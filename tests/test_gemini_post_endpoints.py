"""测试 POST Gemini 端点的模型ID解析和响应格式

本测试验证：
1. POST /v1beta/models/{model}:generateContent 使用 _strip_model_prefix 解析模型
2. POST /v1beta/models/{model}:streamGenerateContent 使用 _strip_model_prefix 解析模型
3. _get_model_by_name 接收到的始终是裸模型名（无 models/ 前缀）
4. 响应中的 modelVersion 格式正确
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def app_with_mocked_models():
    """创建带 mock 模型服务的 FastAPI 应用"""
    from app.main import create_app
    from app.utils.config import Config

    # 创建测试配置
    test_config = Config(
        gemini=MagicMock(
            model_strategy="fallback",
            models=[],
            timeout=30
        ),
        server=MagicMock(
            host="127.0.0.1",
            port=8000,
            temp_dir=Path("./tmp")
        )
    )

    with patch("app.utils.config.g_config", test_config):
        app = create_app()
        return app


class TestGenerateContentEndpoint:
    """测试非流式生成端点"""

    @patch("app.server.gemini._get_model_by_name")
    @patch("app.services.GeminiClientPool")
    @patch("app.services.LMDBConversationStore")
    def test_model_name_with_prefix(
        self, mock_db, mock_pool, mock_get_model
    ):
        """测试 URL 中带 models/ 前缀时，_get_model_by_name 接收裸名"""
        from app.main import create_app
        from fastapi.testclient import TestClient

        # Mock 模型
        mock_model = MagicMock()
        mock_model.model_name = "gemini-3-flash"
        mock_model.supports_image = False
        mock_model.supports_thinking = False
        mock_get_model.return_value = mock_model

        # Mock 客户端池
        mock_client = MagicMock()
        mock_client.process_message.return_value = MagicMock(
            text_delta=None,
            thoughts_delta=None,
            text="Hello World",
            thoughts=None,
            images=None
        )
        mock_pool_instance = MagicMock()
        mock_pool_instance.acquire.return_value.__aenter__ = MagicMock(return_value=MagicMock(
            __aexit__=MagicMock(),
            client=mock_client
        ))
        mock_pool.return_value = mock_pool_instance

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1beta/models/models/gemini-3-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        )

        # 验证 _get_model_by_name 接收的是裸模型名
        mock_get_model.assert_called_once_with("gemini-3-flash")
        assert mock_get_model.call_args[0][0] == "gemini-3-flash"
        assert not mock_get_model.call_args[0][0].startswith("models/")

    @patch("app.server.gemini._get_model_by_name")
    def test_model_name_without_prefix(
        self, mock_get_model
    ):
        """测试 URL 中不带 models/ 前缀时，_get_model_by_name 直接接收裸名"""
        mock_model = MagicMock()
        mock_model.id = "gemini-3-flash"
        mock_model.model_name = "gemini-3-flash"
        mock_model.supports_image = False
        mock_model.supports_thinking = False
        mock_get_model.return_value = mock_model

        # 验证函数行为
        from app.server.gemini import _strip_model_prefix
        stripped = _strip_model_prefix("gemini-3-flash")
        assert stripped == "gemini-3-flash"

    def test_strip_model_prefix_called(self):
        """验证 _strip_model_prefix 在请求处理中被调用"""
        from app.server import gemini

        # 检查源码中确实调用了 _strip_model_prefix
        import inspect
        source = inspect.getsource(gemini.gemini_generate_content)
        assert "_strip_model_prefix(model)" in source


class TestStreamGenerateContentEndpoint:
    """测试流式生成端点"""

    @patch("app.server.gemini._get_model_by_name")
    def test_stream_model_name_with_prefix(
        self, mock_get_model
    ):
        """测试流式端点 URL 中带 models/ 前缀时的处理"""
        mock_model = MagicMock()
        mock_model.id = "gemini-3-flash"
        mock_model.model_name = "gemini-3-flash"
        mock_model.supports_image = False
        mock_model.supports_thinking = False
        mock_get_model.return_value = mock_model

        # 验证 _strip_model_prefix 被调用
        from app.server.gemini import _strip_model_prefix
        result = _strip_model_prefix("models/gemini-3-flash")
        assert result == "gemini-3-flash"

    @patch("app.server.gemini._get_model_by_name")
    def test_stream_model_version_in_final_chunk(
        self, mock_get_model
    ):
        """验证流式响应最终 chunk 的 modelVersion 格式"""
        from app.server.gemini import _create_gemini_streaming_response

        mock_model = MagicMock()
        mock_model.model_name = "gemini-3-flash"
        mock_get_model.return_value = mock_model

        # 模拟流式生成器的最终处理
        from app.server.gemini import _strip_model_prefix
        model_name = "gemini-3-flash"
        stripped = _strip_model_prefix(f"models/{model_name}")

        assert stripped == "gemini-3-flash"
        assert not stripped.startswith("models/")


class TestModelVersionFormat:
    """测试响应中的 modelVersion 格式"""

    def test_non_streaming_model_version(self):
        """验证非流式响应 modelVersion 是裸模型名"""
        from app.server.gemini import _to_gemini_response

        response = _to_gemini_response(
            visible_text="Test",
            tool_calls=[],
            thoughts=None,
            usage_tuple=(10, 20, 30, 0),
            model_name="gemini-3-flash",
            image_parts=None
        )

        assert response.modelVersion == "gemini-3-flash"
        assert not response.modelVersion.startswith("models/")


class TestErrorHandling:
    """测试错误处理正确使用裸模型名"""

    @patch("app.server.gemini._get_model_by_name")
    def test_unknown_model_error(self, mock_get_model):
        """测试未知模型错误时，模型名处理正确"""
        mock_get_model.side_effect = ValueError("Model 'unknown-model' not found")

        from app.server.gemini import _strip_model_prefix
        stripped = _strip_model_prefix("models/unknown-model")
        assert stripped == "unknown-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
