"""测试 Gemini 兼容端点的 modelVersion 和模型ID解析

本测试验证：
1. _strip_model_prefix 正确去除 models/ 前缀
2. _get_model_by_name 使用裸模型名调用
3. generateContent 和 streamGenerateContent 返回正确的 modelVersion 格式
"""

import sys
from pathlib import Path

import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.gemini_models import GeminiGenerateContentResponse
from app.server.gemini import _strip_model_prefix, _to_gemini_response


class TestStripModelPrefix:
    """测试 _strip_model_prefix 函数"""

    def test_strip_models_prefix(self):
        """测试能正确去除 models/ 前缀"""
        assert _strip_model_prefix("models/gemini-3-flash") == "gemini-3-flash"
        assert _strip_model_prefix("models/gemini-1.5-pro") == "gemini-1.5-pro"

    def test_keep_bare_name(self):
        """测试裸模型名保持不变"""
        assert _strip_model_prefix("gemini-3-flash") == "gemini-3-flash"
        assert _strip_model_prefix("gemini-1.5-pro") == "gemini-1.5-pro"

    def test_other_prefix_unchanged(self):
        """测试非 models/ 前缀不会被误删"""
        assert _strip_model_prefix("model/gemini-3-flash") == "model/gemini-3-flash"
        assert _strip_model_prefix("some/gemini") == "some/gemini"


class TestModelVersionFormat:
    """测试 modelVersion 输出格式"""

    def test_model_version_is_bare_name(self):
        """验证 modelVersion 是裸模型名，不是带 models/ 前缀的"""
        model_name = "gemini-3-flash"
        response = _to_gemini_response(
            visible_text="Hi",
            tool_calls=[],
            thoughts=None,
            usage_tuple=(10, 20, 30, 0),
            model_name=model_name,
            image_parts=None
        )

        assert isinstance(response, GeminiGenerateContentResponse)
        assert response.modelVersion == "gemini-3-flash", \
            f"modelVersion 应该是裸模型名，实际是: {response.modelVersion}"
        assert not response.modelVersion.startswith("models/"), \
            "modelVersion 不应该以 models/ 开头"

    def test_model_version_in_response(self):
        """验证响应中包含正确格式的 modelVersion"""
        response = _to_gemini_response(
            visible_text="你好世界",
            tool_calls=[],
            thoughts=None,
            usage_tuple=(5, 8, 13, 0),
            model_name="gemini-1.5-pro",
            image_parts=[]
        )

        assert response.modelVersion == "gemini-1.5-pro"
        assert hasattr(response, 'candidates')


class TestCodePaths:
    """测试关键代码路径"""

    def test_strip_model_prefix_in_routes(self):
        """验证路由中使用 _strip_model_prefix"""
        import inspect

        from app.server import gemini

        source = inspect.getsource(gemini)

        # 应该有 3 处调用
        assert source.count("_strip_model_prefix(model)") == 3, \
            "应该有3处 _strip_model_prefix(model) 调用"

    def test_model_version_assignments(self):
        """验证 modelVersion 赋值使用 model_name"""
        import re
        from pathlib import Path

        gemini_path = Path(__file__).parent.parent / "app" / "server" / "gemini.py"
        source = gemini_path.read_text(encoding="utf-8")

        # 查找 modelVersion 赋值
        # 应该使用裸 model_name，不是 f"models/{model_name}"
        model_version_pattern = r'modelVersion\s*=\s*([^\n,)]+)'
        matches = re.findall(model_version_pattern, source)

        for match in matches:
            assert "f\"models/" not in match, \
                f"modelVersion 不应使用 f'models/...' 格式，当前: {match}"
            assert match.strip() == "model_name", \
                f"modelVersion 应该是 model_name，当前: {match}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
