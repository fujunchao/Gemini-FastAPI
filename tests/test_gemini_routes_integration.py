"""Gemini 路由级 HTTP 集成测试。

使用 FastAPI TestClient 发起真实 HTTP 请求, 验证原生 Gemini 路由的
路径解析、响应格式以及未知模型错误路径。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.gemini_models import GeminiGenerateContentRequest
from app.server.gemini import verify_gemini_api_key

MODEL_NAME = "gemini-3-flash"
MODEL_PATHS = [MODEL_NAME, f"models/{MODEL_NAME}"]


@pytest.fixture
def app():
    """创建带有 mock 依赖的 FastAPI 应用。"""
    startup_pool = MagicMock()
    startup_pool.init = AsyncMock(return_value=None)
    startup_pool.clients = [MagicMock(id="startup-client")]

    startup_store = MagicMock()
    startup_store.retention_days = 0

    route_session = MagicMock(metadata={"session_id": "mock-session"})
    route_client = MagicMock(id="mock-client")
    route_client.start_chat = MagicMock(return_value=route_session)
    route_pool = MagicMock()
    route_pool.acquire = AsyncMock(return_value=route_client)
    route_store = MagicMock()
    route_model = MagicMock(
        id=MODEL_NAME,
        model_name=MODEL_NAME,
        supports_image=False,
        supports_thinking=False,
    )
    async def fake_process_conversation(messages, tmp_dir):
        return ["prepared-input"], []

    async def fake_send_with_split(session, m_input, files, stream):
        from gemini_webapi import Candidate, ModelOutput

        output = ModelOutput(
            metadata=["mock-metadata"],
            candidates=[Candidate(rcid="rcid-1", text="Hello World")],
            chosen=0,
        )
        if stream:

            async def generator():
                yield output

            return generator()

        return output

    with (
        patch("app.main.GeminiClientPool", return_value=startup_pool),
        patch("app.main.LMDBConversationStore", return_value=startup_store),
        patch("app.server.gemini.GeminiClientPool", return_value=route_pool),
        patch("app.server.gemini.LMDBConversationStore", return_value=route_store),
        patch("app.server.gemini._get_model_by_name", return_value=route_model) as mock_get_model,
        patch("app.server.gemini._find_reusable_session", return_value=(None, None, None)),
        patch("app.server.gemini.GeminiClientWrapper.process_conversation", new=AsyncMock(side_effect=fake_process_conversation)),
        patch("app.server.gemini.GeminiClientWrapper.extract_output", return_value="Hello World"),
        patch("app.server.gemini._send_with_split", new=AsyncMock(side_effect=fake_send_with_split)),
        patch("app.server.gemini._calculate_usage", return_value=(10, 20, 30, 0)),
        patch("app.server.gemini._process_llm_output", return_value=(None, "Hello World", "Hello World", [])),
        patch("app.server.gemini._persist_conversation", return_value=None),
    ):
        from app.main import create_app

        test_app = create_app()
        test_app.dependency_overrides[verify_gemini_api_key] = lambda: ""
        test_app.state.route_model = route_model
        test_app.state.route_session = route_session
        test_app.state.route_client = route_client
        test_app.state.mock_get_model = mock_get_model
        yield test_app


def _request_body() -> dict:
    return GeminiGenerateContentRequest(
        contents=[
            {
                "role": "user",
                "parts": [{"text": "你好"}],
            }
        ]
    ).model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_get_model_http_route(app, model_path):
    with TestClient(app) as client:
        response = client.get(f"/v1beta/models/{model_path}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == f"models/{MODEL_NAME}"
    assert payload["displayName"] == MODEL_NAME
    assert payload["supportedGenerationMethods"] == ["generateContent", "streamGenerateContent"]


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_generate_content_http_route(app, model_path):
    with TestClient(app) as client:
        response = client.post(f"/v1beta/models/{model_path}:generateContent", json=_request_body())

    app.state.mock_get_model.assert_called_with(MODEL_NAME)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["modelVersion"] == MODEL_NAME
    assert payload["candidates"][0]["content"]["role"] == "model"
    assert payload["candidates"][0]["content"]["parts"][0]["text"] == "Hello World"


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_stream_generate_content_http_route(app, model_path):
    with TestClient(app) as client, client.stream(
        "POST",
        f"/v1beta/models/{model_path}:streamGenerateContent",
        json=_request_body(),
    ) as response:
        body = "".join(response.iter_text())

    app.state.mock_get_model.assert_called_with(MODEL_NAME)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in body
    assert f'"modelVersion":"{MODEL_NAME}"' in body


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_unknown_model_returns_404(app, model_path):
    from app.server import gemini as gemini_module

    app.dependency_overrides[verify_gemini_api_key] = lambda: ""
    with patch.object(
        gemini_module,
        "_get_model_by_name",
        side_effect=ValueError("Model not found"),
    ), TestClient(app) as client:
        response = client.get(f"/v1beta/models/{model_path}")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == 404
    assert payload["error"]["status"] == "NOT_FOUND"
