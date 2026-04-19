"""Gemini REST API v1beta 原生端点路由。

提供与 Google Gemini API 兼容的端点:
- GET  /v1beta/models             — 列出可用模型
- GET  /v1beta/models/{model}     — 获取单个模型信息
- POST /v1beta/models/{model}:generateContent       — 非流式生成
- POST /v1beta/models/{model}:streamGenerateContent  — 流式生成

内部复用 chat.py 已有的辅助函数, 不修改 chat.py 代码。
"""

from __future__ import annotations

import io
import reprlib
import uuid
from pathlib import Path
from typing import Any, cast

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import ModelOutput
from loguru import logger

from app.models import (
    ContentItem,
    FunctionCall,
    Message,
    Tool,
    ToolCall,
    ToolChoiceFunction,
    ToolFunctionDefinition,
)
from app.models.gemini_models import (
    GeminiCandidate,
    GeminiContent,
    GeminiErrorDetail,
    GeminiErrorResponse,
    GeminiFunctionCall,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiInlineData,
    GeminiModelInfo,
    GeminiModelListResponse,
    GeminiPart,
    GeminiUsageMetadata,
)

# 从 chat.py 导入已有辅助函数(保持不修改 chat.py 的原则)
from app.server.chat import (
    StreamingOutputFilter,
    _build_structured_requirement,
    _calculate_usage,
    _find_reusable_session,
    _get_available_models,
    _get_model_by_name,
    _image_to_base64,
    _persist_conversation,
    _prepare_messages_for_model,
    _process_llm_output,
    _send_with_split,
)
from app.server.middleware import (
    get_image_store_dir,
    get_image_token,
    get_temp_dir,
    verify_gemini_api_key,
)
from app.services import GeminiClientPool, GeminiClientWrapper, LMDBConversationStore

router = APIRouter()

# Gemini 路由的 422 验证错误处理器, 需在 main.py 中通过
# add_gemini_exception_handlers(app) 注册


def add_gemini_exception_handlers(app):
    """注册 Gemini 路由专用的异常处理器。"""

    @app.exception_handler(RequestValidationError)
    async def gemini_validation_exception_handler(request: Request, exc: RequestValidationError):
        """将 Gemini 路由的 422 验证错误转为 Google API 标准错误格式。"""
        # 仅对 /v1beta/ 路径生效
        if request.url.path.startswith("/v1beta/"):
            detail = str(exc.errors()) if exc.errors() else str(exc)
            err = GeminiErrorResponse(
                error=GeminiErrorDetail(
                    code=400,
                    message=f"Invalid request: {detail}",
                    status="INVALID_ARGUMENT",
                )
            )
            return JSONResponse(status_code=400, content=err.model_dump(mode="json"))
        # 非 Gemini 路由走默认 422
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )


# ---------------------------------------------------------------------------
# Gemini ↔ 内部格式转换函数
# ---------------------------------------------------------------------------


def _gemini_contents_to_messages(
    contents: list[GeminiContent],
    system_instruction: Any | None = None,
) -> list[Message]:
    """将 Gemini contents + systemInstruction 转换为内部 Message 列表。

    正确处理:
    - role="model" + functionCall → assistant 消息 + tool_calls
    - role="user" 或 role="function" + functionResponse → tool 消息
    - 多个 functionResponse parts → 多条 tool 消息
    - 多模态 parts 保持原始顺序(text/image 交错)
    """
    messages: list[Message] = []

    # 系统指令 → system message
    if system_instruction:
        sys_parts = system_instruction.parts or []
        sys_texts = [p.text for p in sys_parts if p.text]
        if sys_texts:
            messages.append(Message(role="system", content="\n".join(sys_texts)))

    # 追踪上一个 assistant 消息的 tool_call IDs, 用于关联 functionResponse
    last_tool_call_ids: dict[str, str] = {}  # function_name → call_id

    for content in contents:
        role = content.role or "user"
        parts = content.parts or []

        # 角色映射: model → assistant
        internal_role = role
        if role == "model":
            internal_role = "assistant"
        elif role == "function":
            internal_role = "tool"

        text_fragments: list[str] = []
        content_items: list[ContentItem] = []
        tool_calls: list[ToolCall] = []
        function_responses: list[tuple[str | None, str]] = []  # (name, content_json)

        # 按原始顺序处理 parts, 保持交错关系
        for part in parts:
            if part.text is not None:
                text_fragments.append(part.text)

            if part.inlineData:
                data_url = f"data:{part.inlineData.mimeType};base64,{part.inlineData.data}"
                content_items.append(
                    ContentItem(type="image_url", image_url={"url": data_url, "detail": "auto"})
                )

            if part.fileData:
                file_info: dict[str, Any] = {"url": part.fileData.fileUri}
                if part.fileData.mimeType:
                    file_info["mime_type"] = part.fileData.mimeType
                content_items.append(ContentItem(type="file", file=file_info))

            if part.functionCall:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=FunctionCall(
                            name=part.functionCall.name,
                            arguments=(
                                orjson.dumps(part.functionCall.args).decode("utf-8")
                                if part.functionCall.args
                                else "{}"
                            ),
                        ),
                    )
                )
                # 记录 call_id 以供后续 functionResponse 关联
                last_tool_call_ids[part.functionCall.name] = call_id

            if part.functionResponse:
                resp_content = orjson.dumps(part.functionResponse.response).decode("utf-8")
                function_responses.append((part.functionResponse.name, resp_content))

        # 构建 Message(s)
        if function_responses:
            # functionResponse: 不论原始 role 是 "user" 还是 "function", 都转为 tool 消息
            # 支持多个 functionResponse → 多条 tool 消息
            for fn_name, fn_content in function_responses:
                # 尝试关联前一轮 functionCall 的 call_id
                call_id = last_tool_call_ids.get(fn_name or "", f"call_{uuid.uuid4().hex[:24]}")
                messages.append(
                    Message(
                        role="tool",
                        content=fn_content,
                        name=fn_name,
                        tool_call_id=call_id,
                    )
                )
            # 如果同时有文本, 额外追加一条用户消息
            if text_fragments and internal_role != "tool":
                messages.append(Message(role=internal_role, content="\n".join(text_fragments)))
        elif tool_calls:
            # assistant 消息带 tool_calls
            msg_content = "\n".join(text_fragments) if text_fragments else None
            messages.append(Message(role="assistant", content=msg_content, tool_calls=tool_calls))
        elif content_items or text_fragments:
            # 多模态内容: 按原始顺序构建 content_items
            if content_items:
                ordered_items: list[ContentItem] = []
                text_idx, media_idx = 0, 0
                for part in parts:
                    if part.text is not None and text_idx < len(text_fragments):
                        ordered_items.append(
                            ContentItem(type="text", text=text_fragments[text_idx])
                        )
                        text_idx += 1
                    elif (part.inlineData or part.fileData) and media_idx < len(content_items):
                        ordered_items.append(content_items[media_idx])
                        media_idx += 1
                messages.append(Message(role=internal_role, content=ordered_items))
            else:
                messages.append(Message(role=internal_role, content="\n".join(text_fragments)))
        else:
            # 空 parts, 仍创建消息以保持对话结构
            messages.append(Message(role=internal_role, content=""))

    return messages


def _gemini_tools_to_internal(
    tools: list[Any] | None,
    tool_config: Any | None = None,
) -> tuple[list[Tool] | None, str | ToolChoiceFunction | None]:
    """将 Gemini tools + toolConfig 转为内部 Tool 列表和 tool_choice。"""
    if not tools:
        return None, None

    internal_tools: list[Tool] = []
    for tool in tools:
        for decl in tool.functionDeclarations or []:
            internal_tools.append(
                Tool(
                    type="function",
                    function=ToolFunctionDefinition(
                        name=decl.name,
                        description=decl.description,
                        parameters=decl.parameters,
                    ),
                )
            )

    # tool_choice 映射
    tool_choice: str | ToolChoiceFunction | None = None
    if tool_config and tool_config.functionCallingConfig:
        mode = tool_config.functionCallingConfig.mode.upper()
        if mode == "NONE":
            tool_choice = "none"
        elif mode == "ANY":
            tool_choice = "required"
        else:  # AUTO
            tool_choice = "auto"

    return internal_tools or None, tool_choice


def _to_gemini_response(
    visible_text: str | None,
    tool_calls: list[Any],
    thoughts: str | None,
    usage_tuple: tuple[int, int, int, int],
    model_name: str,
    image_parts: list[GeminiPart] | None = None,
) -> GeminiGenerateContentResponse:
    """将内部处理结果转换为 Gemini API 响应。"""
    parts: list[GeminiPart] = []

    # 思考内容 → thought part (Gemini API 原生格式)
    if thoughts:
        parts.append(GeminiPart(text=thoughts, thought=True))

    if visible_text:
        parts.append(GeminiPart(text=visible_text))

    # 图片 → inlineData parts
    if image_parts:
        parts.extend(image_parts)

    for tc in tool_calls:
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        fn_args_raw = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
        try:
            fn_args = orjson.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
        except orjson.JSONDecodeError:
            fn_args = {}
        parts.append(GeminiPart(functionCall=GeminiFunctionCall(name=fn_name, args=fn_args)))

    finish_reason = "STOP"
    p_tok, c_tok, t_tok, r_tok = usage_tuple

    candidate = GeminiCandidate(
        content=GeminiContent(role="model", parts=parts),
        finishReason=finish_reason,
        index=0,
    )

    usage_meta = GeminiUsageMetadata(
        promptTokenCount=p_tok,
        candidatesTokenCount=c_tok - r_tok,
        totalTokenCount=t_tok,
        thoughtsTokenCount=r_tok if r_tok > 0 else None,
    )

    return GeminiGenerateContentResponse(
        candidates=[candidate],
        usageMetadata=usage_meta,
            modelVersion=f"models/{model_name}",  # 返回 models/ 前缀格式以匹配 Google API 格式
    )


def _to_gemini_error(status_code: int, message: str, grpc_status: str) -> GeminiErrorResponse:
    """构建 Google API 标准错误响应。"""
    return GeminiErrorResponse(
        error=GeminiErrorDetail(
            code=status_code,
            message=message,
            status=grpc_status,
        )
    )


def _strip_model_prefix(model: str) -> str:
    """去除 'models/' 前缀(如果有)。"""
    if model.startswith("models/"):
        return model[len("models/") :]
    return model


def _model_data_to_gemini_info(model_data: Any) -> GeminiModelInfo:
    """将内部 ModelData 转换为 GeminiModelInfo。"""
    return GeminiModelInfo(
        name=f"models/{model_data.id}",
        displayName=model_data.id,
        description=f"Gemini model: {model_data.id}",
        supportedGenerationMethods=["generateContent", "streamGenerateContent"],
    )


# ---------------------------------------------------------------------------
# 路由端点
# ---------------------------------------------------------------------------


@router.get("/v1beta/models")
async def gemini_list_models(api_key: str = Depends(verify_gemini_api_key)):
    """列出所有可用模型(Gemini API 格式)。"""
    models = _get_available_models()
    gemini_models = [_model_data_to_gemini_info(m) for m in models]
    return GeminiModelListResponse(models=gemini_models)


@router.get("/v1beta/models/{model}")
async def gemini_get_model(model: str, api_key: str = Depends(verify_gemini_api_key)):
    """获取单个模型信息(Gemini API 格式)。"""
    model_name = _strip_model_prefix(model)
    try:
        _get_model_by_name(model_name)
    except ValueError as exc:
        err = _to_gemini_error(404, str(exc), "NOT_FOUND")
        return JSONResponse(status_code=404, content=err.model_dump(mode="json"))

    # 查找对应的 ModelData(如果存在)
    all_models = _get_available_models()
    for m in all_models:
        if m.id == model_name:
            return _model_data_to_gemini_info(m)

    # 模型存在但不在列表中(直接从 gemini-webapi 常量获取到的)
    return GeminiModelInfo(
        name=f"models/{model_name}",
        displayName=model_name,
        description=f"Gemini model: {model_name}",
        supportedGenerationMethods=["generateContent", "streamGenerateContent"],
    )


@router.post("/v1beta/models/{model}:generateContent")
async def gemini_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """非流式生成内容(Gemini API 格式)。"""
    model_name = _strip_model_prefix(model)

    try:
        model_obj = _get_model_by_name(model_name)
    except ValueError as exc:
        err = _to_gemini_error(400, str(exc), "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    if not request.contents:
        err = _to_gemini_error(400, "contents is required and cannot be empty.", "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    # 转换为内部格式
    messages = _gemini_contents_to_messages(request.contents, request.systemInstruction)

    # 转换工具
    internal_tools, tool_choice = _gemini_tools_to_internal(request.tools, request.toolConfig)

    # 构建结构化输出需求(如果指定了 responseMimeType=application/json)
    structured_requirement = None
    if request.generationConfig:
        gen_cfg = request.generationConfig
        schema = gen_cfg.responseSchema or gen_cfg.responseJsonSchema
        if gen_cfg.responseMimeType == "application/json" and schema:
            structured_requirement = _build_structured_requirement(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            )

    extra_instr = [structured_requirement.instruction] if structured_requirement else None

    # 准备消息
    msgs = _prepare_messages_for_model(messages, internal_tools, tool_choice, extra_instr)

    pool, db = GeminiClientPool(), LMDBConversationStore()

    # 查找可复用会话
    session, client, remain = await _find_reusable_session(db, pool, model_obj, msgs)

    if session:
        if not remain:
            err = _to_gemini_error(400, "No new messages to send.", "INVALID_ARGUMENT")
            return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

        input_msgs = _prepare_messages_for_model(
            remain, internal_tools, tool_choice, extra_instr, False
        )
        m_input, files = await GeminiClientWrapper.process_conversation(input_msgs, tmp_dir)
        logger.debug(
            f"[Gemini API] Reusing session {reprlib.repr(session.metadata)}"
            f" - sending {len(input_msgs)} message(s)."
        )
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model_obj)
            m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        except Exception as e:
            logger.exception("[Gemini API] Failed to prepare session")
            err = _to_gemini_error(503, str(e), "UNAVAILABLE")
            return JSONResponse(status_code=503, content=err.model_dump(mode="json"))

    try:
        assert session and client
        logger.debug(
            f"[Gemini API] Client: {client.id}, input len: {len(m_input)}, files: {len(files)}"
        )
        resp = await _send_with_split(
            session, m_input, files=cast("list[Path | str | io.BytesIO]", files), stream=False
        )
    except Exception as e:
        logger.exception("[Gemini API] Gemini call failed")
        err = _to_gemini_error(502, str(e), "INTERNAL")
        return JSONResponse(status_code=502, content=err.model_dump(mode="json"))

    try:
        assert isinstance(resp, ModelOutput)
        thoughts = resp.thoughts
        raw_clean = GeminiClientWrapper.extract_output(resp, include_thoughts=False)
    except Exception:
        logger.exception("[Gemini API] Output parsing failed")
        err = _to_gemini_error(502, "Malformed response.", "INTERNAL")
        return JSONResponse(status_code=502, content=err.model_dump(mode="json"))

    thoughts, visible_output, storage_output, tool_calls = _process_llm_output(
        thoughts, raw_clean, structured_requirement
    )

    # 图片处理: 收集 Gemini 返回的图片, 转为 inlineData parts + markdown URL
    image_parts: list[GeminiPart] = []
    seen_hashes: set[str] = set()
    image_store = get_image_store_dir()
    base_url = str(raw_request.base_url).rstrip("/")
    for image in resp.images or []:
        try:
            b64_str, _w, _h, fname, file_hash = await _image_to_base64(image, image_store)
            if file_hash in seen_hashes:
                (image_store / fname).unlink(missing_ok=True)
                continue
            seen_hashes.add(file_hash)
            # 从文件名后缀推导 MIME 类型
            suffix = Path(fname).suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(suffix, "image/png")
            image_parts.append(
                GeminiPart(inlineData=GeminiInlineData(mimeType=mime_type, data=b64_str))
            )
            # 存储用 markdown URL (用于 LMDB 会话持久化)
            token = get_image_token(fname)
            img_url = f"{base_url}/images/{fname}?token={token}"
            storage_output += f"\n\n![{fname}]({img_url})"
        except Exception as exc:
            logger.warning(f"[Gemini API] Failed to process image: {exc}")

    usage_tuple = _calculate_usage(messages, visible_output, tool_calls, thoughts)

    _persist_conversation(
        db,
        model_obj.model_name,
        client.id,
        session.metadata,
        msgs,
        storage_output,
        tool_calls,
        thoughts,
    )

    gemini_resp = _to_gemini_response(
        visible_output, tool_calls, thoughts, usage_tuple, model_name, image_parts
    )
    return gemini_resp


@router.post("/v1beta/models/{model}:streamGenerateContent")
async def gemini_stream_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """流式生成内容(Gemini API 格式)。"""
    model_name = _strip_model_prefix(model)

    try:
        model_obj = _get_model_by_name(model_name)
    except ValueError as exc:
        err = _to_gemini_error(400, str(exc), "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    if not request.contents:
        err = _to_gemini_error(400, "contents is required and cannot be empty.", "INVALID_ARGUMENT")
        return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

    # 转换为内部格式
    messages = _gemini_contents_to_messages(request.contents, request.systemInstruction)
    internal_tools, tool_choice = _gemini_tools_to_internal(request.tools, request.toolConfig)

    structured_requirement = None
    if request.generationConfig:
        gen_cfg = request.generationConfig
        schema = gen_cfg.responseSchema or gen_cfg.responseJsonSchema
        if gen_cfg.responseMimeType == "application/json" and schema:
            structured_requirement = _build_structured_requirement(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            )

    extra_instr = [structured_requirement.instruction] if structured_requirement else None
    msgs = _prepare_messages_for_model(messages, internal_tools, tool_choice, extra_instr)

    pool, db = GeminiClientPool(), LMDBConversationStore()
    session, client, remain = await _find_reusable_session(db, pool, model_obj, msgs)

    if session:
        if not remain:
            err = _to_gemini_error(400, "No new messages to send.", "INVALID_ARGUMENT")
            return JSONResponse(status_code=400, content=err.model_dump(mode="json"))

        input_msgs = _prepare_messages_for_model(
            remain, internal_tools, tool_choice, extra_instr, False
        )
        m_input, files = await GeminiClientWrapper.process_conversation(input_msgs, tmp_dir)
    else:
        try:
            client = await pool.acquire()
            session = client.start_chat(model=model_obj)
            m_input, files = await GeminiClientWrapper.process_conversation(msgs, tmp_dir)
        except Exception as e:
            logger.exception("[Gemini API] Failed to prepare streaming session")
            err = _to_gemini_error(503, str(e), "UNAVAILABLE")
            return JSONResponse(status_code=503, content=err.model_dump(mode="json"))

    try:
        assert session and client
        generator = await _send_with_split(
            session, m_input, files=cast("list[Path | str | io.BytesIO]", files), stream=True
        )
    except Exception as e:
        logger.exception("[Gemini API] Gemini streaming call failed")
        err = _to_gemini_error(502, str(e), "INTERNAL")
        return JSONResponse(status_code=502, content=err.model_dump(mode="json"))

    return _create_gemini_streaming_response(
        generator=generator,
        model_name=model_name,
        messages=msgs,
        original_messages=messages,
        db=db,
        model=model_obj,
        client_wrapper=client,
        session=session,
        structured_requirement=structured_requirement,
        base_url=str(raw_request.base_url).rstrip("/"),
    )


def _create_gemini_streaming_response(
    generator,
    model_name: str,
    messages: list[Message],
    original_messages: list[Message],
    db: LMDBConversationStore,
    model,
    client_wrapper: GeminiClientWrapper,
    session,
    structured_requirement=None,
    base_url: str = "",
) -> StreamingResponse:
    """创建 Gemini 格式的 SSE 流式响应。"""

    async def generate_stream():
        full_thoughts, full_text = "", ""
        last_chunk: ModelOutput | None = None
        all_images: list[Any] = []  # 收集所有 chunk 的图片 (url 去重)
        seen_image_urls: set[str] = set()
        suppressor = StreamingOutputFilter()

        try:
            async for chunk in generator:
                last_chunk = chunk

                # 收集图片 (按 url 去重, 同 OpenAI 路径)
                if chunk.images:
                    for img in chunk.images:
                        if img.url not in seen_image_urls:
                            all_images.append(img)
                            seen_image_urls.add(img.url)

                # 思考增量: 使用 Gemini 原生 thought part
                if t_delta := chunk.thoughts_delta:
                    full_thoughts += t_delta
                    think_resp = GeminiGenerateContentResponse(
                        candidates=[
                            GeminiCandidate(
                                content=GeminiContent(
                                    role="model",
                                    parts=[GeminiPart(text=t_delta, thought=True)],
                                ),
                                index=0,
                            )
                        ],
                    )
                    yield f"data: {orjson.dumps(think_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

                # 文本增量
                if text_delta := chunk.text_delta:
                    full_text += text_delta
                    if visible_delta := suppressor.process(text_delta):
                        chunk_resp = GeminiGenerateContentResponse(
                            candidates=[
                                GeminiCandidate(
                                    content=GeminiContent(
                                        role="model",
                                        parts=[GeminiPart(text=visible_delta)],
                                    ),
                                    index=0,
                                )
                            ],
                        )
                        yield f"data: {orjson.dumps(chunk_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

        except Exception as e:
            logger.exception(f"[Gemini API] Streaming error: {e}")
            err_resp = _to_gemini_error(500, "Streaming error occurred.", "INTERNAL")
            yield f"data: {orjson.dumps(err_resp.model_dump(mode='json')).decode('utf-8')}\n\n"
            return

        # 使用最终 chunk 的完整文本(如果可用)
        if last_chunk is not None:
            if last_chunk.text:
                full_text = last_chunk.text
            if last_chunk.thoughts:
                full_thoughts = last_chunk.thoughts

        # 刷新剩余文本
        if remaining_text := suppressor.flush():
            chunk_resp = GeminiGenerateContentResponse(
                candidates=[
                    GeminiCandidate(
                        content=GeminiContent(
                            role="model",
                            parts=[GeminiPart(text=remaining_text)],
                        ),
                        index=0,
                    )
                ],
            )
            yield f"data: {orjson.dumps(chunk_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

        # --- 后处理: 整体保护层, 避免 SSE 尾段硬中断 ---
        try:
            # 提取 tool calls 等
            _thoughts, visible_output, storage_output, tool_calls = _process_llm_output(
                full_thoughts, full_text, structured_requirement
            )

            # 图片处理: 使用收集的所有图片 (而非仅 last_chunk)
            image_store = get_image_store_dir()
            seen_hashes: set[str] = set()
            for image in all_images:
                try:
                    b64_str, _w, _h, fname, file_hash = await _image_to_base64(image, image_store)
                    if file_hash in seen_hashes:
                        (image_store / fname).unlink(missing_ok=True)
                        continue
                    seen_hashes.add(file_hash)
                    # 从文件名后缀推导 MIME 类型
                    suffix = Path(fname).suffix.lower()
                    mime_map = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                        ".gif": "image/gif",
                    }
                    mime_type = mime_map.get(suffix, "image/png")
                    # 发送图片 inlineData chunk
                    img_chunk = GeminiGenerateContentResponse(
                        candidates=[
                            GeminiCandidate(
                                content=GeminiContent(
                                    role="model",
                                    parts=[
                                        GeminiPart(
                                            inlineData=GeminiInlineData(
                                                mimeType=mime_type, data=b64_str
                                            )
                                        )
                                    ],
                                ),
                                index=0,
                            )
                        ],
                    )
                    yield f"data: {orjson.dumps(img_chunk.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"
                    # 存储用 markdown URL
                    token = get_image_token(fname)
                    img_url = f"{base_url}/images/{fname}?token={token}"
                    storage_output += f"\n\n![{fname}]({img_url})"
                except Exception as exc:
                    logger.warning(f"[Gemini API] Failed to process streaming image: {exc}")
            # 发送最终 chunk(含 finishReason 和 usageMetadata)
            usage_tuple = _calculate_usage(original_messages, visible_output, tool_calls, _thoughts)
            p_tok, c_tok, t_tok, r_tok = usage_tuple

            final_parts: list[GeminiPart] = []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                    fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                    fn_args_raw = (
                        fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                    )
                    try:
                        fn_args = (
                            orjson.loads(fn_args_raw)
                            if isinstance(fn_args_raw, str)
                            else fn_args_raw
                        )
                    except orjson.JSONDecodeError:
                        fn_args = {}
                    final_parts.append(
                        GeminiPart(functionCall=GeminiFunctionCall(name=fn_name, args=fn_args))
                    )

            final_resp = GeminiGenerateContentResponse(
                candidates=[
                    GeminiCandidate(
                        content=GeminiContent(role="model", parts=final_parts)
                        if final_parts
                        else None,
                        finishReason="STOP",
                        index=0,
                    )
                ],
                usageMetadata=GeminiUsageMetadata(
                    promptTokenCount=p_tok,
                    candidatesTokenCount=c_tok - r_tok,
                    totalTokenCount=t_tok,
                    thoughtsTokenCount=r_tok if r_tok > 0 else None,
                ),
                modelVersion=f"models/{model_name}",  # 返回 models/ 前缀格式以匹配 Google API 格式
            )
            yield f"data: {orjson.dumps(final_resp.model_dump(mode='json', exclude_none=True)).decode('utf-8')}\n\n"

            # 持久化会话
            _persist_conversation(
                db,
                model.model_name,
                client_wrapper.id,
                session.metadata,
                messages,
                storage_output,
                tool_calls,
                _thoughts,
            )
        except Exception as exc:
            logger.exception(f"[Gemini API] Post-processing error: {exc}")
            err_resp = _to_gemini_error(500, "Post-processing error.", "INTERNAL")
            yield f"data: {orjson.dumps(err_resp.model_dump(mode='json')).decode('utf-8')}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
