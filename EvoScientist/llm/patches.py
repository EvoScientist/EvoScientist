"""Monkey-patches and utilities for third-party LangChain provider quirks.

All patches follow the same pattern: wrap an existing method/function to
fix upstream bugs, applied at import time or on first use.

Patches:
    - _patch_anthropic_proxy_compat: ccproxy dict→Pydantic model mismatch
    - _patch_openai_compat_content: list content→string for strict APIs
    - _patch_ccproxy_codex_compat: ccproxy model fixes + langchain None guard
    - _patch_ccproxy_system_to_developer: system→developer role for ccproxy
    - _patch_openai_capture_reasoning_content: capture provider
      reasoning_content into AIMessage.additional_kwargs (module-level,
      applied at import)
    - _patch_deepseek_reasoning_passback: re-inject reasoning_content into
      outgoing DeepSeek assistant messages for thinking-mode multi-turn /
      tool_use scenarios

Utilities:
    - _is_ccproxy_codex: detect ccproxy Codex OAuth adapter
    - _flatten_message_content: convert content blocks to plain string
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Patch: langchain-anthropic (>=1.3.4) calls .model_dump() on
# context_management / container objects returned by the Anthropic SDK.
# Proxies like ccproxy may return plain dicts which lack that method.
# We wrap the class method to pre-convert dicts before the original runs.
# ---------------------------------------------------------------------------
def _patch_anthropic_proxy_compat() -> None:
    try:
        import types as _types

        from langchain_anthropic.chat_models import ChatAnthropic as _CA

        _orig = _CA._make_message_chunk_from_anthropic_event

        def _safe(self: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
            for obj, attrs in [
                (event, ("context_management",)),
                (getattr(event, "delta", None), ("container",)),
            ]:
                if obj is None:
                    continue
                for attr in attrs:
                    val = getattr(obj, attr, None)
                    if isinstance(val, dict):
                        d = val.copy()
                        setattr(
                            obj,
                            attr,
                            _types.SimpleNamespace(model_dump=lambda d=d, **kw: d),
                        )
            return _orig(self, event, *args, **kwargs)

        _CA._make_message_chunk_from_anthropic_event = _safe
    except Exception:
        pass


_patch_anthropic_proxy_compat()


# ---------------------------------------------------------------------------
# Patch: ccproxy-api 0.2.7 Codex compatibility.
#
# 1) ResponseObject.output is required but upstream may omit it → 502.
#    Fix: make output default to [].
# 2) CodexMessage.role only allows "user"/"assistant" → 400 on system msgs.
#    Fix: widen to also accept "system" and "developer".
# 3) langchain-openai iterates response.output which can be None after the
#    proxy strips it.  Fix: guard in _construct_lc_result_from_responses_api.
# ---------------------------------------------------------------------------
def _patch_ccproxy_codex_compat() -> None:
    """Patch ccproxy-api models for Responses API compatibility."""
    # 1) Make ResponseObject.output optional (default=[])
    try:
        import ccproxy.llms.models.openai as _oai_mod

        _OrigResponse = _oai_mod.ResponseObject

        from pydantic import Field as _PydanticField

        class _PatchedResponseObject(_OrigResponse):  # type: ignore[misc]
            output: list = _PydanticField(default_factory=list)  # type: ignore[assignment]

            model_config = _OrigResponse.model_config.copy()

        _PatchedResponseObject.__name__ = "ResponseObject"
        _PatchedResponseObject.__qualname__ = "ResponseObject"
        _oai_mod.ResponseObject = _PatchedResponseObject  # type: ignore[misc]

        # Also patch modules that import ResponseObject directly
        for _mod_path in (
            "ccproxy.llms.formatters.openai_to_openai.responses",
            "ccproxy.llms.formatters.anthropic_to_openai.responses",
        ):
            try:
                import importlib

                _mod = importlib.import_module(_mod_path)
                if hasattr(_mod, "ResponseObject"):
                    _mod.ResponseObject = _PatchedResponseObject  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    # 2) Widen CodexMessage.role to accept system/developer
    try:
        from typing import Annotated, Literal

        import ccproxy.plugins.codex.models as _codex_mod

        _OrigMessage = _codex_mod.CodexMessage

        from pydantic import Field as _Field

        class _PatchedCodexMessage(_OrigMessage):  # type: ignore[misc]
            role: Annotated[  # type: ignore[assignment]
                Literal["user", "assistant", "system", "developer"],
                _Field(description="Message role"),
            ]

        _PatchedCodexMessage.__name__ = "CodexMessage"
        _PatchedCodexMessage.__qualname__ = "CodexMessage"
        _codex_mod.CodexMessage = _PatchedCodexMessage  # type: ignore[misc]
    except Exception:
        pass

    # 3) Fix StreamingBufferService returning response.completed event
    #    whose output is None/empty, instead of using accumulated outputs.
    try:
        from ccproxy.llms.streaming.accumulators import ResponsesAccumulator

        _orig_get = ResponsesAccumulator.get_completed_response

        def _patched_get(self: Any) -> dict | None:
            result = _orig_get(self)
            if result is not None:
                output = result.get("output")
                if output is None:
                    # output field lost — force rebuild from accumulated items
                    return None
            return result

        ResponsesAccumulator.get_completed_response = _patched_get  # type: ignore[assignment]
    except Exception:
        pass

    # 4) Guard langchain-openai against None output (final safety net)
    try:
        import langchain_openai.chat_models.base as _base

        _orig_construct = _base._construct_lc_result_from_responses_api

        def _safe(response: Any, *args: Any, **kwargs: Any) -> Any:
            if response.output is None:
                response.output = []
            return _orig_construct(response, *args, **kwargs)

        _base._construct_lc_result_from_responses_api = _safe
    except Exception:
        pass


_patch_ccproxy_codex_compat()


# ---------------------------------------------------------------------------
# Utility: detect ccproxy's Codex adapter (as opposed to generic localhost).
# ---------------------------------------------------------------------------
def _is_ccproxy_codex() -> bool:
    """Return True if the OpenAI endpoint is ccproxy's Codex adapter.

    Checks for the ccproxy-specific markers set by ``setup_codex_env()``
    in ``ccproxy_manager.py``: the sentinel API key and the ``/codex/v1``
    path.  Plain localhost endpoints (vLLM, Ollama, etc.) are not affected.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return (
        ("127.0.0.1" in base_url or "localhost" in base_url)
        and api_key == "ccproxy-oauth"
        and "/codex/" in base_url
    )


# ---------------------------------------------------------------------------
# Utility + Patch: Flatten list content to strings for OpenAI-compatible APIs.
# DeepSeek, SiliconFlow, etc. reject assistant messages whose content is a
# list rather than a string.  Image and file (PDF/document) blocks are
# preserved, not flattened away.
# ---------------------------------------------------------------------------
_SKIP_CONTENT_TYPES = frozenset({"thinking", "reasoning", "reasoning_content"})

# Media block types preserved when flattening (positive allowlist;
# thinking/reasoning still dropped).  Images + files (PDF/documents): both
# serialize on OpenAI-compatible APIs and capable models read them.  `video` is
# deliberately EXCLUDED (langchain-openai raises ValueError on it); `audio` is
# omitted (almost no model support).  is_data_content_block is unreliable
# (False for OpenAI image_url / Anthropic image-source).  Kept as separate
# image/file sets so the no-media fallback can gate per modality.
_IMAGE_CONTENT_TYPES = frozenset({"image", "image_url", "input_image"})
_FILE_CONTENT_TYPES = frozenset({"file", "input_file", "document"})
_MEDIA_CONTENT_TYPES = _IMAGE_CONTENT_TYPES | _FILE_CONTENT_TYPES


def _flatten_message_content(content: Any) -> str | list[Any] | Any:
    """Convert list-of-blocks content to a string, preserving media blocks.

    Thinking/reasoning blocks are dropped.  When a media block (image or file)
    is present, returns a list in the ORIGINAL order — consecutive text is
    joined into a single text block, media blocks kept as-is — so captions stay
    next to the attachment they describe.  Otherwise returns a plain string.
    Non-list input is returned unchanged.

    Args:
        content: Message content — a string, a list of content blocks, or
            another type.

    Returns:
        A plain string for text-only content, a list of blocks when media is
        present, or the input unchanged for non-list input.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    ordered_blocks: list[Any] = []
    saw_media = False

    def _flush_text() -> None:
        if parts:
            ordered_blocks.append({"type": "text", "text": "\n\n".join(parts)})
            parts.clear()

    for block in content:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype in _MEDIA_CONTENT_TYPES:
                # Keep media as-is (never mutate; upstream copy.copy is shallow)
                # and preserve its position relative to surrounding text.
                _flush_text()
                ordered_blocks.append(block)
                saw_media = True
                continue
            if btype in _SKIP_CONTENT_TYPES:
                continue
            text = block.get("text")
            if text:
                parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    if saw_media:
        _flush_text()
        return ordered_blocks
    return "\n\n".join(parts) if parts else ""


def _sanitize_messages(messages: list[Any], hoist_tool_media: bool = True) -> list[Any]:
    """Flatten list content for OpenAI-compatible APIs, preserving media.

    Text/reasoning content is flattened to a string; image blocks are
    preserved.  Tool messages cannot carry media
    on OpenAI-compatible APIs (content must be a string), so when
    ``hoist_tool_media`` is set the media in a tool result is moved into a
    HumanMessage emitted after that turn's (possibly parallel) tool messages.
    Anthropic-routed providers accept tool-result media natively and pass
    ``hoist_tool_media=False`` to keep it inline.
    """
    import copy

    from langchain_core.messages import HumanMessage

    out: list[Any] = []
    pending_media: list[Any] = []  # media hoisted out of a run of tool messages

    def _flush() -> None:
        if pending_media:
            out.append(HumanMessage(content=list(pending_media)))
            pending_media.clear()

    for msg in messages:
        is_tool = getattr(msg, "type", None) == "tool"
        if not is_tool:
            _flush()  # emit hoisted media before any non-tool message
        if not isinstance(msg.content, list):
            out.append(msg)
            continue
        flat = _flatten_message_content(msg.content)
        if hoist_tool_media and is_tool and isinstance(flat, list):
            text_blocks = [
                b for b in flat if isinstance(b, dict) and b.get("type") == "text"
            ]
            media_blocks = [
                b for b in flat if not (isinstance(b, dict) and b.get("type") == "text")
            ]
            tool_msg = copy.copy(msg)
            # Join ALL text runs (interleaved content can yield more than one)
            # so no text is lost; tool content must be a string on OpenAI-compat.
            tool_msg.content = (
                "\n\n".join(b["text"] for b in text_blocks)
                if text_blocks
                else "[media content provided in the following message]"
            )
            out.append(tool_msg)
            pending_media.extend(media_blocks)
        else:
            msg = copy.copy(msg)
            msg.content = flat
            out.append(msg)
    _flush()  # conversation may end with tool messages
    return out


# Fallback for models that reject some media input (no vision / no file
# support): replace blocks of the unsupported types with a placeholder so the
# conversation can keep going instead of erroring every turn that re-sends them.
_UNSUPPORTED_MEDIA_PLACEHOLDER = (
    "[attachment omitted: this model does not support this input type]"
)
# Markers that identify WHICH modality an error rejects, so only the implicated
# type is remembered (not all media).  Generic markers map to all media.
_IMAGE_ERROR_MARKERS = ("image_url", "image input", "support image")
_FILE_ERROR_MARKERS = ("file input", "pdf input", "document input")
# `expected \`text\`` keeps the backticks — the bare "expected text" phrase is
# too broad (matches non-media schema errors like "... expected text").
_GENERIC_MEDIA_MARKERS = ("multimodal", "expected `text`")


def _media_types_in(messages: list[Any]) -> set[str]:
    """Set of preserved-media block types present across the messages."""
    found: set[str] = set()
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in _MEDIA_CONTENT_TYPES:
                    found.add(b["type"])
    return found


def _media_error_types(exc: Exception) -> set[str]:
    """Media modalities the error text implicates (empty if not media-specific).

    Used to remember ONLY the rejected modality, so e.g. a PDF rejection never
    disables images.
    """
    text = str(exc).lower()
    types: set[str] = set()
    if any(m in text for m in _IMAGE_ERROR_MARKERS):
        types |= _IMAGE_CONTENT_TYPES
    if any(m in text for m in _FILE_ERROR_MARKERS):
        types |= _FILE_CONTENT_TYPES
    if any(m in text for m in _GENERIC_MEDIA_MARKERS):
        types |= _MEDIA_CONTENT_TYPES
    return types


def _is_http_400(exc: Exception) -> bool:
    """True if the error carries an HTTP 400 status (bad request)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 400


def _strip_media_types(messages: list[Any], types: set[str]) -> list[Any]:
    """Replace blocks of the given types with a placeholder text block.

    Each stripped block is replaced IN PLACE (consecutive ones collapse into one
    placeholder), so surrounding text/media keep their original positions and a
    model that rejects only one modality still receives the others in order.
    """
    import copy

    out: list[Any] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and b.get("type") in types for b in content
        ):
            out.append(msg)
            continue
        kept: list[Any] = []
        last_was_placeholder = False
        for b in content:
            if isinstance(b, dict) and b.get("type") in types:
                if not last_was_placeholder:
                    kept.append(
                        {"type": "text", "text": _UNSUPPORTED_MEDIA_PLACEHOLDER}
                    )
                    last_was_placeholder = True
                continue
            kept.append(b)
            last_was_placeholder = False
        msg = copy.copy(msg)
        msg.content = kept
        out.append(msg)
    return out


def _patch_openai_compat_content(model: Any, hoist_tool_media: bool = True) -> None:
    """Flatten list content to strings before OpenAI-compatible API calls.

    Wraps ``_generate`` / ``_agenerate`` / ``_stream`` / ``_astream`` to prevent
    "invalid type: sequence, expected a string" errors from strict APIs like
    DeepSeek, while preserving image and file (PDF/document) blocks.  Tracks a
    per-modality ``blocked`` set of media types the model rejects: when a request
    fails with a media-looking error it is retried with the offending modalities
    stripped to a placeholder, and those types are remembered ONLY after the
    stripped retry succeeds — so an unrelated 400 never permanently degrades the
    model, and a PDF rejection never disables images.  The profile pre-seeds the
    set per modality (``image_inputs``/``pdf_inputs`` ``is False``).

    Args:
        model: A LangChain chat model instance to patch in-place.
        hoist_tool_media: When True (OpenAI-compatible), media in a tool result
            is hoisted into a following HumanMessage (tool content must be a
            string).  Anthropic-routed providers pass False (native support).
    """
    import functools

    from langchain_core.messages import BaseMessage

    # Media block types this model rejects.  Pre-seeded per modality from the
    # profile, then grown reactively — but only after a stripped retry succeeds.
    profile = getattr(model, "profile", None)
    blocked: set[str] = set()
    if isinstance(profile, dict):
        if profile.get("image_inputs") is False:
            blocked |= _IMAGE_CONTENT_TYPES
        if profile.get("pdf_inputs") is False:
            blocked |= _FILE_CONTENT_TYPES

    def _prepare(messages: list[BaseMessage]) -> list[BaseMessage]:
        msgs = _strip_media_types(messages, blocked) if blocked else messages
        return _sanitize_messages(msgs, hoist_tool_media)

    def _stripped(messages: list[BaseMessage], suspects: set[str]) -> list[BaseMessage]:
        return _sanitize_messages(
            _strip_media_types(messages, blocked | suspects), hoist_tool_media
        )

    orig_generate = getattr(model, "_generate", None)
    if orig_generate is None:
        return

    @functools.wraps(orig_generate)
    def _patched_generate(
        messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Any:
        prepared = _prepare(messages)
        suspects = _media_types_in(messages) - blocked
        if not suspects:
            return orig_generate(prepared, *args, **kwargs)
        try:
            return orig_generate(prepared, *args, **kwargs)
        except Exception as exc:
            culprit = _media_error_types(exc) & suspects
            if not culprit and not _is_http_400(exc):
                raise
            try:
                result = orig_generate(_stripped(messages, suspects), *args, **kwargs)
            except Exception:
                # stripping didn't help — surface original, don't cache
                raise exc from None
            blocked.update(culprit)  # cache only the marker-identified modality
            return result

    model._generate = _patched_generate

    orig_agenerate = getattr(model, "_agenerate", None)
    if orig_agenerate is not None:

        @functools.wraps(orig_agenerate)
        async def _patched_agenerate(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            prepared = _prepare(messages)
            suspects = _media_types_in(messages) - blocked
            if not suspects:
                return await orig_agenerate(prepared, *args, **kwargs)
            try:
                return await orig_agenerate(prepared, *args, **kwargs)
            except Exception as exc:
                culprit = _media_error_types(exc) & suspects
                if not culprit and not _is_http_400(exc):
                    raise
                try:
                    result = await orig_agenerate(
                        _stripped(messages, suspects), *args, **kwargs
                    )
                except Exception:
                    raise exc from None
                blocked.update(culprit)
                return result

        model._agenerate = _patched_agenerate

    # Also patch streaming paths — CLI/agent uses _stream/_astream, so without
    # these the content flattening is bypassed during normal streaming calls.
    orig_stream = getattr(model, "_stream", None)
    if orig_stream is not None:

        @functools.wraps(orig_stream)
        def _patched_stream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            prepared = _prepare(messages)
            suspects = _media_types_in(messages) - blocked
            if not suspects:
                yield from orig_stream(prepared, *args, **kwargs)
                return
            started = False
            try:
                for chunk in orig_stream(prepared, *args, **kwargs):
                    started = True
                    yield chunk
                return
            except Exception as exc:
                culprit = _media_error_types(exc) & suspects
                if started or (not culprit and not _is_http_400(exc)):
                    raise
                media_exc = exc
                media_culprit = culprit
            retry_started = False
            try:
                for chunk in orig_stream(
                    _stripped(messages, suspects), *args, **kwargs
                ):
                    if not retry_started:
                        retry_started = True
                        blocked.update(media_culprit)
                    yield chunk
            except Exception:
                if retry_started:
                    raise
                raise media_exc from None
            if not retry_started:
                raise media_exc from None  # stripped retry produced nothing

        model._stream = _patched_stream

    orig_astream = getattr(model, "_astream", None)
    if orig_astream is not None:

        @functools.wraps(orig_astream)
        async def _patched_astream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            prepared = _prepare(messages)
            suspects = _media_types_in(messages) - blocked
            if not suspects:
                async for chunk in orig_astream(prepared, *args, **kwargs):
                    yield chunk
                return
            started = False
            try:
                async for chunk in orig_astream(prepared, *args, **kwargs):
                    started = True
                    yield chunk
                return
            except Exception as exc:
                culprit = _media_error_types(exc) & suspects
                if started or (not culprit and not _is_http_400(exc)):
                    raise
                media_exc = exc
                media_culprit = culprit
            retry_started = False
            try:
                async for chunk in orig_astream(
                    _stripped(messages, suspects), *args, **kwargs
                ):
                    if not retry_started:
                        retry_started = True
                        blocked.update(media_culprit)
                    yield chunk
            except Exception:
                if retry_started:
                    raise
                raise media_exc from None
            if not retry_started:
                raise media_exc from None  # stripped retry produced nothing

        model._astream = _patched_astream


# ---------------------------------------------------------------------------
# Patch: ccproxy Codex Responses API rejects "system" role messages.
# Convert SystemMessage to use "developer" role via langchain-openai's
# __openai_role__ mechanism.
# ---------------------------------------------------------------------------
def _patch_ccproxy_system_to_developer(model: Any) -> None:
    """Convert SystemMessage role from 'system' to 'developer' for ccproxy.

    ccproxy's Responses API endpoint rejects system role messages with
    400 "System messages are not allowed".  LangChain's ``langchain_openai``
    checks ``additional_kwargs["__openai_role__"]`` and uses that value as
    the message role when serializing to the API.

    Args:
        model: A LangChain chat model instance to patch in-place.
    """
    import copy
    import functools

    from langchain_core.messages import BaseMessage, SystemMessage

    def _system_to_developer(messages: list[BaseMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                if msg.additional_kwargs.get("__openai_role__") != "developer":
                    msg = copy.copy(msg)
                    msg.additional_kwargs = {
                        **msg.additional_kwargs,
                        "__openai_role__": "developer",
                    }
            out.append(msg)
        return out

    orig_generate = getattr(model, "_generate", None)
    if orig_generate is None:
        return

    @functools.wraps(orig_generate)
    def _patched_generate(
        messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Any:
        return orig_generate(_system_to_developer(messages), *args, **kwargs)

    model._generate = _patched_generate

    orig_agenerate = getattr(model, "_agenerate", None)
    if orig_agenerate is not None:

        @functools.wraps(orig_agenerate)
        async def _patched_agenerate(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return await orig_agenerate(_system_to_developer(messages), *args, **kwargs)

        model._agenerate = _patched_agenerate

    orig_stream = getattr(model, "_stream", None)
    if orig_stream is not None:

        @functools.wraps(orig_stream)
        def _patched_stream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return orig_stream(_system_to_developer(messages), *args, **kwargs)

        model._stream = _patched_stream

    orig_astream = getattr(model, "_astream", None)
    if orig_astream is not None:

        @functools.wraps(orig_astream)
        async def _patched_astream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            async for chunk in orig_astream(
                _system_to_developer(messages), *args, **kwargs
            ):
                yield chunk

        model._astream = _patched_astream


# ---------------------------------------------------------------------------
# Patch (module-level): langchain-openai's _convert_dict_to_message and
# _convert_delta_to_message_chunk discard provider-specific fields like
# `reasoning_content`. We monkey-patch them to capture reasoning_content
# into AIMessage.additional_kwargs so downstream code (incl. our passback
# patch) can find it. Benign for non-DeepSeek providers — they just don't
# return this field, so the patch is a no-op for them.
# ---------------------------------------------------------------------------
_openai_capture_patched = False


def _patch_openai_capture_reasoning_content() -> None:
    global _openai_capture_patched
    if _openai_capture_patched:
        return
    try:
        import langchain_openai.chat_models.base as _base

        _orig_dict_to_msg = _base._convert_dict_to_message
        _orig_delta_to_chunk = _base._convert_delta_to_message_chunk

        def _patched_dict_to_msg(_dict, *args, **kwargs):
            msg = _orig_dict_to_msg(_dict, *args, **kwargs)
            rc = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
            if isinstance(rc, str) and rc and hasattr(msg, "additional_kwargs"):
                msg.additional_kwargs["reasoning_content"] = rc
            return msg

        def _patched_delta_to_chunk(_dict, *args, **kwargs):
            chunk = _orig_delta_to_chunk(_dict, *args, **kwargs)
            rc = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
            if isinstance(rc, str) and rc and hasattr(chunk, "additional_kwargs"):
                # Per-chunk: stash this delta's reasoning_content on the chunk.
                # Cross-chunk accumulation is handled by AIMessageChunk.__add__
                # via merge_dicts (string values in additional_kwargs concatenate).
                chunk.additional_kwargs["reasoning_content"] = (
                    chunk.additional_kwargs.get("reasoning_content", "") + rc
                )
            return chunk

        _base._convert_dict_to_message = _patched_dict_to_msg
        _base._convert_delta_to_message_chunk = _patched_delta_to_chunk
        _openai_capture_patched = True
    except Exception:
        pass


_patch_openai_capture_reasoning_content()


# ---------------------------------------------------------------------------
# Patch: DeepSeek thinking mode requires reasoning_content to be passed back
# in all assistant messages for multi-turn + tool_use scenarios.
# langchain-openai's _convert_message_to_dict drops this field, causing
# HTTP 400 "The reasoning_content in the thinking mode must be passed back".
# Mirrors langchain-ai/langchain PR #34516 (which patches langchain-deepseek;
# we apply equivalent logic to a langchain-openai ChatOpenAI instance).
# ---------------------------------------------------------------------------
def _patch_deepseek_reasoning_passback(model: Any) -> None:
    """Inject reasoning_content into outgoing payload assistant messages.

    DeepSeek V4 thinking mode + tool_use requires every historical assistant
    message to carry its reasoning_content as a top-level field (sibling to
    content / tool_calls).  Without this, multi-turn requests fail with 400.

    For assistant messages where no reasoning_content was captured (e.g.
    history left over from another provider, from DeepSeek Flash, or from an
    older EvoSci version that ran before the capture patch landed), we
    inject an empty string.  This satisfies DeepSeek's format requirement
    in thinking mode.  Non-thinking DeepSeek endpoints are believed to
    accept the extra field without complaint based on observed behavior,
    but this has not been independently audited; if a future DeepSeek
    release rejects empty reasoning_content on non-thinking models, this
    fallback would need a per-call thinking-mode check instead of a blanket
    inject.  The check is intentionally not gated on model name: this
    function is only mounted when provider == "deepseek" (see
    EvoScientist/llm/models.py), so all callers are DeepSeek endpoints.

    Args:
        model: A langchain-openai ChatOpenAI instance configured for DeepSeek.
    """
    import functools

    from langchain_core.messages import AIMessage

    orig = getattr(model, "_get_request_payload", None)
    if orig is None:
        return

    import logging as _logging

    _logger = _logging.getLogger(__name__)

    @functools.wraps(orig)
    def _patched(input_: Any, *, stop: Any = None, **kwargs: Any) -> dict:
        try:
            lc_messages = model._convert_input(input_).to_messages()
        except Exception:
            _logger.warning(
                "DeepSeek passback patch: _convert_input failed, "
                "falling back to unpatched payload (reasoning_content "
                "will not be injected)",
                exc_info=True,
            )
            return orig(input_, stop=stop, **kwargs)

        ai_rcs: list[str | None] = [
            m.additional_kwargs.get("reasoning_content")
            for m in lc_messages
            if isinstance(m, AIMessage)
        ]

        payload = orig(input_, stop=stop, **kwargs)
        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            return payload

        ai_idx = 0
        for msg in msgs:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            rc = ai_rcs[ai_idx] if ai_idx < len(ai_rcs) else None
            if rc:
                msg["reasoning_content"] = rc
            elif "reasoning_content" not in msg:
                # Empty-string fallback for ALL DeepSeek models (not just
                # reasoner). Required when history contains AI messages that
                # came from a different provider (Anthropic / OpenAI /
                # DeepSeek Flash) or from an older EvoSci that didn't capture
                # reasoning_content. Empirically tolerated by non-thinking
                # DeepSeek endpoints; see docstring for the audit caveat.
                msg["reasoning_content"] = ""
            ai_idx += 1

        return payload

    model._get_request_payload = _patched


# ---------------------------------------------------------------------------
# Patch: forward CLI's live (model, model_provider) into deepagents'
# start_async_task / update_async_task tool calls so the deployed graph
# (running in a separate ``langgraph dev`` subprocess) re-resolves the
# chat model per run.
#
# Without this, async sub-agents stay on the model their graph was compiled
# with at langgraph dev boot — `/model` switches in the CLI never reach
# them because they live in another process.
#
# Mechanism: wrap deepagents' ``_build_start_tool`` and ``_build_update_tool``
# factories. Each wrapped factory calls the original with a proxied client
# cache that intercepts ``runs.create(...)`` calls only and injects
# ``config={"configurable": {"model": <cfg.model>, "model_provider": <cfg.provider>}}``.
# All other client methods (``threads.create``, ``runs.get``, ``runs.cancel``,
# ``runs.join_stream``) pass through unchanged. The deployed graph picks up
# ``configurable.model`` via ``ConfigurableModelMiddleware``.
#
# Reads ``_ensure_config()`` at tool-call time (not patch time) so a
# ``/model`` switch in the CLI is reflected on the very next async tool
# call without an agent rebuild.
#
# Upstream PR opportunity: passing ``config`` through ``client.runs.create``
# is generic functionality; worth contributing back to ``langchain-ai/deepagents``
# so this patch can be retired.
# ---------------------------------------------------------------------------
_model_passthrough_patched = False


def _read_cfg_configurable() -> dict[str, str]:
    """Read live ``(model, provider)`` from EvoScientist config.

    Returns a dict suitable for inserting under
    ``RunnableConfig.configurable``. Empty dict on any failure (so the
    patch degrades to a no-op rather than breaking async tool calls).
    """
    try:
        from EvoScientist.EvoScientist import _ensure_config

        cfg = _ensure_config()
    except Exception:
        return {}

    out: dict[str, str] = {}
    model = getattr(cfg, "model", None)
    provider = getattr(cfg, "provider", None)
    if isinstance(model, str) and model:
        out["model"] = model
    if isinstance(provider, str) and provider:
        out["model_provider"] = provider
    return out


def _merge_runs_config_kwargs(kwargs: dict) -> dict:
    """Merge the live model override into ``kwargs`` for ``runs.create``.

    Preserves any caller-supplied ``config.configurable`` keys. EvoScientist's
    keys take precedence on conflict (callers shouldn't be passing model
    overrides — the CLI is the source of truth).
    """
    overrides = _read_cfg_configurable()
    if not overrides:
        return kwargs
    existing = kwargs.get("config")
    if not isinstance(existing, dict):
        existing = {}
    existing_configurable = existing.get("configurable")
    if not isinstance(existing_configurable, dict):
        existing_configurable = {}
    merged_configurable = {**existing_configurable, **overrides}
    kwargs = dict(kwargs)
    kwargs["config"] = {**existing, "configurable": merged_configurable}
    return kwargs


class _SyncRunsProxy:
    """Wraps a sync ``RunsClient`` and injects config into ``create`` only."""

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def create(self, **kwargs: Any) -> Any:
        return self._real.create(**_merge_runs_config_kwargs(kwargs))


class _AsyncRunsProxy:
    """Wraps an async ``RunsClient`` and injects config into ``create`` only."""

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def create(self, **kwargs: Any) -> Any:
        return await self._real.create(**_merge_runs_config_kwargs(kwargs))


class _ClientProxy:
    """Lightweight proxy that swaps ``client.runs`` for a runs proxy.

    Read-only forwarding: only ``__getattr__`` is overridden. Attribute
    *writes* on the proxy land on the proxy itself, NOT on the wrapped real
    client. Current deepagents only reads ``.threads`` / ``.runs``, so this
    is safe — but if a future caller tries ``client.foo = bar`` through the
    proxy, the write will be silently lost. ``object.__setattr__`` in
    ``__init__`` is used solely to avoid infinite recursion when seeding
    the internal slots.

    ``client.runs`` is a stable attribute set in
    ``langgraph_sdk.client.LangGraphClient.__init__`` (``self.runs =
    RunsClient(...)``) — not a property or lazy initializer — so the wrapped
    runs proxy is built once at ``__init__`` time and reused on every
    ``proxy.runs`` access. This avoids the previous behavior of selecting
    sync/async and instantiating a new ``_RunsProxy`` per attribute access.
    """

    def __init__(self, real: Any, *, is_async: bool) -> None:
        runs_proxy_cls = _AsyncRunsProxy if is_async else _SyncRunsProxy
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_runs_proxy", runs_proxy_cls(real.runs))

    def __getattr__(self, name: str) -> Any:
        if name == "runs":
            return self._runs_proxy
        return getattr(self._real, name)


class _ClientCacheProxy:
    """Proxy a ``_ClientCache`` so callers receive config-injecting clients."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def get_sync(self, name: str) -> Any:
        return _ClientProxy(self._real.get_sync(name), is_async=False)

    def get_async(self, name: str) -> Any:
        return _ClientProxy(self._real.get_async(name), is_async=True)


def _patch_deepagents_model_passthrough() -> None:
    """Wrap deepagents' async-launch tool factories to inject CLI model.

    Idempotent: re-invocation is a no-op once the patch is active. Safe to
    call from ``_maybe_swap_async_subagents`` on every CLI startup; both
    that hook and this patch turn on together when async sub-agents are
    enabled.
    """
    global _model_passthrough_patched
    if _model_passthrough_patched:
        return

    try:
        from deepagents.middleware import async_subagents as ds_mod
    except ImportError:
        return

    # Defensive ``getattr`` lookups mirror the rest of this file (lines 254,
    # 266, 279, 290, 339, 351, 362, 373, 473): a deepagents update that
    # renames or removes either private helper degrades to a no-op instead
    # of raising ``AttributeError`` at CLI startup.
    orig_build_start = getattr(ds_mod, "_build_start_tool", None)
    orig_build_update = getattr(ds_mod, "_build_update_tool", None)
    if orig_build_start is None or orig_build_update is None:
        return

    def _patched_build_start(
        agent_map: Any, clients: Any, tool_description: str
    ) -> Any:
        return orig_build_start(agent_map, _ClientCacheProxy(clients), tool_description)

    def _patched_build_update(agent_map: Any, clients: Any) -> Any:
        return orig_build_update(agent_map, _ClientCacheProxy(clients))

    ds_mod._build_start_tool = _patched_build_start
    ds_mod._build_update_tool = _patched_build_update
    _model_passthrough_patched = True
