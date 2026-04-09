"""Monkey-patches and utilities for third-party LangChain provider quirks.

All patches follow the same pattern: wrap an existing method/function to
fix upstream bugs, applied at import time or on first use.

Patches:
    - _patch_anthropic_proxy_compat: ccproxy dict→Pydantic model mismatch
    - _patch_openrouter_reasoning_details: reasoning_details schema errors
    - _patch_openai_compat_content: list content→string for strict APIs
    - _patch_ccproxy_codex_compat: ccproxy model fixes + langchain None guard
    - _patch_ccproxy_system_to_developer: system→developer role for ccproxy

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
                if not output:
                    # output is None/empty — force rebuild from accumulated items
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
# Patch: langchain-openrouter v0.2.1 — _convert_message_to_dict() serializes
# reasoning_details back to the API, but streaming chunks use wrong field
# names per type (thinking→content, reasoning.summary→summary,
# reasoning.encrypted→data), causing Pydantic errors on multi-turn.
# Fix: wrap the function to drop reasoning_details from output.
# ---------------------------------------------------------------------------
_openrouter_patched = False


def _patch_openrouter_reasoning_details() -> None:
    global _openrouter_patched
    if _openrouter_patched:
        return
    try:
        import langchain_openrouter.chat_models as _mod

        _orig = _mod._convert_message_to_dict

        def _patched(message: Any) -> Any:
            result = _orig(message)
            result.pop("reasoning_details", None)
            return result

        _mod._convert_message_to_dict = _patched
        _openrouter_patched = True
    except Exception:
        pass


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
# list rather than a string.
# ---------------------------------------------------------------------------
_SKIP_CONTENT_TYPES = frozenset({"thinking", "reasoning", "reasoning_content"})


def _flatten_message_content(content: Any) -> str | Any:
    """Convert list-of-blocks content to a plain string.

    Args:
        content: Message content — either a string, a list of content blocks
            (dicts with ``type`` and ``text`` keys), or another type.

    Returns:
        A plain string with text blocks joined by double newlines.
        Thinking/reasoning blocks are skipped.  Non-list input is
        returned unchanged.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in _SKIP_CONTENT_TYPES:
                continue
            text = block.get("text")
            if text:
                parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    return "\n\n".join(parts) if parts else ""


def _patch_openai_compat_content(model: Any) -> None:
    """Flatten list content to strings before OpenAI-compatible API calls.

    Wraps ``_generate`` / ``_agenerate`` to prevent "invalid type: sequence,
    expected a string" errors from strict APIs like DeepSeek.

    Args:
        model: A LangChain chat model instance to patch in-place.
    """
    import copy
    import functools

    from langchain_core.messages import BaseMessage

    def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg.content, list):
                msg = copy.copy(msg)
                msg.content = _flatten_message_content(msg.content)
            out.append(msg)
        return out

    orig_generate = getattr(model, "_generate", None)
    if orig_generate is None:
        return

    @functools.wraps(orig_generate)
    def _patched_generate(
        messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Any:
        return orig_generate(_sanitize_messages(messages), *args, **kwargs)

    model._generate = _patched_generate

    orig_agenerate = getattr(model, "_agenerate", None)
    if orig_agenerate is not None:

        @functools.wraps(orig_agenerate)
        async def _patched_agenerate(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return await orig_agenerate(_sanitize_messages(messages), *args, **kwargs)

        model._agenerate = _patched_agenerate

    # Also patch streaming paths — CLI/agent uses _stream/_astream, so without
    # these the content flattening is bypassed during normal streaming calls.
    orig_stream = getattr(model, "_stream", None)
    if orig_stream is not None:

        @functools.wraps(orig_stream)
        def _patched_stream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return orig_stream(_sanitize_messages(messages), *args, **kwargs)

        model._stream = _patched_stream

    orig_astream = getattr(model, "_astream", None)
    if orig_astream is not None:

        @functools.wraps(orig_astream)
        async def _patched_astream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            async for chunk in orig_astream(
                _sanitize_messages(messages), *args, **kwargs
            ):
                yield chunk

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
            return await orig_agenerate(
                _system_to_developer(messages), *args, **kwargs
            )

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
