"""Monkey-patches and utilities for third-party LangChain provider quirks.

All patches follow the same pattern: wrap an existing method/function to
fix upstream bugs, applied at import time or on first use.

Patches:
    - _patch_anthropic_proxy_compat: ccproxy dict→Pydantic model mismatch
    - _patch_openrouter_reasoning_details: reasoning_details schema errors
    - _patch_openai_compat_content: list content→string for strict APIs

Utilities:
    - strip_thinking_tags: remove ccproxy <thinking> tags from content
    - _flatten_message_content: convert content blocks to plain string
"""

from __future__ import annotations

import re
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
# Utility: ccproxy Codex embeds thinking as <thinking>...</thinking> tags
# inside the content string. Strip these so they don't appear in output.
# ---------------------------------------------------------------------------
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)


def strip_thinking_tags(content: str) -> str:
    """Remove ``<thinking>...</thinking>`` tags from ccproxy response content."""
    return _THINKING_TAG_RE.sub("", content)


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
