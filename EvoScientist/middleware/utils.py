"""Shared utilities for EvoScientist middleware.

Functions here are used by multiple middleware modules (memory, tool_selector)
and should not depend on any specific middleware class.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


def disable_thinking(model: BaseChatModel) -> BaseChatModel:
    """Return a copy of the model with thinking/reasoning disabled.

    Anthropic's API does not allow extended thinking when ``tool_choice``
    forces tool use (as ``with_structured_output`` does).  Similarly,
    OpenAI reasoning can conflict.  Strip these settings so structured
    output calls work reliably.

    DeepSeek enables thinking server-side by default (no client field to
    clear), and its thinking mode rejects the forced ``tool_choice`` that
    ``with_structured_output`` sends ("Thinking mode does not support this
    tool_choice"). For DeepSeek models the copy gets an explicit
    ``extra_body["thinking"] = {"type": "disabled"}`` request field instead.

    Uses ``model_copy()`` to produce a real new instance — ``bind()`` only
    wraps the model in a ``RunnableBinding`` whose kwargs do NOT override
    first-class Pydantic fields like ``thinking`` on ``ChatAnthropic``.
    """
    from ..llm.errors import _provider_from_model

    updates: dict[str, Any] = {}
    model_kwargs = getattr(model, "model_kwargs", {}) or {}

    if getattr(model, "thinking", None) or "thinking" in model_kwargs:
        updates["thinking"] = None
    if getattr(model, "reasoning", None) or "reasoning" in model_kwargs:
        updates["reasoning"] = None

    if _provider_from_model(model) == "deepseek":
        from ..llm.deepseek import (
            DEEPSEEK_THINKING_DISABLED,
            is_deepseek_thinking_disabled,
        )

        extra_body = dict(getattr(model, "extra_body", None) or {})
        if not is_deepseek_thinking_disabled(extra_body):
            extra_body["thinking"] = dict(DEEPSEEK_THINKING_DISABLED)
            updates["extra_body"] = extra_body

    if not updates:
        return model

    # Prefer Pydantic model_copy (creates a true new instance with the
    # field cleared) over bind() which only adds invocation kwargs.
    try:
        return model.model_copy(update=updates)
    except Exception:
        # Fallback for non-Pydantic or unusual model classes
        # Note: bind() may not effectively override first-class Pydantic fields
        return model.bind(**updates)


def append_to_system_message(
    system_message: SystemMessage | None, text: str
) -> SystemMessage:
    """Append a text block to a system message, preserving its metadata.

    Used by the memory and scheduler middleware. Unlike building a fresh
    ``SystemMessage``, ``model_copy`` keeps ``additional_kwargs`` (e.g.
    ``cache_control`` prompt-cache breakpoints), ``id``, ``name`` and
    ``response_metadata`` from the original message.
    """
    existing_blocks = list(system_message.content_blocks) if system_message else []
    new_blocks = [*existing_blocks, {"type": "text", "text": text}]
    if system_message is None:
        return SystemMessage(content=new_blocks)
    return system_message.model_copy(update={"content": new_blocks})
