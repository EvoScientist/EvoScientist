"""Shared utilities for EvoScientist middleware.

Functions here are used by multiple middleware modules (memory, tool_selector)
and should not depend on any specific middleware class.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def disable_thinking(model: BaseChatModel) -> BaseChatModel:
    """Return a copy of the model with thinking/reasoning disabled.

    Anthropic's API does not allow extended thinking when ``tool_choice``
    forces tool use (as ``with_structured_output`` does).  Similarly,
    OpenAI reasoning can conflict.  Strip these settings so structured
    output calls work reliably.

    Uses ``model_copy()`` to produce a real new instance — ``bind()`` only
    wraps the model in a ``RunnableBinding`` whose kwargs do NOT override
    first-class Pydantic fields like ``thinking`` on ``ChatAnthropic``.
    """
    updates: dict[str, Any] = {}
    model_kwargs = getattr(model, "model_kwargs", {}) or {}

    if getattr(model, "thinking", None) or "thinking" in model_kwargs:
        updates["thinking"] = None
    if getattr(model, "reasoning", None) or "reasoning" in model_kwargs:
        updates["reasoning"] = None

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
