"""ContextEditingMiddleware configuration for EvoScientist.

Wraps LangChain's built-in ``ContextEditingMiddleware`` with project-specific
defaults: dynamic trigger based on model context window, ``keep=5`` for
multi-step tool chains, and ``think_tool`` excluded from clearing.

Usage::

    from EvoScientist.middleware import create_context_editing_middleware

    middleware = create_context_editing_middleware(model)
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def compute_context_editing_trigger(
    model: BaseChatModel,
    fraction: float = 0.50,
    fallback: int = 100_000,
) -> int:
    """Compute ClearToolUsesEdit trigger based on model context window.

    Uses 50% of ``max_input_tokens`` when a model profile is available,
    otherwise falls back to a fixed token count.  This fires well before
    ``SummarizationMiddleware`` (~85% / 170k).
    """
    profile = getattr(model, "profile", None)
    if (
        profile is not None
        and isinstance(profile, dict)
        and isinstance(profile.get("max_input_tokens"), int)
        and profile["max_input_tokens"] > 0
    ):
        return int(profile["max_input_tokens"] * fraction)
    return fallback


def create_context_editing_middleware(model: BaseChatModel | None = None):
    """Build a ContextEditingMiddleware with EvoScientist defaults.

    Args:
        model: Chat model used to determine context window size.
            If *None*, the default model is resolved via ``_ensure_chat_model()``.
    """
    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

    if model is None:
        from EvoScientist.EvoScientist import _ensure_chat_model

        model = _ensure_chat_model()

    return ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=compute_context_editing_trigger(model),
                keep=5,
                exclude_tools=["think_tool"],
            ),
        ],
    )
