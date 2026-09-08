"""ContextEditingMiddleware configuration for EvoScientist.

Wraps LangChain's built-in ``ContextEditingMiddleware`` with project-specific
defaults: dynamic trigger based on model context window, ``keep=5`` for
multi-step tool chains, and ``think_tool`` excluded from clearing.

Usage::

    from EvoScientist.middleware import create_context_editing_middleware

    middleware = create_context_editing_middleware(model)
"""

from __future__ import annotations

from typing import Any
from weakref import WeakKeyDictionary

from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.language_models import BaseChatModel

from ..llm.context_window import get_context_window


def compute_context_editing_trigger(
    model: BaseChatModel,
    fraction: float = 0.50,
    fallback: int = 100_000,
) -> int:
    """Compute ClearToolUsesEdit trigger based on model context window.

    Uses 50% of the best available model context window when metadata is
    available, otherwise falls back to a fixed token count. This fires well
    before ``SummarizationMiddleware`` (~85% / 170k).
    """
    context_window = get_context_window(model)
    if context_window is not None and context_window > 0:
        return max(1, int(context_window * fraction))
    return fallback


class _PerRunTriggerContextEditingMiddleware(ContextEditingMiddleware):
    """ContextEditingMiddleware whose edit trigger tracks the run's model.

    The base class applies its edits inside ``(a)wrap_model_call``, where the
    current run's model is available on ``request.model`` — which, on the
    server backend, is already the per-run ``configurable.model`` override
    (``ConfigurableModelMiddleware`` sits earlier in the stack and swaps it).
    This subclass resizes the first edit's trigger there, whenever the model
    differs from the one the current trigger was computed for. Triggers are
    cached per model object (``ConfigurableModelMiddleware`` returns the same
    instance per ``(model, provider)`` pair, so steady-state model calls cost
    one dict lookup).

    The trigger mutation happens right before the synchronous edit pass with
    no ``await`` in between, so async runs on one event loop cannot interleave
    it. Parallel sync subagent runs in threads could in principle race the
    mutation; the worst case is one run trimming with the other model's
    trigger — the same over/under-trim the frozen-trigger design had.
    """

    def __init__(
        self,
        *,
        edits: list[Any],
        construction_model: BaseChatModel,
    ) -> None:
        super().__init__(edits=edits)
        self._trigger_model: Any = construction_model
        self._trigger_cache: WeakKeyDictionary = WeakKeyDictionary()

    def _sync_trigger(self, model: Any) -> None:
        if model is self._trigger_model:
            return
        trigger = self._trigger_cache.get(model)
        if trigger is None:
            trigger = compute_context_editing_trigger(model)
            try:
                self._trigger_cache[model] = trigger
            except TypeError:  # unhashable model object - skip caching
                pass
        self._trigger_model = model
        self.edits[0].trigger = trigger

    def wrap_model_call(self, request, handler):
        self._sync_trigger(request.model)
        return super().wrap_model_call(request, handler)

    async def awrap_model_call(self, request, handler) -> Any:
        self._sync_trigger(request.model)
        return await super().awrap_model_call(request, handler)


def create_context_editing_middleware(model: BaseChatModel | None = None):
    """Build a ContextEditingMiddleware with EvoScientist defaults.

    Args:
        model: Chat model used to determine the initial trigger's context
            window. If *None*, the default model is resolved via
            ``_ensure_chat_model()``.
    """
    if model is None:
        from EvoScientist.EvoScientist import _ensure_chat_model

        model = _ensure_chat_model()

    return _PerRunTriggerContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=compute_context_editing_trigger(model),
                keep=5,
                exclude_tools=["think_tool"],
            ),
        ],
        construction_model=model,
    )
