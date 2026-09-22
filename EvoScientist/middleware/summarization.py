"""SummarizationMiddleware whose context limits track the run's model (#466).

deepagents' built-in ``SummarizationMiddleware`` freezes its context limits on
the graph-construction model: langchain's helper resolves fraction-based
trigger/keep clauses via ``self.model.profile`` (the model passed at
construction), and deepagents' overflow fallback reads the same frozen limits.
When ``ConfigurableModelMiddleware`` swaps the model per run via
``configurable.model`` / ``configurable.model_provider``, the summarization
trigger does not adapt — a smaller-window model overflows the provider instead
of being auto-summarized, and a larger-window model gets summarized far too
early.

``request.model`` cannot be used here (unlike
``_PerRunTriggerContextEditingMiddleware``): the middleware list order is
outermost-first and deepagents' summarization slot sits in the *core* stack,
OUTSIDE the user block where ``ConfigurableModelMiddleware`` lives — so when
summarization's ``wrap_model_call`` runs, ``request.model`` is still the
construction model. The run's model must be resolved from
``langgraph.config.get_config()`` directly, mirroring
``ConfigurableModelMiddleware._read_model_override``.

Usage (wired in ``EvoScientist.EvoScientist._get_default_middleware``)::

    from EvoScientist.middleware import create_per_run_summarization_middleware

    middleware = create_per_run_summarization_middleware(construction_model, backend)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents.middleware.summarization import (
    DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain_core.messages.utils import count_tokens_approximately

from ..llm.context_window import get_context_window, resolve_context_window
from .configurable_model import _read_model_override

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest

logger = logging.getLogger(__name__)


class _ProfileWindowShim:
    """Delegate every attribute to *model* but report a chosen profile window.

    langchain's ``SummarizationMiddleware`` helper resolves fraction-based
    trigger and keep clauses through ``self.model.profile`` (plain attribute —
    set once in ``__init__``) and provider-matches reported token usage through
    ``self.model._get_ls_params()``. Re-pointing ``_lc_helper.model`` at this
    shim makes both reads see the per-run context window while everything else
    falls through to the real model. The summary model is NOT affected: it was
    captured at construction (``_summary_model = self.model.with_retry()``).
    """

    def __init__(self, model: Any, window: int) -> None:
        self._model = model
        self.profile = {"max_input_tokens": window}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class _PerRunLimitsSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware with per-run trigger/cutoff/overflow limits.

    Before every model call, ``_sync_limits`` resolves the run's model from
    ``configurable.model`` (falling back to the construction model when absent)
    and, when its resolved context window differs from the installed one:

    - overrides ``_get_profile_limits`` — feeding deepagents' own
      fraction-clause call sites (args-truncation trigger/cutoff and the
      overflow tail-clip budget); and
    - re-points ``_lc_helper.model`` at a profile shim — feeding langchain's
      delegated fraction math (``_should_summarize``,
      ``_determine_cutoff_index`` / ``_find_token_based_cutoff``) with the same
      window; and
    - overrides ``_input_budget`` — deepagents' over-budget check reads
      ``request.model.profile``, which at this layer is still the construction
      model (see module docstring), so the input budget is computed from the
      synced window instead.

    The resolved window is never ``None``: langchain fraction clauses silently
    never fire when ``_get_profile_limits()`` returns ``None``, so an
    unresolvable window falls back to ``resolve_context_window``'s 200k default.

    ``_summary_model`` deliberately stays the construction model (built once at
    langchain ``SummarizationMiddleware.__init__``): summaries don't switch
    models, only the trigger/cutoff/overflow-clip limits do. The approximate
    token counter likewise stays construction-tuned (chars-per-token heuristic
    differences across providers are noise next to the frozen-window bug).

    The sync mutation happens right before the (a)wrap_model_call super call
    with no ``await`` in between, so async runs on one event loop cannot
    interleave it. Parallel sync subagent runs in threads could in principle
    race the mutation; the worst case is one run summarizing against the other
    model's window — the same over/under-trigger the frozen-limits design had.
    """

    @property
    def name(self) -> str:
        """Report the public alias so deepagents' name-based merge REPLACES us.

        The base property deliberately drops the ``SummarizationMiddleware``
        alias for subclasses so user extensions don't shadow the built-in by
        accident. This subclass exists precisely to shadow it: returning the
        alias makes ``_apply_custom_middleware`` swap out the frozen-limit
        instance in place, landing us in the identical core-stack slot
        (preserving position relative to ``ContextOverflowMapperMiddleware``,
        whose mapped ``ContextOverflowError`` the wrapper catches on its
        overflow fallback path).
        """
        return "SummarizationMiddleware"

    def __init__(
        self,
        construction_model: Any,
        backend: Any,
        *,
        summary_prompt: str = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = None,
        token_counter: Any = count_tokens_approximately,
    ) -> None:
        # Mirror deepagents' ``create_summarization_middleware`` so the
        # in-place replacement is behaviorally identical at construction time.
        defaults = compute_summarization_defaults(construction_model)
        super().__init__(
            model=construction_model,
            backend=backend,
            trigger=defaults["trigger"],
            keep=defaults["keep"],
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            truncate_args_settings=defaults["truncate_args_settings"],
        )
        self._construction_model: Any = construction_model
        # Model whose limits are currently installed (identity short-circuit).
        self._limits_model: Any = construction_model
        # Keyed by the resolved context window, not the model object: real
        # chat models are unhashable pydantic objects, and the limits are a
        # pure function of the window.
        self._window_cache: dict[int | None, int] = {}
        # (model, provider) -> resolved chat model, mirroring
        # ConfigurableModelMiddleware's resolution cache.
        self._model_cache: dict[tuple[str, str | None], Any] = {}
        # Active per-run window; ``None`` means "no override synced yet —
        # delegate to the base implementation" (stock behavior).
        self._current_window: int | None = None

    def _sync_limits(self) -> None:
        """Install the current run's context window, if it changed."""
        model_name, provider = _read_model_override()
        if model_name is None:
            target = self._construction_model
        else:
            key = (model_name, provider)
            target = self._model_cache.get(key)
            if target is None:
                try:
                    from ..llm import get_chat_model

                    target = get_chat_model(model=model_name, provider=provider)
                except Exception:
                    logger.warning(
                        "SummarizationMiddleware failed to resolve model=%r "
                        "provider=%r; keeping construction-model context limits",
                        model_name,
                        provider,
                        exc_info=True,
                    )
                    target = self._construction_model
                else:
                    self._model_cache[key] = target
        if target is self._limits_model:
            return
        window = get_context_window(target)
        resolved = self._window_cache.get(window)
        if resolved is None:
            resolved = resolve_context_window(target)
            self._window_cache[window] = resolved
        self._limits_model = target
        self._current_window = resolved
        self._lc_helper.model = _ProfileWindowShim(target, resolved)

    def wrap_model_call(self, request: ModelRequest, handler):
        self._sync_limits()
        return super().wrap_model_call(request, handler)

    async def awrap_model_call(self, request: ModelRequest, handler) -> Any:
        self._sync_limits()
        return await super().awrap_model_call(request, handler)

    def _get_profile_limits(self) -> int | None:
        if self._current_window is not None:
            return self._current_window
        return super()._get_profile_limits()

    def _input_budget(self, request: ModelRequest) -> int | None:
        """Reserve configured output and 5% headroom from the run's window.

        Mirrors deepagents' implementation but sources the input limit from
        the synced per-run window: the stock version reads
        ``request.model.profile``, which at this layer is still the
        construction model (ConfigurableModelMiddleware swaps the model
        further in). The output-token reservation still honors the request's
        settings/model, exactly like the stock version.
        """
        if self._current_window is None:
            return super()._input_budget(request)
        output = 0
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = request.model_settings.get(key, getattr(request.model, key, None))
            if isinstance(value, int) and not isinstance(value, bool):
                output = max(output, value)
        return max(0, int(self._current_window * 0.95) - output)


def create_per_run_summarization_middleware(
    construction_model: Any,
    backend: Any,
) -> _PerRunLimitsSummarizationMiddleware:
    """Build a SummarizationMiddleware whose limits track the run's model.

    Args:
        construction_model: Chat model the graph was built with; sizes the
            construction-time defaults and generates the summaries.
        backend: Backend for conversation-history offload — must be the same
            backend the stock instance would have used, since deepagents'
            name-based merge replaces the built-in with this instance.
    """
    return _PerRunLimitsSummarizationMiddleware(construction_model, backend)
