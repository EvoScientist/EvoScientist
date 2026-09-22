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

import contextvars
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
    """Delegate attribute reads to the per-task model but report the synced window.

    langchain's ``SummarizationMiddleware`` helper resolves fraction-based
    trigger and keep clauses through ``self.model.profile`` (plain attribute
    read) and provider-matches reported token usage through
    ``self.model._get_ls_params()``. Re-pointing ``_lc_helper.model`` at this
    shim makes the first read see the per-run context window and the second
    fall through to the run's model. The summary model is NOT affected: it
    was captured at construction (``_summary_model = self.model.with_retry()``).

    The shim is installed ONCE and never mutated afterwards: both the window
    and the delegation target are read from ``state_var`` — the
    ``contextvars.ContextVar`` the owning middleware ``.set()``s per task —
    so concurrent runs with different windows never see each other's state.
    In a task where no override is synced, the shim reports the construction
    window and delegates to the construction model, i.e. exactly what the
    bare construction model would have answered.
    """

    def __init__(
        self,
        state_var: contextvars.ContextVar[tuple[Any, int] | None],
        fallback_model: Any,
        fallback_window: int,
    ) -> None:
        self._state_var = state_var
        self._fallback_model = fallback_model
        self._fallback_window = fallback_window

    @property
    def profile(self) -> dict[str, int]:
        state = self._state_var.get()
        window = state[1] if state is not None else self._fallback_window
        return {"max_input_tokens": window}

    def __getattr__(self, name: str) -> Any:
        state = self._state_var.get()
        target = state[0] if state is not None else self._fallback_model
        return getattr(target, name)


class _PerRunLimitsSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware with per-run, per-task trigger/cutoff/overflow limits.

    Before every model call, ``_sync_limits`` resolves the run's model from
    ``configurable.model`` (falling back to the construction model when absent)
    and records ``(target_model, resolved_window)`` in a per-task
    ``contextvars.ContextVar``. Downstream, the window is read dynamically from
    that state:

    - ``_get_profile_limits`` — feeding deepagents' own fraction-clause call
      sites (args-truncation trigger/cutoff and the overflow tail-clip
      budget); and
    - ``_lc_helper.model`` (a profile shim) — feeding langchain's delegated
      fraction math (``_should_summarize``,
      ``_determine_cutoff_index`` / ``_find_token_based_cutoff``) with the same
      window; and
    - ``_input_budget`` — deepagents' over-budget check reads
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

    Concurrency (per-task isolation, #466 review): the synced state must NOT
    live on the shared instance, because deepagents' ``awrap_model_call``
    re-reads the window *after* awaits — the real model call, history offload
    and summary generation all precede the overflow-fallback reads
    (``_determine_cutoff_index``, ``_over_budget``/``_input_budget``, and the
    ``max_input_tokens`` handed to the overflow tail-clip). Two interleaved
    sessions with different model windows would otherwise overwrite each
    other's thresholds mid-call. A ``ContextVar`` fixes this: every
    ``asyncio`` task starts from a copy of its parent's context (the same
    propagation ``_read_model_override`` relies on to see the run's
    ``RunnableConfig``), and every OS thread starts with a fresh context, so
    async-task and sync-thread runs are isolated alike. The only shared
    writes left are the one-time shim install (idempotent — racing
    installers write equivalent shims reading the same ContextVar) and the
    resolution caches (``_model_cache`` / ``_window_cache`` are safe to share:
    resolution is deterministic and plain dict get/set are atomic, so the
    worst race is a duplicate resolution producing an identical value).
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
        # Construction-window fallback for the profile shim, resolved once
        # here so unsynced tasks never re-resolve it. Deliberately resolved,
        # not ``get_context_window``: the shim must always answer an int.
        self._construction_window: int = resolve_context_window(construction_model)
        # Per-task synced state: ``(target_model, resolved_window)`` or None.
        # The default is None and __init__ never sets it — "no override synced
        # in this task" means stock delegation semantics (see
        # ``_get_profile_limits`` / ``_input_budget``), so middleware
        # instances that never see an override behave exactly like stock.
        self._synced_state: contextvars.ContextVar[tuple[Any, int] | None] = (
            contextvars.ContextVar("evo_summarization_synced_state", default=None)
        )
        # Keyed by the resolved context window, not the model object: real
        # chat models are unhashable pydantic objects, and the limits are a
        # pure function of the window. Shared across tasks on purpose — see
        # the class docstring's concurrency note.
        self._window_cache: dict[int | None, int] = {}
        # (model, provider) -> resolved chat model, mirroring
        # ConfigurableModelMiddleware's resolution cache. Shared on purpose.
        self._model_cache: dict[tuple[str, str | None], Any] = {}

    def _sync_limits(self) -> None:
        """Record the current run's context window in per-task state.

        Resolves the run's model, then publishes ``(target, window)`` through
        the ContextVar so every later read in this task — including the
        post-``await`` reads on deepagents' overflow fallback path — keeps
        seeing this run's window no matter what other tasks sync in between.
        Writing the var is the only per-run mutation; it lands in this task's
        private context copy, never on the shared instance.
        """
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
        state = self._synced_state.get()
        if state is not None and state[0] is target:
            return
        if target is self._construction_model:
            # No override in effect for this task — clear the synced state so
            # reads delegate to the stock construction-profile behavior.
            if state is not None:
                self._synced_state.set(None)
            return
        window = get_context_window(target)
        resolved = self._window_cache.get(window)
        if resolved is None:
            resolved = resolve_context_window(target)
            self._window_cache[window] = resolved
        # One-time shim install (identity check): the shared write happens
        # only here, and every installer writes an equivalent immutable shim
        # that reads the same ContextVar, so a racing double install is
        # harmless (last write wins).
        if not isinstance(self._lc_helper.model, _ProfileWindowShim):
            self._lc_helper.model = _ProfileWindowShim(
                self._synced_state,
                self._construction_model,
                self._construction_window,
            )
        self._synced_state.set((target, resolved))

    def wrap_model_call(self, request: ModelRequest, handler):
        self._sync_limits()
        return super().wrap_model_call(request, handler)

    async def awrap_model_call(self, request: ModelRequest, handler) -> Any:
        self._sync_limits()
        return await super().awrap_model_call(request, handler)

    def _get_profile_limits(self) -> int | None:
        state = self._synced_state.get()
        if state is not None:
            return state[1]
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
        state = self._synced_state.get()
        if state is None:
            return super()._input_budget(request)
        output = 0
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = request.model_settings.get(key, getattr(request.model, key, None))
            if isinstance(value, int) and not isinstance(value, bool):
                output = max(output, value)
        return max(0, int(state[1] * 0.95) - output)


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
