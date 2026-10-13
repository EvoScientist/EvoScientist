"""Acceptance grading shared by the main agent and the scheduler graph.

A run whose input carries ``rubric`` is graded when the agent stops and sent
back to the model until the grader is satisfied or the pass limit is reached.
On the main agent the run's config must also carry the ``rubric_run`` marker.
Without ``rubric`` the middleware does nothing and builds no grader model.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from deepagents.middleware.rubric import (
    GRADER_SYSTEM_PROMPT,
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricMiddleware,
)
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# Read-only slice of the filesystem tools handed to the grader. A rubric names
# its deliverables, so reading them is enough; ``grep``/``glob`` invited
# whole-workspace scans (15s timeouts per call on large workspaces).
GRADER_TOOLS = ("ls", "read_file")

# Structured-output strategy per OpenRouter model family. langchain picks the
# grader's strategy from the model profile plus a model-name regex table, and
# OpenRouter breaks each family differently: Gemini tool schemas lose the
# criteria ``oneOf`` (every entry comes back null), Anthropic JSON mode returns
# non-JSON, OpenAI's strict JSON mode rejects that ``oneOf`` with a 400.
# Verified live 2026-09-04 and 2026-09-26. Passed explicitly because a profile
# pin loses to the name regex (``anthropic/claude-fable-5``).
_OPENROUTER_GRADER_STRATEGY: dict[str, type[ProviderStrategy] | type[ToolStrategy]] = {
    "google/": ProviderStrategy,
    "anthropic/": ToolStrategy,
    "openai/": ToolStrategy,
}

# Model calls one grader attempt may spend before the run fails closed with
# ``grader_error``. Without it a parse-error retry loop inherits the graph's
# recursion limit and spins for minutes. Sized for a 3-5 bullet rubric over a
# few files: ``ls`` + one ``read_file`` per file + the verdict call.
GRADER_MAX_CALLS = 12

# Grading passes per run: the main agent, and the cheaper unattended scheduler.
MAIN_RUBRIC_MAX_ITERATIONS = 3
SCHEDULER_RUBRIC_MAX_ITERATIONS = 2

# Appended to the upstream grader prompt. Without it the grader either fails a
# file criterion the transcript never mentions, or walks the whole workspace
# looking for a missing file until the call budget ends the pass.
GRADER_FILE_CHECK_PROMPT = (
    "\n\nYou have read-only `ls` and `read_file` tools rooted at the workspace (`/`). "
    "The transcript is bounded and can omit or shorten earlier messages, so when a "
    "criterion is about a file, check that file on disk before you decide it either "
    "way: do not fail it only because the transcript does not show the file, and do "
    "not pass it only because the transcript says it was written. Open only files "
    "that a criterion or the transcript names. A file that is not at the named path "
    "is missing: fail that criterion and return your verdict without looking through "
    "other files or directories."
)

# OpenRouter ids for which neither strategy yields a verdict (re-probed 2026-09-22,
# sonnet-5.5 probed 2026-09-29): the model rejects forced ``tool_choice`` and JSON
# mode rejects the grader schema's ``oneOf``. Exact ids, not families:
# ``anthropic/claude-fable-5`` and the native ``claude-fable-5-1`` /
# ``claude-opus-5-5`` grade fine.
_OPENROUTER_UNGRADABLE_IDS = (
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
)


def _warn_if_grader_unsupported(model: BaseChatModel) -> None:
    from EvoScientist.llm.errors import _provider_from_model

    if _provider_from_model(model) != "openrouter":
        return
    model_id = (getattr(model, "model_name", None) or "").lower()
    if model_id in _OPENROUTER_UNGRADABLE_IDS:
        logger.warning(
            "rubric: grader model %s via OpenRouter cannot return "
            "structured verdicts (this route rejects forced tool_choice and its "
            "JSON mode rejects the grader schema); rubric runs will end in "
            "grader_error. Use the native anthropic provider for this model, or "
            "set auxiliary_model to another model (claude-fable-5, claude-sonnet-5, "
            "Haiku, GPT and Gemini all grade through OpenRouter).",
            model_id,
        )


def _grader_strategy(model: BaseChatModel) -> ProviderStrategy | ToolStrategy | None:
    """Explicit grader strategy on OpenRouter routes; ``None`` defers to langchain."""
    from EvoScientist.llm.errors import _provider_from_model

    if _provider_from_model(model) != "openrouter":
        return None
    model_id = (getattr(model, "model_name", None) or "").lower()
    for prefix, strategy in _OPENROUTER_GRADER_STRATEGY.items():
        if model_id.startswith(prefix):
            return strategy(GraderResponse)
    return None


def _run_carries_rubric_marker() -> bool:
    """Whether the current run's config marks it as carrying a rubric."""
    from langgraph.config import get_config

    from EvoScientist.backends import RUBRIC_RUN_KEY

    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return False
    return configurable.get(RUBRIC_RUN_KEY) is True


class EvoRubricMiddleware(RubricMiddleware):
    """``RubricMiddleware`` adapted to a rubric that belongs to one run.

    Every run that carries a rubric starts a new grading attempt, a final
    verdict clears the rubric from the thread, and the unsupported-grader
    warning is logged on the first graded run. With ``require_run_marker`` an
    invocation is graded only when its config carries ``RUBRIC_RUN_KEY``; any
    other one drops a rubric it finds in the thread. ``_ensure_grader`` mirrors
    upstream except for the model and ``response_format``: a bare
    ``GraderResponse`` there lets langchain choose the strategy, which is
    wrong on OpenRouter (see ``_OPENROUTER_GRADER_STRATEGY``) and native
    DeepSeek (see ``structured_output_for``).
    """

    _warned_unsupported = False

    def __init__(self, *, require_run_marker: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._require_run_marker = require_run_marker

    def _warn_unsupported_once(self, state: Any) -> None:
        if self._warned_unsupported or not state.get("rubric"):
            return
        self._warned_unsupported = True
        _warn_if_grader_unsupported(self._model)

    def _ungraded_run_update(self, state: Any) -> tuple[bool, dict[str, Any] | None]:
        """Whether this run must not be graded, and the update that drops its rubric."""
        if not self._require_run_marker or _run_carries_rubric_marker():
            return False, None
        if not state.get("rubric"):
            return True, None
        logger.debug(
            "rubric: dropped from a run whose config carries no rubric_run "
            "marker; the run is not graded"
        )
        return True, {"rubric": None}

    def before_agent(self, state, runtime):
        ungraded, update = self._ungraded_run_update(state)
        if ungraded:
            return update
        self._warn_unsupported_once(state)
        return super().before_agent(state, runtime)

    async def abefore_agent(self, state, runtime):
        ungraded, update = self._ungraded_run_update(state)
        if ungraded:
            return update
        self._warn_unsupported_once(state)
        return await super().abefore_agent(state, runtime)

    # Checked again before grading: a run parked mid-way can be continued by a
    # client that never asked for it to be graded.
    @hook_config(can_jump_to=["model"])
    def after_agent(self, state, runtime):
        ungraded, update = self._ungraded_run_update(state)
        if ungraded:
            return update
        return super().after_agent(state, runtime)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state, runtime):
        ungraded, update = self._ungraded_run_update(state)
        if ungraded:
            return update
        return await super().aafter_agent(state, runtime)

    def _reset_for_new_rubric(self, state):
        # A rubric belongs to one run here, so a run that carries one always
        # starts a new attempt; upstream would continue the count of a run
        # that was cut short with the same rubric text.
        return super()._reset_for_new_rubric({**state, "_active_rubric": None})

    def _compose_update(self, state, evaluation):
        update = super()._compose_update(state, evaluation)
        if evaluation["result"] != "needs_revision":
            # The rubric belongs to the run that carried it.
            update = {**update, "rubric": None}
        return update

    def _handle_grader_exception(self, runtime, state, grading_run_id, iteration, exc):
        update = super()._handle_grader_exception(
            runtime, state, grading_run_id, iteration, exc
        )
        return {**update, "rubric": None}

    @staticmethod
    def _grader_invocation_config(metadata):
        # Upstream hands the grader the parent's metadata, agent name included,
        # so its work would be shown as a sub-agent called like the main agent.
        config = RubricMiddleware._grader_invocation_config(metadata)
        return {
            "metadata": {
                **config["metadata"],
                "lc_agent_name": RUBRIC_GRADER_MESSAGE_SOURCE,
            }
        }

    def _ensure_grader(self) -> Any:
        if self._grader is not None:
            return self._grader
        from deepagents._models import resolve_model

        from EvoScientist.middleware.utils import structured_output_for

        resolved_model = resolve_model(self._model)
        self._resolved_model = resolved_model
        model, response_format = resolved_model, _grader_strategy(resolved_model)
        if response_format is None:
            model, response_format = structured_output_for(
                resolved_model, GraderResponse
            )
        self._grader = create_agent(
            model=model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=self._grader_middleware,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=response_format,
            state_schema=self._grader_state_schema,
            context_schema=self._grader_context_schema,
        )
        return self._grader


class _GraderCallBudget(AgentMiddleware):
    """Fail closed once one grader attempt has made ``max_calls`` model calls.

    Counts the ``AIMessage``s already in the request, so the budget is per
    grader invocation by construction and no per-run state is needed. Raised
    on the first attempt it surfaces as ``grader_error``; raised on the
    coverage retry, upstream ``_grade`` swallows it and downgrades the first
    (unusable) verdict to ``needs_revision`` instead. Both terminate.
    """

    name = "rubric_grader_budget"

    def __init__(self, *, max_calls: int) -> None:
        self.max_calls = max_calls

    def _check(self, request: Any) -> None:
        spent = sum(isinstance(m, AIMessage) for m in request.messages)
        if spent >= self.max_calls:
            msg = (
                f"rubric grader exceeded {self.max_calls} model calls without a verdict"
            )
            raise RuntimeError(msg)

    def wrap_model_call(self, request, handler):
        self._check(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._check(request)
        return await handler(request)


def _log_rubric_evaluation(evaluation: RubricEvaluation) -> None:
    logger.info(
        "rubric iteration %s: %s — %s",
        evaluation.get("iteration"),
        evaluation.get("result"),
        evaluation.get("explanation"),
    )


def create_rubric_middleware(
    *,
    model: BaseChatModel,
    backend: BackendProtocol,
    max_iterations: int,
    prepare_messages: Callable[[list], list] | None = None,
    require_run_marker: bool = False,
) -> EvoRubricMiddleware:
    """Acceptance grading for a run that carries ``rubric`` (no-op otherwise).

    Must be mounted LAST: ``after_agent`` hooks run in reverse list order, so
    the grader sees the finished run first and a ``needs_revision`` verdict
    jumps back to the model before EvoScientist's other ``after_agent`` hooks
    fire (a harness profile's tail hooks still run first). The
    grader reads the same backend because the deliverables are files the
    transcript alone cannot prove exist.
    """
    from deepagents import FilesystemMiddleware
    from deepagents.middleware import UnsupportedContentMiddleware
    from langchain_core._api import LangChainBetaWarning

    # Eviction thresholds off: both eviction paths write files through the
    # backend, which would let the grader touch the shared workspace.
    grader_fs = FilesystemMiddleware(
        backend=backend,
        tools=list(GRADER_TOOLS),
        tool_token_limit_before_evict=None,
        human_message_token_limit_before_evict=None,
    )
    with warnings.catch_warnings():
        # Beta API; graphs build at langgraph dev import, keep the log clean.
        warnings.simplefilter("ignore", LangChainBetaWarning)
        return EvoRubricMiddleware(
            model=model,
            system_prompt=GRADER_SYSTEM_PROMPT + GRADER_FILE_CHECK_PROMPT,
            grader_middleware=[
                grader_fs,
                _GraderCallBudget(max_calls=GRADER_MAX_CALLS),
                # create_agent does not add it; last so it sees the final request.
                UnsupportedContentMiddleware(),
            ],
            prepare_messages_for_grader=prepare_messages,
            max_iterations=max_iterations,
            on_evaluation=_log_rubric_evaluation,
            require_run_marker=require_run_marker,
        )
