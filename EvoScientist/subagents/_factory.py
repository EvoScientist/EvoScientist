"""Factory for building deployable sub-agent graphs from yaml definitions.

Lives in ``EvoScientist/subagents/`` next to the canonical yaml entries
because the factory is "build a graph from a sub-agent name" — a generic
construction utility, not a deployment concern. Any deployment surface
(``EvoScientist/langgraph_dev/``, future ``langgraph_platform/``, custom
servers) can call ``build_async_subagent_graph(name)`` to materialize the
runnable graph.

Reuses the main EvoScientist agent's chat model, backend, and middleware so
the deployed sub-agent has full capability parity with its in-process
synchronous counterpart: same workspace files, same ``/skills/`` and
``/memories/`` routes, same error-handling and context-overflow middleware.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from deepagents.middleware.rubric import (
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricMiddleware,
)
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Async research agents (no approval path) keep the backend guard forced on;
# internal graphs (scheduler, evomemory, autoskills) run unguarded.
_GUARDED_ASYNC_SUBAGENTS = frozenset({"writing-agent", "data-analysis-agent"})

# Read-only slice of the filesystem tools handed to the scheduler's grader. A
# rubric names its deliverables, so reading them is enough; ``grep``/``glob``
# invited whole-workspace scans (15s timeouts per call on large workspaces).
_SCHEDULER_GRADER_TOOLS = ("ls", "read_file")

# Structured-output strategy per OpenRouter model family. langchain picks the
# grader's strategy from the model profile plus a model-name regex table, and
# OpenRouter breaks each family the other way round: Gemini tool schemas lose
# the criteria ``oneOf`` (every entry comes back null), Anthropic JSON mode
# returns non-JSON. Verified live 2026-09-04. Passed explicitly because a
# profile pin loses to the name regex (``anthropic/claude-fable-5``).
_OPENROUTER_GRADER_STRATEGY: dict[str, type[ProviderStrategy] | type[ToolStrategy]] = {
    "google/": ProviderStrategy,
    "anthropic/": ToolStrategy,
}

# Model calls one grader attempt may spend before the run fails closed with
# ``grader_error``. Without it a parse-error retry loop inherits the scheduler
# graph's recursion limit and spins for minutes. Sized for a 3-5 bullet rubric
# over a few files: ``ls`` + one ``read_file`` per file + the verdict call.
_SCHEDULER_GRADER_MAX_CALLS = 12


# OpenRouter ids for which neither strategy yields a verdict (probed 2026-09-04):
# the route rejects forced ``tool_choice`` (reasoning on or off) and JSON mode
# drops required fields. Exact ids, not families: ``anthropic/claude-fable-5``
# and the native ``claude-fable-5-1`` grade fine.
_OPENROUTER_UNGRADABLE_IDS = ("anthropic/claude-fable-5.1",)


def _warn_if_grader_unsupported(model: BaseChatModel) -> None:
    from EvoScientist.llm.errors import _provider_from_model

    if _provider_from_model(model) != "openrouter":
        return
    model_id = (getattr(model, "model_name", None) or "").lower()
    if model_id in _OPENROUTER_UNGRADABLE_IDS:
        logger.warning(
            "scheduler rubric: grader model %s via OpenRouter cannot return "
            "structured verdicts (this route rejects forced tool_choice and its "
            "JSON mode drops required fields); rubric runs will end in "
            "grader_error. Use the native anthropic provider for this model, or "
            "set auxiliary_model to another model (claude-fable-5, Sonnet, Haiku "
            "and Gemini all grade through OpenRouter).",
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


class _SchedulerRubricMiddleware(RubricMiddleware):
    """``RubricMiddleware`` whose grader gets an explicit structured-output strategy.

    Mirrors upstream ``_ensure_grader`` except for ``response_format``; a bare
    ``GraderResponse`` there lets langchain choose the strategy, which is wrong
    on OpenRouter (see ``_OPENROUTER_GRADER_STRATEGY``).
    """

    def _ensure_grader(self) -> Any:
        if self._grader is not None:
            return self._grader
        from deepagents._models import resolve_model

        resolved_model = resolve_model(self._model)
        self._resolved_model = resolved_model
        self._grader = create_agent(
            model=resolved_model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=self._grader_middleware,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=_grader_strategy(resolved_model) or GraderResponse,
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

    name = "scheduler_rubric_grader_budget"

    def __init__(self, *, max_calls: int) -> None:
        self.max_calls = max_calls

    def _check(self, request: Any) -> None:
        spent = sum(isinstance(m, AIMessage) for m in request.messages)
        if spent >= self.max_calls:
            msg = (
                f"scheduler rubric grader exceeded {self.max_calls} model calls "
                "without a verdict"
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
        "scheduler rubric iteration %s: %s — %s",
        evaluation.get("iteration"),
        evaluation.get("result"),
        evaluation.get("explanation"),
    )


def _scheduler_rubric_middleware(*, model: BaseChatModel, backend: BackendProtocol):
    """Acceptance grading for unattended scheduler runs (no-op without a rubric).

    Must be mounted LAST: ``after_agent`` hooks run in reverse list order, so
    the grader sees the finished run first and a ``needs_revision`` verdict
    jumps back to the model before ``EvoMemoryLifecycleMiddleware`` launches
    its memory worker — the worker fires once, on the accepted run. The grader
    reads the same backend because the deliverables are files the transcript
    alone cannot prove exist.
    """
    import warnings

    from deepagents import FilesystemMiddleware
    from langchain_core._api import LangChainBetaWarning

    # Eviction thresholds off: both eviction paths write files through the
    # backend, which would let the grader touch the shared workspace.
    grader_fs = FilesystemMiddleware(
        backend=backend,
        tools=list(_SCHEDULER_GRADER_TOOLS),
        tool_token_limit_before_evict=None,
        human_message_token_limit_before_evict=None,
    )
    _warn_if_grader_unsupported(model)
    with warnings.catch_warnings():
        # Beta API; graphs build at langgraph dev import, keep the log clean.
        warnings.simplefilter("ignore", LangChainBetaWarning)
        return _SchedulerRubricMiddleware(
            model=model,
            grader_middleware=[
                grader_fs,
                _GraderCallBudget(max_calls=_SCHEDULER_GRADER_MAX_CALLS),
            ],
            max_iterations=2,
            on_evaluation=_log_rubric_evaluation,
        )


def build_async_subagent_graph(name: str) -> Any:
    """Build a deployable graph for the ``name`` sub-agent defined in yaml.

    Args:
        name: The sub-agent's key in one of the ``EvoScientist/subagents/*.yaml``
            files (e.g. ``"writing-agent"``).

    Returns:
        A compiled ``langgraph`` graph ready for registration in ``langgraph.json``.

    Raises:
        ValueError: If ``name`` is not defined under ``EvoScientist/subagents/``.
    """
    # Lazy imports — the factory is invoked at langgraph dev startup time, so
    # all heavy modules (deepagents, llm, MCP) are pulled in here rather than
    # at package import.
    from deepagents import create_deep_agent

    from EvoScientist.config import apply_config_to_env, get_effective_config
    from EvoScientist.EvoScientist import (
        SUBAGENTS_CONFIG,
        _ensure_auxiliary_chat_model,
        _ensure_chat_model,
        _ensure_general_purpose_subagent,
        _get_default_backend,
        _get_default_middleware,
        _inject_subagent_middleware,
    )
    from EvoScientist.tools import skill_manager, tavily_search, think_tool
    from EvoScientist.utils import load_subagents, resolve_subagent_tools

    # Surface API keys as env vars so downstream SDKs (openai, anthropic, …)
    # find them on subprocess invocations from langgraph dev.
    cfg = get_effective_config()
    apply_config_to_env(cfg)

    # Mirror the tool registry constructed in EvoScientist._build_base_kwargs.
    tool_registry = {"think_tool": think_tool, "skill_manager": skill_manager}
    if os.environ.get("TAVILY_API_KEY"):
        tool_registry["tavily_search"] = tavily_search

    # Use the official loader so resolved tools, prompt_refs, and skills are
    # all wired the same way as the in-process sync version.
    specs = load_subagents(
        SUBAGENTS_CONFIG,
    )
    spec = next((s for s in specs if s.get("name") == name), None)
    if spec is None:
        raise ValueError(
            f"Sub-agent {name!r} not found in {SUBAGENTS_CONFIG}. "
            f"Available: {[s.get('name') for s in specs]}"
        )
    resolve_subagent_tools(spec, tool_registry)

    # Load MCP tools routed to THIS agent via ``expose_to: <name>`` in
    # ``mcp.yaml``. Use the cached helper so multiple ``build_async_subagent_graph``
    # calls in the same langgraph dev subprocess (one per registered async graph)
    # share a single MCP connection set per server instead of re-spawning.
    from EvoScientist.EvoScientist import _load_mcp_tools_cached

    mcp_tools_by_agent = _load_mcp_tools_cached()
    agent_mcp_tools = mcp_tools_by_agent.get(name, [])

    # NOTE on HITL: async sub-agents intentionally do NOT set ``interrupt_on``,
    # even though the deployed main agent does. They run as standalone graphs
    # on the langgraph dev subprocess; the parent (CLI main agent) only sees a
    # ``task_id`` from ``start_async_task`` and has no UI path to surface a
    # paused-on-interrupt child to the user. Setting ``interrupt_on`` here
    # would hang the sub-agent on its first ``execute`` call with no one to
    # approve. The user-visible HITL boundary is the parent's
    # ``start_async_task`` decision; restrict the child's reach by limiting
    # ``tools`` in ``subagents/<name>.yaml`` instead.
    #
    # ``for_async_subagent=True`` propagates the same reasoning to the
    # middleware list — specifically, it suppresses ``AskUserMiddleware``,
    # which uses ``interrupt()`` for the same purpose (waiting on a user
    # reply) and would deadlock an async sub-agent for the same reason.
    #
    # Memory middleware is included so async sub-agents get the same profile
    # context and `/memories/profile/...` file guidance as the main agent.
    subagents = []
    _ensure_general_purpose_subagent(subagents)
    _inject_subagent_middleware(subagents)

    middleware = _get_default_middleware(
        for_async_subagent=True,
        memory_source_agent=name,
    )

    # Scheduler is an unattended timer task → use the cheaper auxiliary model.
    model = (
        _ensure_auxiliary_chat_model() if name == "scheduler" else _ensure_chat_model()
    )

    guarded = name in _GUARDED_ASYNC_SUBAGENTS
    backend = _get_default_backend(guard_dangerous=guarded, refuse_delete=guarded)
    if name == "scheduler":
        middleware = [
            *middleware,
            _scheduler_rubric_middleware(model=model, backend=backend),
        ]
    return create_deep_agent(
        name=name,
        model=model,
        system_prompt=spec.get("system_prompt", ""),
        tools=spec.get("tools", []) + agent_mcp_tools,
        skills=spec.get("skills"),
        backend=backend,
        middleware=middleware,
        subagents=subagents,
    ).with_config({"recursion_limit": cfg.recursion_limit})
