"""Async container graph for expert-skill dispatch.

One generic graph that reads ``skill_name`` from initial state and loads
that expert skill's ``SKILL.md`` body as the sub-agent's system prompt at
invocation time. Registered once in ``langgraph.json``; parameterised per
run via the payload the main agent's ``start_async_task`` passes through
:class:`EvoScientist.middleware.expert_async_subagent.EvoAsyncSubAgentMiddleware`.

Rationale — see the follow-up entry in ``notes/teams-and-workflows/
agent-teams-design.md``: the alternative (one graph per expert, registered
statically in ``langgraph.json``) doesn't scale to ``skill_manager install
<expert>`` at runtime, because a new expert would need a repo edit + a
langgraph dev restart. The generic-container approach preserves the
"installable async expert" story.

State schema
------------
Extends ``DeepAgentState`` with ``skill_name`` and ``output_path`` as
optional keys. Both arrive via the ``payload`` the main agent passes; the
middleware validates presence and halts with a clear error message when
either is missing (rather than falling back to an ambient default that
would silently produce the wrong survey).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

from deepagents.graph import DeepAgentState
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage

_logger = logging.getLogger(__name__)


class ExpertContainerState(DeepAgentState):
    """State schema for the async expert container graph.

    Adds ``skill_name`` and ``output_path`` as ``NotRequired`` keys so the
    payload the main agent passes through
    ``EvoAsyncSubAgentMiddleware.start_async_task`` lands cleanly in the
    graph's initial state.
    """

    skill_name: NotRequired[str]
    output_path: NotRequired[str]


class ExpertSkillLoaderMiddleware(AgentMiddleware[Any, Any, Any]):
    """Load the expert skill's SKILL.md body as system prompt on every model call.

    Reads ``state.skill_name``, resolves the corresponding installed expert
    skill via ``list_expert_skills()``, composes the system message from
    ``role`` + SKILL.md body (mirrors the sync path in
    ``EvoScientist.subagents.expert_container._compose_system_prompt``),
    and overrides ``request.system_message`` before the handler runs.

    The container graph's static ``system_prompt`` at construction time is a
    minimal fallback; this middleware is the load-bearing component. If the
    skill_name is missing or resolves to no installed skill, appends an
    explicit error to the system message so the LLM immediately halts and
    returns an error envelope (rather than answering as an ambient
    generalist).
    """

    name = "expert_skill_loader"

    def _compose_prompt(self, state: dict[str, Any]) -> str:
        """Look up the skill and compose its system prompt.

        Returns the composed prompt string, or an error-cue string when the
        skill can't be loaded. Never raises — errors are surfaced through
        the LLM's system prompt so it can return a well-formed error
        envelope rather than crash the graph mid-turn.
        """
        skill_name = state.get("skill_name")
        if not skill_name:
            return (
                "ERROR: The async expert container was invoked without a "
                "``skill_name`` in state. This is a wiring bug in whichever "
                "middleware invoked ``start_async_task``. Return an error "
                "envelope naming the missing field and halt."
            )

        # Lazy import — the loader is a per-turn call, so this stays cheap.
        from ..tools.skills_manager import list_expert_skills

        experts = list_expert_skills(include_system=True)
        match = next((s for s in experts if s.name == skill_name), None)
        if match is None:
            installed = ", ".join(sorted(s.name for s in experts)) or "(none)"
            return (
                f"ERROR: Expert skill '{skill_name}' is not installed. "
                f"Installed experts: {installed}. Return an error envelope "
                "with status='error' explaining the skill is missing."
            )

        # Compose: role prepend (if present) + body.
        body = match.body or ""
        if match.role:
            return f"You are {match.role}.\n\n{body}".rstrip() + "\n"
        return body if body.endswith("\n") else body + "\n"

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        composed = self._compose_prompt(request.state)
        new_system = SystemMessage(content=composed)
        return handler(request.override(system_message=new_system))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        composed = self._compose_prompt(request.state)
        new_system = SystemMessage(content=composed)
        return await handler(request.override(system_message=new_system))


def build_expert_async_subagent_specs(cfg: Any | None = None) -> list[dict[str, Any]]:
    """Build ``AsyncSubAgent``-shaped specs for every ``default_dispatch: async`` expert.

    Each spec is a dict pointing at the shared ``expert-container-async`` graph
    with ``is_expert=True``. The main agent's
    ``EvoAsyncSubAgentMiddleware.start_async_task`` uses ``is_expert`` to
    require a ``payload`` including ``skill_name``.

    Returns an empty list when ``cfg.enable_async_subagents`` is not set, or
    when the langgraph dev subprocess isn't reachable. Both are the same
    conditions used by ``_maybe_swap_async_subagents`` to gate the existing
    ``writing-agent`` / ``data-analysis-agent`` / ``scheduler`` async
    subagents — keeps behaviour consistent across sync-fallback situations.
    """
    from ..config import get_effective_config

    cfg = cfg if cfg is not None else get_effective_config()
    if not getattr(cfg, "enable_async_subagents", False):
        return []
    # Same reachability guard used for standard async subagents in
    # ``_maybe_swap_async_subagents``.
    from ..langgraph_dev.manager import is_async_subagents_available

    if not is_async_subagents_available():
        return []

    from ..tools.skills_manager import list_expert_skills

    port = int(getattr(cfg, "langgraph_dev_port", 6174))
    specs: list[dict[str, Any]] = []
    for skill in list_expert_skills(include_system=True):
        if skill.default_dispatch != "async":
            continue
        specs.append(
            {
                "name": skill.name,
                "description": skill.description,
                "graph_id": "expert-container-async",
                "url": f"http://localhost:{port}",
                "is_expert": True,
            }
        )
    return specs


def build_expert_container_async_graph() -> Any:
    """Build the async expert container graph.

    Called once at langgraph dev startup. The returned graph accepts
    ``{messages, skill_name, output_path}`` as initial state; the
    :class:`ExpertSkillLoaderMiddleware` resolves ``skill_name`` on every
    model call and injects the matching SKILL.md body as system prompt.

    Tool set is intentionally minimal (``think_tool`` + ``skill_manager``)
    — matches the sync ``expert_container`` factory. Once the per-skill
    ``allowed-tools`` follow-up ships, the tool list will union with the
    skill's declared tools.
    """
    from deepagents import create_deep_agent

    from ..config import apply_config_to_env, get_effective_config
    from ..EvoScientist import (
        _ensure_chat_model,
        _ensure_general_purpose_subagent,
        _get_default_backend,
        _get_default_middleware,
        _inject_subagent_middleware,
    )
    from ..tools import skill_manager, think_tool

    cfg = get_effective_config()
    apply_config_to_env(cfg)

    _FALLBACK_SYSTEM_PROMPT = (
        "You are an expert sub-agent. Your specific role is loaded from your "
        "``skill_name`` at every model call. If you see this fallback prompt "
        "instead of your persona, the loader middleware failed — return an "
        "error envelope naming the failure and halt."
    )

    subagents: list[dict[str, Any]] = []
    _ensure_general_purpose_subagent(subagents)
    _inject_subagent_middleware(subagents)

    middleware = [
        # Loader runs FIRST so downstream middleware sees the composed
        # system_message. Ordering matters — put ExpertSkillLoaderMiddleware
        # before context editing / error normalisation so they operate on
        # the already-composed prompt.
        ExpertSkillLoaderMiddleware(),
        *_get_default_middleware(
            for_async_subagent=True,
            memory_source_agent="expert-container-async",
        ),
    ]

    return create_deep_agent(
        name="expert-container-async",
        model=_ensure_chat_model(),
        system_prompt=_FALLBACK_SYSTEM_PROMPT,
        tools=[think_tool, skill_manager],
        skills=["/skills/"],
        backend=_get_default_backend(),
        middleware=middleware,
        subagents=subagents,
        state_schema=ExpertContainerState,
    ).with_config({"recursion_limit": cfg.recursion_limit})
