"""ActiveTeamMiddleware for EvoScientist agent-teams v1.

Reads ``configurable.active_teams: list[str]`` on every model call and
appends a system-prompt cue biasing the main agent to consult the
user-invited expert(s).

The cue names the expert and stops there. It deliberately does not
prescribe a tool: every expert is reachable both in-turn (``task``) and in
the background (``start_async_task``), and choosing between them is the
orchestrator's job per task. The tool shapes, the output-path convention
and the result-envelope handling live once in ``DELEGATION_STRATEGY``
rather than being restated per active expert.

Backend-stateless team binding: WebUI sends ``active_teams`` on every
``stream.submit()`` for as long as the invited expert is active; this
middleware reads it fresh per turn via ``langgraph.config.get_config()``.
Matches the plan's decision to reach for the ``configurable`` primitive
rather than a server-side thread-state store (CLAUDE.md #5).

Naming note: the WIRE FORMAT is ``configurable.active_teams`` (plural,
legacy from the earlier "teams" framing that survived the pivot per the
WebUI section of the design note). Under the current expert-skill
mechanism the semantic content is a list of expert names, but the
wire key stays ``active_teams`` for WebUI compatibility. Internal
system-prompt tags use ``<active_expert>`` / ``<active_experts>``
because that matches what the LLM sees as the semantic target.

No-op when:
- ``configurable.active_teams`` is absent, empty, non-list, or contains
  no non-empty string entries.
- The middleware is invoked outside a runnable context (``get_config``
  raises).

Not included in the async-subagent middleware stack: an expert running
as its own graph would otherwise inject a "prefer expert X" cue into
its own system prompt, where the persona is already baked in. See
``EvoScientist.py::_get_default_middleware``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

_TEMPLATE_SINGLE = (
    "<active_expert>\n"
    "The user has invited the expert `{expert}` to this thread. Prefer it "
    "for requests within its scope. It stays available for the whole "
    "session until the user dismisses it.\n"
    "</active_expert>"
)

_TEMPLATE_MULTI = (
    "<active_experts>\n"
    "The user has invited the following experts to this thread: {experts}. "
    "Consult the right one for the current request; do not consult an expert "
    "if the request is clearly outside its scope. They stay available for "
    "the whole session until the user dismisses them.\n"
    "</active_experts>"
)


def _read_active_teams() -> list[str]:
    """Read ``configurable.active_teams`` from the current RunnableConfig.

    Returns an empty list when the config is absent, malformed, or the
    call happens outside a runnable context.
    """
    try:
        from langgraph.config import get_config

        cfg = get_config()
    except Exception:
        # Outside a runnable context (most common in tests) or
        # langgraph not importable — nothing to inject.
        return []
    if not isinstance(cfg, dict):
        return []
    configurable = cfg.get("configurable") or {}
    if not isinstance(configurable, dict):
        return []
    raw = configurable.get("active_teams")
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, str) and t]


def _dispatchable_names() -> set[str]:
    """Return the names of experts the orchestrator can currently reach.

    Fresh filesystem read every call so a ``skill_manager install <expert>``
    is visible on the next turn without an agent rebuild. Cheap at current
    scale (a handful of skills, cached bodies).

    Sourced from ``list_dispatchable_experts``, which drops empty-body
    experts and names colliding with reserved sub-agents. Keeps the cue
    honest: naming an expert the model cannot reach is worse than saying
    nothing.

    On import failure returns an empty set — the middleware then emits no
    cue, matching the outside-runnable-context no-op path.
    """
    try:
        from ..subagents.expert_container import list_dispatchable_experts
    except Exception:
        return set()
    try:
        return {s.name for s in list_dispatchable_experts()}
    except Exception:
        return set()


class ActiveTeamMiddleware(AgentMiddleware):
    """Bias delegation toward the user's active expert(s) on every turn."""

    name = "active_team"

    def _cue_for(self, experts: list[str]) -> str:
        """Render the cue over the dispatchable subset of ``experts``.

        Invited experts that aren't currently dispatchable (uninstalled,
        empty actor definition, name collision) are dropped from the cue —
        naming an expert the model cannot reach is worse than saying
        nothing. Returns the empty string when nothing survives the
        filter; caller skips the system-prompt append in that case.
        """
        reachable = _dispatchable_names()
        experts = [e for e in experts if e in reachable]
        if not experts:
            return ""
        if len(experts) == 1:
            return _TEMPLATE_SINGLE.format(expert=experts[0])
        return _TEMPLATE_MULTI.format(
            experts=", ".join(f"`{e}`" for e in experts),
        )

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Append the active-expert cue to the request's system message."""
        experts = _read_active_teams()
        if not experts:
            return request
        cue = self._cue_for(experts)
        if not cue:
            return request
        from .utils import append_to_system_message

        new_system = append_to_system_message(request.system_message, cue)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self.modify_request(request))


def create_active_team_middleware() -> ActiveTeamMiddleware:
    """Build ActiveTeamMiddleware."""
    return ActiveTeamMiddleware()
