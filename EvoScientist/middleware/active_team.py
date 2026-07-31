"""ActiveTeamMiddleware for EvoScientist agent-teams v1.

Reads ``configurable.active_teams: list[str]`` on every model call and
appends a system-prompt cue biasing the main agent to consult the
user-invited expert(s).

The cue is dispatch-aware: for each active expert the middleware looks up
its ``default_dispatch`` (via ``list_expert_skills()``) and emits the
tool-shape cue that matches how the expert actually runs:

- ``sync`` / ``panel`` / unset -> ``task({subagent_type: 'X', ...})``.
- ``async`` -> ``start_async_task(subagent_type: 'X', payload: {skill_name:
  'X', output_path: '...'})``, plus a reminder that ``check_async_task``
  returns the status/result later.

Without the dispatch-aware branch, an ``async`` expert like
``literature-review`` gets told to use ``task()``, which routes it back
through the sync ``SubAgentMiddleware`` rather than the async graph the
container registers for it.

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

# Per-expert cue shapes. Composed inside ``_TEMPLATE_SINGLE`` /
# ``_TEMPLATE_MULTI`` at render time so the wrapping tags stay in sync
# with the count of active experts.
_SYNC_CUE = (
    "Consult it via `task({{subagent_type: '{expert}', description: '...'}})`. "
    "It runs synchronously and returns its result to the same turn."
)

_ASYNC_CUE = (
    "Consult it via `start_async_task(description: '<describe the task AND "
    "embed a concrete output path — e.g. write to "
    "``./artifacts/{expert}/<slug>.md`` verbatim>', "
    "subagent_type: '{expert}')`. It runs in the background and returns a "
    "task_id immediately; the result artifact is written to the path you "
    "named in the description. Use ``check_async_task`` to poll status "
    "when the user asks. On ``status: 'success'`` the ``result`` field "
    "contains a JSON envelope with ``output_path``, a one-paragraph "
    "``summary``, and a skill-defined ``metadata`` block (fields vary by "
    "expert — e.g. ``word_count`` / ``section_count`` / ``citations_used`` "
    "for surveys). Render ``summary`` and ``metadata`` directly to the "
    "user — do not re-read the artifact to build a synopsis."
)

_TEMPLATE_SINGLE = (
    "<active_expert>\n"
    "The user has invited the expert `{expert}` to this thread. "
    "{cue} "
    "It stays available for the whole session until the user dismisses it.\n"
    "</active_expert>"
)

_TEMPLATE_MULTI_HEADER = (
    "<active_experts>\n"
    "The user has invited the following experts to this thread: {experts}. "
    "Consult the right one for the current request; do not consult an expert "
    "if the request is clearly outside its scope. Per-expert dispatch:\n"
)

_TEMPLATE_MULTI_FOOTER = "</active_experts>"


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


def _dispatch_by_name() -> dict[str, str]:
    """Return ``{skill_name: default_dispatch}`` for currently dispatchable experts.

    Fresh filesystem read every call so a ``skill_manager install <expert>``
    is visible on the next turn without an agent rebuild. Cheap at current
    scale (a handful of skills, cached bodies).

    Sourced from ``list_dispatchable_experts`` — which filters empty-body
    experts, name collisions with reserved sub-agents, AND async-declared
    experts when async dispatch is unavailable (``enable_async_subagents``
    off or langgraph dev unreachable). Keeps the cue honest: any expert
    named here can actually be reached by the tool shape the cue advertises.

    On import failure returns an empty dict — the middleware then emits no
    cue, matching the outside-runnable-context no-op path.
    """
    try:
        from ..subagents.expert_container import list_dispatchable_experts
    except Exception:
        return {}
    try:
        return {s.name: s.default_dispatch for s in list_dispatchable_experts()}
    except Exception:
        return {}


def _cue_shape_for(dispatch: str, expert: str) -> str:
    """Return the ``task()`` / ``start_async_task(...)`` cue for one expert."""
    if dispatch == "async":
        return _ASYNC_CUE.format(expert=expert)
    return _SYNC_CUE.format(expert=expert)


class ActiveTeamMiddleware(AgentMiddleware):
    """Bias delegation toward the user's active expert(s) on every turn."""

    name = "active_team"

    def _cue_for(self, experts: list[str]) -> str:
        """Render the cue over the dispatchable subset of ``experts``.

        Invited experts that aren't currently dispatchable (uninstalled,
        empty body, name collision, or async-declared while async
        dispatch is unavailable) are dropped from the cue — pointing the
        model at a tool shape that will fail is worse than saying nothing.
        Returns the empty string when nothing survives the filter; caller
        skips the system-prompt append in that case.
        """
        dispatch_map = _dispatch_by_name()
        experts = [e for e in experts if e in dispatch_map]
        if not experts:
            return ""
        if len(experts) == 1:
            expert = experts[0]
            cue = _cue_shape_for(dispatch_map[expert], expert)
            return _TEMPLATE_SINGLE.format(expert=expert, cue=cue)
        experts_str = ", ".join(f"`{e}`" for e in experts)
        per_expert_lines = "\n".join(
            f"- `{e}`: {_cue_shape_for(dispatch_map[e], e)}" for e in experts
        )
        return (
            _TEMPLATE_MULTI_HEADER.format(experts=experts_str)
            + per_expert_lines
            + "\n"
            + _TEMPLATE_MULTI_FOOTER
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
