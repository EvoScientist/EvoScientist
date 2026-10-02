"""ActiveTeamMiddleware: the expert prompt for the main agent.

Injects the ``## Experts`` concept into the system message on every
main-agent turn, so the expert mechanism is always visible — mirroring how
the skill system's guidance is always present. When the user has invited
experts (``configurable.active_teams``), an ``<active_expert>`` block naming
the reachable ones is appended on top.

An expert is a fractal of a skill, so this block is ordered (via the
middleware stack in ``_get_default_middleware``) to land right after
``## Skills System``. Gating the whole block on invitation is the trap this
design avoids: the expert mechanism must not disappear when nothing is
invited, and the invited-expert list must not read as a standalone "always
dispatch an expert" directive.

Backend-stateless team binding: WebUI sends ``active_teams`` on every
``stream.submit()`` for as long as the invited expert is active; this
middleware reads it fresh per turn via ``langgraph.config.get_config()`` —
the ``configurable`` primitive, not a server-side thread-state store
(CLAUDE.md #5). The wire key stays ``active_teams`` (plural, legacy from the
earlier "teams" framing); the semantic content is a list of expert names.

Not included in the async-subagent middleware stack: an expert running as
its own graph would otherwise inject the expert prompt into its own system
message, where its persona is already baked in. See
``EvoScientist.py::_get_default_middleware``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

# The expert concept — injected on every main-agent turn so the mechanism is
# always visible, mirroring how ``## Skills System`` is always present. Moved
# here from ``DELEGATION_STRATEGY`` (an expert is a fractal of a skill, so its
# guidance belongs next to the skill system's). The invited-expert list is
# appended below only when the user has invited experts this session.
EXPERTS_CONCEPT = """## Experts
An expert is an installed skill that also ships an actor definition — a persona and a result-envelope contract. Every installed expert is reachable both ways, and the choice is yours per task, not fixed per expert:

- `task({subagent_type: '<expert>', description: ...})` — runs in-turn and returns into the current turn. Use when the answer is short and the user is waiting on it.
- `start_async_task(subagent_type: '<expert>', description: ...)` — runs in the background, returns a task ID immediately. Use when the work is long-running or its deliverable is a file. **Name a concrete output path in the description** (e.g. "write to `./artifacts/<expert>/<slug>.md`") — the expert honours the path you give it. On `status: 'success'`, `check_async_task` returns a `result` envelope with `output_path`, a one-paragraph `summary`, and an expert-defined `metadata` block; render `summary` and `metadata` to the user directly rather than re-reading the artifact to build a synopsis.

Prefer the background form when unsure — expert work is usually multi-step, and it keeps the conversation responsive.

You need not dispatch at all. An expert's `SKILL.md` is ordinary knowledge on the `/skills/` mount: read it and do the work yourself when the task is small, or when the full conversation context matters more than a fresh sub-agent would. If an `<active_expert>` block appears below, the user invited that expert specifically — prefer it for requests in its scope."""

# Appended to ``EXPERTS_CONCEPT`` only when the user has invited reachable
# experts. One ``<active_expert>`` tag handles one or many names.
_INVITE_TEMPLATE = (
    "\n\n<active_expert>\n"
    "The user has invited {experts} to this thread. Prefer the right one for "
    "requests within its scope; do not consult an expert if the request is "
    "clearly outside its scope. They stay available for the whole session "
    "until the user dismisses them.\n"
    "</active_expert>"
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

    def _invite_block(self, experts: list[str]) -> str:
        """Render the ``<active_expert>`` block over the dispatchable subset.

        Invited experts that aren't currently dispatchable (uninstalled,
        empty actor definition, name collision) are dropped — naming an
        expert the model cannot reach is worse than saying nothing. Returns
        the empty string when nothing survives the filter.
        """
        reachable = _dispatchable_names()
        experts = [e for e in experts if e in reachable]
        if not experts:
            return ""
        names = ", ".join(f"`{e}`" for e in experts)
        return _INVITE_TEMPLATE.format(experts=names)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Append the expert concept (always) plus the invited-expert block
        (when the user has invited reachable experts) to the system message.
        """
        block = EXPERTS_CONCEPT
        invited = _read_active_teams()
        if invited:
            block += self._invite_block(invited)
        from .utils import append_to_system_message

        new_system = append_to_system_message(request.system_message, block)
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
