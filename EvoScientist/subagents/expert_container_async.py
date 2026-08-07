"""Async container graph for expert-skill dispatch.

One generic graph that reads ``skill_name`` from initial state and loads
that expert skill's actor definition as the sub-agent's system prompt at
invocation time — its ``AGENTS.md`` body under the current skill contract,
or its ``SKILL.md`` body for legacy ``type: expert`` skills. Registered once
in ``langgraph.json``; parameterised per run via the payload the main
agent's ``start_async_task`` passes through
:class:`EvoScientist.middleware.expert_async_subagent.EvoAsyncSubAgentMiddleware`.

Why one shared graph rather than one graph per expert: the per-expert
alternative (each expert registered statically in ``langgraph.json``)
doesn't scale to ``skill_manager install <expert>`` at runtime, because a
new expert would need a repo edit + a langgraph dev restart. The
generic-container approach preserves the "installable async expert"
story.

State schema
------------
Extends ``DeepAgentState`` with ``skill_name`` as an optional key,
injected by construction when the spec is built. The middleware validates
its presence and halts with a clear error message when missing (rather
than falling back to an ambient default that would silently produce the
wrong survey). Run-specific values such as the output path travel in the
task description, not in state.
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

# Sentinel token embedded in ``_FALLBACK_SYSTEM_PROMPT`` so
# :class:`ExpertSkillLoaderMiddleware` can locate that block inside the
# base-stack-composed ``system_message`` and swap it for the persona in
# place — preserving the base-stack sections (todo guidance, ``## `task```,
# ``## Filesystem Tools``, ``## Skills System``, ``## Async subagents``,
# ...) that would otherwise be dropped by an ``override(system_message=...)``.
# Double-underscore ASCII cannot occur incidentally in skill bodies or
# base-stack prose, so a substring scan is precise. HTML-comment tokens
# were rejected because pretty-printers can strip them and Claude has
# been observed to echo them back into prose.
_PERSONA_SENTINEL = "__EVOEXPERT_PERSONA_SLOT__"

_FALLBACK_SYSTEM_PROMPT = (
    f"{_PERSONA_SENTINEL}\n\n"
    "Fallback: the expert loader middleware failed to resolve ``skill_name``. "
    "Return an error envelope naming the failure and halt."
)


class ExpertContainerState(DeepAgentState):
    """State schema for the async expert container graph.

    Adds ``skill_name`` as a ``NotRequired`` key —
    ``EvoAsyncSubAgentMiddleware`` injects it by construction from
    ``subagent_type`` so the loader middleware knows which persona to
    load. ``output_path`` is NOT in state; the main agent embeds the
    desired artifact path in the task description (natural language) and
    the expert's actor definition instructs the LLM to pin it in
    ``write_todos`` on turn 1 — surviving summarization via langgraph's
    todo composition.
    """

    skill_name: NotRequired[str]


class ExpertSkillLoaderMiddleware(AgentMiddleware[Any, Any, Any]):
    """Load the expert's actor definition as system prompt on every model call.

    Reads ``state.skill_name``, resolves the corresponding installed expert
    skill via ``list_expert_skills()``, composes the system message from
    ``role`` + the skill's prompt body (AGENTS.md under the current contract,
    SKILL.md for legacy frontmatter experts — resolved by
    ``expert_container.expert_prompt_body``, mirroring the sync path in
    ``expert_container._compose_system_prompt``), and overrides
    ``request.system_message`` before the handler runs.

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

        Emits a "Runtime context" tail re-asserting ``skill_name`` on every
        model call so the expert knows its own persona name after
        summarization. Any run-specific values the LLM needs (output path,
        user goal, ...) travel via the initial user message — the middleware
        does not touch them.
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

        # Mirrors the sync fold-in's empty-body skip in
        # ``expert_container.py::build_expert_subagent_specs``. A skill with
        # only a role line (or nothing) would otherwise run against a
        # persona-less system prompt — a worse failure mode than the expert
        # being absent. Prefer a well-formed error envelope over silent
        # nonsense.
        #
        # Which file is checked follows the skill's contract:
        # ``expert_prompt_body`` reads AGENTS.md for experts declared that
        # way and SKILL.md for legacy frontmatter experts. Checking ``.body``
        # directly would clear a paper-review-shaped skill on the strength of
        # its knowledge file while its actor definition is empty.
        from .expert_container import expert_prompt_body

        body = expert_prompt_body(match)
        if not body.strip():
            source_file = (
                "AGENTS.md" if match.expert_source == "agents_md" else "SKILL.md"
            )
            return (
                f"ERROR: Expert skill '{skill_name}' has an empty {source_file} "
                "body — the persona / pipeline the sub-agent needs is missing. "
                "This is a skill-authoring bug; the sub-agent cannot proceed. "
                "Return an error envelope with status='error' naming the "
                "empty skill."
            )

        # Compose: role prepend (if present) + body + runtime-context tail.
        head = f"You are {match.role}.\n\n" if match.role else ""
        runtime_block = (
            "\n---\n\n"
            "## Runtime context (injected by the container)\n\n"
            f"- ``skill_name``: ``{skill_name}``\n"
        )
        return (head + body).rstrip() + runtime_block

    def _compose_system_message(self, request: ModelRequest[Any]) -> SystemMessage:
        """Return the base-stack ``system_message`` with the persona swapped in.

        Locates the fallback block by ``_PERSONA_SENTINEL`` and replaces its
        text with the composed persona, preserving every other block
        (``## `task```, ``## Filesystem Tools``, ``## Skills System``, …).
        When the sentinel isn't found (e.g. deepagents renamed the
        ``system_prompt=`` handling), degrade to appending the persona so
        the base-stack sections stay live and the expert stays operational;
        surface the drift in logs so it can't rot silently.
        """
        from ..middleware.utils import (
            append_to_system_message,
            replace_block_by_sentinel,
        )

        composed = self._compose_prompt(request.state)
        new_system = replace_block_by_sentinel(
            request.system_message, _PERSONA_SENTINEL, composed
        )
        if new_system is None:
            _logger.warning(
                "Expert persona sentinel %r not found in system_message; "
                "appending persona instead of replacing the fallback block. "
                "Deepagents' base-stack composition may have changed.",
                _PERSONA_SENTINEL,
            )
            new_system = append_to_system_message(request.system_message, composed)
        return new_system

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(
            request.override(system_message=self._compose_system_message(request))
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(
            request.override(system_message=self._compose_system_message(request))
        )


def build_expert_async_subagent_specs(cfg: Any | None = None) -> list[dict[str, Any]]:
    """Build ``AsyncSubAgent``-shaped specs for every installed expert skill.

    Every expert gets a background reach here, and
    ``build_expert_subagent_specs`` independently gives every expert an
    in-turn reach. Nothing classifies a skill into one or the other — the
    orchestrator chooses per task.

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
    from .expert_container import _reserved_subagent_names, expert_prompt_body

    port = int(getattr(cfg, "langgraph_dev_port", 6174))
    # Mirror the sync fold-in's name guard in ``_fold_expert_subagents``:
    # yaml async sub-agents (``writing-agent``, ``data-analysis-agent``,
    # ``scheduler``, …) and ``general-purpose`` share the async spec pool
    # by name, so an expert skill named after any of them would raise
    # ``ValueError: Duplicate async subagent names`` inside
    # ``AsyncSubAgentMiddleware.__init__`` and kill CLI startup. The local
    # ``seen`` set catches the second failure trigger: two workspace-tier
    # skills that share the same frontmatter ``name`` (workspace listing
    # uses ``check_seen=False``, so both survive to this point).
    taken = set(_reserved_subagent_names())
    specs: list[dict[str, Any]] = []
    for skill in list_expert_skills(include_system=True):
        # Same empty-body skip the sync fold-in enforces in
        # ``expert_container.py::build_expert_subagent_specs``. Advertising
        # a body-less expert in ``start_async_task``'s tool schema, then
        # rejecting it at loader time, wastes a launch round-trip; filter
        # upstream so ``start_async_task`` never sees the broken skill.
        if not expert_prompt_body(skill).strip():
            _logger.warning(
                "Expert skill %r: %s body is empty; skipping "
                "async-dispatch registration.",
                skill.name,
                "AGENTS.md" if skill.expert_source == "agents_md" else "SKILL.md",
            )
            continue
        if skill.name in taken:
            _logger.warning(
                "Expert skill %r collides with an existing async sub-agent "
                "name; skipping async-dispatch registration.",
                skill.name,
            )
            continue
        taken.add(skill.name)
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
    ``{messages, skill_name}`` as initial state; the
    :class:`ExpertSkillLoaderMiddleware` resolves ``skill_name`` on every
    model call and injects that expert's actor definition as system prompt.

    Tool set is intentionally minimal (``think_tool`` only) — matches the
    sync ``expert_container`` factory. Once the per-skill ``allowed-tools``
    follow-up ships, the tool list will union with the skill's declared
    tools.
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
    from ..tools import think_tool

    cfg = get_effective_config()
    apply_config_to_env(cfg)

    subagents: list[dict[str, Any]] = []
    # The async expert runs as its own graph inside langgraph-dev, so it
    # does NOT inherit the main agent's subagent stack. Without at least
    # one dispatchable target, the expert's ``task()`` tool has no target
    # to call — long-horizon experts (e.g. literature-review's Phase 2
    # ``paper-navigator`` fan-out) rely on this. ``general-purpose`` is
    # deepagents' default sub-agent and provides the fan-out capability
    # symmetric with what the main agent has by default.
    #
    # Trade-off: an expert LLM that never uses ``task()`` still pays the
    # schema-description tokens on every model call. Acceptable at v1; if
    # experts routinely abuse the fan-out or ignore it entirely, a
    # per-skill ``include_general_purpose: false`` frontmatter opt-out
    # is a natural follow-up.
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
        tools=[think_tool],
        skills=["/skills/"],
        backend=_get_default_backend(),
        middleware=middleware,
        subagents=subagents,
        state_schema=ExpertContainerState,
    ).with_config({"recursion_limit": cfg.recursion_limit})
