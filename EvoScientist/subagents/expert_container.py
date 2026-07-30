"""Expert-subagent-spec factory for the v1 agent-teams feature.

Turns an installed **expert skill** (a `SkillInfo` with `type == "expert"`) into
a deepagents subagent spec dict compatible with `subagents=[...]` on
`create_deep_agent`. The main agent's `_build_base_kwargs` folds these specs
into its subagent list at construction time so the `task` tool can dispatch to
each installed expert for sync consult; the same registry is reused by the
QuickJS `task()` global for panel mode.

The generic-container principle from #361 lives in THIS FUNCTION — one
construction path for all experts, sourcing behaviour from the skill file
rather than a per-expert YAML. There's no deployed graph per expert in v1;
that's async-thread territory (v2) and blocked on the deepagents
`AsyncSubAgent` config-passthrough gap.
"""

from __future__ import annotations

import logging
from typing import Any

from ..tools.skills_manager import SkillInfo, _split_frontmatter_and_body

_logger = logging.getLogger(__name__)

# Default toolset for expert subagents. Kept minimal — most experts are
# "reason about the incoming description and produce structured output";
# they can reach installed utility skills via the `/skills/` mount.
#
# `skill_manager` is included so experts can inspect what utility skills are
# available at runtime (e.g. `idea-brainstorm` checks for `paper-navigator`
# before starting its literature-review phase). Widening beyond these two
# defaults should be a deliberate decision (e.g. adding `execute` only when
# we know experts need to run scripts — deepagents' built-in file/execute
# tools are already available regardless of this list).
_DEFAULT_EXPERT_TOOLS: tuple[str, ...] = ("think_tool", "skill_manager")

# Default skills mount — expert subagents get the same read-only skills view
# as any other subagent (matches `research.yaml` / `writing.yaml` shape).
_DEFAULT_EXPERT_SKILLS: tuple[str, ...] = ("/skills/",)


def _body_of(skill_info: SkillInfo) -> str:
    """Return the SKILL.md body (post-frontmatter content).

    Prefers the body cached on ``SkillInfo`` by ``_parse_skill_md``. Falls
    back to reading SKILL.md fresh if the cached body is empty — that
    handles skills constructed by hand (external callers) without a body
    field populated. Returns an empty string if the file can't be read.
    """
    if skill_info.body:
        return skill_info.body
    skill_md = skill_info.path / "SKILL.md"
    try:
        content = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _logger.warning(
            "Expert skill %r: could not read SKILL.md at %s (%s)",
            skill_info.name,
            skill_md,
            exc,
        )
        return ""
    _, body = _split_frontmatter_and_body(content)
    return body


def _compose_system_prompt(skill_info: SkillInfo, body: str) -> str:
    """Compose the expert's system_prompt from its role + SKILL.md body.

    The `role` frontmatter (one-line role summary) is prepended as an
    orientation line; the body carries the persona voice, rubrics, and
    output-style instructions (all written in second person addressing the
    expert itself, per the expert-skill authoring convention).
    """
    if skill_info.role:
        return f"You are {skill_info.role}.\n\n{body}".rstrip() + "\n"
    return body if body.endswith("\n") else body + "\n"


def build_expert_subagent_spec(
    skill_info: SkillInfo,
    tool_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deepagents subagent spec dict from an expert skill.

    Args:
        skill_info: An expert skill (``type == "expert"``). The caller is
            responsible for filtering — passing a utility skill here builds a
            spec anyway (utility skills just don't have persona content in
            the body, so the result is nonsensical rather than broken).
        tool_registry: Same registry `load_subagents` uses to resolve tool
            names to callables (e.g. `{"think_tool": think_tool, ...}`).
            Unresolved tools are skipped with a warning, matching
            `_build_one` in `EvoScientist/utils.py`.

    Returns:
        A subagent spec dict with the same shape ``load_subagents`` produces:
        ``{name, description, system_prompt, tools, skills, _async}``. Ready
        to append to the main agent's `subagents=[...]` list.
    """
    tool_registry = tool_registry or {}
    body = _body_of(skill_info)
    system_prompt = _compose_system_prompt(skill_info, body)

    resolved_tools: list[Any] = []
    for tool_name in _DEFAULT_EXPERT_TOOLS:
        if tool_name in tool_registry:
            resolved_tools.append(tool_registry[tool_name])
        else:
            _logger.warning(
                "Expert skill %r: default tool %r not in registry, skipping",
                skill_info.name,
                tool_name,
            )

    return {
        "name": skill_info.name,
        "description": skill_info.description,
        "system_prompt": system_prompt,
        "tools": resolved_tools,
        "skills": list(_DEFAULT_EXPERT_SKILLS),
        # v1 is sync-consult + panel only; both use the in-process subagent
        # registry, not the async graph path. Async-thread mode = v2.
        "_async": False,
    }


_reserved_subagent_names_cache: frozenset[str] | None = None


def _reserved_subagent_names() -> frozenset[str]:
    """Names ``_fold_expert_subagents`` refuses for expert registration.

    Union of every static yaml sub-agent name (from ``subagents/*.yaml``)
    plus deepagents' ``general-purpose``. Mirrors the ``taken`` set built
    inline in ``EvoScientist.py::_fold_expert_subagents`` so callers that
    need to know "which names will be rejected at fold time" don't have
    to replay it.

    Cached on first call — yaml sub-agent files are static per process.
    """
    global _reserved_subagent_names_cache
    if _reserved_subagent_names_cache is not None:
        return _reserved_subagent_names_cache

    from pathlib import Path

    import yaml
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    from .. import subagents as _subagents_pkg

    names: set[str] = {GENERAL_PURPOSE_SUBAGENT["name"]}
    for pkg_dir in _subagents_pkg.__path__:
        for yml_path in Path(pkg_dir).glob("*.yaml"):
            if yml_path.name.startswith(("_", ".")):
                continue
            try:
                data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                names.update(str(k) for k in data)
    _reserved_subagent_names_cache = frozenset(names)
    return _reserved_subagent_names_cache


def is_async_dispatch_available(cfg: Any | None = None) -> bool:
    """Return True when async-declared experts can actually be dispatched.

    Both gates must hold: ``enable_async_subagents`` opt-in AND langgraph
    dev subprocess reachable. Same predicate
    ``build_expert_async_subagent_specs`` uses at spec-build time, factored
    out so ``list_dispatchable_experts`` (invite whitelist) and
    ``ActiveTeamMiddleware`` (system-prompt cue) can honour it without
    each re-deriving.
    """
    if cfg is None:
        from ..config import get_effective_config

        cfg = get_effective_config()
    if not getattr(cfg, "enable_async_subagents", False):
        return False
    from ..langgraph_dev.manager import is_async_subagents_available

    return is_async_subagents_available()


def list_dispatchable_experts(
    *, include_system: bool = True, cfg: Any | None = None
) -> list[SkillInfo]:
    """Experts eligible for the ``/expert`` invite whitelist.

    Covers **both** sync (``task()``) and async (``start_async_task``)
    dispatch shapes — an async-declared expert must remain invitable when
    async dispatch is registered, or the active-team cue that instructs
    ``start_async_task(...)`` never fires. Do NOT add a ``default_dispatch
    == "async"`` exclusion here; the sync-specific exclusion lives in
    ``build_expert_subagent_specs``, which is a different surface.

    Async-declared experts are dropped when async dispatch is NOT
    registered (``enable_async_subagents=false`` or langgraph dev
    unreachable). Advertising an expert that resolves to a
    ``start_async_task`` tool that either doesn't exist or doesn't list
    the expert is worse than an honest refusal — see reviewer thread on
    PR #391.

    Combines ``list_expert_skills`` with the two filters that
    construction-time paths already apply (empty body via
    ``build_expert_subagent_specs``; name collision with yaml sub-agents
    or ``general-purpose`` via ``_fold_expert_subagents``). Callers
    surfacing experts to the user (e.g. the ``/expert`` slash command)
    should use this instead of ``list_expert_skills`` directly, otherwise
    they can accept a name that will silently misroute or per-turn error
    at dispatch time.

    Read-only filter — construction-time warnings for empty-body /
    colliding experts are emitted by ``build_expert_subagent_specs`` and
    ``_fold_expert_subagents`` respectively, so nothing is logged here.
    """
    from ..tools.skills_manager import list_expert_skills

    reserved = _reserved_subagent_names()
    async_available = is_async_dispatch_available(cfg=cfg)
    dispatchable: list[SkillInfo] = []
    for info in list_expert_skills(include_system=include_system):
        if not _body_of(info).strip():
            continue
        if info.name in reserved:
            continue
        if info.default_dispatch == "async" and not async_available:
            continue
        dispatchable.append(info)
    return dispatchable


def build_expert_subagent_specs(
    tool_registry: dict[str, Any] | None = None,
    *,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    """Build spec dicts for every installed sync-dispatched expert skill.

    Thin wrapper over ``list_expert_skills()`` + ``build_expert_subagent_spec``.
    Called by the main-agent construction path (``_build_base_kwargs``) to
    fold experts into the ``subagents=[...]`` list.

    Skips (with a warning) any expert whose SKILL.md body is empty — a
    personaless expert advertised in the ``task`` tool schema would let the
    orchestrator dispatch to a blank system prompt, a worse failure mode
    than the expert being absent.

    Experts declared with ``default_dispatch: async`` are excluded — those
    are folded via ``build_expert_async_subagent_specs`` in
    ``expert_container_async.py`` and reached through
    ``EvoAsyncSubAgentMiddleware.start_async_task`` instead of the sync
    ``task`` tool. A skill that appears in both lists would produce two
    competing tool schemas for the same subagent name.
    """
    from ..tools.skills_manager import list_expert_skills

    specs: list[dict[str, Any]] = []
    for info in list_expert_skills(include_system=include_system):
        if info.default_dispatch == "async":
            continue
        if not _body_of(info).strip():
            _logger.warning(
                "Expert skill %r: SKILL.md body is empty; skipping registration.",
                info.name,
            )
            continue
        specs.append(build_expert_subagent_spec(info, tool_registry=tool_registry))
    return specs
