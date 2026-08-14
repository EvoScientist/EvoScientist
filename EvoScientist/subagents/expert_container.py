"""Expert-subagent-spec factory for the agent-teams feature.

Turns an installed **expert skill** (a `SkillInfo` with `type == "expert"`) into
a deepagents subagent spec dict compatible with `subagents=[...]` on
`create_deep_agent`. The main agent's `_build_base_kwargs` folds these specs
into its subagent list at construction time so the `task` tool can dispatch to
each installed expert in-turn; the same registry is reused by the QuickJS
`task()` global for in-eval fan-out.

A skill declares itself an expert by carrying a sibling `AGENTS.md`, whose body
is the actor definition and therefore the runtime prompt — see
`skills_manager._parse_skill_md`. THIS module builds the in-turn (`task()`) spec
for such experts and for legacy `type: expert` frontmatter experts alike;
`expert_container_async.py` independently builds the background
(`start_async_task`) spec for the same experts, so every expert is reachable
both ways and the orchestrator picks per task.

The generic-container principle from #361 lives in THIS FUNCTION — one
construction path for all experts, sourcing behaviour from the skill file
rather than a per-expert YAML.
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


def expert_prompt_body(skill_info: SkillInfo) -> str:
    """Return the text that becomes *skill_info*'s runtime prompt.

    The single answer to "which file carries this expert's persona", for
    every path that builds or validates an expert — sync spec factory, async
    spec factory, and the async loader middleware. The two contracts source
    it differently, and splitting that decision across call sites is how a
    skill ends up registered off one file and prompted off another:

    - ``expert_source == "agents_md"`` — the sibling AGENTS.md body (persona
      + result envelope). SKILL.md stays pure knowledge and stays readable
      in-turn off the ``/skills/`` mount, so it is deliberately NOT the
      prompt here.
    - legacy frontmatter experts — the SKILL.md body, which is where those
      skills put their persona before the actor/knowledge split.

    Prefers the text cached on ``SkillInfo`` by ``_parse_skill_md``. Falls
    back to reading from disk when the cached field is empty — that handles
    ``SkillInfo`` objects constructed by hand (external callers) without the
    body populated. Returns an empty string if the file can't be read; the
    callers treat that as "refuse to register this expert".
    """
    if skill_info.expert_source == "agents_md":
        if skill_info.agents_body:
            return skill_info.agents_body
        from ..tools.skills_manager import _read_agents_md

        return _read_agents_md(skill_info.path) or ""

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
    """Compose the expert's system_prompt from its role + prompt body.

    *body* is what :func:`expert_prompt_body` resolved — an AGENTS.md actor
    definition, or a legacy expert's SKILL.md body. Either way it carries
    the persona voice, rubrics, and output-style instructions, written in
    second person addressing the expert itself.

    The legacy `role` frontmatter (one-line role summary) is prepended as an
    orientation line when set. AGENTS.md experts declare no `role` — their
    persona section opens with the same orientation in prose — so for those
    the body passes through untouched.
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
    body = expert_prompt_body(skill_info)
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


def list_dispatchable_experts(
    *, include_system: bool = True, cfg: Any | None = None
) -> list[SkillInfo]:
    """Experts the orchestrator can actually reach.

    Every installed expert is reachable, so this applies only the two
    filters that would otherwise register a broken entry: an empty actor
    definition (nothing to prompt with) and a name collision with a yaml
    sub-agent or ``general-purpose`` (which would shadow the built-in).

    Deliberately does NOT filter on async availability. An expert whose
    background reach is unavailable is still reachable in-turn via
    ``task()``, so dropping it here would hide a working expert — the
    failure mode that made installed experts vanish whenever langgraph
    dev was unhealthy.

    ``cfg`` is accepted and ignored; kept so callers that thread config
    through don't have to special-case this one. Read-only filter —
    construction-time warnings for empty-body / colliding experts are
    emitted by ``build_expert_subagent_specs`` and
    ``_fold_expert_subagents`` respectively, so nothing is logged here.
    """
    from ..tools.skills_manager import list_expert_skills

    reserved = _reserved_subagent_names()
    dispatchable: list[SkillInfo] = []
    for info in list_expert_skills(include_system=include_system):
        if not expert_prompt_body(info).strip():
            continue
        if info.name in reserved:
            continue
        dispatchable.append(info)
    return dispatchable


def build_expert_subagent_specs(
    tool_registry: dict[str, Any] | None = None,
    *,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    """Build the in-turn (``task``) spec for every installed expert skill.

    Thin wrapper over ``list_expert_skills()`` + ``build_expert_subagent_spec``.
    Called by the main-agent construction path (``_build_base_kwargs``) to
    fold experts into the ``subagents=[...]`` list.

    Every expert gets a spec here, and ``build_expert_async_subagent_specs``
    independently gives every expert a background spec. The same name in
    both is intentional and safe: the two land on different tools
    (``task`` vs ``start_async_task``) with separate schemas, and
    deepagents' duplicate-name check is scoped to the async list alone.
    Two reaches, one expert — the orchestrator picks per task.

    Skips (with a warning) any expert whose actor definition is empty — a
    personaless expert advertised in the ``task`` tool schema would let the
    orchestrator dispatch to a blank system prompt, a worse failure mode
    than the expert being absent.
    """
    from ..tools.skills_manager import list_expert_skills

    specs: list[dict[str, Any]] = []
    for info in list_expert_skills(include_system=include_system):
        if not expert_prompt_body(info).strip():
            _logger.warning(
                "Expert skill %r: %s body is empty; skipping registration.",
                info.name,
                "AGENTS.md" if info.expert_source == "agents_md" else "SKILL.md",
            )
            continue
        specs.append(build_expert_subagent_spec(info, tool_registry=tool_registry))
    return specs
