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


def build_expert_subagent_specs(
    tool_registry: dict[str, Any] | None = None,
    *,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    """Build spec dicts for every installed expert skill.

    Thin wrapper over ``list_expert_skills()`` + ``build_expert_subagent_spec``.
    Called by the main-agent construction path (``_build_base_kwargs``) to
    fold experts into the ``subagents=[...]`` list.

    Skips (with a warning) any expert whose SKILL.md body is empty — a
    personaless expert advertised in the ``task`` tool schema would let the
    orchestrator dispatch to a blank system prompt, a worse failure mode
    than the expert being absent.
    """
    from ..tools.skills_manager import list_expert_skills

    specs: list[dict[str, Any]] = []
    for info in list_expert_skills(include_system=include_system):
        if not _body_of(info).strip():
            _logger.warning(
                "Expert skill %r: SKILL.md body is empty; skipping registration.",
                info.name,
            )
            continue
        specs.append(build_expert_subagent_spec(info, tool_registry=tool_registry))
    return specs
