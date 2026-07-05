"""Slow background agent for proposing AutoSkills from EvoMemory."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from ...backends import build_autoskill_agent_backend
from ...config import get_effective_config
from ..autoskills.proposals import autoskill_proposals_dir
from ..autoskills.tools import (
    create_inspect_autoskill_candidates_tool,
    create_submit_autoskill_proposal_tool,
)
from ..project import resolve_project_id
from ._factory import (
    build_memory_agent_graph,
    memory_agent_middleware,
    resolve_memory_agent_paths,
)

_AUTOSKILLS_EXCLUDED_TOOLS = frozenset({"task", "write_todos"})


def _autoskills_system_prompt() -> str:
    return (
        "You synthesize reusable skills from EvoMemory observation clusters.\n\n"
        "This is slow, conservative background maintenance. Always call "
        "`inspect_autoskill_candidates` first. Consider only candidates that "
        "are not already processed and do not already have a pending proposal. "
        "Propose a skill only when the cluster shows a repeated, procedural "
        "pattern that would materially improve future agent work.\n\n"
        "The inspection result also lists installed workspace/global skills "
        "eligible for updates. If a candidate clearly improves, corrects, or "
        "adds caveats to an existing skill, propose an update instead of a new "
        "skill. Do not update built-in/system skills; only update skills "
        "returned in `installed_skills`. For an update, read the existing "
        "`/skills/<skill>/SKILL.md` first and preserve useful existing "
        "references or scripts unless the observations justify removing them.\n\n"
        "Candidate relations are context, not automatic approval or rejection "
        "rules. Use `complements` to understand supporting observations, "
        "`contradicts` to capture caveats or conditions where a practice fails, "
        "and `supersedes` to prefer newer guidance over older guidance. If the "
        "relations reveal that no coherent reusable procedure exists, do not "
        "propose a skill.\n\n"
        "Use the installed `skill-creator` skill for skill design guidance. "
        "Read its SKILL.md before drafting a proposal. Choose a concise, "
        "lowercase kebab-case skill name; this name is the proposal id. For "
        "updates, use the exact existing skill name.\n\n"
        "Create the proposal as an actual skill folder under "
        "`/autoskill-proposals/<skill-name>/` using `write_file` and "
        "`edit_file`. The folder must contain `SKILL.md` with valid YAML "
        "frontmatter whose `name` matches `<skill-name>` and whose "
        "`description` states when future agents should use it. Add bundled "
        "references or scripts only when they remove real complexity. Keep the "
        "skill concise and operational. Update proposals overlay the existing "
        "workspace/global skill: proposal files replace files with the same "
        "relative path, and omitted installed files are preserved.\n\n"
        "Use `execute` for lightweight validation when useful. Shell commands "
        "run from the autoskill proposal root; keep generated files and logs "
        "under `/autoskill-proposals/`. Do not shell into `/skills` or "
        "`/memories`; read those through file tools instead.\n\n"
        "Do not create a skill for one-off project facts, ordinary summaries, "
        "raw logs, weakly related observations, or clusters dominated by "
        "semantic facts without a reusable procedure. Do not manually edit "
        "`/skills` or `/memories`.\n\n"
        "When the folder is ready, call `submit_autoskill_proposal` with the "
        "exact skill_name, cluster_hash, source observation IDs, rationale, "
        'and `operation`. Use `operation="create"` for a new skill and '
        '`operation="update"` plus `target_skill_name=<skill-name>` for an '
        "existing skill update. If it reports validation errors, edit the "
        "proposal folder and submit again."
    )


def _autoskills_tools(
    *,
    memory_dir: str | Path,
    workspace_dir: str | Path,
) -> list[BaseTool]:
    project_id = resolve_project_id(workspace_dir)
    return [
        create_inspect_autoskill_candidates_tool(
            memory_dir=memory_dir,
            project_id=project_id,
            workspace_dir=workspace_dir,
        ),
        create_submit_autoskill_proposal_tool(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            project_id=project_id,
        ),
    ]


def build_autoskills_graph(
    *,
    memory_dir: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> CompiledStateGraph:
    """Build the registered LangGraph AutoSkills agent."""
    cfg = get_effective_config()
    agent_paths = resolve_memory_agent_paths(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
    )
    proposals_dir = autoskill_proposals_dir(agent_paths.memory_dir)
    return build_memory_agent_graph(
        name="evomemory-autoskills",
        system_prompt=_autoskills_system_prompt(),
        tools=_autoskills_tools(
            memory_dir=agent_paths.memory_dir,
            workspace_dir=agent_paths.workspace_dir,
        ),
        memory_dir=agent_paths.memory_dir,
        workspace_dir=agent_paths.workspace_dir,
        middleware=memory_agent_middleware(
            excluded_tools=_AUTOSKILLS_EXCLUDED_TOOLS,
        ),
        skills=["/skills/"],
        backend=build_autoskill_agent_backend(
            memory_dir=agent_paths.memory_dir,
            proposals_dir=proposals_dir,
            sandbox_timeout=cfg.sandbox_execute_timeout,
        ),
    )
