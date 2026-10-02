"""Agent-facing tools for the AutoSkills graph."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ...config import MemorySkillSynthesisMode, get_effective_config
from ...tools.skills_manager import list_skills
from .candidates import autoskill_candidates
from .proposals import approve_skill_proposal, submit_autoskill_proposal


class SubmitAutoskillProposalArgs(BaseModel):
    """Model-facing arguments for submitting an autoskill proposal folder."""

    skill_name: str = Field(
        min_length=1,
        description=(
            "Exact lowercase kebab-case skill directory name already created "
            "under /autoskill-proposals/."
        ),
    )
    cluster_hash: str = Field(
        min_length=1,
        description="Exact candidate cluster_hash returned by inspect_autoskill_candidates.",
    )
    source_observation_ids: list[str] = Field(
        min_length=1,
        description="Observation IDs that justify the skill.",
    )
    rationale: str = Field(
        min_length=1,
        description=(
            "Concise explanation of the repeated pattern and why it belongs in "
            "a reusable skill."
        ),
    )
    operation: str = Field(
        default="create",
        description=(
            "Use 'create' for a new skill or 'update' when the proposal is a "
            "change to an existing workspace/global skill."
        ),
    )
    target_skill_name: str | None = Field(
        default=None,
        description=(
            "For operation='update', the existing skill being updated. It must "
            "match skill_name."
        ),
    )


def _installed_skills_for_autoskill_context() -> list[dict[str, str]]:
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "source": skill.source,
            "path": f"/skills/{skill.path.name}",
        }
        for skill in list_skills(include_system=False)
        if skill.source in {"workspace", "global"}
    ]


def create_inspect_autoskill_candidates_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
    workspace_dir: str | Path,
) -> BaseTool:
    """Build the read-only candidate-inspection tool for AutoSkills."""

    def _inspect_autoskill_candidates() -> str:
        candidates = autoskill_candidates(
            memory_dir=memory_dir,
            project_id=project_id,
            workspace_dir=workspace_dir,
        )
        return json.dumps(
            {
                "candidates": candidates,
                "installed_skills": _installed_skills_for_autoskill_context(),
            },
            ensure_ascii=False,
            default=str,
        )

    return StructuredTool.from_function(
        func=_inspect_autoskill_candidates,
        name="inspect_autoskill_candidates",
        description=(
            "Inspect linked observation-memory clusters that may justify a "
            "new reusable skill or an update to an existing skill. Call this "
            "before proposing any skill."
        ),
    )


def create_submit_autoskill_proposal_tool(
    *,
    memory_dir: str | Path,
    workspace_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the proposal-registration tool for AutoSkills."""

    def _submit_autoskill_proposal(
        skill_name: str,
        cluster_hash: str,
        source_observation_ids: list[str],
        rationale: str,
        operation: str = "create",
        target_skill_name: str | None = None,
    ) -> str:
        proposal = submit_autoskill_proposal(
            memory_dir=memory_dir,
            skill_name=skill_name,
            cluster_hash=cluster_hash,
            source_observation_ids=source_observation_ids,
            rationale=rationale,
            operation=operation,
            target_skill_name=target_skill_name,
            workspace_dir=workspace_dir,
            project_id=project_id,
        )
        if (
            proposal.get("status") == "pending"
            and get_effective_config().memory_skill_synthesis_mode
            == MemorySkillSynthesisMode.AUTO
        ):
            approved = approve_skill_proposal(
                memory_dir,
                str(proposal["proposal_id"]),
                workspace_dir=workspace_dir,
            )
            proposal["auto_approval"] = approved
        return json.dumps(proposal, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        func=_submit_autoskill_proposal,
        name="submit_autoskill_proposal",
        description=(
            "Validate and register an autoskill proposal after creating its "
            "folder under /autoskill-proposals/<skill-name>. Set operation to "
            "'update' when changing an existing workspace/global skill. In "
            "auto mode, the tool promotes the proposal only if validation and "
            "collision checks pass."
        ),
        args_schema=SubmitAutoskillProposalArgs,
        infer_schema=False,
    )
