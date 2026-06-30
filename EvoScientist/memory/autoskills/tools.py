"""Agent-facing tools for the AutoSkills graph."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

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


def create_inspect_autoskill_candidates_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only candidate-inspection tool for AutoSkills."""

    def _inspect_autoskill_candidates() -> str:
        candidates = autoskill_candidates(
            memory_dir=memory_dir,
            project_id=project_id,
        )
        return json.dumps({"candidates": candidates}, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        func=_inspect_autoskill_candidates,
        name="inspect_autoskill_candidates",
        description=(
            "Inspect linked observation-memory clusters that may justify a "
            "new reusable skill. Call this before proposing any skill."
        ),
    )


def create_submit_autoskill_proposal_tool(
    *,
    memory_dir: str | Path,
    mode: str,
) -> BaseTool:
    """Build the proposal-registration tool for AutoSkills."""

    def _submit_autoskill_proposal(
        skill_name: str,
        cluster_hash: str,
        source_observation_ids: list[str],
        rationale: str,
    ) -> str:
        proposal = submit_autoskill_proposal(
            memory_dir=memory_dir,
            skill_name=skill_name,
            cluster_hash=cluster_hash,
            source_observation_ids=source_observation_ids,
            rationale=rationale,
        )
        if (
            proposal.get("status") == "pending"
            and getattr(mode, "value", mode) == "auto"
        ):
            approved = approve_skill_proposal(
                memory_dir,
                str(proposal["proposal_id"]),
            )
            proposal["auto_approval"] = approved
        return json.dumps(proposal, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        func=_submit_autoskill_proposal,
        name="submit_autoskill_proposal",
        description=(
            "Validate and register an autoskill proposal after creating its "
            "folder under /autoskill-proposals/<skill-name>. In auto mode, "
            "the tool promotes the proposal only if validation and collision checks pass."
        ),
        args_schema=SubmitAutoskillProposalArgs,
    )
