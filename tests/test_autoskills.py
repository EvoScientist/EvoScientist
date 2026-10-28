from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from EvoScientist import paths
from EvoScientist.config import (
    EvoScientistConfig,
    MemorySkillSynthesisCadence,
    MemorySkillSynthesisMode,
    save_config,
    set_config_value,
)
from EvoScientist.memory.autoskills.candidates import autoskill_candidates
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    list_skill_proposals,
    pending_skill_proposal_count,
    reject_skill_proposal,
    submit_autoskill_proposal,
)
from EvoScientist.memory.autoskills.schedule import (
    AUTOSKILL_GRAPH_ID,
    AUTOSKILL_RUN_KIND,
    alist_autoskill_schedules,
    autoskill_cron,
    reconcile_autoskill_schedule,
)
from EvoScientist.memory.autoskills.tools import create_submit_autoskill_proposal_tool
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationRelation,
    link_observation_files,
    record_observation_file,
)
from EvoScientist.paths import Workspace


def _record(
    memory_dir,
    *,
    summary: str,
    observation: str,
    memory_type: MemoryType = MemoryType.PROCEDURAL,
):
    return record_observation_file(
        memory_dir=memory_dir,
        project_id="P-project",
        memory_type=memory_type,
        summary=summary,
        observation=observation,
        why_it_matters=f"Future agents can reuse this pattern: {summary}",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )


def _write_skill_folder(memory_dir, skill_name: str, description: str, body: str):
    proposal_dir = autoskill_proposals_dir(memory_dir) / skill_name
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return proposal_dir


def _write_installed_skill(root, skill_name: str, description: str, body: str):
    skill_dir = root / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_installed_skill_in_dir(
    root,
    directory_name: str,
    *,
    skill_name: str,
    description: str,
    body: str,
):
    skill_dir = root / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_autoskill_cron_uses_presets():
    assert autoskill_cron("nightly", "03:00") == "0 3 * * *"
    assert autoskill_cron("weekly", "04:30") == "30 4 * * 0"
    assert autoskill_cron("monthly", "22:05") == "5 22 1 * *"


def test_autoskill_candidates_use_linked_procedural_clusters(tmp_path):
    memory_dir = tmp_path / "memories"
    first = _record(
        memory_dir,
        summary="Use focused pytest before full suite.",
        observation="Run the focused pytest file before the full test suite.",
    )
    second = _record(
        memory_dir,
        summary="Use ruff on changed Python modules.",
        observation="Run ruff on changed modules before broad validation.",
    )
    third = _record(
        memory_dir,
        summary="Validation workflow benefits from narrow checks.",
        observation="Narrow validation catches regressions before expensive checks.",
        memory_type=MemoryType.SEMANTIC,
    )
    for source, target in ((first, second), (second, third)):
        link_observation_files(
            memory_dir=memory_dir,
            project_id="P-project",
            source_observation_id=source["observation_id"],
            target_observation_id=target["observation_id"],
            reason="These observations describe the same validation workflow.",
        )

    candidates = autoskill_candidates(
        memory_dir=memory_dir,
        project_id="P-project",
    )

    assert len(candidates) == 1
    assert set(candidates[0]["observation_ids"]) == {
        first["observation_id"],
        second["observation_id"],
        third["observation_id"],
    }
    assert candidates[0]["procedural_count"] == 2
    assert candidates[0]["semantic_count"] == 1
    assert candidates[0]["episodic_count"] == 0
    assert candidates[0]["existing_pending_proposal"] is False
    assert candidates[0]["already_processed"] is False


def test_autoskill_candidates_surface_contradiction_clusters(tmp_path):
    memory_dir = tmp_path / "memories"
    first = _record(
        memory_dir,
        summary="Use cached package metadata for offline installs.",
        observation="Cached package metadata works when the network is unavailable.",
    )
    second = _record(
        memory_dir,
        summary="Avoid cached metadata for editable dependency changes.",
        observation="Cached package metadata can hide editable dependency changes.",
    )
    third = _record(
        memory_dir,
        summary="Package validation should check cache freshness.",
        observation="Validation should distinguish offline cache use from stale cache risks.",
        memory_type=MemoryType.SEMANTIC,
    )
    link_observation_files(
        memory_dir=memory_dir,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        relation=ObservationRelation.CONTRADICTS,
        reason="Cached metadata helps offline installs but can hide editable changes.",
    )
    link_observation_files(
        memory_dir=memory_dir,
        project_id="P-project",
        source_observation_id=second["observation_id"],
        target_observation_id=third["observation_id"],
        reason="Both observations describe cache-aware package validation.",
    )

    candidates = autoskill_candidates(
        memory_dir=memory_dir,
        project_id="P-project",
    )

    assert len(candidates) == 1
    assert set(candidates[0]["observation_ids"]) == {
        first["observation_id"],
        second["observation_id"],
        third["observation_id"],
    }
    assert any(
        relation["relation"] == ObservationRelation.CONTRADICTS
        for relation in candidates[0]["relations"]
    )


def test_autoskill_candidate_hash_ignores_mutable_relations(tmp_path):
    memory_dir = tmp_path / "memories"
    first = _record(
        memory_dir,
        summary="Use focused pytest before full suite.",
        observation="Run the focused pytest file before the full test suite.",
    )
    second = _record(
        memory_dir,
        summary="Use ruff on changed Python modules.",
        observation="Run ruff on changed modules before broad validation.",
    )
    third = _record(
        memory_dir,
        summary="Validation workflow benefits from narrow checks.",
        observation="Narrow validation catches regressions before expensive checks.",
        memory_type=MemoryType.SEMANTIC,
    )
    for source, target in ((first, second), (second, third)):
        link_observation_files(
            memory_dir=memory_dir,
            project_id="P-project",
            source_observation_id=source["observation_id"],
            target_observation_id=target["observation_id"],
            reason="These observations describe the same validation workflow.",
        )

    before = autoskill_candidates(memory_dir=memory_dir, project_id="P-project")[0]
    link_observation_files(
        memory_dir=memory_dir,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=third["observation_id"],
        reason="A later linker pass found another relation in the same cluster.",
    )
    after = autoskill_candidates(memory_dir=memory_dir, project_id="P-project")[0]

    assert after["cluster_hash"] == before["cluster_hash"]
    assert after["observation_ids"] == before["observation_ids"]
    assert len(after["relations"]) > len(before["relations"])


def test_skill_proposal_lifecycle_promotes_to_workspace_skill(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"

    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nRun narrow checks before broad ones.",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Three observations describe the same staged validation practice.",
        skills_dir=skills_dir,
    )

    assert proposal["submitted"] is True
    assert pending_skill_proposal_count(memory_dir) == 1
    pending = list_skill_proposals(memory_dir, status="pending")
    assert pending[0].skill_name == "focused-validation"
    assert pending[0].proposal_id == "focused-validation"

    with patch(
        "EvoScientist.memory.autoskills.proposals._notify_approved_skill"
    ) as notify:
        approved = approve_skill_proposal(
            memory_dir,
            pending[0].proposal_id,
            skills_dir=skills_dir,
        )

    assert approved["approved"] is True
    notify.assert_called_once_with()
    skill_md = skills_dir / "focused-validation" / "SKILL.md"
    assert skill_md.exists()
    assert "name: focused-validation" in skill_md.read_text(encoding="utf-8")
    assert pending_skill_proposal_count(memory_dir) == 0
    assert list_skill_proposals(memory_dir)[0].status == "approved"


def test_update_skill_proposal_replaces_workspace_skill(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    _write_installed_skill(
        skills_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nOld workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks and caveats.",
        "# Focused validation\n\nUpdated workflow with caveats.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-update",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="New observations refine the existing validation skill.",
        operation="update",
        target_skill_name="focused-validation",
        skills_dir=skills_dir,
    )
    pending = list_skill_proposals(memory_dir, status="pending")
    approved = approve_skill_proposal(
        memory_dir, proposal["proposal_id"], skills_dir=skills_dir
    )

    skill_md = skills_dir / "focused-validation" / "SKILL.md"
    saved = skill_md.read_text(encoding="utf-8")
    assert proposal["submitted"] is True
    assert proposal["operation"] == "update"
    assert pending[0].operation == "update"
    assert pending[0].target_skill_name == "focused-validation"
    assert approved["approved"] is True
    assert approved["operation"] == "update"
    assert "Updated workflow with caveats." in saved
    assert "Old workflow." not in saved
    assert list_skill_proposals(memory_dir)[0].status == "approved"


def test_update_skill_proposal_preserves_existing_workspace_files(
    tmp_path,
    monkeypatch,
):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    skill_dir = _write_installed_skill(
        skills_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nOld workflow.",
    )
    script_path = skill_dir / "scripts" / "run.sh"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("#!/bin/sh\necho validate\n", encoding="utf-8")
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks and caveats.",
        "# Focused validation\n\nUpdated workflow with caveats.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-update",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="New observations refine the existing validation skill.",
        operation="update",
        target_skill_name="focused-validation",
        skills_dir=skills_dir,
    )
    approved = approve_skill_proposal(
        memory_dir, proposal["proposal_id"], skills_dir=skills_dir
    )

    assert approved["approved"] is True
    assert "Updated workflow with caveats." in (skill_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert script_path.read_text(encoding="utf-8") == "#!/bin/sh\necho validate\n"


def test_update_can_reopen_completed_autoskill_proposal(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    _write_skill_folder(
        memory_dir,
        "reopen-update",
        "Use when testing completed autoskill proposals.",
        "# Reopen update\n\nInitial workflow.",
    )
    first = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="reopen-update",
        cluster_hash="cluster-initial",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Initial proposal.",
        skills_dir=skills_dir,
    )
    approved = approve_skill_proposal(
        memory_dir, first["proposal_id"], skills_dir=skills_dir
    )
    _write_skill_folder(
        memory_dir,
        "reopen-update",
        "Use when testing completed autoskill proposal updates.",
        "# Reopen update\n\nUpdated workflow.",
    )

    reopened = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="reopen-update",
        cluster_hash="cluster-update",
        source_observation_ids=["O-4", "O-5", "O-6"],
        rationale="Later observations refine the existing skill.",
        operation="update",
        target_skill_name="reopen-update",
        skills_dir=skills_dir,
    )
    proposal = list_skill_proposals(memory_dir, status="pending")[0]

    assert approved["approved"] is True
    assert reopened["submitted"] is True
    assert reopened["created"] is False
    assert proposal.operation == "update"
    assert proposal.cluster_hash == "cluster-update"


def test_update_cannot_reopen_rejected_autoskill_proposal(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    _write_skill_folder(
        memory_dir,
        "reject-update",
        "Use when testing rejected autoskill proposal updates.",
        "# Reject update\n\nInitial workflow.",
    )
    first = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="reject-update",
        cluster_hash="cluster-initial",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Initial proposal.",
        skills_dir=skills_dir,
    )
    rejected = reject_skill_proposal(memory_dir, first["proposal_id"])
    _write_installed_skill(
        skills_dir,
        "reject-update",
        "Use when testing rejected autoskill proposal updates.",
        "# Reject update\n\nInstalled workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "reject-update",
        "Use when testing rejected autoskill proposal updates.",
        "# Reject update\n\nUpdated workflow.",
    )

    reopened = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="reject-update",
        cluster_hash="cluster-update",
        source_observation_ids=["O-4", "O-5", "O-6"],
        rationale="Later observations refine the existing skill.",
        operation="update",
        target_skill_name="reject-update",
        skills_dir=skills_dir,
    )

    assert rejected["rejected"] is True
    assert reopened["submitted"] is False
    assert reopened["status"] == "rejected"
    assert list_skill_proposals(memory_dir)[0].status == "rejected"


def test_update_replaces_workspace_skill_matched_by_frontmatter(
    tmp_path,
    monkeypatch,
):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    installed_dir = _write_installed_skill_in_dir(
        skills_dir,
        "legacy-directory",
        skill_name="frontmatter-match",
        description="Use when testing frontmatter skill matching.",
        body="# Frontmatter match\n\nOld workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "frontmatter-match",
        "Use when testing frontmatter skill update matching.",
        "# Frontmatter match\n\nUpdated workflow.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="frontmatter-match",
        cluster_hash="cluster-frontmatter",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Later observations refine the existing skill.",
        operation="update",
        target_skill_name="frontmatter-match",
        skills_dir=skills_dir,
    )
    approved = approve_skill_proposal(
        memory_dir, proposal["proposal_id"], skills_dir=skills_dir
    )

    assert approved["approved"] is True
    assert approved["path"] == str(installed_dir)
    assert "Updated workflow." in (installed_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert not (skills_dir / "frontmatter-match").exists()


def test_update_skill_proposal_requires_existing_skill(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    _write_skill_folder(
        memory_dir,
        "missing-target",
        "Use when testing missing autoskill update targets.",
        "# Missing target\n",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="missing-target",
        cluster_hash="cluster-missing",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This should not submit without an installed target.",
        operation="update",
        target_skill_name="missing-target",
        skills_dir=tmp_path / "skills",
    )

    assert proposal["submitted"] is False
    assert "No installed workspace/global skill" in proposal["error"]
    assert pending_skill_proposal_count(memory_dir) == 0


def test_update_global_skill_writes_back_to_global_scope(tmp_path, monkeypatch):
    """An update to a globally installed skill updates the global skill.

    The old behaviour installed a workspace copy that shadowed the global
    skill without being visible from other workspaces (#330).
    """
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    (global_skills / ".installed.yaml").write_text(
        "global-validation:\n  source: owner/repo@global-validation\n  commit: abc123\n",
        encoding="utf-8",
    )
    global_script = global_skills / "global-validation" / "scripts" / "run.sh"
    global_script.parent.mkdir(parents=True)
    global_script.write_text("#!/bin/sh\necho global\n", encoding="utf-8")
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes from an updated global skill.",
        "# Global validation\n\nGlobal skill update.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-global-update",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Observations refine a global skill for this workspace.",
        operation="update",
        target_skill_name="global-validation",
        skills_dir=workspace_skills,
    )
    assert proposal["submitted"] is True
    # The tier is decided once, when the proposal is submitted.
    assert proposal["scope"] == "global"
    pending = list_skill_proposals(memory_dir, status="pending")
    assert pending[0].upstream_source == "owner/repo@global-validation"
    assert pending[0].upstream_commit == "abc123"
    approved = approve_skill_proposal(
        memory_dir, proposal["proposal_id"], skills_dir=workspace_skills
    )

    assert approved["approved"] is True
    assert approved["operation"] == "update"
    assert approved["scope"] == "global"
    assert approved["upstream_source"] == "owner/repo@global-validation"
    stored = next(
        p
        for p in list_skill_proposals(memory_dir)
        if p.skill_name == "global-validation"
    )
    assert stored.effective_scope == "global"
    # The global skill is updated in place, not shadowed by a copy.
    assert "Global skill update." in (
        global_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    # Files the proposal does not mention stay put.
    assert (global_skills / "global-validation" / "scripts" / "run.sh").read_text(
        encoding="utf-8"
    ) == "#!/bin/sh\necho global\n"
    # No workspace copy is left behind.
    assert not (workspace_skills / "global-validation").exists()


def test_update_global_skill_does_not_overwrite_nonmatching_local_dir(
    tmp_path,
    monkeypatch,
):
    """A global update must not clobber a local skill of the same folder name."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill_in_dir(
        workspace_skills,
        "global-validation",
        skill_name="different-local-skill",
        description="Use when testing nonmatching local skill collisions.",
        body="# Different local skill\n\nKeep this local content.",
    )
    _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes from an updated global skill.",
        "# Global validation\n\nWorkspace shadow update.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-global-update",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Observations refine a global skill for this workspace.",
        operation="update",
        target_skill_name="global-validation",
        skills_dir=workspace_skills,
    )
    assert proposal["submitted"] is False
    assert "No installed workspace/global skill" in proposal["error"]
    # The workspace directory hides the same virtual path, so the runtime does
    # not load the global target in this workspace.
    assert "Keep this local content." in (
        workspace_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Global workflow." in (
        global_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_global_approval_reports_workspace_shadow(tmp_path, monkeypatch):
    """Global approval proceeds but reports a workspace copy still in use."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes from an updated global skill.",
        "# Global validation\n\nUpdated global workflow.",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-shadow-after-submit",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The global update should be audited when a shadow appears.",
        operation="update",
        target_skill_name="global-validation",
        skills_dir=workspace_skills,
    )
    _write_installed_skill_in_dir(
        workspace_skills,
        "global-validation",
        skill_name="workspace-only",
        description="A workspace shadow with a different knowledge name.",
        body="# Workspace shadow\n\nStill active here.",
    )

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope="global",
    )

    assert result["approved"] is True
    assert result["scope"] == "global"
    assert result["workspace_shadow"] == str(workspace_skills / "global-validation")
    assert "Updated global workflow." in (
        global_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_workspace_approval_forks_global_skill(tmp_path, monkeypatch):
    """A workspace approval creates a local shadow of a global skill."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    global_skill = _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    bundled = global_skill / "scripts" / "check.py"
    bundled.parent.mkdir()
    bundled.write_text("print('check')\n", encoding="utf-8")
    (global_skills / ".installed.yaml").write_text(
        "global-validation:\n  source: owner/repo@global-validation\n",
        encoding="utf-8",
    )
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes from a workspace update.",
        "# Global validation\n\nWorkspace workflow.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-workspace-fork",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This update is intentionally workspace-specific.",
        operation="update",
        target_skill_name="global-validation",
        skills_dir=workspace_skills,
    )
    approved = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope="workspace",
    )

    assert proposal["submitted"] is True
    assert proposal["scope"] == "global"
    assert approved["approved"] is True
    assert approved["scope"] == "workspace"
    assert approved["upstream_source"] is None
    assert "Workspace workflow." in (
        workspace_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Global workflow." in (global_skill / "SKILL.md").read_text(encoding="utf-8")
    assert (workspace_skills / "global-validation" / "scripts" / "check.py").is_file()


def test_create_approval_can_target_global_scope(tmp_path, monkeypatch):
    """A global approval may install a new skill into the global tier."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes with staged checks.",
        "# Global validation\n\nShared workflow.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-global-create",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This skill belongs in every workspace.",
        skills_dir=workspace_skills,
    )
    approved = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope="global",
    )

    assert proposal["submitted"] is True
    assert proposal["scope"] == "workspace"
    assert approved["approved"] is True
    assert approved["scope"] == "global"
    assert (global_skills / "global-validation" / "SKILL.md").exists()
    assert not (workspace_skills / "global-validation").exists()


def test_create_proposal_defaults_to_workspace_scope(tmp_path, monkeypatch):
    """A create proposal without a scope stays in the workspace."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nRun narrow checks before broad ones.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-default-scope",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This skill belongs to one workspace.",
        skills_dir=workspace_skills,
    )

    assert proposal["submitted"] is True
    assert proposal["scope"] == "workspace"


def test_create_approval_rejects_frontmatter_name_collision(tmp_path, monkeypatch):
    """A differently named directory must not hide a newly created skill."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill_in_dir(
        workspace_skills,
        "z-existing",
        skill_name="frontmatter-collision",
        description="Already installed under another directory name.",
        body="# Existing skill\n",
    )
    _write_installed_skill_in_dir(
        global_skills,
        "a-global-existing",
        skill_name="another-collision",
        description="A skill in the other tier.",
        body="# Other tier\n",
    )
    _write_skill_folder(
        memory_dir,
        "frontmatter-collision",
        "Use when testing create collisions.",
        "# Proposed skill\n",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="frontmatter-collision",
        cluster_hash="cluster-create-collision",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The create path must reject duplicate runtime names.",
        skills_dir=workspace_skills,
    )
    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert proposal["submitted"] is True
    assert result["approved"] is False
    assert "already exists" in result["error"]
    assert not (workspace_skills / "frontmatter-collision").exists()

    _write_skill_folder(
        memory_dir,
        "another-collision",
        "Use when testing cross-tier create collisions.",
        "# Proposed other-tier skill\n",
    )
    second = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="another-collision",
        cluster_hash="cluster-create-cross-tier-collision",
        source_observation_ids=["O-4", "O-5", "O-6"],
        rationale="A create must reject a collision in the other tier.",
        skills_dir=workspace_skills,
    )
    second_result = approve_skill_proposal(
        memory_dir,
        second["proposal_id"],
        skills_dir=workspace_skills,
    )
    assert second_result["approved"] is False
    assert "already exists" in second_result["error"]


def test_approval_rejects_unknown_scope(tmp_path):
    """An unknown approval scope is rejected and leaves the proposal pending."""
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nRun narrow checks before broad ones.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-bad-scope",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This approval names a scope that does not exist.",
        skills_dir=tmp_path / "skills",
    )
    approved = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=tmp_path / "skills",
        scope="everywhere",
    )

    assert proposal["submitted"] is True
    assert approved["approved"] is False
    assert "scope must be" in approved["error"]
    assert pending_skill_proposal_count(memory_dir) == 1


def test_update_approval_rejects_scope_without_a_target(tmp_path, monkeypatch):
    """A global approval of a workspace-only skill is an error."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill(
        workspace_skills,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nOld workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks and caveats.",
        "# Focused validation\n\nUpdated workflow with caveats.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-wrong-tier",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This proposal names a tier that does not hold the target.",
        operation="update",
        target_skill_name="focused-validation",
        skills_dir=workspace_skills,
    )
    approved = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope="global",
    )

    assert proposal["submitted"] is True
    assert approved["approved"] is False
    assert "No installed global skill" in approved["error"]


def test_update_shadowed_by_local_skill_leaves_global_untouched(tmp_path, monkeypatch):
    """A local skill that shadows a global one must not modify the global copy.

    The workspace tier shadows the global tier, so the update targets the
    local skill. The global copy stays exactly as it was.
    """
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    global_skill = _write_installed_skill(
        global_skills,
        "shared-validation",
        "Use when validating code changes from a global skill.",
        "# Shared validation\n\nGlobal workflow.",
    )
    original_global = (global_skill / "SKILL.md").read_bytes()
    _write_installed_skill(
        workspace_skills,
        "shared-validation",
        "Use when validating code changes from a local skill.",
        "# Shared validation\n\nLocal workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "shared-validation",
        "Use when validating code changes from an updated skill.",
        "# Shared validation\n\nUpdated workflow.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-validation",
        cluster_hash="cluster-shadowed-update",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Observations refine the skill this workspace uses.",
        operation="update",
        target_skill_name="shared-validation",
        skills_dir=workspace_skills,
    )
    approved = approve_skill_proposal(
        memory_dir, proposal["proposal_id"], skills_dir=workspace_skills
    )

    assert proposal["submitted"] is True
    # The local copy shadows the global one, so it is the target.
    assert proposal["scope"] == "workspace"
    assert approved["approved"] is True
    assert "Updated workflow." in (
        workspace_skills / "shared-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    # The global copy is untouched, byte for byte.
    assert (global_skill / "SKILL.md").read_bytes() == original_global


@pytest.mark.parametrize("scope", [None, "workspace"])
def test_legacy_proposal_without_scope_keeps_workspace_behaviour(
    tmp_path,
    monkeypatch,
    scope,
):
    """A proposal recorded before scopes existed must not write to global.

    It keeps the old workspace behaviour, and only an explicit scope
    confirms installing into the global tier.
    """
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes from an updated global skill.",
        "# Global validation\n\nUpdated workflow.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-legacy",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Observations refine a global skill.",
        operation="update",
        target_skill_name="global-validation",
        skills_dir=workspace_skills,
    )
    # Drop the recorded scope to stand in for a proposal written before
    # scopes existed.
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "global-validation" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["scope"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    approved = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope=scope,
    )

    assert approved["approved"] is True
    assert "Updated workflow." in (
        workspace_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    # The global skill is left alone.
    assert "Global workflow." in (
        global_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_legacy_proposal_needs_explicit_scope_for_global(tmp_path, monkeypatch):
    """Confirming the global scope is what lets a legacy proposal write there."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    _write_skill_folder(
        memory_dir,
        "global-validation",
        "Use when validating code changes from an updated global skill.",
        "# Global validation\n\nUpdated workflow.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="global-validation",
        cluster_hash="cluster-legacy-global",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Observations refine a global skill.",
        operation="update",
        target_skill_name="global-validation",
        skills_dir=workspace_skills,
    )
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "global-validation" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["scope"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    approved = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope="global",
    )

    assert approved["approved"] is True
    assert "Updated workflow." in (
        global_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert not (workspace_skills / "global-validation").exists()


def test_approve_command_rejects_an_unknown_scope(tmp_path, monkeypatch):
    """A scope word that is not a tier is an error, not a missing word.

    Falling back to the scope the proposal recorded would install the skill
    somewhere the user did not ask for.
    """
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand

    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    skills_dir = tmp_path / "skills"
    _write_skill_folder(
        memory_dir,
        "typo-scope",
        "Use when testing the scope argument.",
        "# Revised skill\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="typo-scope",
        cluster_hash="cluster-typo-scope",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The scope argument must be checked.",
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is True

    messages: list[str] = []
    ctx = SimpleNamespace(
        workspace=SimpleNamespace(root=tmp_path, skills_dir=skills_dir),
        ui=SimpleNamespace(
            append_system=lambda *args, **kwargs: messages.append(str(args[0])),
            mount_renderable=lambda *args, **kwargs: None,
        ),
    )
    asyncio.run(
        AutoSkillsCommand().execute(
            ctx,
            ["approve", proposal["proposal_id"], "workspce"],
        )
    )
    assert any("scope" in message for message in messages), messages
    # Nothing was installed, and the proposal is still pending.
    assert not (global_skills / "typo-scope").exists()
    assert not (skills_dir / "typo-scope").exists()
    stored = next(
        p for p in list_skill_proposals(memory_dir) if p.skill_name == "typo-scope"
    )
    assert stored.status == "pending"


def test_approval_records_the_scope_it_installed_into(tmp_path, monkeypatch):
    """An approval that overrides the scope records where the skill went."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _write_skill_folder(
        memory_dir,
        "override-scope",
        "Use when testing the scope override.",
        "# Revised skill\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="override-scope",
        cluster_hash="cluster-override-scope",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The scope override must be recorded.",
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is True

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
        scope="global",
    )
    assert result["approved"] is True
    assert result["scope"] == "global"
    stored = next(
        p for p in list_skill_proposals(memory_dir) if p.skill_name == "override-scope"
    )
    assert stored.submitted_scope == "workspace"
    assert stored.effective_scope == "global"
    assert Path(stored.approved_skill_path).parent == global_skills


def test_approve_command_accepts_global_scope(tmp_path, monkeypatch):
    """The slash command forwards a valid global scope override."""
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand

    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    skills_dir = tmp_path / "skills"
    _write_skill_folder(
        memory_dir,
        "command-global",
        "Use when testing global approval from the command.",
        "# Command global\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="command-global",
        cluster_hash="cluster-command-global",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The command should accept a valid scope.",
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    messages: list[str] = []
    ctx = SimpleNamespace(
        workspace=SimpleNamespace(root=tmp_path, skills_dir=skills_dir),
        ui=SimpleNamespace(
            append_system=lambda *args, **kwargs: messages.append(str(args[0])),
            mount_renderable=lambda *args, **kwargs: None,
        ),
    )

    asyncio.run(
        AutoSkillsCommand().execute(
            ctx,
            ["approve", proposal["proposal_id"], "global"],
        )
    )

    assert any("Approved autoskill" in message for message in messages), messages
    assert (global_skills / "command-global" / "SKILL.md").exists()


def test_update_targets_the_tier_the_loader_resolves(tmp_path, monkeypatch):
    """A skill the loader accepts is updated in the tier the loader uses.

    The loader accepts a ``---`` fence with trailing whitespace. A proposal
    lookup that does not would pick the global copy and overwrite it while
    the workspace copy stays in charge.
    """
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    local = _write_installed_skill(
        skills_dir,
        "loader-mismatch",
        "Use when testing the workspace skill.",
        "# Workspace skill\n",
    )
    skill_file = local / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8").replace("---\n", "--- \n", 1),
        encoding="utf-8",
    )
    global_file = (
        _write_installed_skill(
            global_skills,
            "loader-mismatch",
            "Use when testing the global skill.",
            "# Global skill\n",
        )
        / "SKILL.md"
    )
    original_global = global_file.read_bytes()
    _write_skill_folder(
        memory_dir,
        "loader-mismatch",
        "Use when testing the workspace skill.",
        "# Revised skill\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="loader-mismatch",
        cluster_hash="cluster-loader-mismatch",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The update must target the loaded copy.",
        operation="update",
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is True
    assert proposal["scope"] == "workspace"

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    assert result["approved"] is True
    assert global_file.read_bytes() == original_global
    assert "Revised skill" in skill_file.read_text(encoding="utf-8")


def test_update_targets_the_knowledge_skill_behind_an_expert_alias(
    tmp_path,
    monkeypatch,
):
    """A skill that also ships an EXPERT.md is still a knowledge skill.

    The runtime pins a knowledge skill by the name in its SKILL.md
    frontmatter; an expert that ships an EXPERT.md is registered as an actor
    under its directory name. The two identities are not interchangeable: the
    update has to touch the copy the middleware loads, and leave the global
    copy of the same name alone.
    """
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    aliased = _write_installed_skill(
        workspace_skills,
        "z-expert-audit",
        "Use when auditing changes.",
        "# Audit\n",
    )
    skill_file = aliased / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8").replace(
            "name: z-expert-audit",
            "name: shared-audit",
        ),
        encoding="utf-8",
    )
    # The expert declaration makes the runtime register this directory as an
    # actor under its directory name.
    (aliased / "EXPERT.md").write_text("# Auditor\n", encoding="utf-8")
    global_file = (
        _write_installed_skill(
            global_skills,
            "shared-audit",
            "Use when auditing changes.",
            "# Global audit\n",
        )
        / "SKILL.md"
    )
    original_global = global_file.read_bytes()
    _write_skill_folder(
        memory_dir,
        "shared-audit",
        "Use when auditing changes.",
        "# Revised audit\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-audit",
        cluster_hash="cluster-expert-alias",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The update must reach the copy the runtime loads.",
        operation="update",
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is True
    # The workspace copy is the one the runtime loads, so it is the target.
    assert proposal["scope"] == "workspace"

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )
    assert result["approved"] is True
    assert global_file.read_bytes() == original_global
    assert "Revised audit" in skill_file.read_text(encoding="utf-8")


def test_update_uses_the_runtime_winner_for_duplicate_skill_names(
    tmp_path,
    monkeypatch,
):
    """Approval follows the later duplicate name selected by the loader."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    first = _write_installed_skill_in_dir(
        workspace_skills,
        "a-audit",
        skill_name="shared-audit",
        description="Use when auditing changes.",
        body="# First audit\n\nKeep first.",
    )
    winner = _write_installed_skill_in_dir(
        workspace_skills,
        "z-audit",
        skill_name="shared-audit",
        description="Use when auditing changes.",
        body="# Later audit\n\nKeep later.",
    )
    _write_skill_folder(
        memory_dir,
        "shared-audit",
        "Use when auditing changes.",
        "# Revised audit\n",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-audit",
        cluster_hash="cluster-duplicate-name",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The update must target the copy the runtime loads.",
        operation="update",
        target_skill_name="shared-audit",
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )
    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )

    assert result["approved"] is True
    assert Path(result["path"]) == winner
    assert "Revised audit" in (winner / "SKILL.md").read_text(encoding="utf-8")
    assert "Keep first." in (first / "SKILL.md").read_text(encoding="utf-8")


def test_update_uses_virtual_path_order_for_duplicate_skill_names(
    tmp_path,
    monkeypatch,
):
    """The resolver follows slash-delimited virtual path ordering."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    folders = {
        "a": "First.",
        "a-b": "Middle.",
        "a.x": "Last by bare-name order.",
    }
    created = {
        name: _write_installed_skill_in_dir(
            workspace_skills,
            name,
            skill_name="virtual-duplicate",
            description="Use when testing virtual path ordering.",
            body=f"# {name}\n\n{body}",
        )
        for name, body in folders.items()
    }
    _write_skill_folder(
        memory_dir,
        "virtual-duplicate",
        "Use when testing virtual path ordering.",
        "# Revised virtual duplicate\n",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="virtual-duplicate",
        cluster_hash="cluster-virtual-order",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The update must follow the runtime's virtual path winner.",
        operation="update",
        target_skill_name="virtual-duplicate",
        skills_dir=workspace_skills,
    )
    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert result["approved"] is True
    assert Path(result["path"]) == created["a"]
    assert "Revised virtual duplicate" in (created["a"] / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_update_uses_runtime_winner_across_workspace_and_global_tiers(
    tmp_path,
    monkeypatch,
):
    """Duplicate frontmatter names resolve in the same order as the loader."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    workspace = _write_installed_skill_in_dir(
        workspace_skills,
        "a-audit",
        skill_name="shared-audit",
        description="Workspace copy.",
        body="# Workspace audit\n\nKeep workspace.",
    )
    global_copy = _write_installed_skill_in_dir(
        global_skills,
        "z-audit",
        skill_name="shared-audit",
        description="Global copy.",
        body="# Global audit\n\nKeep global.",
    )
    _write_skill_folder(
        memory_dir,
        "shared-audit",
        "Use when auditing changes.",
        "# Revised audit\n",
    )
    from EvoScientist.tools.skills_manager import installed_skill_index

    assert installed_skill_index(Workspace(tmp_path))["shared-audit"] == "Global copy."

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-audit",
        cluster_hash="cluster-cross-tier-duplicate",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The update must reach the runtime-selected copy.",
        operation="update",
        target_skill_name="shared-audit",
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )
    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )

    assert proposal["submitted"] is True
    assert proposal["scope"] == "global"
    assert result["approved"] is True
    assert Path(result["path"]) == global_copy
    assert "Revised audit" in (global_copy / "SKILL.md").read_text(encoding="utf-8")
    assert "Keep workspace." in (workspace / "SKILL.md").read_text(encoding="utf-8")


def test_update_ignores_a_skill_the_runtime_cannot_load(tmp_path, monkeypatch):
    """A file the runtime rejects is not an installed skill.

    The runtime only accepts a frontmatter block whose closing fence is
    followed by a newline. Such a file is never loaded, so it must not be
    picked as the update target - the copy the runtime does load is.
    """
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    unloadable = workspace_skills / "shared-audit"
    unloadable.mkdir(parents=True)
    # The closing fence ends the file: the runtime's frontmatter contract
    # wants a newline after it, so this skill is never loaded.
    (unloadable / "SKILL.md").write_text(
        "---\nname: shared-audit\ndescription: Use when auditing changes.\n---",
        encoding="utf-8",
    )
    global_file = (
        _write_installed_skill(
            global_skills,
            "shared-audit",
            "Use when auditing changes.",
            "# Global audit\n",
        )
        / "SKILL.md"
    )
    original_global = global_file.read_bytes()
    _write_skill_folder(
        memory_dir,
        "shared-audit",
        "Use when auditing changes.",
        "# Revised audit\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-audit",
        cluster_hash="cluster-unloadable",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Only a loadable skill is an update target.",
        operation="update",
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is False
    assert "No installed workspace/global skill" in proposal["error"]
    assert global_file.read_bytes() == original_global


def test_update_is_refused_without_any_loadable_skill(tmp_path, monkeypatch):
    """A skill the runtime cannot load is not an update target at all."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    unloadable = workspace_skills / "shared-audit"
    unloadable.mkdir(parents=True)
    (unloadable / "SKILL.md").write_text(
        "---\nname: shared-audit\ndescription: Use when auditing changes.\n---",
        encoding="utf-8",
    )
    _write_skill_folder(
        memory_dir,
        "shared-audit",
        "Use when auditing changes.",
        "# Revised audit\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-audit",
        cluster_hash="cluster-unloadable-only",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="There is no installed skill to update.",
        operation="update",
        skills_dir=workspace_skills,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is False


def test_approval_refuses_an_unknown_recorded_scope(tmp_path, monkeypatch):
    """A manifest recording an unknown scope is broken, not legacy."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _write_skill_folder(
        memory_dir,
        "unknown-scope",
        "Use when testing a broken manifest.",
        "# Revised skill\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="unknown-scope",
        cluster_hash="cluster-unknown-scope",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="A broken scope must not be ignored.",
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    assert proposal["submitted"] is True
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / proposal["proposal_id"] / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scope"] = "gloabl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )
    assert result["approved"] is False
    assert not (global_skills / "unknown-scope").exists()
    assert not (skills_dir / "unknown-scope").exists()


def test_auto_approval_keeps_new_skill_in_workspace(
    tmp_path,
    monkeypatch,
):
    """AUTO mode cannot choose a tier, so new skills stay in workspace."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace = Workspace(tmp_path)
    _write_skill_folder(
        memory_dir,
        "auto-new",
        "Use when testing the auto mode.",
        "# Auto skill\n",
    )
    tool = create_submit_autoskill_proposal_tool(
        memory_dir=memory_dir,
        workspace=workspace,
        project_id="P-1",
    )
    with patch("EvoScientist.memory.autoskills.tools.get_effective_config") as config:
        config.return_value = SimpleNamespace(
            memory_skill_synthesis_mode=MemorySkillSynthesisMode.AUTO
        )
        result = json.loads(
            tool.invoke(
                {
                    "skill_name": "auto-new",
                    "cluster_hash": "cluster-auto-new",
                    "source_observation_ids": ["O-1", "O-2", "O-3"],
                    "rationale": "Auto mode installs without review.",
                }
            )
        )
    assert "scope" not in tool.args
    assert result["auto_approval"]["approved"] is True
    assert result["auto_approval"]["scope"] == "workspace"
    assert (workspace.skills_dir / "auto-new" / "SKILL.md").is_file()
    assert not (global_skills / "auto-new").exists()


def test_auto_mode_keeps_global_update_pending_for_human_approval(
    tmp_path,
    monkeypatch,
):
    """AUTO mode never writes an update to the shared global tier."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace = Workspace(tmp_path)
    installed = _write_installed_skill(
        global_skills,
        "auto-global-update",
        "Use when testing guarded global updates.",
        "# Global skill\n\nOriginal workflow.\n",
    )
    _write_skill_folder(
        memory_dir,
        "auto-global-update",
        "Use when testing guarded global updates.",
        "# Global skill\n\nUpdated workflow.\n",
    )
    tool = create_submit_autoskill_proposal_tool(
        memory_dir=memory_dir,
        workspace=workspace,
        project_id="P-1",
    )
    with patch("EvoScientist.memory.autoskills.tools.get_effective_config") as config:
        config.return_value = SimpleNamespace(
            memory_skill_synthesis_mode=MemorySkillSynthesisMode.AUTO
        )
        result = json.loads(
            tool.invoke(
                {
                    "skill_name": "auto-global-update",
                    "cluster_hash": "cluster-auto-global-update",
                    "source_observation_ids": ["O-1", "O-2", "O-3"],
                    "rationale": "Auto mode must not silently change every workspace.",
                    "operation": "update",
                    "target_skill_name": "auto-global-update",
                }
            )
        )

    assert result["scope"] == "global"
    assert result["auto_approval"]["approved"] is False
    assert "human approval" in result["auto_approval"]["error"]
    assert pending_skill_proposal_count(memory_dir) == 1
    assert "Original workflow." in (installed / "SKILL.md").read_text(encoding="utf-8")


def test_create_proposal_rejects_target_skill_name(tmp_path):
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "target-on-create",
        "Use when testing create proposal target validation.",
        "# Target on create\n",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="target-on-create",
        cluster_hash="cluster-target",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Create proposals should not name an update target.",
        target_skill_name="target-on-create",
        skills_dir=tmp_path / "skills",
    )

    assert proposal["submitted"] is False
    assert "only valid for update" in proposal["error"]
    assert pending_skill_proposal_count(memory_dir) == 0


def test_submit_autoskill_proposal_defaults_missing_created_at(tmp_path):
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "timestamp-default",
        "Use when testing autoskill proposal timestamp defaults.",
        "# Timestamp default\n",
    )
    first = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="timestamp-default",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Initial proposal.",
        skills_dir=tmp_path / "skills",
    )
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "timestamp-default" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("created_at")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    second = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="timestamp-default",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Resubmitted proposal.",
        skills_dir=tmp_path / "skills",
    )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposal = list_skill_proposals(memory_dir)[0]

    assert first["submitted"] is True
    assert second["submitted"] is True
    assert saved["created_at"] != "None"
    assert saved["created_at"].endswith("Z")
    assert proposal.created_at == saved["created_at"]


def test_approve_skill_proposal_is_scoped_to_recorded_workspace(tmp_path):
    memory_dir = tmp_path / "memories"
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    _write_skill_folder(
        memory_dir,
        "workspace-owned",
        "Use when validating workspace ownership for autoskills.",
        "# Workspace owned\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="workspace-owned",
        cluster_hash="cluster-workspace-a",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="This proposal belongs to workspace A.",
        skills_dir=workspace_a / "skills",
        workspace_dir=workspace_a,
        project_id="P-a",
    )

    assert proposal["submitted"] is True
    assert list_skill_proposals(memory_dir, workspace_dir=workspace_b) == []
    wrong_workspace = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_b / "skills",
        workspace_dir=workspace_b,
    )
    right_workspace = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_a / "skills",
        workspace_dir=workspace_a,
    )

    assert wrong_workspace["approved"] is False
    assert not (workspace_b / "skills" / "workspace-owned").exists()
    assert right_workspace["approved"] is True
    assert (workspace_a / "skills" / "workspace-owned" / "SKILL.md").exists()


def test_submit_autoskill_proposal_does_not_overwrite_other_workspace(tmp_path):
    memory_dir = tmp_path / "memories"
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    _write_skill_folder(
        memory_dir,
        "shared-name",
        "Use when testing cross-workspace proposal name collisions.",
        "# Shared name\n",
    )

    first = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-name",
        cluster_hash="cluster-a",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Workspace A owns this pending proposal.",
        skills_dir=workspace_a / "skills",
        workspace_dir=workspace_a,
        project_id="P-a",
    )
    second = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-name",
        cluster_hash="cluster-b",
        source_observation_ids=["O-4", "O-5", "O-6"],
        rationale="Workspace B must not take over the same proposal id.",
        skills_dir=workspace_b / "skills",
        workspace_dir=workspace_b,
        project_id="P-b",
    )

    assert first["submitted"] is True
    assert second["submitted"] is False
    assert "another workspace" in second["error"]
    assert list_skill_proposals(memory_dir, workspace_dir=workspace_b) == []


def test_submit_tool_reads_live_autoskill_mode_without_rebuild(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "EvoScientist.config.settings.find_dotenv",
        lambda *a, **k: str(tmp_path / ".env"),
    )
    monkeypatch.delenv("EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_MODE", raising=False)
    save_config(
        EvoScientistConfig(
            memory_skill_synthesis_mode=MemorySkillSynthesisMode.REVIEW,
        )
    )
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace = Workspace(workspace_dir)
    tool = create_submit_autoskill_proposal_tool(
        memory_dir=memory_dir,
        workspace=workspace,
        project_id="P-project",
    )

    _write_skill_folder(
        memory_dir,
        "review-mode-skill",
        "Use when testing review mode autoskill submissions.",
        "# Review mode skill\n",
    )
    review_payload = json.loads(
        tool.run(
            {
                "skill_name": "review-mode-skill",
                "cluster_hash": "cluster-review",
                "source_observation_ids": ["O-1", "O-2", "O-3"],
                "rationale": "Review mode should only stage this proposal.",
            }
        )
    )
    set_config_value("memory_skill_synthesis_mode", "auto")
    _write_skill_folder(
        memory_dir,
        "auto-mode-skill",
        "Use when testing auto mode autoskill submissions.",
        "# Auto mode skill\n",
    )
    auto_payload = json.loads(
        tool.run(
            {
                "skill_name": "auto-mode-skill",
                "cluster_hash": "cluster-auto",
                "source_observation_ids": ["O-4", "O-5", "O-6"],
                "rationale": "Auto mode should promote this proposal.",
            }
        )
    )

    assert review_payload["status"] == "pending"
    assert "auto_approval" not in review_payload
    assert auto_payload["auto_approval"]["approved"] is True
    assert (workspace.skills_dir / "auto-mode-skill" / "SKILL.md").exists()


def test_submit_autoskill_proposal_rejects_invalid_generated_folder(tmp_path):
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nTODO: fill this in.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Three observations describe the same staged validation practice.",
        skills_dir=tmp_path / "skills",
    )

    assert proposal["submitted"] is False
    assert proposal["path"] == "/autoskill-proposals/focused-validation"
    assert "TODO placeholders" in proposal["errors"][0]
    assert pending_skill_proposal_count(memory_dir) == 0


def test_reject_skill_proposal_marks_processed(tmp_path):
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "reject-me",
        "Use when testing rejected proposals.",
        "# Reject me\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="reject-me",
        cluster_hash="cluster-rejected",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Test rejection.",
        skills_dir=tmp_path / "skills",
    )

    with patch(
        "EvoScientist.memory.autoskills.proposals._notify_approved_skill"
    ) as notify:
        rejected = reject_skill_proposal(memory_dir, proposal["proposal_id"])

    assert rejected["rejected"] is True
    notify.assert_not_called()
    assert list_skill_proposals(memory_dir)[0].status == "rejected"
    assert (memory_dir / "autoskills" / "processed" / "cluster-rejected.json").exists()


class _FakeCrons:
    def __init__(self):
        self.rows: list[dict] = []
        self.created: list[dict] = []
        self.deleted: list[str] = []

    def search(self, **_kwargs):
        return list(self.rows)

    def create(self, **kwargs):
        row = {
            "cron_id": f"cron-{len(self.rows) + 1}",
            "assistant_id": kwargs["assistant_id"],
            "schedule": kwargs["schedule"],
            "input": kwargs["input"],
            "metadata": kwargs["metadata"],
            "timezone": kwargs["timezone"],
            "enabled": True,
            # The server keeps the run config in the cron's payload.
            "payload": {"config": kwargs["config"]},
        }
        self.rows.append(row)
        self.created.append(row)
        return row

    def delete(self, cron_id: str):
        self.deleted.append(cron_id)
        self.rows = [row for row in self.rows if row["cron_id"] != cron_id]


class _AsyncFakeCrons:
    """Two pages: a full page of another workspace's crons, then ours."""

    def __init__(self, workspace: Workspace, other: str) -> None:
        self.rows = [
            {"cron_id": f"theirs-{i}", "metadata": {"workspace_dir": other}}
            for i in range(1000)
        ] + [{"cron_id": "cron-async", "metadata": {"workspace_dir": workspace.key}}]
        self.calls: list[dict] = []

    async def search(self, *, metadata, limit, offset):
        self.calls.append({"metadata": metadata, "limit": limit, "offset": offset})
        return self.rows[offset : offset + limit]


async def test_alist_autoskill_schedules_uses_async_client(
    monkeypatch, workspace, tmp_path
):
    other = Workspace(tmp_path / "other")
    other.root.mkdir(parents=True)
    crons = _AsyncFakeCrons(workspace, other.key)
    client = SimpleNamespace(crons=crons)
    monkeypatch.setattr("langgraph_sdk.get_client", lambda **_kwargs: client)

    rows = await alist_autoskill_schedules(
        EvoScientistConfig(),
        workspace_dir=workspace.root,
    )

    assert [row["cron_id"] for row in rows] == ["cron-async"]
    assert [c["offset"] for c in crons.calls] == [0, 1000]
    assert all(c["metadata"] == {"run_kind": AUTOSKILL_RUN_KIND} for c in crons.calls)


def test_reconcile_autoskill_schedule_creates_updates_and_disables(
    tmp_path,
    monkeypatch,
):
    crons = _FakeCrons()
    client = SimpleNamespace(crons=crons)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: client)

    cfg = EvoScientistConfig(
        memory_skill_synthesis_enabled=True,
        memory_skill_synthesis_cadence=MemorySkillSynthesisCadence.WEEKLY,
        memory_skill_synthesis_time="03:00",
        scheduler_default_timezone="UTC",
    )

    created = reconcile_autoskill_schedule(cfg, workspace_dir=tmp_path)
    unchanged = reconcile_autoskill_schedule(cfg, workspace_dir=tmp_path)
    updated = reconcile_autoskill_schedule(
        EvoScientistConfig(
            memory_skill_synthesis_enabled=True,
            memory_skill_synthesis_cadence=MemorySkillSynthesisCadence.NIGHTLY,
            memory_skill_synthesis_time="03:00",
            scheduler_default_timezone="UTC",
        ),
        workspace_dir=tmp_path,
    )
    disabled = reconcile_autoskill_schedule(
        EvoScientistConfig(memory_skill_synthesis_enabled=False),
        workspace_dir=tmp_path,
    )

    assert created["status"] == "created"
    assert unchanged["status"] == "unchanged"
    assert updated["status"] == "created"
    assert disabled == {"status": "disabled", "deleted": 1}
    assert crons.deleted == ["cron-1", "cron-1"]
    assert crons.rows == []
    assert created["schedule"] == "0 3 * * 0"
    assert updated["schedule"] == "0 3 * * *"
    assert created["cron_id"] == "cron-1"
    assert [row["assistant_id"] for row in crons.created] == [
        AUTOSKILL_GRAPH_ID,
        AUTOSKILL_GRAPH_ID,
    ]
