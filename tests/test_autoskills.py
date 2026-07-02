from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

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
    AUTOSKILL_SCHEDULE_SEARCH_LIMIT,
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
    )

    assert proposal["submitted"] is True
    assert pending_skill_proposal_count(memory_dir) == 1
    pending = list_skill_proposals(memory_dir, status="pending")
    assert pending[0].skill_name == "focused-validation"
    assert pending[0].proposal_id == "focused-validation"

    approved = approve_skill_proposal(
        memory_dir,
        pending[0].proposal_id,
        skills_dir=skills_dir,
    )

    assert approved["approved"] is True
    skill_md = skills_dir / "focused-validation" / "SKILL.md"
    assert skill_md.exists()
    assert "name: focused-validation" in skill_md.read_text(encoding="utf-8")
    assert pending_skill_proposal_count(memory_dir) == 0
    assert list_skill_proposals(memory_dir)[0].status == "approved"


def test_update_skill_proposal_replaces_workspace_skill(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", skills_dir)
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
    )
    pending = list_skill_proposals(memory_dir, status="pending")
    approved = approve_skill_proposal(memory_dir, proposal["proposal_id"])

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
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", skills_dir)
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
    )
    approved = approve_skill_proposal(memory_dir, proposal["proposal_id"])

    assert approved["approved"] is True
    assert "Updated workflow with caveats." in (skill_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert script_path.read_text(encoding="utf-8") == "#!/bin/sh\necho validate\n"


def test_update_can_reopen_completed_autoskill_proposal(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", skills_dir)
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
    )
    approved = approve_skill_proposal(memory_dir, first["proposal_id"])
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
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", skills_dir)
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
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", skills_dir)
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
    )
    approved = approve_skill_proposal(memory_dir, proposal["proposal_id"])

    assert approved["approved"] is True
    assert approved["path"] == str(installed_dir)
    assert "Updated workflow." in (installed_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert not (skills_dir / "frontmatter-match").exists()


def test_update_skill_proposal_requires_existing_skill(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", tmp_path / "skills")
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
    )

    assert proposal["submitted"] is False
    assert "No installed workspace/global skill" in proposal["error"]
    assert pending_skill_proposal_count(memory_dir) == 0


def test_update_global_skill_creates_workspace_shadow(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", workspace_skills)
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill(
        global_skills,
        "global-validation",
        "Use when validating code changes from a global skill.",
        "# Global validation\n\nGlobal workflow.",
    )
    global_script = global_skills / "global-validation" / "scripts" / "run.sh"
    global_script.parent.mkdir(parents=True)
    global_script.write_text("#!/bin/sh\necho global\n", encoding="utf-8")
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
    )
    approved = approve_skill_proposal(memory_dir, proposal["proposal_id"])

    assert approved["approved"] is True
    assert approved["operation"] == "update"
    assert (workspace_skills / "global-validation" / "SKILL.md").exists()
    assert "Workspace shadow update." in (
        workspace_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert (workspace_skills / "global-validation" / "scripts" / "run.sh").read_text(
        encoding="utf-8"
    ) == "#!/bin/sh\necho global\n"
    assert "Global workflow." in (
        global_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_update_global_skill_does_not_overwrite_nonmatching_local_dir(
    tmp_path,
    monkeypatch,
):
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", workspace_skills)
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
    )
    approved = approve_skill_proposal(memory_dir, proposal["proposal_id"])

    assert proposal["submitted"] is True
    assert approved["approved"] is False
    assert "does not match" in approved["error"]
    assert "Keep this local content." in (
        workspace_skills / "global-validation" / "SKILL.md"
    ).read_text(encoding="utf-8")


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
    )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposal = list_skill_proposals(memory_dir)[0]

    assert first["submitted"] is True
    assert second["submitted"] is True
    assert saved["created_at"] != "None"
    assert saved["created_at"].endswith("Z")
    assert proposal.created_at == saved["created_at"]


def test_approve_skill_proposal_is_scoped_to_recorded_workspace(
    tmp_path,
    monkeypatch,
):
    memory_dir = tmp_path / "memories"
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    active_skills_dir = tmp_path / "active-skills"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", active_skills_dir)
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
        workspace_dir=workspace_a,
        project_id="P-a",
    )

    assert proposal["submitted"] is True
    assert list_skill_proposals(memory_dir, workspace_dir=workspace_b) == []
    wrong_workspace = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        workspace_dir=workspace_b,
    )
    right_workspace = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        workspace_dir=workspace_a,
    )

    assert wrong_workspace["approved"] is False
    assert not (workspace_b / "skills" / "workspace-owned").exists()
    assert right_workspace["approved"] is True
    assert (active_skills_dir / "workspace-owned" / "SKILL.md").exists()


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
        workspace_dir=workspace_a,
        project_id="P-a",
    )
    second = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="shared-name",
        cluster_hash="cluster-b",
        source_observation_ids=["O-4", "O-5", "O-6"],
        rationale="Workspace B must not take over the same proposal id.",
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
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
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
    active_skills_dir = tmp_path / "active-skills"
    monkeypatch.setattr(paths, "USER_SKILLS_DIR", active_skills_dir)
    workspace_dir.mkdir()
    tool = create_submit_autoskill_proposal_tool(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
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
    assert (active_skills_dir / "auto-mode-skill" / "SKILL.md").exists()


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
    )

    rejected = reject_skill_proposal(memory_dir, proposal["proposal_id"])

    assert rejected["rejected"] is True
    assert list_skill_proposals(memory_dir)[0].status == "rejected"
    assert (memory_dir / "autoskills" / "processed" / "cluster-rejected.json").exists()


class _FakeCrons:
    def __init__(self):
        self.rows: list[dict] = []
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.searches: list[dict] = []

    def search(self, **kwargs):
        self.searches.append(kwargs)
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
        }
        self.rows.append(row)
        self.created.append(row)
        return row

    def delete(self, cron_id: str):
        self.deleted.append(cron_id)
        self.rows = [row for row in self.rows if row["cron_id"] != cron_id]


class _AsyncFakeCrons:
    def __init__(self):
        self.searches: list[dict] = []

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return [{"cron_id": "cron-async"}]


def test_alist_autoskill_schedules_uses_async_client_and_explicit_limit(monkeypatch):
    crons = _AsyncFakeCrons()
    client = SimpleNamespace(crons=crons)
    monkeypatch.setattr("langgraph_sdk.get_client", lambda **_kwargs: client)

    rows = asyncio.run(
        alist_autoskill_schedules(
            EvoScientistConfig(),
            limit=3,
        )
    )

    assert rows == [{"cron_id": "cron-async"}]
    assert crons.searches == [
        {
            "metadata": {"run_kind": AUTOSKILL_RUN_KIND},
            "limit": 3,
        }
    ]


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
    assert all(
        search["limit"] == AUTOSKILL_SCHEDULE_SEARCH_LIMIT for search in crons.searches
    )
    assert [row["assistant_id"] for row in crons.created] == [
        AUTOSKILL_GRAPH_ID,
        AUTOSKILL_GRAPH_ID,
    ]
