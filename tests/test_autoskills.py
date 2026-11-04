from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from EvoScientist import paths
from EvoScientist.config import (
    EvoScientistConfig,
    MemorySkillSynthesisMode,
    save_config,
    set_config_value,
)
from EvoScientist.langgraph_dev.assistants import workspace_assistant_id
from EvoScientist.memory.autoskills.candidates import autoskill_candidates
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    find_skill_proposal,
    list_skill_proposals,
    pending_skill_proposal_count,
    processed_cluster_hashes,
    proposal_install_target,
    reject_skill_proposal,
    revert_skill_proposal,
    submit_autoskill_proposal,
)
from EvoScientist.memory.autoskills.schedule import (
    AUTOSKILL_GRAPH_ID,
    AUTOSKILL_RUN_KIND,
    alist_autoskill_schedules,
    autoskill_cron,
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


def _submit_focused_validation(memory_dir, skills_dir):
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nRun narrow checks before broad ones.",
    )
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Three observations describe the same staged validation practice.",
        skills_dir=skills_dir,
        workspace_dir=Path(skills_dir).parent,
    )
    assert result["submitted"] is True
    return result


def _manifest(memory_dir, proposal_id):
    path = autoskill_proposals_dir(memory_dir) / proposal_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_approval_records_user_and_time(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        approved = approve_skill_proposal(
            memory_dir, "focused-validation", skills_dir=skills_dir
        )

    assert approved["approved"] is True
    manifest = _manifest(memory_dir, "focused-validation")
    assert manifest["approved_by"] == "user"
    assert manifest["approved_at"].endswith("Z")
    proposal = list_skill_proposals(memory_dir)[0]
    assert proposal.approved_by == "user"
    assert proposal.approved_at == manifest["approved_at"]


def test_auto_mode_approval_records_auto(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
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

    assert result["auto_approval"]["approved"] is True
    assert _manifest(memory_dir, "auto-new")["approved_by"] == "auto"
    assert list_skill_proposals(memory_dir)[0].approved_by == "auto"


def test_reject_records_time(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)

    rejected = reject_skill_proposal(memory_dir, "focused-validation")

    assert rejected["rejected"] is True
    manifest = _manifest(memory_dir, "focused-validation")
    assert manifest["rejected_at"].endswith("Z")
    assert "approved_by" not in manifest
    assert list_skill_proposals(memory_dir)[0].rejected_at == manifest["rejected_at"]


def test_manifest_without_decision_fields_reads_as_none(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    path = autoskill_proposals_dir(memory_dir) / "focused-validation" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "approved"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    proposal = list_skill_proposals(memory_dir)[0]

    assert proposal.status == "approved"
    assert proposal.approved_by is None
    assert proposal.approved_at is None
    assert proposal.rejected_at is None


def _submit_create(
    memory_dir, skills_dir, skill_name, *, cluster_hash=None, workspace_dir=None
):
    _write_skill_folder(
        memory_dir,
        skill_name,
        f"Use when testing the {skill_name} draft.",
        f"# {skill_name}\n\nSteps.",
    )
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=skill_name,
        cluster_hash=cluster_hash or f"cluster-{skill_name}",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale=f"Observations describe {skill_name}.",
        skills_dir=skills_dir,
        workspace_dir=workspace_dir or Path(skills_dir).parent,
    )
    assert result["submitted"] is True
    return result


def _submit_update(memory_dir, skills_dir, skill_name):
    _write_skill_folder(
        memory_dir,
        skill_name,
        f"Use when testing the updated {skill_name}.",
        f"# {skill_name}\n\nUpdated steps.",
    )
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=skill_name,
        cluster_hash=f"cluster-update-{skill_name}",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale=f"New observations refine {skill_name}.",
        operation="update",
        target_skill_name=skill_name,
        skills_dir=skills_dir,
    )
    assert result["submitted"] is True
    return result


def test_find_skill_proposal_matches_exact_id_only(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "paper-review")
    _submit_create(memory_dir, skills_dir, "paper-review-v2")

    assert find_skill_proposal(memory_dir, "paper-review").proposal_id == "paper-review"
    assert (
        find_skill_proposal(memory_dir, "paper-review-v2").proposal_id
        == "paper-review-v2"
    )
    assert find_skill_proposal(memory_dir, "paper-rev") is None


def test_find_skill_proposal_refuses_bad_ids_without_reading(tmp_path):
    memory_dir = tmp_path / "memories"
    bad_ids = ["", "../paper-review", "a/b", "Paper-Review", "paper-review\n", "-x"]
    with patch(
        "EvoScientist.memory.autoskills.proposals._proposal_root",
        side_effect=AssertionError("read the filesystem"),
    ):
        for bad in bad_ids:
            assert find_skill_proposal(memory_dir, bad) is None


def test_find_skill_proposal_needs_the_folder_to_match_the_id(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "paper-review")
    root = autoskill_proposals_dir(memory_dir)
    shutil.copytree(root / "paper-review", root / "paper-copy")

    assert find_skill_proposal(memory_dir, "paper-copy") is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_find_skill_proposal_refuses_folders_outside_the_root(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "paper-review")
    root = autoskill_proposals_dir(memory_dir)
    manifest = json.loads((root / "paper-review" / "manifest.json").read_text())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text(
        json.dumps({**manifest, "proposal_id": "paper-link"}), encoding="utf-8"
    )
    (root / "paper-link").symlink_to(outside, target_is_directory=True)
    linked_manifest = tmp_path / "linked-manifest.json"
    linked_manifest.write_text(
        json.dumps({**manifest, "proposal_id": "paper-manifest"}), encoding="utf-8"
    )
    (root / "paper-manifest").mkdir()
    (root / "paper-manifest" / "manifest.json").symlink_to(linked_manifest)

    assert find_skill_proposal(memory_dir, "paper-link") is None
    assert find_skill_proposal(memory_dir, "paper-manifest") is None
    assert find_skill_proposal(memory_dir, "paper-review") is not None


def test_approve_and_reject_prefer_exact_id_over_prefix(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "paper-review")
    _submit_create(memory_dir, skills_dir, "paper-review-v2")
    _submit_create(memory_dir, skills_dir, "data-check")
    _submit_create(memory_dir, skills_dir, "data-check-v2")

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        approved = approve_skill_proposal(
            memory_dir, "paper-review", skills_dir=skills_dir
        )
    rejected = reject_skill_proposal(memory_dir, "data-check")

    assert approved["approved"] is True
    assert approved["proposal_id"] == "paper-review"
    assert rejected["rejected"] is True
    assert rejected["proposal_id"] == "data-check"
    statuses = {p.proposal_id: p.status for p in list_skill_proposals(memory_dir)}
    assert statuses == {
        "data-check": "rejected",
        "data-check-v2": "pending",
        "paper-review": "approved",
        "paper-review-v2": "pending",
    }


def test_approve_ignores_a_copied_folder_that_claims_the_id(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "paper-review")
    root = autoskill_proposals_dir(memory_dir)
    shutil.copytree(root / "paper-review", root / "a-copy")
    copy_md = root / "a-copy" / "SKILL.md"
    copy_md.write_text(copy_md.read_text(encoding="utf-8") + "\nTODO\n", "utf-8")

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        approved = approve_skill_proposal(
            memory_dir, "paper-review", skills_dir=skills_dir
        )

    assert approved["approved"] is True
    installed = (skills_dir / "paper-review" / "SKILL.md").read_text(encoding="utf-8")
    assert "TODO" not in installed
    copy_manifest = json.loads((root / "a-copy" / "manifest.json").read_text())
    assert copy_manifest["status"] == "pending"
    assert _manifest(memory_dir, "paper-review")["status"] == "approved"


def test_unique_prefix_still_resolves(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "paper-review-v2")

    rejected = reject_skill_proposal(memory_dir, "paper-rev")

    assert rejected["rejected"] is True
    assert rejected["proposal_id"] == "paper-review-v2"


def test_install_target_for_create_is_new_folder_in_recorded_tier(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "focused-validation")

    target = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )

    assert target.error is None
    assert target.scope == "workspace"
    assert target.path == skills_dir / "focused-validation"
    assert target.base_path is None
    assert target.workspace_shadow is None
    assert not target.path.exists()
    assert list_skill_proposals(memory_dir)[0].status == "pending"


def test_install_target_follows_scope_override(tmp_path, monkeypatch):
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "focused-validation")

    target = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir, scope="global"
    )
    bad = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir, scope="gloabl"
    )

    assert target.scope == "global"
    assert target.path == global_skills / "focused-validation"
    assert bad.path is None
    assert bad.error == "scope must be 'workspace' or 'global'"


def test_install_target_for_update_is_the_installed_skill(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    installed = _write_installed_skill(
        skills_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nOld workflow.",
    )
    _submit_update(memory_dir, skills_dir, "focused-validation")

    target = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )

    assert target.error is None
    assert target.scope == "workspace"
    assert target.path == installed.resolve()
    assert target.base_path == installed.resolve()
    assert "Old workflow." in (installed / "SKILL.md").read_text(encoding="utf-8")


def test_install_target_for_project_fork_starts_from_global_skill(
    tmp_path, monkeypatch
):
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    global_skill = _write_installed_skill(
        global_skills,
        "paper-review",
        "Use when reviewing papers.",
        "# Paper review\n\nGlobal workflow.",
    )
    proposal = _submit_update(memory_dir, skills_dir, "paper-review")
    assert proposal["scope"] == "global"

    target = proposal_install_target(
        memory_dir, "paper-review", skills_dir=skills_dir, scope="workspace"
    )

    assert target.error is None
    assert target.scope == "workspace"
    assert target.path == skills_dir / "paper-review"
    assert target.base_path == global_skill.resolve()
    assert not target.path.exists()


def test_install_target_for_legacy_update_forks_global_skill(tmp_path, monkeypatch):
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    global_skill = _write_installed_skill(
        global_skills,
        "paper-review",
        "Use when reviewing papers.",
        "# Paper review\n\nGlobal workflow.",
    )
    _submit_update(memory_dir, skills_dir, "paper-review")
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "paper-review" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["scope"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target = proposal_install_target(memory_dir, "paper-review", skills_dir=skills_dir)

    assert target.error is None
    assert target.scope == "workspace"
    assert target.path == skills_dir / "paper-review"
    assert target.base_path == global_skill.resolve()


def test_install_target_reports_collision_without_writing(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "focused-validation")
    installed = _write_installed_skill(
        skills_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nHand-written.",
    )

    target = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )

    assert target.path is None
    assert target.base_path is None
    assert "already exists" in target.error
    assert "Hand-written." in (installed / "SKILL.md").read_text(encoding="utf-8")
    assert list_skill_proposals(memory_dir)[0].status == "pending"


def test_install_target_reports_invalid_draft_folder(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "focused-validation")
    skill_md = autoskill_proposals_dir(memory_dir) / "focused-validation" / "SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8") + "\nTODO\n", "utf-8")

    target = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )

    assert target.path is None
    assert target.error.startswith("Proposal skill folder is invalid")
    assert "TODO" in target.error


def test_install_target_for_unknown_or_decided_draft(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_create(memory_dir, skills_dir, "focused-validation")
    reject_skill_proposal(memory_dir, "focused-validation")

    missing = proposal_install_target(memory_dir, "no-such", skills_dir=skills_dir)
    decided = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )

    assert missing.path is None
    assert missing.error
    assert decided.path is None
    assert decided.error == "Only pending proposals can be approved"


def _approve_quietly(memory_dir, proposal_id, skills_dir, **kwargs):
    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = approve_skill_proposal(
            memory_dir, proposal_id, skills_dir=skills_dir, **kwargs
        )
    assert result["approved"] is True
    return result


def _approved_create_with_script(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    script = autoskill_proposals_dir(memory_dir) / "focused-validation" / "scripts"
    script.mkdir()
    (script / "check.py").write_text("print('check')\n", encoding="utf-8")
    _approve_quietly(memory_dir, "focused-validation", skills_dir)
    installed = skills_dir / "focused-validation"
    assert (installed / "scripts" / "check.py").is_file()
    return memory_dir, skills_dir, installed


def test_revert_rejected_returns_to_pending_and_unmarks_cluster(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    reject_skill_proposal(memory_dir, "focused-validation")
    assert "cluster-1" in processed_cluster_hashes(memory_dir)
    path = autoskill_proposals_dir(memory_dir) / "focused-validation" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["updated_at"] = "2000-01-01T00:00:00Z"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result == {
        "reverted": True,
        "proposal_id": "focused-validation",
        "status": "pending",
    }
    assert "cluster-1" not in processed_cluster_hashes(memory_dir)
    proposal = list_skill_proposals(memory_dir)[0]
    assert proposal.status == "pending"
    assert proposal.rejected_at is None
    assert proposal.updated_at != "2000-01-01T00:00:00Z"


def test_revert_without_a_processed_mark(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    reject_skill_proposal(memory_dir, "focused-validation")
    for mark in (memory_dir / "autoskills" / "processed").glob("*.json"):
        mark.unlink()

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True


def test_revert_approved_create_removes_installed_skill(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)

    with patch(
        "EvoScientist.memory.autoskills.proposals._notify_approved_skill"
    ) as notify:
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert not installed.exists()
    notify.assert_called_once_with()
    assert "cluster-1" not in processed_cluster_hashes(memory_dir)
    manifest = _manifest(memory_dir, "focused-validation")
    assert manifest["status"] == "pending"
    for key in ("approved_by", "approved_at", "approved_skill_path"):
        assert key not in manifest
    assert manifest["scope"] == "workspace"


@pytest.mark.parametrize("change", ["edit", "same_size_edit", "add", "add_folder"])
def test_revert_refuses_changed_skill(tmp_path, change):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    if change == "edit":
        skill_md = installed / "SKILL.md"
        skill_md.write_text(skill_md.read_text(encoding="utf-8") + "\nMore.\n")
    elif change == "same_size_edit":
        (installed / "scripts" / "check.py").write_text(
            "print('chock')\n", encoding="utf-8"
        )
    elif change == "add":
        (installed / "notes.md").write_text("Mine.\n", encoding="utf-8")
    else:
        (installed / "data").mkdir()
        (installed / "data" / "run.csv").write_text("a,b\n", encoding="utf-8")

    with patch(
        "EvoScientist.memory.autoskills.proposals._notify_approved_skill"
    ) as notify:
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result == {
        "reverted": False,
        "proposal_id": "focused-validation",
        "error": "skill_changed",
    }
    assert (installed / "SKILL.md").is_file()
    notify.assert_not_called()
    assert _manifest(memory_dir, "focused-validation")["status"] == "approved"
    assert "cluster-1" in processed_cluster_hashes(memory_dir)


def test_revert_deletes_nothing_when_a_later_file_changed(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    (installed / "scripts" / "check.py").write_text("print(1)\n", encoding="utf-8")

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert (installed / "SKILL.md").is_file()


def test_revert_rechecks_each_file_right_before_deleting_it(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)

    with patch(
        "EvoScientist.memory.autoskills.proposals._same_bytes",
        side_effect=[True, True, False],
    ):
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "scripts" / "check.py").is_file()


def test_revert_finishes_a_partly_removed_skill(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    (installed / "SKILL.md").unlink()

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert not installed.exists()


def test_revert_reports_a_failed_removal_and_a_retry_finishes(tmp_path, monkeypatch):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    real_unlink = Path.unlink

    def held_open(self, *args, **kwargs):
        if self.name == "check.py":
            raise PermissionError("held open")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", held_open)
    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        failed = revert_skill_proposal(memory_dir, "focused-validation")
    monkeypatch.setattr(Path, "unlink", real_unlink)

    assert failed["reverted"] is False
    assert failed["error"] == "revert_failed"
    assert "held open" in failed["detail"]
    assert _manifest(memory_dir, "focused-validation")["status"] == "approved"
    assert "cluster-1" in processed_cluster_hashes(memory_dir)

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        retried = revert_skill_proposal(memory_dir, "focused-validation")

    assert retried["reverted"] is True
    assert not installed.exists()
    assert "cluster-1" not in processed_cluster_hashes(memory_dir)


def test_revert_reports_a_failed_unmark_and_keeps_the_decision(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    reject_skill_proposal(memory_dir, "focused-validation")

    with patch(
        "EvoScientist.memory.autoskills.proposals.unmark_cluster_processed",
        side_effect=PermissionError("read-only"),
    ):
        failed = revert_skill_proposal(memory_dir, "focused-validation")

    assert failed["error"] == "revert_failed"
    assert _manifest(memory_dir, "focused-validation")["status"] == "rejected"
    assert "cluster-1" in processed_cluster_hashes(memory_dir)

    retried = revert_skill_proposal(memory_dir, "focused-validation")

    assert retried["reverted"] is True
    assert "cluster-1" not in processed_cluster_hashes(memory_dir)


@pytest.mark.parametrize("step", ["rmdir", "manifest"])
def test_revert_reports_a_failed_late_step_and_a_retry_finishes(
    tmp_path, monkeypatch, step
):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    if step == "rmdir":
        real_rmdir = Path.rmdir

        def failing(self):
            if self.name == "scripts":
                raise PermissionError("busy")
            return real_rmdir(self)

        monkeypatch.setattr(Path, "rmdir", failing)
        target = None
    else:
        target = patch(
            "EvoScientist.memory.autoskills.proposals._write_manifest",
            side_effect=PermissionError("busy"),
        )

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        if target is None:
            failed = revert_skill_proposal(memory_dir, "focused-validation")
        else:
            with target:
                failed = revert_skill_proposal(memory_dir, "focused-validation")
    monkeypatch.undo()

    assert failed["error"] == "revert_failed"
    assert _manifest(memory_dir, "focused-validation")["status"] == "approved"

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        retried = revert_skill_proposal(memory_dir, "focused-validation")

    assert retried["reverted"] is True
    assert not installed.exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, not root",
)
@pytest.mark.parametrize("unreadable", ["skill", "subfolder"])
def test_revert_reports_a_folder_it_cannot_list(tmp_path, unreadable):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    (installed / "scripts" / "mine.py").write_text("print('mine')\n", "utf-8")
    locked = installed if unreadable == "skill" else installed / "scripts"
    locked.chmod(0o300)
    try:
        result = revert_skill_proposal(memory_dir, "focused-validation")
    finally:
        locked.chmod(0o755)

    assert result["error"] == "revert_failed"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "scripts" / "mine.py").is_file()
    assert _manifest(memory_dir, "focused-validation")["status"] == "approved"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_revert_refuses_a_folder_symlink_inside_the_skill(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    outside = tmp_path / "outside-scripts"
    shutil.move(installed / "scripts", outside)
    (installed / "scripts").symlink_to(outside, target_is_directory=True)

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert (outside / "check.py").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_revert_never_deletes_through_a_folder_swapped_after_the_scan(
    tmp_path, monkeypatch
):
    from EvoScientist.memory.autoskills import proposals as proposals_module

    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    outside = tmp_path / "outside-scripts"
    real_scan = proposals_module._regular_files

    def scan_then_swap(root):
        files = real_scan(root)
        shutil.move(installed / "scripts", outside)
        (installed / "scripts").symlink_to(outside, target_is_directory=True)
        return files

    monkeypatch.setattr(proposals_module, "_regular_files", scan_then_swap)
    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is False
    assert (outside / "check.py").is_file()
    assert (installed / "SKILL.md").is_file()


def test_plain_entries_exclude_windows_reparse_points():
    from EvoScientist.memory.autoskills.proposals import _plain

    folder = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_reparse_tag=0)
    junction = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o755,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
        st_reparse_tag=0xA0000003,
    )
    # A OneDrive placeholder is a reparse point but no link.
    cloud_file = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o644,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
        st_reparse_tag=0x9000001A,
    )
    regular = SimpleNamespace(st_mode=stat.S_IFREG | 0o644)

    assert _plain(folder, directory=True) is True
    assert _plain(junction, directory=True) is False
    assert _plain(cloud_file, directory=False) is True
    assert _plain(regular, directory=False) is True
    assert _plain(regular, directory=True) is False


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_revert_refuses_a_symlink_inside_the_skill(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    target = tmp_path / "check.py"
    shutil.move(installed / "scripts" / "check.py", target)
    (installed / "scripts" / "check.py").symlink_to(target)

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert target.is_file()
    assert (installed / "SKILL.md").is_file()


def test_revert_refuses_a_recorded_path_outside_the_tier(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    backup = tmp_path / "backup" / "focused-validation"
    shutil.copytree(installed, backup)
    path = autoskill_proposals_dir(memory_dir) / "focused-validation" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["approved_skill_path"] = str(backup)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert (backup / "SKILL.md").is_file()
    assert (installed / "SKILL.md").is_file()


def _edit_manifest(memory_dir, proposal_id, **changes):
    path = autoskill_proposals_dir(memory_dir) / proposal_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_revert_refuses_a_twin_folder_in_the_same_tier(tmp_path):
    memory_dir, skills_dir, installed = _approved_create_with_script(tmp_path)
    twin = skills_dir / "focused-validation-copy"
    shutil.copytree(installed, twin)
    _edit_manifest(memory_dir, "focused-validation", approved_skill_path=str(twin))

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert (twin / "SKILL.md").is_file()


@pytest.mark.parametrize(
    "changes",
    [
        {"workspace_dir": None},
        {"approved_skill_path": ["not", "a", "path"]},
    ],
)
def test_revert_refuses_a_manifest_it_cannot_check(tmp_path, changes):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    _edit_manifest(memory_dir, "focused-validation", **changes)

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert (installed / "SKILL.md").is_file()


def test_revert_of_an_approval_that_recorded_no_scope(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    _edit_manifest(memory_dir, "focused-validation", scope=None, submitted_scope=None)

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert not installed.exists()


def test_revert_refuses_an_installed_path_that_is_a_file(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    shutil.rmtree(installed)
    installed.write_text("not a folder\n", encoding="utf-8")

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert installed.is_file()


def test_revert_of_an_approval_with_no_recorded_path(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    path = autoskill_proposals_dir(memory_dir) / "focused-validation" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["approved_skill_path"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert (installed / "SKILL.md").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_revert_refuses_installed_path_replaced_by_symlink(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    moved = tmp_path / "moved"
    shutil.move(installed, moved)
    installed.symlink_to(moved, target_is_directory=True)

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["error"] == "skill_changed"
    assert installed.is_symlink()
    assert (moved / "SKILL.md").is_file()


def test_revert_approved_create_whose_folder_is_gone(tmp_path):
    memory_dir, _skills_dir, installed = _approved_create_with_script(tmp_path)
    shutil.rmtree(installed)

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert list_skill_proposals(memory_dir)[0].status == "pending"


def test_revert_restores_the_submitted_scope(tmp_path, monkeypatch):
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    _approve_quietly(memory_dir, "focused-validation", skills_dir, scope="global")
    assert _manifest(memory_dir, "focused-validation")["scope"] == "global"

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert not (global_skills / "focused-validation").exists()
    proposal = list_skill_proposals(memory_dir)[0]
    assert proposal.scope == "workspace"
    assert proposal.submitted_scope == "workspace"


def test_revert_drops_scope_a_legacy_draft_never_had(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    path = autoskill_proposals_dir(memory_dir) / "focused-validation" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["scope"]
    del manifest["submitted_scope"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _approve_quietly(memory_dir, "focused-validation", skills_dir)
    assert _manifest(memory_dir, "focused-validation")["scope"] == "workspace"

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        revert_skill_proposal(memory_dir, "focused-validation")

    assert "scope" not in _manifest(memory_dir, "focused-validation")


def test_reverted_draft_can_be_decided_again(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    reject_skill_proposal(memory_dir, "focused-validation")
    revert_skill_proposal(memory_dir, "focused-validation")

    _approve_quietly(memory_dir, "focused-validation", skills_dir)

    manifest = _manifest(memory_dir, "focused-validation")
    assert manifest["status"] == "approved"
    assert manifest["approved_by"] == "user"
    assert "rejected_at" not in manifest
    assert (skills_dir / "focused-validation" / "SKILL.md").is_file()


def test_revert_refuses_update_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    installed = _write_installed_skill(
        skills_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nOld workflow.",
    )
    _submit_update(memory_dir, skills_dir, "focused-validation")
    _approve_quietly(memory_dir, "focused-validation", skills_dir)

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is False
    assert result["error"] == "revert_unavailable"
    assert (installed / "SKILL.md").is_file()
    assert _manifest(memory_dir, "focused-validation")["status"] == "approved"


def test_revert_refuses_pending_and_unknown(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)

    pending = revert_skill_proposal(memory_dir, "focused-validation")
    unknown = revert_skill_proposal(memory_dir, "focused")

    assert pending == {
        "reverted": False,
        "proposal_id": "focused-validation",
        "error": "not_decided",
    }
    assert unknown == {
        "reverted": False,
        "proposal_id": "focused",
        "error": "not_found",
    }


@pytest.mark.parametrize("other", ["approved", "rejected", "rejected_elsewhere"])
def test_revert_keeps_mark_shared_by_another_decided_draft(tmp_path, other):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    elsewhere = tmp_path / "other-project"
    _submit_create(
        memory_dir,
        skills_dir,
        "other-validation",
        cluster_hash="cluster-1",
        workspace_dir=elsewhere if other == "rejected_elsewhere" else None,
    )
    reject_skill_proposal(memory_dir, "focused-validation")
    if other == "approved":
        _approve_quietly(memory_dir, "other-validation", skills_dir)
    else:
        reject_skill_proposal(memory_dir, "other-validation")

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True
    assert "cluster-1" in processed_cluster_hashes(memory_dir)


@pytest.mark.parametrize(
    "skill_md",
    [
        b"---\nname: focused-validation\ndescription: caf\xe9\n---\n",
        b"---\nname: focused-validation\ndescription: d\n"
        b"metadata: {created: 2026-13-45}\n---\n",
        b"---\nname: focused-validation\ndescription: d\nmetadata: "
        + b"[" * 5000
        + b"]" * 5000
        + b"\n---\n",
    ],
    ids=["not-utf-8", "bad-date", "too-deep"],
)
def test_unreadable_skill_md_is_refused_not_raised(tmp_path, skill_md):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    path = autoskill_proposals_dir(memory_dir) / "focused-validation" / "SKILL.md"
    path.write_bytes(skill_md)

    approved = approve_skill_proposal(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )
    resubmitted = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Again.",
        skills_dir=skills_dir,
    )

    assert approved["approved"] is False
    assert approved["error"] == "Proposal skill folder is invalid"
    assert resubmitted["submitted"] is False
    assert not (skills_dir / "focused-validation").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_draft_linking_outside_its_folder_is_refused(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    folder = autoskill_proposals_dir(memory_dir) / "focused-validation"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not for the skill\n", encoding="utf-8")
    (folder / "notes.txt").symlink_to(outside / "secret.txt")

    approved = approve_skill_proposal(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )
    target = proposal_install_target(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )
    resubmitted = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Again.",
        skills_dir=skills_dir,
    )

    assert approved["approved"] is False
    assert any("outside its folder" in error for error in approved["errors"])
    assert "outside its folder" in target.error
    assert resubmitted["submitted"] is False
    assert not (skills_dir / "focused-validation").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_draft_folder_link_is_not_followed(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    folder = autoskill_proposals_dir(memory_dir) / "focused-validation"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not for the skill\n", encoding="utf-8")
    (folder / "data").symlink_to(outside, target_is_directory=True)

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        approved = approve_skill_proposal(
            memory_dir, "focused-validation", skills_dir=skills_dir
        )

    assert approved["approved"] is True
    assert not (skills_dir / "focused-validation" / "data").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_draft_linking_inside_its_folder_is_approved(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit_focused_validation(memory_dir, skills_dir)
    folder = autoskill_proposals_dir(memory_dir) / "focused-validation"
    (folder / "scripts").mkdir()
    (folder / "scripts" / "check.py").write_text("print(1)\n", encoding="utf-8")
    (folder / "check-link.py").symlink_to(folder / "scripts" / "check.py")

    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        approved = approve_skill_proposal(
            memory_dir, "focused-validation", skills_dir=skills_dir
        )

    assert approved["approved"] is True
    installed = skills_dir / "focused-validation" / "check-link.py"
    assert installed.read_text(encoding="utf-8") == "print(1)\n"


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


def test_global_create_rejects_workspace_collision_masked_by_folder_name(
    tmp_path,
    monkeypatch,
):
    """Create checks each tier independently of runtime masking order."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    _write_installed_skill_in_dir(
        workspace_skills,
        "foo",
        skill_name="bar",
        description="A workspace skill hidden by combined-tier lookup.",
        body="# Workspace bar\n\nKeep workspace.",
    )
    _write_installed_skill(
        global_skills,
        "foo",
        "A different global skill in the same virtual folder.",
        "# Global foo\n\nKeep global.",
    )
    _write_skill_folder(
        memory_dir,
        "bar",
        "Use when testing cross-tier create collisions.",
        "# Proposed bar\n",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="bar",
        cluster_hash="cluster-global-create-masked-collision",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="A collision in either tier must reject a create.",
        skills_dir=workspace_skills,
    )
    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
        scope="global",
    )

    assert result["approved"] is False
    assert "already exists" in result["error"]
    assert not (global_skills / "bar").exists()
    assert "Keep workspace." in (workspace_skills / "foo" / "SKILL.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_create_approval_rejects_broken_destination_symlink(tmp_path, monkeypatch):
    """A broken link is a collision, not a path to create through."""
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    workspace_skills.mkdir()
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    destination = workspace_skills / "broken-create"
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    _write_skill_folder(
        memory_dir,
        "broken-create",
        "Use when testing broken install destinations.",
        "# Broken create\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="broken-create",
        cluster_hash="cluster-broken-create",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="A broken link must not cause an install failure.",
        skills_dir=workspace_skills,
    )

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert result["approved"] is False
    assert "already exists" in result["error"]
    assert destination.is_symlink()


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


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
@pytest.mark.parametrize("link_kind", ["directory", "skill-md"])
def test_auto_mode_holds_global_update_with_external_workspace_symlinks(
    tmp_path,
    monkeypatch,
    link_kind,
):
    """Links the runtime cannot read must not relabel a global update."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace = Workspace(tmp_path)
    installed = _write_installed_skill(
        global_skills,
        "auto-global-link",
        "Use when testing guarded global updates.",
        "# Global skill\n\nOriginal workflow.\n",
    )
    workspace.skills_dir.mkdir(parents=True)
    local = workspace.skills_dir / "auto-global-link"
    if link_kind == "directory":
        local.symlink_to(installed, target_is_directory=True)
    else:
        local.mkdir()
        (local / "SKILL.md").symlink_to(installed / "SKILL.md")
    _write_skill_folder(
        memory_dir,
        "auto-global-link",
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
                    "skill_name": "auto-global-link",
                    "cluster_hash": f"cluster-auto-global-{link_kind}",
                    "source_observation_ids": ["O-1", "O-2", "O-3"],
                    "rationale": "An unreadable workspace alias must not bypass review.",
                    "operation": "update",
                    "target_skill_name": "auto-global-link",
                }
            )
        )

    assert result["scope"] == "global"
    assert result["auto_approval"]["approved"] is False
    assert "human approval" in result["auto_approval"]["error"]
    assert pending_skill_proposal_count(memory_dir) == 1
    assert "Original workflow." in (installed / "SKILL.md").read_text(encoding="utf-8")

    approved = approve_skill_proposal(
        memory_dir,
        result["proposal_id"],
        skills_dir=workspace.skills_dir,
    )
    assert approved["approved"] is True
    assert approved["workspace_shadow"] is None
    assert "Updated workflow." in (installed / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_update_rejects_file_symlink_escaping_install_destination(
    tmp_path,
    monkeypatch,
):
    """A nested link must not redirect part of an update outside its skill."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    installed = _write_installed_skill(
        workspace_skills,
        "nested-link",
        "Use when testing nested update links.",
        "# Nested link\n\nOriginal workspace workflow.\n",
    )
    protected = global_skills / "protected" / "check.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("original global file\n", encoding="utf-8")
    (installed / "scripts").mkdir()
    (installed / "scripts" / "check.py").symlink_to(protected)
    proposal_dir = _write_skill_folder(
        memory_dir,
        "nested-link",
        "Use when testing nested update links.",
        "# Nested link\n\nUpdated workspace workflow.\n",
    )
    (proposal_dir / "scripts").mkdir()
    (proposal_dir / "scripts" / "check.py").write_text(
        "updated global file\n", encoding="utf-8"
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="nested-link",
        cluster_hash="cluster-nested-link",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="No proposal file may write through an external link.",
        operation="update",
        target_skill_name="nested-link",
        skills_dir=workspace_skills,
    )

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert proposal["scope"] == "workspace"
    assert result["approved"] is False
    assert "escapes its destination" in result["error"]
    assert protected.read_text(encoding="utf-8") == "original global file\n"
    assert "Original workspace workflow." in (installed / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert pending_skill_proposal_count(memory_dir) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_update_allows_symlink_inside_workspace_tier(tmp_path, monkeypatch):
    """A link that stays inside its tier remains a valid update target."""
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    installed = _write_installed_skill_in_dir(
        workspace_skills,
        "a-actual",
        skill_name="internal-link",
        description="Use when testing internal update links.",
        body="# Internal link\n\nOriginal workflow.\n",
    )
    (workspace_skills / "z-alias").symlink_to(installed, target_is_directory=True)
    _write_skill_folder(
        memory_dir,
        "internal-link",
        "Use when testing internal update links.",
        "# Internal link\n\nUpdated workflow.\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="internal-link",
        cluster_hash="cluster-internal-link",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Internal links follow the runtime's normal visibility rules.",
        operation="update",
        target_skill_name="internal-link",
        skills_dir=workspace_skills,
    )

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert proposal["submitted"] is True
    assert proposal["scope"] == "workspace"
    assert result["approved"] is True
    assert "Updated workflow." in (installed / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
@pytest.mark.parametrize("target_kind", ["global", "other-workspace"])
def test_auto_mode_holds_when_workspace_tier_root_is_external(
    tmp_path,
    monkeypatch,
    target_kind,
):
    """AUTO must not write through a skills root owned by another tier."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace = Workspace(tmp_path / "workspace")
    workspace.root.mkdir()
    other_workspace = tmp_path / "other-workspace-skills"
    target_root = global_skills if target_kind == "global" else other_workspace
    installed = _write_installed_skill(
        target_root,
        "root-link",
        "Use when testing external tier roots.",
        "# Root link\n\nOriginal workflow.\n",
    )
    workspace.skills_dir.symlink_to(target_root, target_is_directory=True)
    _write_skill_folder(
        memory_dir,
        "root-link",
        "Use when testing external tier roots.",
        "# Root link\n\nUpdated workflow.\n",
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
                    "skill_name": "root-link",
                    "cluster_hash": f"cluster-root-{target_kind}",
                    "source_observation_ids": ["O-1", "O-2", "O-3"],
                    "rationale": "A root alias must not bypass human review.",
                    "operation": "update",
                    "target_skill_name": "root-link",
                }
            )
        )

    expected_scope = "global" if target_kind == "global" else "workspace"
    assert result["scope"] == expected_scope
    assert result["auto_approval"]["approved"] is False
    assert "human approval" in result["auto_approval"]["error"]
    assert "Original workflow." in (installed / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_update_uses_resolved_virtual_path_for_internal_alias(
    tmp_path,
    monkeypatch,
):
    """An internal alias must not invent a path absent from the runtime."""
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "skills"
    first = _write_installed_skill_in_dir(
        workspace_skills,
        "a",
        skill_name="alias-order",
        description="First physical skill.",
        body="# Alias order\n\nKeep first.\n",
    )
    winner = _write_installed_skill_in_dir(
        workspace_skills,
        "m",
        skill_name="alias-order",
        description="Runtime winner.",
        body="# Alias order\n\nRuntime winner.\n",
    )
    (workspace_skills / "z").symlink_to(first, target_is_directory=True)
    from EvoScientist.tools.skills_manager import installed_skill_index

    assert (
        installed_skill_index(Workspace(tmp_path))["alias-order"] == "Runtime winner."
    )
    _write_skill_folder(
        memory_dir,
        "alias-order",
        "Use when testing resolved virtual path ordering.",
        "# Alias order\n\nUpdated winner.\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="alias-order",
        cluster_hash="cluster-alias-order",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="The update must target the resolved runtime winner.",
        operation="update",
        target_skill_name="alias-order",
        skills_dir=workspace_skills,
    )

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert result["approved"] is True
    assert Path(result["path"]) == winner
    assert "Updated winner." in (winner / "SKILL.md").read_text(encoding="utf-8")
    assert "Keep first." in (first / "SKILL.md").read_text(encoding="utf-8")


def test_update_replaces_hardlinked_target_without_touching_global(
    tmp_path,
    monkeypatch,
):
    """Replacing a hardlinked workspace file must leave the global inode alone."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    global_skill = _write_installed_skill(
        global_skills,
        "hardlink-update",
        "Use when testing hardlinked updates.",
        "# Hardlink\n\nOriginal global.\n",
    )
    workspace_skill = _write_installed_skill(
        workspace_skills,
        "hardlink-update",
        "Use when testing hardlinked updates.",
        "# Hardlink\n\nWorkspace copy.\n",
    )
    global_file = global_skill / "SKILL.md"
    workspace_file = workspace_skill / "SKILL.md"
    workspace_file.unlink()
    os.link(global_file, workspace_file)
    _write_skill_folder(
        memory_dir,
        "hardlink-update",
        "Use when testing hardlinked updates.",
        "# Hardlink\n\nUpdated workspace.\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="hardlink-update",
        cluster_hash="cluster-hardlink-update",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="A workspace update must not modify a hardlinked global file.",
        operation="update",
        target_skill_name="hardlink-update",
        skills_dir=workspace_skills,
    )

    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert result["approved"] is True
    assert "Updated workspace." in workspace_file.read_text(encoding="utf-8")
    assert "Original global." in global_file.read_text(encoding="utf-8")
    assert not os.path.samefile(global_file, workspace_file)


def test_proposal_metadata_writes_do_not_follow_hardlinks(tmp_path, monkeypatch):
    """Manifest and rationale replacement must not overwrite linked files."""
    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    global_skills.mkdir()
    memory_dir = tmp_path / "memories"
    workspace_skills = tmp_path / "workspace-skills"
    proposal_dir = _write_skill_folder(
        memory_dir,
        "metadata-links",
        "Use when testing metadata link safety.",
        "# Metadata links\n",
    )
    protected_manifest = global_skills / "protected-manifest"
    protected_rationale = global_skills / "protected-rationale"
    protected_manifest.write_text("original manifest\n", encoding="utf-8")
    protected_rationale.write_text("original rationale\n", encoding="utf-8")
    os.link(protected_manifest, proposal_dir / "manifest.json")
    os.link(protected_rationale, proposal_dir / "RATIONALE.md")

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="metadata-links",
        cluster_hash="cluster-metadata-links",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Metadata writes must not follow hardlinks.",
        skills_dir=workspace_skills,
    )
    result = approve_skill_proposal(
        memory_dir,
        proposal["proposal_id"],
        skills_dir=workspace_skills,
    )

    assert proposal["submitted"] is True
    assert result["approved"] is True
    assert protected_manifest.read_text(encoding="utf-8") == "original manifest\n"
    assert protected_rationale.read_text(encoding="utf-8") == "original rationale\n"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_submit_rejects_proposal_directory_outside_root(tmp_path, monkeypatch):
    """A proposal directory link must not redirect metadata writes."""
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    memory_dir = tmp_path / "memories"
    skill_name = "external-proposal"
    outside = tmp_path / "outside" / skill_name
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: Test external proposals\n---\n\n# External\n",
        encoding="utf-8",
    )
    proposal_dir = autoskill_proposals_dir(memory_dir) / skill_name
    proposal_dir.parent.mkdir(parents=True)
    proposal_dir.symlink_to(outside, target_is_directory=True)

    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=skill_name,
        cluster_hash="cluster-external-proposal",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Proposal metadata must stay inside the proposal root.",
        skills_dir=tmp_path / "workspace-skills",
    )

    assert result["submitted"] is False
    assert "inside /autoskill-proposals" in result["errors"][0]
    assert not (outside / "manifest.json").exists()
    assert not (outside / "RATIONALE.md").exists()


@pytest.mark.parametrize(
    "cluster_hash",
    ["../escape", "/tmp/escape", "folder/name", "folder\\name", "C:\\escape"],
)
def test_submit_rejects_cluster_hash_path_traversal(
    tmp_path,
    monkeypatch,
    cluster_hash,
):
    """A cluster hash is a filename component, never a path."""
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "safe-cluster",
        "Use when testing safe cluster identifiers.",
        "# Safe cluster\n",
    )

    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="safe-cluster",
        cluster_hash=cluster_hash,
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Cluster identifiers must not escape the processed directory.",
        skills_dir=tmp_path / "workspace-skills",
    )

    assert result["submitted"] is False
    assert "cluster_hash" in result["error"]
    assert pending_skill_proposal_count(memory_dir) == 0
    assert not (tmp_path / "escape.json").exists()


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
        theirs = workspace_assistant_id(AUTOSKILL_GRAPH_ID, other)
        ours = workspace_assistant_id(AUTOSKILL_GRAPH_ID, workspace)
        self.rows = [
            {
                "cron_id": f"theirs-{i}",
                "assistant_id": theirs,
                "metadata": {"workspace_dir": other},
            }
            for i in range(1000)
        ] + [
            {
                "cron_id": "cron-async",
                "assistant_id": ours,
                "metadata": {"workspace_dir": workspace.key},
            }
        ]
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
