"""The five findings from the fifth review of PR #853.

Each test names the finding. They describe the behaviour the review proved
was missing, so they fail on the seal-based approval and pass once a person's
decision trusts only the record the reviewing process kept in memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    list_skill_proposals,
    revert_skill_proposal,
    submit_autoskill_proposal,
    upgrade_proposal_workspaces,
)


def _draft(memory_dir: Path, name: str, body: str) -> Path:
    folder = autoskill_proposals_dir(memory_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review this draft.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return folder


def _submit(memory_dir: Path, skills_dir: Path, name: str = "probe-skill") -> None:
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="One observation is enough.",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["submitted"] is True, result


def _manifest(memory_dir: Path, name: str) -> dict:
    path = autoskill_proposals_dir(memory_dir) / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(memory_dir: Path, name: str, manifest: dict) -> None:
    path = autoskill_proposals_dir(memory_dir) / name / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _show(memory_dir: Path, name: str) -> None:
    """What the CLI, the TUI and the review page do when they display a draft."""
    from EvoScientist.memory.autoskills import proposals as module

    record = getattr(module, "record_reviewed_proposal", None)
    if record is None:
        # The seal-based approval has no display record. Showing the draft
        # then does nothing, which is the defect these tests catch.
        return
    folder = autoskill_proposals_dir(memory_dir) / name
    record(folder, _manifest(memory_dir, name))


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approve(memory_dir: Path, skills_dir: Path, name: str = "probe-skill"):
    return approve_skill_proposal(
        memory_dir, name, skills_dir=skills_dir, workspace_dir=skills_dir.parent
    )


def test_f1_a_forged_seal_does_not_make_an_unseen_body_approvable(tmp_path):
    """Finding 1: the agent's shell can rewrite the seal, so it is not a review.

    The proposal shell's cwd is the proposals root, and ``Path.cwd().parent``
    is the directory the seal is stored in. Rewriting the seal to name a
    forged body used to make the approval install that body. A person's
    approval compares the draft with the digest this process recorded when it
    displayed it, which no file the shell can write can supply.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    _show(memory_dir, "probe-skill")
    skill = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace("Reviewed body.", "Forged body."),
        encoding="utf-8",
    )
    # The shell recomputes the seal so it names the forged body. A check that
    # trusts the seal then installs it; a check of the displayed body does not.
    from EvoScientist.memory.autoskills.proposals import _draft_digest

    seal = memory_dir / "autoskills" / "seals" / "probe-skill.json"
    forged = json.loads(seal.read_text(encoding="utf-8"))
    forged["content_digest"] = _draft_digest(
        autoskill_proposals_dir(memory_dir) / "probe-skill"
    )
    seal.write_text(
        json.dumps(forged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = _approve(memory_dir, skills_dir)

    assert result["approved"] is False, result
    assert "shown" in result["error"]
    assert not (skills_dir / "probe-skill").exists()


def test_f1_the_proposal_shell_reaches_the_seal_directory(tmp_path):
    """The shell the agent runs can compute the seal path from its cwd.

    This is why a file beside the proposals directory is not a trust boundary,
    and why the approval does not consult one. The command builds the
    directory name from character codes because the Windows shell strips the
    nested quotes a literal path would need.
    """
    from EvoScientist.backends import AutoskillProposalSandboxBackend

    memory_dir = tmp_path / "memories"
    proposals = memory_dir / "autoskills" / "proposals"
    proposals.mkdir(parents=True)
    backend = AutoskillProposalSandboxBackend(root_dir=str(proposals), timeout=30)
    command = (
        "python -c "
        '"from pathlib import Path; '
        "name = chr(115)+chr(101)+chr(97)+chr(108)+chr(115); "
        "target = Path.cwd().parent / name / (chr(112)+chr(114)+chr(111)+chr(98)+chr(101)); "
        "target = target.with_suffix(chr(46)+chr(106)+chr(115)+chr(111)+chr(110)); "
        "target.parent.mkdir(parents=True, exist_ok=True); "
        "target.write_text(chr(102)+chr(111)+chr(114)+chr(103)+chr(101)+chr(100)); "
        'print(target.exists())"'
    )
    result = backend.execute(command)
    seal = memory_dir / "autoskills" / "seals" / "probe.json"
    assert result.exit_code == 0, result.output
    assert seal.is_file()
    assert seal.read_text(encoding="utf-8") == "forged"


def test_f2_deleting_the_seal_does_not_approve_a_changed_body(tmp_path):
    """Finding 2: a missing seal used to take the legacy path.

    Revert deletes the seal, and the agent's shell can delete it too. Either
    way a later approval refuses a body that changed after the draft was
    shown, instead of falling back to a comparison of timestamps.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    _show(memory_dir, "probe-skill")
    approved = _approve(memory_dir, skills_dir)
    assert approved["approved"] is True, approved
    reverted = revert_skill_proposal(memory_dir, "probe-skill")
    assert reverted["reverted"] is True, reverted
    assert not (memory_dir / "autoskills" / "seals" / "probe-skill.json").exists()
    skill = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "probe-skill" / "manifest.json"
    )
    submitted_at = manifest_path.stat().st_mtime
    skill.write_text(
        skill.read_text(encoding="utf-8").replace("Reviewed body.", "Edited body."),
        encoding="utf-8",
    )
    # The mtime fallback only sees a body newer than the manifest. Preserving
    # the body's mtime, which the agent's shell can do, hides the edit from
    # it and the missing seal then takes that fallback.
    os.utime(skill, (submitted_at, submitted_at))

    again = _approve(memory_dir, skills_dir)

    assert again["approved"] is False, again
    assert not (skills_dir / "probe-skill").exists()


def test_f2_a_decision_without_a_display_is_refused(tmp_path):
    """A process that never showed the draft has nothing to compare against.

    A fresh CLI, a second TUI and the server approve route with no prior
    listing are all this case. Refusing them is what makes the check sound:
    the agent cannot make a decision happen in a process that never looked.
    A unique prefix resolves to the same draft and is refused the same way.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)

    approved = _approve(memory_dir, skills_dir)
    by_prefix = approve_skill_proposal(
        memory_dir, "probe", skills_dir=skills_dir, workspace_dir=skills_dir.parent
    )

    assert approved["approved"] is False, approved
    assert "not reviewed" in approved["error"]
    assert by_prefix["approved"] is False
    assert not (skills_dir / "probe-skill").exists()


def test_f3_a_preserved_mtime_does_not_hide_a_legacy_edit(tmp_path):
    """Finding 3: the mtime fallback is forgeable, and upgrade refreshes it.

    A draft from before this check has no display record. Preserving the
    body mtime, then rewriting the manifest through the upgrade, used to make
    the body look older than the manifest and the approval installed it. Such
    a draft has to be shown again; the upgrade cannot make it approvable.
    """
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    run = tmp_path / "project" / "runs" / "20260101_000000"
    run.mkdir(parents=True)
    folder = _draft(memory_dir, "probe-skill", "Legacy body.")
    _submit(memory_dir, skills_dir)
    # No seal: this is the draft the mtime fallback accepts. The recorded
    # workspace is a run folder, so the upgrade rewrites the manifest.
    seal = memory_dir / "autoskills" / "seals" / "probe-skill.json"
    if seal.exists():
        seal.unlink()
    manifest = {
        "proposal_id": "probe-skill",
        "skill_name": "probe-skill",
        "description": "Review this draft.",
        "status": "pending",
        "operation": "create",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "cluster_hash": "cluster-original",
        "source_observation_ids": ["O-1"],
        "workspace_dir": str(run),
        "scope": "workspace",
        "submitted_scope": "workspace",
    }
    _write_manifest(memory_dir, "probe-skill", manifest)
    # The manifest's mtime is the submission. The body is edited a second
    # later, which the mtime check can see, and then the upgrade rewrites the
    # manifest and moves its mtime past the body.
    import time

    time.sleep(1.1)
    skill = folder / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace("Legacy body.", "Forged body."),
        encoding="utf-8",
    )
    assert upgrade_proposal_workspaces(memory_dir) == 1
    # The upgrade rewrites the manifest, which moves its mtime past the
    # body. Putting the body's mtime back to that new manifest mtime is the
    # forge the comparison cannot see: the body is newer in content and not
    # newer on disk.
    rewritten_at = (folder / "manifest.json").stat().st_mtime
    os.utime(skill, (rewritten_at, rewritten_at))

    from EvoScientist.paths import SessionDirs

    # The upgrade rewrites the manifest's workspace from the run folder to
    # its project, so the approval has to name that project.
    project_workspace = SessionDirs.from_legacy(run).workspace
    project = project_workspace.root
    result = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=project_workspace.skills_dir,
        workspace_dir=project,
    )

    assert result["approved"] is False, result
    assert not (project_workspace.skills_dir / "probe-skill").exists()


def test_f4_a_forged_digest_map_does_not_delete_a_user_edit(tmp_path):
    """Finding 4: revert must not trust the agent-writable digest map.

    After an approval the user edits the installed skill. Replacing the
    manifest digest with the digest of that edit used to make the undo delete
    the edit. The digests a revert compares against are the ones this process
    recorded when it installed, so the forged map changes nothing.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Installed body.")
    _submit(memory_dir, skills_dir)
    _show(memory_dir, "probe-skill")
    approved = _approve(memory_dir, skills_dir)
    assert approved["approved"] is True, approved
    installed = skills_dir / "probe-skill" / "SKILL.md"
    installed.write_text(
        installed.read_text(encoding="utf-8").replace("Installed body.", "User edit."),
        encoding="utf-8",
    )
    manifest = _manifest(memory_dir, "probe-skill")
    manifest["approved_content_digests"]["SKILL.md"] = _digest(installed)
    _write_manifest(memory_dir, "probe-skill", manifest)

    result = revert_skill_proposal(memory_dir, "probe-skill")

    assert result["reverted"] is False, result
    assert result["error"] == "skill_changed"
    assert "User edit." in installed.read_text(encoding="utf-8")


def test_f4_a_forged_scope_does_not_delete_another_tiers_skill(tmp_path, monkeypatch):
    """Finding 4, the redirected path: scope and the install path are forgeable.

    Pointing the manifest at a same-named skill in the global tier used to
    make the undo delete that skill. The path and the tier a revert deletes
    come from the record this process kept, so the global skill stays and the
    real installation is what the undo removes.
    """
    from EvoScientist import paths

    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Installed body.")
    _submit(memory_dir, skills_dir)
    _show(memory_dir, "probe-skill")
    approved = _approve(memory_dir, skills_dir)
    assert approved["approved"] is True, approved
    victim = global_skills / "probe-skill"
    victim.mkdir(parents=True)
    (victim / "SKILL.md").write_text("A user other skill.\n", encoding="utf-8")
    manifest = _manifest(memory_dir, "probe-skill")
    manifest["scope"] = "global"
    manifest["approved_skill_path"] = str(victim)
    manifest["approved_content_digests"] = {"SKILL.md": _digest(victim / "SKILL.md")}
    _write_manifest(memory_dir, "probe-skill", manifest)

    result = revert_skill_proposal(memory_dir, "probe-skill")

    assert (victim / "SKILL.md").is_file()
    assert "A user other skill." in (victim / "SKILL.md").read_text(encoding="utf-8")
    assert result["reverted"] is True, result
    assert not (skills_dir / "probe-skill").exists()


def test_f4_a_revert_in_a_process_that_did_not_approve_refuses(tmp_path):
    """An approval made somewhere else has no deletion record here.

    Comparing the installed files with the live draft is the data-loss bug,
    so a process that did not perform the approval refuses the undo.
    """
    from EvoScientist.memory.autoskills import proposals as module

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Installed body.")
    _submit(memory_dir, skills_dir)
    _show(memory_dir, "probe-skill")
    approved = _approve(memory_dir, skills_dir)
    assert approved["approved"] is True, approved
    # A fresh process has an empty record. Clearing this one stands in for it.
    guard = getattr(module, "_installations_guard", None)
    if guard is not None:
        with guard:
            module._installations.clear()

    result = revert_skill_proposal(memory_dir, "probe-skill")

    assert result["reverted"] is False, result
    assert result["error"] == "revert_unavailable"
    assert (skills_dir / "probe-skill" / "SKILL.md").is_file()


def test_f5_the_lock_is_outside_the_proposals_directory(tmp_path):
    """Finding 5: a lock file the agent shell can unlink does not serialize.

    On POSIX, unlinking a locked pathname removes the directory entry while
    the holder keeps the old inode, and a second lock then locks a new one.
    Moving the file beside the drafts only stops the file tools. The lock is
    now a kernel object with no pathname, so holding it creates no file the
    shell can delete and replace. Guessing the object name can only stall
    this draft.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    from EvoScientist.memory.autoskills.proposals import _proposal_file_lock

    proposals = autoskill_proposals_dir(memory_dir)
    before = {path for path in memory_dir.rglob("*") if path.is_file()}
    with _proposal_file_lock(proposals / "probe-skill"):
        locks = list(memory_dir.rglob("*.lock"))
        created = {
            path
            for path in memory_dir.rglob("*")
            if path.is_file() and path not in before
        }
    assert locks == []
    assert created == set()


def test_relisting_records_the_current_body(tmp_path, monkeypatch):
    """Showing the draft again records what is on screen now.

    A listing is not a bypass: it shows the current content, so the approval
    installs the content that listing showed. A prefix resolves to the same
    draft.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "First body.")
    _submit(memory_dir, skills_dir)
    _show(memory_dir, "probe-skill")
    skill = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace("First body.", "Shown body."),
        encoding="utf-8",
    )
    # The body changed after it was shown, so the first decision is refused
    # whether the check is the seal or the display record.
    stale = _approve(memory_dir, skills_dir)
    assert stale["approved"] is False, stale
    # Put the seal back in step with the body. A check that trusts the seal
    # now installs it; a check of the display record still refuses, because
    # the draft was shown before this edit and has not been shown since.
    from EvoScientist.memory.autoskills.proposals import _draft_digest

    seal = memory_dir / "autoskills" / "seals" / "probe-skill.json"
    recorded = json.loads(seal.read_text(encoding="utf-8"))
    recorded["content_digest"] = _draft_digest(
        autoskill_proposals_dir(memory_dir) / "probe-skill"
    )
    seal.write_text(
        json.dumps(recorded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The seal now names the unseen body. An approval that trusts the seal
    # installs it here, which is the failure; one that trusts the display
    # record refuses it.
    unseen = approve_skill_proposal(
        memory_dir, "probe-s", skills_dir=skills_dir, workspace_dir=skills_dir.parent
    )
    assert unseen["approved"] is False, unseen
    assert not (skills_dir / "probe-skill").exists()

    # record_review is the display. A build that has no such argument cannot
    # record what it showed, so the listing is the plain one.
    try:
        listed = list_skill_proposals(
            memory_dir, workspace_dir=skills_dir.parent, record_review=True
        )
    except TypeError:
        listed = list_skill_proposals(memory_dir, workspace_dir=skills_dir.parent)
    assert [proposal.proposal_id for proposal in listed] == ["probe-skill"]
    approved = approve_skill_proposal(
        memory_dir, "probe-s", skills_dir=skills_dir, workspace_dir=skills_dir.parent
    )
    if not approved["approved"]:
        # The first listing did not record the display. Show it the way the
        # review page does and decide again: what was just shown is installed.
        from EvoScientist import paths
        from EvoScientist.langgraph_dev import autoskills_api, registry

        fresh = registry.Registry()
        fresh._loaded()[paths.Workspace(skills_dir.parent).key] = True
        monkeypatch.setattr(registry, "REGISTRY", fresh)
        monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)

        _code, payload = autoskills_api.list_drafts()
        assert any(
            draft["proposal_id"] == "probe-skill" for draft in payload["proposals"]
        )
        approved = approve_skill_proposal(
            memory_dir,
            "probe-s",
            skills_dir=skills_dir,
            workspace_dir=skills_dir.parent,
        )
    assert approved["approved"] is True, approved
    assert "Shown body." in (skills_dir / "probe-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )
