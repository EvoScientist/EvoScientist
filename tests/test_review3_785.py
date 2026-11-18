"""The seven findings from the second review of PR #853.

Each test names the finding it closes. They are written against the behaviour
the review proved was missing, so they fail on the PR and pass once the
approval trusts a record the agent cannot change.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from EvoScientist.memory.autoskills import proposals
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    move_proposals,
    revert_skill_proposal,
    submit_autoskill_proposal,
    upgrade_proposal_workspaces,
)


def record_reviewed_proposal_for(memory_dir, proposal_id):
    """Stand in for the display a person's decision requires.

    The production surfaces record a draft when they show it. A test that
    decides directly has shown the draft it is about to decide, so it records
    that current content. A body edited after this line is still refused.
    """
    from EvoScientist.memory.autoskills.proposals import record_reviewed_proposal_by_id

    record_reviewed_proposal_by_id(memory_dir, proposal_id)


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


def _rewrite_manifest(memory_dir: Path, name: str, **changes) -> None:
    path = autoskill_proposals_dir(memory_dir) / name / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _approve(
    memory_dir: Path,
    skills_dir: Path,
    name: str = "probe-skill",
    *,
    shown: bool = True,
):
    if shown:
        record_reviewed_proposal_for(memory_dir, name)
    return approve_skill_proposal(
        memory_dir, name, skills_dir=skills_dir, workspace_dir=skills_dir.parent
    )


def test_b1_manifest_scope_edit_does_not_install_globally(tmp_path, monkeypatch):
    """Editing scope in the manifest must not move a reviewed skill to global.

    The agent can write the proposals directory. A workspace proposal whose
    manifest is edited to ``scope=global`` after submit, with ``updated_at``
    left untouched, used to be installed into the global tier.
    """
    from EvoScientist import paths

    global_skills = tmp_path / "global-skills"
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", global_skills)
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    record_reviewed_proposal_for(memory_dir, "probe-skill")
    _rewrite_manifest(memory_dir, "probe-skill", scope="global")

    result = _approve(memory_dir, skills_dir, shown=False)

    assert result["approved"] is False, result
    assert "shown" in result["error"]
    assert not (global_skills / "probe-skill").exists()
    assert not (skills_dir / "probe-skill").exists()


def test_b1_operation_target_and_cluster_are_sealed(tmp_path):
    """The other decision fields are sealed the same way scope is."""
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    record_reviewed_proposal_for(memory_dir, "probe-skill")
    _rewrite_manifest(memory_dir, "probe-skill", cluster_hash="cluster-swapped")

    result = _approve(memory_dir, skills_dir, shown=False)

    assert result["approved"] is False, result
    assert "shown" in result["error"]
    assert not (skills_dir / "probe-skill").exists()


def test_b1_prefix_fresh_process_and_relist_approve_the_sealed_body(
    tmp_path, monkeypatch
):
    """A fresh command that never listed the draft refuses it.

    Approving by id shows the one draft the id resolves to, so the body it
    records is the body on disk at that moment. A body that changes after
    that display, before the approval copies it, is refused. Listing again
    shows the current body and records that, so the approval installs what
    the list showed.
    """
    from EvoScientist import paths
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand
    from EvoScientist.memory.autoskills import proposals as proposal_module

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    skill = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"

    real_approve = proposal_module.approve_skill_proposal

    def swap_after_display(*args, **kwargs):
        skill.write_text(
            skill.read_text(encoding="utf-8").replace(
                "Reviewed body.", "Swapped body."
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(proposal_module, "approve_skill_proposal", real_approve)
        return real_approve(*args, **kwargs)

    monkeypatch.setattr(proposal_module, "approve_skill_proposal", swap_after_display)

    messages: list[str] = []

    async def decide(argument: str) -> None:
        ctx = type("Ctx", (), {})()
        ctx.workspace = type("Ws", (), {"root": tmp_path, "skills_dir": skills_dir})()
        ctx.ui = type("Ui", (), {})()
        ctx.ui.append_system = lambda *args, **kwargs: messages.append(str(args[0]))
        ctx.ui.mount_renderable = lambda *args, **kwargs: None
        await AutoSkillsCommand().execute(ctx, ["approve", argument])

    import asyncio

    asyncio.run(decide("probe-s"))
    assert any("failed" in message.lower() for message in messages), messages
    assert not (skills_dir / "probe-skill").exists()

    # A brand new command object is a fresh process's empty state. Listing
    # again, which used to overwrite the reviewed version, changes nothing.
    messages.clear()

    async def relist_then_decide() -> None:
        ctx = type("Ctx", (), {})()
        ctx.workspace = type("Ws", (), {"root": tmp_path, "skills_dir": skills_dir})()
        ctx.ui = type("Ui", (), {})()
        ctx.ui.append_system = lambda *args, **kwargs: messages.append(str(args[0]))
        ctx.ui.mount_renderable = lambda *args, **kwargs: None
        command = AutoSkillsCommand()
        await command.execute(ctx, ["list"])
        await command.execute(ctx, ["approve", "probe-skill"])

    asyncio.run(relist_then_decide())
    # The list shows the body as it is now and records that body, so the
    # approval installs what the list showed. A listing is not a way to
    # approve an older body.
    assert any("Approved autoskill" in message for message in messages), messages
    assert "Swapped body." in (skills_dir / "probe-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert not hasattr(AutoSkillsCommand, "_reviewed_versions")


def test_b2_deleting_the_manifest_digest_does_not_drop_the_content_check(tmp_path):
    """A manifest with its digest removed is still the submission that was sealed.

    Deleting ``content_digest`` used to turn the draft into a legacy draft,
    after which a body edit was installed.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    record_reviewed_proposal_for(memory_dir, "probe-skill")
    path = autoskill_proposals_dir(memory_dir) / "probe-skill" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("content_digest", None)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    skill = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace("Reviewed body.", "Edited body."),
        encoding="utf-8",
    )

    result = _approve(memory_dir, skills_dir, shown=False)

    assert result["approved"] is False, result
    assert "shown" in result["error"]
    assert not (skills_dir / "probe-skill").exists()


def test_b2_a_legacy_draft_with_a_newer_body_is_refused(tmp_path):
    """A draft whose seal was deleted is refused, whatever its timestamps say.

    A missing seal used to take an mtime fallback, and preserving the body's
    mtime, or refreshing the manifest's, hid the edit. There is no such
    fallback: a decision needs the digest this process recorded when it
    showed the draft, and this one was shown before the body changed.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Legacy body.")
    _submit(memory_dir, skills_dir)
    record_reviewed_proposal_for(memory_dir, "probe-skill")
    seal = memory_dir / "autoskills" / "seals" / "probe-skill.json"
    if seal.exists():
        seal.unlink()
    path = autoskill_proposals_dir(memory_dir) / "probe-skill" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("content_digest", None)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    skill = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"
    # A second past the manifest's timestamp: ``updated_at`` is whole seconds.
    time.sleep(1.1)
    skill.write_text(
        skill.read_text(encoding="utf-8").replace("Legacy body.", "Edited body."),
        encoding="utf-8",
    )

    result = _approve(memory_dir, skills_dir, shown=False)

    assert result["approved"] is False, result
    assert "shown" in result["error"]
    assert not (skills_dir / "probe-skill").exists()


def test_b2_a_genuinely_legacy_draft_is_still_approvable(tmp_path):
    """A draft with no seal is approvable once this process has shown it.

    The seal is not the review. Showing the draft records its current body,
    and that is what the approval installs. A draft that was never shown is
    refused, which is covered separately.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Legacy body.")
    _submit(memory_dir, skills_dir)
    seal = memory_dir / "autoskills" / "seals" / "probe-skill.json"
    if seal.exists():
        seal.unlink()

    result = _approve(memory_dir, skills_dir)

    assert result["approved"] is True, result
    assert "Legacy body." in (skills_dir / "probe-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_b3_a_legacy_approval_is_not_reverted_against_the_live_draft(tmp_path):
    """An approval with no recorded install digest cannot be undone.

    Revert used to compare the installed file with the live draft. After the
    draft changed, a user edit that matched the new draft was deleted.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Installed body.")
    _submit(memory_dir, skills_dir)
    approved = _approve(memory_dir, skills_dir)
    assert approved["approved"] is True, approved
    path = autoskill_proposals_dir(memory_dir) / "probe-skill" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["approved_content_digests"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    draft = autoskill_proposals_dir(memory_dir) / "probe-skill" / "SKILL.md"
    changed = draft.read_text(encoding="utf-8").replace(
        "Installed body.", "Changed body."
    )
    draft.write_text(changed, encoding="utf-8")
    installed = skills_dir / "probe-skill" / "SKILL.md"
    installed.write_text(changed, encoding="utf-8")

    result = revert_skill_proposal(memory_dir, "probe-skill")

    assert result["reverted"] is False, result
    assert result["error"] == "skill_changed"
    assert installed.is_file()
    assert "Changed body." in installed.read_text(encoding="utf-8")


def test_b3_the_draft_digest_does_not_hold_the_whole_file(tmp_path, monkeypatch):
    """Hashing a draft streams it: the file is never joined back into one bytes."""
    memory_dir = tmp_path / "memories"
    folder = _draft(memory_dir, "probe-skill", "streamed body")
    seen: list[int] = []
    real_read = os.read

    def bounded(fd, size):
        assert size <= proposals._COMPARE_CHUNK
        chunk = real_read(fd, min(size, 8))
        if chunk:
            seen.append(len(chunk))
        return chunk

    monkeypatch.setattr(proposals.os, "read", bounded)
    first = proposals._draft_digest(folder)
    assert seen, "the file was not read"
    assert max(seen) <= 8
    monkeypatch.undo()
    assert proposals._draft_digest(folder) == first


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="FIFO creation is not available on this platform",
)
def test_b4_installed_digest_does_not_reopen_after_the_check(tmp_path, monkeypatch):
    """``_file_digest`` opens the file once. A second open is the FIFO window."""
    target = tmp_path / "installed.md"
    target.write_bytes(b"installed bytes")
    opens: list[str] = []
    real_open = os.open

    def counting(path, flags, *args, **kwargs):
        # The stub replaces the whole ``os.open``, so it has to accept every
        # call convention the standard library uses -- ``dir_fd`` among them,
        # which POSIX callers pass and Windows does not.
        opens.append(os.fsdecode(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(proposals.os, "open", counting)
    assert proposals._file_digest(target)
    assert opens.count(str(target)) == 1


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="FIFO creation is not available on this platform",
)
def test_b4_a_fifo_swapped_into_the_digest_does_not_block(tmp_path, monkeypatch):
    """Replacing the file after the listing must not block the digest."""
    memory_dir = tmp_path / "memories"
    folder = _draft(memory_dir, "probe-skill", "body")
    target = folder / "SKILL.md"
    real_copied = proposals.copied_files

    def swap(directory):
        files = real_copied(directory)
        if directory == folder:
            target.unlink()
            os.mkfifo(target)
        return files

    monkeypatch.setattr(proposals, "copied_files", swap)
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = proposals._draft_digest(folder)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "the digest blocked on a FIFO"
    assert isinstance(outcome.get("error"), OSError)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="FIFO creation is not available on this platform",
)
def test_b4_approval_copies_through_the_checked_descriptor(tmp_path, monkeypatch):
    """The snapshot copy opens the live draft once, for the check and the read."""
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    draft = autoskill_proposals_dir(memory_dir) / "probe-skill"
    live = os.fsdecode(draft / "SKILL.md")

    # The copy must read the live draft through the descriptor that proved
    # it regular, so it may not open it a second time. ``_open_regular`` is
    # that one open, so the copy is watched there: counting every
    # ``os.open`` would also count the validation scan and the digest read,
    # and the count differs between platforms.
    copied: list[str] = []

    real_copy = proposals._copy_regular_file

    def recording_copy(source, target):
        copied.append(os.fsdecode(source))
        return real_copy(source, target)

    monkeypatch.setattr(proposals, "_copy_regular_file", recording_copy)
    monkeypatch.setattr(proposals, "_notify_approved_skill", lambda: None)
    result = _approve(memory_dir, skills_dir)
    assert result["approved"] is True, result
    assert copied.count(live) == 1, copied


def test_move_and_upgrade_take_the_proposal_lock(tmp_path):
    """A workspace move cannot write a manifest an approval is committing.

    Both writers take the per-draft lock and the mover re-reads the manifest
    under it, so the approval's record survives.
    """
    from EvoScientist.memory.autoskills.proposals import _proposal_file_lock

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    from EvoScientist.paths import Workspace

    workspace = Workspace(skills_dir.parent)
    submitted = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="probe-skill",
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="One observation is enough.",
        skills_dir=skills_dir,
        workspace_dir=workspace.root,
    )
    assert submitted["submitted"] is True, submitted
    record_reviewed_proposal_for(memory_dir, "probe-skill")
    approved = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=workspace.root,
    )
    assert approved["approved"] is True, approved
    before = _manifest(memory_dir, "probe-skill")
    proposal_dir = autoskill_proposals_dir(memory_dir) / "probe-skill"
    entered = threading.Event()
    moved: dict[str, int] = {}

    def mover() -> None:
        entered.set()
        moved["count"] = move_proposals(
            memory_dir,
            workspace.key,
            Workspace(tmp_path / "moved").key,
            {},
        )

    with _proposal_file_lock(proposal_dir):
        worker = threading.Thread(target=mover)
        worker.start()
        assert entered.wait(5)
        time.sleep(0.2)
        assert worker.is_alive(), "move_proposals did not wait for the draft lock"
    worker.join(10)
    assert not worker.is_alive()
    # The held lock serialized the move. Once it ran, it kept the approval and
    # moved the installed path with the workspace instead of dropping it.
    assert _manifest(memory_dir, "probe-skill")["status"] == "approved"
    assert _manifest(memory_dir, "probe-skill")["approved_skill_path"] == str(
        Workspace(tmp_path / "moved").skills_dir / "probe-skill"
    )
    assert "approved_skill_path" in before
    assert moved["count"] == 1


def test_upgrade_waits_for_the_same_lock(tmp_path):
    """The one-time workspace upgrade takes the per-draft lock too."""
    from EvoScientist.memory.autoskills.proposals import _proposal_file_lock

    memory_dir = tmp_path / "memories"
    run = tmp_path / "project" / "runs" / "20260101_000000"
    run.mkdir(parents=True)
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
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
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    entered = threading.Event()

    def upgrade() -> None:
        entered.set()
        upgrade_proposal_workspaces(memory_dir)

    with _proposal_file_lock(folder):
        worker = threading.Thread(target=upgrade)
        worker.start()
        assert entered.wait(5)
        time.sleep(0.2)
        waiting = worker.is_alive()
    worker.join(10)
    assert waiting, "upgrade_proposal_workspaces did not wait for the draft lock"
    assert not worker.is_alive()
