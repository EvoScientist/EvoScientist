from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
from filelock import Timeout

from EvoScientist.memory.autoskills import proposals as module
from EvoScientist.memory.autoskills.proposals import (
    _draft_lock,
    approve_skill_proposal,
    autoskill_proposals_dir,
    reject_skill_proposal,
    revert_skill_proposal,
    submit_autoskill_proposal,
)


def _write_draft(memory_dir: Path, name: str, body: str = "Original body.") -> Path:
    folder = autoskill_proposals_dir(memory_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Exercise the draft lock.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return folder


def _submit(
    memory_dir: Path, skills_dir: Path, name: str, cluster: str, *, must_submit=True
):
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash=cluster,
        source_observation_ids=["O-1"],
        rationale="Exercise the per-draft writer lock.",
        skills_dir=skills_dir,
        workspace_dir=memory_dir.parent,
    )
    assert result["submitted"] is must_submit, result
    return result


def _manifest(memory_dir: Path, name: str) -> dict:
    path = autoskill_proposals_dir(memory_dir) / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _call(results: dict, key: str, done: threading.Event, function) -> None:
    try:
        results[key] = function()
    except BaseException as exc:
        results[key] = exc
    finally:
        done.set()


@pytest.mark.parametrize("first_action", ["submit", "reject"])
def test_draft_writers_same_draft_do_not_interleave(
    tmp_path, monkeypatch, first_action
):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    folder = _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    original_version = _manifest(memory_dir, name)["updated_at"]
    manifest_path = folder / "manifest.json"
    entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()
    real_write = module._write_manifest
    real_lock = module._draft_lock
    paused_status = "pending" if first_action == "submit" else "rejected"

    def paused_write(path, payload):
        if (
            threading.current_thread().name == "first-writer"
            and path == manifest_path
            and payload.get("status") == paused_status
        ):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("the test did not release the first writer")
        return real_write(path, payload)

    def observed_lock(proposal_dir):
        if threading.current_thread().name == "second-writer":
            second_attempted.set()
        return real_lock(proposal_dir)

    def first():
        if first_action == "submit":
            return _submit(memory_dir, skills_dir, name, "cluster-resubmitted")
        return reject_skill_proposal(memory_dir, name)

    def second():
        if first_action == "submit":
            return reject_skill_proposal(
                memory_dir, name, expected_updated_at=original_version
            )
        return _submit(
            memory_dir, skills_dir, name, "cluster-resubmitted", must_submit=False
        )

    monkeypatch.setattr(module, "_write_manifest", paused_write)
    monkeypatch.setattr(module, "_draft_lock", observed_lock)
    results = {}
    first_thread = threading.Thread(
        target=_call,
        args=(results, "first", first_done, first),
        name="first-writer",
    )
    second_thread = threading.Thread(
        target=_call,
        args=(results, "second", second_done, second),
        name="second-writer",
    )
    first_thread.start()
    second_started = False
    try:
        assert entered.wait(10)
        second_thread.start()
        second_started = True
        assert second_attempted.wait(10)
        assert not second_done.is_set()
    finally:
        release.set()
        first_thread.join(15)
        if second_started:
            second_thread.join(15)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not isinstance(results["first"], BaseException)
    assert not isinstance(results["second"], BaseException)
    manifest = _manifest(memory_dir, name)
    if first_action == "submit":
        assert results["first"]["submitted"] is True
        assert results["second"]["rejected"] is False
        assert results["second"]["error"] == "Proposal changed since it was read"
        assert manifest["status"] == "pending"
        assert manifest["cluster_hash"] == "cluster-resubmitted"
    else:
        assert results["first"]["rejected"] is True
        assert results["second"]["submitted"] is False
        assert results["second"]["status"] == "rejected"
        assert manifest["status"] == "rejected"
        assert manifest["cluster_hash"] == "cluster-original"


def test_draft_writers_different_drafts_do_not_block_each_other(tmp_path, monkeypatch):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    for name in ("alpha", "beta"):
        _write_draft(memory_dir, name)
        _submit(memory_dir, skills_dir, name, f"cluster-{name}")
    alpha_manifest = autoskill_proposals_dir(memory_dir) / "alpha" / "manifest.json"
    entered = threading.Event()
    release = threading.Event()
    beta_done = threading.Event()
    alpha_done = threading.Event()
    real_write = module._write_manifest

    def pause_alpha(path, payload):
        if (
            threading.current_thread().name == "alpha-writer"
            and path == alpha_manifest
            and payload.get("status") == "rejected"
        ):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("the test did not release alpha")
        return real_write(path, payload)

    monkeypatch.setattr(module, "_write_manifest", pause_alpha)
    results = {}
    alpha = threading.Thread(
        target=_call,
        args=(
            results,
            "alpha",
            alpha_done,
            lambda: reject_skill_proposal(memory_dir, "alpha"),
        ),
        name="alpha-writer",
    )
    beta = threading.Thread(
        target=_call,
        args=(
            results,
            "beta",
            beta_done,
            lambda: reject_skill_proposal(memory_dir, "beta"),
        ),
        name="beta-writer",
    )
    alpha.start()
    try:
        assert entered.wait(10)
        beta.start()
        beta_finished_while_alpha_was_paused = beta_done.wait(10)
    finally:
        release.set()
        alpha.join(15)
        if beta.ident is not None:
            beta.join(15)

    assert beta_finished_while_alpha_was_paused
    assert results["alpha"]["rejected"] is True
    assert results["beta"]["rejected"] is True
    assert _manifest(memory_dir, "alpha")["status"] == "rejected"
    assert _manifest(memory_dir, "beta")["status"] == "rejected"


@pytest.mark.parametrize(
    ("action", "flag"),
    [
        ("submit", "submitted"),
        ("approve", "approved"),
        ("reject", "rejected"),
        ("revert", "reverted"),
    ],
)
def test_draft_lock_timeout_is_a_refusal_for_every_writer(
    tmp_path, monkeypatch, action, flag
):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    folder = _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    monkeypatch.setattr(module, "_DRAFT_LOCK_TIMEOUT", 0.01)
    calls = {
        "submit": lambda: _submit(
            memory_dir, skills_dir, name, "cluster-new", must_submit=False
        ),
        "approve": lambda: approve_skill_proposal(
            memory_dir, name, skills_dir=skills_dir
        ),
        "reject": lambda: reject_skill_proposal(memory_dir, name),
        "revert": lambda: revert_skill_proposal(memory_dir, name),
    }
    done = threading.Event()
    results = {}
    with _draft_lock(folder):
        writer = threading.Thread(
            target=_call,
            args=(results, action, done, calls[action]),
            name="contending-writer",
        )
        writer.start()
        finished = done.wait(5)
    writer.join(5)

    assert finished, "the contending writer hung instead of refusing"
    assert not writer.is_alive()
    assert results[action][flag] is False
    assert "timed out waiting for another writer" in results[action]["error"]
    assert _manifest(memory_dir, name)["status"] == "pending"


def test_decision_manifest_reread_sees_write_before_lock_acquisition(
    tmp_path, monkeypatch
):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    original_version = _manifest(memory_dir, name)["updated_at"]
    real_lock = module._draft_lock

    def write_between_lookup_and_lock(proposal_dir):
        manifest_path = proposal_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["updated_at"] = "2026-01-02T03:04:05.006Z"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return real_lock(proposal_dir)

    monkeypatch.setattr(module, "_draft_lock", write_between_lookup_and_lock)

    result = reject_skill_proposal(
        memory_dir, name, expected_updated_at=original_version
    )

    assert result == {
        "rejected": False,
        "proposal_id": name,
        "updated_at": "2026-01-02T03:04:05.006Z",
        "error": "Proposal changed since it was read",
    }
    assert _manifest(memory_dir, name)["status"] == "pending"


def test_draft_lock_is_shared_by_aliased_proposals_roots(tmp_path):
    real_memory, alias_memory = tmp_path / "real", tmp_path / "alias"
    draft = _write_draft(real_memory, "focused-validation")
    alias_root = autoskill_proposals_dir(alias_memory)
    alias_root.parent.mkdir(parents=True)
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(draft.parent), str(alias_root))
    else:
        alias_root.symlink_to(draft.parent, target_is_directory=True)
    alias = alias_root / draft.name
    assert alias.resolve() == draft.resolve()

    lock, alias_lock = _draft_lock(draft), _draft_lock(alias)
    assert Path(lock.lock_file) == Path(alias_lock.lock_file)
    with lock, pytest.raises(Timeout):
        alias_lock.acquire(timeout=0)


def test_approval_refuses_a_workspace_move_between_lookup_and_lock(
    tmp_path, monkeypatch
):
    old, new = tmp_path / "old", tmp_path / "new"
    new.mkdir()
    memory_dir, skills_dir = old / "memories", old / "skills"
    name = "focused-validation"
    draft = _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    old_key = _manifest(memory_dir, name)["workspace_dir"]
    new_key = module._normalize_workspace_dir(new)
    assert new_key is not None
    real_lock = module._draft_lock
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    moved = False
    moving = False

    def move_then_lock(proposal_dir):
        nonlocal moved, moving
        assert proposal_dir == draft
        if not moved and not moving:
            # The move takes each draft's lock itself; those must not come back here.
            moving = True
            try:
                assert module.move_proposals(memory_dir, old_key, new_key, {}) == 1
            finally:
                moving = False
            moved = True
        return real_lock(proposal_dir)

    monkeypatch.setattr(module, "_draft_lock", move_then_lock)
    result = approve_skill_proposal(
        memory_dir, name, workspace_dir=old, skills_dir=skills_dir
    )

    assert moved
    assert result["approved"] is False, result
    assert "moved to a different workspace" in result["error"]
    manifest = _manifest(memory_dir, name)
    assert manifest["workspace_dir"] == new_key
    assert manifest["status"] == "pending"
    assert not (skills_dir / name).exists()


def test_a_resubmit_waits_for_an_approval_that_is_installing(tmp_path, monkeypatch):
    """The draft's lock is held until the approval has committed.

    A resubmit that arrives while the skill is being installed used to land
    before the approval's manifest write, which then overwrote it. It now
    waits, and starts from the approval the first writer committed.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    installed = skills_dir / name
    installed.mkdir(parents=True)
    base = f"---\nname: {name}\ndescription: Use when testing updates.\n---\n\nBase.\n"
    (installed / "SKILL.md").write_text(base, encoding="utf-8")

    def submit(body, cluster):
        _write_draft(memory_dir, name, body)
        result = submit_autoskill_proposal(
            memory_dir=memory_dir,
            skill_name=name,
            cluster_hash=cluster,
            source_observation_ids=["O-1"],
            rationale="Refine the installed skill.",
            skills_dir=skills_dir,
            workspace_dir=tmp_path,
            operation="update",
            target_skill_name=name,
        )
        assert result["submitted"] is True, result
        return result

    submit("First version.", "cluster-first")
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    approved = threading.Event()
    resubmitted = threading.Event()
    real_copy = module._copy_proposed_skill
    real_lock = module._draft_lock

    def pause_the_install(source, destination):
        if threading.current_thread().name == "approver":
            entered.set()
            if not release.wait(10):
                raise TimeoutError("the test did not release the approval")
        return real_copy(source, destination)

    def observed_lock(proposal_dir):
        if threading.current_thread().name == "resubmitter":
            second_attempted.set()
        return real_lock(proposal_dir)

    monkeypatch.setattr(module, "_copy_proposed_skill", pause_the_install)
    monkeypatch.setattr(module, "_draft_lock", observed_lock)
    results = {}
    approver = threading.Thread(
        target=_call,
        args=(
            results,
            "approve",
            approved,
            lambda: approve_skill_proposal(memory_dir, name, skills_dir=skills_dir),
        ),
        name="approver",
    )
    resubmitter = threading.Thread(
        target=_call,
        args=(
            results,
            "resubmit",
            resubmitted,
            lambda: submit("Second version.", "cluster-second"),
        ),
        name="resubmitter",
    )
    approver.start()
    resubmitter_started = False
    try:
        assert entered.wait(10)
        resubmitter.start()
        resubmitter_started = True
        assert second_attempted.wait(10)
        # The approval still holds the lock, so the resubmit cannot have finished.
        resubmitter.join(0.3)
        assert resubmitter.is_alive()
    finally:
        release.set()
        approver.join(15)
        if resubmitter_started:
            resubmitter.join(15)

    assert not approver.is_alive()
    assert not resubmitter.is_alive()
    assert not isinstance(results["approve"], BaseException), results["approve"]
    assert results["approve"]["approved"] is True, results["approve"]
    skill_md = installed / "SKILL.md"
    assert "First version." in skill_md.read_text(encoding="utf-8")
    assert results["resubmit"]["submitted"] is True, results["resubmit"]
    # The resubmission came last, so it is what the draft now holds.
    manifest = _manifest(memory_dir, name)
    assert manifest["status"] == "pending"
    assert manifest["cluster_hash"] == "cluster-second"


def test_the_skill_index_is_told_once_the_draft_lock_is_released(tmp_path, monkeypatch):
    """A callback that reads the draft must not wait on the approval's own lock.

    The skill index is told after every approval. Had the approval still held
    the draft's lock then, a callback that took it would wait out the timeout.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    folder = _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    monkeypatch.setattr(module, "_DRAFT_LOCK_TIMEOUT", 0.01)
    told = []

    def notify():
        with _draft_lock(folder):
            told.append(True)

    monkeypatch.setattr(module, "_notify_approved_skill", notify)

    result = approve_skill_proposal(memory_dir, name, skills_dir=skills_dir)

    assert result["approved"] is True, result
    assert told == [True]


def test_the_skill_index_is_told_once_the_revert_lock_is_released(
    tmp_path, monkeypatch
):
    """A callback that reads the draft must not wait on the revert's own lock.

    The skill index is told after an Undo as well, and it was told under the
    draft's lock until the approval and the revert were made to agree.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    folder = _write_draft(memory_dir, name)
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="Exercise the revert lock.",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["submitted"] is True, result
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    approved = approve_skill_proposal(memory_dir, name, skills_dir=skills_dir)
    assert approved["approved"] is True, approved
    monkeypatch.setattr(module, "_DRAFT_LOCK_TIMEOUT", 0.01)
    told = []

    def notify():
        with _draft_lock(folder):
            told.append(True)

    monkeypatch.setattr(module, "_notify_approved_skill", notify)

    reverted = revert_skill_proposal(memory_dir, name)

    assert reverted["reverted"] is True, reverted
    assert told == [True]


def test_the_skill_index_is_told_when_the_manifest_write_fails(tmp_path, monkeypatch):
    """A skill that is gone has to be told about, even if the Undo fails after.

    The removal happens before the manifest is written, so a write that fails
    leaves the skill gone and the draft decided. The index still has to hear
    about the removal, or it keeps listing a skill that is no longer there.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    _write_draft(memory_dir, name)
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="Exercise the revert lock.",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["submitted"] is True, result
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    approved = approve_skill_proposal(memory_dir, name, skills_dir=skills_dir)
    assert approved["approved"] is True, approved
    told = []
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: told.append(True))
    real_write = module._write_manifest

    def failing_write(path, payload):
        if payload.get("status") == "pending":
            raise OSError("disk full")
        return real_write(path, payload)

    monkeypatch.setattr(module, "_write_manifest", failing_write)

    reverted = revert_skill_proposal(memory_dir, name)

    assert reverted["reverted"] is False, reverted
    assert reverted["error"] == "revert_failed"
    assert told == [True]
    assert not (skills_dir / name).exists()


def test_a_move_does_not_overwrite_an_approval_committed_after_its_listing(
    tmp_path, monkeypatch
):
    old, new = tmp_path / "old", tmp_path / "new"
    new.mkdir()
    memory_dir, skills_dir = old / "memories", old / "skills"
    name = "focused-validation"
    _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    old_key = _manifest(memory_dir, name)["workspace_dir"]
    new_key = module._normalize_workspace_dir(new)
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    listed = module._proposal_manifests

    def approve_after_listing(memory, workspace_key):
        candidates = list(listed(memory, workspace_key))
        # The approval commits between the move's read and its write.
        decision = approve_skill_proposal(
            memory_dir, name, skills_dir=skills_dir, workspace_dir=old
        )
        assert decision["approved"] is True, decision
        return candidates

    monkeypatch.setattr(module, "_proposal_manifests", approve_after_listing)

    assert module.move_proposals(memory_dir, old_key, new_key, {}) == 1

    # Written from the listing, this would be the pending manifest without Undo.
    manifest = _manifest(memory_dir, name)
    assert manifest["status"] == "approved"
    assert manifest["workspace_dir"] == new_key
    assert manifest["approved_skill_path"] == str(new / "skills" / name)


def test_a_move_leaves_a_draft_reached_through_a_link_alone(tmp_path, monkeypatch):
    """A folder that resolves outside the proposals root is not served.

    Its lock would follow the link, so neither a move nor the upgrade may
    take one for it: the lock file would be created outside the proposals
    area. The project only such a draft was made under is left alone as well,
    since the draft that names it stays where it is.
    """
    old, new = tmp_path / "old", tmp_path / "new"
    new.mkdir()
    memory_dir, skills_dir = old / "memories", old / "skills"
    name = "focused-validation"
    _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    old_key = _manifest(memory_dir, name)["workspace_dir"]
    new_key = module._normalize_workspace_dir(new)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = autoskill_proposals_dir(memory_dir) / "linked"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(outside), str(linked))
    else:
        linked.symlink_to(outside, target_is_directory=True)
    unserved = json.dumps(
        {
            **_manifest(memory_dir, name),
            "proposal_id": "linked",
            "project_id": "linked-project",
        }
    )
    (outside / "manifest.json").write_text(unserved, encoding="utf-8")
    locked = []
    real_lock = module._draft_lock

    def observed_lock(proposal_dir):
        locked.append(Path(proposal_dir).name)
        return real_lock(proposal_dir)

    monkeypatch.setattr(module, "_draft_lock", observed_lock)

    # The linked draft is the only one that names a project.
    assert module.proposal_project_ids(memory_dir, old_key) == set()

    assert module.move_proposals(memory_dir, old_key, new_key, {}) == 1
    assert module.upgrade_proposal_workspaces(memory_dir) == 0

    assert module.proposal_project_ids(memory_dir, old_key) == set()
    assert locked == [name, name]
    assert (outside / "manifest.json").read_text(encoding="utf-8") == unserved


def test_rejection_sticks_to_the_draft_selected_before_locking(tmp_path, monkeypatch):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    prefix, selected = "paper-review", "paper-review-v2"
    selected_dir = _write_draft(memory_dir, selected)
    _submit(memory_dir, skills_dir, selected, "cluster-selected")
    real_lock = module._draft_lock

    def add_exact_then_lock(proposal_dir):
        if proposal_dir == selected_dir:
            _write_draft(memory_dir, prefix)
            _submit(memory_dir, skills_dir, prefix, "cluster-exact")
        return real_lock(proposal_dir)

    monkeypatch.setattr(module, "_draft_lock", add_exact_then_lock)
    result = reject_skill_proposal(memory_dir, prefix)

    assert result["rejected"] is True, result
    assert result["proposal_id"] == selected
    assert _manifest(memory_dir, selected)["status"] == "rejected"
    assert _manifest(memory_dir, prefix)["status"] == "pending"
