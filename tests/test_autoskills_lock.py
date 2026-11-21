from __future__ import annotations

import hashlib
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

    def move_then_lock(proposal_dir):
        nonlocal moved
        assert proposal_dir == draft
        if not moved:
            assert module.move_proposals(memory_dir, old_key, new_key, {}) == 1
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


def test_approvals_cannot_interleave_installs_of_two_draft_versions(
    tmp_path, monkeypatch
):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    installed = skills_dir / name
    installed.mkdir(parents=True)
    base = f"---\nname: {name}\ndescription: Use when testing updates.\n---\n\nBase.\n"
    skill_md = installed / "SKILL.md"
    skill_md.write_text(base, encoding="utf-8")

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

    submit("First version.", "cluster-first")
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    monkeypatch.setattr(module, "_DRAFT_LOCK_TIMEOUT", 0)
    real_copy = module._copy_proposed_skill
    competing = None
    interleaved = False

    def approve():
        return approve_skill_proposal(memory_dir, name, skills_dir=skills_dir)

    def copy_with_competing_approval(source, destination):
        nonlocal competing, interleaved
        if destination == installed and not interleaved:
            interleaved = True
            # Force the newer approval into the gap before the old snapshot writes.
            submit("Second version.", "cluster-second")
            competing = approve()
        return real_copy(source, destination)

    monkeypatch.setattr(module, "_copy_proposed_skill", copy_with_competing_approval)
    first = approve()
    assert first["approved"] is False, first
    assert "changed while it was being approved" in first["error"]
    assert competing is not None

    # Without serialization, the second approval commits before the first overwrites it.
    manifest = _manifest(memory_dir, name)
    if competing["approved"]:
        assert (
            manifest["approved_content_digests"]["SKILL.md"]
            == hashlib.sha256(skill_md.read_bytes()).hexdigest()
        )
    assert competing["approved"] is False, competing
    assert "waiting for another installer" in competing["error"]
    assert manifest["status"] == "pending"

    retry = approve()
    assert retry["approved"] is True, retry
    assert "Second version." in skill_md.read_text(encoding="utf-8")
    manifest = _manifest(memory_dir, name)
    assert manifest["status"] == "approved"
    assert manifest["cluster_hash"] == "cluster-second"
    assert (
        manifest["approved_content_digests"]["SKILL.md"]
        == hashlib.sha256(skill_md.read_bytes()).hexdigest()
    )


def test_approval_rolls_back_a_created_skill_on_commit_lock_timeout(
    tmp_path, monkeypatch
):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    name = "focused-validation"
    _write_draft(memory_dir, name)
    _submit(memory_dir, skills_dir, name, "cluster-original")
    installed = skills_dir / name
    real_lock = module._draft_lock
    timed_out = False
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)

    def lock(proposal_dir):
        nonlocal timed_out
        draft_lock = real_lock(proposal_dir)
        if (installed / "SKILL.md").is_file() and not timed_out:
            timed_out = True
            raise Timeout(draft_lock.lock_file)
        return draft_lock

    monkeypatch.setattr(module, "_draft_lock", lock)
    result = approve_skill_proposal(memory_dir, name, skills_dir=skills_dir)

    assert timed_out
    assert result["approved"] is False
    assert "Could not finish approval: timed out" in result["error"]
    assert _manifest(memory_dir, name)["status"] == "pending"
    assert not installed.exists()
    retry = approve_skill_proposal(memory_dir, name, skills_dir=skills_dir)
    assert retry["approved"] is True, retry


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
