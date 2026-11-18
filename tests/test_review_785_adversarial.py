import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    processed_cluster_hashes,
    reject_skill_proposal,
    submit_autoskill_proposal,
)


def record_reviewed_proposal_for(memory_dir, proposal_id):
    """Stand in for the display a person's decision requires.

    The production surfaces record a draft when they show it. A test that
    decides directly has shown the draft it is about to decide, so it records
    that current content. A body edited after this line is still refused.
    """
    from EvoScientist.memory.autoskills.proposals import record_reviewed_proposal_by_id

    record_reviewed_proposal_by_id(memory_dir, proposal_id)


def _write_draft(memory_dir, name, body):
    folder = autoskill_proposals_dir(memory_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review draft.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return folder


def _submit(memory_dir, skills_dir, name, *, cluster_hash="cluster-original"):
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash=cluster_hash,
        source_observation_ids=["O-1"],
        rationale="Original rationale.",
        skills_dir=skills_dir,
        workspace_dir=Path(skills_dir).parent,
    )
    assert result["submitted"] is True


def _wait_for(path, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


def _run(command, *args):
    return subprocess.Popen(
        [sys.executable, "-c", command, *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_review_two_processes_serialize_opposed_lifecycle_changes(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _write_draft(memory_dir, "focused-validation", "Original body.")
    _submit(memory_dir, skills_dir, "focused-validation")
    signals = tmp_path / "signals"
    signals.mkdir()
    approver = tmp_path / "approver.py"
    approver.write_text(
        "\n".join(
            [
                "import sys, time",
                "from pathlib import Path",
                "from EvoScientist.memory.autoskills.proposals import approve_skill_proposal, review_skill_proposal",
                "memory_dir, skills_dir, ready, release = map(Path, sys.argv[1:])",
                "module = sys.modules['EvoScientist.memory.autoskills.proposals']",
                "original = module._write_manifest",
                "def paused_write(path, payload):",
                "    ready.write_text('ready', encoding='utf-8')",
                "    while not release.exists():",
                "        time.sleep(.01)",
                "    return original(path, payload)",
                "module._write_manifest = paused_write",
                "module._notify_approved_skill = lambda: None",
                "review_skill_proposal(memory_dir, 'focused-validation')",
                "result = approve_skill_proposal(memory_dir, 'focused-validation', skills_dir=skills_dir)",
                "print(result['approved'])",
            ]
        ),
        encoding="utf-8",
    )
    rejecter = (
        "import sys\n"
        "from pathlib import Path\n"
        "from EvoScientist.memory.autoskills.proposals import reject_skill_proposal, review_skill_proposal\n"
        "memory_dir = Path(sys.argv[1])\n"
        "review_skill_proposal(memory_dir, 'focused-validation')\n"
        "print(reject_skill_proposal(memory_dir, 'focused-validation')['rejected'])"
    )
    ready = signals / "approver-ready"
    release = signals / "release"
    first = _run(
        approver.read_text(encoding="utf-8"), memory_dir, skills_dir, ready, release
    )
    try:
        _wait_for(ready)
        second = _run(rejecter, memory_dir)
        try:
            time.sleep(0.75)
            assert second.poll() is None
            release.write_text("release", encoding="utf-8")
            first_output, first_error = first.communicate(timeout=20)
            second_output, second_error = second.communicate(timeout=20)
        finally:
            if second.poll() is None:
                second.kill()
                second.wait()
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()
    assert first.returncode == 0, first_error
    assert second.returncode == 0, second_error
    assert first_output.strip() == "True"
    assert second_output.strip() == "False"


def test_review_different_proposals_do_not_share_one_lock(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    for name in ("alpha", "beta"):
        _write_draft(memory_dir, name, f"{name} body.")
        _submit(memory_dir, skills_dir, name, cluster_hash=f"cluster-{name}")
    signals = tmp_path / "signals"
    signals.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import sys, time",
                "from pathlib import Path",
                "from EvoScientist.memory.autoskills.proposals import reject_skill_proposal, review_skill_proposal",
                "memory_dir, name, ready, peer = map(Path, sys.argv[1:])",
                "module = sys.modules['EvoScientist.memory.autoskills.proposals']",
                "original = module._write_manifest",
                "def signal_write(path, payload):",
                "    ready.write_text('ready', encoding='utf-8')",
                "    while not peer.exists():",
                "        time.sleep(.01)",
                "    return original(path, payload)",
                "module._write_manifest = signal_write",
                "review_skill_proposal(memory_dir, name.name)",
                "result = reject_skill_proposal(memory_dir, name.name)",
                "print(result['rejected'])",
            ]
        ),
        encoding="utf-8",
    )
    alpha_ready = signals / "alpha"
    beta_ready = signals / "beta"
    first = _run(
        worker.read_text(encoding="utf-8"), memory_dir, "alpha", alpha_ready, beta_ready
    )
    try:
        _wait_for(alpha_ready)
        second = _run(
            worker.read_text(encoding="utf-8"),
            memory_dir,
            "beta",
            beta_ready,
            beta_ready,
        )
        try:
            _wait_for(beta_ready)
            first_output, first_error = first.communicate(timeout=20)
            second_output, second_error = second.communicate(timeout=20)
        finally:
            if second.poll() is None:
                second.kill()
                second.wait()
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()
    assert first.returncode == 0, first_error
    assert second.returncode == 0, second_error
    assert first_output.strip() == second_output.strip() == "True"


def test_review_leftover_lock_file_does_not_block_a_live_process(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _write_draft(memory_dir, "focused-validation", "Original body.")
    _submit(memory_dir, skills_dir, "focused-validation")
    # A lock file left from the old pathname must not be what a decision
    # waits on: the lock is a kernel object, so a file here is inert.
    lock = memory_dir / "autoskills" / "locks" / "focused-validation.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b"left from a dead process")
    record_reviewed_proposal_for(memory_dir, "focused-validation")
    result = reject_skill_proposal(memory_dir, "focused-validation")
    assert result["rejected"] is True


def test_review_stale_cli_style_decision_after_resubmission_is_refused(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    folder = _write_draft(memory_dir, "focused-validation", "Old body.")
    _submit(memory_dir, skills_dir, "focused-validation")
    (folder / "SKILL.md").write_text(
        folder.joinpath("SKILL.md")
        .read_text(encoding="utf-8")
        .replace("Old body.", "New body."),
        encoding="utf-8",
    )
    _submit(memory_dir, skills_dir, "focused-validation", cluster_hash="cluster-new")
    # The version the user reviewed is the one before the resubmission.
    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "focused-validation" / "manifest.json"
    )
    reviewed_updated_at = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "updated_at"
    ]
    _submit(memory_dir, skills_dir, "focused-validation", cluster_hash="cluster-newer")
    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        record_reviewed_proposal_for(memory_dir, "focused-validation")
        result = approve_skill_proposal(
            memory_dir,
            "focused-validation",
            skills_dir=skills_dir,
            workspace_dir=Path(skills_dir).parent,
            expected_updated_at=reviewed_updated_at,
        )
    assert result["approved"] is False
    assert result["error"] == "Proposal changed since it was read"


def test_review_a_draft_with_no_seal_is_approvable_once_it_was_shown(
    tmp_path,
):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _write_draft(memory_dir, "focused-validation", "Old body.")
    _submit(memory_dir, skills_dir, "focused-validation")
    # The seal is a race record, not the review. Deleting it leaves a draft
    # a person can still approve, because the approval compares the draft
    # with what this process displayed. The body is left untouched.
    seal = memory_dir / "autoskills" / "seals" / "focused-validation.json"
    seal.unlink()
    record_reviewed_proposal_for(memory_dir, "focused-validation")
    result = approve_skill_proposal(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )
    assert result["approved"] is True, result


def test_review_rationale_is_a_digest_exclusion_but_decision_fields_are_not(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    folder = _write_draft(memory_dir, "focused-validation", "Old body.")
    _submit(memory_dir, skills_dir, "focused-validation")
    record_reviewed_proposal_for(memory_dir, "focused-validation")
    (folder / "RATIONALE.md").write_text("Changed rationale.\n", encoding="utf-8")
    (folder / ".ordinary.tmp").write_text("Excluded bytes\n", encoding="utf-8")
    # Rationale and scratch files are not installed, so they are not part of
    # the review. The decision fields are: an edit of cluster_hash in the
    # manifest, made after the draft was shown, must not be what the approval
    # marks processed.
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cluster_hash"] = "cluster-manifest-edit"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        result = approve_skill_proposal(
            memory_dir, "focused-validation", skills_dir=skills_dir
        )
    assert result["approved"] is False, result
    assert "shown" in result["error"]
    assert "cluster-manifest-edit" not in processed_cluster_hashes(memory_dir)
    assert "cluster-original" not in processed_cluster_hashes(memory_dir)

    manifest["cluster_hash"] = "cluster-original"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with patch("EvoScientist.memory.autoskills.proposals._notify_approved_skill"):
        record_reviewed_proposal_for(memory_dir, "focused-validation")
        restored = approve_skill_proposal(
            memory_dir, "focused-validation", skills_dir=skills_dir
        )
    assert restored["approved"] is True, restored
    assert "cluster-original" in processed_cluster_hashes(memory_dir)


def test_review_draft_can_change_after_snapshot_verification(tmp_path, monkeypatch):
    from EvoScientist.memory.autoskills import proposals as module

    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    folder = _write_draft(memory_dir, "focused-validation", "Old body.")
    original = folder / "SKILL.md"
    _submit(memory_dir, skills_dir, "focused-validation")
    real_copy = module._copy_proposed_skill

    def change_after_verified_snapshot(source, destination):
        blocked = real_copy(source, destination)
        original.write_text(
            original.read_text(encoding="utf-8").replace("Old body.", "Racing body."),
            encoding="utf-8",
        )
        return blocked

    monkeypatch.setattr(module, "_copy_proposed_skill", change_after_verified_snapshot)
    monkeypatch.setattr(module, "_notify_approved_skill", lambda: None)
    record_reviewed_proposal_for(memory_dir, "focused-validation")
    result = approve_skill_proposal(
        memory_dir, "focused-validation", skills_dir=skills_dir
    )
    assert result["approved"] is True
    assert "Racing body." in original.read_text(encoding="utf-8")
    assert "Old body." in (skills_dir / "focused-validation" / "SKILL.md").read_text(
        encoding="utf-8"
    )
