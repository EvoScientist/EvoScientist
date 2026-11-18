"""Regression tests for two defects found in review of #785.

1. A file swapped for a FIFO between the regular-file check and the read made
   the digest read block forever, while the proposal lock was held.
2. A revert compared the installed skill against the LIVE DRAFT instead of
   what the approval actually installed, so an edit that happened to match the
   changed draft was deleted as if the approval had never run.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    revert_skill_proposal,
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


def _write_skill_folder(
    memory_dir: Path,
    skill_name: str,
    description: str,
    body: str,
) -> None:
    folder = autoskill_proposals_dir(memory_dir) / skill_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _submit(memory_dir: Path, skills_dir: Path, body: str = "Draft body.") -> None:
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        body,
    )
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1"],
        rationale="One observation describes the practice.",
        skills_dir=skills_dir,
        workspace_dir=Path(skills_dir).parent,
    )
    assert result["submitted"] is True


def _approve(memory_dir: Path, skills_dir: Path) -> Path:
    record_reviewed_proposal_for(memory_dir, "focused-validation")
    result = approve_skill_proposal(
        memory_dir,
        "focused-validation",
        skills_dir=skills_dir,
        workspace_dir=Path(skills_dir).parent,
        approved_by="user",
    )
    assert result.get("approved") is True, result
    return skills_dir / "focused-validation"


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="FIFO creation is not available on this platform"
)
def test_a_draft_file_swapped_for_a_fifo_never_blocks_the_digest(tmp_path):
    """A FIFO in place of a draft file must not hang the decision path.

    The digest read happens while the proposal lock is held, so a blocking
    read there would stall every other process waiting on the same draft.
    """
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit(memory_dir, skills_dir)

    proposal_dir = autoskill_proposals_dir(memory_dir) / "focused-validation"
    target = proposal_dir / "SKILL.md"
    target.unlink()
    os.mkfifo(target)

    # The FIFO has no writer: a blocking read would hang forever. Run the
    # decision in a thread and require it to finish.
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            record_reviewed_proposal_for(memory_dir, "focused-validation")
            outcome["result"] = approve_skill_proposal(
                memory_dir,
                "focused-validation",
                skills_dir=skills_dir,
                workspace_dir=Path(skills_dir).parent,
                approved_by="user",
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=20)

    assert not worker.is_alive(), (
        "the decision blocked on a FIFO instead of refusing the non-regular file"
    )
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"].get("approved") is not True, outcome["result"]


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="FIFO creation is not available on this platform"
)
def test_a_fifo_replaced_after_the_check_never_blocks_the_digest(tmp_path):
    """The FIFO can appear AFTER the regular-file check has passed.

    The file can be replaced between the check and the read. The digest read
    must then refuse it -- neither block on it nor read it as an empty file,
    because a digest over zero files would pass the "content changed" check
    for a skill that is missing that file.
    """
    from EvoScientist.memory.autoskills import proposals as proposals_module

    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit(memory_dir, skills_dir)

    proposal_dir = autoskill_proposals_dir(memory_dir) / "focused-validation"
    target = proposal_dir / "SKILL.md"
    original = target.read_bytes()

    # The exact window CodeRabbit reported: the file is listed as a regular
    # file, and is replaced with a FIFO AFTER that listing and BEFORE the
    # read. Without the protection the read opens the FIFO in blocking mode
    # and waits for a writer that never comes.
    real_copied_files = proposals_module.copied_files

    def listing_then_swap(directory):
        files = real_copied_files(directory)
        target.unlink()
        os.mkfifo(target)
        return files

    proposals_module.copied_files = listing_then_swap
    try:
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["result"] = proposals_module._draft_digest(proposal_dir)
            except BaseException as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=20)
    finally:
        proposals_module.copied_files = real_copied_files

    assert not worker.is_alive(), "the digest read blocked on a FIFO"
    assert "error" in outcome, "a FIFO should be refused, not read"
    assert isinstance(outcome["error"], OSError)
    assert original  # the submitted content existed before the swap


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="FIFO creation is not available on this platform"
)
def test_a_fifo_in_the_draft_is_skipped_and_never_blocks(tmp_path):
    """A FIFO present before the decision is skipped, and never blocks.

    The upstream contract is that an entry that is not a regular file is
    skipped, not refused: it is not a skill file the approval installs.
    """
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit(memory_dir, skills_dir)

    proposal_dir = autoskill_proposals_dir(memory_dir) / "focused-validation"
    os.mkfifo(proposal_dir / "fifo")

    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            record_reviewed_proposal_for(memory_dir, "focused-validation")
            outcome["result"] = approve_skill_proposal(
                memory_dir,
                "focused-validation",
                skills_dir=skills_dir,
                workspace_dir=Path(skills_dir).parent,
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=20)

    assert not worker.is_alive(), "the decision blocked on a FIFO"
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"].get("approved") is True, outcome["result"]
    installed = skills_dir / "focused-validation"
    assert sorted(p.name for p in installed.iterdir()) == ["SKILL.md"]


def test_the_digest_read_is_byte_exact_on_crlf_files(tmp_path):
    """A draft file with CRLF must digest to its own bytes.

    ``os.open`` defaults to TEXT mode on Windows, which rewrites CRLF as LF
    and changes every digest. A digest that differs from the file's own bytes
    makes every approval read as "the content changed since submission".
    The read is hashed as a stream, so the length is taken from the opened
    descriptor and the bytes follow it one chunk at a time.
    """
    from EvoScientist.memory.autoskills.proposals import _draft_digest

    memory_dir = tmp_path / "memories"
    proposal_dir = autoskill_proposals_dir(memory_dir) / "focused-validation"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    raw = b"line one\r\nline two\r\n\x00\xff binary tail"
    (proposal_dir / "SKILL.md").write_bytes(raw)

    digest = _draft_digest(proposal_dir)

    import hashlib

    expected = hashlib.sha256()
    expected.update(len(b"SKILL.md").to_bytes(8, "big"))
    expected.update(b"SKILL.md")
    expected.update(len(raw).to_bytes(8, "big"))
    file_digest = hashlib.sha256(raw)
    expected.update(file_digest.digest())
    assert digest == expected.hexdigest(), "the digest is not over the file's own bytes"


def test_a_revert_compares_the_installed_skill_with_what_the_approval_installed(
    tmp_path,
):
    """A revert must not delete an edit that matches a CHANGED draft.

    The approval installs a digest-verified snapshot of the submitted content.
    If the draft changes afterwards, comparing the installed skill against the
    live draft is comparing against the wrong thing: an edit that happens to
    match the new draft looks "unchanged" and gets deleted.
    """
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit(memory_dir, skills_dir, body="Submitted body.")
    installed = _approve(memory_dir, skills_dir)

    installed_skill = installed / "SKILL.md"
    assert installed_skill.is_file()

    # The draft changes after the approval.
    proposal_dir = autoskill_proposals_dir(memory_dir) / "focused-validation"
    (proposal_dir / "SKILL.md").write_text(
        "---\nname: focused-validation\ndescription: changed\n---\n\nChanged body.\n",
        encoding="utf-8",
    )

    # The user edits the INSTALLED skill to match the changed draft.
    installed_skill.write_text(
        "---\nname: focused-validation\ndescription: changed\n---\n\nChanged body.\n",
        encoding="utf-8",
    )

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is False, (
        "a revert deleted an edit that was not the approved content"
    )
    assert result["error"] == "skill_changed"
    assert installed_skill.is_file(), "the user's edit was deleted"


def test_a_revert_still_removes_an_untouched_installed_skill(tmp_path):
    """The normal case must keep working: an untouched install is removed."""
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit(memory_dir, skills_dir, body="Submitted body.")
    installed = _approve(memory_dir, skills_dir)

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is True, result
    assert not installed.exists()


def test_a_revert_refuses_a_skill_edited_after_the_approval(tmp_path):
    """An edit that does NOT match the draft is still refused."""
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    _submit(memory_dir, skills_dir, body="Submitted body.")
    installed = _approve(memory_dir, skills_dir)

    (installed / "SKILL.md").write_text("user edit\n", encoding="utf-8")

    result = revert_skill_proposal(memory_dir, "focused-validation")

    assert result["reverted"] is False
    assert result["error"] == "skill_changed"
    assert (installed / "SKILL.md").read_text(encoding="utf-8") == "user edit\n"


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    sys.exit(pytest.main([__file__, "-v"]))
