"""The round-15 finding: the body a decision reviewed is the body installed.

The approve path read the draft twice -- once inside the review check, once
afterwards to get the digest the install is verified against. A writer that
landed between the two reads produced a digest of the NEW body, the snapshot
matched it, and the NEW body was installed: content nobody was shown.

The lock does not close that hole: the agent writes the draft with its shell,
and the shell does not take the lock. Only one read can close it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from EvoScientist import paths
from EvoScientist.memory.autoskills import proposals


def _draft(memory_dir: Path, name: str, body: str) -> Path:
    folder = proposals.autoskill_proposals_dir(memory_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review this draft.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    (folder / "notes.txt").write_text(f"note for {name}\n", encoding="utf-8")
    return folder


def _submit(memory_dir: Path, skills_dir: Path, name: str = "probe-skill") -> None:
    result = proposals.submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="One observation is enough.",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["submitted"] is True, result


def _rewrite_body(folder: Path, name: str, body: str) -> None:
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review this draft.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_a_write_after_the_review_is_not_installed(tmp_path, monkeypatch):
    """A writer that lands after the review cannot have its body installed.

    The review compared the body the person saw with the draft on disk. The
    install has to be verified against that same body, so the copy of a
    changed draft is refused instead of installed.
    """
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "OLD BODY")
    _submit(memory_dir, skills_dir)
    folder = proposals.autoskill_proposals_dir(memory_dir) / "probe-skill"

    # Stand in for the display: the record is of the body that was shown.
    manifest = proposals._read_manifest(folder / "manifest.json")
    assert manifest is not None
    proposals.record_rendered_proposal(
        folder, manifest, proposals.rendered_files(folder)
    )

    # A writer lands right after the review check has read the body.
    original_check = proposals._check_human_review

    def check_then_write(*args, **kwargs):
        refused = original_check(*args, **kwargs)
        if refused is None:
            _rewrite_body(folder, "probe-skill", "NEW BODY")
        return refused

    monkeypatch.setattr(proposals, "_check_human_review", check_then_write)

    result = proposals.approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )

    assert result["approved"] is False, result
    assert not (skills_dir / "probe-skill").exists(), "the changed body was installed"
    installed = skills_dir / "probe-skill" / "SKILL.md"
    assert not installed.exists() or "NEW BODY" not in installed.read_text(
        encoding="utf-8"
    )


def test_the_reviewed_body_is_the_one_installed(tmp_path, monkeypatch):
    """Without a writer, the body that was shown is the body installed."""
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "SHOWN BODY")
    _submit(memory_dir, skills_dir)
    folder = proposals.autoskill_proposals_dir(memory_dir) / "probe-skill"

    manifest = proposals._read_manifest(folder / "manifest.json")
    assert manifest is not None
    proposals.record_rendered_proposal(
        folder, manifest, proposals.rendered_files(folder)
    )

    result = proposals.approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )

    assert result["approved"] is True, result
    installed = (skills_dir / "probe-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert "SHOWN BODY" in installed, installed


@pytest.mark.parametrize("count", [1])
def test_the_body_is_read_once_by_the_approve_path(tmp_path, monkeypatch, count):
    """The approve path digests the draft once, not once per stage."""
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "BODY")
    _submit(memory_dir, skills_dir)
    folder = proposals.autoskill_proposals_dir(memory_dir) / "probe-skill"

    manifest = proposals._read_manifest(folder / "manifest.json")
    assert manifest is not None
    proposals.record_rendered_proposal(
        folder, manifest, proposals.rendered_files(folder)
    )

    reads: list[str] = []
    real_digest = proposals._draft_digest

    def counting_digest(pdir):
        reads.append(str(pdir))
        return real_digest(pdir)

    monkeypatch.setattr(proposals, "_draft_digest", counting_digest)

    result = proposals.approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )

    assert result["approved"] is True, result
    # One read of the live draft. The other digest is of the snapshot, which
    # is a different directory.
    live = [path for path in reads if path == str(folder)]
    assert len(live) == count, (live, reads)
