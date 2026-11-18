"""The two blocking findings from the twelfth review of PR #853.

The review record has to describe the body that was shown, and it has to be
written only once that body reached the deciding surface. Both fail on
``a8025a6``: the record compared the displayed text with one read of the
draft and then digested the draft with a second read, so a writer landing
between the two reads produced a review of a body nobody was shown; and the
record was written before the body was handed to the screen.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist import paths
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    reviewed_decision_digest,
    submit_autoskill_proposal,
)


def _draft(memory_dir: Path, name: str, body: str) -> Path:
    folder = autoskill_proposals_dir(memory_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review this draft.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    (folder / "notes.txt").write_text(f"note for {name}\n", encoding="utf-8")
    return folder


def _write_body(folder: Path, name: str, body: str) -> None:
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review this draft.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _submit(
    memory_dir: Path,
    skills_dir: Path,
    name: str = "probe-skill",
    *,
    workspace_dir: Path | None = None,
) -> None:
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name=name,
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="One observation is enough.",
        skills_dir=skills_dir,
        workspace_dir=workspace_dir or skills_dir.parent,
    )
    assert result["submitted"] is True, result


def _manifest(memory_dir: Path, name: str) -> dict:
    path = autoskill_proposals_dir(memory_dir) / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare(tmp_path, monkeypatch, body: str = "OLD BODY"):
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", body)
    _submit(memory_dir, skills_dir)
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    return memory_dir, skills_dir, folder, _manifest(memory_dir, "probe-skill")


def test_the_installed_body_is_the_body_that_was_shown(tmp_path, monkeypatch):
    """A writer that lands mid-record must not decide for the person.

    The record compared the displayed text with one read of the draft and
    then digested the draft with a second read. A writer landing between
    those two reads produced a review of a body nobody was shown, and the
    approval installed that body.
    """
    from EvoScientist.memory.autoskills import proposals

    memory_dir, skills_dir, folder, manifest = _prepare(tmp_path, monkeypatch)

    # What the deciding surface shows.
    shown = proposals.rendered_files(folder)

    # A writer lands between the read that compares the text and the read
    # that digests it. Before the fix those were two reads.
    real_digest = proposals._draft_digest

    def digest_after_a_write(pdir: Path) -> str:
        _write_body(pdir, "probe-skill", "NEW BODY")
        return real_digest(pdir)

    monkeypatch.setattr(proposals, "_draft_digest", digest_after_a_write)
    proposals.record_rendered_proposal(folder, manifest, shown)
    monkeypatch.setattr(proposals, "_draft_digest", real_digest)

    result = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["approved"] is True, result
    installed = (skills_dir / "probe-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert "OLD BODY" in installed, installed
    assert "NEW BODY" not in installed, installed


def test_a_draft_changed_after_it_was_shown_is_not_reviewed(tmp_path, monkeypatch):
    """A writer that lands before the record leaves the draft unarmed."""
    from EvoScientist.memory.autoskills import proposals

    memory_dir, skills_dir, folder, manifest = _prepare(tmp_path, monkeypatch)

    shown = proposals.rendered_files(folder)
    _write_body(folder, "probe-skill", "NEW BODY")

    assert proposals.record_rendered_proposal(folder, manifest, shown) is None
    result = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["approved"] is False, result
    assert not (skills_dir / "probe-skill").exists()


def test_a_mount_that_raised_leaves_the_draft_unarmed(tmp_path, monkeypatch):
    """A body that never reached the screen must not arm the decision.

    The command recorded the review while it built the body, before the
    mount. A screen that refused the body still left the draft armed, and a
    later approval in that process installed a body nobody was shown.
    """
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand

    memory_dir, skills_dir, _, _ = _prepare(tmp_path, monkeypatch, body="Shown body.")
    messages: list[str] = []
    mounted: list = []

    def mount(item, **kwargs):
        # The command mounts the proposals table first and the draft bodies
        # second. Only the body mount is refused: the table carries no
        # content, so refusing it would test nothing.
        mounted.append(item)
        if len(mounted) > 1:
            raise RuntimeError("the screen refused the body")

    ctx = SimpleNamespace(
        workspace=SimpleNamespace(root=tmp_path, skills_dir=skills_dir),
        ui=SimpleNamespace(
            append_system=lambda *args, **kwargs: messages.append(str(args[0])),
            mount_renderable=mount,
        ),
    )

    with pytest.raises(RuntimeError):
        asyncio.run(AutoSkillsCommand().execute(ctx, ["review"]))

    assert len(mounted) == 2, "the table was mounted and the body was refused"

    result = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=skills_dir.parent,
    )
    assert result["approved"] is False, result
    assert not (skills_dir / "probe-skill").exists()


def test_the_webui_arms_a_draft_only_after_the_response_left(tmp_path, monkeypatch):
    """The list route hands the review records to the caller, unrecorded.

    The payload is the text the page shows. The record is written once that
    payload has been delivered, so a request that never delivers it leaves
    the draft unarmed.
    """
    from EvoScientist.langgraph_dev import autoskills_api, registry

    _memory_dir, skills_dir, folder, manifest = _prepare(tmp_path, monkeypatch)

    fresh = registry.Registry()
    fresh._loaded()[paths.Workspace(skills_dir.parent).key] = True
    monkeypatch.setattr(registry, "REGISTRY", fresh)

    records: list = []
    code, payload = autoskills_api.list_drafts(None, records)
    assert code == 200
    assert payload["proposals"], payload
    assert records, "the payload carries a draft that has to be recorded"
    # The response is still on its way out: nothing is armed yet.
    assert reviewed_decision_digest(folder, manifest["updated_at"]) is None

    autoskills_api.record_shown_drafts(records)
    assert reviewed_decision_digest(folder, manifest["updated_at"]) is not None


def test_the_list_body_returns_the_records_unrecorded(tmp_path, monkeypatch):
    """``list_drafts_body`` hands back the records without writing them."""
    from EvoScientist.langgraph_dev import autoskills_api, registry

    _memory_dir, skills_dir, folder, manifest = _prepare(tmp_path, monkeypatch)

    fresh = registry.Registry()
    fresh._loaded()[paths.Workspace(skills_dir.parent).key] = True
    monkeypatch.setattr(registry, "REGISTRY", fresh)

    code, body, records = autoskills_api.list_drafts_body()
    assert code == 200
    assert body
    assert records
    assert reviewed_decision_digest(folder, manifest["updated_at"]) is None
