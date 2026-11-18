"""The three blocking findings from the eleventh review of PR #853.

Each test is the reviewer's reproduction. They fail on ``23f62cd``, where a
decision records a review without rendering the draft, where the review
digest does not pin the workspace, and where the per-draft lock is a file
the agent shell can unlink.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from EvoScientist import paths
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    list_skill_proposals,
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


def _ctx(root: Path, skills_dir: Path, messages: list, rendered: list):
    return SimpleNamespace(
        workspace=SimpleNamespace(root=root, skills_dir=skills_dir),
        ui=SimpleNamespace(
            append_system=lambda *args, **kwargs: messages.append(str(args[0])),
            mount_renderable=lambda item, **kwargs: rendered.append(item),
        ),
    )


def test_direct_approve_without_a_display_does_not_install(tmp_path, monkeypatch):
    """``/autoskills approve <id>`` with no prior display refuses the draft.

    The command used to hash the body inside the decision and call that a
    display. Nothing was rendered, and the skill was installed.
    """
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "Secret body the user never saw.")
    _submit(memory_dir, skills_dir)
    messages: list[str] = []
    rendered: list = []
    ctx = _ctx(tmp_path, skills_dir, messages, rendered)

    asyncio.run(AutoSkillsCommand()._approve(ctx, "probe-skill"))

    assert rendered == []
    assert not any("Secret body" in message for message in messages)
    assert not (skills_dir / "probe-skill").exists()
    assert any("not reviewed" in message.lower() for message in messages), messages


def test_review_renders_the_body_and_then_a_decision_installs_it(tmp_path, monkeypatch):
    """``/autoskills review`` shows the copied files, and that arms the decision."""
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "Shown body.")
    _submit(memory_dir, skills_dir)
    messages: list[str] = []
    rendered: list = []
    ctx = _ctx(tmp_path, skills_dir, messages, rendered)
    command = AutoSkillsCommand()

    asyncio.run(command.execute(ctx, ["review"]))
    from rich.console import Console

    shown_parts = []
    for item in rendered:
        buffer = __import__("io").StringIO()
        Console(file=buffer, force_terminal=False, width=120, color_system=None).print(
            item
        )
        shown_parts.append(buffer.getvalue())
    shown = "\n".join(shown_parts)
    assert "Shown body." in shown
    assert "notes.txt" in shown
    assert "note for probe-skill" in shown

    asyncio.run(command._approve(ctx, "probe-skill"))

    assert "Shown body." in (skills_dir / "probe-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_tui_count_notice_does_not_arm_a_decision(tmp_path, monkeypatch):
    """The startup notice shows a count, so it must not record a review."""
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "Unseen body.")
    _submit(memory_dir, skills_dir)
    # The notice is a nested method. Calling it unbound on a stand-in is the
    # startup path: it has the workspace and a place to print the count, and
    # nothing else.
    from EvoScientist.cli.tui_interactive import run_textual_interactive

    source = Path(run_textual_interactive.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    start = source.index("        def _append_pending_skill_proposals_notice")
    end = source.index("        def _render_status", start)
    method = textwrap.dedent(source[start:end])
    namespace: dict[str, object] = {}
    exec(method, run_textual_interactive.__globals__, namespace)
    notices: list[str] = []
    app = SimpleNamespace(
        _dirs=SimpleNamespace(workspace=SimpleNamespace(root=tmp_path)),
        _append_system=lambda text, **kwargs: notices.append(text),
    )
    namespace["_append_pending_skill_proposals_notice"](app)

    assert any("1 autoskill" in notice for notice in notices), notices
    assert "Unseen body." not in "\n".join(notices)
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    assert (
        reviewed_decision_digest(
            folder, _manifest(memory_dir, "probe-skill")["updated_at"]
        )
        is None
    )
    result = approve_skill_proposal(
        memory_dir, "probe-skill", skills_dir=skills_dir, workspace_dir=tmp_path
    )
    assert result["approved"] is False, result
    assert not (skills_dir / "probe-skill").exists()


def test_filtered_list_does_not_arm_a_hidden_pending_draft(tmp_path, monkeypatch):
    """A status filter that returns no pending draft records none either."""
    from EvoScientist.langgraph_dev import autoskills_api, registry

    memory_dir = tmp_path / "memories"
    workspace = paths.Workspace(tmp_path / "a")
    workspace.root.mkdir()
    skills_dir = workspace.skills_dir
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    monkeypatch.setattr(registry, "REGISTRY", registry.Registry())
    registry.REGISTRY.set_registered(workspace.key, True)
    _draft(memory_dir, "probe-skill", "Hidden pending body.")
    _submit(memory_dir, skills_dir, workspace_dir=workspace.root)

    code, payload = autoskills_api.list_drafts("approved")

    assert code == 200
    assert payload["proposals"] == []
    assert payload["counts"]["pending"] == 1
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    assert (
        reviewed_decision_digest(
            folder, _manifest(memory_dir, "probe-skill")["updated_at"]
        )
        is None
    )
    result = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=workspace.root,
    )
    assert result["approved"] is False, result


def test_reading_the_installed_file_does_not_arm_the_pending_draft(
    tmp_path, monkeypatch
):
    """Viewing the installed copy is not a review of the pending draft."""
    from EvoScientist.langgraph_dev import autoskills_api, registry
    from EvoScientist.memory.autoskills.proposals import record_reviewed_proposal_by_id

    memory_dir = tmp_path / "memories"
    workspace = paths.Workspace(tmp_path / "a")
    workspace.root.mkdir()
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    monkeypatch.setattr(registry, "REGISTRY", registry.Registry())
    registry.REGISTRY.set_registered(workspace.key, True)
    installed = workspace.skills_dir / "probe-skill"
    installed.mkdir(parents=True)
    (installed / "SKILL.md").write_text(
        "---\nname: probe-skill\ndescription: Review this draft.\n---\n\n"
        "Installed body.\n",
        encoding="utf-8",
    )
    _draft(memory_dir, "probe-skill", "Pending body the page did not return.")
    result = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="probe-skill",
        cluster_hash="cluster-original",
        source_observation_ids=["O-1"],
        rationale="One observation is enough.",
        operation="update",
        target_skill_name="probe-skill",
        skills_dir=workspace.skills_dir,
        workspace_dir=workspace.root,
    )
    assert result["submitted"] is True, result
    record_reviewed_proposal_by_id(memory_dir, "probe-skill")
    from EvoScientist.memory.autoskills import proposals as module

    module._reviews.clear()

    code, body = autoskills_api.read_draft_file(
        "probe-skill", "SKILL.md", installed=True
    )

    assert code == 200
    assert "Installed body." in body
    assert "Pending body" not in body
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    assert (
        reviewed_decision_digest(
            folder, _manifest(memory_dir, "probe-skill")["updated_at"]
        )
        is None
    )


def test_channel_approve_without_a_review_message_does_not_install(
    tmp_path, monkeypatch
):
    """A channel slash approval carries no draft body, so it is refused.

    The dispatcher sends the slash text to the same command object the CLI
    uses. The channel UI collects what that command renders; an approval
    that never rendered the body must not install it.
    """
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.manager import manager as cmd_manager

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    _draft(memory_dir, "probe-skill", "Channel body never sent.")
    _submit(memory_dir, skills_dir)
    delivered: list[object] = []

    class _UI:
        sent_to_channel = False

        def append_system(self, text, style="dim"):
            delivered.append(text)

        def mount_renderable(self, item):
            delivered.append(item)

        async def flush(self):
            return None

    ctx = CommandContext(
        agent=None,
        thread_id="thread",
        ui=_UI(),
        dirs=SimpleNamespace(
            workspace=SimpleNamespace(root=tmp_path, skills_dir=skills_dir)
        ),
    )

    async def run_command(text, context):
        return await cmd_manager.execute(text, context)

    handled = asyncio.run(run_command("/autoskills approve probe-skill", ctx))

    assert handled is True
    assert not any("Channel body never sent." in str(item) for item in delivered)
    assert not (skills_dir / "probe-skill").exists()
    assert any("not reviewed" in item.lower() for item in delivered), delivered

    delivered.clear()
    reviewed = asyncio.run(run_command("/autoskills review", ctx))
    assert reviewed is True
    from io import StringIO

    from rich.console import Console

    rendered_text = []
    for item in delivered:
        buffer = StringIO()
        Console(file=buffer, force_terminal=False, width=120, color_system=None).print(
            item
        )
        rendered_text.append(buffer.getvalue())
    assert any("Channel body never sent." in item for item in rendered_text), (
        rendered_text
    )
    asyncio.run(run_command("/autoskills approve probe-skill", ctx))
    assert "Channel body never sent." in (
        skills_dir / "probe-skill" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_workspace_swap_after_display_does_not_install_elsewhere(tmp_path, monkeypatch):
    """Changing only ``workspace_dir`` after the display refuses the decision.

    The WebUI used to read the mutated manifest and install into workspace B
    while the reviewer had seen workspace A.
    """
    from EvoScientist.langgraph_dev import autoskills_api, registry

    memory_dir = tmp_path / "memories"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    monkeypatch.setattr(registry, "REGISTRY", registry.Registry())
    workspace_a = paths.Workspace(tmp_path / "a")
    workspace_b = paths.Workspace(tmp_path / "b")
    workspace_a.root.mkdir()
    workspace_b.root.mkdir()
    registry.REGISTRY.set_registered(workspace_a.key, True)
    registry.REGISTRY.set_registered(workspace_b.key, True)
    _draft(memory_dir, "probe-skill", "Reviewed under A.")
    _submit(memory_dir, workspace_a.skills_dir, workspace_dir=workspace_a.root)
    code, payload = autoskills_api.list_drafts()
    assert code == 200
    assert payload["proposals"][0]["workspace_dir"] == workspace_a.key
    assert "Reviewed under A." in (
        payload["proposals"][0]["skill_md"]
        + "".join(payload["proposals"][0].get("contents", {}).values())
    )

    manifest_path = (
        autoskill_proposals_dir(memory_dir) / "probe-skill" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stamp = manifest["updated_at"]
    manifest["workspace_dir"] = workspace_b.key
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rewritten = json.loads(manifest_path.read_text(encoding="utf-8"))
    rewritten["updated_at"] = stamp
    manifest_path.write_text(json.dumps(rewritten), encoding="utf-8")

    status, result = autoskills_api.decide(
        "probe-skill",
        "approve",
        review_token=payload["proposals"][0]["review_token"],
    )

    assert status == 409, result
    assert result["error"] in {"draft_changed", "approve_failed"}
    assert not (workspace_a.skills_dir / "probe-skill").exists()
    assert not (workspace_b.skills_dir / "probe-skill").exists()


def test_list_leaves_out_a_draft_whose_manifest_names_an_outside_folder(tmp_path):
    """A manifest that does not live in its draft folder is not listed.

    On POSIX the manifest is a symlink to a file outside the proposals root,
    and the list must leave that draft out. Creating that link needs no
    privilege there. Windows skips link creation unless it is allowed, and
    the same assertion still runs: a manifest whose resolved path is outside
    the root is not a draft the list returns.
    """
    from EvoScientist.langgraph_dev import autoskills_api
    from EvoScientist.memory.autoskills import proposals as module

    memory_dir = tmp_path / "memories"
    workspace = paths.Workspace(tmp_path / "a")
    workspace.root.mkdir()
    _draft(memory_dir, "probe-skill", "Inside body.")
    _submit(memory_dir, workspace.skills_dir, workspace_dir=workspace.root)
    outside = tmp_path / "outside" / "manifest.json"
    outside.parent.mkdir()
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    manifest = folder / "manifest.json"
    manifest.replace(outside)
    linked = False
    try:
        manifest.symlink_to(outside)
        linked = True
    except OSError:
        manifest.write_bytes(outside.read_bytes())
        # No link privilege: point the resolver at the outside file for this
        # one manifest, which is what a symlink would make ``resolve`` return.
        real_resolve = Path.resolve

        def resolve_manifest_outside(self, *args, **kwargs):
            if self == manifest:
                return real_resolve(outside)
            return real_resolve(self, *args, **kwargs)

        Path.resolve = resolve_manifest_outside  # type: ignore[method-assign]
    try:
        proposals = module.list_skill_proposals(memory_dir)
    finally:
        if not linked:
            Path.resolve = real_resolve  # type: ignore[method-assign]
    assert proposals == []
    del autoskills_api


def test_per_draft_lock_has_no_unlinkable_pathname(tmp_path):
    """The lifecycle lock is not a file the shell can delete and replace."""
    from EvoScientist.memory.autoskills.proposals import _proposal_file_lock

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    proposals = autoskill_proposals_dir(memory_dir)
    before = {path for path in memory_dir.rglob("*") if path.is_file()}
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with _proposal_file_lock(proposals / "probe-skill"):
            entered.set()
            assert release.wait(5)

    worker = threading.Thread(target=hold)
    worker.start()
    assert entered.wait(5)
    created = {
        path for path in memory_dir.rglob("*") if path.is_file() and path not in before
    }
    release.set()
    worker.join(5)
    assert created == set()
    assert not list(memory_dir.rglob("*.lock"))


def test_per_draft_lock_still_serializes_two_threads(tmp_path):
    """A nameless lock still makes the second writer wait."""
    from EvoScientist.memory.autoskills.proposals import _proposal_file_lock

    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Reviewed body.")
    _submit(memory_dir, skills_dir)
    lock_path = proposals = autoskill_proposals_dir(memory_dir) / "probe-skill"
    order: list[str] = []
    release = threading.Event()

    def first() -> None:
        with _proposal_file_lock(lock_path):
            order.append("first-in")
            assert release.wait(5)
            order.append("first-out")

    def second() -> None:
        while "first-in" not in order:
            time.sleep(0.01)
        with _proposal_file_lock(proposals):
            order.append("second-in")

    one = threading.Thread(target=first)
    two = threading.Thread(target=second)
    one.start()
    two.start()
    while "first-in" not in order:
        time.sleep(0.01)
    time.sleep(0.2)
    assert "second-in" not in order
    release.set()
    one.join(5)
    two.join(5)
    assert order == ["first-in", "first-out", "second-in"]


def test_counting_a_draft_does_not_record_it(tmp_path):
    """A list used as a count, even with the old flag, is not a display."""
    memory_dir, skills_dir = tmp_path / "memories", tmp_path / "skills"
    _draft(memory_dir, "probe-skill", "Counted body.")
    _submit(memory_dir, skills_dir)
    listed = list_skill_proposals(
        memory_dir,
        status="pending",
        workspace_dir=skills_dir.parent,
        record_review=True,
    )
    assert [proposal.proposal_id for proposal in listed] == ["probe-skill"]
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    assert reviewed_decision_digest(folder, listed[0].updated_at) is None
