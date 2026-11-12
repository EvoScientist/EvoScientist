"""Scheduled tasks belong to a workspace: they run on its assistant, and are
listed and managed within it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.cron import schedule as crons
from EvoScientist.langgraph_dev.assistants import workspace_assistant_id
from EvoScientist.paths import SessionDirs, Workspace


def _cron(cron_id: str, workspace_dir: str | None = None, **metadata) -> dict:
    """A scheduled task, on *workspace_dir*'s scheduler assistant if given."""
    meta = {"run_kind": crons.SCHEDULED_RUN_KIND, "name": cron_id, **metadata}
    cron = {"cron_id": cron_id, "schedule": "0 7 * * *", "metadata": meta}
    if workspace_dir is not None:
        meta["workspace_dir"] = workspace_dir
        cron["assistant_id"] = workspace_assistant_id("scheduler", workspace_dir)
    return cron


@pytest.fixture
def other(tmp_path) -> Workspace:
    """Another live workspace on this machine (its folder exists)."""
    ws = Workspace(tmp_path / "other")
    ws.root.mkdir(parents=True)
    return ws


@pytest.fixture
def client(workspace, other, monkeypatch) -> MagicMock:
    """A server with one task of each workspace and one on no workspace's
    assistant."""
    fake = MagicMock()
    fake.crons.search.return_value = [
        _cron("mine", workspace.key),
        _cron("theirs", other.key),
        _cron("untagged"),
    ]
    monkeypatch.setattr(crons, "_client", lambda: fake)
    monkeypatch.setattr(crons, "_default_timezone", lambda: "UTC")
    return fake


def test_belongs_to_matches_workspace_assistant(workspace, other):
    assert crons.belongs_to(_cron("a", workspace.key), workspace)
    assert not crons.belongs_to(_cron("a", other.key), workspace)
    # The tag alone is for display: a cron on no workspace assistant is no one's.
    untagged = _cron("a")
    untagged["metadata"]["workspace_dir"] = workspace.key
    assert not crons.belongs_to(untagged, workspace)


def test_list_schedules_filters_by_workspace(workspace, client):
    ids = [row["cron_id"] for row in crons.list_schedules(workspace=workspace)]
    assert ids == ["mine"]


def test_list_schedules_reads_every_page(workspace, other, client):
    """Workspaces are filtered after the search, so every page is read."""
    rows = [_cron(f"theirs-{i}", other.key) for i in range(1500)]
    rows.append(_cron("mine", workspace.key))
    client.crons.search.side_effect = lambda *, metadata, limit, offset: rows[
        offset : offset + limit
    ]

    ids = [row["cron_id"] for row in crons.list_schedules(workspace=workspace)]
    assert ids == ["mine"]


def test_list_schedules_dedupes_shifted_pages(workspace, client):
    """Pages are newest first; a cron created between two fetches shifts the
    rest down by one, so the row at the page boundary comes back twice."""
    rows = [_cron(f"c{i}", workspace.key) for i in range(1200)]
    fetches = 0

    def _search(*, metadata, limit, offset):
        nonlocal fetches
        fetches += 1
        if fetches == 2:
            rows.insert(0, _cron("newest", workspace.key))
        return rows[offset : offset + limit]

    client.crons.search.side_effect = _search

    ids = [row["cron_id"] for row in crons.list_schedules(workspace=workspace)]
    assert len(ids) == len(set(ids)) == 1200
    assert ids[0] == "c0"


def test_delete_and_set_enabled_refuse_other_workspace(workspace, other, client):
    theirs = _cron("theirs", other.key)
    with pytest.raises(LookupError):
        crons.delete_schedule(theirs, workspace=workspace)
    with pytest.raises(LookupError):
        crons.set_enabled(theirs, False, workspace=workspace)
    client.crons.delete.assert_not_called()
    client.crons.update.assert_not_called()


def test_create_and_run_now_tag_workspace(workspace, client):
    client.threads.create.return_value = {"thread_id": "t1"}
    crons.create_schedule(
        name="digest", schedule="0 7 * * *", prompt="p", workspace=workspace
    )
    crons.run_now("p", workspace=workspace)

    for call in (client.crons.create.call_args, client.runs.create.call_args):
        assert call.kwargs["metadata"]["workspace_dir"] == workspace.key
        assert call.kwargs["config"] == {
            "configurable": {"workspace_dir": workspace.key}
        }
    thread_metadata = client.threads.create.call_args.kwargs["metadata"]
    assert thread_metadata["workspace_dir"] == workspace.key


def test_scheduling_tools_scoped_to_workspace(workspace, client, monkeypatch):
    from EvoScientist.middleware.scheduler import make_scheduling_tools

    monkeypatch.setattr(crons, "is_available", lambda: True)
    tools = {t.name: t for t in make_scheduling_tools(workspace)}

    tools["schedule_task"].invoke({"name": "n", "cron": "0 7 * * *", "prompt": "p"})
    created = client.crons.create.call_args.kwargs["metadata"]
    assert created["workspace_dir"] == workspace.key

    listed = tools["list_scheduled_tasks"].invoke({})
    assert "mine" in listed
    assert "theirs" not in listed

    refused = tools["cancel_scheduled_task"].invoke({"cron_id": "theirs"})
    assert "No scheduled task" in refused
    client.crons.delete.assert_not_called()


async def test_schedule_command_scoped_to_workspace(workspace, client, monkeypatch):
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    monkeypatch.setattr(crons, "is_available", lambda: True)
    ui = MagicMock()
    ctx = CommandContext(dirs=SessionDirs(workspace), agent=None, thread_id="t", ui=ui)

    await ScheduleCommand().execute(ctx, ["add", "0", "7", "*", "*", "*", "p"])
    created = client.crons.create.call_args.kwargs["metadata"]
    assert created["workspace_dir"] == workspace.key

    await ScheduleCommand().execute(ctx, ["remove", "theirs"])
    await ScheduleCommand().execute(ctx, ["pause", "theirs"])
    client.crons.delete.assert_not_called()
    client.crons.update.assert_not_called()
    # Another workspace's id never resolves: the user sees a plain miss.
    messages = [c.args[0] for c in ui.append_system.call_args_list]
    assert messages.count("No schedule matching theirs.") == 2

    await ScheduleCommand().execute(ctx, ["pause", "mine"])
    client.crons.update.assert_called_once_with("mine", enabled=False)


# ---------------------------------------------------------------------------
# AutoSkills keeps one cron per workspace
# ---------------------------------------------------------------------------


def _autoskill_cron(
    cron_id: str,
    workspace_dir: str,
    *,
    forwarded: bool,
    assistant_id: str | None = None,
    step_limit: bool = True,
) -> dict:
    """An AutoSkills cron as the server returns it; by default on its
    workspace's assistant, its runs limited to the memory step limit."""
    from EvoScientist.langgraph_dev.assistants import workspace_assistant_id
    from EvoScientist.memory.autoskills.schedule import (
        AUTOSKILL_GRAPH_ID,
        AUTOSKILL_RUN_KIND,
    )
    from EvoScientist.memory.types import MEMORY_AGENT_RECURSION_LIMIT

    config: dict = {"configurable": {"workspace_dir": workspace_dir}}
    if step_limit:
        config["recursion_limit"] = MEMORY_AGENT_RECURSION_LIMIT
    payload = {"config": config}
    return {
        "cron_id": cron_id,
        "assistant_id": assistant_id
        or workspace_assistant_id(AUTOSKILL_GRAPH_ID, Workspace(workspace_dir)),
        "schedule": "0 3 * * *",
        "timezone": "UTC",
        "enabled": True,
        "metadata": {
            "run_kind": AUTOSKILL_RUN_KIND,
            "workspace_dir": workspace_dir,
            "mode": "propose",
        },
        "payload": payload if forwarded else {},
    }


@pytest.fixture
def plan():
    """Plan the reconcile of a workspace's AutoSkills crons among *rows*."""
    from EvoScientist.cron.schedule import owned
    from EvoScientist.memory.autoskills.schedule import plan_autoskill_schedule

    config = SimpleNamespace(
        memory_skill_synthesis_enabled=True,
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="nightly"),
        memory_skill_synthesis_time="03:00",
        scheduler_default_timezone="UTC",
    )

    def _plan(workspace, rows):
        return plan_autoskill_schedule(config, workspace, owned(rows, workspace))

    return _plan


def test_reconcile_autoskills_ignores_other_workspaces(
    workspace, other, plan, tmp_path
):
    # Ours forwards the workspace written another way: still up to date.
    link = tmp_path / "link"
    link.symlink_to(workspace.root, target_is_directory=True)
    rows = [
        _autoskill_cron("mine", f"{link}/", forwarded=True),
        _autoskill_cron("theirs", other.key, forwarded=True),
    ]

    result = plan(workspace, rows)

    assert result.keep["cron_id"] == "mine"
    assert result.delete == []


SYSTEM_ID = "1f1e4f53-0b5e-5c8b-9a43-0e0f8c3f6a11"


def test_reconcile_autoskills_leaves_cron_on_other_assistant(workspace, plan):
    """A cron on the graph's own assistant is no project's: the reconcile
    creates the workspace's and leaves that one alone."""
    rows = [
        # The server stores the graph's own assistant by its ID.
        _autoskill_cron("old", workspace.key, forwarded=True, assistant_id=SYSTEM_ID)
    ]

    result = plan(workspace, rows)

    assert result.create is not None
    assert result.delete == []


def test_reconcile_autoskills_replaces_cron_without_step_limit(workspace, plan):
    """A cron created before its runs carried the memory step limit."""
    rows = [_autoskill_cron("old", workspace.key, forwarded=True, step_limit=False)]

    result = plan(workspace, rows)

    assert result.create is not None
    assert [r["cron_id"] for r in result.delete] == ["old"]


def test_reconcile_autoskills_converges_after_failed_delete(workspace, plan):
    """A delete that failed last time: the next reconcile keeps the new cron
    and removes the old one."""
    rows = [
        _autoskill_cron("old", workspace.key, forwarded=False),
        _autoskill_cron("new", workspace.key, forwarded=True),
    ]

    result = plan(workspace, rows)

    assert result.keep["cron_id"] == "new"
    assert result.create is None
    assert [r["cron_id"] for r in result.delete] == ["old"]


def test_reconcile_autoskills_keeps_one_of_duplicate_crons(workspace, plan):
    rows = [
        _autoskill_cron("first", workspace.key, forwarded=True),
        _autoskill_cron("second", workspace.key, forwarded=True),
    ]

    result = plan(workspace, rows)

    assert result.keep["cron_id"] == "first"
    assert [r["cron_id"] for r in result.delete] == ["second"]


def test_reconcile_autoskills_replaces_unforwarded_cron(workspace, other, plan):
    """A cron from before runs forwarded the workspace is recreated once."""
    rows = [
        _autoskill_cron("old", workspace.key, forwarded=False),
        _autoskill_cron("theirs", other.key, forwarded=False),
    ]

    result = plan(workspace, rows)

    assert result.create is not None
    assert [r["cron_id"] for r in result.delete] == ["old"]
