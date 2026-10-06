"""Scheduled tasks belong to a workspace: tagged, listed and managed within it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.cron import schedule as crons
from EvoScientist.paths import SessionDirs, Workspace


def _cron(cron_id: str, workspace_dir: str | None = None, **metadata) -> dict:
    meta = {"run_kind": crons.SCHEDULED_RUN_KIND, "name": cron_id, **metadata}
    if workspace_dir is not None:
        meta["workspace_dir"] = workspace_dir
    return {"cron_id": cron_id, "schedule": "0 7 * * *", "metadata": meta}


@pytest.fixture
def other(tmp_path) -> Workspace:
    """Another live workspace on this machine (its folder exists)."""
    ws = Workspace(tmp_path / "other")
    ws.root.mkdir(parents=True)
    return ws


@pytest.fixture
def client(workspace, other, monkeypatch) -> MagicMock:
    """A server with one task of each workspace and one untagged task."""
    fake = MagicMock()
    fake.crons.search.return_value = [
        _cron("mine", workspace.key),
        _cron("theirs", other.key),
        _cron("untagged"),
    ]
    monkeypatch.setattr(crons, "_client", lambda: fake)
    monkeypatch.setattr(crons, "_default_timezone", lambda: "UTC")
    return fake


def test_belongs_to_matches_tagged_workspace(workspace, other, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(workspace.root, target_is_directory=True)

    assert crons.belongs_to(_cron("a", workspace.key), workspace)
    assert crons.belongs_to(_cron("a", f"{link}/"), workspace)
    assert not crons.belongs_to(_cron("a", other.key), workspace)
    # A tag that is no path here names no workspace here.
    assert not crons.belongs_to(_cron("a", "/tmp/evil\x00x"), workspace)
    # Created before tasks were tagged: the server serves one workspace.
    assert crons.belongs_to(_cron("a"), workspace)
    # The project was moved or renamed: its store, and the cron in it, came along.
    assert crons.belongs_to(_cron("a", str(tmp_path / "gone")), workspace)


def test_adopt_stale_tasks_retags_untagged_and_moved_crons(workspace, other, tmp_path):
    fake = MagicMock()
    fake.crons.search.return_value = [
        _cron("mine", workspace.key),
        _cron("untagged"),
        _cron("moved", str(tmp_path / "gone")),
        _cron("theirs", other.key),
    ]

    adopted = crons.adopt_stale_tasks(
        fake, workspace, run_kind=crons.SCHEDULED_RUN_KIND
    )

    assert adopted == 2
    assert sorted(c.args[0] for c in fake.crons.update.call_args_list) == [
        "moved",
        "untagged",
    ]
    for call in fake.crons.update.call_args_list:
        cron_id = call.args[0]
        assert call.kwargs == {
            # The rest of the metadata is kept, whatever the server merges.
            "metadata": {
                "run_kind": crons.SCHEDULED_RUN_KIND,
                "name": cron_id,
                "workspace_dir": workspace.key,
            },
            "config": {"configurable": {"workspace_dir": workspace.key}},
        }


def test_list_schedules_filters_by_workspace(workspace, client):
    ids = [row["cron_id"] for row in crons.list_schedules(workspace=workspace)]
    assert ids == ["mine", "untagged"]


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


def _autoskill_cron(cron_id: str, workspace_dir: str, *, forwarded: bool) -> dict:
    from EvoScientist.memory.autoskills.schedule import AUTOSKILL_RUN_KIND

    payload = {"config": {"configurable": {"workspace_dir": workspace_dir}}}
    return {
        "cron_id": cron_id,
        "schedule": "0 3 * * *",
        "enabled": True,
        "metadata": {
            "run_kind": AUTOSKILL_RUN_KIND,
            "workspace_dir": workspace_dir,
            "mode": "propose",
        },
        "payload": payload if forwarded else {},
    }


@pytest.fixture
def autoskills(monkeypatch):
    import EvoScientist.langgraph_dev.manager as manager
    import EvoScientist.memory.autoskills.schedule as schedule

    fake = MagicMock()
    fake.crons.create.return_value = {"cron_id": "new"}
    monkeypatch.setattr(schedule, "get_langgraph_sync_client", lambda **_k: fake)
    monkeypatch.setattr(schedule, "langgraph_dev_url", lambda _c: "http://x")
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    config = SimpleNamespace(
        memory_skill_synthesis_enabled=True,
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="nightly"),
        memory_skill_synthesis_time="03:00",
    )
    return schedule, fake, config


def test_reconcile_autoskills_ignores_other_workspaces(
    workspace, other, autoskills, tmp_path
):
    schedule, fake, config = autoskills
    # Ours forwards the workspace written another way: still up to date.
    link = tmp_path / "link"
    link.symlink_to(workspace.root, target_is_directory=True)
    fake.crons.search.return_value = [
        _autoskill_cron("mine", f"{link}/", forwarded=True),
        _autoskill_cron("theirs", other.key, forwarded=True),
    ]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "unchanged"
    fake.crons.delete.assert_not_called()


def test_reconcile_autoskills_replaces_unforwarded_cron(workspace, other, autoskills):
    """A cron from before runs forwarded the workspace is recreated once."""
    schedule, fake, config = autoskills
    fake.crons.search.return_value = [
        _autoskill_cron("old", workspace.key, forwarded=False),
        _autoskill_cron("theirs", other.key, forwarded=False),
    ]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "created"
    fake.crons.delete.assert_called_once_with("old")
    assert fake.crons.create.call_args.kwargs["config"] == {
        "configurable": {"workspace_dir": workspace.key}
    }


def test_reconcile_autoskills_replaces_moved_folder_cron(
    workspace, autoskills, tmp_path
):
    """The project was renamed: the store, and its cron tagged with the old
    folder, came along. The cron is ours and is recreated for the new root."""
    schedule, fake, config = autoskills
    gone = str(tmp_path / "gone")
    fake.crons.search.return_value = [_autoskill_cron("stale", gone, forwarded=True)]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "created"
    fake.crons.delete.assert_called_once_with("stale")
