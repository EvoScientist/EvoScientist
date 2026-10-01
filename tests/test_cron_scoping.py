"""Scheduled tasks belong to a workspace: tagged, listed and managed within it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.cron import schedule as crons
from EvoScientist.paths import Workspace


def _cron(cron_id: str, workspace_dir: str | None = None, **metadata) -> dict:
    meta = {"run_kind": crons.SCHEDULED_RUN_KIND, "name": cron_id, **metadata}
    if workspace_dir is not None:
        meta["workspace_dir"] = workspace_dir
    return {"cron_id": cron_id, "schedule": "0 7 * * *", "metadata": meta}


@pytest.fixture
def other(tmp_path) -> Workspace:
    return Workspace(tmp_path / "other")


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
    return fake


def test_a_task_belongs_to_the_workspace_it_names(workspace, other, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(workspace.root, target_is_directory=True)

    assert crons.belongs_to(_cron("a", workspace.key), workspace)
    assert crons.belongs_to(_cron("a", f"{link}/"), workspace)
    assert not crons.belongs_to(_cron("a", other.key), workspace)
    # Created before tasks were tagged: the server serves one workspace.
    assert crons.belongs_to(_cron("a"), workspace)


def test_listing_shows_only_the_workspace_s_tasks(workspace, client):
    ids = [row["cron_id"] for row in crons.list_schedules(workspace=workspace)]
    assert ids == ["mine", "untagged"]


def test_another_workspace_s_task_cannot_be_deleted_or_paused(workspace, other, client):
    with pytest.raises(LookupError):
        crons.delete_schedule("theirs", workspace=workspace)
    with pytest.raises(LookupError):
        crons.set_enabled("theirs", False, workspace=workspace)
    client.crons.delete.assert_not_called()
    client.crons.update.assert_not_called()

    crons.delete_schedule("mine", workspace=workspace)
    client.crons.delete.assert_called_once_with("mine")


def test_new_tasks_are_tagged_and_run_in_the_workspace_root(workspace, client):
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


def test_the_agent_s_tools_stay_in_their_workspace(workspace, client, monkeypatch):
    from EvoScientist.middleware.scheduler import make_scheduling_tools

    monkeypatch.setattr(crons, "is_available", lambda: True)
    tools = {t.name: t for t in make_scheduling_tools(workspace)}

    listed = tools["list_scheduled_tasks"].invoke({})
    assert "mine" in listed
    assert "theirs" not in listed

    refused = tools["cancel_scheduled_task"].invoke({"cron_id": "theirs"})
    assert "No scheduled task" in refused
    client.crons.delete.assert_not_called()


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
    monkeypatch.setattr(schedule, "autoskill_cron", lambda *_a: "0 3 * * *")
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    config = SimpleNamespace(
        memory_skill_synthesis_enabled=True,
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="daily"),
        memory_skill_synthesis_time="03:00",
    )
    return schedule, fake, config


def test_reconcile_leaves_other_workspaces_autoskills_alone(
    workspace, other, autoskills
):
    schedule, fake, config = autoskills
    fake.crons.search.return_value = [
        _autoskill_cron("mine", workspace.key, forwarded=True),
        _autoskill_cron("theirs", other.key, forwarded=True),
    ]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "unchanged"
    fake.crons.delete.assert_not_called()


def test_reconcile_replaces_a_cron_whose_runs_forward_nothing(
    workspace, other, autoskills
):
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
