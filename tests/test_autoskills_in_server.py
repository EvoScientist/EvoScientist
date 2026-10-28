"""The server reconciles each project's AutoSkills cron, in process, through
the runtime's own ops: at the project's setup, and for every project when a
client changed the settings (``/autoskills``)."""

from __future__ import annotations

import httpx
import pytest

from EvoScientist.config import EvoScientistConfig
from EvoScientist.langgraph_dev import server_crons, workspace_setup
from EvoScientist.langgraph_dev.assistants import ensure_workspace_assistants
from EvoScientist.memory.autoskills.schedule import (
    AUTOSKILL_RUN_KIND,
    plan_autoskill_schedule,
)
from EvoScientist.paths import Workspace


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "project")


def _config(
    *, enabled: bool = True, time: str = "03:00", timezone: str = "UTC"
) -> EvoScientistConfig:
    config = EvoScientistConfig()
    config.memory_skill_synthesis_enabled = enabled
    config.memory_skill_synthesis_time = time
    config.scheduler_default_timezone = timezone
    return config


def _autoskill_crons(store) -> list[dict]:
    return [
        c
        for c in store["crons"]
        if (c.get("metadata") or {}).get("run_kind") == AUTOSKILL_RUN_KIND
    ]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_plan_autoskill_schedule_disabled_deletes_every_cron(workspace):
    rows = [{"cron_id": "a"}, {"cron_id": "b"}]

    plan = plan_autoskill_schedule(_config(enabled=False), workspace, rows)

    assert plan.keep is None
    assert plan.create is None
    assert plan.delete == rows


def test_plan_autoskill_schedule_without_match_creates_then_deletes(workspace):
    stale = {"cron_id": "old", "schedule": "1 2 * * *"}

    plan = plan_autoskill_schedule(_config(), workspace, [stale])

    assert plan.create.schedule == "0 3 * * 0"
    assert plan.create.timezone == "UTC"
    assert plan.delete == [stale]


def test_plan_autoskill_schedule_refuses_unknown_timezone(workspace):
    with pytest.raises(ValueError, match="Europe/Atlantis"):
        plan_autoskill_schedule(_config(timezone="Europe/Atlantis"), workspace, [])


# ---------------------------------------------------------------------------
# In the server
# ---------------------------------------------------------------------------


async def test_reconcile_in_server_creates_cron_on_workspace_assistant(
    runtime_store, workspace
):
    from EvoScientist.langgraph_dev.assistants import launch_assistant_id

    await ensure_workspace_assistants(workspace)

    assert await server_crons.reconcile_autoskills(_config(), [workspace]) == []

    (cron,) = _autoskill_crons(runtime_store)
    assert str(cron["assistant_id"]) == launch_assistant_id(
        "evomemory-autoskills", workspace
    )
    assert cron["metadata"]["workspace_dir"] == workspace.key
    assert cron["payload"]["config"]["configurable"]["workspace_dir"] == workspace.key
    assert cron["timezone"] == "UTC"


async def test_reconcile_in_server_cron_runs_take_memory_step_limit(
    runtime_store, workspace
):
    """langgraph-api fills a run's limit with its own unless the cron's run
    config names one."""
    from EvoScientist.memory.types import MEMORY_AGENT_RECURSION_LIMIT

    await ensure_workspace_assistants(workspace)
    await server_crons.reconcile_autoskills(_config(), [workspace])

    (cron,) = _autoskill_crons(runtime_store)
    assert cron["payload"]["config"]["recursion_limit"] == MEMORY_AGENT_RECURSION_LIMIT


async def test_reconcile_in_server_keeps_matching_cron(runtime_store, workspace):
    await ensure_workspace_assistants(workspace)
    await server_crons.reconcile_autoskills(_config(), [workspace])
    (first,) = _autoskill_crons(runtime_store)

    await server_crons.reconcile_autoskills(_config(), [workspace])

    assert _autoskill_crons(runtime_store) == [first]


async def test_reconcile_in_server_replaces_cron_after_config_change(
    runtime_store, workspace
):
    await ensure_workspace_assistants(workspace)
    await server_crons.reconcile_autoskills(_config(), [workspace])

    await server_crons.reconcile_autoskills(_config(time="04:30"), [workspace])

    (cron,) = _autoskill_crons(runtime_store)
    assert cron["schedule"] == "30 4 * * 0"


async def test_reconcile_in_server_reconciles_each_project(
    runtime_store, workspace, tmp_path
):
    """A setting is the server's, so every served project follows it."""
    other = Workspace(tmp_path / "other")
    for project in (workspace, other):
        await ensure_workspace_assistants(project)
    await server_crons.reconcile_autoskills(_config(), [workspace, other])
    assert len(_autoskill_crons(runtime_store)) == 2

    await server_crons.reconcile_autoskills(_config(enabled=False), [workspace])

    (cron,) = _autoskill_crons(runtime_store)
    assert cron["metadata"]["workspace_dir"] == other.key


async def test_reconcile_in_server_failed_create_keeps_old_cron(
    runtime_store, workspace, monkeypatch, caplog
):
    """The replacement is created first: one that fails leaves the old cron
    running, and the other projects go on."""
    await ensure_workspace_assistants(workspace)
    await server_crons.reconcile_autoskills(_config(), [workspace])
    (old,) = _autoskill_crons(runtime_store)

    async def fail(*_a, **_k):
        raise RuntimeError("assistant not found")

    monkeypatch.setattr(server_crons, "_put", fail)

    (reason,) = await server_crons.reconcile_autoskills(
        _config(time="04:30"), [workspace]
    )
    assert "assistant not found" in reason
    assert _autoskill_crons(runtime_store) == [old]
    assert "assistant not found" in caplog.text


# ---------------------------------------------------------------------------
# At a project's setup, and after /autoskills
# ---------------------------------------------------------------------------


def _fake_setup(monkeypatch) -> list:
    reconciled: list = []

    async def ensure(_workspace):
        return None

    async def reconcile(_config, projects):
        reconciled.extend(projects)
        return []

    monkeypatch.setattr(
        workspace_setup.assistants, "ensure_workspace_assistants", ensure
    )
    monkeypatch.setattr(server_crons, "reconcile_autoskills", reconcile)
    return reconciled


async def test_project_setup_reconciles_autoskills(monkeypatch, workspace):
    workspace.root.mkdir(parents=True)
    reconciled = _fake_setup(monkeypatch)

    await workspace_setup._set_up(workspace, restore=False)

    assert reconciled == [workspace]


async def test_project_setup_survives_reconcile_failure(monkeypatch, workspace, caplog):
    """The project is served either way; its next setup tries again."""
    from EvoScientist.langgraph_dev import registry

    workspace.root.mkdir(parents=True)
    _fake_setup(monkeypatch)

    async def fail(_config, _projects):
        raise RuntimeError("store busy")

    monkeypatch.setattr(server_crons, "reconcile_autoskills", fail)

    await workspace_setup._set_up(workspace, restore=False)

    assert registry.REGISTRY.is_registered(workspace.key)
    assert "store busy" in caplog.text


async def test_reconcile_everywhere_covers_served_projects(monkeypatch, tmp_path):
    from EvoScientist.langgraph_dev import registry

    served = Workspace(tmp_path / "served")
    served.root.mkdir()
    gone = Workspace(tmp_path / "gone")
    for project in (served, gone):
        registry.REGISTRY.set_registered(project.key, True)
    reconciled = _fake_setup(monkeypatch)

    assert await workspace_setup.reconcile_autoskills_everywhere() == (1, [])
    assert reconciled == [served]


def test_autoskills_route_needs_local_secret(monkeypatch):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev.http import app

    monkeypatch.setattr(workspace_setup, "set_up_at_start", lambda: _none())
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/autoskills/reconcile")

    assert response.status_code == 403


async def _none():
    return None


@pytest.mark.parametrize("running", [True, False])
def test_autoskills_setting_asks_server_to_reconcile(monkeypatch, running):
    """The server reconciles with its own settings; without a server, each
    project follows at its next setup."""
    from EvoScientist.langgraph_dev import manager, registration

    asked = []
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: running)
    monkeypatch.setattr(
        registration, "reconcile_autoskills", lambda *, url: asked.append(url) or []
    )

    assert registration.apply_autoskills_settings(_config()) is None
    assert asked == (["http://127.0.0.1:6174"] if running else [])


def test_autoskills_setting_reports_failed_reconcile(monkeypatch):
    from EvoScientist.langgraph_dev import manager, registration

    def refuse(*, url):
        raise registration.RegistrationError("not now")

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(registration, "reconcile_autoskills", refuse)

    assert "not now" in registration.apply_autoskills_settings(_config())


def test_reconcile_autoskills_client_posts_with_secret(monkeypatch, tmp_path):
    from EvoScientist import local_secret, paths
    from EvoScientist.langgraph_dev import registration

    home = tmp_path / "server"
    home.mkdir()
    (home / local_secret.FILE_NAME).write_text("s3cret\n")
    monkeypatch.setattr(paths, "SERVER_HOME", home)
    sent = []
    monkeypatch.setattr(
        registration.httpx,
        "get",
        lambda _url, **_k: httpx.Response(
            200, json={"instance": local_secret.instance_of("s3cret")}
        ),
    )

    def post(url, **kwargs):
        sent.append((url, kwargs["headers"][local_secret.HEADER]))
        return httpx.Response(200, json={"projects": 3, "failed": ["/a: bad"]})

    monkeypatch.setattr(registration.httpx, "post", post)

    assert registration.reconcile_autoskills(url="http://127.0.0.1:6174") == ["/a: bad"]
    assert sent == [("http://127.0.0.1:6174/api/autoskills/reconcile", "s3cret")]


async def test_reconcile_in_server_replaces_cron_after_timezone_change(
    runtime_store, workspace
):
    await ensure_workspace_assistants(workspace)
    await server_crons.reconcile_autoskills(_config(), [workspace])

    await server_crons.reconcile_autoskills(
        _config(timezone="Asia/Shanghai"), [workspace]
    )

    (cron,) = _autoskill_crons(runtime_store)
    assert cron["timezone"] == "Asia/Shanghai"


def test_autoskills_setting_reports_projects_not_rescheduled(monkeypatch):
    from EvoScientist.langgraph_dev import manager, registration

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(
        registration, "reconcile_autoskills", lambda *, url: ["/a: bad timezone"]
    )

    assert "/a: bad timezone" in registration.apply_autoskills_settings(_config())


@pytest.mark.parametrize(
    ("body", "reconciles"),
    [
        ({"memory_skill_synthesis_time": "04:30"}, True),
        ({"show_thinking": False}, False),
    ],
)
def test_settings_change_reconciles_autoskills(monkeypatch, body, reconciles):
    """The WebUI's settings page changes the same settings."""
    from starlette.testclient import TestClient

    from EvoScientist import local_secret
    from EvoScientist.langgraph_dev import http, settings_api
    from EvoScientist.langgraph_dev.http import app

    monkeypatch.setattr(workspace_setup, "set_up_at_start", lambda: _none())
    monkeypatch.setattr(settings_api, "patch_settings", lambda _b: (200, {}))
    monkeypatch.setattr(local_secret, "_value", "s3cret")
    monkeypatch.setattr(http, "load_local_secret", lambda: None)
    calls = []

    async def everywhere():
        calls.append(1)
        return 0, []

    monkeypatch.setattr(workspace_setup, "reconcile_autoskills_everywhere", everywhere)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.patch(
            "/api/settings", json=body, headers={local_secret.HEADER: "s3cret"}
        )

    assert response.status_code == 200
    assert calls == ([1] if reconciles else [])


@pytest.mark.parametrize(
    ("value", "ok"), [("", True), ("UTC", True), ("Europe/Atlantis", False)]
)
def test_scheduler_timezone_setting_is_checked(value, ok):
    """A bad one would make every project's reconcile fail."""
    from EvoScientist.config.settings import validate_setting

    if ok:
        validate_setting("scheduler_default_timezone", value)
    else:
        with pytest.raises(ValueError, match="unknown time zone"):
            validate_setting("scheduler_default_timezone", value)


def test_reconcile_autoskills_client_shows_server_reason(monkeypatch, tmp_path):
    from EvoScientist import local_secret, paths
    from EvoScientist.langgraph_dev import registration

    home = tmp_path / "server"
    home.mkdir()
    (home / local_secret.FILE_NAME).write_text("s3cret\n")
    monkeypatch.setattr(paths, "SERVER_HOME", home)
    monkeypatch.setattr(
        registration.httpx,
        "get",
        lambda _url, **_k: httpx.Response(
            200, json={"instance": local_secret.instance_of("s3cret")}
        ),
    )
    monkeypatch.setattr(
        registration.httpx,
        "post",
        lambda _url, **_k: httpx.Response(
            500, json={"error": "reconcile_failed", "detail": "registry locked"}
        ),
    )

    with pytest.raises(registration.RegistrationError, match="registry locked"):
        registration.reconcile_autoskills(url="http://127.0.0.1:6174")


@pytest.mark.parametrize(
    ("args", "asked"),
    [
        (["set", "memory_skill_synthesis_enabled", "false"], 1),
        (["set", "scheduler_default_timezone", "Asia/Shanghai"], 1),
        (["set", "log_level", "debug"], 0),
        (["reset", "--yes"], 1),
    ],
)
def test_config_commands_reschedule_autoskills(monkeypatch, args, asked):
    """A running server would keep every project's cron on the old settings;
    after a reset it is still reached on the port it started on."""
    from typer.testing import CliRunner

    from EvoScientist.cli.commands import config_app
    from EvoScientist.config.settings import get_config_path
    from EvoScientist.langgraph_dev import registration

    get_config_path().parent.mkdir(parents=True, exist_ok=True)
    get_config_path().write_text(
        "log_level: info\nlanggraph_dev_port: 16123\n", encoding="utf-8"
    )
    ports = []
    monkeypatch.setattr(
        registration,
        "apply_autoskills_settings",
        lambda config: ports.append(config.langgraph_dev_port),
    )

    result = CliRunner().invoke(config_app, args)

    assert result.exit_code == 0, result.output
    assert ports == [16123] * asked
