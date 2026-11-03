"""The registry: the projects this server serves, their setup, and the routes
that register and unregister them."""

from __future__ import annotations

import asyncio

import pytest

from EvoScientist import local_secret, paths, sessions
from EvoScientist.langgraph_dev import (
    factories,
    registry,
    server_crons,
    workspace_setup,
)
from EvoScientist.langgraph_dev.assistants import (
    ensure_workspace_assistants,
    workspace_assistant_id,
)
from EvoScientist.paths import SessionDirs, Workspace


@pytest.fixture
def projects(monkeypatch) -> registry.Registry:
    fresh = registry.Registry()
    monkeypatch.setattr(registry, "REGISTRY", fresh)
    return fresh


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_registry_set_registered_persists_in_sessions_db(projects, workspace):
    projects.set_registered(workspace.key, True)

    assert registry.Registry().registered() == [workspace.key]
    projects.set_registered(workspace.key, False)
    reloaded = registry.Registry()
    assert reloaded.registered() == []
    assert reloaded.known() == {workspace.key}


def test_registry_maps_workspace_assistant_to_project(projects, workspace, tmp_path):
    projects.set_registered(workspace.key, False)  # known, not registered
    assistant = workspace_assistant_id("scheduler", workspace)

    assert projects.project_of_assistant("scheduler", assistant) == workspace
    # Another graph's assistant, an unknown project's, or no assistant at all.
    assert projects.project_of_assistant("writing-agent", assistant) is None
    other = workspace_assistant_id("scheduler", Workspace(tmp_path / "other"))
    assert projects.project_of_assistant("scheduler", other) is None
    assert projects.project_of_assistant("scheduler", None) is None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def _no_assistants(_workspace):
    return None


@pytest.fixture
def setup_calls(monkeypatch) -> list[tuple]:
    """Record the setup steps instead of running them."""
    calls: list[tuple] = []

    async def restore(key):
        calls.append(("restore", key))
        sessions.RESTORED_WORKSPACES.add(key)
        return True

    async def ensure(workspace):
        calls.append(("assistants", workspace.key))

    monkeypatch.setattr(workspace_setup, "restore_workspace_threads", restore)
    monkeypatch.setattr(
        workspace_setup.assistants, "ensure_workspace_assistants", ensure
    )
    monkeypatch.setattr(
        workspace_setup,
        "warm_up",
        lambda workspace: calls.append(("warm up", workspace.key)),
    )
    return calls


async def test_register_workspace_sets_project_up_then_registers_it(
    projects, setup_calls, workspace
):
    dirs = SessionDirs(workspace)
    await workspace_setup.register_workspace(dirs)

    assert setup_calls == [
        ("restore", workspace.key),
        ("assistants", workspace.key),
        ("warm up", workspace.key),
    ]
    assert projects.is_registered(workspace.key)


@pytest.mark.parametrize("failing", ["restore", "assistants"])
async def test_register_workspace_failed_setup_leaves_project_unregistered(
    projects, setup_calls, monkeypatch, workspace, failing
):
    async def restore_fails(_key):
        return False

    async def assistants_fail(_workspace):
        raise RuntimeError("store unavailable")

    if failing == "restore":
        monkeypatch.setattr(workspace_setup, "restore_workspace_threads", restore_fails)
    else:
        monkeypatch.setattr(
            workspace_setup.assistants, "ensure_workspace_assistants", assistants_fail
        )
    with pytest.raises(RuntimeError, match="Could not"):
        await workspace_setup.register_workspace(SessionDirs(workspace))

    assert not projects.is_registered(workspace.key)
    assert ("warm up", workspace.key) not in setup_calls


async def test_register_workspace_again_restores_threads(
    projects, setup_calls, workspace
):
    """Clients register their project at every start: each registration adds
    the threads CLI sessions made since the last one."""
    await workspace_setup.register_workspace(SessionDirs(workspace))
    setup_calls.clear()

    await workspace_setup.register_workspace(SessionDirs(workspace))

    assert setup_calls == [
        ("restore", workspace.key),
        ("assistants", workspace.key),
        ("warm up", workspace.key),
    ]


async def test_register_workspace_again_survives_failed_restore(
    projects, setup_calls, monkeypatch, workspace
):
    await workspace_setup.register_workspace(SessionDirs(workspace))
    setup_calls.clear()
    failed = []

    async def restore_fails(key):
        failed.append(key)
        return False

    monkeypatch.setattr(workspace_setup, "restore_workspace_threads", restore_fails)
    await workspace_setup.register_workspace(SessionDirs(workspace))

    assert failed == [workspace.key]
    assert setup_calls == [("assistants", workspace.key), ("warm up", workspace.key)]
    assert projects.is_registered(workspace.key)


async def test_register_workspace_checks_folder_before_restoring(
    projects, setup_calls, workspace
):
    await workspace_setup.register_workspace(SessionDirs(workspace))
    setup_calls.clear()
    workspace.root.rmdir()

    with pytest.raises(workspace_setup.FolderMissingError):
        await workspace_setup.register_workspace(SessionDirs(workspace))

    assert setup_calls == []


async def test_register_workspace_restores_registered_project_not_restored(
    projects, setup_calls, workspace
):
    """The restore at the server's start failed: the next registration
    restores the project's threads."""
    projects.set_registered(workspace.key, True)

    await workspace_setup.register_workspace(SessionDirs(workspace))

    assert setup_calls[0] == ("restore", workspace.key)


@pytest.mark.parametrize("failing", ["restore", "assistants"])
async def test_register_workspace_failure_keeps_registered_project(
    projects, setup_calls, monkeypatch, workspace, failing
):
    projects.set_registered(workspace.key, True)

    async def restore_fails(_key):
        return False

    async def assistants_fail(_workspace):
        raise RuntimeError("store unavailable")

    if failing == "restore":
        monkeypatch.setattr(workspace_setup, "restore_workspace_threads", restore_fails)
    else:
        monkeypatch.setattr(
            workspace_setup.assistants, "ensure_workspace_assistants", assistants_fail
        )
    with pytest.raises(RuntimeError):
        await workspace_setup.register_workspace(SessionDirs(workspace))

    assert projects.is_registered(workspace.key)


async def test_register_workspace_refuses_missing_folder(
    projects, setup_calls, tmp_path
):
    gone = Workspace(tmp_path / "gone")
    with pytest.raises(workspace_setup.FolderMissingError):
        await workspace_setup.register_workspace(SessionDirs(gone))

    assert setup_calls == []
    assert projects.known() == set()
    assert not gone.root.exists()


async def test_server_start_sets_registered_projects_up(
    projects, setup_calls, tmp_path
):
    """The checkpointer restored every project's threads already; a project
    whose folder is gone stays registered."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    gone = Workspace(tmp_path / "gone")
    for key in (other.key, gone.key):
        projects.set_registered(key, True)

    await workspace_setup.set_up_at_start()

    assert setup_calls == [("assistants", other.key)]
    assert set(projects.registered()) == {other.key, gone.key}


async def test_warm_up_builds_chat_graph_in_background(monkeypatch, workspace):
    built = []
    monkeypatch.setattr(
        factories.GRAPHS, "get", lambda graph_id, dirs: built.append((graph_id, dirs))
    )

    workspace_setup.warm_up(workspace)
    await asyncio.gather(*workspace_setup._warm_ups)

    assert built == [("EvoScientist", SessionDirs(workspace))]


async def test_warm_ups_build_one_at_a_time(monkeypatch, workspace, tmp_path):
    """Builds take turns anyway; a queued warm-up holds no worker thread."""
    import threading
    import time

    building, most = [], []
    lock = threading.Lock()

    def get(_graph_id, _dirs):
        with lock:
            building.append(1)
            most.append(len(building))
        time.sleep(0.02)
        with lock:
            building.pop()

    monkeypatch.setattr(factories.GRAPHS, "get", get)
    for name in ("a", "b", "c"):
        workspace_setup.warm_up(Workspace(tmp_path / name))
    await asyncio.gather(*workspace_setup._warm_ups)

    assert max(most) == 1


# ---------------------------------------------------------------------------
# A project's crons, through the runtime's own store
# ---------------------------------------------------------------------------


async def _put_cron(workspace: Workspace, graph_id: str, run_kind: str, **kwargs):
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Crons

    async with connect() as conn:
        rows = await Crons.put(
            conn,
            payload={"assistant_id": workspace_assistant_id(graph_id, workspace)},
            schedule="0 3 * * *",
            metadata={"run_kind": run_kind},
            enabled=kwargs.get("enabled", True),
        )
        return await anext(rows)


async def test_project_crons_are_those_on_its_assistants(
    runtime_store, workspace, tmp_path
):
    other = Workspace(tmp_path / "other")
    await ensure_workspace_assistants(workspace)
    await ensure_workspace_assistants(other)
    mine = await _put_cron(workspace, "scheduler", "scheduled_task")
    await _put_cron(other, "scheduler", "scheduled_task")

    crons = await server_crons.project_crons(workspace)

    assert [cron["cron_id"] for cron in crons] == [mine["cron_id"]]


async def test_drop_system_cron_deletes_only_project_system_cron(
    runtime_store, workspace, tmp_path
):
    """A user cron stays as it is: its fires are refused until the folder is
    back, and leave nothing in the store."""
    other = Workspace(tmp_path / "other")
    await ensure_workspace_assistants(workspace)
    await ensure_workspace_assistants(other)
    task = await _put_cron(workspace, "scheduler", "scheduled_task")
    autoskills = await _put_cron(
        workspace, "evomemory-autoskills", "evomemory_autoskills"
    )
    others = await _put_cron(other, "evomemory-autoskills", "evomemory_autoskills")

    for cron in (task, autoskills, others):
        await server_crons.drop_system_cron(str(cron["cron_id"]), workspace)

    kept = {cron["cron_id"]: cron for cron in runtime_store["crons"]}
    assert set(kept) == {task["cron_id"], others["cron_id"]}
    assert kept[task["cron_id"]]["enabled"] is True


@pytest.fixture
def missing_project(monkeypatch, projects, workspace) -> dict:
    """A cron's run config for a registered project whose folder is gone."""
    projects.set_registered(workspace.key, True)
    workspace.root.rmdir()
    monkeypatch.setattr(
        factories, "thread_binding", lambda _thread_id: SessionDirs(workspace)
    )
    return {
        "configurable": {
            "thread_id": "t",
            "assistant_id": workspace_assistant_id("scheduler", workspace),
            "cron_id": "c1",
        }
    }


async def _access(config: dict, access: str):
    from types import SimpleNamespace

    return await factories.served_graph("scheduler")(
        config, SimpleNamespace(access_context=access)
    )


async def test_factory_drops_cron_whose_project_folder_is_missing(
    monkeypatch, missing_project, workspace
):
    from starlette.exceptions import HTTPException

    from EvoScientist.langgraph_dev.folder_check import ProjectNotServedError

    dropped = []

    async def drop(cron_id, project):
        dropped.append((cron_id, project))

    monkeypatch.setattr(server_crons, "drop_system_cron", drop)
    with pytest.raises(ProjectNotServedError):
        await _access(missing_project, "threads.create_run")
    # An update of a thread a cron created drops nothing.
    with pytest.raises(HTTPException):
        await _access(missing_project, "threads.update")

    assert dropped == [("c1", workspace)]


async def test_factory_refuses_run_when_cron_drop_fails(monkeypatch, missing_project):
    from EvoScientist.langgraph_dev.folder_check import ProjectNotServedError

    async def fail(_cron_id, _project):
        raise ValueError("badly formed hexadecimal UUID string")

    monkeypatch.setattr(server_crons, "drop_system_cron", fail)
    with pytest.raises(ProjectNotServedError):
        await _access(missing_project, "threads.create_run")


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


@pytest.fixture
def server(monkeypatch, no_workspace_setup, projects):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev.http import app

    monkeypatch.setattr(local_secret, "_value", None)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


@pytest.fixture
def secret(server) -> dict[str, str]:
    value = (paths.SERVER_HOME / local_secret.FILE_NAME).read_text()
    return {local_secret.HEADER: value}


def test_list_workspaces_shows_registered_projects(
    server, projects, workspace, tmp_path
):
    gone = Workspace(tmp_path / "gone")
    projects.set_registered(workspace.key, True)
    projects.set_registered(gone.key, True)
    projects.set_registered(Workspace(tmp_path / "unregistered").key, False)

    body = server.get("/api/workspaces").json()

    assert body == {
        "workspaces": [
            {
                "workspace_dir": workspace.key,
                "name": workspace.root.name,
                "missing": False,
            },
            {"workspace_dir": gone.key, "name": "gone", "missing": True},
        ]
    }


def test_register_workspace_route_needs_secret(server, workspace):
    response = server.post("/api/workspaces", json={"workspace_dir": workspace.key})
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("api_url", "tunnel"),
    [
        ("https://abc-def.trycloudflare.com", True),
        # A tunnel that did not come up in time: the local URL.
        ("http://0.0.0.0:6174", False),
    ],
)
def test_register_workspace_route_sets_project_up(
    server, secret, monkeypatch, workspace, api_url, tunnel
):
    registered = []

    async def register(dirs):
        registered.append(dirs)

    monkeypatch.setattr(workspace_setup, "register_workspace", register)
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_HOST", "0.0.0.0")
    monkeypatch.setenv("LANGGRAPH_API_URL", api_url)
    response = server.post(
        "/api/workspaces", json={"workspace_dir": f"{workspace.root}/"}, headers=secret
    )

    assert response.status_code == 200
    assert response.json() == {
        "workspace_dir": workspace.key,
        "host": "0.0.0.0",
        "tunnel": tunnel,
    }
    assert registered == [SessionDirs(workspace)]


def test_register_workspace_route_reports_failed_setup(
    server, secret, monkeypatch, workspace
):
    async def fail(_dirs):
        raise RuntimeError("Could not restore the threads.")

    monkeypatch.setattr(workspace_setup, "register_workspace", fail)
    response = server.post(
        "/api/workspaces", json={"workspace_dir": workspace.key}, headers=secret
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "setup_failed",
        "detail": "Could not restore the threads.",
    }


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({"workspace_dir": "relative/ws"}, "invalid_path"),
        ({"workspace_dir": 123}, "invalid_path"),
        ({}, "invalid_path"),
        (None, "folder_missing"),
    ],
)
def test_register_workspace_route_refuses_bad_folder(
    server, secret, body, error, tmp_path
):
    if body is None:
        body = {"workspace_dir": str(tmp_path / "nonexistent")}
    response = server.post("/api/workspaces", json=body, headers=secret)
    assert response.status_code == 400
    assert response.json() == {"error": error}


def test_unregister_workspace_route_refuses_unregistered(server, secret, workspace):
    response = server.delete(
        "/api/workspaces", params={"workspace_dir": workspace.key}, headers=secret
    )
    assert response.status_code == 404
    assert response.json() == {"error": "not_registered"}


def test_unregister_workspace_route_refuses_while_user_cron_enabled(
    server, secret, projects, monkeypatch, workspace
):
    projects.set_registered(workspace.key, True)
    cron = {
        "cron_id": "c1",
        "enabled": True,
        "metadata": {"run_kind": "scheduled_task"},
    }

    async def crons(_workspace):
        return [cron]

    monkeypatch.setattr(server_crons, "project_crons", crons)
    response = server.delete(
        "/api/workspaces", params={"workspace_dir": workspace.key}, headers=secret
    )

    assert response.status_code == 409
    assert response.json() == {"error": "enabled_user_crons", "cron_ids": ["c1"]}
    assert projects.is_registered(workspace.key)


def test_unregister_workspace_route_keeps_project_and_drops_system_crons(
    server, secret, projects, monkeypatch, workspace
):
    projects.set_registered(workspace.key, True)
    workspace_setup.SET_UP.add(workspace.key)
    rows = [
        {
            "cron_id": "task",
            "enabled": False,
            "metadata": {"run_kind": "scheduled_task"},
        },
        {
            "cron_id": "skills",
            "enabled": True,
            "metadata": {"run_kind": "evomemory_autoskills"},
        },
    ]
    deleted = []

    async def crons(_workspace):
        return rows

    async def delete(cron_ids):
        deleted.extend(cron_ids)

    monkeypatch.setattr(server_crons, "project_crons", crons)
    monkeypatch.setattr(server_crons, "delete_crons", delete)
    response = server.delete(
        "/api/workspaces", params={"workspace_dir": workspace.key}, headers=secret
    )

    assert response.status_code == 200
    assert deleted == ["skills"]
    assert not projects.is_registered(workspace.key)
    assert workspace.key not in workspace_setup.SET_UP
    assert projects.known() == {workspace.key}
    assert workspace.root.is_dir()


def test_unregister_workspace_route_reports_failure(
    server, secret, projects, monkeypatch, workspace
):
    projects.set_registered(workspace.key, True)

    async def fail(_workspace):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(server_crons, "project_crons", fail)
    response = server.delete(
        "/api/workspaces", params={"workspace_dir": workspace.key}, headers=secret
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "unregister_failed",
        "detail": "store unavailable",
    }
    assert projects.is_registered(workspace.key)


async def test_unregister_workspace_route_waits_for_registration_in_progress(
    monkeypatch, projects, setup_calls, workspace
):
    """A DELETE sent while the project is being registered answers once that
    is done, so the registration cannot record the project behind it."""
    import httpx

    from EvoScientist.langgraph_dev.http import app

    restoring, release = asyncio.Event(), asyncio.Event()

    async def restore(_key):
        restoring.set()
        await release.wait()
        return True

    async def no_crons(_workspace):
        return []

    async def delete(_cron_ids):
        return None

    monkeypatch.setattr(workspace_setup, "restore_workspace_threads", restore)
    monkeypatch.setattr(server_crons, "project_crons", no_crons)
    monkeypatch.setattr(server_crons, "delete_crons", delete)
    monkeypatch.setattr(local_secret, "_value", "secret")
    headers = {local_secret.HEADER: "secret"}
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1"
    ) as client:
        register = asyncio.create_task(
            client.post(
                "/api/workspaces",
                json={"workspace_dir": workspace.key},
                headers=headers,
            )
        )
        await restoring.wait()
        unregister = asyncio.create_task(
            client.delete(
                "/api/workspaces",
                params={"workspace_dir": workspace.key},
                headers=headers,
            )
        )
        await asyncio.sleep(0.05)
        assert not unregister.done()
        release.set()
        assert (await register).status_code == 200
        response = await unregister

    assert response.status_code == 200
    assert not projects.is_registered(workspace.key)


async def test_registration_restore_runs_off_event_loop(runtime_store, workspace):
    """Registering happens while the server serves, where blockbuster refuses
    blocking calls on the event loop."""
    from blockbuster import BlockBuster

    blocker = BlockBuster(scanned_modules=[sessions])
    blocker.activate()
    try:
        restored = await sessions.restore_workspace_threads(workspace.key)
    finally:
        blocker.deactivate()
    assert restored
    assert workspace.key in sessions.RESTORED_WORKSPACES
