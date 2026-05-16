from __future__ import annotations

import asyncio

from EvoScientist.channels.webui.channel import WebUIChannel, WebUIConfig
from EvoScientist.langgraph_dev import manager as langgraph_manager


class _FakeProcess:
    def poll(self):
        return None


def test_webui_run_mode_uses_per_thread_workspace(tmp_path):
    thread_id = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )

    assert channel._workspace_path_for_thread(thread_id) == (
        tmp_path / "runs" / f"webui_{thread_id}"
    )


def test_webui_thread_registry_discovers_run_workspaces(tmp_path):
    thread_id = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    workspace = tmp_path / "runs" / f"webui_{thread_id}"
    workspace.mkdir(parents=True)
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )

    threads = channel._load_thread_registry_sync()

    assert len(threads) == 1
    assert threads[0]["threadId"] == thread_id
    assert threads[0]["workspaceDir"] == str(workspace.resolve())


def test_webui_thread_registry_order_uses_created_at(tmp_path):
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )
    older = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    newer = "6cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"

    channel._save_thread_registry_sync(
        [
            {
                "threadId": older,
                "title": "Older",
                "createdAt": "2026-01-01T00:00:00+00:00",
                "updatedAt": "2026-12-01T00:00:00+00:00",
                "workspaceDir": str(tmp_path / "runs" / f"webui_{older}"),
            },
            {
                "threadId": newer,
                "title": "Newer",
                "createdAt": "2026-02-01T00:00:00+00:00",
                "updatedAt": "2026-02-01T00:00:00+00:00",
                "workspaceDir": str(tmp_path / "runs" / f"webui_{newer}"),
            },
        ]
    )

    threads = channel._load_thread_registry_sync()

    assert [thread["threadId"] for thread in threads] == [newer, older]


def test_webui_thread_registry_persists_manual_title_source(tmp_path):
    thread_id = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )

    record = asyncio.run(
        channel._ensure_thread_record(
            thread_id,
            title="Readable research thread",
            title_source="manual",
        )
    )
    threads = channel._load_thread_registry_sync()

    assert record["title"] == "Readable research thread"
    assert record["titleSource"] == "manual"
    assert threads[0]["titleSource"] == "manual"


def test_webui_daemon_mode_uses_workspace_root(tmp_path):
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="daemon", workspace_root=str(tmp_path))
    )

    assert (
        channel._workspace_path_for_thread("5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb")
        == tmp_path
    )


def test_webui_runtime_status_is_backend_owned():
    channel = WebUIChannel(WebUIConfig())

    status = asyncio.run(channel._runtime_status_payload())

    assert status == {
        "isRunning": False,
        "activeThreadId": None,
        "activeThreadIds": [],
        "activeWorkspaceDir": None,
        "activeWorkspaces": {},
        "createdAt": None,
    }


def test_webui_runtime_pool_starts_isolated_runtime_per_thread(tmp_path, monkeypatch):
    thread_id = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )
    calls = []

    def fake_start_isolated_langgraph_dev(
        workspace_dir,
        *,
        port,
        log_file,
        file_persistence=True,
        jobs_per_worker=10,
    ):
        calls.append(
            {
                "workspace_dir": workspace_dir,
                "port": port,
                "log_file": log_file,
                "file_persistence": file_persistence,
                "jobs_per_worker": jobs_per_worker,
            }
        )
        return _FakeProcess()

    monkeypatch.setattr(
        langgraph_manager,
        "start_isolated_langgraph_dev",
        fake_start_isolated_langgraph_dev,
    )
    monkeypatch.setattr(langgraph_manager, "is_langgraph_dev_running", lambda **_: True)
    channel._next_runtime_port_locked = lambda: 7001

    async def run():
        async with channel._runtime_lock:
            first = await channel._ensure_thread_runtime_locked(thread_id)
            second = await channel._ensure_thread_runtime_locked(thread_id)
        return first, second

    first_runtime, second_runtime = asyncio.run(run())

    assert first_runtime is second_runtime
    assert first_runtime.workspace_dir == str(tmp_path / "runs" / f"webui_{thread_id}")
    assert first_runtime.base_url == "http://localhost:7001"
    assert len(calls) == 1


def test_webui_command_catalog_prefers_native_actions():
    channel = WebUIChannel(WebUIConfig())
    catalog = {item["name"]: item for item in channel._command_catalog()}

    assert catalog["/model"]["nativeAction"] == "switch_model"
    assert catalog["/skills"]["nativeAction"] == "show_skills"
    assert catalog["/mcp"]["nativeAction"] == "manage_mcp"
    assert catalog["/compact"]["nativeAction"] is None
