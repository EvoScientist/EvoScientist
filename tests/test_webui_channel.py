from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from EvoScientist.channels.webui.channel import WebUIChannel, WebUIConfig


def _request(headers=None, **query):
    return SimpleNamespace(
        headers={"Host": "localhost", **(headers or {})},
        query=query,
        match_info={},
    )


def test_webui_uses_shared_workspace_root(tmp_path):
    thread_id = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )

    assert channel._workspace_path_for_thread(thread_id) == tmp_path.resolve()


def test_webui_route_normalizes_configured_base_path():
    channel = WebUIChannel(WebUIConfig(base_path="webui/"))

    assert channel._route("/healthz") == "/webui/healthz"


def test_webui_cors_allows_configured_public_origin():
    channel = WebUIChannel(WebUIConfig(allowed_origins="https://ui.example.com"))
    request = _request(headers={"Origin": "https://ui.example.com"})

    response = asyncio.run(channel._handle_options(request))

    assert response.status == 204
    assert response.headers["Access-Control-Allow-Origin"] == "https://ui.example.com"


def test_webui_host_allows_configured_public_host():
    channel = WebUIChannel(
        WebUIConfig(bind_host="0.0.0.0", allowed_hosts="api.example.com")
    )

    assert channel._request_host_allowed(
        _request(headers={"Host": "api.example.com:8010"})
    )


def test_webui_proxy_strips_control_auth_headers():
    channel = WebUIChannel(WebUIConfig())

    headers = channel._proxied_request_headers(
        {
            "Authorization": "Bearer webui-secret",
            "X-API-Key": "webui-key",
            "Content-Type": "application/json",
        }
    )

    assert headers == {"Content-Type": "application/json"}


def test_webui_thread_registry_does_not_discover_run_workspaces(tmp_path):
    thread_id = "5cd2e8b1-b4a6-48cb-9fe0-11441a6ffddb"
    workspace = tmp_path / "runs" / f"webui_{thread_id}"
    workspace.mkdir(parents=True)
    channel = WebUIChannel(
        WebUIConfig(workspace_mode="run", workspace_root=str(tmp_path))
    )

    threads = channel._load_thread_registry_sync()

    assert threads == []


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


def test_webui_runtime_status_points_to_shared_runtime(tmp_path):
    channel = WebUIChannel(WebUIConfig(workspace_root=str(tmp_path)))

    status = asyncio.run(channel._runtime_status_payload())

    assert status == {
        "isRunning": False,
        "activeThreadId": None,
        "activeThreadIds": [],
        "activeWorkspaceDir": str(tmp_path.resolve()),
        "activeWorkspaces": {},
        "createdAt": None,
        "langGraphApiUrl": "/api/langgraph",
        "langGraphBaseUrl": "http://localhost:6174",
        "workspaceDir": str(tmp_path.resolve()),
    }


def test_webui_file_tree_rejects_invalid_path(tmp_path):
    channel = WebUIChannel(WebUIConfig(workspace_root=str(tmp_path)))

    response = asyncio.run(channel._handle_ui_files_tree(_request(path="../secret")))

    assert response.status == 400
    assert json.loads(response.text) == {"error": "invalid path"}


def test_webui_file_read_rejects_invalid_path(tmp_path):
    channel = WebUIChannel(WebUIConfig(workspace_root=str(tmp_path)))

    response = asyncio.run(channel._handle_ui_files_read(_request(path="../secret")))

    assert response.status == 400
    assert json.loads(response.text) == {"error": "invalid path"}


def test_webui_command_catalog_prefers_native_actions():
    channel = WebUIChannel(WebUIConfig())
    catalog = {item["name"]: item for item in channel._command_catalog()}

    assert catalog["/model"]["nativeAction"] == "switch_model"
    assert catalog["/skills"]["nativeAction"] == "show_skills"
    assert catalog["/mcp"]["nativeAction"] == "manage_mcp"
    assert catalog["/compact"]["nativeAction"] is None


def test_webui_workspace_zip_helper_creates_downloadable_archive(tmp_path):
    channel = WebUIChannel(WebUIConfig())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hello")

    archive_path = channel._create_workspace_zip_sync(workspace)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.read("notes.txt") == b"hello"
    finally:
        Path(archive_path).unlink()
