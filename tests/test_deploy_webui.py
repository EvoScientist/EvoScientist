from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.deploy.webui import (
    WebUIConfig,
    WebUIConfigurationError,
    WebUIControlServer,
)

_AUTH_HEADERS = {"X-API-Key": "secret"}


def _request(headers=None, payload=None, **query):
    async def _json():
        return payload or {}

    return SimpleNamespace(
        headers={"Host": "localhost", **(headers or {})},
        query=query,
        match_info={},
        json=_json,
    )


def _server(tmp_path, **config):
    return WebUIControlServer(
        WebUIConfig(**config),
        workspace_dir=tmp_path,
        langgraph_base_url="http://localhost:6174",
    )


def test_webui_route_normalizes_configured_base_path(tmp_path):
    server = _server(tmp_path, base_path="webui/")

    assert server._route("/healthz") == "/webui/healthz"


def test_webui_health_reports_auth_required_without_rejecting(tmp_path):
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(server._handle_health(_request()))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["authRequired"] is True
    assert payload["authenticated"] is False


def test_webui_health_reports_authenticated_with_api_key(tmp_path):
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(
        server._handle_health(_request(headers={"X-API-Key": "secret"}))
    )
    payload = json.loads(response.text)

    assert payload["authRequired"] is True
    assert payload["authenticated"] is True


def test_webui_rejects_non_loopback_without_api_key(tmp_path):
    server = _server(tmp_path, bind_host="0.0.0.0", api_key="")

    with pytest.raises(WebUIConfigurationError) as exc_info:
        server._validate_start_security()

    assert "webui_api_key is required" in str(exc_info.value)


def test_webui_cors_allows_configured_public_origin(tmp_path):
    server = _server(tmp_path, allowed_origins="https://ui.example.com")
    request = _request(headers={"Origin": "https://ui.example.com"})

    response = asyncio.run(server._handle_options(request))

    assert response.status == 204
    assert response.headers["Access-Control-Allow-Origin"] == "https://ui.example.com"


def test_webui_cors_does_not_reflect_unknown_private_origin(tmp_path):
    server = _server(tmp_path)
    request = _request(headers={"Origin": "http://192.168.1.100:3000"})

    response = asyncio.run(server._handle_options(request))

    assert response.status == 403
    assert "Access-Control-Allow-Origin" not in response.headers


def test_webui_file_tree_rejects_invalid_path(tmp_path):
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(
        server._handle_ui_files_tree(_request(headers=_AUTH_HEADERS, path="../secret"))
    )

    assert response.status == 400
    assert json.loads(response.text) == {"error": "invalid path"}


def test_webui_file_read_rejects_invalid_path(tmp_path):
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(
        server._handle_ui_files_read(_request(headers=_AUTH_HEADERS, path="../secret"))
    )

    assert response.status == 400
    assert json.loads(response.text) == {"error": "invalid path"}


def test_webui_file_read_is_bounded_and_marks_truncated(tmp_path):
    (tmp_path / "large.txt").write_text("a" * (256 * 1024 + 5))
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(
        server._handle_ui_files_read(_request(headers=_AUTH_HEADERS, path="large.txt"))
    )
    payload = json.loads(response.text)

    assert payload["truncated"] is True
    assert len(payload["content"]) == 256 * 1024


def test_webui_file_tree_hides_dotfiles_and_internal_state(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".langgraph_api").mkdir()
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(
        server._handle_ui_files_tree(_request(headers=_AUTH_HEADERS))
    )
    payload = json.loads(response.text)

    assert [entry["name"] for entry in payload["entries"]] == ["notes.txt"]
    assert payload["truncated"] is False


def test_webui_file_tree_reports_truncated_results(monkeypatch, tmp_path):
    from EvoScientist.deploy import webui as webui_mod

    for index in range(3):
        (tmp_path / f"{index}.txt").write_text(str(index))
    server = _server(tmp_path, api_key="secret")
    monkeypatch.setattr(webui_mod, "_MAX_WORKSPACE_TREE_ENTRIES", 2)

    response = asyncio.run(
        server._handle_ui_files_tree(_request(headers=_AUTH_HEADERS))
    )
    payload = json.loads(response.text)

    assert [entry["name"] for entry in payload["entries"]] == ["0.txt", "1.txt"]
    assert payload["truncated"] is True
    assert payload["limit"] == 2


def test_webui_workspace_zip_excludes_dotfiles_and_internal_state(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".langgraph_api").mkdir()
    (tmp_path / ".langgraph_api" / "store.sqlite").write_text("secret")
    server = _server(tmp_path)

    archive_path = server._create_workspace_zip_sync(tmp_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.namelist() == ["notes.txt"]
            assert archive.read("notes.txt") == b"hello"
    finally:
        Path(archive_path).unlink()


def test_webui_provider_key_response_mentions_restart(monkeypatch, tmp_path):
    server = _server(tmp_path, api_key="secret")
    monkeypatch.setattr(
        server,
        "_load_config",
        lambda: SimpleNamespace(openai_api_key="", provider="openai", model="gpt-4o"),
    )
    monkeypatch.setattr(server, "_models_overview", lambda: {"providers": []})

    from EvoScientist.config import settings as settings_mod

    monkeypatch.setattr(settings_mod, "set_config_value", lambda *_args: True)

    response = asyncio.run(
        server._handle_ui_provider_key(
            _request(
                headers=_AUTH_HEADERS,
                payload={"provider": "openai", "apiKey": "sk-test"},
            )
        )
    )
    payload = json.loads(response.text)

    assert payload["ok"] is True
    assert "Restart the LangGraph run" in payload["message"]


def test_webui_provider_base_url_response_mentions_restart(monkeypatch, tmp_path):
    server = _server(tmp_path, api_key="secret")
    monkeypatch.setattr(
        server,
        "_load_config",
        lambda: SimpleNamespace(
            custom_openai_base_url="",
            provider="custom-openai",
            model="gpt-4o",
        ),
    )
    monkeypatch.setattr(server, "_models_overview", lambda: {"providers": []})

    from EvoScientist.config import settings as settings_mod

    monkeypatch.setattr(settings_mod, "set_config_value", lambda *_args: True)

    response = asyncio.run(
        server._handle_ui_provider_base_url(
            _request(
                headers=_AUTH_HEADERS,
                payload={
                    "provider": "custom-openai",
                    "baseUrl": "https://api.example.com/v1",
                },
            )
        )
    )
    payload = json.loads(response.text)

    assert payload["ok"] is True
    assert "Restart the LangGraph run" in payload["message"]


def test_webui_sensitive_endpoint_requires_configured_api_key(tmp_path):
    server = _server(tmp_path)

    response = asyncio.run(
        server._handle_ui_provider_key(
            _request(payload={"provider": "openai", "apiKey": "sk-test"})
        )
    )
    payload = json.loads(response.text)

    assert response.status == 403
    assert payload == {"error": "webui_api_key is required for this endpoint"}


def test_webui_sensitive_endpoint_rejects_missing_api_key(tmp_path):
    server = _server(tmp_path, api_key="secret")

    response = asyncio.run(
        server._handle_ui_provider_key(
            _request(payload={"provider": "openai", "apiKey": "sk-test"})
        )
    )

    assert response.status == 401
