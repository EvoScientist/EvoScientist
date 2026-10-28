"""A client's side of the project registry: the folder check every launcher
runs, and the registration of its project with the server it uses."""

from __future__ import annotations

import os

import httpx
import pytest

from EvoScientist import local_secret, paths
from EvoScientist.langgraph_dev import registration
from EvoScientist.langgraph_dev.registration import (
    OlderServerError,
    ProjectFolderMissingError,
    RegistrationError,
    prepare_workdir,
)
from EvoScientist.langgraph_dev.registration import register_folder as real_register
from EvoScientist.paths import Workspace

URL = "http://127.0.0.1:6174"


@pytest.fixture
def secret(monkeypatch, tmp_path) -> str:
    """The server's local secret, in this test's server home."""
    home = tmp_path / "server"
    home.mkdir()
    (home / local_secret.FILE_NAME).write_text("s3cret\n")
    monkeypatch.setattr(paths, "SERVER_HOME", home)
    return "s3cret"


def _answer(
    monkeypatch, status: int, body: dict | None = None, *, instance: str = "s3cret"
) -> list[dict]:
    """Fake a server holding the secret *instance* and its answer to the
    registration; return the registrations sent."""
    sent: list[dict] = []

    def get(url, **_kwargs):
        assert url.endswith("/api/instance")
        return httpx.Response(
            200, json={"instance": local_secret.instance_of(instance)}
        )

    def post(url, **kwargs):
        sent.append({"url": url, **kwargs})
        return httpx.Response(status, json=body or {})

    monkeypatch.setattr(registration.httpx, "get", get)
    monkeypatch.setattr(registration.httpx, "post", post)
    return sent


# ---------------------------------------------------------------------------
# The folder check
# ---------------------------------------------------------------------------


def test_prepare_workdir_creates_new_folder(tmp_path):
    folder = tmp_path / "new" / "project"

    assert prepare_workdir(folder) == folder
    assert folder.is_dir()


def test_prepare_workdir_refuses_known_project_with_missing_folder(tmp_path):
    """Recreating it empty would hide its threads and scheduled tasks."""
    from EvoScientist.langgraph_dev.registry import Registry

    gone = Workspace(tmp_path / "gone")
    Registry().set_registered(gone.key, False)  # known, not even registered

    with pytest.raises(ProjectFolderMissingError, match="Restore the folder"):
        prepare_workdir(gone.root)
    assert not gone.root.exists()


def test_prepare_workdir_without_registry_creates_folder(tmp_path):
    """Before any server created the table, no path is known."""
    folder = tmp_path / "project"

    prepare_workdir(folder)

    assert folder.is_dir()


def test_is_known_project_reads_data_folder_with_uri_characters(monkeypatch, tmp_path):
    from EvoScientist.langgraph_dev.registry import Registry

    name = "r&d #2 100%" if os.name == "nt" else "r&d #2 ?100%"
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / name)
    gone = Workspace(tmp_path / "gone")
    Registry().set_registered(gone.key, True)

    assert registration.is_known_project(gone)


def test_is_known_project_never_creates_data_folder(monkeypatch, tmp_path):
    data = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_DIR", data)

    assert not registration.is_known_project(Workspace(tmp_path / "project"))
    assert not data.exists()


# ---------------------------------------------------------------------------
# The secret
# ---------------------------------------------------------------------------


def test_read_local_secret_never_creates_file(monkeypatch, tmp_path):
    home = tmp_path / "server"
    monkeypatch.setattr(paths, "SERVER_HOME", home)

    assert local_secret.read_local_secret() is None
    assert not home.exists()


def test_read_local_secret_empty_file_reads_none(monkeypatch, tmp_path):
    home = tmp_path / "server"
    home.mkdir()
    (home / local_secret.FILE_NAME).write_text("\n")
    monkeypatch.setattr(paths, "SERVER_HOME", home)

    assert local_secret.read_local_secret() is None


# ---------------------------------------------------------------------------
# The registration
# ---------------------------------------------------------------------------


def test_register_folder_sends_project_root_and_secret(monkeypatch, secret, tmp_path):
    sent = _answer(monkeypatch, 200, {"workspace_dir": "x"})
    workspace = Workspace(tmp_path)

    real_register(workspace, url=URL)

    (request,) = sent
    assert request["url"] == f"{URL}/api/workspaces"
    assert request["json"] == {"workspace_dir": workspace.key}
    assert request["headers"][local_secret.HEADER] == secret
    # Loopback: a system proxy must not intercept it.
    assert request["trust_env"] is False
    assert request["timeout"] == registration.REGISTRATION_TIMEOUT


@pytest.mark.parametrize(
    ("body", "public", "tunnel"),
    [
        ({"host": "0.0.0.0", "tunnel": True}, True, True),
        ({"host": " localhost ", "tunnel": False}, False, False),
        ({"host": None, "tunnel": "yes"}, False, False),
        # An empty bind address listens on every interface.
        ({"host": " ", "tunnel": False}, True, False),
        ({}, False, False),
    ],
)
def test_register_folder_returns_server_exposure(
    monkeypatch, secret, tmp_path, body, public, tunnel
):
    _answer(monkeypatch, 200, {"workspace_dir": "x"} | body)

    answered = real_register(Workspace(tmp_path), url=URL)

    assert (answered.public_host is not None, answered.tunnel) == (public, tunnel)


@pytest.mark.parametrize(
    ("status", "body", "error", "match"),
    [
        (400, {"error": "folder_missing"}, RegistrationError, "cannot see"),
        (403, {"error": "local_secret_required"}, RegistrationError, "data folder"),
        (
            500,
            {"error": "setup_failed", "detail": "locked"},
            RegistrationError,
            "setup_failed: locked",
        ),
        (400, {"error": "invalid_path"}, RegistrationError, "invalid_path"),
    ],
)
def test_register_folder_reports_refusals(
    monkeypatch, secret, tmp_path, status, body, error, match
):
    _answer(monkeypatch, status, body)

    with pytest.raises(error, match=match):
        real_register(Workspace(tmp_path), url=URL)


def test_register_folder_missing_known_project_refuses(monkeypatch, secret, tmp_path):
    from EvoScientist.langgraph_dev.registry import Registry

    gone = Workspace(tmp_path / "gone")
    Registry().set_registered(gone.key, True)
    _answer(monkeypatch, 400, {"error": "folder_missing"})

    with pytest.raises(ProjectFolderMissingError, match="Restore"):
        real_register(gone, url=URL)


def test_register_folder_reports_failed_instance_check(monkeypatch, secret, tmp_path):
    """A server error is not reported as another user's server."""
    sent = _answer(monkeypatch, 200)
    monkeypatch.setattr(
        registration.httpx, "get", lambda _url, **_k: httpx.Response(500, text="boom")
    )

    with pytest.raises(RegistrationError, match="HTTP 500"):
        real_register(Workspace(tmp_path), url=URL)
    assert sent == []


def test_register_folder_server_without_instance_is_older(monkeypatch, tmp_path):
    """A server from before the local secret: no secret file, no route."""
    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path / "server")
    monkeypatch.setattr(
        registration.httpx, "get", lambda _url, **_k: httpx.Response(404)
    )

    with pytest.raises(OlderServerError):
        real_register(Workspace(tmp_path), url=URL)


def test_register_folder_existing_folder_server_cannot_see(
    monkeypatch, secret, tmp_path
):
    """A known project whose folder exists here, which the server cannot
    see, is not refused as missing."""
    from EvoScientist.langgraph_dev.registry import Registry

    project = Workspace(tmp_path)
    Registry().set_registered(project.key, True)
    _answer(monkeypatch, 400, {"error": "folder_missing"})

    with pytest.raises(RegistrationError, match="cannot see"):
        real_register(project, url=URL)


def test_register_folder_never_sends_secret_to_other_server(
    monkeypatch, secret, tmp_path
):
    """Another user's server on the port shows another instance."""
    sent = _answer(monkeypatch, 200, instance="theirs")

    with pytest.raises(RegistrationError, match="another user"):
        real_register(Workspace(tmp_path), url=URL)
    assert sent == []


def test_register_folder_reports_unreachable_server(monkeypatch, secret, tmp_path):
    def unreachable(_url, **_kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(registration.httpx, "get", unreachable)

    with pytest.raises(RegistrationError, match="refused"):
        real_register(Workspace(tmp_path), url=URL)


def test_register_folder_without_secret_names_data_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path / "server")
    sent = _answer(monkeypatch, 200)

    with pytest.raises(RegistrationError, match="EVOSCIENTIST_DATA_DIR"):
        real_register(Workspace(tmp_path), url=URL)
    assert sent == []


# ---------------------------------------------------------------------------
# Where clients register
# ---------------------------------------------------------------------------


@pytest.fixture
def reused_server(monkeypatch, runtime_paths):
    """``ensure_langgraph_dev`` finds a server already running."""
    from EvoScientist.langgraph_dev import manager

    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(
        manager,
        "_ensure_langgraph_dev_locked",
        lambda *_a, **_k: manager._Found(available=True),
    )
    monkeypatch.setattr(manager, "_ASYNC_SUBAGENTS_AVAILABLE", True)
    monkeypatch.setattr(manager, "needs_langgraph_dev", lambda *_a, **_k: True)
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    return cfg


def test_ensure_registers_project_after_reuse(monkeypatch, reused_server, tmp_path):
    from EvoScientist.langgraph_dev import manager

    registered = []
    monkeypatch.setattr(
        registration,
        "register_folder",
        lambda workspace, *, url: registered.append((workspace, url)),
    )

    manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)

    assert registered == [(Workspace(tmp_path), "http://127.0.0.1:6174")]
    assert manager.is_async_subagents_available()


def test_ensure_failed_registration_counts_as_server_unavailable(
    monkeypatch, reused_server, tmp_path, caplog
):
    """Every caller then takes its existing fallback (sub-agents run in
    process, the server backend's gateway falls back to the in-process one)."""
    from EvoScientist.langgraph_dev import manager

    def fail(_workspace, *, url):
        raise RegistrationError("setup failed")

    monkeypatch.setattr(registration, "register_folder", fail)

    manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)

    assert not manager.is_async_subagents_available()
    assert "setup failed" in caplog.text


def test_ensure_missing_folder_keeps_current_session_server(
    monkeypatch, reused_server, tmp_path
):
    """A session that stays where it is keeps the server serving its
    project."""
    from EvoScientist.langgraph_dev import manager

    def missing(workspace, *, url):
        raise ProjectFolderMissingError(workspace.root)

    monkeypatch.setattr(registration, "register_folder", missing)

    with pytest.raises(ProjectFolderMissingError):
        manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)
    assert manager.is_async_subagents_available()


def test_ensure_unusable_lock_is_failed_start(monkeypatch, reused_server, tmp_path):
    from EvoScientist.langgraph_dev import manager

    class _Lock:
        def __init__(self, *_a, **_k):
            pass

        def acquire(self):
            raise PermissionError("read-only")

    monkeypatch.setattr(manager, "FileLock", _Lock)

    assert manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path) is False
    assert manager.SERVER_UNAVAILABLE


@pytest.mark.parametrize("error", [PermissionError("pid file"), ValueError("bug")])
def test_ensure_start_error_leaves_no_server(
    monkeypatch, reused_server, tmp_path, error
):
    """The session must not keep the server it had before the call."""
    from EvoScientist.langgraph_dev import manager

    def broken(*_a, **_k):
        raise error

    monkeypatch.setattr(manager, "_ensure_langgraph_dev_locked", broken)

    with pytest.raises(type(error)):
        manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)
    assert not manager.is_async_subagents_available()


def test_ensure_older_server_serves_no_project(monkeypatch, reused_server, tmp_path):
    """The client stops, or a session that stays where it is goes on
    without the server."""
    from EvoScientist.langgraph_dev import manager

    def older(_workspace, *, url):
        raise OlderServerError()

    monkeypatch.setattr(registration, "register_folder", older)

    with pytest.raises(OlderServerError):
        manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)
    assert not manager.is_async_subagents_available()


def test_cli_refuses_known_project_with_missing_folder(tmp_path):
    import typer

    from EvoScientist.cli.commands import _prepare_workdir_or_exit
    from EvoScientist.langgraph_dev.registry import Registry

    gone = Workspace(tmp_path / "gone")
    Registry().set_registered(gone.key, True)

    with pytest.raises(typer.Exit):
        _prepare_workdir_or_exit(gone.root)


def test_cli_refusal_prints_path_with_brackets(monkeypatch, tmp_path):
    """A segment like ``[lab]`` is part of the path, not markup."""
    import typer

    from EvoScientist.cli import commands
    from EvoScientist.langgraph_dev.registry import Registry

    gone = Workspace(tmp_path / "[lab]" / "gone")
    Registry().set_registered(gone.key, True)

    with commands.console.capture() as captured, pytest.raises(typer.Exit):
        commands._prepare_workdir_or_exit(gone.root)

    assert "[lab]" in captured.get()


async def test_session_move_refuses_known_project_with_missing_folder(tmp_path):
    """``/resume`` into another project, whatever the backend."""
    from EvoScientist.cli.commands import _register_or_keep_session
    from EvoScientist.langgraph_dev.registry import Registry
    from EvoScientist.paths import SessionDirs

    gone = Workspace(tmp_path / "gone")
    Registry().set_registered(gone.key, True)
    here = SessionDirs(Workspace(tmp_path))

    with pytest.raises(ProjectFolderMissingError):
        await _register_or_keep_session(
            object(), target=SessionDirs(gone), current=here
        )
    assert not gone.root.exists()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (OlderServerError(), "server_outdated"),
        (RegistrationError("setup failed"), "registration_failed"),
    ],
)
def test_launch_registration_failure_raises_launcher_code(monkeypatch, error, code):
    from EvoScientist.deploy.launcher import LauncherError, register_launch_folder

    def fail(_workspace, *, url):
        raise error

    monkeypatch.setattr(registration, "register_folder", fail)

    with pytest.raises(LauncherError) as raised:
        register_launch_folder("/project", url=URL)
    assert raised.value.code == code


def test_launch_registration_missing_folder_names_workspace(monkeypatch, tmp_path):
    from EvoScientist.deploy.launcher import LauncherError, register_launch_folder

    def missing(workspace, *, url):
        raise ProjectFolderMissingError(workspace.root)

    monkeypatch.setattr(registration, "register_folder", missing)

    with pytest.raises(LauncherError) as raised:
        register_launch_folder(str(tmp_path), url=URL)
    assert raised.value.code == "folder_missing"
    assert raised.value.fields == {"workspace": str(tmp_path)}


def test_json_mode_reports_launcher_registration_codes():
    from EvoScientist.deploy.json_mode import _Failure, _launcher_failures
    from EvoScientist.deploy.launcher import LauncherError

    with pytest.raises(_Failure) as raised, _launcher_failures():
        raise LauncherError("server_outdated", "stop it")
    assert raised.value.code == "server_outdated"


async def test_session_move_reports_folder_it_cannot_create(monkeypatch, tmp_path):
    """Not a server failure: the session cannot work there."""
    from EvoScientist.cli import commands
    from EvoScientist.paths import SessionDirs

    def unwritable(_path):
        raise PermissionError("read-only")

    monkeypatch.setattr(registration, "prepare_workdir", unwritable)

    with pytest.raises(PermissionError):
        await commands._register_or_keep_session(
            object(),
            target=SessionDirs(Workspace(tmp_path / "other")),
            current=SessionDirs(Workspace(tmp_path)),
        )


def test_ensure_settles_registered_server_exposure(
    monkeypatch, reused_server, tmp_path
):
    from EvoScientist.langgraph_dev import manager

    exposure = registration.ServerExposure.of("0.0.0.0", True)
    monkeypatch.setattr(
        registration, "register_folder", lambda _workspace, *, url: exposure
    )

    manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)
    assert manager.SERVER_EXPOSURE == exposure

    def fail(_workspace, *, url):
        raise RegistrationError("setup failed")

    monkeypatch.setattr(registration, "register_folder", fail)
    manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)
    assert manager.SERVER_EXPOSURE is None


def test_tui_notices_name_exposed_server(monkeypatch):
    """The terminal, where the CLI prints it, is hidden while the TUI runs."""
    from EvoScientist.cli.tui_interactive import _agent_shell_notices
    from EvoScientist.langgraph_dev import manager

    exposure = registration.ServerExposure.of("0.0.0.0", True)
    monkeypatch.setattr(manager, "SERVER_EXPOSURE", exposure)
    expected = registration.exposure_messages(exposure)
    assert len(expected) == 2
    assert set(expected) <= set(_agent_shell_notices())


@pytest.mark.parametrize(
    ("flagged", "backend", "says"),
    [
        (True, "local", "applies to this session's own agent"),
        (True, "langgraph_server", "only turns approvals off"),
        (False, "local", None),
    ],
)
def test_dangerous_flag_note_says_where_it_applies(monkeypatch, flagged, backend, says):
    """The background server's agents follow config.yaml; the TUI, whose
    terminal is hidden, shows the note as well."""
    from EvoScientist.cli import commands
    from EvoScientist.cli.tui_interactive import _agent_shell_notices
    from EvoScientist.config import settings

    if flagged:
        settings.record_env_origin("EVOSCIENTIST_DANGEROUS_MODE", "cli")

    note = commands.dangerous_flag_note(backend)
    notices = _agent_shell_notices(backend)

    if says is None:
        assert note is None
    else:
        assert says in note
        assert note in notices


def test_dangerous_flag_note_for_webui():
    """The WebUI's runs carry none of this session's approval settings."""
    from EvoScientist.cli import commands
    from EvoScientist.config import settings

    settings.record_env_origin("EVOSCIENTIST_DANGEROUS_MODE", "cli")
    note = commands.dangerous_flag_note("langgraph_server", webui=True)
    assert "has no effect here" in note


def test_tui_notices_name_config_drift(monkeypatch):
    """The terminal, where the CLI prints it, is hidden while the TUI runs."""
    from EvoScientist.cli.tui_interactive import _agent_shell_notices
    from EvoScientist.langgraph_dev import manager

    monkeypatch.setattr(manager, "CONFIG_DRIFT_SINCE_LAUNCH", True)
    assert manager.CONFIG_DRIFT_MESSAGE in _agent_shell_notices()
    monkeypatch.setattr(manager, "CONFIG_DRIFT_SINCE_LAUNCH", False)
    assert manager.CONFIG_DRIFT_MESSAGE not in _agent_shell_notices()


def test_tui_notices_name_unavailable_server(monkeypatch):
    """The terminal is hidden while the TUI runs."""
    from EvoScientist.cli.tui_interactive import _agent_shell_notices
    from EvoScientist.langgraph_dev import manager

    monkeypatch.setattr(manager, "SERVER_UNAVAILABLE", True)
    assert manager.SERVER_UNAVAILABLE_MESSAGE in _agent_shell_notices()
    monkeypatch.setattr(manager, "SERVER_UNAVAILABLE", False)
    assert manager.SERVER_UNAVAILABLE_MESSAGE not in _agent_shell_notices()


def test_ensure_unexpected_registration_error_leaves_no_server(
    monkeypatch, reused_server, tmp_path
):
    from EvoScientist.langgraph_dev import manager

    def broken(_workspace, *, url):
        raise ValueError("bug")

    monkeypatch.setattr(registration, "register_folder", broken)

    with pytest.raises(ValueError, match="bug"):
        manager.ensure_langgraph_dev(reused_server, workspace_dir=tmp_path)
    assert not manager.is_async_subagents_available()
