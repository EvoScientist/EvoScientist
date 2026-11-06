"""``EvoSci workspace move`` and its client call: the counts it prints, and
the server's refusals it passes on."""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from EvoScientist import local_secret, paths
from EvoScientist.cli import commands  # noqa: F401 - registers the commands
from EvoScientist.cli._app import app
from EvoScientist.langgraph_dev import manager, registration
from EvoScientist.paths import Workspace


@pytest.fixture
def folders(tmp_path):
    new = tmp_path / "new"
    new.mkdir()
    return tmp_path / "old", new


@pytest.fixture
def server(monkeypatch):
    """A server that is up; the moves asked of it, and its answer."""
    asked: list = []
    answer: dict = {"report": {}, "error": None}

    monkeypatch.setattr(
        manager,
        "serve_project",
        lambda config, workspace, **_k: manager.ServerResult(
            available=True, url="http://127.0.0.1:6174"
        ),
    )

    def move(old, new, *, url):
        asked.append((old.key, new.key, url))
        if answer["error"] is not None:
            raise registration.RegistrationError(answer["error"])
        return answer["report"]

    monkeypatch.setattr(registration, "move_folder", move)
    return asked, answer


def test_workspace_move_prints_what_moved(folders, server):
    old, new = folders
    asked, answer = server
    answer["report"] = {"threads": 3, "tasks": 1, "memory": ["P-1"], "proposals": 0}

    result = CliRunner().invoke(app, ["workspace", "move", str(old), str(new)])

    assert result.exit_code == 0, result.output
    assert asked == [(Workspace(old).key, Workspace(new).key, "http://127.0.0.1:6174")]
    assert "3 conversation(s)" in result.output
    assert "1 scheduled task(s)" in result.output


def test_workspace_move_passes_refusal_on(folders, server):
    old, new = folders
    _asked, answer = server
    answer["error"] = "it still exists, so this is a copy"

    result = CliRunner().invoke(app, ["workspace", "move", str(old), str(new)])

    assert result.exit_code == 1
    assert "copy" in result.output


def test_workspace_move_needs_new_folder(tmp_path, server):
    asked, _answer = server

    result = CliRunner().invoke(
        app, ["workspace", "move", str(tmp_path / "old"), str(tmp_path / "nowhere")]
    )

    assert result.exit_code == 1
    assert asked == []


@pytest.fixture
def own_server(monkeypatch, tmp_path):
    """The server at the address is this data folder's: it gets the secret."""
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

    def answer(status, body):
        monkeypatch.setattr(
            registration.httpx,
            "post",
            lambda url, **_k: httpx.Response(status, json=body),
        )

    return answer


def test_move_folder_returns_report(own_server, folders):
    own_server(200, {"threads": 2})
    old, new = folders

    report = registration.move_folder(
        Workspace(old), Workspace(new), url="http://127.0.0.1:6174"
    )

    assert report == {"threads": 2}


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (
            409,
            {"error": "old_exists", "detail": "a copy"},
            registration.RegistrationError,
        ),
        (404, {"detail": "Not Found"}, registration.OlderServerError),
        (500, {"error": "move_failed"}, registration.RegistrationError),
    ],
    ids=["refused", "older_server", "failed"],
)
def test_move_folder_raises_on_refusal(own_server, folders, status, body, error):
    own_server(status, body)
    old, new = folders

    with pytest.raises(error):
        registration.move_folder(
            Workspace(old), Workspace(new), url="http://127.0.0.1:6174"
        )


def test_move_folder_reads_404_with_error_code_as_refusal(own_server, folders):
    own_server(404, {"error": "not_found", "detail": "no such thing"})
    old, new = folders

    with pytest.raises(registration.RegistrationError):
        registration.move_folder(
            Workspace(old), Workspace(new), url="http://127.0.0.1:6174"
        )


def test_workspace_move_prints_finished_move_and_warnings(folders, server):
    old, new = folders
    _asked, answer = server
    answer["report"] = {"resumed": True, "warnings": ["default_workdir still names"]}

    result = CliRunner().invoke(app, ["workspace", "move", str(old), str(new)])

    assert result.exit_code == 0, result.output
    assert "Finished moving" in result.output
    assert "default_workdir still names" in result.output
