"""Tests for ``EvoSci login chatgpt`` / ``EvoSci logout chatgpt`` and their JSON mode."""

from __future__ import annotations

import functools
import io
import json
import logging
import os
import socket
import sys
import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import typer
from langchain_openai.chatgpt_oauth import _ChatGPTToken, _FileChatGPTOAuthTokenProvider
from typer.testing import CliRunner

import EvoScientist.cli.commands  # noqa: F401  registers the commands
from EvoScientist.cli import login_cmd
from EvoScientist.cli._app import app
from EvoScientist.llm import chatgpt_oauth
from EvoScientist.setup.protocol import JsonEmitter

URL = "https://auth.openai.com/oauth/authorize?state=s1"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("EvoScientist.paths.DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def signed_in(data_dir):
    path = chatgpt_oauth.token_path()
    path.parent.mkdir(parents=True)
    token = _ChatGPTToken(
        access_token="tok-access",
        refresh_token="tok-refresh",
        expires_at=datetime.now(UTC) + timedelta(days=1),
        account_id="acct_1",
        plan_type="plus",
    )
    _FileChatGPTOAuthTokenProvider(path=path).save(token)
    return path


@pytest.fixture
def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _events(fn, *args):
    out = io.StringIO()
    code = fn(JsonEmitter(out), *args)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def _fake_sign_in(*, url=URL, plan="plus", error=None, before_return=None):
    def _sign_in(*, on_url, stop=None, timeout=None):
        on_url(url)
        if error is not None:
            raise error
        if before_return is not None:
            before_return()
        return chatgpt_oauth.ChatGPTLogin(signed_in=True, plan=plan)

    return _sign_in


def test_json_login_reports_url_then_signed_in(monkeypatch):
    monkeypatch.setattr(chatgpt_oauth, "sign_in", _fake_sign_in())

    code, events = _events(login_cmd.run_login_json, threading.Event())

    assert code == 0
    assert events == [
        {"protocol": 1, "event": "url", "url": URL},
        {"protocol": 1, "event": "signed_in", "plan": "plus"},
    ]


@pytest.mark.parametrize(
    "code_", ["port_in_use", "timed_out", "cancelled", "sign_in_failed"]
)
def test_json_login_errors_end_with_one_error_line(monkeypatch, code_):
    error = chatgpt_oauth.SignInError(code_, "msg")
    monkeypatch.setattr(chatgpt_oauth, "sign_in", _fake_sign_in(error=error))

    code, events = _events(login_cmd.run_login_json, threading.Event())

    assert code == 1
    assert events[-1] == {
        "protocol": 1,
        "event": "error",
        "code": code_,
        "message": "msg",
    }
    assert sum(e["event"] in ("signed_in", "error") for e in events) == 1


def test_json_login_unexpected_failure_is_internal_error(monkeypatch):
    error = ValueError("boom")
    monkeypatch.setattr(chatgpt_oauth, "sign_in", _fake_sign_in(error=error))

    code, events = _events(login_cmd.run_login_json, threading.Event())

    assert code == 1
    assert (events[-1]["event"], events[-1]["code"]) == ("error", "internal_error")


def test_json_login_network_failure_is_sign_in_failed(monkeypatch):
    error = httpx.ConnectError("down")
    monkeypatch.setattr(chatgpt_oauth, "sign_in", _fake_sign_in(error=error))

    code, events = _events(login_cmd.run_login_json, threading.Event())

    assert code == 1
    assert events[-1]["code"] == "sign_in_failed"
    assert events[-1]["message"] == chatgpt_oauth.SIGN_IN_INCOMPLETE


def test_json_login_ctrl_c_ends_with_cancelled(monkeypatch):
    monkeypatch.setattr(
        chatgpt_oauth, "sign_in", _fake_sign_in(error=KeyboardInterrupt())
    )

    try:
        code, events = _events(login_cmd.run_login_json, threading.Event())
    except KeyboardInterrupt:
        pytest.fail("Ctrl+C escaped without a terminal line")

    assert code == 1
    assert events[-1] == {
        "protocol": 1,
        "event": "error",
        "code": "cancelled",
        "message": chatgpt_oauth.SIGN_IN_CANCELLED,
    }


def test_json_command_watches_a_piped_stdin(monkeypatch, capsys):
    """The command itself starts the watcher: closing stdin cancels."""
    from EvoScientist.setup import protocol

    read, write = os.pipe()
    stdin = os.fdopen(read, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(protocol, "stdin_is_pipe", lambda: True)
    monkeypatch.setattr(
        "EvoScientist.stream.json_sink.redirect_console_to_stderr", lambda: None
    )

    def _sign_in(*, on_url, stop=None, timeout=None):
        on_url(URL)
        os.close(write)  # the program went away
        if stop is not None and stop.wait(5):
            raise chatgpt_oauth.SignInError(
                "cancelled", chatgpt_oauth.SIGN_IN_CANCELLED
            )
        return chatgpt_oauth.ChatGPTLogin(signed_in=True, plan="plus")

    monkeypatch.setattr(chatgpt_oauth, "sign_in", _sign_in)
    try:
        with pytest.raises(typer.Exit) as exit_info:
            login_cmd.login_chatgpt_command(json_output=True)
    finally:
        stdin.close()

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_info.value.exit_code == 1
    assert [e["event"] for e in events] == ["url", "error"]
    assert events[-1]["code"] == "cancelled"


def test_json_login_closed_stdin_reports_cancelled(data_dir, free_port, monkeypatch):
    from EvoScientist.setup.protocol import watch_stdin

    real_authorize = chatgpt_oauth._authorize
    monkeypatch.setattr(
        chatgpt_oauth, "_authorize", functools.partial(real_authorize, port=free_port)
    )
    read, write = os.pipe()
    stop = threading.Event()
    threading.Thread(target=watch_stdin, args=(read, stop), daemon=True).start()
    out = io.StringIO()
    inner = JsonEmitter(out)

    def emit(event):
        inner(event)
        if event["event"] == "url":
            os.close(write)  # the app went away while the browser was open

    code = login_cmd.run_login_json(emit, stop)

    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert code == 1
    assert [e["event"] for e in events] == ["url", "error"]
    assert events[-1]["code"] == "cancelled"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as again:
        again.bind(("127.0.0.1", free_port))


def test_json_logout(signed_in):
    assert _events(login_cmd.run_logout_json) == (
        0,
        [{"protocol": 1, "event": "signed_out"}],
    )
    assert not signed_in.exists()


def test_json_logout_without_a_sign_in(data_dir):
    assert _events(login_cmd.run_logout_json) == (
        0,
        [{"protocol": 1, "event": "signed_out"}],
    )


def test_json_stdout_is_only_json_lines(data_dir, monkeypatch):
    def _noisy():
        logging.getLogger("langchain_openai.chatgpt_oauth").error("upstream noise")

    monkeypatch.setattr(chatgpt_oauth, "sign_in", _fake_sign_in(before_return=_noisy))

    result = CliRunner().invoke(app, ["login", "chatgpt", "--json"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["url", "signed_in"]


def test_login_without_json_runs_the_terminal_sign_in(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "EvoScientist.config.onboard.helpers._run_chatgpt_login",
        lambda: calls.append(True) or False,
    )

    result = CliRunner().invoke(app, ["login", "chatgpt"])

    assert (calls, result.exit_code) == ([True], 1)


def test_logout_without_json_says_what_happened(signed_in):
    first = CliRunner().invoke(app, ["logout", "chatgpt"])
    second = CliRunner().invoke(app, ["logout", "chatgpt"])

    assert first.exit_code == 0
    assert "Signed out of ChatGPT." in first.stdout
    assert "Not signed in to ChatGPT." in second.stdout
