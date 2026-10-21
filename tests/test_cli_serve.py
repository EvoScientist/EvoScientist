"""Tests for CLI serve/channel glue behavior."""

from __future__ import annotations

import os
import signal

from EvoScientist.cli import commands
from EvoScientist.paths import SessionDirs, Workspace
from tests.fakes.serve import run_serve_once, serve_config


def test_serve_workdir_has_highest_priority(monkeypatch, tmp_path):
    cfg_ws = tmp_path / "cfg_ws"
    cli_ws = tmp_path / "cli_ws"
    config = serve_config(default_workdir=str(cfg_ws), channel_send_thinking=True)

    order, captured = run_serve_once(
        monkeypatch,
        config,
        workdir=str(cli_ws),
    )

    expected = str(cli_ws.resolve())
    assert captured["work_dir"] is None
    assert captured["workspace"] == Workspace(expected)
    assert captured["dirs"] == SessionDirs(Workspace(expected))
    assert captured["media_dir"] == Workspace(expected).media_dir
    assert cli_ws.is_dir()
    # The project .env can override the memories folder; it must be re-read
    # before ensure_dirs() creates it.
    assert order.index(("reload_env_dirs", None)) < order.index(("ensure_dirs", None))


def test_serve_uses_config_default_workdir_when_no_cli_workdir(monkeypatch, tmp_path):
    cfg_ws = tmp_path / "cfg_ws"
    config = serve_config(default_workdir=str(cfg_ws), channel_send_thinking=True)

    _, captured = run_serve_once(monkeypatch, config)

    expected = str(cfg_ws.resolve())
    assert captured["work_dir"] is None
    assert captured["workspace"] == Workspace(expected)
    assert captured["dirs"] == SessionDirs(Workspace(expected))


def test_serve_uses_cwd_when_no_workdir_config(monkeypatch, tmp_path):
    cwd = str(tmp_path.resolve())
    config = serve_config(default_workdir="", channel_send_thinking=True)

    _, captured = run_serve_once(
        monkeypatch,
        config,
        cwd=cwd,
    )

    assert captured["work_dir"] is None
    assert captured["workspace"] == Workspace(cwd)
    assert captured["dirs"] == SessionDirs(Workspace(cwd))


def test_serve_channel_thinking_respects_config_and_no_thinking(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())

    _, captured_default = run_serve_once(
        monkeypatch,
        serve_config(default_workdir=ws, channel_send_thinking=True),
    )
    assert captured_default["send_thinking"] is True

    _, captured_cfg_off = run_serve_once(
        monkeypatch,
        serve_config(default_workdir=ws, channel_send_thinking=False),
    )
    assert captured_cfg_off["send_thinking"] is False

    _, captured_cli_off = run_serve_once(
        monkeypatch,
        serve_config(default_workdir=ws, channel_send_thinking=True),
        no_thinking=True,
    )
    assert captured_cli_off["send_thinking"] is False


def test_serve_debug_sets_log_level_and_channel_trace(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws, channel_send_thinking=True)
    configure_calls: list[tuple[str | None, str | None]] = []
    monkeypatch.delenv("EVOSCIENTIST_LOG_LEVEL", raising=False)
    monkeypatch.delenv("EVOSCIENTIST_CHANNEL_DEBUG_TRACING", raising=False)

    monkeypatch.setattr(
        commands,
        "_configure_logging",
        lambda: configure_calls.append(
            (
                os.environ.get("EVOSCIENTIST_LOG_LEVEL"),
                os.environ.get("EVOSCIENTIST_CHANNEL_DEBUG_TRACING"),
            )
        ),
    )

    _, captured = run_serve_once(
        monkeypatch,
        config,
        workdir=ws,
        debug=True,
    )

    assert captured["cli_overrides"] == {
        "log_level": "DEBUG",
        "channel_debug_tracing": True,
    }
    assert configure_calls == [("DEBUG", "true")]


def test_serve_host_flag_narrows_or_widens_backend_bind(monkeypatch, tmp_path):
    """`EvoSci serve --host` must reach langgraph_dev_host — the flag was
    silently dropped before, leaving the backend bind config-only."""
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws)

    _, captured = run_serve_once(monkeypatch, config, workdir=ws, host="0.0.0.0")

    assert captured["cli_overrides"] == {"langgraph_dev_host": "0.0.0.0"}
    # The effective config carrying the override must be the one handed to the
    # backend launcher — not just recorded in the overrides dict.
    assert captured["ensure_config"].langgraph_dev_host == "0.0.0.0"


def test_serve_host_flag_is_stripped(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws)

    _, captured = run_serve_once(
        monkeypatch, config, workdir=ws, host="  192.168.1.5  "
    )

    assert captured["cli_overrides"] == {"langgraph_dev_host": "192.168.1.5"}


def test_serve_blank_host_flag_leaves_config_defaults(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws)

    _, captured = run_serve_once(monkeypatch, config, workdir=ws, host="   ")

    assert captured["cli_overrides"] == {}


def test_serve_auto_approve_only_sets_auto_approve(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws, enable_ask_user=True)

    _, captured = run_serve_once(
        monkeypatch,
        config,
        workdir=ws,
        auto_approve=True,
    )

    assert captured["cli_overrides"] == {"auto_approve": True}


def test_serve_auto_mode_implies_auto_approve_and_disables_ask_user(
    monkeypatch, tmp_path
):
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws, enable_ask_user=True)

    _, captured = run_serve_once(
        monkeypatch,
        config,
        workdir=ws,
        auto_mode=True,
        ask_user=True,
    )

    assert captured["cli_overrides"] == {
        "auto_mode": True,
        "auto_approve": True,
        "enable_ask_user": False,
    }


def test_serve_dangerous_sets_dangerous_mode(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())
    config = serve_config(default_workdir=ws)

    _, captured = run_serve_once(
        monkeypatch,
        config,
        workdir=ws,
        dangerous=True,
    )

    assert captured["cli_overrides"] == {"dangerous_mode": True}


def test_serve_sigterm_cancels_active_message_scope_before_shutdown(
    monkeypatch, tmp_path
):
    handlers = {}
    message = commands.ChannelMessage(
        msg_id="message-1",
        content="run a long command",
        sender="user",
        channel_type="telegram",
        chat_id="chat-1",
    )

    class _OneMessageQueue:
        def get(self, timeout=None):
            return message

    def _fake_signal(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    cancelled_scopes = []

    def _fake_process_message(*args, **kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(
        "EvoScientist.stream.display.request_stream_cancel",
        cancelled_scopes.append,
    )

    run_serve_once(
        monkeypatch,
        serve_config(default_workdir=str(tmp_path)),
        message_queue=_OneMessageQueue(),
        process_message=_fake_process_message,
    )

    assert cancelled_scopes == ["channel:telegram:chat-1:message-1"]


def test_serve_sigint_cancels_active_message_scope_before_interrupt(
    monkeypatch, tmp_path
):
    handlers = {}
    message = commands.ChannelMessage(
        msg_id="message-2",
        content="run a long command",
        sender="user",
        channel_type="telegram",
        chat_id="chat-2",
    )

    class _OneMessageQueue:
        def get(self, timeout=None):
            return message

    def _fake_signal(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    cancelled_scopes = []

    def _fake_process_message(*args, **kwargs):
        handlers[signal.SIGINT](signal.SIGINT, None)

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(
        "EvoScientist.stream.display.request_stream_cancel",
        cancelled_scopes.append,
    )

    run_serve_once(
        monkeypatch,
        serve_config(default_workdir=str(tmp_path)),
        message_queue=_OneMessageQueue(),
        process_message=_fake_process_message,
    )

    assert cancelled_scopes == ["channel:telegram:chat-2:message-2"]
