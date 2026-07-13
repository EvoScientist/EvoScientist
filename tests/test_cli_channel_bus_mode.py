"""Tests for channel bus-mode thinking propagation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from EvoScientist.cli import channel as channel_cli


@pytest.fixture(autouse=True)
def _restore_channel_globals():
    """Restore the bus-mode globals after each test."""
    original = {
        "_manager": channel_cli._manager,
        "_bus_loop": channel_cli._bus_loop,
        "_bus_thread": channel_cli._bus_thread,
    }
    yield
    channel_cli._manager = original["_manager"]
    channel_cli._bus_loop = original["_bus_loop"]
    channel_cli._bus_thread = original["_bus_thread"]


def test_auto_start_channel_passes_send_thinking(monkeypatch):
    from EvoScientist.commands.base import ChannelRuntime

    captured = {}

    def _fake_start(config, agent, thread_id, *, send_thinking=None):
        captured["send_thinking"] = send_thinking
        captured["thread_id"] = thread_id
        captured["agent"] = agent
        return [("telegram", True, "connected (bus)")]

    monkeypatch.setattr(channel_cli, "_start_channels_bus_mode", _fake_start)
    monkeypatch.setattr(channel_cli, "_print_channel_panel", lambda _rows: None)

    config = SimpleNamespace(channel_enabled="telegram")
    agent = object()
    runtime = ChannelRuntime()
    channel_cli._auto_start_channel(
        agent,
        "thread-1",
        config,
        send_thinking=False,
        runtime=runtime,
    )

    assert captured["send_thinking"] is False
    assert captured["thread_id"] == "thread-1"
    assert captured["agent"] is agent
    assert runtime.agent is agent
    assert runtime.thread_id == "thread-1"


def test_auto_start_channel_reports_startup_failure(monkeypatch):
    from EvoScientist.commands.base import ChannelRuntime

    rows = [("telegram", False, "failed: dependency missing")]
    rendered = []
    monkeypatch.setattr(
        channel_cli,
        "_start_channels_bus_mode",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(channel_cli, "_print_channel_panel", rendered.append)
    runtime = ChannelRuntime()

    result = channel_cli._auto_start_channel(
        object(),
        "thread-1",
        SimpleNamespace(channel_enabled="telegram"),
        runtime=runtime,
    )

    assert result == rows
    assert rendered == [rows]
    assert runtime.agent is None
    assert runtime.thread_id is None


def test_auto_start_channel_binds_runtime_while_starting(monkeypatch):
    from EvoScientist.channels.channel_manager import CHANNEL_STARTUP_PENDING_DETAIL
    from EvoScientist.commands.base import ChannelRuntime

    rows = [("telegram", False, CHANNEL_STARTUP_PENDING_DETAIL)]
    monkeypatch.setattr(
        channel_cli,
        "_start_channels_bus_mode",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(channel_cli, "_print_channel_panel", lambda _rows: None)
    agent = object()
    runtime = ChannelRuntime()

    result = channel_cli._auto_start_channel(
        agent,
        "thread-1",
        SimpleNamespace(channel_enabled="telegram"),
        runtime=runtime,
    )

    assert result == rows
    assert runtime.agent is agent
    assert runtime.thread_id == "thread-1"


def test_get_channel_startup_results_without_manager():
    channel_cli._manager = None

    assert channel_cli.get_channel_startup_results() == []


def test_get_channel_startup_results_uses_manager_snapshot():
    rows = [("telegram", True, "connected (bus)")]

    class Manager:
        def startup_results(self):
            return rows

    channel_cli._manager = Manager()

    assert channel_cli.get_channel_startup_results() is rows
