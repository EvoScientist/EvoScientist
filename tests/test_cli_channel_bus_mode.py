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
