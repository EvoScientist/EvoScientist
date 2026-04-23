"""Tests for channel bus-mode thinking propagation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from EvoScientist.cli import channel as channel_cli


@pytest.fixture(autouse=True)
def _restore_channel_globals():
    """Restore mutable module globals after each test."""
    original = {
        "_manager": channel_cli._manager,
        "_bus_loop": channel_cli._bus_loop,
        "_bus_thread": channel_cli._bus_thread,
        "_cli_agent": channel_cli._cli_agent,
        "_cli_thread_id": channel_cli._cli_thread_id,
    }
    yield
    channel_cli._manager = original["_manager"]
    channel_cli._bus_loop = original["_bus_loop"]
    channel_cli._bus_thread = original["_bus_thread"]
    channel_cli._cli_agent = original["_cli_agent"]
    channel_cli._cli_thread_id = original["_cli_thread_id"]


def test_auto_start_channel_passes_send_thinking(monkeypatch):
    captured = {}

    def _fake_start(config, agent, thread_id, *, send_thinking=None):
        captured["send_thinking"] = send_thinking
        captured["thread_id"] = thread_id
        captured["agent"] = agent

    monkeypatch.setattr(channel_cli, "_start_channels_bus_mode", _fake_start)
    monkeypatch.setattr(channel_cli, "_print_channel_panel", lambda _rows: None)

    config = SimpleNamespace(channel_enabled="telegram")
    agent = object()
    channel_cli._auto_start_channel(
        agent,
        "thread-1",
        config,
        send_thinking=False,
    )

    assert captured["send_thinking"] is False
    assert captured["thread_id"] == "thread-1"
    assert captured["agent"] is agent
