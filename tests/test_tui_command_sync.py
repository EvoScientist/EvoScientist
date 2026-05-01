"""Tests for TUI command-completion state sync."""

from types import SimpleNamespace

import pytest

from EvoScientist.commands.base import ChannelRuntime, CommandContext
from tests.conftest import run_async as _run

pytest.importorskip("textual")


class _Loader:
    def __init__(self) -> None:
        self.adopt_calls: list[object] = []
        self.agent: object | None = None

    def adopt(self, agent: object) -> None:
        self.adopt_calls.append(agent)
        self.agent = agent


class _StubApp:
    def __init__(self) -> None:
        self._agent_loader = _Loader()
        self._conversation_tid = "thread-1"
        self._channel_runtime = ChannelRuntime()
        self.model_updates: list[tuple[str, str | None]] = []
        self.refresh_calls: list[bool] = []

    def update_status_after_model_change(
        self,
        new_model: str,
        new_provider: str | None = None,
    ) -> None:
        self.model_updates.append((new_model, new_provider))

    async def _refresh_status_snapshot(
        self,
        *,
        reset_streaming_text: bool = True,
    ) -> None:
        self.refresh_calls.append(reset_streaming_text)


def test_sync_tui_command_completion_adopts_agent_swap(monkeypatch):
    import EvoScientist.cli.tui_interactive as tui_mod
    from EvoScientist import EvoScientist as evosci_mod

    app = _StubApp()
    ctx = CommandContext(
        agent="new-agent",
        thread_id="thread-1",
        ui=SimpleNamespace(),
    )
    cmd = SimpleNamespace(name="/model")

    monkeypatch.setattr(
        evosci_mod,
        "_ensure_config",
        lambda: SimpleNamespace(model="gpt-5.5", provider="openai"),
    )
    monkeypatch.setattr(tui_mod, "_channels_is_running", lambda: True)
    app._channel_runtime.bind("old-agent", "old-thread")

    _run(tui_mod._sync_tui_command_completion(app, ctx, "old-agent", cmd))

    assert app._agent_loader.adopt_calls == ["new-agent"]
    assert app.model_updates == [("gpt-5.5", "openai")]
    assert app.refresh_calls == [True]
    assert app._channel_runtime.agent == "new-agent"
    assert app._channel_runtime.thread_id == "thread-1"


def test_sync_tui_command_completion_refreshes_without_agent_swap(monkeypatch):
    import EvoScientist.cli.tui_interactive as tui_mod

    app = _StubApp()
    ctx = CommandContext(
        agent="same-agent",
        thread_id="thread-1",
        ui=SimpleNamespace(),
    )
    cmd = SimpleNamespace(name="/compact")

    monkeypatch.setattr(tui_mod, "_channels_is_running", lambda: False)

    _run(tui_mod._sync_tui_command_completion(app, ctx, "same-agent", cmd))

    assert app._agent_loader.adopt_calls == []
    assert app.model_updates == []
    assert app.refresh_calls == [True]


def test_sync_tui_rebinds_runtime_on_thread_rotation_without_agent_swap(monkeypatch):
    """Regression: ``/new`` and ``/resume`` rotate ``app._conversation_tid``
    without swapping the agent.  The runtime must still pick up the new
    thread id so the bus contract stays consistent with serve mode."""
    import EvoScientist.cli.tui_interactive as tui_mod

    app = _StubApp()
    app._conversation_tid = "rotated-thread"
    app._agent_loader.agent = "same-agent"
    app._channel_runtime.bind("same-agent", "old-thread")
    ctx = CommandContext(
        agent="same-agent",
        thread_id="rotated-thread",
        ui=SimpleNamespace(),
    )
    cmd = SimpleNamespace(name="/new")

    monkeypatch.setattr(tui_mod, "_channels_is_running", lambda: True)

    _run(tui_mod._sync_tui_command_completion(app, ctx, "same-agent", cmd))

    assert app._channel_runtime.agent == "same-agent"
    assert app._channel_runtime.thread_id == "rotated-thread"
