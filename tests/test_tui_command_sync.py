"""Tests for TUI command-completion state sync."""

from types import SimpleNamespace

import pytest

from EvoScientist.commands.base import CommandContext
from tests.conftest import run_async as _run

pytest.importorskip("textual")


class _Loader:
    def __init__(self) -> None:
        self.adopt_calls: list[object] = []

    def adopt(self, agent: object) -> None:
        self.adopt_calls.append(agent)


class _StubApp:
    def __init__(self) -> None:
        self._agent_loader = _Loader()
        self._conversation_tid = "thread-1"
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
    monkeypatch.setattr(tui_mod._ch_mod, "_cli_agent", "old-agent", raising=False)
    monkeypatch.setattr(tui_mod._ch_mod, "_cli_thread_id", "old-thread", raising=False)

    _run(tui_mod._sync_tui_command_completion(app, ctx, "old-agent", cmd))

    assert app._agent_loader.adopt_calls == ["new-agent"]
    assert app.model_updates == [("gpt-5.5", "openai")]
    assert app.refresh_calls == [True]
    assert tui_mod._ch_mod._cli_agent == "new-agent"
    assert tui_mod._ch_mod._cli_thread_id == "thread-1"


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
