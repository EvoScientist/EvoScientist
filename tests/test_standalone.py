"""Tests for the headless standalone channel runner's dev-server startup."""

from __future__ import annotations

import os
from types import SimpleNamespace

import EvoScientist.channels.standalone as standalone
from EvoScientist.paths import Workspace


def _patch_manager(monkeypatch, *, gateway_backend, default_workdir):
    """Stub the dev-server spawn + workspace helpers; return (config, ensure_calls).

    ``ensure_calls`` captures each ``ensure_langgraph_dev`` invocation so a test
    can assert whether (and with what workspace) the dev server was ensured.
    """
    import EvoScientist.langgraph_dev.manager as manager_mod
    import EvoScientist.paths as paths_mod

    ensure_calls: list[dict] = []
    monkeypatch.setattr(
        manager_mod,
        "ensure_langgraph_dev",
        lambda cfg, *, workspace_dir, backend=None: ensure_calls.append(
            {"config": cfg, "workspace_dir": workspace_dir, "backend": backend}
        ),
    )
    # Avoid creating the real data dirs.
    monkeypatch.setattr(paths_mod, "ensure_dirs", lambda: None)
    config = SimpleNamespace(
        gateway_backend=gateway_backend,
        default_workdir=default_workdir,
    )
    return config, ensure_calls


def test_ensure_dev_server_spawns_on_server_backend(monkeypatch, tmp_path):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )

    standalone._ensure_standalone_dev_server(
        config, workspace_dir=str(tmp_path), backend="langgraph_server"
    )

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == str(tmp_path)
    # The resolved backend is forwarded so the manager spawns in full mode.
    assert ensure_calls[0]["backend"] == "langgraph_server"


def test_ensure_dev_server_noop_on_local_backend(monkeypatch, tmp_path):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="local", default_workdir=str(tmp_path)
    )

    standalone._ensure_standalone_dev_server(
        config, workspace_dir=str(tmp_path), backend="local"
    )

    assert ensure_calls == []


def test_run_standalone_dev_server_falls_back_to_cwd(monkeypatch):
    import EvoScientist.config as config_mod

    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=""
    )
    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == os.getcwd()


def test_run_standalone_uses_default_workdir(monkeypatch, tmp_path):
    import EvoScientist.config as config_mod

    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )
    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    seen: dict[str, object] = {}

    def _fake_async_main(*args, workspace, **kwargs):
        seen["workspace"] = workspace
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(standalone, "_async_main", _fake_async_main)
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert ensure_calls[0]["workspace_dir"] == str(Workspace(tmp_path).root)
    assert seen["workspace"] == Workspace(tmp_path)


def test_run_standalone_ensures_dev_server_only_with_agent(monkeypatch):
    """``run_standalone`` resolves config and ensures the dev server iff use_agent."""
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(gateway_backend="local", default_workdir=""),
    )
    ensure_configs: list[object] = []
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda cfg, *, workspace_dir, backend=None: ensure_configs.append(cfg),
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)
    assert len(ensure_configs) == 1

    ensure_configs.clear()
    standalone.run_standalone(channel=None, bus=None, use_agent=False)
    assert ensure_configs == []


import asyncio
import signal
import threading
import time

import pytest

from EvoScientist.channels.bus.message_bus import MessageBus
from tests.fakes import QueueFakeChannel


class _SigintChannel(QueueFakeChannel):
    """Fake channel that schedules SIGINT once its worker is running."""

    def __init__(self, *, second_sigint_at: float | None = None, hang_stop: bool = False):
        super().__init__()
        self._second_sigint_at = second_sigint_at
        self._hang_stop = hang_stop
        self._stop_release: asyncio.Event | None = None

    async def run(self) -> None:
        threading.Timer(0.1, signal.raise_signal, [signal.SIGINT]).start()
        if self._second_sigint_at is not None:
            threading.Timer(
                self._second_sigint_at, signal.raise_signal, [signal.SIGINT]
            ).start()
        await asyncio.Future()

    async def stop(self) -> None:
        if self._hang_stop:
            # Simulate a stop step that blocks (e.g. consumer drain wait).
            self._stop_release = asyncio.Event()
            await self._stop_release.wait()
        await super().stop()


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_run_standalone_exits_on_sigint():
    """Ctrl+C runs the graceful path and asyncio.run returns (issue #565)."""
    channel = _SigintChannel()
    bus = MessageBus()

    started = time.monotonic()
    standalone.run_standalone(channel, bus, use_agent=False)
    assert time.monotonic() - started < 20
    assert channel._stopped


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_second_sigint_forces_shutdown():
    """A second signal cancels a blocked graceful path instead of no-op."""
    channel = _SigintChannel(second_sigint_at=0.3, hang_stop=True)
    bus = MessageBus()

    started = time.monotonic()
    standalone.run_standalone(channel, bus, use_agent=False)
    # Forced exit: the hanging stop() never completed.
    assert time.monotonic() - started < 20
    assert not channel._stopped
