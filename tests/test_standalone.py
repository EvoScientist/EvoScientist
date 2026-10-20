"""Tests for the headless standalone channel runner's dev-server startup."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import EvoScientist.channels.standalone as standalone
from EvoScientist.channels.base import Channel
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.paths import Workspace
from tests.fakes import QueueFakeChannel


@pytest.fixture(autouse=True)
def _restore_http_log_levels():
    """Reset the process-global logger levels run_standalone sets after every test."""
    loggers = [logging.getLogger(n) for n in ("httpx", "httpcore", "telegram")]
    saved = [lg.level for lg in loggers]
    yield
    for lg, level in zip(loggers, saved, strict=False):
        lg.setLevel(level)


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


def _standalone_ready(monkeypatch, *, signed_in, **config_fields):
    """Stub everything ``run_standalone`` starts; return (ensure_calls, status_calls)."""
    import EvoScientist.config as config_mod
    from EvoScientist.llm.chatgpt_oauth import ChatGPTLogin

    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=""
    )
    for name, value in config_fields.items():
        setattr(config, name, value)
    status_calls: list[bool] = []

    def _status():
        status_calls.append(True)
        return ChatGPTLogin(signed_in=signed_in)

    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    monkeypatch.setattr("EvoScientist.llm.chatgpt_oauth.status", _status)
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())
    return ensure_calls, status_calls


def test_run_standalone_without_chatgpt_sign_in_exits_before_anything_starts(
    monkeypatch, caplog
):
    from EvoScientist.llm.chatgpt_oauth import SIGN_IN_HINT

    ensure_calls, _ = _standalone_ready(
        monkeypatch, signed_in=False, openai_auth_mode="oauth"
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
        standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert exc_info.value.code == 1
    assert [record.getMessage() for record in caplog.records] == [SIGN_IN_HINT]
    assert ensure_calls == []


def test_run_standalone_names_an_unreadable_token_before_the_hint(monkeypatch, caplog):
    from EvoScientist.llm.chatgpt_oauth import SIGN_IN_HINT, ChatGPTLogin

    _standalone_ready(monkeypatch, signed_in=False, openai_auth_mode="oauth")
    monkeypatch.setattr(
        "EvoScientist.llm.chatgpt_oauth.status",
        lambda: ChatGPTLogin(signed_in=False, problem="token file is unreadable"),
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert [record.getMessage() for record in caplog.records] == [
        "ChatGPT: token file is unreadable",
        SIGN_IN_HINT,
    ]


def test_run_standalone_starts_when_signed_in(monkeypatch):
    ensure_calls, status_calls = _standalone_ready(
        monkeypatch, signed_in=True, openai_auth_mode="oauth"
    )

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensure_calls) == 1
    assert status_calls == [True]


@pytest.mark.parametrize("fields", [{}, {"openai_auth_mode": "api_key"}])
def test_run_standalone_does_not_look_at_the_sign_in_in_api_key_mode(
    monkeypatch, fields
):
    ensure_calls, status_calls = _standalone_ready(
        monkeypatch, signed_in=False, **fields
    )

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensure_calls) == 1
    assert status_calls == []


class _SigintChannel(QueueFakeChannel):
    """Fake channel that schedules SIGINT once its worker is running."""

    def __init__(
        self, *, second_sigint_at: float | None = None, hang_stop: bool = False
    ):
        super().__init__()
        self._second_sigint_at = second_sigint_at
        self._hang_stop = hang_stop

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
    sys.platform == "win32", reason="Windows loops always take this path"
)
@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_run_standalone_exits_on_sigint_without_loop_signal_handlers(monkeypatch):
    """A loop without add_signal_handler (Windows) still stops on Ctrl+C."""
    from asyncio import unix_events

    def _unsupported(self, sig, callback, *args):
        raise NotImplementedError

    monkeypatch.setattr(
        unix_events._UnixSelectorEventLoop, "add_signal_handler", _unsupported
    )
    handler_before = signal.getsignal(signal.SIGINT)
    channel = _SigintChannel()

    started = time.monotonic()
    standalone.run_standalone(channel, MessageBus(), use_agent=False)

    assert time.monotonic() - started < 20
    assert channel._stopped
    assert signal.getsignal(signal.SIGINT) is handler_before


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


class _HangingCleanupChannel(QueueFakeChannel):
    """Keeps Channel.run() and Channel.stop(); its cleanup never finishes."""

    stop = Channel.stop

    def __init__(self) -> None:
        super().__init__()
        self.cleanups = 0

    async def receive(self):
        threading.Timer(0.1, signal.raise_signal, [signal.SIGINT]).start()
        threading.Timer(0.5, signal.raise_signal, [signal.SIGINT]).start()
        await asyncio.Future()
        yield  # pragma: no cover

    async def _cleanup(self) -> None:
        self.cleanups += 1
        await asyncio.Event().wait()


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_second_sigint_exits_when_channel_cleanup_hangs():
    """Cancelling the channel's worker must not start the hung cleanup again."""
    channel = _HangingCleanupChannel()

    started = time.monotonic()
    standalone.run_standalone(channel, MessageBus(), use_agent=False)

    assert time.monotonic() - started < 20
    assert channel.cleanups == 1


def test_run_standalone_quiets_httpx(monkeypatch):
    """HTTP client logs must not expose channel tokens in request URLs."""
    leaking_level = {
        "httpx": logging.INFO,
        "httpcore": logging.INFO,
        "telegram": logging.DEBUG,
    }
    for name in leaking_level:
        logging.getLogger(name).setLevel(logging.DEBUG)

    enabled_at_start: dict[str, bool] = {}

    def _run(coro):
        # The channel's first request happens inside asyncio.run.
        for name, level in leaking_level.items():
            enabled_at_start[name] = logging.getLogger(name).isEnabledFor(level)
        coro.close()

    monkeypatch.setattr(standalone.asyncio, "run", _run)

    standalone.run_standalone(channel=None, bus=None, use_agent=False)

    assert enabled_at_start == dict.fromkeys(leaking_level, False)
