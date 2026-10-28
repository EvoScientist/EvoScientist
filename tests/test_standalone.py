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
from unittest.mock import AsyncMock

import pytest

import EvoScientist.channels.standalone as standalone
from EvoScientist.channels.base import Channel, ChannelError
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.paths import Workspace
from tests.fakes.common import QueueFakeChannel


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


def test_ensure_dev_server_logs_server_exposure(monkeypatch, tmp_path, caplog):
    import EvoScientist.langgraph_dev.manager as manager_mod
    from EvoScientist.langgraph_dev.registration import (
        ServerExposure,
        exposure_messages,
    )

    config, _calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )
    exposure = ServerExposure.of("0.0.0.0", False)
    monkeypatch.setattr(manager_mod, "SERVER_EXPOSURE", exposure)

    with caplog.at_level("WARNING"):
        standalone._ensure_standalone_dev_server(
            config, workspace_dir=str(tmp_path), backend="langgraph_server"
        )

    assert set(exposure_messages(exposure)) <= set(caplog.messages)


def test_ensure_dev_server_unusable_server_exits(monkeypatch, tmp_path):
    import EvoScientist.langgraph_dev.manager as manager_mod
    from EvoScientist.langgraph_dev.registration import OlderServerError

    config, _calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )

    def older(_cfg, *, workspace_dir, backend=None):
        raise OlderServerError()

    monkeypatch.setattr(manager_mod, "ensure_langgraph_dev", older)

    with pytest.raises(SystemExit) as exited:
        standalone._ensure_standalone_dev_server(
            config, workspace_dir=str(tmp_path), backend="langgraph_server"
        )
    assert exited.value.code == 1


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


@pytest.fixture(autouse=True)
def _no_health_server(monkeypatch):
    """Keep the runner off 0.0.0.0:8080; a forced exit never stops the server."""
    from EvoScientist.channels.channel_manager import ChannelManager

    async def _noop(self):
        return None

    monkeypatch.setattr(ChannelManager, "start_health", _noop)


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


class _SameTickChannel(QueueFakeChannel):
    """Both signals land while a sync step holds the loop."""

    async def run(self) -> None:
        def _fire() -> None:
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)

        threading.Timer(0.1, _fire).start()
        time.sleep(0.3)
        await asyncio.Future()


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_two_signals_in_one_loop_pass_still_exit():
    """The second signal must not cancel the shutdown before it has started."""
    started = time.monotonic()
    standalone.run_standalone(_SameTickChannel(), MessageBus(), use_agent=False)

    assert time.monotonic() - started < 20


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


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_sigint_closes_the_code_interpreter_of_an_unfinished_turn(
    monkeypatch, tmp_path
):
    """A turn cut off by the signal never evicts its QuickJS slot."""
    from EvoScientist.middleware.code_interpreter import (
        create_code_interpreter_middleware,
    )

    interpreter = create_code_interpreter_middleware()
    interpreter._registry.get("unfinished-turn")

    async def _agent(workspace):
        return object()

    monkeypatch.setattr(standalone, "_create_standalone_agent", _agent)
    monkeypatch.setattr(
        "EvoScientist.gateway.create_runtime_gateways_for_config",
        lambda config, backend=None: SimpleNamespace(graph_gateway=None),
    )
    try:
        asyncio.run(
            standalone._async_main(
                _SigintChannel(),
                MessageBus(),
                use_agent=True,
                send_thinking=False,
                workspace=Workspace(tmp_path),
            )
        )
        assert interpreter._registry.get_if_exists("unfinished-turn") is None
    finally:
        interpreter._registry.close()


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_forced_exit_settles_turns_before_closing_code_interpreters(
    monkeypatch, tmp_path
):
    """A turn still running after a forced exit must not reopen its QuickJS slot."""
    from EvoScientist.middleware.code_interpreter import (
        create_code_interpreter_middleware,
    )

    interpreter = create_code_interpreter_middleware()

    class _TurnConsumer:
        def __init__(self, **_kwargs) -> None:
            self.metrics = {}

        async def run(self) -> None:
            self._turn = asyncio.create_task(self._reopening_turn())
            threading.Timer(0.2, signal.raise_signal, [signal.SIGINT]).start()
            threading.Timer(0.5, signal.raise_signal, [signal.SIGINT]).start()
            await asyncio.Future()

        async def _reopening_turn(self) -> None:
            while True:
                interpreter._registry.get("running-turn")
                await asyncio.sleep(0.01)

        async def stop(self) -> None:
            await asyncio.Future()

    async def _agent(_workspace):
        return object()

    monkeypatch.setattr(standalone, "_create_standalone_agent", _agent)
    monkeypatch.setattr(standalone, "InboundConsumer", _TurnConsumer)
    monkeypatch.setattr(
        "EvoScientist.gateway.create_runtime_gateways_for_config",
        lambda *a, **k: SimpleNamespace(graph_gateway=None),
    )
    try:
        asyncio.run(
            standalone._async_main(
                QueueFakeChannel(),
                MessageBus(),
                True,
                False,
                workspace=Workspace(tmp_path),
            )
        )
        assert interpreter._registry.get_if_exists("running-turn") is None
    finally:
        interpreter._registry.close()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows loops always take this path"
)
@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_ctrl_c_while_turns_settle_still_closes_code_interpreters(
    monkeypatch, tmp_path
):
    """A third Ctrl+C during the final cleanup must not skip the interpreter close."""
    from asyncio import unix_events

    from EvoScientist.middleware.code_interpreter import (
        create_code_interpreter_middleware,
    )

    def _unsupported(self, sig, callback, *args):
        raise NotImplementedError

    monkeypatch.setattr(
        unix_events._UnixSelectorEventLoop, "add_signal_handler", _unsupported
    )
    interpreter = create_code_interpreter_middleware()
    interpreter._registry.get("slow-turn")

    class _TurnConsumer:
        def __init__(self, **_kwargs) -> None:
            self.metrics = {}

        async def run(self) -> None:
            self._turn = asyncio.create_task(self._slow_to_cancel_turn())
            await asyncio.sleep(0.1)
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(0.1)
            signal.raise_signal(signal.SIGINT)
            await asyncio.Future()

        async def _slow_to_cancel_turn(self) -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(2)
                raise

        async def stop(self) -> None:
            await asyncio.Future()

    async def _agent(_workspace):
        return object()

    monkeypatch.setattr(standalone, "_create_standalone_agent", _agent)
    monkeypatch.setattr(standalone, "InboundConsumer", _TurnConsumer)
    monkeypatch.setattr(
        "EvoScientist.gateway.create_runtime_gateways_for_config",
        lambda *a, **k: SimpleNamespace(graph_gateway=None),
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(
                standalone._async_main(
                    QueueFakeChannel(),
                    MessageBus(),
                    True,
                    False,
                    workspace=Workspace(tmp_path),
                )
            )
        assert interpreter._registry.get_if_exists("slow-turn") is None
    finally:
        interpreter._registry.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sleep")
@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_sigint_stops_a_shell_command_the_turn_is_running(monkeypatch, tmp_path):
    """Ctrl+C kills the shell command an in-process turn is running."""
    from EvoScientist.backends import CustomSandboxBackend

    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    result = {}

    def _tool() -> None:
        result["resp"] = backend.execute("sleep 15")

    class _ToolConsumer:
        def __init__(self, **_kwargs) -> None:
            self.metrics = {}

        async def run(self) -> None:
            threading.Timer(0.5, signal.raise_signal, [signal.SIGINT]).start()
            await asyncio.create_task(asyncio.to_thread(_tool))

        async def stop(self) -> None:
            pass

    async def _agent(_workspace):
        return object()

    monkeypatch.setattr(standalone, "_create_standalone_agent", _agent)
    monkeypatch.setattr(standalone, "InboundConsumer", _ToolConsumer)
    monkeypatch.setattr(
        "EvoScientist.gateway.create_runtime_gateways_for_config",
        lambda *a, **k: SimpleNamespace(graph_gateway=None),
    )

    started = time.monotonic()
    asyncio.run(
        standalone._async_main(
            QueueFakeChannel(), MessageBus(), True, False, workspace=Workspace(tmp_path)
        )
    )

    assert time.monotonic() - started < 10
    assert result["resp"].exit_code == 130


async def test_stop_during_start_still_cleans_what_start_opened():
    """A stop() during start() still closes what start() opens afterwards."""
    gate = asyncio.Event()

    class _Channel(QueueFakeChannel):
        stop = Channel.stop
        receive = Channel.receive

        def __init__(self):
            super().__init__()
            self.opened = self.closed = 0
            self._resource = None

        async def start(self):
            await gate.wait()
            self._resource = object()
            self.opened += 1
            self._running = True

        async def _cleanup(self):
            if self._resource is not None:
                self._resource = None
                self.closed += 1

    channel = _Channel()
    task = asyncio.create_task(channel.run())
    await asyncio.sleep(0)
    await channel.stop()
    gate.set()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert (channel.opened, channel.closed) == (1, 1)


def _oauth_config(**overrides):
    base = {"anthropic_auth_mode": "api_key", "openai_auth_mode": "api_key"}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_ensure_ccproxy_noop_without_oauth(monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    calls: list[object] = []
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, **_k: calls.append(config)
    )

    standalone._ensure_ccproxy(_oauth_config())

    assert calls == []


def test_ensure_ccproxy_outlives_runner(monkeypatch):
    """It is shared by every session and the background server; EvoSci
    ccproxy stop stops it."""
    import atexit as atexit_mod

    import EvoScientist.ccproxy_manager as ccproxy_mod

    calls: list = []
    monkeypatch.setattr(
        ccproxy_mod,
        "maybe_start_ccproxy",
        lambda config, *, lasting: calls.append((config, lasting)),
    )
    registered: list[tuple] = []
    monkeypatch.setattr(atexit_mod, "register", lambda *args: registered.append(args))

    cfg = _oauth_config(anthropic_auth_mode="oauth")
    standalone._ensure_ccproxy(cfg)

    assert calls == [(cfg, True)]
    assert registered == []


def test_ensure_ccproxy_reused_instance_registers_no_shutdown(monkeypatch):
    """An already-running ccproxy (proc is None) needs no atexit shutdown."""
    import atexit as atexit_mod

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(ccproxy_mod, "maybe_start_ccproxy", lambda config, **_k: None)
    registered: list[tuple] = []
    monkeypatch.setattr(atexit_mod, "register", lambda *args: registered.append(args))

    standalone._ensure_ccproxy(_oauth_config(anthropic_auth_mode="oauth"))

    assert registered == []


def test_ensure_ccproxy_noop_with_openai_oauth_only(monkeypatch):
    """OpenAI's OAuth mode signs in to ChatGPT directly: no proxy needed."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    calls: list[object] = []
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, **_k: calls.append(config)
    )

    standalone._ensure_ccproxy(_oauth_config(openai_auth_mode="oauth"))

    assert calls == []


def test_ensure_ccproxy_exits_when_the_proxy_cannot_start(monkeypatch, caplog):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    def _boom(config, **_k):
        raise RuntimeError("ccproxy is required for OAuth mode but not found")

    monkeypatch.setattr(ccproxy_mod, "maybe_start_ccproxy", _boom)

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
        standalone._ensure_ccproxy(_oauth_config(anthropic_auth_mode="oauth"))

    assert exc_info.value.code == 1
    assert "ccproxy startup failed: ccproxy is required" in caplog.text


def test_run_standalone_ensures_ccproxy_only_with_agent(monkeypatch):
    """``run_standalone`` starts ccproxy (OAuth) iff an agent is configured."""
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(
            gateway_backend="local",
            default_workdir="",
            anthropic_auth_mode="oauth",
            openai_auth_mode="api_key",
        ),
    )
    ensured: list[object] = []
    monkeypatch.setattr(
        standalone, "_ensure_ccproxy", lambda config, **_k: ensured.append(config)
    )
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda cfg, *, workspace_dir, backend=None: None,
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)
    assert len(ensured) == 1

    ensured.clear()
    standalone.run_standalone(channel=None, bus=None, use_agent=False)
    assert ensured == []


def test_run_standalone_skips_ccproxy_when_chatgpt_sign_in_missing(monkeypatch):
    """ccproxy starts only after the ChatGPT sign-in check passes."""
    _, _ = _standalone_ready(
        monkeypatch,
        signed_in=False,
        openai_auth_mode="oauth",
        anthropic_auth_mode="oauth",
    )
    ensured: list[object] = []
    monkeypatch.setattr(
        standalone, "_ensure_ccproxy", lambda config, **_k: ensured.append(config)
    )

    with pytest.raises(SystemExit):
        standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert ensured == []


def test_run_standalone_starts_ccproxy_after_chatgpt_sign_in(monkeypatch):
    """Signed-in ChatGPT OAuth + Anthropic OAuth: the proxy starts."""
    _, _ = _standalone_ready(
        monkeypatch,
        signed_in=True,
        openai_auth_mode="oauth",
        anthropic_auth_mode="oauth",
    )
    ensured: list[object] = []
    monkeypatch.setattr(
        standalone, "_ensure_ccproxy", lambda config, **_k: ensured.append(config)
    )

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensured) == 1


def test_run_standalone_sets_the_proxy_env_before_the_dev_server(monkeypatch):
    """The dev server copies env from ``os.environ`` at spawn, so the proxy
    env must be force-set before the dev server starts."""
    import EvoScientist.ccproxy_manager as ccproxy_mod
    import EvoScientist.langgraph_dev.manager as manager_mod

    _standalone_ready(
        monkeypatch, signed_in=True, anthropic_auth_mode="oauth", ccproxy_port=8000
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_available", lambda: True)
    monkeypatch.setattr(ccproxy_mod, "check_ccproxy_auth", lambda provider: (True, ""))
    monkeypatch.setattr(ccproxy_mod, "ensure_ccproxy", lambda port, **_k: None)
    seen: list[str] = []
    monkeypatch.setattr(
        manager_mod,
        "ensure_langgraph_dev",
        lambda cfg, *, workspace_dir, backend=None: seen.append(
            os.environ["ANTHROPIC_BASE_URL"]
        ),
    )

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert seen == ["http://127.0.0.1:8000/claude"]


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


@pytest.mark.parametrize("use_agent", [False, True])
@pytest.mark.parametrize("outcome", ["return", "fatal_start"])
def test_channel_worker_exit_shuts_down_standalone(
    monkeypatch, tmp_path, caplog, use_agent, outcome
):
    from EvoScientist.channels.bus.events import OutboundMessage
    from EvoScientist.channels.channel_manager import ChannelManager

    bus = MessageBus()
    consumer_started = asyncio.Event()
    consumer_stopped = False
    consumer_cancelled = False

    class _FailedStartChannel(QueueFakeChannel):
        async def start(self):
            raise ChannelError("invalid test credentials")

    class _ExitingChannel(QueueFakeChannel):
        async def run(self):
            if use_agent:
                await consumer_started.wait()

    class _Consumer:
        def __init__(self, **_kwargs):
            self.metrics = {}

        async def run(self):
            nonlocal consumer_cancelled
            consumer_started.set()
            try:
                await asyncio.Future()
            finally:
                consumer_cancelled = True

        async def stop(self):
            nonlocal consumer_stopped
            consumer_stopped = True
            await bus.publish_outbound(
                OutboundMessage(channel="fake", chat_id="chat", content="last reply")
            )

    monkeypatch.setattr(standalone, "InboundConsumer", _Consumer)
    monkeypatch.setattr(standalone, "_create_standalone_agent", AsyncMock())
    monkeypatch.setattr(
        "EvoScientist.gateway.create_runtime_gateways_for_config",
        lambda *a, **k: SimpleNamespace(graph_gateway=None),
    )
    stop_health = AsyncMock()
    monkeypatch.setattr(ChannelManager, "stop_health", stop_health)
    channel = _FailedStartChannel() if outcome == "fatal_start" else _ExitingChannel()

    async def _run():
        async with asyncio.timeout(1):
            return await standalone._async_main(
                channel, bus, use_agent, False, workspace=Workspace(tmp_path)
            )

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(_run()) is True

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == standalone.__name__
    ] == ["Channel fake stopped; shutting down"]
    assert channel._stopped
    stop_health.assert_awaited_once()
    if use_agent:
        assert consumer_stopped
        assert consumer_cancelled
        assert [message.content for message in channel._sent] == ["last reply"]
    if outcome == "fatal_start":
        assert "invalid test credentials" in caplog.text


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
@pytest.mark.timeout(30)
def test_run_standalone_exits_nonzero_when_the_channel_cannot_start(
    monkeypatch, tmp_path
):
    class _FailedStartChannel(QueueFakeChannel):
        async def start(self):
            raise ChannelError("invalid test credentials")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        standalone.run_standalone(_FailedStartChannel(), MessageBus(), use_agent=False)

    assert exc_info.value.code == 1


def test_channel_exit_during_shutdown_does_not_force_exit(
    monkeypatch, tmp_path, caplog
):
    handlers = {}
    exited = asyncio.Event()

    class _StoppingChannel(QueueFakeChannel):
        async def run(self):
            handlers[signal.SIGINT]()
            await exited.wait()

        async def stop(self):
            exited.set()
            # Let the worker finish and its done callbacks run before stop completes.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await super().stop()

    channel = _StoppingChannel()

    async def _run():
        monkeypatch.setattr(
            asyncio.get_running_loop(),
            "add_signal_handler",
            lambda sig, callback: handlers.__setitem__(sig, callback),
        )
        async with asyncio.timeout(1):
            return await standalone._async_main(
                channel, MessageBus(), False, False, workspace=Workspace(tmp_path)
            )

    assert asyncio.run(_run()) is False
    assert channel._stopped
    assert "forcing" not in caplog.text
    assert "Channel fake stopped; shutting down" not in caplog.messages


def test_outer_cancellation_does_not_start_shutdown(monkeypatch, tmp_path, caplog):
    class _CancelledChannel(QueueFakeChannel):
        async def run(self):
            runner.cancel()
            await asyncio.Future()

    channel = _CancelledChannel()
    runner = None

    async def _run():
        nonlocal runner
        runner = asyncio.current_task()
        await standalone._async_main(
            channel, MessageBus(), False, False, workspace=Workspace(tmp_path)
        )

    with caplog.at_level(logging.INFO), pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())

    assert not channel._stopped
    assert "Graceful shutdown initiated" not in caplog.text
    assert "Channel fake stopped; shutting down" not in caplog.messages


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
def test_sigint_forces_shutdown_after_channel_worker_exits(tmp_path, caplog):
    class _Channel(QueueFakeChannel):
        signalled = False

        async def run(self):
            return

        async def stop(self):
            if not self.signalled:
                self.signalled = True
                asyncio.get_running_loop().call_soon(signal.raise_signal, signal.SIGINT)
            await asyncio.Future()

    channel = _Channel()

    async def _run():
        async with asyncio.timeout(1):
            await standalone._async_main(
                channel, MessageBus(), False, False, workspace=Workspace(tmp_path)
            )

    asyncio.run(_run())
    assert "Signal received during shutdown, forcing exit" in caplog.text
