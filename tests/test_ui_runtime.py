"""Tests for UI backend runtime selection."""

import asyncio
import threading
import time
from dataclasses import dataclass

import pytest

from EvoScientist.cli.tui_runtime import (
    StreamCancellationTimeout,
    normalize_ui_backend,
    resolve_ui_backend,
    run_streaming,
    run_streaming_async,
)
from EvoScientist.runtime import AsyncRuntimeError
from tests.fakes import FakeGraphGateway


def test_normalize_ui_backend_defaults_to_cli():
    assert normalize_ui_backend(None) == "cli"
    assert normalize_ui_backend("") == "cli"


def test_normalize_ui_backend_accepts_known_values():
    assert normalize_ui_backend("cli") == "cli"
    assert normalize_ui_backend("tui") == "tui"
    assert normalize_ui_backend("TUI") == "tui"


def test_normalize_ui_backend_maps_legacy_values():
    assert normalize_ui_backend("textual") == "tui"
    assert normalize_ui_backend("Textual") == "tui"
    assert normalize_ui_backend("rich") == "cli"
    assert normalize_ui_backend("Rich") == "cli"


def test_normalize_ui_backend_unknown_falls_back_to_cli():
    assert normalize_ui_backend("unknown-ui") == "cli"


def test_resolve_ui_backend_falls_back_when_textual_unavailable(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime._has_textual_support", lambda: False
    )
    assert resolve_ui_backend("tui") == "cli"


def test_resolve_ui_backend_keeps_tui_when_available(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime._has_textual_support", lambda: True
    )
    assert resolve_ui_backend("tui") == "tui"


@dataclass
class _BrokenBackend:
    name: str = "tui"

    def run_streaming(self, **kwargs):
        raise RuntimeError("boom")


def test_run_streaming_falls_back_to_cli_on_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.get_backend", lambda *a, **k: _BrokenBackend()
    )

    class _RichStub:
        def run_streaming(self, **kwargs):
            return "fallback-ok"

    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.RichStreamingBackend", lambda: _RichStub()
    )

    result = run_streaming(
        ui_backend="tui",
        agent=object(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        gateway=FakeGraphGateway(),
    )
    assert result == "fallback-ok"


def test_run_streaming_does_not_retry_on_owned_runtime_error(monkeypatch):
    attempts = 0

    class _RuntimeFailureBackend:
        def run_streaming(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise AsyncRuntimeError("owned runtime failed")

    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.get_backend",
        lambda *a, **k: _RuntimeFailureBackend(),
    )
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.RichStreamingBackend",
        lambda: _RuntimeFailureBackend(),
    )

    with pytest.raises(AsyncRuntimeError, match="owned runtime failed"):
        run_streaming(
            ui_backend="tui",
            agent=object(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=FakeGraphGateway(),
        )

    assert attempts == 1


async def test_async_streaming_cancellation_stops_and_joins_worker(monkeypatch):
    from EvoScientist.stream.display import (
        discard_stream_cancel,
        is_stream_cancel_requested,
    )

    scope = "test:async-renderer-cancel"
    started = threading.Event()
    finished = threading.Event()

    def fake_run_streaming(**kwargs):
        assert kwargs["cancel_scope"] == scope
        started.set()
        while not is_stream_cancel_requested(scope):
            time.sleep(0.001)
        finished.set()
        return "stopped"

    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.run_streaming", fake_run_streaming
    )

    task = asyncio.create_task(run_streaming_async(cancel_scope=scope))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    discard_stream_cancel(scope)


async def test_async_streaming_can_recover_foreground_task_after_cancel(monkeypatch):
    from EvoScientist.stream.display import (
        discard_stream_cancel,
        is_stream_cancel_requested,
    )

    scope = "test:async-renderer-recover"
    started = threading.Event()

    def fake_run_streaming(**kwargs):
        started.set()
        while not is_stream_cancel_requested(scope):
            time.sleep(0.001)
        return "[Stopped.]"

    cleanup_called = False

    async def fake_cleanup() -> None:
        nonlocal cleanup_called
        cleanup_called = True

    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.run_streaming", fake_run_streaming
    )
    monkeypatch.setattr(
        "EvoScientist.middleware.code_interpreter.aclose_code_interpreters",
        fake_cleanup,
    )

    task = asyncio.create_task(
        run_streaming_async(cancel_scope=scope, recover_on_cancel=True)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    assert await task == "[Stopped.]"
    assert cleanup_called
    assert not task.cancelled()
    discard_stream_cancel(scope)


async def test_noncooperative_worker_reports_settlement_timeout(monkeypatch):
    """Cancellation timeout is an ordinary lifecycle error, not BaseException."""
    from EvoScientist.cli import tui_runtime
    from EvoScientist.stream.display import discard_stream_cancel

    scope = "test:async-renderer-timeout"
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def fake_run_streaming(**_kwargs):
        started.set()
        release.wait()
        finished.set()
        return "late"

    async def fake_cleanup() -> None:
        return None

    monkeypatch.setattr(tui_runtime, "run_streaming", fake_run_streaming)
    monkeypatch.setattr(tui_runtime, "STREAM_CANCEL_SETTLE_TIMEOUT", 0.05)
    monkeypatch.setattr(
        "EvoScientist.middleware.code_interpreter.aclose_code_interpreters",
        fake_cleanup,
    )

    task = asyncio.create_task(
        run_streaming_async(cancel_scope=scope, recover_on_cancel=True)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(StreamCancellationTimeout, match="did not stop"):
        await task

    assert not finished.is_set()
    release.set()
    assert await asyncio.to_thread(finished.wait, 1)
    await asyncio.sleep(0)
    discard_stream_cancel(scope)
