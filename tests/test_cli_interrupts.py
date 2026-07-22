"""Signal-level regression tests for Rich CLI turn cancellation."""

import asyncio
import signal
import threading

import pytest

from EvoScientist.cli import interactive


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="process signal handlers require the main thread",
)
def test_ctrl_c_can_cancel_two_separate_rich_cli_turns(monkeypatch):
    """A recovered turn must not consume asyncio.run's force-quit budget."""
    started = asyncio.Event()
    calls = 0

    async def fake_run_streaming_async(**kwargs):
        nonlocal calls
        assert kwargs["recover_on_cancel"] is True
        calls += 1
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            current = asyncio.current_task()
            assert current is not None
            current.uncancel()
            return "[Stopped.]"

    monkeypatch.setattr(interactive, "run_streaming_async", fake_run_streaming_async)
    original_sigint = signal.getsignal(signal.SIGINT)

    async def cancel_started_turn() -> None:
        await started.wait()
        signal.raise_signal(signal.SIGINT)

    async def scenario() -> None:
        runner_sigint = signal.getsignal(signal.SIGINT)
        for _ in range(2):
            started.clear()
            sender = asyncio.create_task(cancel_started_turn())
            assert await interactive._run_rich_cli_streaming_turn() == "[Stopped.]"
            await sender
            assert signal.getsignal(signal.SIGINT) is runner_sigint

    asyncio.run(scenario())

    assert calls == 2
    assert signal.getsignal(signal.SIGINT) is original_sigint
