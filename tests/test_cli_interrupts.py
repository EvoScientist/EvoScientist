"""Signal-level regression tests for Rich CLI turn cancellation."""

import asyncio
import signal
import threading

import pytest

from EvoScientist.cli import interactive


@pytest.mark.asyncio
async def test_session_turns_are_serialized() -> None:
    """A channel turn cannot start while a foreground turn owns the session."""
    turn_lock = asyncio.Lock()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    order: list[str] = []

    async def first_turn() -> None:
        order.append("first-started")
        first_started.set()
        await release_first.wait()
        order.append("first-finished")

    async def second_turn() -> None:
        order.append("second-started")
        second_started.set()

    first = asyncio.create_task(interactive._run_serialized_turn(turn_lock, first_turn))
    await first_started.wait()
    second = asyncio.create_task(
        interactive._run_serialized_turn(turn_lock, second_turn)
    )
    await asyncio.sleep(0)

    assert not second_started.is_set()
    release_first.set()
    await asyncio.gather(first, second)
    assert order == ["first-started", "first-finished", "second-started"]


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
