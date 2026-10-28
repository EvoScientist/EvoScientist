"""New-message turns on the LangGraph server gateway (#558).

A new message reads ``stream.events``: it skips the thread's history replay,
starts the run, keeps only that run's envelopes and stops at its root
completed/failed or at a pause. Resume turns keep main's ``subscribe`` path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.types import Command

from EvoScientist.gateway import (
    LangGraphServerGateway,
    LangGraphServerThreadStore,
    RunRequest,
)
from EvoScientist.gateway import server as server_mod
from EvoScientist.gateway.server import _ThreadEvents
from tests.fakes.common import (
    FakeLangGraphClient,
    FakeLangGraphThreadsClient,
    FakeLangGraphThreadStream,
)

THREAD_ID = "abc12345"
# Each turn gets a guard so a hang fails the test instead of the suite.
# Nothing below asserts on elapsed time.
GUARD_SECONDS = 10

_ACTION_REQUESTS = [
    {"name": "execute", "args": {"command": "echo hello"}, "id": "tool-1"}
]
_REVIEW_CONFIGS = [
    {"action_name": "execute", "allowed_decisions": ["approve", "reject"]}
]
_STATE_INTERRUPT = {
    "id": "interrupt-1",
    "value": {
        "action_requests": _ACTION_REQUESTS,
        "review_configs": _REVIEW_CONFIGS,
    },
}
_INTERRUPT_EVENT = {
    "type": "interrupt",
    "interrupt_id": "interrupt-1",
    "action_requests": _ACTION_REQUESTS,
    "review_configs": _REVIEW_CONFIGS,
}


def _lifecycle(run_id: str, phase: str, namespace: list[str] | None = None):
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace or [], "data": {"event": phase}},
        "event_id": f"synth:{run_id}:lc:{'/'.join(namespace or [])}:{phase}",
    }


def _text(text: str, entry_id: str = "1-0") -> dict[str, Any]:
    return {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": {
                "event": "content-block-delta",
                "delta": {"type": "text-delta", "text": text},
            },
        },
        "event_id": entry_id,
    }


def _finish(entry_id: str = "1-1") -> dict[str, Any]:
    return {
        "method": "messages",
        "params": {"namespace": [], "data": {"event": "message-finish"}},
        "event_id": entry_id,
    }


def _input_requested(namespace: list[str]) -> dict[str, Any]:
    """Shaped like langchain_protocol ``InputRequestedData``: ``payload``."""
    return {
        "method": "input.requested",
        "params": {
            "namespace": namespace,
            "data": {
                "interrupt_id": "interrupt-1",
                "payload": _STATE_INTERRUPT["value"],
            },
        },
        "event_id": "2-0",
    }


def _earlier_run(run_id: str, text: str) -> list[dict[str, Any]]:
    return [
        _lifecycle(run_id, "running"),
        _text(text, "0-1"),
        _finish("0-2"),
        _lifecycle(run_id, "completed"),
    ]


_EARLIER_STATE = {
    "messages": [
        {"type": "human", "content": "first", "id": "human-0"},
        {"type": "ai", "content": "old answer", "id": "ai-0"},
    ]
}


def _gateway(
    stream: FakeLangGraphThreadStream,
    *,
    values: dict[str, Any] | None = None,
    interrupts: list[dict[str, Any]] | None = None,
) -> tuple[LangGraphServerGateway, FakeLangGraphThreadsClient]:
    state: dict[str, Any] = {"values": values or {}}
    if interrupts is not None:
        state["interrupts"] = interrupts
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": THREAD_ID, "metadata": {"graph_id": "EvoScientist"}}],
        states={THREAD_ID: state},
        streams={THREAD_ID: stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    return gateway, threads


async def _turn(gateway: LangGraphServerGateway, message: object) -> list[dict]:
    async def collect() -> list[dict]:
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(message=message, thread_id=THREAD_ID)
            )
        ]

    async with asyncio.timeout(GUARD_SECONDS):
        return await collect()


def _texts(events: list[dict]) -> list[str]:
    return [event["content"] for event in events if event["type"] == "text"]


# --- state: pause on a new thread / in a subgraph (regression 1) -------------


@pytest.mark.parametrize(
    "pause",
    [
        pytest.param([_input_requested([])], id="root"),
        pytest.param(
            [_input_requested(["child:1"]), _lifecycle("run-1", "interrupted")],
            id="subgraph",
        ),
    ],
)
async def test_pause_reads_interrupt_from_state_when_lifecycle_lags(pause):
    # ``stream.interrupted`` comes from a separate SSE and can still be False
    # when the pause envelope arrives; the envelope carries ``payload``, which
    # the event processor does not parse, so the interrupt must come from state.
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), *pause],
        interrupted=False,
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[_STATE_INTERRUPT])

    events = await _turn(gateway, "hi")

    # A subgraph namespace also opens and closes a subagent in the tracker.
    assert [event for event in events if event["type"] == "interrupt"] == [
        _INTERRUPT_EVENT
    ]
    assert events[-1] == {"type": "done", "content": "", "response": ""}
    assert THREAD_ID in threads.state_gets


async def test_completed_run_does_not_read_state_interrupts():
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("hello"), _finish(), _lifecycle("run-1", "completed")],
        stay_open=True,
    )
    gateway, _ = _gateway(stream, interrupts=[_STATE_INTERRUPT])

    events = await _turn(gateway, "hi")

    assert [event["type"] for event in events] == ["text", "done"]


# --- state: pause after a completed turn ------------------------------------


async def test_pause_after_a_completed_turn_skips_the_replayed_turn():
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), _input_requested([])],
        replay=_earlier_run("run-0", "old answer"),
        stay_open=True,
    )
    gateway, _ = _gateway(stream, values=_EARLIER_STATE, interrupts=[_STATE_INTERRUPT])

    events = await _turn(gateway, "hi")

    assert events == [_INTERRUPT_EVENT, {"type": "done", "content": "", "response": ""}]
    assert stream.lost_live_events is False


# --- state: second turn (#558) and long history (regression 2) --------------


async def test_second_turn_ignores_replayed_history():
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _text("new answer", "1-0"),
            _finish(),
            _lifecycle("run-1", "completed"),
        ],
        replay=_earlier_run("run-0", "old answer"),
        stay_open=True,
    )
    gateway, _ = _gateway(stream, values=_EARLIER_STATE)

    events = await _turn(gateway, "second")

    assert _texts(events) == ["new answer"]
    assert events[-1] == {
        "type": "done",
        "content": "new answer",
        "response": "new answer",
    }
    assert stream.event_streams_opened == 1


async def test_starts_the_run_after_a_long_history_replay():
    replay = [
        event
        for index in range(120)
        for event in _earlier_run(f"run-old-{index}", f"old {index}")
    ]
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("short", "1-0"), _finish(), _lifecycle("run-1", "completed")],
        replay=replay,
        stay_open=True,
    )
    gateway, _ = _gateway(stream, values=_EARLIER_STATE)

    events = await _turn(gateway, "hi")

    # The fake drops live envelopes for a run started before the replay ended.
    assert stream.lost_live_events is False
    assert _texts(events) == ["short"]


async def test_run_started_during_the_replay_loses_its_events():
    """The fake models regression 2: starting first loses the live envelopes."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("short"), _lifecycle("run-1", "completed")],
        replay=_earlier_run("run-0", "old"),
        stay_open=True,
    )
    await stream.run.start(input={})
    received = []

    with suppress(TimeoutError):
        async with asyncio.timeout(0.2):
            async for event in stream.events:
                received.append(event)

    assert stream.lost_live_events is True
    assert received == _earlier_run("run-0", "old")


# --- state: first turn on a new thread --------------------------------------


@pytest.mark.parametrize(
    ("values", "replay", "skips"),
    [
        pytest.param({}, None, 0, id="new-thread"),
        pytest.param(
            _EARLIER_STATE, _earlier_run("run-0", "old"), 1, id="thread-with-state"
        ),
    ],
)
async def test_replay_is_skipped_only_when_the_thread_has_state(values, replay, skips):
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("hello"), _finish(), _lifecycle("run-1", "completed")],
        replay=replay,
        stay_open=True,
    )
    gateway, _ = _gateway(stream, values=values)
    calls = 0
    original = _ThreadEvents.skip_replay

    async def counting_skip(self):
        nonlocal calls
        calls += 1
        return await original(self)

    with patch.object(_ThreadEvents, "skip_replay", counting_skip):
        events = await _turn(gateway, "hi")

    assert calls == skips
    assert _texts(events) == ["hello"]
    assert len(stream.run.starts) == 1


async def test_events_from_another_run_are_dropped():
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-other", "running"),
            _text("other", "1-0"),
            _lifecycle("run-1", "running"),
            _text("mine", "1-1"),
            _lifecycle("run-other", "completed"),
            _lifecycle("run-1", "completed"),
        ],
        stay_open=True,
    )
    gateway, _ = _gateway(stream)

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["mine"]


async def test_events_stream_ending_early_raises():
    stream = FakeLangGraphThreadStream(THREAD_ID, events=[])

    async def ended() -> AsyncIterator[dict[str, Any]]:
        await stream.run.started.wait()
        return
        yield

    gateway, _ = _gateway(stream)
    with (
        patch.object(
            FakeLangGraphThreadStream,
            "events",
            property(lambda self: ended()),
        ),
        pytest.raises(RuntimeError, match="ended before the run finished"),
    ):
        await _turn(gateway, "hi")


# --- state: resume and retry keep main's subscribe path ---------------------


async def test_resume_keeps_the_subscribe_path():
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("resumed: ok"), _finish()],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    gateway, _ = _gateway(stream)

    events = await _turn(
        gateway, Command(resume={"interrupt-1": {"decisions": [{"type": "approve"}]}})
    )

    assert stream.event_streams_opened == 0
    assert stream.run.starts == []
    assert stream.run.responses == [
        {
            "response": {"decisions": [{"type": "approve"}]},
            "interrupt_id": "interrupt-1",
        }
    ]
    assert _texts(events) == ["resumed: ok"]


async def test_retrying_the_same_resume_payload_resumes_again():
    payload = {"interrupt-1": {"decisions": [{"type": "approve"}]}}
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("resumed: ok"), _finish()],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    gateway, _ = _gateway(stream)

    first = await _turn(gateway, Command(resume=payload))
    second = await _turn(gateway, Command(resume=payload))

    assert _texts(first) == _texts(second) == ["resumed: ok"]
    assert len(stream.run.responses) == 2
    assert stream.event_streams_opened == 0


# --- _ThreadEvents.skip_replay: controllable clock, no real waiting ---------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _fake_wait_for(clock: _Clock, step: float):
    """``wait_for`` on a fake clock: an item costs ``step``, an empty queue times out."""

    async def wait_for(awaitable, timeout):
        task = asyncio.ensure_future(awaitable)
        for _ in range(5):
            await asyncio.sleep(0)
        if task.done():
            clock.now += step
            return task.result()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        clock.now += timeout
        raise TimeoutError

    return wait_for


async def _source(events: list[Any], *, then_wait: bool) -> AsyncIterator[Any]:
    for event in events:
        yield event
    if then_wait:
        await asyncio.Event().wait()


async def test_skip_replay_stops_at_the_quiet_window():
    clock = _Clock()
    replay = [{"n": index} for index in range(7)]
    async with _ThreadEvents(
        _source(replay, then_wait=True),
        quiet_seconds=0.5,
        max_seconds=30.0,
        clock=clock,
        wait_for=_fake_wait_for(clock, step=0.25),
    ) as events:
        skipped = await events.skip_replay()

    assert skipped == 7


async def test_skip_replay_raises_when_the_cap_is_hit_while_still_receiving():
    async def endless() -> AsyncIterator[Any]:
        index = 0
        while True:
            yield {"n": index}
            index += 1
            await asyncio.sleep(0)

    clock = _Clock()
    async with _ThreadEvents(
        endless(),
        quiet_seconds=0.5,
        max_seconds=1.0,
        clock=clock,
        wait_for=_fake_wait_for(clock, step=0.25),
    ) as events:
        with pytest.raises(RuntimeError, match="replay did not finish within 1s"):
            await events.skip_replay()


async def test_skip_replay_raises_when_the_cap_trims_the_quiet_window():
    """A quiet wait cut short by the cap is unfinished replay, not a gap."""
    clock = _Clock()
    yielded = {"n": 0}

    async def source() -> AsyncIterator[Any]:
        while yielded["n"] < 3:
            yield {"n": yielded["n"]}
            yielded["n"] += 1
        await asyncio.Event().wait()

    async def wait_for(awaitable, timeout):
        task = asyncio.ensure_future(awaitable)
        for _ in range(5):
            await asyncio.sleep(0)
        if task.done():
            clock.now += 0.2
            return task.result()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        # After three events, clock is 0.6; left under max=1.0 is 0.4, so the
        # caller asks for timeout=0.4 which is under quiet=0.5.
        clock.now += timeout
        raise TimeoutError

    async with _ThreadEvents(
        source(),
        quiet_seconds=0.5,
        first_event_seconds=2.0,
        max_seconds=1.0,
        clock=clock,
        wait_for=wait_for,
    ) as events:
        with pytest.raises(RuntimeError, match="replay did not finish within 1s"):
            await events.skip_replay()


async def test_skip_replay_keeps_the_end_of_stream():
    clock = _Clock()
    async with _ThreadEvents(
        _source([{"n": 0}, {"n": 1}], then_wait=False),
        clock=clock,
        wait_for=_fake_wait_for(clock, step=0.25),
    ) as events:
        skipped = await events.skip_replay()
        rest = [event async for event in events]

    assert skipped == 2
    assert rest == []


async def test_skip_replay_keeps_a_stream_error():
    async def failing() -> AsyncIterator[Any]:
        yield {"n": 0}
        raise ConnectionError("provider connection lost")

    clock = _Clock()
    async with _ThreadEvents(
        failing(), clock=clock, wait_for=_fake_wait_for(clock, step=0.25)
    ) as events:
        skipped = await events.skip_replay()
        with pytest.raises(ConnectionError, match="provider connection lost"):
            await anext(events)

    assert skipped == 1


async def test_skip_replay_waits_for_the_first_event_before_the_quiet_window():
    """Setup delay before the first envelope is not the end of the replay."""
    clock = _Clock()
    released = asyncio.Event()

    async def delayed() -> AsyncIterator[Any]:
        await released.wait()
        for index in range(3):
            yield {"n": index}
        await asyncio.Event().wait()

    async def wait_for(awaitable, timeout):
        task = asyncio.ensure_future(awaitable)
        for _ in range(5):
            await asyncio.sleep(0)
        if not released.is_set() and not task.done():
            # First attempt times out under the quiet window; the production
            # code then keeps waiting up to first_event_seconds.
            if timeout <= 0.05:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                clock.now += timeout
                raise TimeoutError
            released.set()
            for _ in range(5):
                await asyncio.sleep(0)
        if task.done():
            clock.now += 0.25
            return task.result()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        clock.now += timeout
        raise TimeoutError

    async with _ThreadEvents(
        delayed(),
        quiet_seconds=0.02,
        first_event_seconds=2.0,
        max_seconds=5.0,
        clock=clock,
        wait_for=wait_for,
    ) as events:
        skipped = await events.skip_replay()

    assert skipped == 3
    assert clock.now >= 0.02


async def test_skip_replay_returns_zero_when_no_envelope_arrives():
    clock = _Clock()
    async with _ThreadEvents(
        _source([], then_wait=True),
        quiet_seconds=0.5,
        first_event_seconds=1.0,
        max_seconds=5.0,
        clock=clock,
        wait_for=_fake_wait_for(clock, step=0.25),
    ) as events:
        skipped = await events.skip_replay()

    assert skipped == 0
    assert clock.now == 1.0


async def test_thread_events_closes_its_reader():
    closed = asyncio.Event()

    async def watched() -> AsyncIterator[Any]:
        try:
            await asyncio.Event().wait()
            yield
        finally:
            closed.set()

    async with _ThreadEvents(watched()):
        await asyncio.sleep(0)

    assert closed.is_set()


async def test_pause_state_read_failure_does_not_cancel_the_paused_run():
    """A transient get_state error must not cancel an already paused run.

    The turn emits the gateway ``error`` shape (and raises) so callers do not
    treat an empty ``done`` as a normal finish / "No response".
    """
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), _input_requested([])],
        interrupted=False,
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[_STATE_INTERRUPT])
    client = gateway.thread_store.client
    cancelled: list[list[str]] = []

    class _Runs:
        async def list(self, thread_id: str, **kwargs):
            return [{"run_id": "run-1", "status": "running", "metadata": {}}]

        async def cancel_many(self, *, thread_id: str, run_ids):
            cancelled.append(list(run_ids))

    client.runs = _Runs()

    # Fail only the post-pause read; the pre-run state fetch must still work.
    real_get_state = threads.get_state
    calls = {"n": 0}

    async def boom(thread_id: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return await real_get_state(thread_id)
        raise ConnectionError("state store unavailable")

    # Force the paused-state fallback: no stream.interrupts, interrupted False.
    stream.interrupts = []
    threads.get_state = boom  # type: ignore[method-assign]

    events: list[dict] = []

    async def collect() -> None:
        async with asyncio.timeout(GUARD_SECONDS):
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id=THREAD_ID)
            ):
                events.append(event)

    with pytest.raises(
        server_mod._PauseStateReadError,
        match="Could not read the pause state",
    ):
        await collect()

    assert cancelled == []
    assert [event["type"] for event in events] == ["error"]
    assert events[-1] == {
        "type": "error",
        "message": f"Could not read the pause state for thread {THREAD_ID}",
    }


async def test_root_failed_lifecycle_emits_done_like_main():
    """A root ``failed`` ends the turn the same way main's subscribe path does.

    Main ends ``subscribe`` at root failed and then emits ``done`` with whatever
    text was streamed. The new-message path matches that: it breaks on failed
    and emits ``done``. An ``error`` event is reserved for raised exceptions.
    """
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _text("partial", "1-0"),
            _lifecycle("run-1", "failed"),
        ],
        stay_open=True,
    )
    gateway, _ = _gateway(stream)

    events = await _turn(gateway, "hi")

    assert [event["type"] for event in events] == ["text", "done"]
    assert events[-1] == {
        "type": "done",
        "content": "partial",
        "response": "partial",
    }


def test_quiet_window_defaults():
    assert server_mod._REPLAY_QUIET_SECONDS < server_mod._REPLAY_MAX_SECONDS
