"""Turns on the LangGraph server gateway (#558, #738).

Every turn, a new message or a resume, reads ``stream.events``: it skips the
thread's history replay, starts the run, keeps only that run's envelopes and
stops at its root completed/failed or at a pause.
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


@pytest.fixture(autouse=True)
def _fast_thread_replay_skip_timeouts(monkeypatch):
    """Keep ``_ThreadEvents.skip_replay`` off the wall clock in the suite.

    Production defaults stay in ``EvoScientist.gateway.server``; skip_replay
    reads them at call time when the constructor left timeouts unset.
    """
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_QUIET_SECONDS", 0.01)
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_FIRST_EVENT_SECONDS", 0.05)
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_MAX_SECONDS", 0.5)


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


# --- state: pause on a new thread / in a subgraph ---------------------------


@pytest.mark.parametrize("namespace", [[], ["child:1"]], ids=["root", "subgraph"])
async def test_pause_prompt_comes_from_the_envelope_before_state_lists_it(namespace):
    """#706: when ``input.requested`` arrives the thread state may not list
    the interrupt yet; the prompt comes from the envelope's ``payload``. A
    subgraph's pause ends the turn too, at the root ``interrupted`` the server
    sends right after it."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _input_requested(namespace),
            _lifecycle("run-1", "interrupted"),
        ],
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[])

    events = await _turn(gateway, "hi")

    # A subgraph namespace also opens and closes a subagent in the tracker.
    assert [event for event in events if event["type"] == "interrupt"] == [
        _INTERRUPT_EVENT
    ]
    assert events[-1] == {"type": "done", "content": "", "response": ""}
    # The pre-run read and the pause read, which found nothing yet.
    assert threads.state_gets == [THREAD_ID, THREAD_ID]


async def test_pause_of_several_interrupts_prompts_each_once():
    """The turn stops at the root ``interrupted`` that follows the first
    interrupting task's envelope; the step's other interrupts come from thread
    state, without repeating the envelope's prompt."""
    second = {"id": "interrupt-2", "value": _STATE_INTERRUPT["value"]}
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _input_requested([]),
            _lifecycle("run-1", "interrupted"),
        ],
        stay_open=True,
    )
    gateway, _ = _gateway(stream, interrupts=[_STATE_INTERRUPT, second])

    events = await _turn(gateway, "hi")

    assert [e["interrupt_id"] for e in events if e["type"] == "interrupt"] == [
        "interrupt-1",
        "interrupt-2",
    ]
    assert events[-1]["type"] == "done"


_ANSWERED_INTERRUPT = {
    "id": "interrupt-0",
    "value": {
        "action_requests": [{"name": "execute", "args": {"command": "echo old"}}],
        "review_configs": _REVIEW_CONFIGS,
    },
}
_SECOND_INTERRUPT = {"id": "interrupt-2", "value": _STATE_INTERRUPT["value"]}


@pytest.mark.parametrize(
    ("envelope", "written_by", "interrupts", "prompts"),
    [
        (True, "run-0", [_STATE_INTERRUPT, _ANSWERED_INTERRUPT], ["interrupt-1"]),
        (
            True,
            "run-1",
            [_STATE_INTERRUPT, _SECOND_INTERRUPT],
            ["interrupt-1", "interrupt-2"],
        ),
        (False, "run-0", [_STATE_INTERRUPT], ["interrupt-1"]),
    ],
    ids=["earlier-run", "own-run", "no-envelope"],
)
async def test_resume_pause_adds_state_prompts_only_from_its_own_run(
    envelope, written_by, interrupts, prompts
):
    """Once the envelope gave the prompt, only a checkpoint the resumed run
    wrote adds prompts: the earlier run's can still list the interrupt just
    answered. Without an envelope prompt the state is the only source."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            *([_input_requested([])] if envelope else []),
            _lifecycle("run-1", "interrupted"),
        ],
        interrupted=True,
        replay=[_lifecycle("run-0", "running"), _lifecycle("run-0", "interrupted")],
        stay_open=True,
    )
    gateway, threads = _gateway(
        stream, values=_EARLIER_STATE, interrupts=[_ANSWERED_INTERRUPT]
    )
    threads.states[THREAD_ID]["metadata"] = {"run_id": "run-0"}
    respond = stream.run.respond_command

    async def respond_and_pause_again(params):
        reply = await respond(params)
        threads.states[THREAD_ID].update(
            interrupts=interrupts, metadata={"run_id": written_by}
        )
        return reply

    stream.run.respond_command = respond_and_pause_again

    events = await _turn(
        gateway, Command(resume={"interrupt-0": {"decisions": [{"type": "approve"}]}})
    )

    assert [e["interrupt_id"] for e in events if e["type"] == "interrupt"] == prompts


async def test_failed_pause_read_after_a_prompt_keeps_the_turn():
    """The envelope already showed the prompt: a failed state read only logs."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _input_requested([]),
            _lifecycle("run-1", "interrupted"),
        ],
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[])
    real_get_state = threads.get_state

    async def fail_after_the_pre_run_read(thread_id: str):
        if threads.state_gets:
            raise ConnectionError("state store unavailable")
        return await real_get_state(thread_id)

    threads.get_state = fail_after_the_pre_run_read  # type: ignore[method-assign]

    events = await _turn(gateway, "hi")

    assert events == [_INTERRUPT_EVENT, {"type": "done", "content": "", "response": ""}]
    assert gateway.thread_store.client.runs.cancelled == []


async def test_pause_without_an_envelope_prompt_reads_it_from_state():
    """A pause seen only as a root ``interrupted`` gives no prompt; it is read
    from thread state, which lists it once the run has stopped."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), _lifecycle("run-1", "interrupted")],
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[_STATE_INTERRUPT])

    events = await _turn(gateway, "hi")

    assert events == [_INTERRUPT_EVENT, {"type": "done", "content": "", "response": ""}]
    assert threads.state_gets == [THREAD_ID, THREAD_ID]


async def test_completed_run_does_not_read_state_interrupts():
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("hello"), _finish(), _lifecycle("run-1", "completed")],
        stay_open=True,
    )
    gateway, _ = _gateway(stream, interrupts=[_STATE_INTERRUPT])

    events = await _turn(gateway, "hi")

    assert [event["type"] for event in events] == ["text", "done"]


async def test_completed_turn_skips_pause_read_when_not_paused():
    """A finished turn must not read ``stream.interrupts`` or report a pause error.

    ``stream.interrupted`` can still be true and ``get_state`` can fail after a
    completed root run; the pause helper returns before either is consulted.
    """
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("hello"), _finish(), _lifecycle("run-1", "completed")],
        interrupted=True,
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[_STATE_INTERRUPT])
    stream.interrupts = [_STATE_INTERRUPT]

    real_get_state = threads.get_state
    calls = {"n": 0}

    async def boom(thread_id: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return await real_get_state(thread_id)
        raise ConnectionError("state store unavailable")

    threads.get_state = boom  # type: ignore[method-assign]

    events = await _turn(gateway, "hi")

    assert [event["type"] for event in events] == ["text", "done"]
    assert calls["n"] == 1


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


# --- state: second turn (#558) and long history ----------------------------


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
    assert stream.lost_live_events is False


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


# --- state: first turn on a new thread --------------------------------------


@pytest.mark.parametrize(
    ("values", "replay", "skips", "fail_state"),
    [
        pytest.param({}, None, 0, False, id="new-thread"),
        pytest.param(
            _EARLIER_STATE,
            _earlier_run("run-0", "old"),
            1,
            False,
            id="thread-with-state",
        ),
        pytest.param(
            {},
            _earlier_run("run-0", "old"),
            1,
            True,
            id="state-read-fails",
        ),
    ],
)
async def test_replay_is_skipped_only_when_the_thread_has_state(
    values, replay, skips, fail_state
):
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_text("hello"), _finish(), _lifecycle("run-1", "completed")],
        replay=replay,
        stay_open=True,
    )
    gateway, threads = _gateway(stream, values=values)
    if fail_state:

        async def boom(thread_id: str):
            raise ConnectionError("state store unavailable")

        threads.get_state = boom  # type: ignore[method-assign]
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
    assert stream.lost_live_events is False


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


# --- state: resume and retry read events like a new message ----------------


async def test_resume_reads_its_run_from_the_thread_events():
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=[_text("resumed: ok"), _finish()]
    )
    gateway, _ = _gateway(stream, interrupts=[{"id": "interrupt-1"}])

    events = await _turn(
        gateway, Command(resume={"interrupt-1": {"decisions": [{"type": "approve"}]}})
    )

    assert stream.event_streams_opened == 1
    assert [respond["input"] for respond in stream.run.responds] == [
        {"interrupt-1": {"decisions": [{"type": "approve"}]}}
    ]
    assert _texts(events) == ["resumed: ok"]


_ANSWERED_PROMPT = {
    "method": "input.requested",
    "params": {
        "namespace": [],
        "data": {
            "interrupt_id": "interrupt-0",
            "payload": {
                "action_requests": [
                    {"name": "execute", "args": {"command": "echo old"}, "id": "t0"}
                ],
                "review_configs": _REVIEW_CONFIGS,
            },
        },
    },
    "event_id": "0-2",
}


@pytest.mark.parametrize(
    "replayed_prompt",
    [False, True],
    ids=["replay-without-prompt", "replay-with-prompt"],
)
async def test_resume_whose_run_pauses_again_ends_at_the_new_pause(replayed_prompt):
    """#738: the replayed pause already set ``stream.interrupted``, so the turn
    must end at the resumed run's own pause, not wait for a rising edge; its
    prompt is the new pause's, and the replayed ``input.requested`` of the
    interrupt just answered is not prompted again."""
    replayed_pause = [_ANSWERED_PROMPT] if replayed_prompt else []
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _text("first call done"),
            _finish(),
            _input_requested([]),
            _lifecycle("run-1", "interrupted"),
        ],
        interrupted=True,
        replay=[
            _lifecycle("run-0", "running"),
            *replayed_pause,
            _lifecycle("run-0", "interrupted"),
        ],
        stay_open=True,
    )
    gateway, threads = _gateway(
        stream, values=_EARLIER_STATE, interrupts=[{"id": "interrupt-0"}]
    )
    respond = stream.run.respond_command

    async def respond_and_pause_again(params):
        reply = await respond(params)
        threads.states[THREAD_ID]["interrupts"] = [_STATE_INTERRUPT]
        return reply

    stream.run.respond_command = respond_and_pause_again

    events = await _turn(
        gateway, Command(resume={"interrupt-0": {"decisions": [{"type": "approve"}]}})
    )

    assert [respond["input"] for respond in stream.run.responds] == [
        {"interrupt-0": {"decisions": [{"type": "approve"}]}}
    ]
    assert _texts(events) == ["first call done"]
    assert [event for event in events if event["type"] == "interrupt"] == [
        _INTERRUPT_EVENT
    ]
    assert events[-1]["type"] == "done"
    assert not stream.lost_live_events


async def test_retrying_the_same_resume_payload_resumes_again():
    """Each resume turn is its own run: the retry reads the second run's events
    and drops the first run's, which the thread stream sends again."""
    payload = {"interrupt-1": {"decisions": [{"type": "approve"}]}}
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _text("first run", "1-0"),
            _lifecycle("run-1", "completed"),
            _lifecycle("run-2", "running"),
            _text("second run", "2-0"),
            _lifecycle("run-2", "completed"),
        ],
    )
    gateway, _ = _gateway(stream, interrupts=[{"id": "interrupt-1"}])

    first = await _turn(gateway, Command(resume=payload))
    second = await _turn(gateway, Command(resume=payload))

    assert _texts(first) == ["first run"]
    assert _texts(second) == ["second run"]
    assert [respond["input"] for respond in stream.run.responds] == [payload, payload]
    assert stream.event_streams_opened == 2


class _BusyThreadStream(FakeLangGraphThreadStream):
    """Another task of the paused run streams model tokens, which carry no run
    id, so the SSE never goes quiet; the answer's run starts once that run
    ends."""

    async def _thread_events(self) -> AsyncIterator[dict[str, Any]]:
        while not self.run.started.is_set():
            yield _text("sibling token", "0-1")
            await asyncio.sleep(0.005)
        # The answer is queued: the busy run streams on, then ends.
        for _ in range(3):
            yield _text("sibling token", "0-1")
        yield _lifecycle("run-0", "interrupted")
        async for event in self._iter_events():
            yield event


@pytest.mark.parametrize(
    ("message", "busy", "answered"),
    [
        pytest.param(
            Command(resume={"interrupt-1": {"decisions": [{"type": "approve"}]}}),
            True,
            True,
            id="resume-busy",
        ),
        pytest.param(
            Command(resume={"interrupt-1": {"decisions": [{"type": "approve"}]}}),
            False,
            False,
            id="resume-idle",
        ),
        pytest.param("hi", True, False, id="new-message-busy"),
    ],
)
async def test_resume_to_a_busy_run_is_sent_past_the_replay_cap(
    message, busy, answered, monkeypatch
):
    """A sibling task's envelopes keep the SSE from going quiet; a resume on a
    busy thread is sent at the cap and queues behind that run. Anything else
    still fails at the cap."""
    # The sibling's gaps must stay under the quiet window on a ~15 ms timer
    # (Windows).
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_QUIET_SECONDS", 0.2)
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_MAX_SECONDS", 0.6)
    stream = _BusyThreadStream(
        THREAD_ID,
        events=[
            _lifecycle("run-1", "running"),
            _text("resumed"),
            _lifecycle("run-1", "completed"),
        ],
    )
    gateway, threads = _gateway(
        stream, values=_EARLIER_STATE, interrupts=[{"id": "interrupt-1"}]
    )
    if busy:
        threads.threads[0]["status"] = "busy"

    if answered:
        events = await _turn(gateway, message)
        assert _texts(events) == ["resumed"]
        assert len(stream.run.responds) == 1
    else:
        with pytest.raises(RuntimeError, match="replay did not finish"):
            await _turn(gateway, message)
        assert stream.run.responds == stream.run.starts == []


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
        quiet_seconds=0.5,
        first_event_seconds=2.0,
        max_seconds=30.0,
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
        failing(),
        quiet_seconds=0.5,
        first_event_seconds=2.0,
        max_seconds=30.0,
        clock=clock,
        wait_for=_fake_wait_for(clock, step=0.25),
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
    treat an empty ``done`` as a normal finish / "No response". A pause seen
    only as ``interrupted`` gives no prompt, so it is read from state.
    """
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), _lifecycle("run-1", "interrupted")],
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


async def test_pause_state_read_failure_on_a_resume_does_not_cancel():
    """A resume whose run pauses again leaves that run paused when the state
    read fails."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), _lifecycle("run-1", "interrupted")],
        stay_open=True,
    )
    gateway, threads = _gateway(stream, interrupts=[{"id": "interrupt-0"}])

    real_get_state = threads.get_state
    calls = {"n": 0}

    async def boom(thread_id: str):
        calls["n"] += 1
        # The pre-run read and the resume's pending-id read succeed.
        if calls["n"] <= 2:
            return await real_get_state(thread_id)
        raise ConnectionError("state store unavailable")

    threads.get_state = boom  # type: ignore[method-assign]
    answer = {"interrupt-0": {"decisions": [{"type": "approve"}]}}

    events: list[dict] = []

    async def collect() -> None:
        async with asyncio.timeout(GUARD_SECONDS):
            async for event in gateway.stream_events(
                RunRequest(message=Command(resume=answer), thread_id=THREAD_ID)
            ):
                events.append(event)

    with pytest.raises(
        server_mod._PauseStateReadError,
        match="Could not read the pause state",
    ):
        await collect()

    assert [respond["input"] for respond in stream.run.responds] == [answer]
    assert gateway.thread_store.client.runs.cancelled == []
    assert threads.state_updates == []
    assert [event["type"] for event in events] == ["error"]
    assert events[-1] == {
        "type": "error",
        "message": f"Could not read the pause state for thread {THREAD_ID}",
    }


async def test_root_failed_lifecycle_emits_partial_text_as_done():
    """A root ``failed`` ends the turn with ``done`` and any partial text.

    The path for a new message breaks on root failed and emits ``done``. An
    ``error`` event is reserved for raised exceptions.
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


async def test_skip_replay_uses_module_timeout_constants(monkeypatch):
    """Patched ``_REPLAY_*`` values apply without passing timeouts to the constructor."""
    monkeypatch.setattr(server_mod, "_REPLAY_FIRST_EVENT_SECONDS", 0.05)
    monkeypatch.setattr(server_mod, "_REPLAY_QUIET_SECONDS", 0.05)
    monkeypatch.setattr(server_mod, "_REPLAY_MAX_SECONDS", 1.0)

    async def empty() -> AsyncIterator[Any]:
        if False:  # pragma: no cover
            yield None
        await asyncio.Event().wait()

    async with asyncio.timeout(0.5):
        async with _ThreadEvents(empty()) as events:
            skipped = await events.skip_replay()

    assert skipped == 0


def test_synth_event_run_id_parses_server_lifecycle_ids(monkeypatch):
    monkeypatch.setenv("REDIS_URI", "redis://localhost")
    from langgraph_api.event_streaming.session import _synth_event_id

    from EvoScientist.gateway.server import _synth_event_run_id

    event = {"event_id": _synth_event_id("run-abc", "lc", "", "running")}
    assert _synth_event_run_id(event) == "run-abc"
