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


async def test_events_stream_ending_early_while_status_is_unreadable_raises(
    quiet_poll,
):
    """A turn whose stream ended waits for its run by status; when the status
    cannot be read three times in a row it fails as "ended before the run finished"."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs

    async def unreadable(thread_id: str, run_id: str) -> dict[str, Any]:
        runs.gets.append((thread_id, run_id))
        raise ConnectionError("server busy")

    runs.get = unreadable

    with pytest.raises(RuntimeError, match="ended before the run finished"):
        await _turn(gateway, "hi")

    assert len(runs.gets) == 3
    # The run that could not be followed is cancelled.
    assert runs.cancelled == [(THREAD_ID, ["run-1"])]


@pytest.mark.parametrize("drop", ["raises", "ends"])
async def test_events_stream_lost_after_a_prompt_keeps_the_pause(drop):
    """A stream lost between the envelope's prompt and the root
    ``interrupted``: the turn waits for the run and reads it back. With
    nothing more to recover (the read-back ends the same way, the state is
    empty) it fails, without cancelling or repairing the run the prompt
    belongs to."""
    stream = FakeLangGraphThreadStream(THREAD_ID, events=[])

    async def lost() -> AsyncIterator[dict[str, Any]]:
        await stream.run.started.wait()
        yield _lifecycle("run-1", "running")
        yield _input_requested([])
        if drop == "raises":
            raise ConnectionError("provider connection lost")

    gateway, threads = _gateway(stream)
    events: list[dict] = []

    async def collect() -> None:
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id=THREAD_ID)
        ):
            events.append(event)

    with (
        patch.object(
            FakeLangGraphThreadStream, "events", property(lambda self: lost())
        ),
        pytest.raises(RuntimeError, match=r"^Lost the run's live events$"),
    ):
        await collect()

    assert [e["type"] for e in events] == ["interrupt", "error"]
    # It waited for the run by its status, then read the run back.
    assert gateway.thread_store.client.runs.gets == [(THREAD_ID, "run-1")]
    assert gateway.thread_store.client.runs.cancelled == []
    assert threads.state_updates == []


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


async def test_root_failed_lifecycle_ends_the_turn_with_an_error():
    """A root ``failed`` ends the turn with ``error`` after any partial text."""
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
    events: list[dict] = []

    async def collect() -> None:
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id=THREAD_ID)
        ):
            events.append(event)

    with pytest.raises(RuntimeError, match="Run failed"):
        async with asyncio.timeout(GUARD_SECONDS):
            await collect()

    assert [event["type"] for event in events] == ["text", "error"]


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


# --- #709: the run finishes but its live envelopes do not arrive -------------


@pytest.fixture
def quiet_poll(monkeypatch):
    """Short quiet and grace windows, so a silent stream is noticed soon.

    Well above a busy runner's scheduling gaps: a window that runs out early
    sends the turn down the recovery path while envelopes are still coming.
    """
    monkeypatch.setattr(server_mod, "_QUIET_POLL_SECONDS", 0.2)
    monkeypatch.setattr(server_mod, "_FINISHED_GRACE_SECONDS", 0.2)


def _tool(phase: str, entry_id: str, **data: Any) -> dict[str, Any]:
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {"event": phase, "tool_call_id": "call-1", **data},
        },
        "event_id": entry_id,
    }


_TEXT_RUN = [
    _lifecycle("run-1", "running"),
    _text("Hello", "1-0"),
    _text(" world", "1-1"),
    _finish("1-2"),
    _lifecycle("run-1", "completed"),
]
_TOOL_RUN = [
    _lifecycle("run-1", "running"),
    _tool("tool-started", "1-0", tool_name="execute", input={"command": "ls"}),
    _tool("tool-finished", "1-1", tool_name="execute", output="a.txt"),
    _text("One file", "1-2"),
    _finish("1-3"),
    _lifecycle("run-1", "completed"),
]
_PAUSE_RUN = [
    _lifecycle("run-1", "running"),
    _text("Let me check", "1-0"),
    _finish("1-1"),
    _input_requested([]),
    _lifecycle("run-1", "interrupted"),
]
_SUBAGENT_RUN = [
    _lifecycle("run-1", "running"),
    _lifecycle("run-1", "started", ["research:task-1"]),
    {
        "method": "custom",
        "params": {
            "namespace": [],
            "data": {
                "evoscientist": {
                    "kind": "fallback_notice",
                    "text": "fb",
                    "style": "red",
                }
            },
        },
        "event_id": "1-0",
    },
    _lifecycle("run-1", "completed", ["research:task-1"]),
    _text("Done", "1-1"),
    _finish("1-2"),
    _lifecycle("run-1", "completed"),
]


class _Notices:
    """Session sink double: the lines a frontend would show in the turn."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        self.lines.append(f"fallback: {text}")

    def emit_notice(self, text: str, style: str = "yellow") -> None:
        self.lines.append(text)


async def _live_events(run: list[dict[str, Any]], **gateway_kwargs: Any) -> list[dict]:
    gateway, _ = _gateway(
        FakeLangGraphThreadStream(THREAD_ID, events=run, stay_open=True),
        **gateway_kwargs,
    )
    return await _turn(gateway, "hi")


@pytest.mark.parametrize("live_limit", [0, 2], ids=["nothing-live", "partly-live"])
@pytest.mark.parametrize(
    ("run", "interrupts"),
    [
        pytest.param(_TEXT_RUN, None, id="text"),
        pytest.param(_TOOL_RUN, None, id="tool-call"),
        pytest.param(_PAUSE_RUN, [], id="pause"),
        pytest.param(_SUBAGENT_RUN, None, id="subagent-and-custom"),
    ],
)
async def test_finished_run_with_lost_live_events_is_read_back(
    run, interrupts, live_limit, quiet_poll, caplog
):
    """The turn's events equal a live turn's: nothing twice, nothing missing."""
    expected = await _live_events(run, interrupts=interrupts)
    stream = FakeLangGraphThreadStream(THREAD_ID, events=run, live_limit=live_limit)
    gateway, _ = _gateway(stream, interrupts=interrupts)
    gateway.events = _Notices()

    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        events = await _turn(gateway, "hi")

    assert events == expected
    if run is _SUBAGENT_RUN:
        assert [e["type"] for e in events if e["type"].startswith("subagent_")] == [
            "subagent_start",
            "subagent_end",
        ]
    assert stream.event_streams_opened == 2
    assert gateway.thread_store.client.runs.cancelled == []
    assert "Lost the live events of run run-1" in caplog.text
    # The run's custom middleware line arrives once, with the read-back's events.
    custom = ["fallback: fb"] if run is _SUBAGENT_RUN else []
    assert gateway.events.lines == [server_mod._RECOVERED_TURN_NOTICE, *custom]


async def test_read_back_drops_the_earlier_runs_it_replays(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=_TEXT_RUN,
        replay=_earlier_run("run-0", "old answer"),
        live_limit=0,
    )
    gateway, _ = _gateway(stream, values=_EARLIER_STATE)

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert events[-1]["response"] == "Hello world"


async def test_resume_turn_with_lost_live_events_is_read_back(quiet_poll):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream, interrupts=[{"id": "interrupt-0"}])

    events = await _turn(
        gateway, Command(resume={"interrupt-0": {"decisions": [{"type": "approve"}]}})
    )

    assert _texts(events) == ["Hello", " world"]
    assert len(stream.run.responds) == 1


async def test_turn_with_flowing_envelopes_asks_for_no_status(quiet_poll):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, stay_open=True)
    gateway, _ = _gateway(stream)
    gateway.events = _Notices()

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert gateway.thread_store.client.runs.gets == []
    assert stream.event_streams_opened == 1
    assert gateway.events.lines == []


async def test_lost_failed_run_raises_its_own_error(quiet_poll):
    failed = _lifecycle("run-1", "failed")
    failed["params"]["data"]["error"] = "boom"
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=[_lifecycle("run-1", "running"), failed], live_limit=0
    )
    stream.run.status = "error"
    gateway, _ = _gateway(stream)
    gateway.events = _Notices()

    with pytest.raises(RuntimeError, match="Run failed: boom"):
        await _turn(gateway, "hi")

    assert gateway.thread_store.client.runs.cancelled == []
    # The turn ends with the run's error, not a result.
    assert gateway.events.lines == []


@pytest.mark.parametrize(
    "statuses",
    [
        pytest.param(["pending", "running", "retry", "success"], id="still-active"),
        pytest.param([ConnectionError("down"), "success"], id="request-fails-once"),
    ],
)
async def test_turn_keeps_waiting_until_the_run_has_finished(
    statuses, quiet_poll, caplog
):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    answers = list(statuses)

    async def get(thread_id: str, run_id: str) -> dict[str, Any]:
        runs.gets.append((thread_id, run_id))
        # Nothing is read back while the run is still active.
        assert stream.event_streams_opened == 1
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"run_id": run_id, "status": answer}

    runs.get = get

    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert len(runs.gets) == len(statuses)
    assert ("Could not read the status of run run-1" in caplog.text) == isinstance(
        statuses[0], Exception
    )


async def test_run_the_server_no_longer_has_ends_the_turn(quiet_poll):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    start = stream.run.start

    async def start_and_forget(**kwargs: Any) -> dict[str, Any]:
        reply = await start(**kwargs)
        stream.run.runs.clear()
        return reply

    stream.run.start = start_and_forget

    with pytest.raises(RuntimeError, match="Run run-1 no longer exists"):
        await _turn(gateway, "hi")

    assert gateway.thread_store.client.runs.cancelled == []


def _state_after_start(
    stream: FakeLangGraphThreadStream,
    threads: FakeLangGraphThreadsClient,
    state: dict[str, Any],
) -> None:
    """Thread state as the run leaves it, also its last checkpoint; the
    pre-run read and the run's input checkpoint see the old one."""
    start = stream.run.start

    async def start_and_write(**kwargs: Any) -> dict[str, Any]:
        reply = await start(**kwargs)
        threads.input_states[(THREAD_ID, reply["run_id"])] = threads.states[THREAD_ID]
        threads.states[THREAD_ID] = state
        threads.run_states[(THREAD_ID, reply["run_id"])] = state
        return reply

    stream.run.start = start_and_write


_NEW_STATE = {
    "messages": [
        *_EARLIER_STATE["messages"],
        {"type": "human", "content": "hi", "id": "human-1"},
        {"type": "ai", "content": "Hello world", "id": "ai-1"},
    ]
}


_OTHER_STATE = {
    "messages": [
        *_EARLIER_STATE["messages"],
        {"type": "ai", "content": "Something else", "id": "ai-1"},
    ]
}
_TWO_MESSAGE_RUN = [
    _lifecycle("run-1", "running"),
    _text("First", "1-0"),
    _text(" part", "1-1"),
    _finish("1-2"),
    _tool("tool-started", "1-3", tool_name="execute", input={"command": "ls"}),
    _tool("tool-finished", "1-4", tool_name="execute", output="a.txt"),
    _text("Second", "1-5"),
    _finish("1-6"),
    _lifecycle("run-1", "completed"),
]
_TWO_MESSAGE_STATE = {
    "messages": [
        *_EARLIER_STATE["messages"],
        {"type": "human", "content": "hi", "id": "human-1"},
        {
            "type": "ai",
            "content": "First part",
            "id": "ai-1",
            "tool_calls": [
                {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
            ],
        },
        {
            "type": "tool",
            "content": "a.txt",
            "id": "tool-1",
            "name": "execute",
            "tool_call_id": "call-1",
        },
        {"type": "ai", "content": [{"type": "text", "text": "Second"}], "id": "ai-2"},
    ]
}


@pytest.mark.parametrize(
    "live_limit", range(len(_TWO_MESSAGE_RUN)), ids=lambda n: f"{n}-live"
)
async def test_cut_off_answer_is_completed_from_the_thread_state(
    live_limit, quiet_poll
):
    """Wherever the live envelopes stop, the turn ends like a live one: the
    whole answer, and each tool call and result once."""
    expected = await _live_events(_TWO_MESSAGE_RUN, values=_EARLIER_STATE)
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TWO_MESSAGE_RUN, live_limit=live_limit, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _TWO_MESSAGE_STATE})

    events = await _turn(gateway, "hi")

    assert "".join(_texts(events)) == "".join(_texts(expected))
    assert [e for e in events if e["type"] != "text"] == [
        e for e in expected if e["type"] != "text"
    ]
    assert events[-1]["response"] == "First part\n\nSecond"


async def test_failed_read_back_falls_back_to_the_thread_state(quiet_poll, caplog):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=1
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _NEW_STATE})

    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello world"]
    assert events[-1]["response"] == "Hello world"
    assert "using the thread state" in caplog.text


async def test_failed_read_back_of_a_pause_shows_the_prompt_from_state(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_PAUSE_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, interrupts=[])
    _state_after_start(
        stream, threads, {"values": {}, "interrupts": [_STATE_INTERRUPT]}
    )

    events = await _turn(gateway, "hi")

    assert [event for event in events if event["type"] == "interrupt"] == [
        _INTERRUPT_EVENT
    ]
    assert gateway.thread_store.client.runs.cancelled == []


_UNSHOWN = r"^The run finished, but its reply could not be shown$"


@pytest.mark.parametrize(
    ("live_limit", "status", "state", "message"),
    [
        # The run succeeded: its reply is in its checkpoint, not shown here.
        pytest.param(0, "success", {"values": {}}, _UNSHOWN, id="empty-state"),
        pytest.param(0, "success", None, _UNSHOWN, id="state-read-fails"),
        # The text shown is not the start of the state's answer.
        pytest.param(2, "success", {"values": _OTHER_STATE}, _UNSHOWN, id="other-text"),
        pytest.param(
            0,
            "interrupted",
            {"values": _NEW_STATE},
            r"^Lost the run's live events: .*status interrupted$",
        ),
        pytest.param(
            0,
            "error",
            {"values": _NEW_STATE},
            r"^Lost the run's live events: .*status error$",
        ),
    ],
)
async def test_turn_raises_when_read_back_and_state_both_fail(
    live_limit, status, state, message, quiet_poll
):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=live_limit, read_back_limit=0
    )
    stream.run.status = status
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    if state is not None:
        _state_after_start(stream, threads, state)
    else:
        start = stream.run.start

        async def start_and_break_state(**kwargs: Any) -> dict[str, Any]:
            reply = await start(**kwargs)
            del threads.states[THREAD_ID]
            return reply

        stream.run.start = start_and_break_state
    events: list[dict] = []

    async def collect() -> None:
        async with asyncio.timeout(GUARD_SECONDS):
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id=THREAD_ID)
            ):
                events.append(event)

    gateway.events = _Notices()

    with pytest.raises(RuntimeError, match=message):
        await collect()

    assert events[-1]["type"] == "error"
    # No "the result is shown now" for a turn that ends without one.
    assert gateway.events.lines == []
    # The run has finished: the cleanup has nothing to cancel.
    assert gateway.thread_store.client.runs.cancelled == []


class _OtherRunStream(FakeLangGraphThreadStream):
    """The live connection only ever carries another run's envelopes."""

    async def _thread_events(self) -> AsyncIterator[dict[str, Any]]:
        if self.event_streams_opened > 1:
            for event in self.queued_events:
                yield event
            return
        await self.run.started.wait()
        yield _lifecycle("run-9", "running")
        while True:
            yield _text("someone else's token", "9-0")
            await asyncio.sleep(0.005)


async def test_another_runs_envelopes_do_not_count_as_this_runs(quiet_poll):
    gateway, _ = _gateway(_OtherRunStream(THREAD_ID, events=_TEXT_RUN))

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]


async def test_recovered_turn_shows_one_line_through_the_session_sink(quiet_poll):
    """The TUI and the Rich CLI render the session sink's lines in the turn."""
    from EvoScientist.stream.sink import SessionEventSink

    shown: list[tuple[str, str]] = []
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    gateway.events = SessionEventSink(
        fallback_display=lambda text, style: shown.append((text, style))
    )

    await _turn(gateway, "hi")

    assert shown == [(server_mod._RECOVERED_TURN_NOTICE, "yellow")]


async def test_recovered_turn_without_a_frontend_sink_only_logs(quiet_poll):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    assert gateway.events is None

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]


def _model_call(call: str, phase: str, entry_id: str, **data: Any) -> dict[str, Any]:
    return {
        "method": "custom",
        "params": {
            "namespace": [],
            "data": {"type": "model_call", "call": call, "phase": phase, **data},
        },
        "event_id": entry_id,
    }


async def test_answer_cut_off_after_a_discarded_retry_is_completed(quiet_poll):
    """Text a failed provider call streamed is rolled back before the match:
    state only has the answer of the call that replaced it."""
    run = [
        _lifecycle("run-1", "running"),
        _model_call("call-1", "start", "1-0"),
        _text("A draft that fails", "1-1"),
        _model_call("call-1", "end", "1-2", status="error"),
        _model_call("call-2", "start", "1-3"),
        _text("Hello", "1-4"),
        _text(" world", "1-5"),
        _finish("1-6"),
        _lifecycle("run-1", "completed"),
    ]
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=6, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _NEW_STATE})

    events = await _turn(gateway, "hi")

    assert [event["type"] for event in events if event["type"] != "text"] == [
        "model_output_discarded",
        "done",
    ]
    assert events[-1]["response"] == "Hello world"


def _thinking(text: str, entry_id: str) -> dict[str, Any]:
    return {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": {
                "event": "content-block-delta",
                "delta": {"type": "reasoning-delta", "reasoning": text},
            },
        },
        "event_id": entry_id,
    }


_THINKING_RUN = [
    _lifecycle("run-1", "running"),
    _text("First", "1-0"),
    _finish("1-1"),
    _thinking("weighing it", "1-2"),
    _text("Second", "1-3"),
    _text(" ", "1-4"),
    _text("part", "1-5"),
    _finish("1-6"),
    _lifecycle("run-1", "completed"),
]
_THINKING_STATE = {
    "messages": [
        *_EARLIER_STATE["messages"],
        {"type": "ai", "content": "First", "id": "ai-1"},
        {
            "type": "ai",
            "content": [
                {"type": "reasoning", "reasoning": "weighing it"},
                {"type": "text", "text": "Second part"},
            ],
            "id": "ai-2",
        },
    ]
}


@pytest.mark.parametrize(
    "live_limit", range(len(_THINKING_RUN)), ids=lambda n: f"{n}-live"
)
async def test_state_completion_shows_thinking_once_and_keeps_word_spacing(
    live_limit, quiet_poll
):
    """Thinking of a message the live stream had not begun comes from state;
    thinking already streamed is not repeated; a space the live stream was
    holding back for the next word is not doubled."""
    expected = await _live_events(_THINKING_RUN, values=_EARLIER_STATE)
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_THINKING_RUN, live_limit=live_limit, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _THINKING_STATE})

    events = await _turn(gateway, "hi")

    assert events[-1]["response"] == "First\n\nSecond part"
    assert [e for e in events if e["type"] != "text"] == [
        e for e in expected if e["type"] != "text"
    ]


_PAUSED_ON_A_TOOL = [
    {"type": "human", "content": "list files", "id": "human-0"},
    {
        "type": "ai",
        "content": "",
        "id": "ai-0",
        "tool_calls": [{"name": "execute", "args": {"command": "ls"}, "id": "call-1"}],
    },
]
_RESUMED_RUN = [
    _lifecycle("run-1", "running"),
    _tool("tool-finished", "1-0", tool_name="execute", output="a.txt"),
    _text("One file", "1-1"),
    _finish("1-2"),
    _lifecycle("run-1", "completed"),
]
_RESUME = Command(resume={"interrupt-0": {"decisions": [{"type": "approve"}]}})


def _resumed_gateway(
    read_back_limit: int | None,
    interrupts: list[dict[str, Any]] | None = None,
    run: list[dict[str, Any]] = _RESUMED_RUN,
) -> tuple[LangGraphServerGateway, FakeLangGraphThreadsClient]:
    """A thread paused on a tool call; the resumed run adds the tool's result
    and the answer, and pauses again on ``interrupts``; its live envelopes
    are lost."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=0, read_back_limit=read_back_limit
    )
    gateway, threads = _gateway(
        stream,
        values={"messages": _PAUSED_ON_A_TOOL},
        interrupts=[{"id": "interrupt-0"}],
    )
    respond = stream.run.respond_command

    async def respond_and_finish(params):
        reply = await respond(params)
        resumed = [
            *_PAUSED_ON_A_TOOL,
            {
                "type": "tool",
                "content": "a.txt",
                "id": "tool-1",
                "name": "execute",
                "tool_call_id": "call-1",
            },
            {"type": "ai", "content": "One file", "id": "ai-1"},
        ]
        state: dict[str, Any] = {"values": {"messages": resumed}}
        if interrupts:
            state["interrupts"] = interrupts
        threads.states[THREAD_ID] = state
        threads.run_states[(THREAD_ID, stream.run.run_id)] = state
        return reply

    stream.run.respond_command = respond_and_finish
    return gateway, threads


async def test_resume_turn_read_back_in_full_shows_the_resumed_tools_result_once(
    quiet_poll,
):
    gateway, _ = _resumed_gateway(read_back_limit=None)

    events = await _turn(gateway, _RESUME)

    assert [(e["type"], e.get("content")) for e in events] == [
        ("tool_result", "a.txt"),
        ("text", "One file"),
        ("done", "One file"),
    ]


async def test_resume_turn_is_not_completed_from_state(quiet_poll):
    """A resumed run has no input checkpoint, and a run it queued behind can
    add messages after the pre-run read: the turn cannot tell its own apart."""
    gateway, threads = _resumed_gateway(read_back_limit=0)

    with pytest.raises(RuntimeError, match=_UNSHOWN):
        await _turn(gateway, _RESUME)

    assert gateway.thread_store.client.runs.cancelled == []
    # Only the run's own last checkpoint, for its prompts.
    assert [metadata for _, _, metadata in threads.history_gets] == [
        {"run_id": "run-1"}
    ]


async def test_resume_turn_that_paused_again_shows_its_prompt_from_state(
    quiet_poll,
):
    """The prompts in the run's own checkpoint are its own: the turn ends
    with them, without rebuilding the answer from the messages."""
    gateway, _ = _resumed_gateway(read_back_limit=0, interrupts=[_STATE_INTERRUPT])

    events = await _turn(gateway, _RESUME)

    assert [e["type"] for e in events] == ["interrupt", "done"]
    assert gateway.thread_store.client.runs.cancelled == []


async def test_resume_turn_shows_a_prompt_the_read_back_held_once(quiet_poll):
    """The read-back holds the new prompt but not the run's end, and the run's
    checkpoint has it too: it is shown once."""
    run = [
        _lifecycle("run-1", "running"),
        _tool("tool-finished", "1-0", tool_name="execute", output="a.txt"),
        _input_requested([]),
        _lifecycle("run-1", "interrupted"),
    ]
    gateway, _ = _resumed_gateway(
        read_back_limit=3, interrupts=[_STATE_INTERRUPT], run=run
    )

    events = await _turn(gateway, _RESUME)

    assert [e for e in events if e["type"] == "interrupt"] == [_INTERRUPT_EVENT]


async def test_resume_turn_takes_no_prompt_from_another_runs_checkpoint(
    quiet_poll,
):
    """Only this run's checkpoint is read: a later run's prompt is not this
    turn's."""
    gateway, threads = _resumed_gateway(read_back_limit=0)
    threads.run_states[(THREAD_ID, "run-9")] = {
        "values": {"messages": _PAUSED_ON_A_TOOL},
        "interrupts": [_STATE_INTERRUPT],
    }

    with pytest.raises(RuntimeError, match=_UNSHOWN):
        await _turn(gateway, _RESUME)


@pytest.mark.parametrize(
    "live_limit", [0, 2, 3, 4], ids=["nothing", "cut", "all-text", "finished"]
)
async def test_state_completion_reports_the_usage_live_never_got(
    live_limit, quiet_poll
):
    """Usage arrives with a message's end; a message whose end was not shown
    live gets its usage from the stored message, once."""
    finish = _finish("1-2")
    finish["params"]["data"]["usage"] = {"input_tokens": 10, "output_tokens": 5}
    run = [
        _lifecycle("run-1", "running"),
        _text("Hello", "1-0"),
        _text(" world", "1-1"),
        finish,
        _lifecycle("run-1", "completed"),
    ]
    state = {
        "messages": [
            *_EARLIER_STATE["messages"],
            {"type": "human", "content": "hi", "id": "human-1"},
            {
                "type": "ai",
                "content": "Hello world",
                "id": "ai-1",
                "usage_metadata": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]
    }
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=live_limit, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": state})

    events = await _turn(gateway, "hi")

    assert [e for e in events if e["type"] == "usage_stats"] == [
        {"type": "usage_stats", "input_tokens": 10, "output_tokens": 5}
    ]
    assert events[-1]["response"] == "Hello world"


class _TailAfterStatusStream(FakeLangGraphThreadStream):
    """The run's first envelopes arrive, then nothing until the turn has asked
    for the status, then the rest: the grace period must catch them live."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.status_asked = asyncio.Event()

    async def _thread_events(self) -> AsyncIterator[dict[str, Any]]:
        await self.run.started.wait()
        yield self.queued_events[0]
        yield self.queued_events[1]
        await self.status_asked.wait()
        for event in self.queued_events[2:]:
            yield event
        await asyncio.Event().wait()


async def test_envelopes_arriving_in_the_grace_period_are_read_live(quiet_poll):
    stream = _TailAfterStatusStream(THREAD_ID, events=_TEXT_RUN)
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    get = runs.get

    async def get_and_release(thread_id: str, run_id: str) -> dict[str, Any]:
        run = await get(thread_id, run_id)
        stream.status_asked.set()
        return run

    runs.get = get_and_release
    gateway.events = _Notices()

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert stream.event_streams_opened == 1
    assert gateway.events.lines == []


class _BrokenReadBackStream(FakeLangGraphThreadStream):
    """The read-back's connection fails."""

    @property
    def events(self) -> AsyncIterator[dict[str, Any]]:
        self.event_streams_opened += 1
        if self.event_streams_opened > 1:
            raise ConnectionError("server gone")
        return self._lossy_events(False)


async def test_failed_read_back_connection_falls_back_to_state(quiet_poll, caplog):
    stream = _BrokenReadBackStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _NEW_STATE})

    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello world"]
    assert "Could not read the finished run back" in caplog.text


async def test_a_failing_notice_sink_does_not_end_the_turn(quiet_poll):
    class _Broken(_Notices):
        def emit_notice(self, text: str, style: str = "yellow") -> None:
            raise RuntimeError("display gone")

    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    gateway.events = _Broken()

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]


async def test_abandoning_a_recovered_turn_does_not_cancel_the_finished_run(
    quiet_poll,
):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    turn = gateway.stream_events(RunRequest(message="hi", thread_id=THREAD_ID))

    async with asyncio.timeout(GUARD_SECONDS):
        first = await anext(turn)
        await turn.aclose()

    assert first["type"] == "text"
    assert gateway.thread_store.client.runs.cancelled == []


async def test_shown_text_longer_than_the_stored_answer_raises(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=3, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    short = {
        "messages": [
            *_EARLIER_STATE["messages"],
            {"type": "ai", "content": "Hello", "id": "ai-1"},
        ]
    }
    _state_after_start(stream, threads, {"values": short})

    with pytest.raises(RuntimeError, match=_UNSHOWN):
        await _turn(gateway, "hi")


async def test_thinking_of_a_message_whose_tool_call_was_shown_is_not_repeated(
    quiet_poll,
):
    """A provider that does not stream its reasoning still stores it; a
    message whose tool call the live stream showed has begun."""
    run = [
        _lifecycle("run-1", "running"),
        _text("First", "1-0"),
        _finish("1-1"),
        _tool("tool-started", "1-2", tool_name="execute", input={"command": "ls"}),
        _tool("tool-finished", "1-3", tool_name="execute", output="a.txt"),
        _lifecycle("run-1", "completed"),
    ]
    state = {
        "messages": [
            *_EARLIER_STATE["messages"],
            {"type": "human", "content": "hi", "id": "human-1"},
            {"type": "ai", "content": "First", "id": "ai-1"},
            {
                "type": "ai",
                "content": [{"type": "reasoning", "reasoning": "list it"}],
                "id": "ai-2",
                "tool_calls": [
                    {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
                ],
            },
            {
                "type": "tool",
                "content": "a.txt",
                "id": "tool-1",
                "name": "execute",
                "tool_call_id": "call-1",
            },
        ]
    }
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=4, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": state})

    events = await _turn(gateway, "hi")

    assert [e["type"] for e in events] == ["text", "tool_call", "tool_result", "done"]


async def test_repeated_status_failures_log_one_traceback(quiet_poll, caplog):
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN, live_limit=0)
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    get = runs.get
    calls = {"n": 0}

    async def flaky(thread_id: str, run_id: str) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] <= 3:
            raise ConnectionError("busy")
        return await get(thread_id, run_id)

    runs.get = flaky

    with caplog.at_level("DEBUG", logger="EvoScientist.gateway.server"):
        events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    failures = [r for r in caplog.records if "Could not read the status" in r.message]
    assert [r.levelname for r in failures] == ["WARNING", "DEBUG", "DEBUG"]
    assert [bool(r.exc_info) for r in failures] == [True, False, False]


async def test_next_within_returns_a_queued_envelope_with_no_time_left():
    async with _ThreadEvents(_source([{"n": 1}], then_wait=True)) as events:
        while events._queue.empty():
            await asyncio.sleep(0)

        assert await events.next_within(0) == {"n": 1}


class _EndAfterStatusStream(FakeLangGraphThreadStream):
    """The live connection carries the run's first envelopes and ends once the
    turn has asked for the status; a new connection replays the whole run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.status_asked = asyncio.Event()

    async def _thread_events(self) -> AsyncIterator[dict[str, Any]]:
        if self.event_streams_opened > 1:
            for event in self.queued_events:
                yield event
            await asyncio.Event().wait()
        await self.run.started.wait()
        yield self.queued_events[0]
        yield self.queued_events[1]
        await self.status_asked.wait()


async def test_stream_ending_after_a_finished_status_reads_the_run_back(quiet_poll):
    stream = _EndAfterStatusStream(THREAD_ID, events=_TEXT_RUN)
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    get = runs.get

    async def get_and_end(thread_id: str, run_id: str) -> dict[str, Any]:
        run = await get(thread_id, run_id)
        stream.status_asked.set()
        return run

    runs.get = get_and_end
    gateway.events = _Notices()

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert stream.event_streams_opened == 2
    # The run had finished when the stream ended: no wait to announce.
    assert gateway.events.lines == [server_mod._RECOVERED_TURN_NOTICE]


def _two_runs_state(later: bool) -> dict[str, Any]:
    messages = [*_NEW_STATE["messages"]]
    if later:
        messages += [
            {"type": "human", "content": "next", "id": "human-2"},
            {"type": "ai", "content": "Next answer", "id": "ai-2"},
        ]
    return {"values": {"messages": messages}}


async def test_state_completion_uses_the_runs_own_checkpoint(quiet_poll):
    """Another run may already have answered on the thread: the turn shows
    only what this run wrote."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, _two_runs_state(later=True))
    start = stream.run.start

    async def start_then_a_later_run(**kwargs: Any) -> dict[str, Any]:
        reply = await start(**kwargs)
        threads.run_states[(THREAD_ID, reply["run_id"])] = _two_runs_state(later=False)
        return reply

    stream.run.start = start_then_a_later_run

    events = await _turn(gateway, "hi")

    assert events[-1]["response"] == "Hello world"
    assert threads.history_gets == [
        (THREAD_ID, 1, {"run_id": "run-1", "source": "input"}),
        (THREAD_ID, 1, {"run_id": "run-1"}),
    ]


async def test_state_completion_without_a_checkpoint_of_the_run_raises(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    start = stream.run.start

    async def start_without_a_checkpoint(**kwargs: Any) -> dict[str, Any]:
        # The thread has a state, but no checkpoint of this run.
        reply = await start(**kwargs)
        threads.states[THREAD_ID] = {"values": _NEW_STATE}
        return reply

    stream.run.start = start_without_a_checkpoint

    with pytest.raises(RuntimeError, match=_UNSHOWN):
        await _turn(gateway, "hi")


def _message_start(message_id: str, entry_id: str) -> dict[str, Any]:
    return {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": {"event": "message-start", "role": "ai", "id": message_id},
        },
        "event_id": entry_id,
    }


def _finish_with_usage(entry_id: str, inp: int, out: int) -> dict[str, Any]:
    finish = _finish(entry_id)
    finish["params"]["data"]["usage"] = {"input_tokens": inp, "output_tokens": out}
    return finish


def _usage(inp: int, out: int) -> dict[str, int]:
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


# A tool-only message, then an answer that thinks first; ids as the server
# sends them in ``message-start``.
_TOOL_THEN_THINKING_RUN = [
    _lifecycle("run-1", "running"),
    _message_start("ai-1", "1-0"),
    {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": {
                "event": "content-block-finish",
                "content": {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "execute",
                    "args": {"command": "ls"},
                },
            },
        },
        "event_id": "1-1",
    },
    _finish_with_usage("1-2", 10, 5),
    _tool("tool-started", "1-3", tool_name="execute", input={"command": "ls"}),
    _tool("tool-finished", "1-4", tool_name="execute", output="a.txt"),
    _message_start("ai-2", "1-5"),
    _thinking("weighing it", "1-6"),
    _text("One file", "1-7"),
    _finish_with_usage("1-8", 20, 7),
    _lifecycle("run-1", "completed"),
]
_TOOL_THEN_THINKING_STATE = {
    "messages": [
        *_EARLIER_STATE["messages"],
        {"type": "human", "content": "hi", "id": "human-1"},
        {
            "type": "ai",
            "content": "",
            "id": "ai-1",
            "tool_calls": [
                {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
            ],
            "usage_metadata": _usage(10, 5),
        },
        {
            "type": "tool",
            "content": "a.txt",
            "id": "tool-1",
            "name": "execute",
            "tool_call_id": "call-1",
        },
        {
            "type": "ai",
            "content": [
                {"type": "reasoning", "reasoning": "weighing it"},
                {"type": "text", "text": "One file"},
            ],
            "id": "ai-2",
            "usage_metadata": _usage(20, 7),
        },
    ]
}


@pytest.mark.parametrize(
    "live_limit", range(len(_TOOL_THEN_THINKING_RUN)), ids=lambda n: f"{n}-live"
)
async def test_state_completion_after_a_tool_only_message_repeats_nothing(
    live_limit, quiet_poll
):
    """A tool-only message whose end arrived live does not get its usage
    again, and thinking streamed for the next message is not repeated."""
    expected = await _live_events(_TOOL_THEN_THINKING_RUN, values=_EARLIER_STATE)
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=_TOOL_THEN_THINKING_RUN,
        live_limit=live_limit,
        read_back_limit=0,
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _TOOL_THEN_THINKING_STATE})

    events = await _turn(gateway, "hi")

    assert [e for e in events if e["type"] != "text"] == [
        e for e in expected if e["type"] != "text"
    ]
    assert events[-1]["response"] == "One file"


_TWO_REPLIES_RUN = [
    _lifecycle("run-1", "running"),
    _text("First", "1-0"),
    _finish_with_usage("1-1", 10, 5),
    _text("Second", "1-2"),
    _finish_with_usage("1-3", 20, 7),
    _lifecycle("run-1", "completed"),
]
_TWO_REPLIES_STATE = {
    "messages": [
        *_EARLIER_STATE["messages"],
        {"type": "human", "content": "hi", "id": "human-1"},
        {
            "type": "ai",
            "content": "First",
            "id": "ai-1",
            "usage_metadata": _usage(10, 5),
        },
        {
            "type": "ai",
            "content": "Second",
            "id": "ai-2",
            "usage_metadata": _usage(20, 7),
        },
    ]
}


# The first reply also calls a tool: the stream can drop while its tool call
# streams, after its text and before its end.
_REPLY_WITH_TOOL_RUN = [
    _lifecycle("run-1", "running"),
    _text("First", "1-0"),
    {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": {
                "event": "content-block-finish",
                "content": {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "execute",
                    "args": {"command": "ls"},
                },
            },
        },
        "event_id": "1-1",
    },
    _finish_with_usage("1-2", 10, 5),
    _tool("tool-started", "1-3", tool_name="execute", input={"command": "ls"}),
    _tool("tool-finished", "1-4", tool_name="execute", output="a.txt"),
    _text("Second", "1-5"),
    _finish_with_usage("1-6", 20, 7),
    _lifecycle("run-1", "completed"),
]
_REPLY_WITH_TOOL_STATE = {
    "messages": [
        *_TWO_REPLIES_STATE["messages"][:-2],
        {
            **_TWO_REPLIES_STATE["messages"][-2],
            "tool_calls": [
                {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
            ],
        },
        {
            "type": "tool",
            "content": "a.txt",
            "id": "tool-1",
            "name": "execute",
            "tool_call_id": "call-1",
        },
        _TWO_REPLIES_STATE["messages"][-1],
    ]
}


@pytest.mark.parametrize(
    ("run", "state", "live_limit"),
    [
        *(
            pytest.param(_TWO_REPLIES_RUN, _TWO_REPLIES_STATE, n, id=f"text-{n}-live")
            for n in range(len(_TWO_REPLIES_RUN))
        ),
        *(
            pytest.param(
                _REPLY_WITH_TOOL_RUN, _REPLY_WITH_TOOL_STATE, n, id=f"tool-{n}-live"
            )
            for n in range(len(_REPLY_WITH_TOOL_RUN))
        ),
    ],
)
async def test_state_completion_reports_each_replys_usage_once(
    run, state, live_limit, quiet_poll
):
    """A reply whose text all arrived live but whose end did not gets its
    usage, also when the next reply holds the cut."""
    expected = await _live_events(run, values=_EARLIER_STATE)
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=live_limit, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": state})

    events = await _turn(gateway, "hi")

    assert [e for e in events if e["type"] != "text"] == [
        e for e in expected if e["type"] != "text"
    ]
    assert events[-1]["response"] == "First\n\nSecond"


# Another run answered on the thread after the pre-run read and before this
# turn's run started (it queued behind that run).
_OTHER_RUN_MESSAGES = [
    {"type": "human", "content": "other question", "id": "human-b"},
    {"type": "ai", "content": "Other answer", "id": "ai-b"},
]
_AFTER_THE_OTHER_RUN = {"messages": [*_EARLIER_STATE["messages"], *_OTHER_RUN_MESSAGES]}
_THIS_RUN_DONE = {
    "messages": [
        *_AFTER_THE_OTHER_RUN["messages"],
        {"type": "human", "content": "hi", "id": "human-1"},
        {"type": "ai", "content": "Hello world", "id": "ai-1"},
    ]
}


def _start_after_another_run(
    stream: FakeLangGraphThreadStream,
    threads: FakeLangGraphThreadsClient,
    *,
    input_checkpoint: dict[str, Any] | None,
) -> None:
    """The pre-run read saw ``_EARLIER_STATE``; the run starts at
    ``input_checkpoint`` and leaves ``_THIS_RUN_DONE``."""
    start = stream.run.start

    async def start_and_write(**kwargs: Any) -> dict[str, Any]:
        reply = await start(**kwargs)
        if input_checkpoint is not None:
            threads.input_states[(THREAD_ID, reply["run_id"])] = input_checkpoint
        threads.states[THREAD_ID] = {"values": _THIS_RUN_DONE}
        threads.run_states[(THREAD_ID, reply["run_id"])] = {"values": _THIS_RUN_DONE}
        return reply

    stream.run.start = start_and_write


async def test_state_completion_starts_at_the_runs_input_checkpoint(quiet_poll):
    """The other run's answer is after the pre-run read but before this run's
    start: it is not this turn's."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _start_after_another_run(
        stream, threads, input_checkpoint={"values": _AFTER_THE_OTHER_RUN}
    )

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello world"]
    assert events[-1]["response"] == "Hello world"


def _values(messages: list[dict[str, Any]], entry_id: str) -> dict[str, Any]:
    return {
        "method": "values",
        "params": {"namespace": [], "data": {"messages": messages}},
        "event_id": entry_id,
    }


async def test_read_back_value_snapshots_start_at_the_runs_input_checkpoint(
    quiet_poll,
):
    """A run whose answer arrives only in value snapshots: the read-back shows
    it, not the other run's answer the snapshots also hold."""
    run = [
        _lifecycle("run-1", "running"),
        _values(_THIS_RUN_DONE["messages"][:-1], "1-0"),
        _values(_THIS_RUN_DONE["messages"], "1-1"),
        _lifecycle("run-1", "completed"),
    ]
    stream = FakeLangGraphThreadStream(THREAD_ID, events=run, live_limit=0)
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _start_after_another_run(
        stream, threads, input_checkpoint={"values": _AFTER_THE_OTHER_RUN}
    )

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello world"]


async def test_state_completion_without_an_input_checkpoint_raises(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _start_after_another_run(stream, threads, input_checkpoint=None)

    with pytest.raises(RuntimeError, match=_UNSHOWN):
        await _turn(gateway, "hi")


async def test_state_completion_after_a_summary_rewrote_the_start(quiet_poll):
    """None of the start's messages is left (a summary replaced them): the
    turn starts at the first assistant message the start did not have."""
    summarized = {
        "messages": [
            {"type": "human", "content": "Summary of the conversation", "id": "sum-1"},
            {"type": "human", "content": "hi", "id": "human-1"},
            {"type": "ai", "content": "Hello world", "id": "ai-1"},
        ]
    }
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": summarized})

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello world"]


async def test_finished_reply_with_only_thinking_is_not_repeated(quiet_poll):
    """Its end arrived live: neither its thinking nor its usage again."""
    run = [
        _lifecycle("run-1", "running"),
        _message_start("ai-1", "1-0"),
        _thinking("weighing it", "1-1"),
        _finish_with_usage("1-2", 10, 5),
        _message_start("ai-2", "1-3"),
        _text("Done", "1-4"),
        _finish_with_usage("1-5", 20, 7),
        _lifecycle("run-1", "completed"),
    ]
    state = {
        "messages": [
            *_EARLIER_STATE["messages"],
            {"type": "human", "content": "hi", "id": "human-1"},
            {
                "type": "ai",
                "content": [{"type": "reasoning", "reasoning": "weighing it"}],
                "id": "ai-1",
                "usage_metadata": _usage(10, 5),
            },
            {
                "type": "ai",
                "content": "Done",
                "id": "ai-2",
                "usage_metadata": _usage(20, 7),
            },
        ]
    }
    expected = await _live_events(run, values=_EARLIER_STATE)
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=4, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": state})

    events = await _turn(gateway, "hi")

    assert [e for e in events if e["type"] != "text"] == [
        e for e in expected if e["type"] != "text"
    ]


async def test_finished_reply_whose_tool_call_was_not_shown_gets_it(quiet_poll):
    """A finish does not mean every tool call was shown: the stored one is."""
    run = [
        _lifecycle("run-1", "running"),
        _message_start("ai-1", "1-0"),
        _finish_with_usage("1-1", 10, 5),
        _tool("tool-started", "1-2", tool_name="execute", input={"command": "ls"}),
        _tool("tool-finished", "1-3", tool_name="execute", output="a.txt"),
        _lifecycle("run-1", "completed"),
    ]
    state = {
        "messages": [
            *_EARLIER_STATE["messages"],
            {"type": "human", "content": "hi", "id": "human-1"},
            {
                "type": "ai",
                "content": "",
                "id": "ai-1",
                "tool_calls": [
                    {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
                ],
                "usage_metadata": _usage(10, 5),
            },
            {
                "type": "tool",
                "content": "a.txt",
                "id": "tool-1",
                "name": "execute",
                "tool_call_id": "call-1",
            },
        ]
    }
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=3, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": state})

    events = await _turn(gateway, "hi")

    assert [e["type"] for e in events] == [
        "usage_stats",
        "tool_call",
        "tool_result",
        "done",
    ]


async def test_prompt_held_by_the_read_back_is_shown_without_a_boundary(
    quiet_poll,
):
    """No input checkpoint, and the read-back stops after the prompt but
    before the run's end: the prompt it holds still ends the turn."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_PAUSE_RUN, live_limit=0, read_back_limit=4
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    start = stream.run.start

    async def start_and_pause(**kwargs: Any) -> dict[str, Any]:
        reply = await start(**kwargs)
        paused = {"values": _EARLIER_STATE, "interrupts": [_STATE_INTERRUPT]}
        threads.states[THREAD_ID] = paused
        threads.run_states[(THREAD_ID, reply["run_id"])] = paused
        return reply

    stream.run.start = start_and_pause

    events = await _turn(gateway, "hi")

    assert [e for e in events if e["type"] == "interrupt"] == [_INTERRUPT_EVENT]
    assert gateway.thread_store.client.runs.cancelled == []


@pytest.mark.parametrize(
    ("run", "read_back_limit"),
    [
        pytest.param(_TEXT_RUN, 0, id="nothing-read-back"),
        pytest.param(
            [*_TEXT_RUN[:4], _values(_NEW_STATE["messages"], "1-3"), _TEXT_RUN[4]],
            5,
            id="read-back-holds-a-snapshot",
        ),
    ],
)
async def test_nothing_to_show_from_state_shows_no_line(
    run, read_back_limit, quiet_poll
):
    """The live stream showed the whole answer; only the run's end was lost."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=4, read_back_limit=read_back_limit
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _NEW_STATE})
    gateway.events = _Notices()

    events = await _turn(gateway, "hi")

    assert events[-1]["response"] == "Hello world"
    assert gateway.events.lines == []


_FALLBACK_NOTICE_RUN = [
    _lifecycle("run-1", "running"),
    _text("Before", "1-0"),
    {
        "method": "custom",
        "params": {
            "namespace": [],
            "data": {
                "evoscientist": {
                    "kind": "fallback_notice",
                    "text": "fb",
                    "style": "red",
                }
            },
        },
        "event_id": "1-1",
    },
    _text(" after", "1-2"),
    _finish("1-3"),
    _lifecycle("run-1", "completed"),
]


@pytest.mark.parametrize(
    ("read_back_limit", "shown", "lost"),
    [
        pytest.param(
            None,
            [server_mod._RECOVERED_TURN_NOTICE, "Before", "fallback: fb", " after"],
            False,
            id="read-back",
        ),
        pytest.param(3, [], True, id="read-back-cut-then-lost"),
    ],
)
async def test_read_back_shows_middleware_notices_with_its_events(
    read_back_limit, shown, lost, quiet_poll
):
    """Held read-back events and their notices are shown together, in order."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=_FALLBACK_NOTICE_RUN,
        live_limit=0,
        read_back_limit=read_back_limit,
    )
    gateway, _ = _gateway(stream)
    gateway.events = _Notices()

    async def turn() -> None:
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id=THREAD_ID)
        ):
            if event["type"] == "text":
                gateway.events.lines.append(event["content"])

    if lost:
        with pytest.raises(RuntimeError, match=_UNSHOWN):
            async with asyncio.timeout(GUARD_SECONDS):
                await turn()
    else:
        async with asyncio.timeout(GUARD_SECONDS):
            await turn()

    assert gateway.events.lines == shown


async def test_resume_turn_shows_a_held_prompt_without_a_checkpoint_of_its_own(
    quiet_poll,
):
    """A resume that pauses again in the same step writes no checkpoint of its
    own: the prompt the read-back holds is still shown."""
    run = [
        _lifecycle("run-1", "running"),
        _tool("tool-finished", "1-0", tool_name="execute", output="a.txt"),
        _input_requested([]),
        _lifecycle("run-1", "interrupted"),
    ]
    gateway, threads = _resumed_gateway(
        read_back_limit=3, interrupts=[_STATE_INTERRUPT], run=run
    )
    stream = threads.streams[THREAD_ID]
    respond = stream.run.respond_command

    async def respond_without_a_checkpoint(params):
        reply = await respond(params)
        del threads.run_states[(THREAD_ID, stream.run.run_id)]
        return reply

    stream.run.respond_command = respond_without_a_checkpoint

    events = await _turn(gateway, _RESUME)

    assert [e["type"] for e in events if e["type"] in ("interrupt", "error")] == [
        "interrupt"
    ]


@pytest.mark.parametrize("calls", ["tool_calls", "invalid_tool_calls"])
@pytest.mark.parametrize("as_objects", [False, True], ids=["dicts", "messages"])
async def test_state_completion_skips_the_result_of_an_earlier_turns_call(
    as_objects, calls, quiet_poll
):
    """An earlier turn left a call without a result (stopped, or malformed),
    and the next run's middleware writes one before the new message: it is
    not this turn's."""
    from langchain_core.messages import messages_from_dict

    call = {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
    if calls == "invalid_tool_calls":
        call = {**call, "args": '{"command": "l', "error": None}
    dangling = [
        {"type": "human", "content": "list files", "id": "human-0"},
        {"type": "ai", "content": "", "id": "ai-0", calls: [call]},
    ]
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    # The pre-run state is the run's input checkpoint too.
    start = (
        messages_from_dict([{"type": m["type"], "data": m} for m in dangling])
        if as_objects
        else dangling
    )
    gateway, threads = _gateway(stream, values={"messages": start})
    closed = {
        "type": "tool",
        "content": "Tool call execute with id call-1 did not complete"
        if calls == "tool_calls"
        else "Tool call execute with id call-1 could not be executed - arguments "
        "were malformed or truncated.",
        "id": "tool-0",
        "name": "execute",
        "tool_call_id": "call-1",
    }
    new = [
        {"type": "human", "content": "hi", "id": "human-1"},
        {"type": "ai", "content": "Hello world", "id": "ai-1"},
    ]
    _state_after_start(
        stream, threads, {"values": {"messages": [*dangling, closed, *new]}}
    )

    events = await _turn(gateway, "hi")

    assert [e["type"] for e in events if e["type"] != "text"] == ["done"]
    assert events[-1]["response"] == "Hello world"


@pytest.mark.parametrize("answered", [True, False], ids=["answered", "dangling"])
async def test_state_completion_keeps_the_result_of_a_reused_call_id(
    answered, quiet_poll
):
    """An earlier call with the same id, answered or left dangling, does not
    hide this turn's result for its own call."""
    earlier_call = {
        "type": "ai",
        "content": "",
        "id": "ai-0",
        "tool_calls": [{"name": "execute", "args": {"command": "ls"}, "id": "call-1"}],
    }
    earlier_result = {
        "type": "tool",
        "content": "a.txt",
        "id": "tool-0",
        "name": "execute",
        "tool_call_id": "call-1",
    }
    earlier = [
        {"type": "human", "content": "list files", "id": "human-0"},
        earlier_call,
        *(
            [earlier_result, {"type": "ai", "content": "One file", "id": "ai-1"}]
            if answered
            else []
        ),
    ]
    # Left dangling, the next run's middleware closes it before the new message.
    closed = (
        []
        if answered
        else [
            {
                **earlier_result,
                "content": "Tool call execute with id call-1 did not complete",
            }
        ]
    )
    new = [
        {"type": "human", "content": "again", "id": "human-1"},
        {
            "type": "ai",
            "content": "",
            "id": "ai-2",
            "tool_calls": [
                {"name": "execute", "args": {"command": "ls"}, "id": "call-1"}
            ],
        },
        {
            "type": "tool",
            "content": "a.txt b.txt",
            "id": "tool-1",
            "name": "execute",
            "tool_call_id": "call-1",
        },
        {"type": "ai", "content": "Two files", "id": "ai-3"},
    ]
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, read_back_limit=0
    )
    gateway, threads = _gateway(stream, values={"messages": earlier})
    _state_after_start(
        stream, threads, {"values": {"messages": [*earlier, *closed, *new]}}
    )

    events = await _turn(gateway, "again")

    assert [e["type"] for e in events if e["type"] != "text"] == [
        "tool_call",
        "tool_result",
        "done",
    ]


# --- #711: the event stream ends or fails before the run does ---------------


class _StatusAfter:
    """``runs.get`` answering ``running`` for the first *n* requests, then the
    run's own status. ``read_back_early`` records a read-back opened while the
    run still ran (an assert here would be swallowed: the gateway counts any
    exception from ``runs.get`` as a failed status request)."""

    def __init__(self, gateway, stream, n: int, fail_first: int = 0) -> None:
        self.runs = gateway.thread_store.client.runs
        self.stream = stream
        self.n = n
        self.fail_first = fail_first
        self.calls = 0
        self.read_back_early = False
        self.real = self.runs.get
        self.runs.get = self

    async def __call__(self, thread_id: str, run_id: str) -> dict[str, Any]:
        self.calls += 1
        # The read-back waits until the run has finished.
        self.read_back_early |= self.stream.event_streams_opened != 1
        if self.calls <= self.fail_first:
            raise ConnectionError("server busy")
        if self.calls <= self.n:
            return {"run_id": run_id, "status": "running"}
        return await self.real(thread_id, run_id)


@pytest.mark.parametrize(
    "live_end",
    [True, ConnectionError("read failed")],
    ids=["stream-ends", "stream-raises"],
)
@pytest.mark.parametrize("live_limit", [0, 2], ids=["nothing-live", "partly-live"])
@pytest.mark.parametrize(
    ("run", "interrupts"),
    [
        pytest.param(_TEXT_RUN, None, id="text"),
        pytest.param(_TOOL_RUN, None, id="tool-call"),
        pytest.param(_PAUSE_RUN, [], id="pause"),
    ],
)
async def test_run_outlives_its_event_stream_and_is_read_back(
    run, interrupts, live_limit, live_end, quiet_poll, caplog
):
    """The stream ends (or fails) while the run works on: the turn waits for
    the run's status, then reads the finished run back. Its events equal a
    live turn's, and the run is not cancelled."""
    expected = await _live_events(run, interrupts=interrupts)
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=run, live_limit=live_limit, live_end=live_end
    )
    gateway, _ = _gateway(stream, interrupts=interrupts)
    status = _StatusAfter(gateway, stream, n=2)
    gateway.events = _Notices()

    with caplog.at_level("INFO", logger="EvoScientist.gateway.server"):
        events = await _turn(gateway, "hi")

    assert events == expected
    # Said when the stream closed, not again with the result.
    assert gateway.events.lines == [server_mod._STREAM_CLOSED_NOTICE]
    assert status.calls == 3
    assert not status.read_back_early
    assert gateway.thread_store.client.runs.cancelled == []
    how = "failed" if isinstance(live_end, Exception) else "ended"
    closed = [
        record
        for record in caplog.records
        if record.getMessage().startswith(
            f"The event stream of run run-1 on thread {THREAD_ID} {how} after "
            f"{live_limit} of the run's envelopes"
        )
    ]
    # The notice is the frontend's line: the log does not warn again.
    assert [record.levelname for record in closed] == ["INFO"]


@pytest.mark.parametrize("finished", [False, True], ids=["running", "finished"])
async def test_closed_stream_warns_a_turn_without_a_frontend(
    finished, quiet_poll, caplog
):
    """With no sink to show the notice, the log is the only trace; it says
    the turn waits only while the run has not finished."""
    if finished:
        stream = _EndAfterStatusStream(THREAD_ID, events=_TEXT_RUN)
        gateway, _ = _gateway(stream)
        runs = gateway.thread_store.client.runs
        get = runs.get

        async def get_and_end(thread_id: str, run_id: str) -> dict[str, Any]:
            run = await get(thread_id, run_id)
            stream.status_asked.set()
            return run

        runs.get = get_and_end
    else:
        stream = FakeLangGraphThreadStream(
            THREAD_ID, events=_TEXT_RUN, live_limit=1, live_end=True
        )
        gateway, _ = _gateway(stream)
        _StatusAfter(gateway, stream, n=1)

    with caplog.at_level("INFO", logger="EvoScientist.gateway.server"):
        await _turn(gateway, "hi")

    closed = [
        record
        for record in caplog.records
        if record.getMessage().startswith("The event stream of run run-1")
    ]
    assert [record.levelname for record in closed] == ["WARNING"]
    waits = closed[0].getMessage().endswith("waiting for the run to finish")
    assert waits is not finished


async def test_stream_ending_after_the_run_finished_is_read_back(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert gateway.thread_store.client.runs.gets == [(THREAD_ID, "run-1")]


async def test_run_failing_after_its_stream_ended_raises_its_own_error(quiet_poll):
    failed = _lifecycle("run-1", "failed")
    failed["params"]["data"]["error"] = "boom"
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=[_lifecycle("run-1", "running"), failed],
        live_limit=1,
        live_end=True,
    )
    stream.run.status = "error"
    gateway, _ = _gateway(stream)
    _StatusAfter(gateway, stream, n=1)

    with pytest.raises(RuntimeError, match="Run failed: boom"):
        await _turn(gateway, "hi")

    assert gateway.thread_store.client.runs.cancelled == []


async def test_failed_status_requests_after_the_stream_ended_are_retried(
    quiet_poll,
):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)
    status = _StatusAfter(gateway, stream, n=0, fail_first=2)

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert status.calls == 3
    assert gateway.thread_store.client.runs.cancelled == []


async def test_failed_status_requests_must_come_in_a_row_to_end_the_turn(
    quiet_poll,
):
    """Two failures, a success that says ``running``, two more failures: the
    count starts again, so the turn still recovers."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    real = runs.get
    answers = ["fail", "fail", "running", "fail", "fail", "real"]

    async def flaky(thread_id: str, run_id: str) -> dict[str, Any]:
        answer = answers.pop(0)
        if answer == "fail":
            raise ConnectionError("server busy")
        if answer == "running":
            return {"run_id": run_id, "status": "running"}
        return await real(thread_id, run_id)

    runs.get = flaky

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert answers == []


async def test_a_stream_error_the_status_cannot_settle_keeps_its_message(
    quiet_poll,
):
    stream = FakeLangGraphThreadStream(
        THREAD_ID,
        events=_TEXT_RUN,
        live_limit=0,
        live_end=ConnectionError("read failed"),
    )
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs

    raised: list[Exception] = []

    async def unreadable(thread_id: str, run_id: str) -> dict[str, Any]:
        raised.append(ConnectionError(f"server busy {len(raised)}"))
        raise raised[-1]

    runs.get = unreadable

    with pytest.raises(ConnectionError, match="read failed") as caught:
        await _turn(gateway, "hi")

    assert caught.value.__cause__ is raised[-1]


async def test_closed_stream_asks_for_the_status_once_per_poll_window(quiet_poll):
    """After the stream closed the turn waits a poll window between status
    requests instead of asking again at once."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    real = runs.get
    loop = asyncio.get_running_loop()
    asked: list[float] = []
    answered: list[float] = []

    async def running_for_a_while(thread_id: str, run_id: str) -> dict[str, Any]:
        asked.append(loop.time())
        try:
            if asked[-1] < asked[0] + 0.5:
                return {"run_id": run_id, "status": "running"}
            return await real(thread_id, run_id)
        finally:
            answered.append(loop.time())

    runs.get = running_for_a_while

    events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]
    assert len(asked) >= 2
    # Each answer is followed by a full 0.2 s window before the next request.
    assert min(asked[i + 1] - answered[i] for i in range(len(asked) - 1)) >= 0.15


async def test_status_failures_before_the_stream_closed_do_not_count(quiet_poll):
    """Two failed status requests while the stream is still open, then it
    ends and one more fails: the turn keeps waiting for the run."""
    stream = FakeLangGraphThreadStream(THREAD_ID, events=_TEXT_RUN)
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs
    real = runs.get
    answers = ["fail", "fail", "fail", "real"]
    two_failed = asyncio.Event()

    async def flaky(thread_id: str, run_id: str) -> dict[str, Any]:
        answer = answers.pop(0)
        if len(answers) == 2:
            two_failed.set()
        if answer == "fail":
            raise ConnectionError("server busy")
        return await real(thread_id, run_id)

    runs.get = flaky

    async def quiet_then_closed() -> AsyncIterator[dict[str, Any]]:
        await stream.run.started.wait()
        yield _lifecycle("run-1", "running")
        await two_failed.wait()

    real_events = FakeLangGraphThreadStream.events
    opened: list[object] = []

    def events(self: FakeLangGraphThreadStream) -> AsyncIterator[dict[str, Any]]:
        opened.append(self)
        if len(opened) == 1:
            return quiet_then_closed()
        return real_events.fget(self)

    with patch.object(FakeLangGraphThreadStream, "events", property(events)):
        turn_events = await _turn(gateway, "hi")

    assert _texts(turn_events) == ["Hello", " world"]
    assert answers == []


async def test_resume_turn_outlives_its_event_stream(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=1, live_end=True
    )
    gateway, _ = _gateway(stream, interrupts=[{"id": "interrupt-0"}])
    _StatusAfter(gateway, stream, n=1)

    events = await _turn(
        gateway, Command(resume={"interrupt-0": {"decisions": [{"type": "approve"}]}})
    )

    assert _texts(events) == ["Hello", " world"]
    assert gateway.thread_store.client.runs.cancelled == []


async def test_run_the_server_no_longer_has_after_the_stream_ended(quiet_poll):
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)
    start = stream.run.start

    async def start_and_forget(**kwargs: Any) -> dict[str, Any]:
        reply = await start(**kwargs)
        stream.run.runs.clear()
        return reply

    stream.run.start = start_and_forget

    with pytest.raises(RuntimeError, match="Run run-1 no longer exists"):
        await _turn(gateway, "hi")

    assert gateway.thread_store.client.runs.cancelled == []


async def test_status_is_asked_as_soon_as_the_stream_ends(monkeypatch):
    """No wait before the first status request, and none after a finished
    status once the stream is closed: with long windows the turn still ends
    at once."""
    monkeypatch.setattr(server_mod, "_QUIET_POLL_SECONDS", 30.0)
    monkeypatch.setattr(server_mod, "_FINISHED_GRACE_SECONDS", 30.0)
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=1, live_end=True
    )
    gateway, _ = _gateway(stream)

    async with asyncio.timeout(5):
        events = await _turn(gateway, "hi")

    assert _texts(events) == ["Hello", " world"]


async def test_giving_up_keeps_the_last_status_error_as_the_cause(quiet_poll):
    """A server that died answers neither the stream nor the status: the
    turn's error carries the status request's error as its cause."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=0, live_end=True
    )
    gateway, _ = _gateway(stream)
    runs = gateway.thread_store.client.runs

    async def dead(thread_id: str, run_id: str) -> dict[str, Any]:
        raise ConnectionError("All connection attempts failed")

    runs.get = dead

    with pytest.raises(RuntimeError, match="ended before the run finished") as caught:
        await _turn(gateway, "hi")

    assert isinstance(caught.value.__cause__, ConnectionError)


async def test_closed_stream_completed_from_the_state_says_it_once(quiet_poll):
    """The line shown when the stream closed covers a result that comes from
    the run's checkpoint too."""
    stream = FakeLangGraphThreadStream(
        THREAD_ID, events=_TEXT_RUN, live_limit=2, read_back_limit=0, live_end=True
    )
    gateway, threads = _gateway(stream, values=_EARLIER_STATE)
    _state_after_start(stream, threads, {"values": _NEW_STATE})
    _StatusAfter(gateway, stream, n=1)
    gateway.events = _Notices()

    events = await _turn(gateway, "hi")

    assert events[-1]["response"] == "Hello world"
    assert gateway.events.lines == [server_mod._STREAM_CLOSED_NOTICE]
