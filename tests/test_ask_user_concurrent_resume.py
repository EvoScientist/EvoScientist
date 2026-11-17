"""Regression tests for issue #762: one run per thread at a time.

Two clients that resume the SAME thread at once used to write their answer
into the checkpoint independently. The graph ended up with every answer
recorded and no node scheduled: the model was never called again and the
turn produced no response at all.

The local path now serialises the runs of one thread, so the second resume
waits for the first to finish and then sees the state the first one left
behind.

These tests drive the REAL main graph (``create_cli_agent``) through the
REAL stream funnel (``stream_agent_events``); only the human is scripted.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.paths import Workspace
from EvoScientist.stream.events import stream_agent_events

pytestmark = pytest.mark.usefixtures("no_mcp")

ANSWER_BY_QUESTION = {"Colour?": "red", "Size?": "large"}


class _Scripted(BaseChatModel):
    """Gives the scripted replies in order; ``seen`` records every call."""

    replies: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0) if self.replies else AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _ask(tool_call_id: str, question: str) -> dict:
    return {
        "name": "ask_user",
        "args": {"questions": [{"question": question, "type": "text"}]},
        "id": tool_call_id,
        "type": "tool_call",
    }


def _agent(tmp_path, replies) -> tuple[Any, _Scripted]:
    model = _Scripted(replies=replies)
    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=EvoScientistConfig(
            enable_async_subagents=False, memory_workers_enabled=False
        ),
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    return agent, model


def _two_ask_user_replies() -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[_ask("q1", "Colour?"), _ask("q2", "Size?")]),
        AIMessage("Both answers received."),
    ]


async def _drain(agent, thread_id: str, message) -> list[dict]:
    """Run one message through the product's stream funnel."""
    events: list[dict] = []
    async for event in stream_agent_events(agent, message, thread_id):
        events.append(event)
    return events


def _answer(payload: str) -> dict:
    return {"answers": [payload], "status": "answered"}


@pytest.mark.asyncio
async def test_two_concurrent_resumes_of_one_thread_do_not_stall_the_graph(tmp_path):
    """Two clients answering two questions at once must still finish the turn.

    Before the fix both answers landed in the checkpoint, the graph had no
    node scheduled, the model was never called again and the turn produced no
    response.
    """
    from EvoScientist.backends import build_ask_user_resume

    agent, model = _agent(tmp_path, _two_ask_user_replies())

    # The first run parks the thread with two pending questions.
    await _drain(agent, "t", "pick for me")

    snapshot = await agent.aget_state({"configurable": {"thread_id": "t"}})
    pending = [i.id for i in snapshot.interrupts if getattr(i, "id", None)]
    assert len(pending) == 2, pending

    # Two clients answer at the same time, each keyed by its own id.
    first, second = pending[0], pending[1]
    results = await asyncio.gather(
        _drain(agent, "t", build_ask_user_resume(first, _answer("red"))),
        _drain(agent, "t", build_ask_user_resume(second, _answer("large"))),
    )

    # The turn produced a response: the model ran again and answered.
    finals = [
        event
        for events in results
        for event in events
        if event.get("type") == "text"
        and str(event.get("content") or event.get("text") or "")
    ]
    assert any(
        "Both answers received." in str(event.get("content") or event.get("text"))
        for event in finals
    ), finals

    # Both answers reached the graph, each for its own question.
    answered: dict[str, str] = {}
    for messages in model.seen:
        for message in messages:
            if message.type == "tool" and message.tool_call_id in {"q1", "q2"}:
                answered.setdefault(message.tool_call_id, str(message.content))
    assert answered.get("q1") == "Q: Colour?\nA: red", answered
    assert answered.get("q2") == "Q: Size?\nA: large", answered


@pytest.mark.asyncio
async def test_a_resume_of_one_thread_waits_for_the_run_in_flight(tmp_path):
    """The second resume of a thread waits for the first run to finish.

    The two runs must not interleave: every event of one run comes before
    any event of the other.
    """
    from EvoScientist.backends import build_ask_user_resume

    agent, _model = _agent(tmp_path, _two_ask_user_replies())
    await _drain(agent, "t", "pick for me")
    snapshot = await agent.aget_state({"configurable": {"thread_id": "t"}})
    pending = [i.id for i in snapshot.interrupts if getattr(i, "id", None)]

    timeline: list[str] = []

    async def resume(tag: str, interrupt_id: str) -> None:
        async for _event in stream_agent_events(
            agent,
            build_ask_user_resume(interrupt_id, _answer("red")),
            "t",
        ):
            timeline.append(tag)

    await asyncio.gather(resume("a", pending[0]), resume("b", pending[1]))

    assert "a" in timeline, timeline
    assert "b" in timeline, timeline
    # One contiguous block each: the runs did not interleave.
    blocks = [tag for tag, _ in itertools.groupby(timeline)]
    assert len(blocks) == 2, timeline


@pytest.mark.asyncio
async def test_two_event_loops_drive_one_thread_without_overlapping(tmp_path):
    """Two OS threads, two event loops, one thread id: still serialised.

    The lock must not be tied to one event loop. A lock bound to the loop
    that first awaited it is replaced when another loop asks for the same
    thread, and both loops then hold a different lock -- the exact race the
    lock exists to prevent.
    """
    from EvoScientist.backends import build_ask_user_resume

    agent, model = _agent(tmp_path, _two_ask_user_replies())
    await _drain(agent, "t", "pick for me")
    snapshot = await agent.aget_state({"configurable": {"thread_id": "t"}})
    pending = [i.id for i in snapshot.interrupts if getattr(i, "id", None)]
    assert len(pending) == 2, pending

    barrier = threading.Barrier(2)
    results: dict[str, list[dict]] = {}
    errors: list[BaseException] = []

    def drive(interrupt_id: str, payload: str) -> list[dict]:
        async def run() -> list[dict]:
            # Both threads reach the funnel at the same time.
            barrier.wait(timeout=10)
            events: list[dict] = []
            async for event in stream_agent_events(
                agent, build_ask_user_resume(interrupt_id, _answer(payload)), "t"
            ):
                events.append(event)
            return events

        return asyncio.run(run())

    def worker(interrupt_id: str, payload: str) -> None:
        try:
            results[payload] = drive(interrupt_id, payload)
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(pending[0], "red")),
        threading.Thread(target=worker, args=(pending[1], "large")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert set(results) == {"red", "large"}, results

    # The model was called again and the turn produced its response.
    finals = [
        event
        for events in results.values()
        for event in events
        if event.get("type") == "text"
    ]
    assert any(
        "Both answers received." in str(event.get("content") or event.get("text"))
        for event in finals
    ), finals

    # Both answers reached the graph, each for its own question.
    answered: dict[str, str] = {}
    for messages in model.seen:
        for message in messages:
            if message.type == "tool" and message.tool_call_id in {"q1", "q2"}:
                answered.setdefault(message.tool_call_id, str(message.content))
    assert answered.get("q1") == "Q: Colour?\nA: red", answered
    assert answered.get("q2") == "Q: Size?\nA: large", answered


@pytest.mark.asyncio
async def test_the_lock_is_released_when_the_stream_is_closed_early(tmp_path):
    """Closing the stream early must not strand the thread's lock.

    A close that is cancelled while the inner generator is being closed must
    still give the lock back: otherwise every later run of that thread waits
    forever.
    """
    from EvoScientist.backends import _THREAD_RUN_LOCKS, thread_run_lock

    agent, _model = _agent(tmp_path, _two_ask_user_replies())
    await _drain(agent, "t", "pick for me")

    stream = stream_agent_events(agent, "another message", "t")
    await stream.__anext__()
    await stream.aclose()

    # The registry holds no entry for a thread nobody is running.
    assert "t" not in _THREAD_RUN_LOCKS, list(_THREAD_RUN_LOCKS)

    # And the thread can be run again straight away.
    entry = thread_run_lock("t")
    await entry.wait_and_acquire()
    entry.release()


@pytest.mark.asyncio
async def test_many_threads_do_not_grow_the_lock_registry():
    """The registry does not keep the threads nobody is running.

    Every entry that nobody holds or waits for is dropped, so a process
    that serves many threads does not accumulate one lock per thread.
    """
    from EvoScientist.backends import (
        _THREAD_RUN_LOCKS,
        _THREAD_RUN_LOCKS_MAX,
        release_thread_run_lock,
        thread_run_lock,
    )

    for index in range(_THREAD_RUN_LOCKS_MAX * 10):
        entry = thread_run_lock(f"t{index}")
        await entry.wait_and_acquire()
        entry.release()
        release_thread_run_lock(f"t{index}", entry)

    assert len(_THREAD_RUN_LOCKS) <= _THREAD_RUN_LOCKS_MAX, len(_THREAD_RUN_LOCKS)


@pytest.mark.asyncio
async def test_a_waiter_keeps_the_entry_alive():
    """A thread id always maps to the SAME lock while anyone references it.

    A lock that is briefly unlocked between its holder's release and its
    waiter's wake-up looks free. Evicting it then would let a new caller
    take a SECOND lock for the same thread -- the very race the lock exists
    to prevent. The waiter here is queued and the holder has already let go
    of the lock, which is exactly that window.
    """
    from EvoScientist.backends import (
        _THREAD_RUN_LOCKS,
        release_thread_run_lock,
        thread_run_lock,
    )

    holder = thread_run_lock("t")
    await holder.wait_and_acquire()

    waiter = thread_run_lock("t")
    assert waiter is holder

    # A waiter is queued for the lock.
    wait_task = asyncio.create_task(waiter.wait_and_acquire())
    await asyncio.sleep(0.05)

    # The holder lets go of the lock. The waiter is woken but has not run
    # yet, so the lock looks free to anyone who looks now.
    holder.release()

    # A new request for the same thread must land on the SAME lock, not on
    # a second one.
    assert thread_run_lock("t") is holder
    assert _THREAD_RUN_LOCKS.get("t") is holder, list(_THREAD_RUN_LOCKS)

    await wait_task
    waiter.release()
    release_thread_run_lock("t", holder)
