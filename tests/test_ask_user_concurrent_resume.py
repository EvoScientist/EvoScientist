"""Regression tests for issue #762: one run per thread at a time.

Two clients that resume the SAME thread at once used to write their answer
into the checkpoint independently. The graph ended up with every answer
recorded and no node scheduled: the model was never called again and the turn
produced no response at all.

The local path now serialises the runs of one thread, so the second resume
waits for the first to finish and then sees the state the first one left
behind.

These tests drive the REAL main graph (``create_cli_agent``) through the REAL
stream funnel (``stream_agent_events``); only the human is scripted.
"""

from __future__ import annotations

import asyncio
import itertools
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
        _drain(
            agent,
            "t",
            build_ask_user_resume(first, {"answers": ["red"], "status": "answered"}),
        ),
        _drain(
            agent,
            "t",
            build_ask_user_resume(second, {"answers": ["large"], "status": "answered"}),
        ),
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
            build_ask_user_resume(
                interrupt_id, {"answers": ["red"], "status": "answered"}
            ),
            "t",
        ):
            timeline.append(tag)

    await asyncio.gather(resume("a", pending[0]), resume("b", pending[1]))

    assert "a" in timeline, timeline
    assert "b" in timeline, timeline
    # One contiguous block each: the runs did not interleave.
    blocks = [tag for tag, _ in itertools.groupby(timeline)]
    assert len(blocks) == 2, timeline
