"""Regression tests for issue #762: two ask_user questions in one message.

When the model calls ``ask_user`` twice in a single message, the graph parks
with TWO pending interrupts. The clients used to resume with a flat
``Command(resume=payload)``; with more than one pending interrupt that raises
``RuntimeError`` (langgraph in-process, the server gateway's
``_resume_target``). The fix keys every ask_user resume by the interrupt id of
the event the client actually displayed (``build_ask_user_resume``), which
answers that interrupt and re-parks the rest — they re-emit on the next stream
and are drained one at a time, exactly like keyed HITL approvals.

These tests drive the REAL main graph (``create_cli_agent``) through the REAL
Rich CLI resume loop (``stream.display._run_streaming`` over a
``LocalGraphGateway``); only the human is scripted via ``ask_user_prompt_fn``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.paths import Workspace
from EvoScientist.stream import display as display_mod

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


@dataclass
class _RecordingGateway:
    """LocalGraphGateway wrapper that records every RunRequest and the
    order the ask_user events arrive in."""

    inner: LocalGraphGateway = field(default_factory=LocalGraphGateway)
    requests: list[Any] = field(default_factory=list)
    ask_user_arrivals: list[str] = field(default_factory=list)

    def stream_events(self, request):
        self.requests.append(request)
        inner = self.inner.stream_events(request)

        async def _recording():
            async for event in inner:
                if event.get("type") == "ask_user":
                    question = event["questions"][0]["question"]
                    self.ask_user_arrivals.append(question)
                yield event

        return _recording()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _ask(call_id: str, question: str) -> dict:
    return _call(
        call_id, "ask_user", questions=[{"question": question, "type": "text"}]
    )


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


def _run(agent, prompt_fn) -> tuple[str, _RecordingGateway]:
    gateway = _RecordingGateway()
    result = display_mod._run_streaming(
        agent=agent,
        message="pick for me",
        thread_id="t",
        show_thinking=False,
        interactive=True,
        ask_user_prompt_fn=prompt_fn,
        gateway=gateway,
    )
    return result, gateway


def _ask_user_tool_results(model) -> dict[str, str]:
    """Every ask_user ToolMessage the model received, by tool_call_id."""
    results: dict[str, str] = {}
    for messages in model.seen:
        for m in messages:
            if m.type == "tool" and m.tool_call_id in {"q1", "q2"}:
                results.setdefault(m.tool_call_id, str(m.content))
    return results


def _two_ask_user_replies() -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[_ask("q1", "Colour?"), _ask("q2", "Size?")]),
        AIMessage("Both answers received."),
    ]


async def _parked_interrupt_count(agent) -> int:
    parked = await agent.ainvoke(
        {"messages": [("user", "pick for me")]},
        config={"configurable": {"thread_id": "t"}},
    )
    return len(parked.get("__interrupt__") or [])


def test_two_ask_user_calls_in_one_message_are_both_answered(tmp_path):
    """The model asks Colour? and Size? in ONE message: the client must answer
    each question by its OWN interrupt id, the turn must not raise, and BOTH
    tool results must reach the model, each carrying its own answer."""
    parked_agent, _parked_model = _agent(tmp_path, _two_ask_user_replies())
    # The issue's scenario: one message, TWO pending ask_user interrupts.
    assert asyncio.run(_parked_interrupt_count(parked_agent)) == 2

    agent, model = _agent(tmp_path / "turn", _two_ask_user_replies())
    asked: list[tuple[str, str]] = []

    def _prompt(ask_user_data: dict) -> dict:
        question = ask_user_data["questions"][0]["question"]
        asked.append((ask_user_data["interrupt_id"], question))
        return {"answers": [ANSWER_BY_QUESTION[question]], "status": "answered"}

    result, gateway = _run(agent, _prompt)

    assert result == "Both answers received."
    # Both questions were displayed, one per interrupt, in first-emitted
    # order. LangGraph emits parallel-tool interrupts in completion order,
    # which varies run to run; what the CLI's keep-first block guarantees is
    # that the questions are ASKED in the order the events arrive — the
    # same order the TUI and the channels consumer ask in. (A still-pending
    # sibling re-emits on the resume stream, hence the first-arrival dedupe.)
    assert len(asked) == 2
    assert len({interrupt_id for interrupt_id, _ in asked}) == 2
    first_arrivals = list(dict.fromkeys(gateway.ask_user_arrivals))
    assert [question for _, question in asked] == first_arrivals
    assert sorted(first_arrivals) == ["Colour?", "Size?"]
    # Every resume is keyed by the interrupt id of the event that was shown.
    resumes = [r.message for r in gateway.requests[1:]]
    assert len(resumes) == 2
    for resume, (interrupt_id, _) in zip(resumes, asked, strict=True):
        assert list(resume.resume) == [interrupt_id]
        assert resume.resume[interrupt_id]["status"] == "answered"
    # Both answers reached the model, each on its own tool call.
    results = _ask_user_tool_results(model)
    assert results == {
        "q1": "Q: Colour?\nA: red",
        "q2": "Q: Size?\nA: large",
    }


def test_two_ask_user_calls_with_cancelled_answers(tmp_path):
    """A keyed CANCEL ({"status": "cancelled"}) must drain the second pending
    interrupt just like an answered one."""
    agent, model = _agent(tmp_path, _two_ask_user_replies())

    def _prompt(_ask_user_data: dict) -> dict:
        return {"status": "cancelled"}

    result, gateway = _run(agent, _prompt)

    assert result == "Both answers received."
    for resume in [r.message for r in gateway.requests[1:]]:
        (interrupt_id,) = resume.resume
        assert resume.resume[interrupt_id] == {"status": "cancelled"}
    results = _ask_user_tool_results(model)
    assert results == {
        "q1": "Q: Colour?\nA: (cancelled)",
        "q2": "Q: Size?\nA: (cancelled)",
    }


def test_a_single_ask_user_still_round_trips(tmp_path):
    """N=1 is the common path: one question, one keyed answer, no regression."""
    agent, model = _agent(
        tmp_path,
        [
            AIMessage("", tool_calls=[_ask("q1", "Colour?")]),
            AIMessage("Answer received."),
        ],
    )
    asked: list[str] = []

    def _prompt(ask_user_data: dict) -> dict:
        asked.append(ask_user_data["questions"][0]["question"])
        return {"answers": ["red"], "status": "answered"}

    result, gateway = _run(agent, _prompt)

    assert result == "Answer received."
    assert asked == ["Colour?"]
    (resume,) = [r.message for r in gateway.requests[1:]]
    (interrupt_id,) = resume.resume
    assert resume.resume[interrupt_id] == {"answers": ["red"], "status": "answered"}
    assert _ask_user_tool_results(model) == {"q1": "Q: Colour?\nA: red"}


def test_the_gateway_guard_still_rejects_unkeyed_multi_interrupt_resumes():
    """The server gateway must keep refusing a flat payload with two pending
    interrupts — the fix is the keyed client, not a weakened guard."""
    from EvoScientist.gateway.server import _resume_target

    with pytest.raises(RuntimeError, match="2 pending interrupts"):
        _resume_target(
            "t", {"answers": ["red"], "status": "answered"}, ["id-a", "id-b"]
        )
    # A keyed payload resolves to exactly the interrupt it names.
    assert _resume_target("t", {"id-a": {"status": "cancelled"}}, ["id-a", "id-b"]) == (
        "id-a",
        {"status": "cancelled"},
    )
