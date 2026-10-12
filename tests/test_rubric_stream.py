"""A graded run through the real stream pipeline, with a nested grader agent."""

from __future__ import annotations

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from EvoScientist.middleware.rubric import create_rubric_middleware
from EvoScientist.stream.events import stream_agent_events
from tests.test_rubric_middleware import FAIL, PASS, _Main, _ScriptedGrader


def _deep(tmp_path, verdicts, replies, **middleware_kwargs):
    grader = _ScriptedGrader(verdicts=verdicts)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    rubric = create_rubric_middleware(
        model=grader, backend=backend, max_iterations=3, **middleware_kwargs
    )
    agent = create_deep_agent(
        model=_Main(messages=iter([AIMessage(r) for r in replies])),
        backend=backend,
        middleware=[rubric],
        checkpointer=InMemorySaver(),
        name="EvoScientist",
    )
    return agent, grader


async def _collect(agent, message, thread_id, rubric=None):
    return [
        event
        async for event in stream_agent_events(agent, message, thread_id, rubric=rubric)
    ]


_EXPERIMENTAL_STREAM = "ignore:The v3 streaming protocol on Pregel is experimental"


@pytest.mark.filterwarnings(_EXPERIMENTAL_STREAM)
async def test_nested_grader_is_not_shown_as_a_subagent(tmp_path):
    # One-word replies: the fake model streams word by word and the stream
    # pipeline drops whitespace-only chunks.
    agent, _grader = _deep(tmp_path, [FAIL, PASS], ["draft", "revised"])

    events = await _collect(agent, "do it", "t1", rubric="- c1")

    assert [e["type"] for e in events if e["type"].startswith("subagent")] == []
    assert [e["type"] for e in events if e["type"].startswith("rubric_")] == [
        "rubric_evaluation_start",
        "rubric_evaluation",
        "rubric_evaluation_start",
        "rubric_evaluation",
    ]
    assert [e for e in events if e["type"] == "done"][-1][
        "response"
    ] == "draft\n\nrevised"


@pytest.mark.filterwarnings(_EXPERIMENTAL_STREAM)
async def test_stream_marks_the_run_so_the_main_agent_grader_runs(tmp_path):
    agent, grader = _deep(tmp_path, [PASS], ["done"], require_run_marker=True)

    events = await _collect(agent, "do it", "t1", rubric="- c1")

    assert [e["result"] for e in events if e["type"] == "rubric_evaluation"] == [
        "satisfied"
    ]
    assert grader.calls == 1


@pytest.mark.filterwarnings(_EXPERIMENTAL_STREAM)
async def test_plain_turn_after_a_stale_rubric_is_not_graded(tmp_path):
    agent, grader = _deep(tmp_path, [PASS], ["ok"], require_run_marker=True)
    cfg = {"configurable": {"thread_id": "t1"}}
    agent.update_state(cfg, {"rubric": "- stale"})

    events = await _collect(agent, "thanks", "t1")

    assert not [e for e in events if e["type"].startswith("rubric_")]
    assert grader.calls == 0
    assert agent.get_state(cfg).values["rubric"] is None


class _ScriptedMain(BaseChatModel):
    """Main model that replays a script, so a turn can contain a tool call."""

    script: list
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-main"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        message = item if isinstance(item, AIMessage) else AIMessage(item)
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def needs_approval() -> str:
    """A tool gated by an approval prompt."""
    return "done"


_TOOL_CALL = AIMessage(
    content="",
    tool_calls=[
        {"name": "needs_approval", "args": {}, "id": "tc1", "type": "tool_call"}
    ],
)
_APPROVE = {"decisions": [{"type": "approve"}]}


async def _parked_at_an_approval(tmp_path):
    grader = _ScriptedGrader(verdicts=[PASS])
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    rubric = create_rubric_middleware(
        model=grader, backend=backend, max_iterations=3, require_run_marker=True
    )
    agent = create_deep_agent(
        model=_ScriptedMain(script=[_TOOL_CALL, "answer"]),
        backend=backend,
        tools=[needs_approval],
        interrupt_on={"needs_approval": True},
        middleware=[rubric],
        checkpointer=InMemorySaver(),
        name="EvoScientist",
    )
    first = await _collect(agent, "do it", "t", rubric="- c1")
    assert any(event["type"] == "interrupt" for event in first)
    return agent, grader


@pytest.mark.filterwarnings(_EXPERIMENTAL_STREAM)
async def test_approval_from_the_grading_client_keeps_the_run_graded(tmp_path):
    agent, grader = await _parked_at_an_approval(tmp_path)

    events = await _collect(agent, Command(resume=_APPROVE), "t", rubric="- c1")

    assert [e["result"] for e in events if e["type"] == "rubric_evaluation"] == [
        "satisfied"
    ]
    assert grader.calls == 1


@pytest.mark.filterwarnings(_EXPERIMENTAL_STREAM)
async def test_approval_from_another_client_is_not_graded(tmp_path):
    agent, grader = await _parked_at_an_approval(tmp_path)

    events = await _collect(agent, Command(resume=_APPROVE), "t")

    assert not [e for e in events if e["type"].startswith("rubric_")]
    assert grader.calls == 0
    state = await agent.aget_state({"configurable": {"thread_id": "t"}})
    assert state.values["rubric"] is None


async def test_a_subagent_that_only_shares_the_graders_name_is_shown():
    from tests.stream_v3_fakes import (
        FakeSubagent,
        FakeV3Agent,
        collect_events,
        tool_finished,
        tool_started,
    )

    # Started by the task tool, so it does not run below the grading hook.
    namespace = ("tools:1",)
    agent = FakeV3Agent(
        [
            tool_started("ls", tool_call_id="t1", namespace=namespace),
            tool_finished("[]", tool_call_id="t1", namespace=namespace),
        ],
        subagents=[FakeSubagent(namespace, "rubric_grader")],
    )

    events = await collect_events(agent)

    assert [e["type"] for e in events if e["type"].startswith("subagent")] == [
        "subagent_start",
        "subagent_tool_call",
        "subagent_tool_result",
        "subagent_end",
    ]
