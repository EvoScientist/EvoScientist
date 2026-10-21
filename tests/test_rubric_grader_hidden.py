"""The grader's own run on the real main graph: none of it reaches the stream."""

from __future__ import annotations

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
from tests.rubric_fakes import PASS

pytestmark = pytest.mark.usefixtures("no_mcp")


class _Roles(BaseChatModel):
    """Main model and grader in one, told apart by the system prompt.

    The grader plays ``grading`` in order: an exception is raised, ``"look"``
    lists the folder, a dict is handed over as the verdict.
    """

    profile: dict = Field(default_factory=lambda: {"max_input_tokens": 200_000})
    grading: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-roles"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = (
            str(messages[0].content)
            if messages and messages[0].type == "system"
            else ""
        )
        if not system.startswith("You are a grader"):
            message = AIMessage("done")
        else:
            step = self.grading.pop(0) if self.grading else PASS
            if isinstance(step, Exception):
                raise step
            if step == "look":
                call = {"name": "ls", "args": {"path": "/"}, "id": "look"}
            else:
                call = {"name": "GraderResponse", "args": step, "id": "verdict"}
            message = AIMessage("", tool_calls=[{**call, "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


async def _graded_turn(tmp_path, grading, **config):
    cfg = EvoScientistConfig(
        enable_async_subagents=False, memory_workers_enabled=False, **config
    )
    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=_Roles(grading=list(grading)),
        checkpointer=InMemorySaver(),
    )
    return [
        event async for event in stream_agent_events(agent, "hi", "t1", rubric="- c1")
    ]


def _kinds(events, prefix):
    return [e["type"] for e in events if e["type"].startswith(prefix)]


async def test_a_grading_pass_shows_only_the_rubric_events(tmp_path):
    events = await _graded_turn(tmp_path, ["look", PASS])

    assert _kinds(events, "rubric_") == ["rubric_evaluation_start", "rubric_evaluation"]
    assert [e["result"] for e in events if e["type"] == "rubric_evaluation"] == [
        "satisfied"
    ]
    # Neither the grader nor its file check nor its verdict call is a row.
    assert _kinds(events, "subagent") == []
    assert _kinds(events, "tool_") == []
    assert [e for e in events if e["type"] == "done"][-1]["response"] == "done"


async def test_a_retried_grader_call_is_still_announced(tmp_path):
    events = await _graded_turn(
        tmp_path, [ConnectionError("dropped"), PASS], model_retry_wait_seconds=60
    )

    retries = [e for e in events if e["type"] == "model_retry"]
    assert [r["subagent"] for r in retries] == ["rubric_grader"]
    assert retries[0]["discarded"] == {
        "text_chars": 0,
        "thinking_chars": 0,
        "tool_call_ids": [],
    }
    assert _kinds(events, "subagent") == []
    # Announced while the grader waits, not when the stream closes.
    kinds = [e["type"] for e in events]
    assert kinds.index("model_retry") < kinds.index("rubric_evaluation")
    assert [e["result"] for e in events if e["type"] == "rubric_evaluation"] == [
        "satisfied"
    ]
