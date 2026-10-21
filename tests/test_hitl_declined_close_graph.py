"""Closing a declined approval on the real main graph: what is written and what the next turn sees."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.backends import build_hitl_resume, close_parked_checkpoint
from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.gateway.types import GraphTarget
from EvoScientist.paths import Workspace
from EvoScientist.stream.events import stream_agent_events
from tests.fakes.common import close_result

pytestmark = pytest.mark.usefixtures("no_mcp")


class _Scripted(BaseChatModel):
    """Replays a script for the main agent and its sub-agents; records what each call saw."""

    script: list[Any]
    seen: list[list[Any]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        message = self.script.pop(0) if self.script else AIMessage("extra")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _agent(tmp_path, script):
    model = _Scripted(script=list(script), seen=[])
    cfg = EvoScientistConfig(enable_async_subagents=False, memory_workers_enabled=False)
    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    return agent, model


async def _turn(agent, message, *, extra=None) -> list[dict]:
    return [
        event
        async for event in stream_agent_events(
            agent,
            message,
            "t",
            configurable_extra={"hitl_suppressed": False, **(extra or {})},
        )
    ]


async def _state(agent):
    return await agent.aget_state({"configurable": {"thread_id": "t"}})


async def _close(agent, reason: str) -> None:
    await close_parked_checkpoint(
        LocalGraphGateway(), GraphTarget(local_graph=agent), "t", reason=reason
    )


def _results(state) -> dict[str, str]:
    return {
        message.tool_call_id: str(message.content)
        for message in state.values["messages"]
        if message.type == "tool"
    }


def _facts(state) -> dict[str, tuple[str, str, bool]]:
    """Per call: what its result says (``close_result``)."""
    return {call: close_result(text) for call, text in _results(state).items()}


def _files(tmp_path) -> list[str]:
    return sorted(path.name for path in tmp_path.glob("*.txt"))


async def test_a_declined_command_is_closed_and_the_next_turn_starts_clean(tmp_path):
    agent, model = _agent(
        tmp_path,
        [
            AIMessage(
                "", tool_calls=[_call("c1", "execute", command="echo 1 > one.txt")]
            ),
            AIMessage("second answer"),
        ],
    )
    first = await _turn(agent, "go")
    assert [e["type"] for e in first if e["type"] in ("interrupt", "done")] == [
        "interrupt",
        "done",
    ]

    await _close(agent, "rejected")

    state = await _state(agent)
    assert state.next == ()
    assert not any(task.interrupts for task in state.tasks)
    assert _facts(state) == {"c1": ("asked", "rejected", True)}

    second = await _turn(agent, "what happened?")
    assert [e["type"] for e in second if e["type"] == "tool_call"] == []
    told = [str(m.content) for m in model.seen[-1] if m.type == "tool"]
    assert told == [_results(state)["c1"]]
    assert _files(tmp_path) == []


async def test_a_batch_tells_the_asked_call_from_its_ungated_sibling(tmp_path):
    agent, _ = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call("w1", "write_file", file_path="/note.txt", content="x"),
                    _call("c1", "execute", command="echo 1 > one.txt"),
                ],
            )
        ],
    )
    await _turn(agent, "go")
    await _close(agent, "unanswered")

    assert _facts(await _state(agent)) == {
        "c1": ("asked", "unanswered", True),
        "w1": ("not executed", "unanswered", False),
    }
    assert _files(tmp_path) == []


async def test_a_declined_command_inside_a_sub_agent_leaves_the_task_unfinished(
    tmp_path,
):
    """The sub-agent had already written a file: the ``task`` call did not finish,
    it was not "not executed"."""
    agent, _ = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="write two files",
                    )
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    _call("s0", "write_file", file_path="/done.txt", content="x")
                ],
            ),
            AIMessage(
                "", tool_calls=[_call("s1", "execute", command="echo 2 > two.txt")]
            ),
        ],
    )
    await _turn(agent, "go")
    assert _files(tmp_path) == ["done.txt"]

    await _close(agent, "rejected")

    state = await _state(agent)
    assert state.next == ()
    assert _facts(state) == {"t1": ("did not finish", "rejected", False)}
    assert _files(tmp_path) == ["done.txt"]


async def test_a_question_nobody_answered_is_not_called_a_rejection(tmp_path):
    agent, _ = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "q1",
                        "ask_user",
                        questions=[{"question": "Continue?", "type": "text"}],
                    )
                ],
            )
        ],
    )
    events = await _turn(agent, "go")
    assert "ask_user" in [e["type"] for e in events]

    await _close(agent, "unanswered")

    assert _facts(await _state(agent)) == {
        "q1": ("did not finish", "unanswered", False)
    }


async def test_closing_leaves_a_one_shot_checklist_as_an_interrupted_turn_does(
    tmp_path,
):
    """A turn under a ``next`` checklist that ends at a declined approval keeps
    the checklist and its grading for the next message, closed or not."""
    marker = {
        "rubric_run": {
            "mode": "thread",
            "changes": [{"session": "s1", "seq": 1, "next": "- one.txt exists"}],
        }
    }
    keys = ("rubric", "_next_rubric", "_sticky_rubric", "_rubric_applied")

    async def _parked(folder, *, close: bool):
        folder.mkdir()
        agent, _ = _agent(
            folder,
            [
                AIMessage(
                    "", tool_calls=[_call("c1", "execute", command="echo 1 > one.txt")]
                )
            ],
        )
        await _turn(agent, "go", extra=marker)
        if close:
            await _close(agent, "rejected")
        values = (await _state(agent)).values
        grading = dict(values.get("_rubric_grading") or {})
        first_human = next(m.id for m in values["messages"] if m.type == "human")
        started_at_the_task = grading.pop("start", None) == first_human
        return {key: values.get(key) for key in keys}, grading, started_at_the_task

    left, closed = (
        await _parked(tmp_path / "left", close=False),
        await _parked(tmp_path / "closed", close=True),
    )
    assert closed == left
    assert left[0]["_next_rubric"] == {"text": "- one.txt exists", "id": "s1:1"}
    assert left[2] is True


async def test_a_sibling_sub_agent_that_finished_keeps_its_result(tmp_path):
    """Two sub-agents in one step: one asks to run a command, the other does its
    work. Once the stream has ended the decline closes the first and leaves the
    second's result as it is."""
    agent, _ = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1", "task", subagent_type="general-purpose", description="a"
                    ),
                    _call(
                        "t2", "task", subagent_type="general-purpose", description="b"
                    ),
                ],
            ),
            # The two sub-agents take these in whichever order they start.
            AIMessage(
                "", tool_calls=[_call("s1", "execute", command="echo 1 > one.txt")]
            ),
            AIMessage(
                "",
                tool_calls=[
                    _call("s2", "write_file", file_path="/sibling.txt", content="x")
                ],
            ),
            AIMessage("sibling done"),
        ],
    )
    events = await _turn(agent, "go")
    assert [e["type"] for e in events].count("interrupt") == 1
    assert _files(tmp_path) == ["sibling.txt"]

    await _close(agent, "rejected")

    state = await _state(agent)
    assert state.next == ()
    results = sorted(_results(state).values())
    assert len(results) == 2
    assert "sibling done" in results
    assert ("did not finish", "rejected", False) in map(close_result, results)
    # Nothing is written after the close.
    assert (await _state(agent)).values["messages"] == state.values["messages"]
    assert _files(tmp_path) == ["sibling.txt"]


async def test_the_stream_says_which_sub_agent_ends_are_pauses(tmp_path):
    """A sub-agent that stops at an approval has not ended: it goes on once
    the approval is granted, and only then ends."""
    agent, _ = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="run one command",
                    )
                ],
            ),
            AIMessage(
                "", tool_calls=[_call("s1", "execute", command="echo 1 > one.txt")]
            ),
            AIMessage("sub done"),
            AIMessage("all done"),
        ],
    )

    def _ends(events: list[dict]) -> list[bool]:
        return [bool(e.get("paused")) for e in events if e["type"] == "subagent_end"]

    asked = await _turn(agent, "go")
    assert _ends(asked) == [True]

    (approval,) = [e for e in asked if e["type"] == "interrupt"]
    granted = build_hitl_resume(approval["interrupt_id"], [{"type": "approve"}])
    resumed = await _turn(agent, granted)
    assert _ends(resumed) == [False]
    assert _files(tmp_path) == ["one.txt"]
