"""A session "approve all" on the real main graph: the grant key and the order it must travel in."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.paths import Workspace
from EvoScientist.stream.events import stream_agent_events


class _Scripted(GenericFakeChatModel):
    """Returns the scripted messages in order, for the main agent and its sub-agents alike."""

    def bind_tools(self, tools, **kwargs):
        return self


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _execute(call_id: str, command: str) -> AIMessage:
    return AIMessage("", tool_calls=[_call(call_id, "execute", command=command)])


def _agent(tmp_path, messages):
    cfg = EvoScientistConfig(enable_async_subagents=False, memory_workers_enabled=False)
    return create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=_Scripted(messages=iter([*messages, *[AIMessage("extra")] * 4])),
        checkpointer=InMemorySaver(),
    )


def _config(*, granted: bool = False, suppressed: bool = False) -> dict:
    configurable = {"thread_id": "t", "hitl_suppressed": suppressed}
    if granted:
        configurable["hitl_approve_all"] = True
    return {"configurable": configurable}


def _resume(interrupt, *decisions) -> Command:
    return Command(resume={interrupt.id: {"decisions": list(decisions)}})


def _written(tmp_path) -> list[str]:
    return sorted(p.name for p in tmp_path.glob("*.txt"))


def _tool_text(result, call_id: str) -> str:
    return next(
        str(m.content)
        for m in result["messages"]
        if m.type == "tool" and m.tool_call_id == call_id
    )


APPROVE = {"type": "approve"}
REJECT = {"type": "reject", "message": "not this one"}


def _edit(command: str) -> dict:
    return {
        "type": "edit",
        "edited_action": {"name": "execute", "args": {"command": command}},
    }


async def test_a_granted_call_runs_gated_tools_without_interrupting(tmp_path):
    armed = _agent(tmp_path / "armed", [_execute("c1", "echo 1 > one.txt")])
    (tmp_path / "armed").mkdir(exist_ok=True)
    parked = await armed.ainvoke({"messages": [("user", "go")]}, config=_config())
    assert len(parked["__interrupt__"]) == 1
    assert _written(tmp_path / "armed") == []

    (tmp_path / "granted").mkdir(exist_ok=True)
    granted = _agent(
        tmp_path / "granted", [_execute("c1", "echo 1 > one.txt"), AIMessage("Done.")]
    )
    result = await granted.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert not result.get("__interrupt__")
    assert _written(tmp_path / "granted") == ["one.txt"]


async def test_the_granting_answer_holds_and_the_next_call_silences_the_turn(tmp_path):
    """The answer that grants goes out without the key, so its decisions apply;
    the key rides on the call after it and nothing interrupts again."""
    agent = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call("a", "execute", command="echo a > a.txt"),
                    _call("b", "execute", command="echo b > b.txt"),
                    _call("x", "execute", command="echo x > x.txt"),
                ],
            ),
            _execute("c", "echo c > c.txt"),
            _execute("d", "echo d > d.txt"),
            AIMessage("All done."),
        ],
    )
    first = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    (parked,) = first["__interrupt__"]
    assert len(parked.value["action_requests"]) == 3

    second = await agent.ainvoke(
        _resume(parked, APPROVE, REJECT, _edit("echo edited > e.txt")), config=_config()
    )
    assert _written(tmp_path) == ["a.txt", "e.txt"]
    (next_parked,) = second["__interrupt__"]

    third = await agent.ainvoke(
        _resume(next_parked, APPROVE), config=_config(granted=True)
    )
    assert not third.get("__interrupt__")
    assert _written(tmp_path) == ["a.txt", "c.txt", "d.txt", "e.txt"]
    assert "rejected" in _tool_text(third, "b")
    assert third["messages"][-1].text == "All done."


async def test_a_sync_subagent_sees_the_parents_grant(tmp_path):
    def _script():
        return [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="write a file",
                    )
                ],
            ),
            _execute("s1", "echo sub > sub.txt"),
            AIMessage("Sub done."),
            AIMessage("Main done."),
        ]

    (tmp_path / "armed").mkdir()
    armed = await _agent(tmp_path / "armed", _script()).ainvoke(
        {"messages": [("user", "go")]}, config=_config()
    )
    assert len(armed["__interrupt__"]) == 1
    assert _written(tmp_path / "armed") == []

    (tmp_path / "granted").mkdir()
    granted = await _agent(tmp_path / "granted", _script()).ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert not granted.get("__interrupt__")
    assert _written(tmp_path / "granted") == ["sub.txt"]


async def test_a_granted_fresh_turn_does_not_run_a_batch_left_parked(tmp_path):
    """A declined approval leaves the thread parked; the next message is new
    input, not an answer, so the grant must not release the declined call."""
    agent = _agent(
        tmp_path, [_execute("c1", "echo 1 > declined.txt"), AIMessage("Fresh.")]
    )
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    assert len(parked["__interrupt__"]) == 1

    events = [
        event
        async for event in stream_agent_events(
            agent, "something else", "t", configurable_extra={"hitl_approve_all": True}
        )
    ]
    assert _written(tmp_path) == []
    assert not [e for e in events if e["type"] == "interrupt"]
    assert next(e for e in events if e["type"] == "done")["response"] == "Fresh."


async def test_a_granted_fresh_turn_does_not_run_a_subagents_parked_batch(tmp_path):
    """The same for an approval parked inside a sync sub-agent."""
    agent = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="write a file",
                    )
                ],
            ),
            _execute("s1", "echo sub > declined.txt"),
            AIMessage("Fresh."),
        ],
    )
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    assert len(parked["__interrupt__"]) == 1

    events = [
        event
        async for event in stream_agent_events(
            agent, "something else", "t", configurable_extra={"hitl_approve_all": True}
        )
    ]
    assert _written(tmp_path) == []
    assert not [e for e in events if e["type"] == "interrupt"]
    assert next(e for e in events if e["type"] == "done")["response"] == "Fresh."


async def test_the_grant_does_not_arm_the_dangerous_guard_but_suppression_does(
    tmp_path,
):
    (tmp_path / "granted").mkdir()
    granted = await _agent(
        tmp_path / "granted", [_execute("c1", "echo true | bash"), AIMessage("Done.")]
    ).ainvoke({"messages": [("user", "go")]}, config=_config(granted=True))
    assert not granted.get("__interrupt__")
    assert "Command blocked" not in _tool_text(granted, "c1")

    (tmp_path / "both").mkdir()
    both = await _agent(
        tmp_path / "both", [_execute("c1", "echo true | bash"), AIMessage("Done.")]
    ).ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True, suppressed=True)
    )
    assert not both.get("__interrupt__")
    assert "Command blocked" in _tool_text(both, "c1")
