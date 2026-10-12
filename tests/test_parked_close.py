"""Closing a parked approval or question: the reason, the sentence per call, the completeness check."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from EvoScientist.backends import (
    HITL_ROUND_LIMIT_REJECT_MESSAGE,
    abandoned_tool_messages,
    close_parked_checkpoint,
)
from EvoScientist.gateway import GraphTarget
from EvoScientist.gateway.server import (
    LangGraphServerGateway,
    LangGraphServerThreadStore,
)
from tests.fakes import (
    FakeCheckpointAgent,
    FakeGraphGateway,
    FakeLangGraphClient,
    FakeLangGraphThreadsClient,
)

BEFORE_TOOLS = ("HumanInTheLoopMiddleware.after_model",)
NO_RETRY = "Do not retry this tool call unless the user explicitly requests it."
CAUTION = (
    "Anything it had already done may have taken effect or may still be "
    "finishing; check the current state instead of re-running it unless the "
    "user asks."
)


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _invalid(call_id: str | None, name: str = "execute") -> dict:
    return {
        "name": name,
        "args": "not-json",
        "id": call_id,
        "error": "parse",
        "type": "invalid_tool_call",
    }


def _ai(*calls: dict, invalid: tuple[dict, ...] = ()) -> AIMessage:
    return AIMessage(
        content="", tool_calls=list(calls), invalid_tool_calls=list(invalid)
    )


def _texts(patch: list) -> dict[str, str]:
    return {message.tool_call_id: message.content for message in patch}


# -- the sentence per call ---------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            "rejected",
            "User rejected the tool call for `execute` with id c1. The tool was "
            f"not executed. {NO_RETRY}",
        ),
        (
            "unanswered",
            "The approval request for the tool call `execute` with id c1 was "
            f"left unanswered. The tool was not executed. {NO_RETRY}",
        ),
        (
            "round_limit",
            "User rejected the tool call for `execute` with id c1 (approval "
            f"round limit reached). The tool was not executed. {NO_RETRY}",
        ),
    ],
)
def test_a_gated_call_parked_before_tools_was_asked_and_not_executed(reason, expected):
    patch = abandoned_tool_messages(
        [_ai(_call("c1", "execute", command="ls"))],
        reason=reason,
        parked_before_tools=True,
    )
    assert [message.content for message in patch] == [expected]
    assert patch[0].status == "error"
    assert patch[0].name == "execute"


@pytest.mark.parametrize(
    ("reason", "clause"),
    [
        ("rejected", "the turn ended when the user rejected an approval request"),
        (
            "unanswered",
            "the turn ended with an approval request or question left unanswered",
        ),
        ("round_limit", HITL_ROUND_LIMIT_REJECT_MESSAGE),
    ],
)
def test_calls_the_user_was_not_asked_about_get_no_retry_ban(reason, clause):
    """An ungated sibling, a question that was never shown and a malformed call."""
    patch = abandoned_tool_messages(
        [
            _ai(
                _call("c1", "execute", command="ls"),
                _call("w1", "write_file", file_path="/a.txt", content="x"),
                _call("q1", "ask_user", questions=[]),
                invalid=(_invalid("bad-1"),),
            )
        ],
        reason=reason,
        parked_before_tools=True,
    )
    texts = _texts(patch)
    for call_id, name in (
        ("w1", "write_file"),
        ("q1", "ask_user"),
        ("bad-1", "execute"),
    ):
        assert texts[call_id] == (
            f"The tool call for `{name}` with id {call_id} was not executed: {clause}."
        )
    assert NO_RETRY in texts["c1"]


@pytest.mark.parametrize(
    ("reason", "clause"),
    [
        ("rejected", "the user rejected an approval request"),
        ("unanswered", "an approval request or question was left unanswered"),
        ("round_limit", HITL_ROUND_LIMIT_REJECT_MESSAGE),
    ],
)
def test_a_call_that_had_started_did_not_finish(reason, clause):
    patch = abandoned_tool_messages(
        [_ai(_call("t1", "task", description="work"))], reason=reason
    )
    assert _texts(patch) == {
        "t1": f"The tool call for `task` with id t1 did not finish ({clause}). {CAUTION}"
    }


def test_an_invalid_call_beside_a_running_question_was_not_executed():
    """Parked in ``tools`` by the question; the malformed call never ran."""
    patch = abandoned_tool_messages(
        [_ai(_call("q1", "ask_user", questions=[]), invalid=(_invalid("bad-1"),))],
        reason="unanswered",
    )
    texts = _texts(patch)
    assert "did not finish" in texts["q1"]
    assert texts["bad-1"] == (
        "The tool call for `execute` with id bad-1 was not executed: the turn "
        "ended with an approval request or question left unanswered."
    )


def test_a_call_id_answered_in_an_earlier_turn_is_still_closed():
    """Some providers repeat call ids across turns: only results after the last
    AI message answer its calls."""
    patch = abandoned_tool_messages(
        [
            _ai(_call("c1", "execute", command="ls")),
            ToolMessage(content="done", name="execute", tool_call_id="c1"),
            _ai(_call("c1", "execute", command="pwd")),
        ],
        reason="rejected",
        parked_before_tools=True,
    )
    assert [message.tool_call_id for message in patch] == ["c1"]


def test_a_gated_name_parked_in_tools_is_not_called_unexecuted():
    """Only the position before tools proves that nothing ran."""
    patch = abandoned_tool_messages(
        [_ai(_call("c1", "execute", command="ls"))], reason="rejected"
    )
    assert "did not finish" in patch[0].content
    assert "was not executed" not in patch[0].content


# -- through the close ---------------------------------------------------------


async def _close(agent: Any, reason: str = "rejected") -> None:
    await close_parked_checkpoint(
        FakeGraphGateway(checkpoint=agent), GraphTarget(), "t-close", reason=reason
    )


def _written(agent: Any) -> dict[str, str]:
    for values, _node in agent.updates:
        if isinstance(values, dict):
            return _texts(values["messages"])
    return {}


async def test_parked_before_tools_separates_asked_from_not_asked():
    agent = FakeCheckpointAgent(
        next=BEFORE_TOOLS,
        values={
            "messages": [
                _ai(
                    _call("c1", "execute", command="ls"),
                    _call("w1", "write_file", file_path="/a.txt", content="x"),
                )
            ]
        },
    )
    await _close(agent)
    written = _written(agent)
    assert written["c1"].startswith("User rejected the tool call for `execute`")
    assert written["w1"] == (
        "The tool call for `write_file` with id w1 was not executed: the turn "
        "ended when the user rejected an approval request."
    )


@pytest.mark.parametrize(
    "position",
    [
        ("tools",),
        ("tools", "tools"),
        ("somewhere",),
        # Only a park where every node is the approval node proves nothing ran.
        (*BEFORE_TOOLS, "tools"),
    ],
)
async def test_any_other_position_gets_the_cautious_sentence(position):
    agent = FakeCheckpointAgent(
        next=position,
        values={"messages": [_ai(_call("t1", "task", description="work"))]},
    )
    await _close(agent, "unanswered")
    assert _written(agent) == {
        "t1": "The tool call for `task` with id t1 did not finish (an approval "
        f"request or question was left unanswered). {CAUTION}"
    }


async def test_a_pause_without_a_position_gets_the_cautious_sentence():
    agent = FakeCheckpointAgent(
        next=(), values={"messages": [_ai(_call("c1", "execute", command="ls"))]}
    )
    agent.interrupts = (object(),)
    await _close(agent, "rejected")
    assert _written(agent) == {
        "c1": "The tool call for `execute` with id c1 did not finish (the user "
        f"rejected an approval request). {CAUTION}"
    }


async def test_the_default_reason_is_the_round_limit():
    agent = FakeCheckpointAgent(
        next=BEFORE_TOOLS,
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
    )
    await close_parked_checkpoint(
        FakeGraphGateway(checkpoint=agent), GraphTarget(), "t-limit"
    )
    assert _written(agent) == {
        "c1": "User rejected the tool call for `execute` with id c1 (approval "
        f"round limit reached). The tool was not executed. {NO_RETRY}"
    }


class _ParkedServerThreads(FakeLangGraphThreadsClient):
    """Applies state updates the way the server does, on the dict state it returns."""

    async def update_state(self, thread_id, values, *, as_node=None):
        self.state_updates.append((thread_id, values, as_node))
        state = self.states[thread_id]
        if as_node == "__end__":
            parked = list(state["next"]) not in ([], ["model"])
            self.states[thread_id] = {
                **state,
                "next": ["model"] if parked else [],
                "tasks": [],
                "interrupts": [],
            }
        elif isinstance(values, dict):
            sent = [
                {
                    "type": "tool",
                    "content": message.content,
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "status": message.status,
                }
                for message in values["messages"]
            ]
            messages = [*state["values"]["messages"], *sent]
            self.states[thread_id] = {**state, "values": {"messages": messages}}
        return {"checkpoint": {"thread_id": thread_id}}


def _parked_server_state() -> dict:
    request = {
        "id": "i-1",
        "value": {"action_requests": [{"name": "execute", "args": {"command": "ls"}}]},
    }
    return {
        "next": list(BEFORE_TOOLS),
        "tasks": [{"name": BEFORE_TOOLS[0], "interrupts": [request]}],
        "interrupts": [request],
        "values": {
            "messages": [
                {"type": "human", "content": "go"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        _call("c1", "execute", command="ls"),
                        _call("w1", "write_file", file_path="/a.txt", content="x"),
                    ],
                },
            ]
        },
    }


def _server_gateway(threads) -> LangGraphServerGateway:
    return LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )


async def test_the_close_through_the_server_gateway_classifies_like_the_local_one():
    """The server returns dict state; the three writes and the sentences are the same."""
    threads = _ParkedServerThreads(threads=[], states={"t-srv": _parked_server_state()})
    await close_parked_checkpoint(
        _server_gateway(threads), GraphTarget(), "t-srv", reason="rejected"
    )
    assert [node for _tid, _values, node in threads.state_updates] == [
        "__end__",
        BEFORE_TOOLS[0],
        "__end__",
    ]
    written = _texts(threads.state_updates[1][1]["messages"])
    assert written["c1"].startswith("User rejected the tool call for `execute`")
    assert written["w1"] == (
        "The tool call for `write_file` with id w1 was not executed: the turn "
        "ended when the user rejected an approval request."
    )
    assert threads.states["t-srv"]["next"] == []


async def test_a_server_that_refuses_the_write_fails_the_close_with_one_short_line(
    caplog,
):
    """A run still in flight on the server: ``update_state`` answers 409."""
    import httpx
    from langgraph_sdk.errors import ConflictError

    class _Busy(_ParkedServerThreads):
        async def update_state(self, thread_id, values, *, as_node=None):
            raise ConflictError(
                "Thread has in-flight runs\n" + "x" * 5000,
                response=httpx.Response(
                    409, request=httpx.Request("POST", "http://server/threads/t/state")
                ),
                body=None,
            )

    threads = _Busy(threads=[], states={"t-busy": _parked_server_state()})
    with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
        with pytest.raises(
            RuntimeError, match="Could not close parked HITL checkpoint"
        ):
            await close_parked_checkpoint(
                _server_gateway(threads), GraphTarget(), "t-busy", reason="rejected"
            )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert len(warnings[0].getMessage()) < 400
    assert "\n" not in warnings[0].getMessage()
    assert warnings[0].exc_info is None
    assert threads.states["t-busy"]["next"] == list(BEFORE_TOOLS)


# -- completeness ----------------------------------------------------------------


class _LosesTheResults(FakeCheckpointAgent):
    """The write of the results reports success but nothing lands."""

    async def aupdate_state(self, _config, values, as_node=None):
        if isinstance(values, dict):
            self.updates.append((values, as_node))
            return
        await super().aupdate_state(_config, values, as_node)


async def test_a_close_whose_results_did_not_land_fails():
    agent = _LosesTheResults(
        next=BEFORE_TOOLS,
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
    )
    with pytest.raises(RuntimeError, match="Could not close parked HITL checkpoint"):
        await _close(agent)


async def test_a_half_closed_thread_is_reported_not_called_clean():
    """Nothing is pending, yet a call has no result: an earlier close stopped half-way."""
    agent = FakeCheckpointAgent(
        next=(), values={"messages": [_ai(_call("c1", "execute", command="ls"))]}
    )
    with pytest.raises(RuntimeError, match="Could not close parked HITL checkpoint"):
        await _close(agent)
    assert agent.updates == []


async def test_crash_recovery_does_not_apply_the_close_checks(caplog):
    """Only a close reports a call left without a result; recovery after a
    crash answers as before."""
    from EvoScientist.stream.events import _recover_interrupted_graph_state

    config = {"configurable": {"thread_id": "t1"}}
    messages = [_ai(_call("c1", "execute", command="ls"))]
    clean = FakeCheckpointAgent(next=(), values={"messages": list(messages)})
    crashed = _LosesTheResults(next=("tools",), values={"messages": list(messages)})
    crashed.interrupts = ()
    with caplog.at_level(logging.WARNING, logger="EvoScientist"):
        assert await _recover_interrupted_graph_state(clean, config) is True
        assert await _recover_interrupted_graph_state(crashed, config) is True
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert clean.updates == []


async def test_messages_that_cannot_be_read_are_not_proof_of_a_clean_close():
    agent = FakeCheckpointAgent(next=(), values={"messages": [object()]})
    with pytest.raises(RuntimeError, match="Could not close parked HITL checkpoint"):
        await _close(agent)


async def test_a_closed_thread_can_be_closed_again_without_writes():
    agent = FakeCheckpointAgent(
        next=(),
        values={
            "messages": [
                _ai(_call("c1", "execute", command="ls")),
                ToolMessage(content="done", name="execute", tool_call_id="c1"),
            ]
        },
    )
    await _close(agent)
    assert agent.updates == []


async def test_an_invalid_call_without_an_id_does_not_fail_a_healthy_close():
    agent = FakeCheckpointAgent(
        next=BEFORE_TOOLS,
        values={
            "messages": [
                _ai(_call("c1", "execute", command="ls"), invalid=(_invalid(None),))
            ]
        },
    )
    await _close(agent)
    assert list(_written(agent)) == ["c1"]


async def test_a_failing_close_logs_one_short_line(caplog):
    agent = FakeCheckpointAgent(
        next=BEFORE_TOOLS,
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
        update_error=RuntimeError("409 conflict\nrun kwargs:\n" + "x" * 5000),
    )
    with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
        with pytest.raises(RuntimeError):
            await _close(agent)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert len(warnings[0].getMessage()) < 400
    assert "\n" not in warnings[0].getMessage()
    assert warnings[0].exc_info is None


class _NeverClears(FakeCheckpointAgent):
    """Accepts every write and stays parked, with a task that prints at length."""

    async def aupdate_state(self, _config, values, as_node=None):
        self.updates.append((values, as_node))


async def test_a_thread_that_stays_stuck_is_reported_in_one_short_line(caplog):
    agent = _NeverClears(
        next=BEFORE_TOOLS,
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
    )
    # No ``next``, so the position is read from the task, which prints at length.
    agent.next = ()
    agent.tasks = (
        SimpleNamespace(name="tools", interrupts=(), payload="line\n" + "y" * 5000),
    )
    with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
        with pytest.raises(RuntimeError):
            await _close(agent)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert len(warnings[0].getMessage()) < 400
    assert "\n" not in warnings[0].getMessage()


async def test_crash_recovery_keeps_its_full_stuck_report(caplog):
    """Only the close shortens the line; recovery after a crash reports as before."""
    from EvoScientist.stream.events import _recover_interrupted_graph_state

    agent = _NeverClears(
        next=BEFORE_TOOLS,
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
    )
    agent.next = ()
    agent.tasks = (
        SimpleNamespace(name="tools", interrupts=(), payload="line\n" + "y" * 5000),
    )
    agent.interrupts = ()  # a crash, not a pause: recovery acts on it
    with caplog.at_level(logging.WARNING, logger="EvoScientist"):
        ok = await _recover_interrupted_graph_state(
            agent, {"configurable": {"thread_id": "t1"}}
        )
    assert ok is False
    (warning,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning.getMessage()) > 5000


async def test_a_cancelled_close_says_so_once(caplog):
    """A cancellation is not an error the close reports on its own."""
    agent = FakeCheckpointAgent(
        next=BEFORE_TOOLS,
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
        update_error=asyncio.CancelledError(),
    )
    with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
        with pytest.raises(asyncio.CancelledError):
            await _close(agent)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "was cancelled before it finished" in warnings[0].getMessage()
    assert warnings[0].exc_info is None


async def test_a_cancelled_crash_recovery_logs_nothing(caplog):
    from EvoScientist.stream.events import _recover_interrupted_graph_state

    agent = FakeCheckpointAgent(
        next=("tools",),
        values={"messages": [_ai(_call("c1", "execute", command="ls"))]},
        update_error=asyncio.CancelledError(),
    )
    agent.interrupts = ()
    with caplog.at_level(logging.WARNING, logger="EvoScientist"):
        with pytest.raises(asyncio.CancelledError):
            await _recover_interrupted_graph_state(
                agent, {"configurable": {"thread_id": "t1"}}
            )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
