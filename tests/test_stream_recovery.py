"""Targeted tests for interrupted-graph-state recovery.

These run against a real compiled LangGraph graph with a checkpointer,
so they actually verify the claims the recovery rests on:

1. After a mid-run crash, ``aupdate_state(config, None, as_node=END)`` clears the
   stuck ``next`` tuple while preserving channel values.
2. A legitimate human-in-the-loop ``interrupt()`` (also a non-empty ``next``) is
   left intact, so a pending question is never silently discarded.
3. A run cancelled mid-tools leaves dangling tool calls; recovery closes them
   with honest synthetic results so the next turn neither replays the tool
   batch nor floods the UI with historical tool calls.
4. If recovery itself fails (e.g. a transient checkpoint error), the next run
   is refused instead of risking a replay of the interrupted tool batch.
"""

import asyncio
from typing import Any, TypedDict
from unittest.mock import patch as mock_patch

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from EvoScientist.middleware.tool_history_repair import ToolHistoryRepairMiddleware
from EvoScientist.stream import events as events_module
from EvoScientist.stream.events import (
    _INTERRUPTED_TOOL_RESULT,
    _recover_interrupted_graph_state,
    stream_agent_events,
)


class _S(TypedDict):
    """Minimal state schema for the hand-built test graphs below."""

    x: int


def _crashing_app():
    """Build a graph whose node 'b' crashes once, then succeeds.

    A post-recovery run can therefore complete and prove the graph is
    genuinely unstuck (not replaying the dead step).
    """
    crashed = {"v": False}

    def a(state):
        return {"x": state["x"] + 1}

    def b(state):
        if not crashed["v"]:
            crashed["v"] = True
            raise RuntimeError("boom")
        return {"x": state["x"] + 100}

    g = StateGraph(_S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=InMemorySaver())


def _interrupting_app():
    """Build a graph that parks at a genuine ``interrupt()`` HITL pause."""

    def ask(state):
        interrupt({"question": "continue?"})
        return {"x": state["x"] + 1}

    g = StateGraph(_S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    return g.compile(checkpointer=InMemorySaver())


async def test_recovery_clears_stuck_state_after_crash():
    """Recovery clears a crash-stuck ``next`` and preserves channel values."""
    app = _crashing_app()
    cfg = {"configurable": {"thread_id": "t1"}}
    try:
        app.invoke({"x": 0}, cfg)
    except Exception:
        pass  # LangGraph re-raises the node error (wrapped); we only care about state
    # The crash left the graph frozen at node 'b'.
    assert app.get_state(cfg).next == ("b",)

    assert await _recover_interrupted_graph_state(app, cfg) is True

    snap = app.get_state(cfg)
    assert snap.next == ()  # stuck state actually cleared
    assert snap.values == {"x": 1}  # channel values (history) preserved

    # And the graph is genuinely unstuck: a fresh run completes (a: +1, b: +100)
    # instead of replaying the dead node.
    assert app.invoke({"x": 41}, cfg)["x"] == 142


async def test_recovery_preserves_pending_hitl_interrupt():
    """A pending ``interrupt()`` pause is left intact and stays resumable."""
    app = _interrupting_app()
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"x": 0}, cfg)  # parks at interrupt()
    before = app.get_state(cfg)
    assert before.next == ("ask",)
    assert before.interrupts

    assert await _recover_interrupted_graph_state(app, cfg) is True

    after = app.get_state(cfg)
    assert after.next == ("ask",)  # interrupt left intact, still resumable
    assert after.interrupts


# --- Full-stack regression: Ctrl+C mid-tools must not replay the batch -------
#
# These mirror the real construction path (deepagents create_deep_agent with
# the project's ToolHistoryRepairMiddleware) so deepagents'
# PatchToolCallsMiddleware — the layer whose "was cancelled" rewrite causes the
# replay — is in the stack too.

_TOOL_SIDE_EFFECTS: list[str] = []
_MODEL_REQUESTS: list[list[BaseMessage]] = []
_SCRIPT: list[AIMessage] = []
_script_idx = {"i": 0}


class _ScriptedModel(BaseChatModel):
    """Replays a scripted AI-message sequence; records every request."""

    @property
    def _llm_type(self) -> str:
        """Identifier required by the BaseChatModel interface."""
        return "scripted"

    def bind_tools(self, tools, **kwargs: Any) -> "_ScriptedModel":
        """Accept any tool binding; the scripted responses ignore tools."""
        return self

    def _generate(
        self, messages, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        """Return the next scripted message, recording the request history."""
        _MODEL_REQUESTS.append(list(messages))
        msg = _SCRIPT[_script_idx["i"]]
        _script_idx["i"] += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def _slow_edit(path: str, content: str = "data") -> str:
    """Edit a file; the side effect lands immediately, like edit_file."""
    _TOOL_SIDE_EFFECTS.append(path)
    await asyncio.sleep(2.0)
    return f"edited {path}"


def _tool_call(id_: str, path: str) -> dict[str, Any]:
    """Build a tool-call entry for ``_slow_edit`` on ``path``."""
    return {
        "id": id_,
        "name": _slow_edit.__name__,
        "args": {"path": path},
        "type": "tool_call",
    }


def _build_agent():
    """Build a deep agent mirroring the real EvoScientist middleware stack."""
    return create_deep_agent(
        model=_ScriptedModel(),
        tools=[_slow_edit],
        middleware=[ToolHistoryRepairMiddleware()],
        checkpointer=InMemorySaver(),
    )


def _script_tool_batch_then_ack() -> None:
    """Reset shared script/log state: 3-call tool batch, then an ack."""
    _TOOL_SIDE_EFFECTS.clear()
    _MODEL_REQUESTS.clear()
    _SCRIPT.clear()
    _script_idx["i"] = 0
    _SCRIPT.extend(
        [
            AIMessage(
                content="applying the batch of edits",
                tool_calls=[
                    _tool_call("call_1", "a.py"),
                    _tool_call("call_2", "b.py"),
                    _tool_call("call_3", "c.py"),
                ],
            ),
            AIMessage(content="acknowledged"),
        ]
    )


async def _run_turn(agent, message: str, thread_id: str) -> list[dict[str, Any]]:
    """Stream one full turn through the real ``stream_agent_events``."""
    collected = []
    async for ev in stream_agent_events(agent, message, thread_id):
        collected.append(ev)
    return collected


async def _cancel_mid_tools(agent, thread_id: str) -> None:
    """Run a turn whose tool batch is mid-flight, then cancel it (Ctrl+C)."""
    task = asyncio.create_task(_run_turn(agent, "apply the edits", thread_id))
    deadline = asyncio.get_running_loop().time() + 10
    while len(_TOOL_SIDE_EFFECTS) < 3 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _tool_messages(messages: Any) -> list[ToolMessage]:
    """Extract the ToolMessages from a message sequence."""
    return [m for m in messages if isinstance(m, ToolMessage)]


async def test_cancel_mid_tools_closes_dangling_calls_and_next_turn_is_clean():
    """Ctrl+C mid-tools must close the batch honestly; next turn stays clean."""
    _script_tool_batch_then_ack()
    agent = _build_agent()
    cfg = {"configurable": {"thread_id": "t-cancel"}}

    await _cancel_mid_tools(agent, "t-cancel")

    # The cancel-path recovery closed the turn out in the generator's finally:
    # no pending tasks, and every dangling call got an honest synthetic result.
    snap = await agent.aget_state(cfg)
    assert snap.next == ()
    synth = _tool_messages(snap.values["messages"])
    assert len(synth) == 3
    assert all(m.content == _INTERRUPTED_TOOL_RESULT for m in synth)

    # The next user message must NOT replay the interrupted tool batch...
    before = len(_TOOL_SIDE_EFFECTS)
    events = await _run_turn(agent, "hello, new instruction", "t-cancel")
    assert len(_TOOL_SIDE_EFFECTS) == before
    # ...nor re-broadcast historical tool calls to the UI...
    assert all(ev.get("type") != "tool_call" for ev in events)
    # ...nor tell the model the calls "were cancelled" (the replay invitation).
    request = _MODEL_REQUESTS[-1]
    closed = _tool_messages(request)
    assert len(closed) == 3
    assert all(m.content == _INTERRUPTED_TOOL_RESULT for m in closed)
    assert not any("another message came in" in str(m.content) for m in closed)


async def test_hard_killed_thread_recovered_at_start_of_next_run():
    """A hard-killed thread (no cleanup ran) is repaired on the next run."""
    _script_tool_batch_then_ack()
    agent = _build_agent()
    cfg = {"configurable": {"thread_id": "t-kill"}}

    async def _no_recovery(agent, config, snapshot=None):
        return None

    with mock_patch.object(
        events_module, "_recover_interrupted_graph_state", _no_recovery
    ):
        await _cancel_mid_tools(agent, "t-kill")

    snap = await agent.aget_state(cfg)
    assert snap.next  # still stuck — the "killed" process never cleaned up
    assert not _tool_messages(snap.values["messages"])  # calls left dangling

    before = len(_TOOL_SIDE_EFFECTS)
    events = await _run_turn(agent, "hello, new instruction", "t-kill")

    # Start-of-run recovery kicked in: no replay, no UI flood, honest results.
    assert len(_TOOL_SIDE_EFFECTS) == before
    assert all(ev.get("type") != "tool_call" for ev in events)
    request = _MODEL_REQUESTS[-1]
    closed = _tool_messages(request)
    assert len(closed) == 3
    assert all(m.content == _INTERRUPTED_TOOL_RESULT for m in closed)
    assert not any("another message came in" in str(m.content) for m in closed)

    snap = await agent.aget_state(cfg)
    assert snap.next == ()


async def test_start_of_run_recovery_failure_refuses_the_run(monkeypatch):
    """If start-of-run recovery fails, the new run is refused, not replayed.

    A transient checkpoint failure (e.g. sqlite error) during recovery must
    not fall through to ``astream_events`` on a still-stuck thread — LangGraph
    could resume the interrupted tools node and replay its side effects.
    """
    _script_tool_batch_then_ack()
    agent = _build_agent()
    cfg = {"configurable": {"thread_id": "t-fail"}}

    async def _no_recovery(agent, config, snapshot=None):
        return None

    with mock_patch.object(
        events_module, "_recover_interrupted_graph_state", _no_recovery
    ):
        await _cancel_mid_tools(agent, "t-fail")
    snap = await agent.aget_state(cfg)
    assert snap.next  # still stuck — recovery has not run yet

    async def _failing_aupdate_state(_config, _values=None, as_node=None):
        raise RuntimeError("sqlite: disk I/O error")

    monkeypatch.setattr(agent, "aupdate_state", _failing_aupdate_state)

    before = len(_TOOL_SIDE_EFFECTS)
    events: list[dict[str, Any]] = []

    async def _collect_until_refused() -> None:
        async for ev in stream_agent_events(agent, "hello", "t-fail"):
            events.append(ev)

    with pytest.raises(RuntimeError, match="Could not repair the interrupted state"):
        await _collect_until_refused()

    # The turn was refused before any streaming: an error event for the UI,
    # no tool side effects, and the stuck checkpoint untouched (still stuck,
    # so a later retry can attempt recovery again).
    assert events
    assert events[0]["type"] == "error"
    assert "Could not repair the interrupted state" in events[0]["message"]
    assert len(_TOOL_SIDE_EFFECTS) == before
    snap = await agent.aget_state(cfg)
    assert snap.next
