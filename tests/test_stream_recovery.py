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
4. A call that finished before the cancel keeps its real result; only the
   unfinished calls get the synthetic one.
5. A SIGKILL after every tool wrote, before the super-step checkpoint, still
   recovers (``next`` empty, ``tasks`` present, empty patch).
"""

import asyncio
from typing import Any, TypedDict

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from EvoScientist.middleware.tool_history_repair import ToolHistoryRepairMiddleware
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
    done = app.get_state(cfg)
    assert done.next == ()
    assert done.tasks == ()


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
_slow_tool_gate: asyncio.Event | None = None
"""Unset event tests may install so matching tool calls park (deterministically
cancellable mid-batch) instead of finishing on fixed delays."""
_slow_tool_gate_paths: frozenset[str] | None = None
"""Paths the gate applies to; ``None`` gates every call."""


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
    """Edit a file; the side effect lands immediately, like edit_file.

    Tests may install ``_slow_tool_gate`` (an unset ``asyncio.Event``) to park
    matching calls after their side effect lands — cancellation then always
    lands mid-batch instead of racing fixed delays. Ungated calls finish on
    staggered delays (``a.py`` fast, the rest slow).
    """
    _TOOL_SIDE_EFFECTS.append(path)
    if _slow_tool_gate is not None and (
        _slow_tool_gate_paths is None or path in _slow_tool_gate_paths
    ):
        await _slow_tool_gate.wait()
    else:
        await asyncio.sleep(0.3 if path == "a.py" else 2.0)
    return f"edited {path}"


def _tool_call(id_: str, path: str) -> dict[str, Any]:
    """Build a tool-call entry for ``_slow_edit`` on ``path``."""
    return {
        "id": id_,
        "name": _slow_edit.__name__,
        "args": {"path": path},
        "type": "tool_call",
    }


def _build_agent(checkpointer: Any | None = None):
    """Build a deep agent mirroring the real EvoScientist middleware stack."""
    return create_deep_agent(
        model=_ScriptedModel(),
        tools=[_slow_edit],
        middleware=[ToolHistoryRepairMiddleware()],
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


def _gate_slow_tools(paths: frozenset[str] | None = None) -> None:
    """Hold matching tool calls (all of them when ``paths`` is None) in place.

    Gated calls park on an unset event after landing their side effect, so
    they can never complete before the test cancels the turn.
    """
    global _slow_tool_gate, _slow_tool_gate_paths
    _slow_tool_gate = asyncio.Event()
    _slow_tool_gate_paths = paths


def _ungate_slow_tools() -> None:
    """Clear the test gate installed by ``_gate_slow_tools``."""
    global _slow_tool_gate, _slow_tool_gate_paths
    _slow_tool_gate = None
    _slow_tool_gate_paths = None


async def _await_tool_result(
    agent: Any, cfg: dict[str, Any], tool_call_id: str, timeout: float = 10.0
) -> None:
    """Wait until a finished call's result is visible on the stuck checkpoint.

    ``aget_state`` folds pending writes into ``values``, so the ToolMessage of
    a finished call appears in the snapshot's messages before its super-step
    commits. That visibility is the deterministic signal that a
    partially-finished batch is actually in place — no fixed sleeps.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        snap = await agent.aget_state(cfg)
        values = (getattr(snap, "values", None) or {}).get("messages") or []
        if any(
            isinstance(m, ToolMessage) and m.tool_call_id == tool_call_id
            for m in values
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"result for {tool_call_id} never reached the checkpoint")


def _script_tool_batch_then_ack() -> None:
    """Reset shared script/log state: 3-call tool batch, then an ack."""
    _TOOL_SIDE_EFFECTS.clear()
    _MODEL_REQUESTS.clear()
    _SCRIPT.clear()
    _script_idx["i"] = 0
    _ungate_slow_tools()
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


class _KillAfterToolWrites(InMemorySaver):
    """SIGKILL between ``put_writes`` and ``put_checkpoint``.

    Drops the super-step commit once three tool-task writes are pending, then
    freezes further durable writes so the dying process cannot store a later
    checkpoint. ``thaw()`` re-enables writes for the next process's recovery.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase = "live"
        self.killed = asyncio.Event()

    def thaw(self) -> None:
        """Allow the next process to persist recovery writes."""
        self._phase = "thawed"

    def _frozen_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
                "checkpoint_id": config["configurable"].get("checkpoint_id"),
            }
        }

    def put(self, config, checkpoint, metadata, new_versions):
        """Drop the tools super-step commit, then freeze until ``thaw()``."""
        if self._phase == "frozen":
            return self._frozen_config(config)
        if self._phase == "live":
            parent_id = config["configurable"].get("checkpoint_id")
            if parent_id:
                thread_id = config["configurable"]["thread_id"]
                checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
                pending = self.writes.get((thread_id, checkpoint_ns, parent_id)) or {}
                if len({entry[0] for entry in pending.values()}) >= 3:
                    self._phase = "frozen"
                    self.killed.set()
                    return self._frozen_config(config)
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes, task_id, task_path=""):
        """Ignore writes from the dying process after the dropped commit."""
        if self._phase == "frozen":
            return None
        return super().put_writes(config, writes, task_id, task_path)


def _tool_messages(messages: Any) -> list[ToolMessage]:
    """Extract the ToolMessages from a message sequence."""
    return [m for m in messages if isinstance(m, ToolMessage)]


async def test_cancel_mid_tools_closes_dangling_calls_and_next_turn_is_clean():
    """Ctrl+C mid-tools must close the batch honestly; next turn stays clean."""
    _script_tool_batch_then_ack()
    agent = _build_agent()
    cfg = {"configurable": {"thread_id": "t-cancel"}}

    _gate_slow_tools()  # park every call: the cancel always lands mid-batch
    try:
        await _cancel_mid_tools(agent, "t-cancel")
    finally:
        _ungate_slow_tools()

    # Cancellation leaves the checkpoint parked; the next run repairs it.
    assert (await agent.aget_state(cfg)).next

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
    """A hard-killed process is repaired when a new agent resumes the thread.

    Models a process restart: cancel with agent A, then build agent B over the
    same saver and run the next turn through B. Recovery is not invoked at
    cancel time; start-of-run on B's first ``stream_agent_events`` does it.
    """
    _script_tool_batch_then_ack()
    saver = InMemorySaver()
    killed = _build_agent(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-kill"}}

    _gate_slow_tools()  # park every call: the "kill" always lands mid-batch
    try:
        await _cancel_mid_tools(killed, "t-kill")
    finally:
        _ungate_slow_tools()

    snap = await killed.aget_state(cfg)
    assert snap.next  # still stuck — the "killed" process never cleaned up
    assert not _tool_messages(snap.values["messages"])  # calls left dangling

    resumed = _build_agent(checkpointer=saver)
    before = len(_TOOL_SIDE_EFFECTS)
    events = await _run_turn(resumed, "hello, new instruction", "t-kill")

    # Start-of-run recovery kicked in: no replay, no UI flood, honest results.
    assert len(_TOOL_SIDE_EFFECTS) == before
    assert all(ev.get("type") != "tool_call" for ev in events)
    request = _MODEL_REQUESTS[-1]
    closed = _tool_messages(request)
    assert len(closed) == 3
    assert all(m.content == _INTERRUPTED_TOOL_RESULT for m in closed)
    assert not any("another message came in" in str(m.content) for m in closed)

    snap = await resumed.aget_state(cfg)
    assert snap.next == ()


async def test_partially_finished_batch_keeps_real_result_and_closes_the_rest():
    """A finished call keeps its real result; only the rest get synthetic.

    Regression for the patch-before-clear ordering: ``aupdate_state(...,
    as_node="tools")`` reuses a finished task's id, and the checkpointer keeps
    the first write per ``(task_id, idx)``, so the synthetic patch was silently
    dropped and the dangling calls came back as "was cancelled" on the next
    turn. Clearing first commits the finished call's pending write; the patch
    then only covers the unfinished calls.
    """
    _script_tool_batch_then_ack()
    agent = _build_agent()
    cfg = {"configurable": {"thread_id": "t-partial"}}

    _gate_slow_tools(frozenset({"b.py", "c.py"}))  # a.py finishes; the rest hold
    try:
        task = asyncio.create_task(_run_turn(agent, "apply the edits", "t-partial"))
        while len(_TOOL_SIDE_EFFECTS) < 3:
            await asyncio.sleep(0.05)
        # Deterministic partial batch: a.py's real result has reached the
        # stuck checkpoint as a pending write, and b.py / c.py are parked.
        await _await_tool_result(agent, cfg, "call_1")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        _ungate_slow_tools()

    events = await _run_turn(agent, "hello, new instruction", "t-partial")

    assert all(ev.get("type") != "tool_call" for ev in events)
    seen = {m.tool_call_id: m.content for m in _tool_messages(_MODEL_REQUESTS[-1])}
    assert seen == {
        "call_1": "edited a.py",
        "call_2": _INTERRUPTED_TOOL_RESULT,
        "call_3": _INTERRUPTED_TOOL_RESULT,
    }
    assert (await agent.aget_state(cfg)).next == ()


async def test_uncommitted_finished_batch_is_recovered():
    """SIGKILL after every tool wrote, before the super-step checkpoint.

    ``next`` is empty (every task has a write) while ``tasks`` still holds
    the tools. Recovery must still commit those writes — and must not
    warn-fail just because the first END schedules ``model``.
    """
    _script_tool_batch_then_ack()
    saver = _KillAfterToolWrites()
    agent = _build_agent(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-uncommitted"}}

    # Let every call finish immediately so all three pending writes land.
    _gate_slow_tools()
    assert _slow_tool_gate is not None
    _slow_tool_gate.set()
    try:
        task = asyncio.create_task(_run_turn(agent, "apply the edits", "t-uncommitted"))
        await asyncio.wait_for(saver.killed.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        _ungate_slow_tools()

    saver.thaw()
    snap = await agent.aget_state(cfg)
    assert snap.next == ()
    assert snap.tasks
    assert await _recover_interrupted_graph_state(agent, cfg, snapshot=snap) is True
    recovered = await agent.aget_state(cfg)
    assert recovered.next == ()
    assert recovered.tasks == ()

    before = len(_TOOL_SIDE_EFFECTS)
    events = await _run_turn(agent, "hello, new instruction", "t-uncommitted")
    assert len(_TOOL_SIDE_EFFECTS) == before
    assert all(ev.get("type") != "tool_call" for ev in events)
    seen = {m.tool_call_id: m.content for m in _tool_messages(_MODEL_REQUESTS[-1])}
    assert seen == {
        "call_1": "edited a.py",
        "call_2": "edited b.py",
        "call_3": "edited c.py",
    }
    assert (await agent.aget_state(cfg)).next == ()
