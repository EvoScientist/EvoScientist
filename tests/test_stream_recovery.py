"""Targeted tests for interrupted-graph-state recovery.

These run against a real compiled LangGraph graph with a checkpointer,
so they actually verify the two claims the recovery rests on:

1. After a mid-run crash, ``aupdate_state(config, None, as_node=END)`` clears the
   stuck ``next`` tuple while preserving channel values.
2. A legitimate human-in-the-loop ``interrupt()`` (also a non-empty ``next``) is
   left intact, so a pending question is never silently discarded.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from EvoScientist.stream.events import _clear_interrupted_graph_state
from tests.conftest import run_async as _run


class _S(TypedDict):
    x: int


def _crashing_app():
    # Node 'b' crashes once, then succeeds — so a post-recovery run can complete
    # and prove the graph is genuinely unstuck (not replaying the dead step).
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
    def ask(state):
        interrupt({"question": "continue?"})
        return {"x": state["x"] + 1}

    g = StateGraph(_S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    return g.compile(checkpointer=InMemorySaver())


def test_recovery_clears_stuck_state_after_crash():
    app = _crashing_app()
    cfg = {"configurable": {"thread_id": "t1"}}
    try:
        app.invoke({"x": 0}, cfg)
    except Exception:
        pass  # LangGraph re-raises the node error (wrapped); we only care about state
    # The crash left the graph frozen at node 'b'.
    assert app.get_state(cfg).next == ("b",)

    _run(_clear_interrupted_graph_state(app, cfg))

    snap = app.get_state(cfg)
    assert snap.next == ()  # stuck state actually cleared
    assert snap.values == {"x": 1}  # channel values (history) preserved

    # And the graph is genuinely unstuck: a fresh run completes (a: +1, b: +100)
    # instead of replaying the dead node.
    assert app.invoke({"x": 41}, cfg)["x"] == 142


def test_recovery_preserves_pending_hitl_interrupt():
    app = _interrupting_app()
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"x": 0}, cfg)  # parks at interrupt()
    before = app.get_state(cfg)
    assert before.next == ("ask",)
    assert before.interrupts

    _run(_clear_interrupted_graph_state(app, cfg))

    after = app.get_state(cfg)
    assert after.next == ("ask",)  # interrupt left intact, still resumable
    assert after.interrupts
