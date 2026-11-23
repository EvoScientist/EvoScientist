"""Regression tests for the code_interpreter PTC allowlist and the
``EvoCodeInterpreterMiddleware`` subclass shape.

langchain-quickjs >=0.3 reserves the ``task`` sub-agent dispatch tool as the
top-level REPL global and raises ``ValueError`` if ``task`` appears in the
``ptc`` allowlist. EvoScientist must therefore keep ``task`` out of the
allowlist (``task()`` stays reachable as the REPL global, with responseSchema).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from EvoScientist.middleware.code_interpreter import (
    _DEFAULT_PTC_ALLOWLIST,
    create_code_interpreter_middleware,
)


def test_task_excluded_from_ptc_allowlist():
    # Exposing `task` via ptc raises on quickjs >=0.3 — it is the REPL global.
    assert "task" not in _DEFAULT_PTC_ALLOWLIST


def test_async_tools_that_track_tasks_stay_out_of_allowlist():
    # Starting, updating or cancelling a task from code is untested, so those
    # stay out; reading tracked tasks is allowed.
    for name in ("start_async_task", "update_async_task", "cancel_async_task"):
        assert name not in _DEFAULT_PTC_ALLOWLIST
    for name in ("check_async_task", "list_async_tasks"):
        assert name in _DEFAULT_PTC_ALLOWLIST


def test_filter_tools_for_ptc_accepts_default_allowlist():
    # End-to-end guard: the live quickjs filter must accept our allowlist even
    # when a `task` tool is present in the agent toolset.
    _ptc = pytest.importorskip("langchain_quickjs._ptc")
    from langchain_core.tools import tool

    @tool
    def task(description: str) -> str:
        """dummy sub-agent dispatch tool"""
        return "ok"

    _ptc.filter_tools_for_ptc(
        [task], _DEFAULT_PTC_ALLOWLIST, self_tool_name="code_interpreter"
    )


def test_create_code_interpreter_middleware_builds():
    assert create_code_interpreter_middleware() is not None


def test_after_agent_without_a_slot_returns_nothing():
    # A run that ends in its first start hook never ran before_agent.
    mw = create_code_interpreter_middleware()
    assert mw.after_agent({"messages": []}, runtime=None) is None


async def test_aafter_agent_without_a_slot_returns_nothing():
    mw = create_code_interpreter_middleware()
    assert await mw.aafter_agent({"messages": []}, runtime=None) is None


def test_middleware_uses_thread_mode():
    """Upstream ``mode="thread"`` (the default) preserves cross-turn REPL
    state as ``langchain-ai/deepagents#3064`` shipped it. The wire-cost
    bloat that motivated the earlier ``mode="turn"`` regression guard is
    fixed at the API serialization layer (``EvoFilteredGraph`` in
    ``EvoScientist/langgraph_dev/main_graph.py``), not by revoking the
    persistence feature.
    """
    mw = create_code_interpreter_middleware()
    assert mw._mode == "thread"


def test_evo_filtered_graph_strips_private_snapshot_field():
    """The ``StateSnapshot`` returned by ``EvoScientist_agent.get_state`` must
    not contain ``_quickjs_snapshot_payload`` in either ``values`` (the
    materialized channel payload) or ``metadata['writes']`` (the raw write
    records surfaced by ``get_state_history``).
    """
    from EvoScientist.langgraph_dev.main_graph import _EvoFilteredGraph, _strip_private

    snap = MagicMock()
    snap.values = {
        "messages": ["m1"],
        "_quickjs_snapshot_payload": b"x" * 100,
        "skills_metadata": [],
    }
    snap.metadata = {
        "source": "loop",
        "step": 42,
        "writes": {
            "CodeInterpreterMiddleware.after_agent": {
                "_quickjs_snapshot_payload": ("snap", b"y" * 1_400_000),
                "messages": [],
            },
            "model": {"messages": ["m1"]},
        },
        "parents": {},
    }
    _strip_private(snap)
    snap._replace.assert_called_once()
    kwargs = snap._replace.call_args.kwargs
    assert "_quickjs_snapshot_payload" not in kwargs["values"]
    assert "messages" in kwargs["values"]
    assert "skills_metadata" in kwargs["values"]
    scrubbed_writes = kwargs["metadata"]["writes"]
    assert (
        "_quickjs_snapshot_payload"
        not in scrubbed_writes["CodeInterpreterMiddleware.after_agent"]
    )
    assert "messages" in scrubbed_writes["CodeInterpreterMiddleware.after_agent"]
    assert scrubbed_writes["model"] == {"messages": ["m1"]}
    # Non-writes metadata keys are preserved.
    assert kwargs["metadata"]["source"] == "loop"
    assert kwargs["metadata"]["step"] == 42
    # Sanity: the class exists and inherits from CompiledStateGraph.
    from langgraph.graph.state import CompiledStateGraph

    assert issubclass(_EvoFilteredGraph, CompiledStateGraph)


def test_strip_private_handles_missing_metadata_writes():
    """``metadata['writes']`` can be missing or ``None`` on some snapshots
    (e.g. initial state). The filter must not crash and must still strip
    values.
    """
    from EvoScientist.langgraph_dev.main_graph import _strip_private

    snap = MagicMock()
    snap.values = {"_quickjs_snapshot_payload": b"x", "messages": []}
    snap.metadata = {"source": "input", "step": -1, "writes": None}
    snap.tasks = ()
    _strip_private(snap)
    kwargs = snap._replace.call_args.kwargs
    assert "_quickjs_snapshot_payload" not in kwargs["values"]
    # writes was None, metadata passes through unchanged.
    assert kwargs["metadata"]["writes"] is None


def test_strip_private_scrubs_task_result_snapshot_blob():
    """``tasks[*].result`` is where ``after_agent``'s return dict lands.
    When the middleware snapshots, ``result`` carries
    ``{"_quickjs_snapshot_payload": ("snap", ~1.4 MB bytes)}``. Verified
    on live history: this is the dominant per-response leak, larger than
    ``values`` and ``metadata.writes`` combined for anchor checkpoints.
    """
    from EvoScientist.langgraph_dev.main_graph import _strip_private

    class FakeTask:
        def __init__(self, id_, result):
            self.id = id_
            self.name = "CodeInterpreterMiddleware.after_agent"
            self.result = result

        def _replace(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            return self

    leaking_task = FakeTask(
        "t1", {"_quickjs_snapshot_payload": ("snap", b"z" * 1_400_000), "messages": []}
    )
    clean_task = FakeTask("t2", {"messages": ["hi"]})
    snap = MagicMock()
    snap.values = {}
    snap.metadata = {"source": "loop", "step": 5}
    snap.tasks = (leaking_task, clean_task)
    _strip_private(snap)
    kwargs = snap._replace.call_args.kwargs
    tasks_after = kwargs["tasks"]
    assert "_quickjs_snapshot_payload" not in tasks_after[0].result
    assert "messages" in tasks_after[0].result
    # Clean task is passed through untouched.
    assert tasks_after[1] is clean_task


def test_every_served_graph_uses_filtered_graph_class(
    monkeypatch, tmp_path, tiny_graph
):
    """Every graph the server builds (main + all subagents) gets the
    ``__class__`` swap. Iterates the registered builders, so a new graph needs
    no test update.

    Subagents get ``create_code_interpreter_middleware`` unconditionally
    (``EvoScientist.py:_build_middleware_stack``), so they can touch the
    QuickJS REPL and write ``_quickjs_snapshot_payload`` on their own
    checkpoint namespace, read through their own ``/threads/{id}/state``.
    """
    from EvoScientist.langgraph_dev import factories
    from EvoScientist.langgraph_dev.main_graph import _EvoFilteredGraph
    from EvoScientist.paths import SessionDirs, Workspace

    for graph_id in list(factories.BUILDERS):
        monkeypatch.setitem(factories.BUILDERS, graph_id, tiny_graph)
    graphs = factories.GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    for graph_id in factories.BUILDERS:
        assert isinstance(graphs.get(graph_id, dirs), _EvoFilteredGraph), graph_id


def test_strip_private_recurses_into_nested_subgraph_state():
    """When ``subgraphs=True``, ``PregelTask.state`` holds a nested
    ``StateSnapshot`` for the subgraph. Its ``values`` (and its own nested
    tasks) can carry ``_quickjs_snapshot_payload`` just like the parent.
    Recursion covers the compound leak path CodeRabbit flagged.
    """
    from langgraph.types import StateSnapshot

    from EvoScientist.langgraph_dev.main_graph import _strip_private

    nested_snap = StateSnapshot(
        values={"_quickjs_snapshot_payload": b"n" * 1_400_000, "messages": []},
        next=(),
        config={},
        metadata={"source": "loop", "step": 3},
        created_at="2026-07-01T12:00:00Z",
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    class FakeTask:
        def __init__(self, state):
            self.id = "sub-1"
            self.name = "subgraph"
            self.result = None
            self.state = state

        def _replace(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            return self

    task_with_nested = FakeTask(nested_snap)
    task_with_config_state = FakeTask({"configurable": {"thread_id": "t"}})
    snap = MagicMock()
    snap.values = {}
    snap.metadata = {"source": "loop", "step": 5}
    snap.tasks = (task_with_nested, task_with_config_state)
    _strip_private(snap)
    kwargs = snap._replace.call_args.kwargs
    tasks_after = kwargs["tasks"]
    # Nested StateSnapshot got recursively scrubbed.
    assert "_quickjs_snapshot_payload" not in tasks_after[0].state.values
    assert "messages" in tasks_after[0].state.values
    # A dict (RunnableConfig-shaped) state passes through unchanged — we only
    # recurse into ``StateSnapshot`` instances.
    assert tasks_after[1].state == {"configurable": {"thread_id": "t"}}


def test_strip_private_scrubs_delta_counters():
    """``metadata['counters_since_delta_snapshot']`` is a small
    ``{channel: [count, superstep]}`` bookkeeping map. Not a size problem,
    but leaks the channel name — strip for consistency with the private
    annotation.
    """
    from EvoScientist.langgraph_dev.main_graph import _strip_private

    snap = MagicMock()
    snap.values = {}
    snap.metadata = {
        "source": "loop",
        "step": 5,
        "counters_since_delta_snapshot": {
            "_quickjs_snapshot_payload": [1, 14],
            "messages": [3, 14],
        },
    }
    snap.tasks = ()
    _strip_private(snap)
    kwargs = snap._replace.call_args.kwargs
    counters = kwargs["metadata"]["counters_since_delta_snapshot"]
    assert "_quickjs_snapshot_payload" not in counters
    assert "messages" in counters
