"""Tests for the session-emptying ``Command(goto=None)`` crash.

Root cause
----------
When the WebUI sends a ``command`` parameter (e.g. ``{"resume": {...}}``) in
the run request body **without** a ``goto`` field, ``langgraph_api.command.map_cmd``
produces ``Command(goto=None, resume=...)``.  LangGraph 1.2.9's
``_control_branch`` (``langgraph/graph/state.py:1754``) then tries to iterate
over ``None``::

    goto_targets = (
        [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
    )
    for go in goto_targets:        # ← TypeError: 'NoneType' object is not iterable

The crash happens at the ``__start__`` pseudo-node — before any model call,
before any middleware — so the thread's persisted checkpoint is **not**
corrupted, but the thread status is set to ``"error"`` with empty ``values``
and the WebUI renders a blank session.

These tests assert the **desired** behaviour: ``Command(goto=None)`` should be
handled gracefully (treated as "no goto targets"), not crash.  They **fail**
while the bug is live in langgraph 1.2.9 and **pass** once the fix is applied
(either upstream in ``_control_branch`` or locally in ``map_cmd``).

See ``notes/glm-5.2/emptied-session/root-cause.md`` for the full analysis.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

# Apply the upstream-workaround patches (map_cmd + _control_branch) so the
# tests verify the post-fix behaviour.
from EvoScientist.llm import patches  # noqa: F401  (side-effect)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _State(TypedDict):
    messages: Annotated[list, add_messages]


@pytest.fixture
def minimal_graph():
    """A minimal compiled StateGraph with one node and a MemorySaver checkpointer.

    The checkpointer is required for ``Command(resume=...)`` — without it
    LangGraph raises ``RuntimeError: Cannot use Command(resume=...) without
    checkpointer`` before reaching the crash site.
    """
    builder = StateGraph(_State)
    builder.add_node("model", lambda s: s)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    return builder.compile(checkpointer=MemorySaver())


@pytest.fixture
def thread_config():
    return {"configurable": {"thread_id": "test-thread"}}


# ---------------------------------------------------------------------------
# 1. map_cmd should not produce goto=None
# ---------------------------------------------------------------------------


class TestMapCmdGotoNone:
    """``map_cmd`` must not emit ``Command(goto=None)``.

    A ``Command`` with ``goto=None`` crashes ``_control_branch`` when used as
    graph input.  The fix is to emit ``goto=[]`` (or omit the kwarg) so that
    the "no goto targets" path is taken cleanly.

    These tests fail while the bug is live (``goto`` is ``None``) and pass
    after ``map_cmd`` is patched.
    """

    def test_resume_only_command_goto_is_not_none(self):
        """A resume command without goto must not produce ``goto=None``."""
        from langgraph_api.command import map_cmd

        result = map_cmd({"resume": {"status": "answered", "answers": ["yes"]}})

        assert result.goto is not None, (
            "map_cmd produces Command(goto=None) for resume-only commands, "
            "which crashes _control_branch when used as graph input"
        )

    def test_update_only_command_goto_is_not_none(self):
        """An update command without goto must not produce ``goto=None``."""
        from langgraph_api.command import map_cmd

        result = map_cmd({"update": {"messages": [{"role": "user", "content": "hi"}]}})

        assert result.goto is not None, (
            "map_cmd produces Command(goto=None) for update-only commands, "
            "which crashes _control_branch when used as graph input"
        )


# ---------------------------------------------------------------------------
# 2. Command(goto=None) as graph input should not crash
# ---------------------------------------------------------------------------


class TestCommandGotoNoneAsInput:
    """Feeding ``Command(goto=None)`` as graph input should be handled
    gracefully, not crash with ``TypeError``.

    These tests fail while the bug is live in ``_control_branch``
    (langgraph 1.2.9) and pass once the fix — ``command.goto or []`` — is
    applied.
    """

    def test_resume_command_does_not_crash(self, minimal_graph, thread_config):
        """``Command(goto=None, resume=...)`` — the exact shape ``map_cmd``
        produces for a WebUI resume request — should not raise."""
        cmd = Command(goto=None, resume={"answer": "yes"})
        # Should complete without TypeError
        asyncio.run(minimal_graph.ainvoke(cmd, config=thread_config))

    def test_update_command_does_not_crash(self, minimal_graph, thread_config):
        """``Command(goto=None, update=...)`` should also not raise."""
        cmd = Command(
            goto=None,
            update={"messages": [{"role": "user", "content": "hi"}]},
        )
        asyncio.run(minimal_graph.ainvoke(cmd, config=thread_config))

    def test_normal_dict_input_still_works(self, minimal_graph, thread_config):
        """Regression guard: regular dict input is unaffected."""
        result = asyncio.run(
            minimal_graph.ainvoke(
                {"messages": [{"role": "user", "content": "hello"}]},
                config=thread_config,
            )
        )
        assert len(result["messages"]) == 1

    def test_command_with_goto_still_works(self, minimal_graph, thread_config):
        """Regression guard: a Command with a valid goto is unaffected."""
        cmd = Command(goto="model", update={})
        asyncio.run(minimal_graph.ainvoke(cmd, config=thread_config))
