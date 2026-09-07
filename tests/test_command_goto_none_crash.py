"""Tests for the session-emptying ``Command(goto=None)`` crash.

Root cause
----------
When a client sends a ``command`` parameter (e.g. ``{"resume": {...}}``) in
the run request body **without** a ``goto`` field, ``langgraph_api.command.map_cmd``
produces ``Command(goto=None, resume=...)``.  LangGraph 1.2.9's
``_control_branch`` (``langgraph/graph/state.py``) then tries to iterate
over ``None``::

    goto_targets = (
        [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
    )
    for go in goto_targets:        # ← TypeError: 'NoneType' object is not iterable

The crash happens at the ``__start__`` pseudo-node — before any model call,
before any middleware.  The thread status is set to ``"error"`` and the
checkpoint ``values`` are reset to defaults (``messages: []``, ``files: {}``,
``async_tasks: {}``), corrupting the previous conversation state.  Every UI
(WebUI, TUI, CLI) renders a blank session.

These tests verify two properties:

1. **No crash**: ``Command(goto=None)`` is handled gracefully (treated as
   "no goto targets"), not as a ``TypeError``.
2. **Crash path coverage**: when a thread's checkpoint has been deleted
   (mimicking a cancelled/rolled-back run), a subsequent ``goto=None``
   command reaches ``__start__`` and ``_control_branch`` — the exact crash
   path. On ``main`` this crashes; with the patch it completes cleanly and
   any update payload is applied.

Matches upstream issue langchain-ai/langgraph#5656.
"""

from __future__ import annotations

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
    graph input.  The fix is to emit ``goto=[]`` so that the "no goto targets"
    path is taken cleanly.
    """

    def test_resume_only_command_goto_is_empty_list(self):
        """A resume command without goto must produce ``goto=[]``, not ``None``."""
        from langgraph_api.command import map_cmd

        result = map_cmd({"resume": {"status": "answered", "answers": ["yes"]}})

        assert result.goto == [], (
            "map_cmd produces Command(goto=None) for resume-only commands, "
            "which crashes _control_branch when used as graph input"
        )

    def test_update_only_command_goto_is_empty_list(self):
        """An update command without goto must produce ``goto=[]``, not ``None``."""
        from langgraph_api.command import map_cmd

        result = map_cmd({"update": {"messages": [{"role": "user", "content": "hi"}]}})

        assert result.goto == [], (
            "map_cmd produces Command(goto=None) for update-only commands, "
            "which crashes _control_branch when used as graph input"
        )


# ---------------------------------------------------------------------------
# 2. Command(goto=None) as graph input should not crash
# ---------------------------------------------------------------------------


class TestCommandGotoNoneAsInput:
    """Feeding ``Command(goto=None)`` as graph input should be handled
    gracefully, not crash with ``TypeError``.
    """

    async def test_resume_command_does_not_crash(self, minimal_graph, thread_config):
        """``Command(goto=None, resume=...)`` — the exact shape ``map_cmd``
        produces for a resume request — should not raise."""
        cmd = Command(goto=None, resume={"answer": "yes"})
        # Should complete without TypeError
        await minimal_graph.ainvoke(cmd, config=thread_config)

    async def test_update_command_does_not_crash(self, minimal_graph, thread_config):
        """``Command(goto=None, update=...)`` should also not raise."""
        cmd = Command(
            goto=None,
            update={"messages": [{"role": "user", "content": "hi"}]},
        )
        await minimal_graph.ainvoke(cmd, config=thread_config)

    async def test_normal_dict_input_still_works(self, minimal_graph, thread_config):
        """Regression guard: regular dict input is unaffected."""
        result = await minimal_graph.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            config=thread_config,
        )
        assert len(result["messages"]) == 1

    async def test_command_with_goto_still_works(self, minimal_graph, thread_config):
        """Regression guard: a Command with a valid goto is unaffected."""
        cmd = Command(goto="model", update={})
        result = await minimal_graph.ainvoke(cmd, config=thread_config)
        assert len(result["messages"]) == 0, (
            "Command with valid goto should complete and return state"
        )


# ---------------------------------------------------------------------------
# 3. Crash path via __start__: goto=None command after checkpoint deletion
# ---------------------------------------------------------------------------


class TestGotoNoneCrashPath:
    """Tests that exercise the actual crash path: ``_control_branch`` via the
    ``__start__`` pseudo-node.

    The scenario mimics the production trigger: a thread that previously had
    messages has its checkpoint deleted (e.g. by a cancelled/rolled-back run),
    then a resume or update command with ``goto=None`` fires against the same
    thread. With no checkpoint, ``is_resuming`` is ``False``, so LangGraph
    takes the ``map_input`` path, writes the ``Command`` to the ``START``
    channel, and ``__start__`` fires — calling ``_control_branch`` with
    ``Command(goto=None)``.

    On ``main`` (unpatched), this crashes with ``TypeError: 'NoneType' object
    is not iterable`` and the thread is left in an error state with default
    values. With the patch, ``goto=None`` is normalized to ``[]`` and the
    command completes cleanly.
    """

    async def test_resume_after_checkpoint_deletion_does_not_crash(
        self, minimal_graph, thread_config
    ):
        """Seed messages, delete the checkpoint (mimicking rollback), then
        resume — reaches ``__start__`` with ``goto=None``, crashes on main."""
        # Seed: create a checkpoint with one message
        await minimal_graph.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            config=thread_config,
        )
        # Delete the checkpoint — mimics a cancelled/rolled-back run
        minimal_graph.checkpointer.delete_thread(
            thread_config["configurable"]["thread_id"]
        )
        # Act: resume command with goto=None — __start__ fires, _control_branch
        # is called with Command(goto=None). Crashes on main, completes with patch.
        cmd = Command(goto=None, resume={"answer": "yes"})
        await minimal_graph.ainvoke(cmd, config=thread_config)

    async def test_update_after_checkpoint_deletion_applies_update(
        self, minimal_graph, thread_config
    ):
        """Seed messages, delete the checkpoint (mimicking rollback), then
        send an update command — the update must be applied despite going
        through ``__start__`` with ``goto=None``."""
        # Seed: create a checkpoint with one message
        await minimal_graph.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            config=thread_config,
        )
        # Delete the checkpoint — mimics a cancelled/rolled-back run
        minimal_graph.checkpointer.delete_thread(
            thread_config["configurable"]["thread_id"]
        )
        # Act: update command with goto=None — __start__ fires, _control_branch
        # is called. Crashes on main, applies the update with patch.
        cmd = Command(
            goto=None,
            update={"messages": [{"role": "user", "content": "world"}]},
        )
        await minimal_graph.ainvoke(cmd, config=thread_config)
        # Assert: the update was applied (not lost to a crash)
        state = await minimal_graph.aget_state(thread_config)
        assert len(state.values.get("messages", [])) >= 1, (
            "Update was lost: no messages after goto=None command via __start__"
        )
        assert state.values["messages"][0].content == "world"


# ---------------------------------------------------------------------------
# 4. _control_branch directly: END routing must not produce a branch
# ---------------------------------------------------------------------------


class TestControlBranchEndRouting:
    """Regression guard: ``Command(goto=END)`` must not produce a spurious
    ``branch:to:__end__`` channel.  The original ``_control_branch`` skips
    branching for ``END``; the patched version must preserve that behavior.
    """

    def test_goto_end_is_not_a_branch_target(self):
        """``_control_branch(Command(goto=END))`` must return ``[]`` — END
        is a terminal sentinel, not a node to branch to."""
        from langgraph.graph.state import _control_branch

        assert _control_branch(Command(goto=END)) == []

    def test_goto_end_in_list_is_not_a_branch_target(self):
        """Same check with ``goto=[END]`` (list form)."""
        from langgraph.graph.state import _control_branch

        assert _control_branch(Command(goto=[END])) == []
