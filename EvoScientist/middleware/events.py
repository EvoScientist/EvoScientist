"""Typed middleware → frontend event sink.

This module lives deliberately inside ``middleware/`` so the dependency
direction is always **frontends → middleware**, never the reverse. Middleware
reports structured facts about what happened during a model call; a frontend
supplies a sink implementation that owns its own display state and renders (or
ignores) those facts.

Two families of events are evidenced today and modelled here:

* **Tool selection** — the adaptive ``LLMToolSelectorMiddleware`` wrapper
  reports when a selection LLM call starts, which tools survived filtering,
  and when it ends.
* **Model fallback** — the fallback middleware reports each transition from a
  failing model to the next candidate in the chain.

Events are **structured, not pre-formatted**. Formatting (Rich styles, TUI
widgets, etc.) belongs in the sink implementation, where the frontend owns it.

Threading / blocking contract
-----------------------------
Sink methods may be called **from any thread**: synchronous middleware hooks
run on LangChain worker threads, async hooks on whichever loop runs the graph.
A sink implementation therefore MUST be:

* **thread-safe** — any state it mutates is touched under its own lock, and
* **non-blocking** — it marshals to its UI itself (Textual:
  ``call_from_thread`` / ``post_message``; Rich: the console's internal lock)
  and returns promptly.

A sink that blocks stalls the model call that emitted the event — the emitting
worker thread is held until the sink returns. Nothing in the framework isolates
a slow sink from the run; see ``tests/test_middleware_event_sink.py`` for the
contract test that pins this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MiddlewareEventSink(Protocol):
    """Structured display events emitted by middleware hooks.

    Implementations are supplied by frontends and injected at the agent
    composition root (see ``EvoScientist.EvoScientist``). The default for
    headless, gateway, and subagent stacks is :class:`NoOpSink`.

    All methods must honour the module-level threading/blocking contract:
    callable from any thread, thread-safe, and non-blocking.
    """

    def on_tool_selection_started(self, total_tools: int) -> None:
        """A tool-selection LLM call has begun over ``total_tools`` tools."""
        ...

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        """The selection kept ``selected`` out of ``total_tools`` tools."""
        ...

    def on_tool_selection_ended(self) -> None:
        """The tool-selection LLM call has finished (or failed)."""
        ...

    def on_model_fallback(self, from_model: str, to_model: str, reason: str) -> None:
        """A model call fell back from ``from_model`` to ``to_model``."""
        ...


class ToolSelectionView(Protocol):
    """Read side of the tool-selection state the stream suppressor consumes.

    A frontend sink both *records* tool-selection facts (via the
    :class:`MiddlewareEventSink` write side) and *exposes* them here so
    ``stream/tool_selection.py`` can decide whether to suppress selector chatter
    and when to surface the selection widget. Ownership lives in the frontend;
    the stream layer only reads. :class:`NoOpSink` implements this as "never
    active, nothing pending" so headless stacks render no widget.
    """

    @property
    def tool_selection_active(self) -> bool:
        """Whether a selection LLM call is currently in flight."""
        ...

    def tool_selection_pending(self) -> bool:
        """Whether an unconsumed selection result is waiting to render."""
        ...

    def consume_tool_selection(self) -> tuple[bool, list[str] | None]:
        """Consume the pending selection once, applying dedup-vs-last-emitted.

        Returns ``(had_pending, render)``:

        * ``had_pending`` — a pending selection existed and was consumed.
        * ``render`` — the tool list to display, or ``None`` when the selection
          should not render (it kept every tool, or duplicates the last one
          shown). ``None`` with ``had_pending=True`` still counts as consumed.
        """
        ...


class NoOpSink:
    """Default sink: drops every event and never renders a selection.

    Used for headless / gateway / deploy paths and for every subagent stack,
    where there is no frontend to render middleware events. Trivially
    thread-safe and non-blocking. Implements both the write
    (:class:`MiddlewareEventSink`) and read (:class:`ToolSelectionView`) sides.
    """

    __slots__ = ()

    def on_tool_selection_started(self, total_tools: int) -> None:
        pass

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        pass

    def on_tool_selection_ended(self) -> None:
        pass

    def on_model_fallback(self, from_model: str, to_model: str, reason: str) -> None:
        pass

    # --- ToolSelectionView (read side) -----------------------------------
    @property
    def tool_selection_active(self) -> bool:
        return False

    def tool_selection_pending(self) -> bool:
        return False

    def consume_tool_selection(self) -> tuple[bool, list[str] | None]:
        return (False, None)
