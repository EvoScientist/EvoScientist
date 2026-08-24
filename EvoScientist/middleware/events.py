"""Typed middleware → frontend event sink.

This module lives deliberately inside ``middleware/`` so the dependency
direction is always **frontends → middleware**, never the reverse. Middleware
reports facts about what happened during a model call; a frontend supplies a
sink implementation that owns its own display state and renders (or ignores)
those facts.

Two families of events are evidenced today and modelled here:

* **Tool selection** — the adaptive ``LLMToolSelectorMiddleware`` wrapper
  reports when a selection LLM call starts, which tools survived filtering,
  and when it ends.
* **Model fallback** — the fallback middleware reports lifecycle narration for
  failed primary calls and fallback attempts.

Tool-selection events are structured. Fallback notices are pre-formatted
narration plus a style, because the middleware owns the wording and the sink
only decides where to display it.

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
a slow sink from the run.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Any, Protocol, runtime_checkable

MIDDLEWARE_EVENT_TAG = "evoscientist"
"""Wire tag for middleware events carried on the LangGraph ``custom`` channel.

Payloads are ``{MIDDLEWARE_EVENT_TAG: {"kind": ..., ...}}`` so consumers can
distinguish middleware events from any other custom-channel traffic without
inspecting payload internals."""


@runtime_checkable
class MiddlewareEventSink(Protocol):
    """Structured display events emitted by middleware hooks.

    Implementations are supplied by frontends/sessions and injected at the
    agent composition root (see ``EvoScientist.EvoScientist``). Main agents
    built without an explicit sink use :class:`RunScopedEventSink`; subagent
    stacks use :class:`NoOpSink`.

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

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        """Render a pre-formatted fallback lifecycle line."""
        ...


@runtime_checkable
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


@runtime_checkable
class SessionEvents(MiddlewareEventSink, ToolSelectionView, Protocol):
    """Gateway-carried session sink for both middleware writes and stream reads."""


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

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        pass

    # --- ToolSelectionView (read side) -----------------------------------
    @property
    def tool_selection_active(self) -> bool:
        return False

    def tool_selection_pending(self) -> bool:
        return False

    def consume_tool_selection(self) -> tuple[bool, list[str] | None]:
        return (False, None)


NO_OP_SINK = NoOpSink()


_current_run_event_sink: ContextVar[MiddlewareEventSink | None] = ContextVar(
    "evoscientist_current_run_event_sink", default=None
)


def bind_run_event_sink(
    events: MiddlewareEventSink,
) -> Token[MiddlewareEventSink | None]:
    """Bind middleware events to the sink for the current streamed run."""
    return _current_run_event_sink.set(events)


def reset_run_event_sink(token: Token[MiddlewareEventSink | None]) -> None:
    """Restore the previous run-scoped event sink binding."""
    _current_run_event_sink.reset(token)


class RunScopedEventSink:
    """Proxy sink for default main agents.

    A main agent can be constructed before the frontend or local gateway exists.
    This proxy lets that agent report middleware events to whichever sink the
    active ``stream_agent_events`` call bound for the current run. If the agent
    is invoked outside that streaming path, events are dropped.
    """

    __slots__ = ()

    def _sink(self) -> MiddlewareEventSink:
        return _current_run_event_sink.get() or NO_OP_SINK

    def on_tool_selection_started(self, total_tools: int) -> None:
        self._sink().on_tool_selection_started(total_tools)

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        self._sink().on_tool_selection(selected, total_tools)

    def on_tool_selection_ended(self) -> None:
        self._sink().on_tool_selection_ended()

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        self._sink().emit_fallback_notice(text, style)


class StreamBroadcastSink:
    """Mirror sink writes onto the run's LangGraph ``custom`` stream channel.

    Installed on main-agent stacks inside the langgraph dev subprocess, where
    no frontend sink can exist: the wrapped :class:`RunScopedEventSink` has no
    binder there, so its writes are silent, and the custom stream event —
    delivered to the client on the ``custom`` channel and mapped by
    ``LangGraphServerGateway`` onto its ``events`` sink — is the actual
    delivery channel. Every write still forwards to the wrapped sink first,
    preserving local in-process consumers if a binder ever exists.

    Delivery is single-rendered per backend by construction, with no
    suppression flag: the local CLI process never sets
    ``EVOSCIENTIST_DEPLOY_MODE``, so local stacks never broadcast and render
    once via the injected frontend sink; server stacks broadcast, and their
    silent wrapped sink renders nothing.

    The stream write is best-effort and must never break a model call:
    ``get_stream_writer()`` raises ``RuntimeError`` outside a runnable
    context (e.g. background in-process invocations) — such writes are
    skipped, and any other writer failure is swallowed after logging. Honours
    the module threading/blocking contract.
    """

    __slots__ = ("_sink",)

    def __init__(self, sink: MiddlewareEventSink) -> None:
        self._sink = sink

    def _emit(self, payload: dict[str, Any]) -> None:
        try:
            from langgraph.config import get_stream_writer

            get_stream_writer()({MIDDLEWARE_EVENT_TAG: payload})
        except RuntimeError:
            # Outside a runnable context — nothing subscribes anyway.
            pass
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "custom-channel mirror failed for %r", payload, exc_info=True
            )

    def on_tool_selection_started(self, total_tools: int) -> None:
        self._sink.on_tool_selection_started(total_tools)
        self._emit({"kind": "tool_selection_started", "total_tools": total_tools})

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        self._sink.on_tool_selection(selected, total_tools)
        self._emit(
            {
                "kind": "tool_selection",
                "selected": list(selected),
                "total_tools": total_tools,
            }
        )

    def on_tool_selection_ended(self) -> None:
        self._sink.on_tool_selection_ended()
        self._emit({"kind": "tool_selection_ended"})

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        self._sink.emit_fallback_notice(text, style)
        self._emit({"kind": "fallback_notice", "text": text, "style": style})


def resolve_middleware_event_sink(
    events: MiddlewareEventSink | None,
    *,
    for_async_subagent: bool,
) -> MiddlewareEventSink:
    """Select the middleware event sink for a middleware stack.

    Policy (single home for the decision):

    * subagent stacks always use :class:`NoOpSink` — they never drive the
      main-agent frontend widgets, and their results reach the client via
      thread state, not live custom events;
    * main stacks without an explicit sink use :class:`RunScopedEventSink`;
    * main stacks inside a langgraph dev subprocess (``EVOSCIENTIST_DEPLOY_MODE``
      set — i.e. the process serves graphs over HTTP, where no frontend sink
      can exist) are additionally wrapped in :class:`StreamBroadcastSink` so
      middleware events reach the client over the ``custom`` channel. The
      local CLI process never sets that variable, so local stacks never
      broadcast (single render via the injected sink).
    """
    if for_async_subagent:
        return NO_OP_SINK
    sink = events if events is not None else RunScopedEventSink()
    if os.environ.get("EVOSCIENTIST_DEPLOY_MODE"):
        return StreamBroadcastSink(sink)
    return sink
