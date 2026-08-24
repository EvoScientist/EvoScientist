"""Tests for the middleware event sink selection policy and custom-channel bridge.

``resolve_middleware_event_sink`` owns the sink decision for every middleware
stack; ``StreamBroadcastSink`` mirrors sink writes onto the run's LangGraph
``custom`` stream channel so server-side middleware events (tool-selection
lifecycle, fallback narration) reach the client — the server process has no
frontend sink to render them.
"""

from __future__ import annotations

from unittest.mock import patch

from EvoScientist.middleware.events import (
    MIDDLEWARE_EVENT_TAG,
    NO_OP_SINK,
    NoOpSink,
    RunScopedEventSink,
    StreamBroadcastSink,
    resolve_middleware_event_sink,
)


class _RecordingWriter:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> None:
        self.payloads.append(payload)


class _RecordingSink:
    """Minimal MiddlewareEventSink double that records every write."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def on_tool_selection_started(self, total_tools: int) -> None:
        self.calls.append(("started", total_tools))

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        self.calls.append(("selection", list(selected), total_tools))

    def on_tool_selection_ended(self) -> None:
        self.calls.append(("ended",))

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        self.calls.append(("fallback", text, style))


class TestResolveMiddlewareEventSink:
    def test_subagent_stacks_always_get_noop(self, monkeypatch):
        monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "full")
        sink = resolve_middleware_event_sink(_RecordingSink(), for_async_subagent=True)
        assert sink is NO_OP_SINK

    def test_local_main_stack_keeps_caller_sink_unwrapped(self, monkeypatch):
        monkeypatch.delenv("EVOSCIENTIST_DEPLOY_MODE", raising=False)
        caller = _RecordingSink()
        sink = resolve_middleware_event_sink(caller, for_async_subagent=False)
        assert sink is caller

    def test_local_main_stack_defaults_to_run_scoped(self, monkeypatch):
        monkeypatch.delenv("EVOSCIENTIST_DEPLOY_MODE", raising=False)
        sink = resolve_middleware_event_sink(None, for_async_subagent=False)
        assert isinstance(sink, RunScopedEventSink)

    def test_server_subprocess_wraps_in_broadcast(self, monkeypatch):
        monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "stripped")
        sink = resolve_middleware_event_sink(None, for_async_subagent=False)
        assert isinstance(sink, StreamBroadcastSink)
        assert isinstance(sink._sink, RunScopedEventSink)

    def test_server_subprocess_wraps_caller_sink(self, monkeypatch):
        monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "full")
        caller = _RecordingSink()
        sink = resolve_middleware_event_sink(caller, for_async_subagent=False)
        assert isinstance(sink, StreamBroadcastSink)
        assert sink._sink is caller


class TestStreamBroadcastSink:
    def test_writes_mirror_tagged_payloads_to_writer(self):
        writer = _RecordingWriter()
        broadcast = StreamBroadcastSink(_RecordingSink())
        with patch("langgraph.config.get_stream_writer", return_value=writer):
            broadcast.on_tool_selection_started(7)
            broadcast.on_tool_selection(["a", "b"], 7)
            broadcast.on_tool_selection_ended()
            broadcast.emit_fallback_notice("falling back", "red")
        assert writer.payloads == [
            {
                MIDDLEWARE_EVENT_TAG: {
                    "kind": "tool_selection_started",
                    "total_tools": 7,
                }
            },
            {
                MIDDLEWARE_EVENT_TAG: {
                    "kind": "tool_selection",
                    "selected": ["a", "b"],
                    "total_tools": 7,
                }
            },
            {MIDDLEWARE_EVENT_TAG: {"kind": "tool_selection_ended"}},
            {
                MIDDLEWARE_EVENT_TAG: {
                    "kind": "fallback_notice",
                    "text": "falling back",
                    "style": "red",
                }
            },
        ]

    def test_wrapped_sink_still_receives_every_write(self):
        writer = _RecordingWriter()
        inner = _RecordingSink()
        broadcast = StreamBroadcastSink(inner)
        with patch("langgraph.config.get_stream_writer", return_value=writer):
            broadcast.on_tool_selection_started(3)
            broadcast.emit_fallback_notice("x")
        assert inner.calls == [("started", 3), ("fallback", "x", "yellow")]

    def test_runtime_error_outside_runnable_context_is_swallowed(self):
        def _raising_writer_factory():
            raise RuntimeError("Called get_config outside of a runnable context")

        inner = _RecordingSink()
        broadcast = StreamBroadcastSink(inner)
        with patch("langgraph.config.get_stream_writer", _raising_writer_factory):
            # Must not raise — the bridge can never break a model call.
            broadcast.on_tool_selection_started(3)
        assert inner.calls == [("started", 3)]


class TestNoOpSinkContract:
    def test_noop_sink_is_default_constructible_and_silent(self):
        sink = NoOpSink()
        sink.on_tool_selection_started(1)
        sink.on_tool_selection(["x"], 1)
        sink.on_tool_selection_ended()
        sink.emit_fallback_notice("ignored")
        assert sink.tool_selection_active is False
        assert sink.tool_selection_pending() is False
        assert sink.consume_tool_selection() == (False, None)
