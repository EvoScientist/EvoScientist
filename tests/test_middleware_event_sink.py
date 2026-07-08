"""Contract tests for the middleware event sink.

Pins two things:

1. The protocol / :class:`NoOpSink` shape is stable and structural.
2. The threading/blocking contract: sinks may be called from any thread, and
   a sink that blocks stalls its caller (nothing isolates a slow sink from the
   emitting thread). The deliberately-slow fake sink documents this.
"""

from __future__ import annotations

import threading
import time

from EvoScientist.middleware.events import MiddlewareEventSink, NoOpSink


class _SlowSink:
    """A deliberately-slow, thread-safe sink used to exercise the contract.

    Every event method sleeps ``delay`` seconds under a lock and records the
    thread it was called on. A real frontend must NOT do this — it exists only
    to demonstrate that a blocking sink holds the emitting thread.
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._lock = threading.Lock()
        self.calls: list[tuple[str, threading.Thread]] = []

    def _record(self, name: str) -> None:
        time.sleep(self._delay)
        with self._lock:
            self.calls.append((name, threading.current_thread()))

    def on_tool_selection_started(self, total_tools: int) -> None:
        self._record("started")

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        self._record("selection")

    def on_tool_selection_ended(self) -> None:
        self._record("ended")

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        self._record("notice")


def test_noopsink_satisfies_protocol():
    sink = NoOpSink()
    assert isinstance(sink, MiddlewareEventSink)
    # Every event is a no-op and returns None regardless of arguments.
    assert sink.on_tool_selection_started(10) is None
    assert sink.on_tool_selection(["a", "b"], 10) is None
    assert sink.on_tool_selection_ended() is None
    assert sink.emit_fallback_notice("fallback notice") is None


def test_slow_sink_satisfies_protocol():
    assert isinstance(_SlowSink(0.0), MiddlewareEventSink)


def test_sink_is_callable_from_any_thread():
    """Sink methods may be invoked from worker threads (the sync-hook world)."""
    sink = _SlowSink(0.0)
    main = threading.current_thread()

    def _worker() -> None:
        sink.on_tool_selection_started(3)
        sink.on_tool_selection(["think_tool"], 3)
        sink.on_tool_selection_ended()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()

    names = [name for name, _ in sink.calls]
    assert names == ["started", "selection", "ended"]
    # All calls landed on the worker thread, not the caller's thread.
    assert all(thread is not main for _, thread in sink.calls)
    assert all(thread is t for _, thread in sink.calls)


def test_blocking_sink_stalls_the_emitting_thread():
    """A slow sink holds its caller: the contract requires non-blocking sinks.

    This is the negative proof — nothing in the framework isolates the caller
    from a blocking sink, so the emitting thread waits the full delay.
    """
    delay = 0.2
    sink = _SlowSink(delay)

    start = time.perf_counter()
    sink.emit_fallback_notice("fallback notice")
    elapsed = time.perf_counter() - start

    # The caller was blocked for at least the sink's delay.
    assert elapsed >= delay
    assert [name for name, _ in sink.calls] == ["notice"]


def test_noopsink_never_blocks():
    sink = NoOpSink()
    start = time.perf_counter()
    for _ in range(10_000):
        sink.on_tool_selection_started(50)
        sink.emit_fallback_notice("fallback notice")
    elapsed = time.perf_counter() - start
    # 20k no-op calls are effectively free.
    assert elapsed < 0.5
