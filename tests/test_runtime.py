"""Unit tests for the owned async runtime (``EvoScientist.runtime``).

These are deliberately *synchronous* tests (pytest-asyncio ``auto`` mode would
otherwise wrap them in its own per-test loop). Each drives a fresh
``AgentRuntime`` — which owns its own daemon loop thread — from the test thread
and from worker threads, exactly as the sync frontends will.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time

import pytest

from EvoScientist.runtime import AgentRuntime


def _wait(predicate, *, timeout=5.0, interval=0.005):
    """Poll ``predicate`` until true or fail after ``timeout`` seconds.

    Done-callbacks that release the turn slot / log spawn errors run on the
    loop thread and can lag a ``Future.result()`` return, so cross-thread
    assertions poll rather than read once.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    assert predicate(), "condition was not met within timeout"
    return True


@pytest.fixture
def rt():
    """A fresh, isolated runtime (not the module singleton), closed on teardown."""
    runtime = AgentRuntime()
    try:
        yield runtime
    finally:
        runtime.close(timeout=5.0)


# -- start() / lifecycle ---------------------------------------------------


def test_start_is_idempotent_and_names_the_thread(rt):
    rt.start()
    first = rt._thread
    rt.start()
    assert rt._thread is first
    assert first is not None
    assert first.name == "evosci-runtime"
    assert first.daemon is True
    assert first.is_alive()


def test_loop_accessor_none_before_start_then_set(rt):
    assert rt.loop is None
    rt.start()
    assert rt.loop is not None
    assert rt.loop.is_running()


def test_start_applies_winloop_policy_before_loop_creation(monkeypatch):
    """``start()`` must call the Windows policy helper before creating a loop."""
    calls = []
    import EvoScientist.runtime as runtime_mod

    real_new_event_loop = asyncio.new_event_loop

    def spy_policy():
        calls.append("policy")
        return False

    def spy_new_loop():
        calls.append("new_loop")
        return real_new_event_loop()

    monkeypatch.setattr(runtime_mod, "ensure_proactor_event_loop_policy", spy_policy)
    monkeypatch.setattr(runtime_mod.asyncio, "new_event_loop", spy_new_loop)

    runtime = AgentRuntime()
    try:
        runtime.start()
        assert calls == ["policy", "new_loop"]
    finally:
        runtime.close(timeout=5.0)


# -- submit ----------------------------------------------------------------


def test_submit_runs_on_runtime_loop(rt):
    async def add():
        return 1 + 2

    assert rt.submit(add()).result(5) == 3


def test_submit_executes_on_runtime_thread(rt):
    async def who():
        return threading.current_thread().name

    assert rt.submit(who()).result(5) == "evosci-runtime"


def test_submit_from_worker_thread(rt):
    result = {}

    async def who():
        return threading.current_thread().name

    def worker():
        result["name"] = rt.submit(who()).result(5)

    t = threading.Thread(target=worker)
    t.start()
    t.join(5)
    assert result["name"] == "evosci-runtime"


# -- run_sync --------------------------------------------------------------


def test_run_sync_returns_result(rt):
    async def val():
        await asyncio.sleep(0)
        return 42

    assert rt.run_sync(val()) == 42


def test_run_sync_propagates_exception(rt):
    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        rt.run_sync(boom())


def test_run_sync_from_worker_thread(rt):
    result = {}

    async def val():
        return "worked"

    def worker():
        result["value"] = rt.run_sync(val())

    t = threading.Thread(target=worker)
    t.start()
    t.join(5)
    assert result["value"] == "worked"


def test_run_sync_on_runtime_thread_raises(rt):
    """The on-loop-thread guard: blocking the loop from inside itself is loud."""

    async def inner():
        try:
            rt.run_sync(asyncio.sleep(0))
        except RuntimeError as exc:
            return f"guard:{exc.__class__.__name__}"
        return "no-error"

    assert rt.run_sync(inner()) == "guard:RuntimeError"


def test_run_sync_keyboardinterrupt_cancels_and_reraises(rt, monkeypatch):
    """Ctrl+C while waiting cancels the underlying task, then re-raises KI."""
    started = threading.Event()
    cancelled = threading.Event()

    async def long_task():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    real_submit = rt.submit

    class _KIOnFirstResult:
        def __init__(self, fut):
            self._fut = fut
            self._raised = False

        def result(self, timeout=None):
            if not self._raised:
                self._raised = True
                # Ensure the task is genuinely running before the interrupt.
                started.wait(5)
                raise KeyboardInterrupt
            return self._fut.result(timeout)

        def cancel(self):
            return self._fut.cancel()

        def wait_settled(self, timeout=None):
            return self._fut.wait_settled(timeout)

    def fake_submit(coro):
        return _KIOnFirstResult(real_submit(coro))

    monkeypatch.setattr(rt, "submit", fake_submit)

    with pytest.raises(KeyboardInterrupt):
        rt.run_sync(long_task())

    assert cancelled.wait(5), "underlying task was not cancelled on Ctrl+C"


def test_run_sync_keyboardinterrupt_waits_for_async_cleanup(rt, monkeypatch):
    """Re-raise only after the cancelled task has finished its ``finally``."""
    started = threading.Event()
    cleanup_done = threading.Event()

    async def long_task():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.05)
            cleanup_done.set()

    real_submit = rt.submit

    class _KIOnFirstResult:
        def __init__(self, fut):
            self._fut = fut
            self._raised = False

        def result(self, timeout=None):
            if not self._raised:
                self._raised = True
                started.wait(5)
                raise KeyboardInterrupt
            return self._fut.result(timeout)

        def cancel(self):
            return self._fut.cancel()

        def wait_settled(self, timeout=None):
            return self._fut.wait_settled(timeout)

    monkeypatch.setattr(rt, "submit", lambda coro: _KIOnFirstResult(real_submit(coro)))

    with pytest.raises(KeyboardInterrupt):
        rt.run_sync(long_task())

    assert cleanup_done.is_set()


def test_run_sync_timeout_cancels_underlying_task_after_cleanup(rt):
    cleanup_done = threading.Event()

    async def long_task():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.05)
            cleanup_done.set()

    with pytest.raises(concurrent.futures.TimeoutError):
        rt.run_sync(long_task(), timeout=0.01)

    assert cleanup_done.is_set()


# -- spawn -----------------------------------------------------------------


def test_spawn_runs_coroutine(rt):
    ran = threading.Event()

    async def work():
        ran.set()

    rt.spawn(work())
    assert ran.wait(5)


def test_spawn_from_worker_thread(rt):
    ran = threading.Event()

    async def work():
        ran.set()

    def worker():
        rt.spawn(work(), name="worker-spawn")

    t = threading.Thread(target=worker)
    t.start()
    t.join(5)
    assert ran.wait(5)


def test_spawn_logs_unhandled_exception(rt, caplog):
    done = threading.Event()

    async def boom():
        try:
            raise RuntimeError("spawn-fail")
        finally:
            done.set()

    with caplog.at_level(logging.ERROR, logger="EvoScientist.runtime"):
        rt.spawn(boom(), name="boomer")
        assert done.wait(5)
        _wait(
            lambda: any(
                r.name == "EvoScientist.runtime"
                and r.exc_info is not None
                and isinstance(r.exc_info[1], RuntimeError)
                for r in caplog.records
            )
        )

    rec = next(
        r
        for r in caplog.records
        if r.name == "EvoScientist.runtime" and r.exc_info is not None
    )
    assert isinstance(rec.exc_info[1], RuntimeError)
    assert str(rec.exc_info[1]) == "spawn-fail"
    assert "boomer" in rec.getMessage()


def test_spawn_does_not_log_on_cancellation(rt, caplog):
    started = threading.Event()

    async def never():
        started.set()
        await asyncio.Event().wait()

    with caplog.at_level(logging.ERROR, logger="EvoScientist.runtime"):
        rt.spawn(never(), name="cancel-me")
        assert started.wait(5)
        # Close cancels the pending task; cancellation must not be logged.
        rt.close(timeout=5.0)

    assert [r for r in caplog.records if r.name == "EvoScientist.runtime"] == []


# -- close() ---------------------------------------------------------------


def test_close_drains_pending_queue_get_task(rt):
    """The exact scenario the old ``run_async`` conftest helper guarded.

    A task parked on ``asyncio.Queue.get`` must be cancelled and drained so the
    loop closes cleanly (no "Event loop is closed") and the thread joins.
    """
    started = threading.Event()

    async def blocked_getter():
        queue = asyncio.Queue()
        started.set()
        await queue.get()  # blocks forever

    rt.spawn(blocked_getter())
    assert started.wait(5)

    thread = rt._thread
    assert thread is not None
    assert thread.is_alive()

    rt.close(timeout=5.0)

    assert not thread.is_alive()
    assert rt.loop is None


def test_close_is_idempotent_and_safe_before_start(rt):
    rt.close(timeout=5.0)  # never started — no-op
    rt.start()
    rt.close(timeout=5.0)
    rt.close(timeout=5.0)  # already closed — no-op

    with pytest.raises(RuntimeError, match="is closed"):
        rt.start()


def test_submit_after_close_is_rejected_and_closes_coroutine(rt):
    rt.start()
    rt.close(timeout=5.0)

    class CloseSpy:
        closed = False

        def close(self):
            self.closed = True

    coro = CloseSpy()

    with pytest.raises(RuntimeError, match="is closed"):
        rt.submit(coro)

    assert coro.closed is True


def test_spawn_after_close_is_rejected_and_closes_coroutine(rt):
    rt.start()
    rt.close(timeout=5.0)

    class CloseSpy:
        closed = False

        def close(self):
            self.closed = True

    coro = CloseSpy()

    with pytest.raises(RuntimeError, match="is closed"):
        rt.spawn(coro)

    assert coro.closed is True


def test_submit_enqueue_is_atomic_with_close():
    """close() cannot detach the loop between submit validation and enqueue."""
    submit_holds_lock = threading.Event()
    release_submit = threading.Event()

    class PausedRuntime(AgentRuntime):
        def _ensure_started_locked(self):
            loop = super()._ensure_started_locked()
            submit_holds_lock.set()
            release_submit.wait(5)
            return loop

    runtime = PausedRuntime()
    submitted = {}
    submit_thread = threading.Thread(
        target=lambda: submitted.setdefault(
            "future", runtime.submit(asyncio.sleep(0, result="done"))
        )
    )
    submit_thread.start()
    assert submit_holds_lock.wait(5)

    close_thread = threading.Thread(target=lambda: runtime.close(timeout=5.0))
    close_thread.start()
    release_submit.set()
    submit_thread.join(5)
    close_thread.join(5)

    assert not submit_thread.is_alive()
    assert not close_thread.is_alive()
    future = submitted["future"]
    assert future.done()
    assert future.wait_settled(5)


def test_close_from_runtime_thread_raises_without_detaching_runtime(rt):
    async def close_from_runtime_thread():
        with pytest.raises(RuntimeError, match="runtime loop thread"):
            rt.close(timeout=5.0)
        return threading.current_thread().name

    assert rt.run_sync(close_from_runtime_thread()) == "evosci-runtime"

    thread = rt._thread
    assert thread is not None
    assert thread.is_alive()
    assert rt.loop is not None
    assert rt.run_sync(asyncio.sleep(0, result="still-running")) == "still-running"
