"""Focused tests for the application-scoped owned async runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import threading
import time
from typing import Any

import pytest

from EvoScientist.runtime import (
    AsyncRuntime,
    AsyncRuntimeClosedError,
    AsyncRuntimeError,
    RuntimeHandle,
)


def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "condition was not met before timeout"


@pytest.fixture
def runtime():
    instance = AsyncRuntime(cancellation_timeout=1.0)
    try:
        yield instance
    finally:
        instance.close(timeout=5.0)


def test_constructor_validates_timeouts():
    with pytest.raises(ValueError, match="start_timeout"):
        AsyncRuntime(start_timeout=0)
    with pytest.raises(ValueError, match="cancellation_timeout"):
        AsyncRuntime(cancellation_timeout=-1)


def test_start_is_idempotent_and_waits_until_loop_runs(runtime):
    runtime.start()
    first_thread = runtime._thread
    runtime.start()

    assert runtime._thread is first_thread
    assert first_thread is not None
    assert first_thread.name == "evosci-async-runtime"
    assert first_thread.daemon
    assert first_thread.is_alive()
    assert runtime.is_running


def test_context_manager_owns_start_and_close():
    with AsyncRuntime(thread_name="context-runtime") as runtime:
        thread = runtime._thread
        assert runtime.is_running
        assert runtime.run_sync(lambda: asyncio.sleep(0, result=3)) == 3

    assert thread is not None
    assert not thread.is_alive()
    assert not runtime.is_running


def test_start_applies_windows_policy_before_creating_loop(monkeypatch):
    import EvoScientist.runtime as runtime_module

    calls: list[str] = []
    real_new_event_loop = asyncio.new_event_loop

    def policy_spy() -> bool:
        calls.append("policy")
        return False

    def loop_spy() -> asyncio.AbstractEventLoop:
        calls.append("loop")
        return real_new_event_loop()

    monkeypatch.setattr(runtime_module, "ensure_proactor_event_loop_policy", policy_spy)
    monkeypatch.setattr(runtime_module.asyncio, "new_event_loop", loop_spy)

    runtime = AsyncRuntime()
    try:
        runtime.start()
        assert calls == ["policy", "loop"]
    finally:
        runtime.close()


def test_close_after_startup_timeout_does_not_leave_runtime_thread(monkeypatch):
    import EvoScientist.runtime as runtime_module

    loop_creation_started = threading.Event()
    release_loop_creation = threading.Event()
    real_new_event_loop = asyncio.new_event_loop

    def delayed_new_event_loop() -> asyncio.AbstractEventLoop:
        loop_creation_started.set()
        assert release_loop_creation.wait(5)
        return real_new_event_loop()

    monkeypatch.setattr(
        runtime_module.asyncio, "new_event_loop", delayed_new_event_loop
    )
    runtime = AsyncRuntime(start_timeout=0.01)

    with pytest.raises(TimeoutError, match="did not start"):
        runtime.start()
    assert loop_creation_started.is_set()

    release_loop_creation.set()
    runtime.close(timeout=5)

    assert runtime._thread is not None
    assert not runtime._thread.is_alive()


def test_runtime_is_instance_scoped_not_a_module_singleton():
    import EvoScientist.runtime as runtime_module

    assert not hasattr(runtime_module, "runtime")
    assert AsyncRuntime() is not AsyncRuntime()


def test_submit_invokes_factory_on_owned_thread(runtime):
    factory_thread: list[str] = []

    async def identify() -> str:
        return threading.current_thread().name

    def factory():
        factory_thread.append(threading.current_thread().name)
        return identify()

    handle = runtime.submit(factory)

    assert isinstance(handle, RuntimeHandle)
    assert handle.result(5) == "evosci-async-runtime"
    assert handle.wait_settled(5)
    assert factory_thread == ["evosci-async-runtime"]


def test_submissions_share_one_owned_loop(runtime):
    async def current_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = runtime.run_sync(current_loop)
    second = runtime.run_sync(current_loop)

    assert first is second


def test_submit_propagates_context_variables(runtime):
    request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id")
    token = request_id.set("request-42")
    try:
        handle = runtime.submit(
            lambda: asyncio.sleep(0, result=request_id.get("missing"))
        )
        request_id.set("changed-after-submit")
        assert handle.result(5) == "request-42"
    finally:
        request_id.reset(token)


def test_submit_is_safe_from_worker_threads(runtime):
    results: list[int] = []

    def worker(value: int) -> None:
        result = runtime.submit(lambda: asyncio.sleep(0, result=value * 2)).result(5)
        results.append(result)

    threads = [threading.Thread(target=worker, args=(value,)) for value in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(results) == [0, 2, 4, 6]


def test_handle_distinguishes_public_cancellation_from_task_settlement(runtime):
    started = threading.Event()
    release_cleanup = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            while not release_cleanup.is_set():
                await asyncio.sleep(0.005)

    handle = runtime.submit(blocked)
    assert started.wait(5)

    assert handle.cancel()
    assert handle.done()
    assert handle.cancelled()
    assert not handle.settled

    release_cleanup.set()
    assert handle.wait_settled(5)


async def test_cancelling_async_waiter_does_not_cancel_settlement_signal(runtime):
    started = threading.Event()
    release = threading.Event()

    async def blocked() -> None:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.005)

    handle = runtime.submit(blocked)
    assert await asyncio.to_thread(started.wait, 5)

    waiter = asyncio.create_task(handle.wait_settled_async())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert not handle.settled
    assert handle.wait_settled(0) is False

    release.set()
    assert handle.result(5) is None
    assert handle.wait_settled(5)


def test_cancelling_before_task_creation_never_invokes_factory(runtime):
    loop_blocked = threading.Event()
    release_loop = threading.Event()
    factory_called = False

    def block_loop() -> None:
        loop_blocked.set()
        assert release_loop.wait(5)

    runtime.start()
    assert runtime._loop is not None
    runtime._loop.call_soon_threadsafe(block_loop)
    assert loop_blocked.wait(5)

    async def operation() -> None:
        nonlocal factory_called
        factory_called = True

    handle = runtime.submit(operation)
    assert handle.cancel()
    release_loop.set()

    assert handle.wait_settled(5)
    assert not factory_called


def test_run_sync_returns_result_and_propagates_exception(runtime):
    assert runtime.run_sync(lambda: asyncio.sleep(0, result=42)) == 42

    async def fail() -> None:
        raise ValueError("broken")

    with pytest.raises(ValueError, match="broken"):
        runtime.run_sync(fail)


async def test_run_sync_rejects_every_running_event_loop(runtime):
    called = False

    async def operation() -> None:
        nonlocal called
        called = True

    with pytest.raises(AsyncRuntimeError, match="cannot block a running event loop"):
        runtime.run_sync(operation)

    assert not called


def test_run_sync_timeout_cancels_and_waits_for_cleanup(runtime):
    cleanup_finished = threading.Event()

    async def blocked() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished.set()

    with pytest.raises(concurrent.futures.TimeoutError):
        runtime.run_sync(blocked, timeout=0.01)

    assert cleanup_finished.is_set()


def test_run_sync_interrupt_cancels_and_waits_for_cleanup(runtime, monkeypatch):
    started = threading.Event()
    cleanup_finished = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished.set()

    real_submit = runtime.submit

    class InterruptingResult:
        def __init__(self, handle: RuntimeHandle[Any]) -> None:
            self._handle = handle

        def result(self, timeout: float | None = None) -> Any:
            assert started.wait(5)
            raise KeyboardInterrupt

        def done(self) -> bool:
            return self._handle.done()

        def cancelled(self) -> bool:
            return self._handle.cancelled()

        def cancel(self) -> bool:
            return self._handle.cancel()

        def wait_settled(self, timeout: float | None = None) -> bool:
            return self._handle.wait_settled(timeout)

    monkeypatch.setattr(
        runtime,
        "submit",
        lambda factory: InterruptingResult(real_submit(factory)),
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.run_sync(blocked)

    assert cleanup_finished.is_set()


async def test_run_async_bridges_without_blocking_callers_loop(runtime):
    caller_loop = asyncio.get_running_loop()
    runtime_loop, thread_name = await runtime.run_async(lambda: _loop_and_thread())

    assert runtime_loop is not caller_loop
    assert thread_name == "evosci-async-runtime"


async def _loop_and_thread() -> tuple[asyncio.AbstractEventLoop, str]:
    return asyncio.get_running_loop(), threading.current_thread().name


async def test_run_async_propagates_caller_context(runtime):
    request_id: contextvars.ContextVar[str] = contextvars.ContextVar("async_request_id")
    token = request_id.set("from-ui-loop")
    try:
        assert (
            await runtime.run_async(
                lambda: asyncio.sleep(0, result=request_id.get("missing"))
            )
            == "from-ui-loop"
        )
    finally:
        request_id.reset(token)


async def test_run_async_cancellation_waits_for_runtime_cleanup(runtime):
    started = threading.Event()
    cleanup_finished = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished.set()

    caller = asyncio.create_task(runtime.run_async(blocked))
    assert await asyncio.to_thread(started.wait, 5)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller

    assert cleanup_finished.is_set()


def test_run_async_rejects_calls_from_owned_loop(runtime):
    factory_called = False

    async def operation() -> None:
        nonlocal factory_called
        factory_called = True

    async def invoke_from_runtime() -> None:
        with pytest.raises(AsyncRuntimeError, match="owned loop"):
            await runtime.run_async(operation)

    runtime.run_sync(invoke_from_runtime)
    assert not factory_called


def test_spawn_runs_durable_work_and_returns_named_handle(runtime):
    release = threading.Event()
    finished = threading.Event()

    async def background() -> str:
        while not release.is_set():
            await asyncio.sleep(0.005)
        finished.set()
        return "complete"

    handle = runtime.spawn(background, name="durable-work")
    assert handle.name == "durable-work"
    assert not handle.done()

    release.set()
    assert handle.result(5) == "complete"
    assert handle.wait_settled(5)
    assert finished.is_set()


def test_spawn_logs_unhandled_failures(runtime, caplog):
    async def fail() -> None:
        raise RuntimeError("background exploded")

    with caplog.at_level(logging.ERROR, logger="EvoScientist.runtime"):
        handle = runtime.spawn(fail, name="failing-background")
        with pytest.raises(RuntimeError, match="background exploded"):
            handle.result(5)
        assert handle.wait_settled(5)
        _wait_until(
            lambda: any(
                "failing-background" in record.getMessage() for record in caplog.records
            )
        )

    record = next(
        record
        for record in caplog.records
        if "failing-background" in record.getMessage()
    )
    assert isinstance(record.exc_info[1], RuntimeError)


def test_spawn_cancellation_is_not_logged(runtime, caplog):
    started = threading.Event()

    async def blocked() -> None:
        started.set()
        await asyncio.Event().wait()

    with caplog.at_level(logging.ERROR, logger="EvoScientist.runtime"):
        handle = runtime.spawn(blocked, name="cancelled-background")
        assert started.wait(5)
        handle.cancel()
        assert handle.wait_settled(5)

    assert not caplog.records


def test_close_cancels_and_settles_pending_work(runtime):
    started = threading.Event()
    cleanup_finished = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            await asyncio.Queue().get()
        finally:
            cleanup_finished.set()

    handle = runtime.spawn(blocked, name="pending")
    assert started.wait(5)
    thread = runtime._thread

    runtime.close(timeout=5)

    assert handle.cancelled()
    assert handle.wait_settled(5)
    assert cleanup_finished.is_set()
    assert thread is not None
    assert not thread.is_alive()
    assert not runtime.is_running


def test_close_waits_for_default_executor_work():
    runtime = AsyncRuntime()
    started = threading.Event()
    finished = threading.Event()

    def blocking_job() -> None:
        started.set()
        time.sleep(0.1)
        finished.set()

    runtime.spawn(
        lambda: asyncio.to_thread(blocking_job),
        name="executor-job",
    )
    assert started.wait(2)

    runtime.close(timeout=2)

    assert finished.is_set()
    assert runtime._thread is not None
    assert not runtime._thread.is_alive()


def test_close_timeout_never_reports_success_while_executor_is_active():
    runtime = AsyncRuntime()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_job() -> None:
        started.set()
        release.wait()
        finished.set()

    runtime.spawn(
        lambda: asyncio.to_thread(blocking_job),
        name="blocked-executor-job",
    )
    assert started.wait(2)

    with pytest.raises(TimeoutError, match="did not settle"):
        runtime.close(timeout=0.05)

    assert not finished.is_set()
    assert runtime._thread is not None
    assert runtime._thread.is_alive()

    release.set()
    runtime.close(timeout=2)

    assert finished.is_set()
    assert not runtime._thread.is_alive()


def test_close_is_idempotent_and_close_before_start_seals_runtime():
    runtime = AsyncRuntime()
    runtime.close()
    runtime.close()

    with pytest.raises(AsyncRuntimeClosedError, match="closed"):
        runtime.start()
    with pytest.raises(AsyncRuntimeClosedError, match="closed"):
        runtime.submit(lambda: asyncio.sleep(0))


def test_close_rejects_calls_from_runtime_thread(runtime):
    async def close_from_runtime() -> str:
        with pytest.raises(AsyncRuntimeError, match="application owner"):
            runtime.close()
        return threading.current_thread().name

    assert runtime.run_sync(close_from_runtime) == "evosci-async-runtime"
    assert runtime.run_sync(lambda: asyncio.sleep(0, result="still alive")) == (
        "still alive"
    )


def test_submit_enqueue_is_atomic_with_close():
    submit_holds_lock = threading.Event()
    release_submit = threading.Event()

    class PausedRuntime(AsyncRuntime):
        def _enqueue_locked(self, loop, callback, context):
            submit_holds_lock.set()
            assert release_submit.wait(5)
            super()._enqueue_locked(loop, callback, context)

    runtime = PausedRuntime()
    submitted: dict[str, RuntimeHandle[str]] = {}

    def submit() -> None:
        submitted["handle"] = runtime.submit(
            lambda: asyncio.sleep(0, result="accepted")
        )

    submit_thread = threading.Thread(target=submit)
    close_thread = threading.Thread(target=lambda: runtime.close(timeout=5))
    submit_thread.start()
    assert submit_holds_lock.wait(5)
    close_thread.start()

    release_submit.set()
    submit_thread.join(5)
    close_thread.join(5)

    assert not submit_thread.is_alive()
    assert not close_thread.is_alive()
    handle = submitted["handle"]
    assert handle.done()
    assert handle.wait_settled(5)


def test_submission_after_started_runtime_is_closed_never_invokes_factory(runtime):
    runtime.start()
    runtime.close()
    called = False

    async def operation() -> None:
        nonlocal called
        called = True

    with pytest.raises(AsyncRuntimeClosedError, match="closed"):
        runtime.submit(operation)

    assert not called
