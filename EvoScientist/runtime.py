"""Application-scoped ownership for EvoScientist async work.

``AsyncRuntime`` owns one continuously running event loop on a dedicated
thread. Callers share the runtime, never its raw loop:

* synchronous code uses :meth:`AsyncRuntime.run_sync`;
* code already on another event loop uses :meth:`AsyncRuntime.run_async`;
* durable background work uses :meth:`AsyncRuntime.spawn`.

The API accepts factories rather than pre-created coroutines so construction
happens on the owned loop. There is deliberately no module singleton: an
application bootstrap owns an instance, passes it to consumers, and closes it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from EvoScientist._winloop import ensure_proactor_event_loop_policy

logger = logging.getLogger(__name__)

T = TypeVar("T")
AsyncFactory = Callable[[], Awaitable[T]]


class AsyncRuntimeError(RuntimeError):
    """Base error raised by :class:`AsyncRuntime`."""


class AsyncRuntimeClosedError(AsyncRuntimeError):
    """Raised when work is submitted after shutdown begins."""


class RuntimeHandle(concurrent.futures.Future[T], Generic[T]):
    """Cross-thread result with a separate coroutine-settlement signal.

    Cancelling a concurrent future marks it done immediately, while the
    asyncio task may still be running ``finally`` blocks. ``wait_settled``
    distinguishes those two moments for cancellation and shutdown paths.
    """

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__()
        self._name = name
        self._settled: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._settle_lock = threading.Lock()

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def settled(self) -> bool:
        return self._settled.done()

    def wait_settled(self, timeout: float | None = None) -> bool:
        """Block for task settlement; return ``False`` on timeout."""
        try:
            self._settled.result(timeout)
        except concurrent.futures.TimeoutError:
            return False
        return True

    async def wait_settled_async(self) -> None:
        """Wait for task settlement without blocking the caller's loop."""
        # ``wrap_future`` propagates cancellation back to the concurrent
        # future.  Settlement is a shared, one-way runtime signal rather than
        # work owned by any individual waiter, so a cancelled waiter must not
        # cancel or falsely complete it for everyone else.
        await asyncio.shield(asyncio.wrap_future(self._settled))

    def _mark_settled(self) -> None:
        # The lock makes the check-and-set atomic during forced shutdown.
        with self._settle_lock:
            if not self._settled.done():
                self._settled.set_result(None)


class AsyncRuntime:
    """Own a persistent asyncio loop and its task lifecycle.

    Instances start lazily on first submission or eagerly through
    :meth:`start`. A closed instance is permanently sealed; create a new
    instance for a new application lifetime.
    """

    def __init__(
        self,
        *,
        thread_name: str = "evosci-async-runtime",
        start_timeout: float = 5.0,
        cancellation_timeout: float = 2.0,
    ) -> None:
        if start_timeout <= 0:
            raise ValueError("start_timeout must be greater than zero")
        if cancellation_timeout < 0:
            raise ValueError("cancellation_timeout must not be negative")

        self._thread_name = thread_name
        self._start_timeout = start_timeout
        self._cancellation_timeout = cancellation_timeout

        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._closed = False
        self._failure: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Runtime-loop-only mapping. Strong references prevent pending tasks
        # and their handles from being garbage-collected.
        self._loop_tasks: dict[asyncio.Task[Any], RuntimeHandle[Any]] = {}

    def __enter__(self) -> AsyncRuntime:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            return (
                not self._closed
                and thread is not None
                and thread.is_alive()
                and self._ready.is_set()
                and not self._stopped.is_set()
            )

    def start(self) -> None:
        """Start the loop thread idempotently and wait until it serves work."""
        with self._lock:
            self._ensure_started_locked()

    def _ensure_started_locked(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise AsyncRuntimeClosedError(f"{self._thread_name} is closed")

        thread = self._thread
        if thread is not None:
            if thread.is_alive() and self._ready.is_set() and self._loop is not None:
                return self._loop
            error = AsyncRuntimeError(f"{self._thread_name} stopped unexpectedly")
            if self._failure is not None:
                raise error from self._failure
            raise error

        thread = threading.Thread(
            target=self._thread_main,
            name=self._thread_name,
            daemon=True,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._thread = None
            raise

        if not self._ready.wait(self._start_timeout):
            # A late-created loop observes this seal in _thread_main and exits
            # instead of becoming an orphan after its owner saw startup fail.
            self._closed = True
            raise TimeoutError(
                f"{self._thread_name} did not start within {self._start_timeout:.1f}s"
            )
        if self._failure is not None:
            raise AsyncRuntimeError(f"{self._thread_name} failed to start") from (
                self._failure
            )
        if self._loop is None or not thread.is_alive():
            raise AsyncRuntimeError(f"{self._thread_name} failed to start")
        return self._loop

    def _thread_main(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            ensure_proactor_event_loop_policy()
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)

            # A startup timeout may let close() seal the instance before loop
            # creation finishes. Do not leave a late-starting daemon behind.
            if self._closed:
                self._ready.set()
                return

            # The callback proves run_forever is serving work before start()
            # returns; merely allocating a loop is not sufficient.
            loop.call_soon(self._ready.set)
            loop.run_forever()
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
            logger.exception("%s loop failed", self._thread_name)
        finally:
            if loop is not None:
                self._settle_abandoned_tasks()
                loop.close()
                asyncio.set_event_loop(None)
            self._loop = None
            self._stopped.set()

    def _settle_abandoned_tasks(self) -> None:
        """Resolve handles when a stopped loop cannot unwind further."""
        for task, handle in list(self._loop_tasks.items()):
            if not task.done():
                task.cancel()
            if not handle.done():
                handle.cancel()
            handle._mark_settled()
        self._loop_tasks.clear()

    def submit(self, factory: AsyncFactory[T]) -> RuntimeHandle[T]:
        """Schedule a factory on the owned loop and return its handle.

        Submission is atomic with :meth:`close`: work is either queued before
        shutdown is sealed or rejected.
        """
        return self._submit(factory, name=None)

    def _submit(
        self,
        factory: AsyncFactory[T],
        *,
        name: str | None,
    ) -> RuntimeHandle[T]:
        if not callable(factory):
            raise TypeError("factory must be callable")

        handle: RuntimeHandle[T] = RuntimeHandle(name=name)
        context = contextvars.copy_context()

        def create_task() -> None:
            if handle.cancelled():
                handle._mark_settled()
                return

            async def invoke_factory() -> T:
                return await factory()

            try:
                task = asyncio.create_task(invoke_factory(), name=name)
            except BaseException as exc:
                self._set_handle_exception(handle, exc)
                handle._mark_settled()
                return

            self._loop_tasks[task] = handle

            def cancel_task(done: concurrent.futures.Future[T]) -> None:
                if not done.cancelled() or task.done():
                    return
                try:
                    task.get_loop().call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    handle._mark_settled()

            handle.add_done_callback(cancel_task)
            task.add_done_callback(self._copy_task_result)
            if handle.cancelled() and not task.done():
                task.cancel()

        try:
            with self._lock:
                loop = self._ensure_started_locked()
                self._enqueue_locked(loop, create_task, context)
        except BaseException:
            handle._mark_settled()
            raise
        return handle

    def _enqueue_locked(
        self,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[[], None],
        context: contextvars.Context,
    ) -> None:
        """Enqueue under the lifecycle lock; isolated for race testing."""
        try:
            loop.call_soon_threadsafe(callback, context=context)
        except RuntimeError as exc:
            raise AsyncRuntimeError(
                f"{self._thread_name} stopped while submitting work"
            ) from exc

    def _copy_task_result(self, task: asyncio.Task[Any]) -> None:
        handle = self._loop_tasks.pop(task)
        try:
            result = task.result()
        except asyncio.CancelledError:
            handle.cancel()
        except BaseException as exc:
            self._set_handle_exception(handle, exc)
        else:
            self._set_handle_result(handle, result)
        finally:
            handle._mark_settled()

    @staticmethod
    def _set_handle_result(handle: RuntimeHandle[Any], result: Any) -> None:
        try:
            handle.set_result(result)
        except concurrent.futures.InvalidStateError:
            pass  # External cancellation won; discard the completed result.

    @staticmethod
    def _set_handle_exception(handle: RuntimeHandle[Any], exc: BaseException) -> None:
        try:
            handle.set_exception(exc)
        except concurrent.futures.InvalidStateError:
            pass

    def run_sync(
        self,
        factory: AsyncFactory[T],
        *,
        timeout: float | None = None,
        on_submitted: Callable[[RuntimeHandle[T]], None] | None = None,
    ) -> T:
        """Run async work from sync code, blocking for its result.

        Any thread already running an event loop must use :meth:`run_async`;
        blocking it would freeze that frontend even if it is not the owned loop.
        ``on_submitted`` may retain the handle for cross-thread cancellation;
        it runs after submission and before this method starts blocking.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AsyncRuntimeError(
                "run_sync() cannot block a running event loop; use "
                "`await runtime.run_async(...)` instead"
            )

        handle = self.submit(factory)
        if on_submitted is not None:
            try:
                on_submitted(handle)
            except BaseException:
                handle.cancel()
                self._wait_for_cancellation(handle)
                raise
        try:
            return handle.result(timeout)
        except BaseException:
            if not handle.done():
                handle.cancel()
                self._wait_for_cancellation(handle)
            elif handle.cancelled():
                self._wait_for_cancellation(handle)
            raise

    async def run_async(self, factory: AsyncFactory[T]) -> T:
        """Await owned work without blocking the caller's event loop.

        Current UI adapters cross through ``to_thread`` and :meth:`run_sync`.
        This public bridge is retained for embedders and future async surfaces
        whose event loop must remain responsive while the owned loop does work.
        """
        caller_loop = asyncio.get_running_loop()
        with self._lock:
            runtime_loop = self._loop
        if caller_loop is runtime_loop:
            raise AsyncRuntimeError(
                "run_async() called from the owned loop; await directly instead"
            )

        handle = self.submit(factory)
        try:
            return await asyncio.wrap_future(handle)
        except asyncio.CancelledError:
            handle.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(handle.wait_settled_async()),
                    timeout=self._cancellation_timeout,
                )
            except TimeoutError:
                logger.warning(
                    "%s task did not settle within %.1fs after cancellation",
                    self._thread_name,
                    self._cancellation_timeout,
                )
            raise

    def spawn(
        self,
        factory: AsyncFactory[Any],
        *,
        name: str,
    ) -> RuntimeHandle[Any]:
        """Start durable work, retaining it and logging unhandled failures.

        This public primitive is reserved for runtime-owned background
        services; scoped request work should continue to use :meth:`submit`.
        """
        handle = self._submit(factory, name=name)
        handle.add_done_callback(self._on_background_done)
        return handle

    @staticmethod
    def _on_background_done(handle: concurrent.futures.Future[Any]) -> None:
        if handle.cancelled():
            return
        try:
            exc = handle.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            name = getattr(handle, "name", None)
            logger.error(
                "unhandled exception in runtime task %r",
                name,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _wait_for_cancellation(self, handle: RuntimeHandle[Any]) -> None:
        if not handle.wait_settled(self._cancellation_timeout):
            logger.warning(
                "%s task did not settle within %.1fs after cancellation",
                self._thread_name,
                self._cancellation_timeout,
            )

    @staticmethod
    async def _drain() -> None:
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        loop = asyncio.get_running_loop()
        await loop.shutdown_asyncgens()
        # Cancelling a task awaiting ``to_thread`` / ``run_in_executor`` does
        # not stop its underlying callable.  Do not report a clean runtime
        # shutdown until the owned loop's default executor is actually idle.
        await loop.shutdown_default_executor()

    def close(self, *, timeout: float = 5.0) -> None:
        """Seal intake, settle pending tasks, stop the loop, and join its thread.

        Shutdown is bounded by ``timeout``. Executor work cannot be preempted by
        asyncio cancellation; if it outlives the deadline this call raises
        :class:`TimeoutError` and the sealed runtime finishes shutting down in
        the background. A later ``close()`` waits for that shutdown and only
        succeeds after the executor is idle and the loop thread has stopped.
        """
        if timeout < 0:
            raise ValueError("timeout must not be negative")
        deadline = time.monotonic() + timeout

        with self._lock:
            thread = self._thread
            if thread is threading.current_thread():
                raise AsyncRuntimeError(
                    "close() cannot join the owned loop thread; close the "
                    "runtime from its application owner"
                )

            if self._closed:
                wait_for_existing_close = thread is not None and thread.is_alive()
                drain = None
                loop = self._loop
            else:
                self._closed = True
                wait_for_existing_close = False
                loop = self._loop
                if thread is None:
                    self._stopped.set()
                    return
                if loop is None:
                    # start() timed out while the runtime thread was still
                    # creating its loop. _thread_main observes _closed and
                    # exits as soon as creation finishes.
                    drain = None
                else:
                    # The lifecycle lock orders this after every accepted
                    # task-creation callback queued by submit().
                    try:
                        drain = asyncio.run_coroutine_threadsafe(self._drain(), loop)
                    except RuntimeError:
                        drain = None

        if wait_for_existing_close:
            if not self._stopped.wait(max(0.0, deadline - time.monotonic())):
                raise TimeoutError(
                    f"timed out waiting for {self._thread_name} shutdown"
                )
            return
        if thread is None:
            return

        if loop is None:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise TimeoutError(
                    f"{self._thread_name} did not stop within {timeout:.1f}s"
                )
            return

        drain_timed_out = False
        drain_error: BaseException | None = None
        if drain is not None:
            try:
                drain.result(max(0.0, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError:
                drain_timed_out = True

                def stop_after_drain(
                    _done: concurrent.futures.Future[None],
                ) -> None:
                    try:
                        loop.call_soon_threadsafe(loop.stop)
                    except RuntimeError:
                        pass

                # Keep the loop alive while executor work finishes. This
                # callback completes the already-sealed shutdown afterward.
                drain.add_done_callback(stop_after_drain)
            except BaseException as exc:
                drain_error = exc

        if drain_timed_out:
            raise TimeoutError(
                f"{self._thread_name} tasks did not settle within {timeout:.1f}s"
            )

        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        thread.join(max(0.0, deadline - time.monotonic()))

        if thread.is_alive():
            raise TimeoutError(
                f"{self._thread_name} did not stop within {timeout:.1f}s"
            )
        if drain_error is not None:
            raise drain_error


__all__ = [
    "AsyncFactory",
    "AsyncRuntime",
    "AsyncRuntimeClosedError",
    "AsyncRuntimeError",
    "RuntimeHandle",
]
