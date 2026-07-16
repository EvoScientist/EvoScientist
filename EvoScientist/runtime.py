"""Process-wide owned async runtime for EvoScientist.

A single persistent event loop, owned by a dedicated daemon thread
(``evosci-runtime``), is the one place agent coroutines run in *every* mode.
Frontends (Textual, prompt_toolkit, the serve poll loop) stay synchronous and
reach the loop through exactly three primitives — :meth:`AgentRuntime.submit`,
:meth:`AgentRuntime.run_sync`, :meth:`AgentRuntime.spawn` — plus the single-turn
:meth:`AgentRuntime.turn` gate. Everything else is ``await``.

This module owns the runtime loop creation path and the Windows Proactor policy
application (see :mod:`EvoScientist._winloop` and issue #283).
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from typing import Any

from EvoScientist._winloop import ensure_proactor_event_loop_policy

logger = logging.getLogger(__name__)


class _RuntimeFuture(concurrent.futures.Future[Any]):
    """Cross-thread future that tracks the underlying asyncio task settling.

    ``asyncio.run_coroutine_threadsafe`` completes its public future as soon as
    that future is cancelled, before the loop task has run its ``finally``
    blocks.  Runtime callers need the second event to distinguish those two
    moments.
    """

    def __init__(self) -> None:
        super().__init__()
        self._task_settled = threading.Event()

    def mark_settled(self) -> None:
        self._task_settled.set()

    def wait_settled(self, timeout: float | None = None) -> bool:
        return self._task_settled.wait(timeout)


class TurnInProgressError(RuntimeError):
    """Raised by :meth:`AgentRuntime.turn` when a foreground turn is active.

    "At most one foreground turn per process" is an explicit runtime invariant.
    Raising rather than queueing is deliberate: every frontend owns a queue with
    its own deferral policy, and an accidental second concurrent turn fails
    loudly instead of racing on process-global agent/middleware state.
    """


class AgentRuntime:
    """Owns the process-wide agent event loop and its lifecycle.

    A module-level singleton :data:`runtime` is provided; entrypoints call
    :meth:`start` eagerly if they wish, but any primitive starts the loop lazily
    on first use, so pure-server processes never pay for the thread.
    """

    #: How long ``run_sync`` waits for cancellation to settle before
    #: re-raising the original interruption (seconds).
    _CANCEL_WAIT = 2.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._atexit_registered = False
        # Once a started runtime is closed it is sealed permanently.  A no-op
        # close before first start remains harmless for fixture/entrypoint code.
        self._closed = False
        # Strong refs to fire-and-forget tasks so the loop can't GC them
        # mid-flight (asyncio only holds weak refs). Only mutated on the loop
        # thread (in the ``spawn`` closure and the done-callback).
        self._tasks: set[asyncio.Task[Any]] = set()
        # The single foreground-turn slot.
        self._turn_lock = threading.Lock()
        self._turn_active = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the runtime loop thread if not already running (idempotent).

        Applies the Windows Proactor loop policy *before* the loop is created —
        this is the only place loops get created, so it is the correct home for
        the ``_winloop`` safeguard (issue #283).
        """
        with self._lock:
            self._ensure_started_locked()

    def _ensure_started_locked(self) -> asyncio.AbstractEventLoop:
        """Return the live loop while ``self._lock`` is held."""
        if self._closed:
            raise RuntimeError("evosci-runtime is closed")
        if self._thread is not None and self._thread.is_alive():
            assert self._loop is not None
            return self._loop

        # Must run before ``new_event_loop`` — swapping the policy after a
        # loop exists does not change the already-running loop.
        ensure_proactor_event_loop_policy()
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(loop,),
            name="evosci-runtime",
            daemon=True,
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True
        return loop

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The runtime loop, or ``None`` before :meth:`start`.

        Exposed read-only for the rare "am I on the runtime thread?" check.
        Nothing else about the loop is public.
        """
        return self._loop

    def _is_on_runtime_thread(self) -> bool:
        thread = self._thread
        return thread is not None and threading.current_thread() is thread

    # -- primitives --------------------------------------------------------

    def submit(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
        """Schedule ``coro`` on the runtime loop; return its future.

        Callable from any thread *except* the runtime thread itself.
        """
        future = _RuntimeFuture()

        def _create_task() -> None:
            if future.cancelled():
                coro.close()
                future.mark_settled()
                return

            try:
                task = asyncio.create_task(coro)
            except BaseException as exc:
                coro.close()
                if not future.cancelled():
                    try:
                        future.set_exception(exc)
                    except concurrent.futures.InvalidStateError:
                        pass
                future.mark_settled()
                return

            def _cancel_task(done: concurrent.futures.Future) -> None:
                if done.cancelled() and not task.done():
                    task_loop = task.get_loop()
                    try:
                        task_loop.call_soon_threadsafe(task.cancel)
                    except RuntimeError:
                        if task_loop.is_closed():
                            # A timed-out shutdown may have closed around a
                            # cancellation-resistant task. No further cleanup
                            # can run once the loop is closed.
                            future.mark_settled()
                        else:  # pragma: no cover - defensive loop failure
                            raise

            def _copy_task_result(done: asyncio.Task[Any]) -> None:
                try:
                    result = done.result()
                except asyncio.CancelledError:
                    if not future.cancelled():
                        future.cancel()
                except BaseException as exc:
                    if not future.cancelled():
                        try:
                            future.set_exception(exc)
                        except concurrent.futures.InvalidStateError:
                            pass
                else:
                    if not future.cancelled():
                        try:
                            future.set_result(result)
                        except concurrent.futures.InvalidStateError:
                            pass
                finally:
                    future.mark_settled()

            future.add_done_callback(_cancel_task)
            task.add_done_callback(_copy_task_result)
            if future.cancelled():
                task.cancel()

        try:
            # Keep lifecycle validation, loop capture, and enqueue atomic with
            # close().  Once this callback is queued, close() queues its drain
            # after it, so the newly-created task is included in that drain.
            with self._lock:
                loop = self._ensure_started_locked()
                loop.call_soon_threadsafe(_create_task)
        except BaseException:
            coro.close()
            future.mark_settled()
            raise
        return future

    def run_sync(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        """Run ``coro`` on the runtime loop and block for its result.

        Guards:

        * Called *on* the runtime thread → :class:`RuntimeError`. Blocking the
          loop from inside itself is exactly the bug ``nest_asyncio`` has been
          hiding; we make it loud.
        * Timeout or ``KeyboardInterrupt`` while waiting → cancel the future
          (which cancels the underlying task thread-safely), wait briefly for
          cancellation to settle, then re-raise the original interruption.
        """
        if self._is_on_runtime_thread():
            coro.close()
            raise RuntimeError(
                "run_sync() called on the evosci-runtime loop thread; use await "
                "(or spawn) instead of blocking the loop from inside itself"
            )
        future = self.submit(coro)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            if timeout is not None and not future.done():
                self._cancel_and_drain_future(future)
            raise
        except KeyboardInterrupt:
            self._cancel_and_drain_future(future)
            raise
        except concurrent.futures.CancelledError:
            # External cancellation also completes the proxy future before the
            # asyncio task has necessarily unwound.
            self._wait_for_future_settlement(future)
            raise

    def _cancel_and_drain_future(self, future: concurrent.futures.Future) -> None:
        future.cancel()
        self._wait_for_future_settlement(future)

    def _wait_for_future_settlement(self, future: concurrent.futures.Future) -> None:
        wait_settled = getattr(future, "wait_settled", None)
        if callable(wait_settled):
            wait_settled(self._CANCEL_WAIT)
            return
        # Compatibility fallback for Future-like test doubles and callers that
        # did not originate from submit().
        try:
            future.result(self._CANCEL_WAIT)
        except (Exception, asyncio.CancelledError):
            pass

    def spawn(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        """Fire-and-forget ``coro`` on the runtime loop, logging any error.

        Thread-safe. Holds a strong reference to the task until it finishes and
        logs unhandled exceptions with full ``exc_info`` — this is the new home
        for ``async_notifier.spawn_watcher`` and kills the
        fire-and-forget-swallows-errors pattern.
        """

        def _create(loop: asyncio.AbstractEventLoop) -> None:
            try:
                task = loop.create_task(coro, name=name)
            except BaseException:
                coro.close()
                logger.exception("failed to create spawned task %r", name)
                return
            self._tasks.add(task)
            task.add_done_callback(self._on_spawn_done)

        try:
            # See submit(): enqueue while holding the lifecycle lock so close()
            # cannot stop this loop between validation and scheduling.
            with self._lock:
                loop = self._ensure_started_locked()
                loop.call_soon_threadsafe(_create, loop)
        except BaseException:
            coro.close()
            raise

    def _on_spawn_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(
                "unhandled exception in spawned task %r",
                task.get_name(),
                exc_info=exc,
            )

    # -- the single turn slot ---------------------------------------------

    @property
    def turn_active(self) -> bool:
        """Whether a foreground turn currently holds the slot.

        Readable from any thread (e.g. ``_check_channel_queue`` handlers that
        check-and-defer mid-turn).
        """
        return self._turn_active

    def turn(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
        """Run a foreground agent turn, gated by the single turn slot.

        Atomically acquires the process's one turn slot and raises
        :class:`TurnInProgressError` if a turn is already active. The slot is
        released in a done-callback, so completion, cancellation, and error all
        free it.
        """
        with self._turn_lock:
            if self._turn_active:
                coro.close()
                raise TurnInProgressError(
                    "a foreground agent turn is already active; "
                    "check runtime.turn_active and defer this work"
                )
            self._turn_active = True
        try:
            future = self.submit(coro)
        except BaseException:
            # submit() itself failed — release the slot we just took.
            with self._turn_lock:
                self._turn_active = False
            raise
        future.add_done_callback(self._release_turn_slot)
        return future

    def _release_turn_slot(self, _future: concurrent.futures.Future) -> None:
        with self._turn_lock:
            self._turn_active = False

    # -- shutdown ----------------------------------------------------------

    @staticmethod
    async def _drain(loop: asyncio.AbstractEventLoop) -> None:
        """Cancel every other pending task, gather them, shut down asyncgens.

        Replicates the semantics the old test-suite ``run_async`` helper
        documented as load-bearing: without this, tasks parked on
        ``asyncio.Queue.get`` raise "Event loop is closed" at teardown.
        """
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks(loop) if t is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await loop.shutdown_asyncgens()

    def close(self, *, timeout: float = 5.0) -> None:
        """Cancel pending work, drain the loop, and join the thread.

        Registered via ``atexit`` as a backstop; entrypoints may also call it
        explicitly. Idempotent and safe to call when never started.
        """
        with self._lock:
            thread = self._thread
            loop = self._loop
            if thread is None or loop is None:
                return
            if thread is threading.current_thread():
                raise RuntimeError(
                    "close() called on the evosci-runtime loop thread; call it "
                    "from another thread so shutdown can join the runtime safely"
                )
            self._closed = True
            self._thread = None
            self._loop = None
        if thread.is_alive():
            try:
                drain = asyncio.run_coroutine_threadsafe(self._drain(loop), loop)
                drain.result(timeout)
            except (Exception, asyncio.CancelledError):
                logger.exception("error draining evosci-runtime loop during close")
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                # The loop may have failed and closed independently while the
                # drain was timing out.  It is already stopped in that case.
                pass
            thread.join(timeout)


#: Module singleton; started lazily by the first primitive call, or eagerly by
#: an entrypoint's ``runtime.start()``.
runtime = AgentRuntime()
