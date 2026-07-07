"""Process-wide owned async runtime for EvoScientist (audit finding #8).

A single persistent event loop, owned by a dedicated daemon thread
(``evosci-runtime``), is the one place agent coroutines run in *every* mode.
Frontends (Textual, prompt_toolkit, the serve poll loop) stay synchronous and
reach the loop through exactly three primitives — :meth:`AgentRuntime.submit`,
:meth:`AgentRuntime.run_sync`, :meth:`AgentRuntime.spawn` — plus the single-turn
:meth:`AgentRuntime.turn` gate. Everything else is ``await``.

This module owns the sole ``new_event_loop()`` call in the codebase and the
Windows Proactor policy application (see :mod:`EvoScientist._winloop` and issue
#283). No other module should create loops, call ``asyncio.run``, or apply
``nest_asyncio``; the ``tests/test_no_adhoc_event_loops.py`` guard enforces this.

See ``design-async-runtime.md`` §3 for the full rationale.
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


class TurnInProgressError(RuntimeError):
    """Raised by :meth:`AgentRuntime.turn` when a foreground turn is active.

    "At most one foreground turn per process" is an explicit invariant of the
    runtime (see ``design-async-runtime.md`` §2 non-goals and R4). Raising —
    rather than queueing — is deliberate: every frontend owns a queue with its
    own deferral policy, and an accidental second concurrent turn fails loudly
    instead of racing on process-global agent/middleware state.
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
            if self._thread is not None and self._thread.is_alive():
                return
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
        self.start()
        assert self._loop is not None  # start() guarantees it
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

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

    def _cancel_and_drain_future(self, future: concurrent.futures.Future) -> None:
        future.cancel()
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
        self.start()
        assert self._loop is not None
        loop = self._loop

        def _create() -> None:
            task = loop.create_task(coro, name=name)
            self._tasks.add(task)
            task.add_done_callback(self._on_spawn_done)

        loop.call_soon_threadsafe(_create)

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
            self._thread = None
            self._loop = None
        if thread.is_alive():
            try:
                drain = asyncio.run_coroutine_threadsafe(self._drain(loop), loop)
                drain.result(timeout)
            except (Exception, asyncio.CancelledError):
                logger.exception("error draining evosci-runtime loop during close")
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout)


#: Module singleton; started lazily by the first primitive call, or eagerly by
#: an entrypoint's ``runtime.start()``.
runtime = AgentRuntime()
