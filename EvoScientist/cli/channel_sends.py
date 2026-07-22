"""Non-blocking bridge for streaming callbacks sent through a channel loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from typing import Any


class PendingChannelSends:
    """Schedule channel I/O without blocking the caller's event loop.

    Streaming callbacks run on the owned async runtime, while channel clients
    belong to the channel bus loop.  Submissions therefore only enqueue work;
    the frontend settles the returned futures after streaming has unwound.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None,
        logger: logging.Logger,
    ) -> None:
        self._loop = loop
        self._logger = logger
        self._lock = threading.Lock()
        self._pending: list[tuple[concurrent.futures.Future[Any], str, int]] = []

    @staticmethod
    def _close(coro: Coroutine[Any, Any, Any]) -> None:
        coro.close()

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        label: str,
        timeout: int = 15,
    ) -> None:
        """Schedule one send and return immediately."""
        if self._loop is None:
            self._close(coro)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception as exc:
            self._close(coro)
            self._logger.debug("%s send failed: %s", label, exc)
            return
        with self._lock:
            self._pending.append((future, label, timeout))

    def _take_pending(
        self,
    ) -> list[tuple[concurrent.futures.Future[Any], str, int]]:
        with self._lock:
            pending = self._pending
            self._pending = []
        return pending

    def settle(self) -> None:
        """Wait for scheduled sends from a synchronous frontend thread."""
        for future, label, timeout in self._take_pending():
            try:
                future.result(timeout=timeout)
            except Exception as exc:
                future.cancel()
                self._logger.debug("%s send failed: %s", label, exc)

    async def settle_async(self) -> None:
        """Wait for scheduled sends without blocking the frontend loop."""
        pending = self._take_pending()
        try:
            for future, label, timeout in pending:
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
                except TimeoutError as exc:
                    future.cancel()
                    self._logger.debug("%s send failed: %s", label, exc)
                except asyncio.CancelledError as exc:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    self._logger.debug("%s send failed: %s", label, exc)
                except Exception as exc:
                    self._logger.debug("%s send failed: %s", label, exc)
        except asyncio.CancelledError:
            for future, _label, _timeout in pending:
                future.cancel()
            raise
