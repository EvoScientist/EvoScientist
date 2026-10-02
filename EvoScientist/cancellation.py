"""Cancellation context shared by streaming frontends and blocking tools."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_current_cancel_event: contextvars.ContextVar[threading.Event | None] = (
    contextvars.ContextVar("evoscientist_cancel_event", default=None)
)


@contextmanager
def bind_cancel_event(event: threading.Event) -> Iterator[None]:
    """Make a stream's cancellation event visible to nested sync tool calls."""
    token = _current_cancel_event.set(event)
    try:
        yield
    finally:
        _current_cancel_event.reset(token)


def current_cancel_event() -> threading.Event | None:
    """Return the cancellation event bound to the current agent run, if any."""
    return _current_cancel_event.get()
