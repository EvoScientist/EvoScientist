"""Shared terminal stop-signal handling for deploy and WebUI commands."""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Iterator
from typing import Any

from rich.console import Console

from ..stream.console import console


def stop_signals() -> list[signal.Signals]:
    """The signals that end deploy and the terminal WebUI. On macOS and Linux that
    includes SIGHUP: the terminal closed or the SSH session dropped. (On
    Windows the server shares the terminal's console, so closing it ends the
    server too.)"""
    stops = [signal.SIGINT, signal.SIGTERM]
    hangup = getattr(signal, "SIGHUP", None)
    if hangup is not None:
        stops.append(hangup)
    return stops


def say(markup: str, *, to: Console = console) -> None:
    """Print a stop message; after a hang-up the terminal is gone, and
    writing to it fails without stopping the stop."""
    with contextlib.suppress(OSError):
        to.print(markup)


def ignore_stop_signal(signum: int, frame: Any) -> None:
    """A stop signal after the first: the stop is already under way."""


def interrupt_stop_signal(signum: int, frame: Any) -> None:
    """End deploy as Ctrl+C does, with the signals that end it ignored from
    now on, so a second one cannot cut the stop short. Ignored by a no-op:
    with ``SIG_IGN``, a second signal already pending prints an error."""
    ignore_further_stop_signals()
    signal.default_int_handler(signum, frame)


def ignore_further_stop_signals() -> None:
    """Shield cleanup without replacing inherited or caller-owned handlers.

    Unlike the scoped signal guard, this does not restore handlers: the outer
    interrupt_on_stop_signals context restores them after cleanup finishes.
    """
    for sig in stop_signals():
        if signal.getsignal(sig) is interrupt_stop_signal:
            signal.signal(sig, ignore_stop_signal)


@contextlib.contextmanager
def interrupt_on_stop_signals() -> Iterator[None]:
    """Each signal that ends deploy raises ``KeyboardInterrupt``, as Ctrl+C
    does, so the stop in deploy's ``finally`` runs: SIGTERM and SIGHUP would
    otherwise end the process without it, and leave the server, which runs
    in its own session or process group, running. Installed before the
    start, so a signal while the server starts stops it too. A SIGHUP this
    process started with ignored stays ignored: ``nohup EvoSci deploy &``
    keeps deploy and its server up after a logout. (SIGINT is taken over
    even when ignored, as a shell ignores it for ``EvoSci deploy &`` and a
    later ``kill -INT`` stops deploy.)

    The first such signal also ignores the rest from then on: every stop
    that follows it, the one ``ensure_server`` makes of a server it was
    starting as well as deploy's own, runs to the end."""
    hangup = getattr(signal, "SIGHUP", None)
    previous = {
        sig: signal.signal(sig, interrupt_stop_signal)
        for sig in stop_signals()
        if sig != hangup or signal.getsignal(sig) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
