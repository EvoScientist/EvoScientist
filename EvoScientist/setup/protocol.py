"""The ``EvoSci setup`` event protocol.

Every stage reports through an emitter as a sequence of events. With ``--json``
each event is one JSON object per line on stdout, which the desktop app reads
to show installation progress. The field names are a contract: later changes
may add fields, never rename or remove them, and ``PROTOCOL`` is bumped only for
a breaking change.

Event fields:

- ``protocol``: always :data:`PROTOCOL`.
- ``stage``: the stage id, e.g. ``"node"``.
- ``status``: ``running``, ``done``, ``skipped`` or ``error``.
- ``progress``: a float in ``[0, 1]`` on ``running`` events, when known.
- ``message``: a human-readable line.
- ``detail``: an object on ``done`` and ``skipped`` events, e.g.
  ``{"source": "system", ...}``.
- ``code``: a stable error code on ``error`` events (see :data:`ERROR_CODES`).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TextIO, TypeVar

T = TypeVar("T")

PROTOCOL = 1

# ``download_failed``: a download EvoScientist makes itself, or its archive
# content; ``install_failed``: the local install step (file system, venv, pip).
# A pip that cannot reach its index is ``install_failed`` too: its failure does
# not tell network errors from others.
ERROR_CODES = frozenset(
    {
        "download_failed",
        "checksum_mismatch",
        "probe_failed",
        "unsupported_platform",
        "install_failed",
    }
)


class StageError(Exception):
    """A stage failure carrying one of :data:`ERROR_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown setup error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StageResult:
    """How a stage ended without an error: ``done`` when it provided the
    dependency, ``skipped`` when there was nothing to do. The runner turns it
    into the stage's one terminal event."""

    message: str
    detail: dict[str, Any]
    status: Literal["done", "skipped"] = "done"


def make_event(
    stage: str,
    status: str,
    *,
    progress: float | None = None,
    message: str | None = None,
    detail: dict[str, Any] | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build one protocol event, leaving out fields that are not set."""
    event: dict[str, Any] = {"protocol": PROTOCOL, "stage": stage, "status": status}
    if progress is not None:
        event["progress"] = round(min(max(progress, 0.0), 1.0), 3)
    if message is not None:
        event["message"] = message
    if detail is not None:
        event["detail"] = detail
    if code is not None:
        event["code"] = code
    return event


class Emitter(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...


class JsonEmitter:
    """Write each event as one JSON line and flush, so a reader sees it at once.

    Lines are pure ASCII (non-ASCII as ``\\uXXXX`` escapes): a piped stdout on
    Windows uses the ANSI code page, which cannot encode every path.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def __call__(self, event: dict[str, Any]) -> None:
        stream = self._stream or sys.stdout
        stream.write(json.dumps(event) + "\n")
        stream.flush()


# How often a silent step repeats its last ``running`` event, so a reader
# (the desktop app) can tell a slow step from a stuck one.
HEARTBEAT_INTERVAL = 5.0


class StepStalled(Exception):
    """A silent step showed no activity for its whole silence limit."""


def wait_with_heartbeat(
    poll: Callable[[float], T | None],
    beat: Callable[[], None],
    activity: Callable[[], object],
    *,
    silence_limit: float,
    interval: float = HEARTBEAT_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Wait for a step that reports nothing, with heartbeats and a silence limit.

    ``poll(timeout)`` waits up to ``timeout`` seconds and returns the step's
    result, or None while it still runs. After each empty poll ``beat()`` runs
    (re-emit the last ``running`` event) and ``activity()`` is sampled: a value
    that changes counts as progress (for example a growing directory).
    Heartbeats themselves never count. Raises :class:`StepStalled` when the
    activity value has not changed for ``silence_limit`` seconds; the caller
    stops the step and maps it to a stage error.
    """
    last = activity()
    changed_at = clock()
    while True:
        result = poll(interval)
        if result is not None:
            return result
        beat()
        current = activity()
        if current != last:
            last, changed_at = current, clock()
        elif clock() - changed_at >= silence_limit:
            raise StepStalled(f"no progress for {silence_limit:g} s")


class ProgressThrottle:
    """Pick the progress updates a human should see.

    An update is shown when its message differs from the last one shown for
    the stage, or when the whole-number percentage has moved by at least 10
    points, so a download does not print a line per chunk but every step
    still appears.
    """

    def __init__(self) -> None:
        self._last_shown: dict[str, tuple[int, str]] = {}

    def should_show(self, stage: str, pct: int, message: str) -> bool:
        last = self._last_shown.get(stage)
        if last is not None and message == last[1] and pct - last[0] < 10 and pct < 100:
            return False
        self._last_shown[stage] = (pct, message)
        return True


class ConsoleEmitter:
    """Render events as human-readable lines on the Rich console, with
    ``running`` progress thinned by :class:`ProgressThrottle`."""

    def __init__(self, console: Any) -> None:
        self._console = console
        self._throttle = ProgressThrottle()

    def __call__(self, event: dict[str, Any]) -> None:
        from rich.markup import escape

        # Messages carry paths and exception text: unescaped, `[/x]` raises
        # MarkupError and a Windows `\[` loses its backslash.
        stage = escape(event["stage"])
        status = event["status"]
        message = escape(event.get("message", ""))
        if status == "running":
            progress = event.get("progress")
            if progress is not None:
                pct = int(progress * 100)
                if not self._throttle.should_show(stage, pct, message):
                    return
                self._console.print(f"  [dim]{stage}: {message} ({pct}%)[/dim]")
            else:
                self._console.print(f"  [dim]{stage}: {message}[/dim]")
        elif status == "done":
            self._console.print(f"  [green]✓ {stage}: {message}[/green]")
        elif status == "skipped":
            self._console.print(f"  [dim]- {stage}: {message}[/dim]")
        else:
            self._console.print(
                f"  [red]✗ {stage}: {message} ({event.get('code')})[/red]"
            )
