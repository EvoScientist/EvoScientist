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
- ``detail``: an object on ``done`` events, e.g. ``{"source": "system", ...}``.
- ``code``: a stable error code on ``error`` events (see :data:`ERROR_CODES`).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol, TextIO

PROTOCOL = 1

ERROR_CODES = frozenset(
    {"download_failed", "checksum_mismatch", "probe_failed", "unsupported_platform"}
)


class StageError(Exception):
    """A stage failure carrying one of :data:`ERROR_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown setup error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message


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
    """Write each event as one JSON line and flush, so a reader sees it at once."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def __call__(self, event: dict[str, Any]) -> None:
        stream = self._stream or sys.stdout
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()


class ConsoleEmitter:
    """Render events as human-readable lines on the Rich console.

    ``running`` events with a progress value are shown only when the rounded
    percentage moves by at least 10 points, so a download does not print a
    line per chunk.
    """

    def __init__(self, console: Any) -> None:
        self._console = console
        self._last_shown: dict[str, int] = {}

    def __call__(self, event: dict[str, Any]) -> None:
        stage = event["stage"]
        status = event["status"]
        message = event.get("message", "")
        if status == "running":
            progress = event.get("progress")
            if progress is not None:
                pct = int(progress * 100)
                last = self._last_shown.get(stage)
                if last is not None and pct - last < 10 and pct < 100:
                    return
                self._last_shown[stage] = pct
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
