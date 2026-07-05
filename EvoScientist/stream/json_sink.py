"""Line-delimited JSON (JSONL) output sink for headless / programmatic use.

Selected by ``--output-format stream-json``. Serializes EvoScientist's native
event stream verbatim — one JSON object per line — onto a writable stream
(stdout in production). External clients (e.g. an agent daemon) read the stream
with a line scanner and ``json.loads`` per line, dispatching on each object's
``type`` field.

This module deliberately contains no rendering logic: it writes the normalized
event dicts produced by :meth:`EvoScientist.gateway.GraphGateway.stream_events`
verbatim — the same events the Rich renderer consumes — so it stays agnostic to
whether the run executes locally or against a langgraph server.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from ..gateway import GraphGateway, RunRequest


async def write_events_as_json(
    events: AsyncIterable[dict[str, Any]],
    out: TextIO,
) -> str:
    """Serialize an async stream of event dicts to ``out`` as JSONL.

    Each event is written as a single line and flushed immediately so consumers
    receive events in real time. ``default=str`` ensures an unexpected
    non-serializable value (e.g. a tool argument carrying a rich object) degrades
    to its string form instead of raising and tearing down the stream.

    Returns the final response text carried by the terminal ``done`` event
    (empty string if the stream ends without one).
    """
    final = ""
    async for event in events:
        out.write(json.dumps(event, default=str))
        out.write("\n")
        out.flush()
        if event.get("type") == "done":
            final = event.get("response", "") or ""
    return final


def redirect_console_to_stderr() -> None:
    """Route all human-facing Rich output to stderr.

    In stream-json mode stdout must carry only JSONL event lines, but the shared
    ``console`` singleton (used everywhere for status lines, separators, resume
    hints, error panels) writes to stdout by default. Reassigning its ``file``
    once moves every existing ``console.print`` call to stderr without touching
    individual call sites.
    """
    from .console import console

    console.file = sys.stderr


async def stream_json(
    gateway: GraphGateway,
    request: RunRequest,
    *,
    out: TextIO | None = None,
) -> str:
    """Run a graph turn through the gateway and emit its event stream as JSONL.

    Sources events from :meth:`GraphGateway.stream_events` — the same normalized
    event dicts the Rich renderer consumes — so it works for both local and
    langgraph-server execution. ``out`` defaults to stdout.
    """
    sink = out if out is not None else sys.stdout
    return await write_events_as_json(gateway.stream_events(request), sink)
