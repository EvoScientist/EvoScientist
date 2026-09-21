"""Pure, GUI-free helpers for safe desktop shutdown (issue #484 Part 5).

Answers two questions the pywebview ``closing`` handler needs — "is the backend
busy?" and "should we confirm before closing?" — without importing pywebview or
requiring a live server, so both unit-test in isolation. The GUI glue (the
confirmation dialog and the close-event wiring) lives in
:mod:`EvoScientist.desktop.shell`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("EvoScientist.desktop")


def should_confirm_close(backend_started: bool, has_active: bool) -> bool:
    """Whether closing should prompt the user first.

    Only when this app OWNS the backend (``backend_started``) AND it is busy:
    closing then tears the backend down and kills its in-flight runs. When the
    backend was reused (``backend_started`` is False), our close does not stop
    it, so its runs survive and there is nothing to confirm.
    """
    return bool(backend_started and has_active)


def backend_has_active_runs(url: str) -> bool:
    """Return True if the backend at ``url`` has any busy thread.

    A "busy" thread is one with a pending or running run, which includes
    background sub-agents (they run on their own threads). Answered with a single
    ``threads.search(status="busy", limit=1)`` round-trip.

    Fails OPEN (returns False) when the backend is unreachable or the probe
    errors: an unreachable backend has no reachable runs to protect, and a close
    must never hang or be vetoed by a flaky probe.
    """
    from ..langgraph_dev.manager import is_langgraph_dev_running
    from ..langgraph_dev.sdk import get_langgraph_sync_client

    if not is_langgraph_dev_running(base_url=url):
        return False
    try:
        client = get_langgraph_sync_client(url=url)
        return bool(client.threads.search(status="busy", limit=1))
    except Exception as exc:  # never trap the user in an unclosable window
        logger.warning("Active-run probe failed for %s: %s", url, exc)
        return False
