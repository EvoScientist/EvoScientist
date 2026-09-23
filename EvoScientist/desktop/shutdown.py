"""Pure, GUI-free helpers for safe desktop shutdown (issue #484 Part 5).

Answers two questions the pywebview ``closing`` handler needs — "is the backend
busy?" and "should we confirm before closing?" — without importing pywebview or
requiring a live server, so both unit-test in isolation. The GUI glue (the
confirmation dialog and the close-event wiring) lives in
:mod:`EvoScientist.desktop.shell`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Literal

logger = logging.getLogger("EvoScientist.desktop")

BusyState = Literal["idle", "busy", "unknown"]


def should_confirm_close(backend_started: bool, has_active: bool) -> bool:
    """Whether closing should prompt the user first.

    Only when this app OWNS the backend (``backend_started``) AND it is busy:
    closing then tears the backend down and kills its in-flight runs. When the
    backend was reused (``backend_started`` is False), our close does not stop
    it, so its runs survive and there is nothing to confirm.
    """
    return bool(backend_started and has_active)


def _probe_busy_state(url: str, *, timeout: float = 3.0) -> BusyState:
    """Tri-state active-run probe for the backend at ``url``.

    A "busy" thread is one with a pending or running run, which includes
    background sub-agents (they run on their own threads). Answered with a single
    ``POST /threads/search`` for ``status="busy"`` (limit 1).

    The probe hits the endpoint with a raw ``httpx`` call using
    ``trust_env=False`` and a short timeout, NOT the langgraph SDK client. The
    SDK's client (``get_sync_client``) builds its ``httpx.Client`` without
    ``trust_env=False`` and with a 300s read timeout, so on Windows it routes
    this 127.0.0.1 call through the system/registry proxy (present even with no
    ``*_PROXY`` env vars) — a VPN/corporate proxy then swallows the loopback
    request, making a genuinely busy backend look idle (and can stall the GUI
    thread for the long timeout). ``trust_env=False`` bypasses the proxy, exactly
    as ``is_langgraph_dev_running`` does for the health check.

    Returns:
        - ``"idle"``  — backend unreachable (its runs are already gone), or a
          successful probe found no busy threads.
        - ``"busy"``  — a successful probe found at least one busy thread.
        - ``"unknown"`` — the backend is up but the probe request itself errored
          (timeout, refused, bad response); the caller cannot tell whether work
          is in flight.

    The two callers collapse this differently, which is the whole point of the
    tri-state: :func:`backend_has_active_runs` (close path) folds ``"unknown"``
    to *not busy* so a flaky probe never traps the user in an unclosable window;
    :func:`wait_for_backend_idle` (switch path) folds ``"unknown"`` to *keep
    waiting* so a transient error never ends the wait and kills a live run.
    """
    import httpx

    from ..langgraph_dev.manager import is_langgraph_dev_running
    from ..langgraph_dev.sdk import langgraph_dev_headers

    if not is_langgraph_dev_running(base_url=url):
        return "idle"
    try:
        resp = httpx.post(
            f"{url}/threads/search",
            json={"status": "busy", "limit": 1},
            headers=langgraph_dev_headers(),
            timeout=timeout,
            trust_env=False,
        )
        resp.raise_for_status()
        return "busy" if resp.json() else "idle"
    except Exception as exc:
        logger.warning("Active-run probe failed for %s: %s", url, exc)
        return "unknown"


def backend_has_active_runs(url: str, *, timeout: float = 3.0) -> bool:
    """Return True if the backend at ``url`` has any busy thread.

    Fails OPEN (returns False) when the backend is unreachable or the probe
    errors: an unreachable backend has no reachable runs to protect, and a close
    must never hang or be vetoed by a flaky probe. Thin bool view over
    :func:`_probe_busy_state` — only a confirmed ``"busy"`` counts.
    """
    return _probe_busy_state(url, timeout=timeout) == "busy"


def wait_for_backend_idle(
    url: str,
    *,
    probe: Callable[[str], BusyState] = _probe_busy_state,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 1.0,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """Block until the backend at ``url`` has no active runs, then return True.

    Waits INDEFINITELY (issue #484 Part 6: a workspace switch restarts the
    app-owned backend only once all active runs, including background
    sub-agents, have finished). Proceeds only on a CONFIRMED-idle state; an
    ``"unknown"`` result (backend up but the probe errored) is treated as still
    busy, so a transient blip never ends the wait early and kills a live run.

    ``should_cancel`` is polled each iteration; when it returns True the wait
    aborts and returns False without restarting anything (the desktop wires this
    to the window-close event so quitting mid-wait does not relaunch a backend on
    an app that is shutting down). Returns True once the backend is idle.
    """
    while True:
        if should_cancel is not None and should_cancel():
            return False
        if probe(url) == "idle":
            return True
        sleep(poll_interval)
