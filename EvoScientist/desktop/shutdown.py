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

ActiveState = Literal["idle", "active", "unknown"]

# Thread statuses that mean work is in flight and the app-owned backend must not
# be torn down under it. ``busy`` = a run is executing (includes background
# sub-agents, which run on their own threads); ``interrupted`` = a run is paused
# awaiting human input (ask_user / HITL) — still a live turn, not a finished one.
# (langgraph ThreadStatus is idle | busy | interrupted | error; ``error`` is
# terminal, ``idle`` is done.)
_ACTIVE_THREAD_STATUSES = ("busy", "interrupted")


def should_confirm_close(backend_started: bool, has_active: bool) -> bool:
    """Whether closing should prompt the user first.

    Only when this app OWNS the backend (``backend_started``) AND it is busy:
    closing then tears the backend down and kills its in-flight runs. When the
    backend was reused (``backend_started`` is False), our close does not stop
    it, so its runs survive and there is nothing to confirm.
    """
    return bool(backend_started and has_active)


def _probe_active_state(url: str, *, timeout: float = 3.0) -> ActiveState:
    """Tri-state active-work probe for the backend at ``url``.

    Work is "active" when any thread is ``busy`` (a run executing, including
    background sub-agents on their own threads) OR ``interrupted`` (a run paused
    awaiting human input — ask_user / HITL — which is still a live turn, not a
    finished one). Answered with a ``POST /threads/search`` (limit 1) per status
    in :data:`_ACTIVE_THREAD_STATUSES`; the endpoint filters one status at a
    time, so this is one short request each, short-circuiting on the first hit.

    Each request is a raw ``httpx`` call using ``trust_env=False`` and a short
    timeout, NOT the langgraph SDK client. The SDK's client (``get_sync_client``)
    builds its ``httpx.Client`` without ``trust_env=False`` and with a 300s read
    timeout, so on Windows it routes this 127.0.0.1 call through the
    system/registry proxy (present even with no ``*_PROXY`` env vars) — a
    VPN/corporate proxy then swallows the loopback request, making a genuinely
    active backend look idle (and can stall the GUI thread for the long timeout).
    ``trust_env=False`` bypasses the proxy, exactly as ``is_langgraph_dev_running``
    does for the health check.

    Returns:
        - ``"idle"`` — backend unreachable (its runs are already gone), or every
          status was probed successfully and none matched.
        - ``"active"`` — a successful probe found at least one busy/interrupted
          thread.
        - ``"unknown"`` — the backend is up but a probe request errored (timeout,
          refused, bad response) and no other probe confirmed activity, so the
          caller cannot tell whether work is in flight.

    The two callers collapse this differently, which is the whole point of the
    tri-state: :func:`backend_has_active_runs` (close path) folds ``"unknown"``
    to *not active* so a flaky probe never traps the user in an unclosable
    window; :func:`wait_for_backend_idle` (switch path) folds ``"unknown"`` to
    *keep waiting* so a transient error never ends the wait and kills a live run.
    """
    import httpx

    from ..langgraph_dev.manager import is_langgraph_dev_running
    from ..langgraph_dev.sdk import langgraph_dev_headers

    if not is_langgraph_dev_running(base_url=url):
        return "idle"
    saw_unknown = False
    for status in _ACTIVE_THREAD_STATUSES:
        try:
            resp = httpx.post(
                f"{url}/threads/search",
                json={"status": status, "limit": 1},
                headers=langgraph_dev_headers(),
                timeout=timeout,
                trust_env=False,
            )
            resp.raise_for_status()
            if resp.json():
                return "active"
        except Exception as exc:
            logger.warning("Active-work probe (%s) failed for %s: %s", status, url, exc)
            saw_unknown = True
    return "unknown" if saw_unknown else "idle"


def backend_has_active_runs(url: str, *, timeout: float = 3.0) -> bool:
    """Return True if the backend at ``url`` has any busy or interrupted thread.

    Fails OPEN (returns False) when the backend is unreachable or the probe
    errors: an unreachable backend has no reachable runs to protect, and a close
    must never hang or be vetoed by a flaky probe. Thin bool view over
    :func:`_probe_active_state` — only a confirmed ``"active"`` counts.
    """
    return _probe_active_state(url, timeout=timeout) == "active"


def wait_for_backend_idle(
    url: str,
    *,
    probe: Callable[[str], ActiveState] = _probe_active_state,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 1.0,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """Block until the backend at ``url`` has no active work, then return True.

    Waits INDEFINITELY (issue #484 Part 6: a workspace switch restarts the
    app-owned backend only once all active work — runs, background sub-agents,
    and turns paused awaiting human input — has finished). Proceeds only on a
    CONFIRMED-idle state; an ``"unknown"`` result (backend up but the probe
    errored) is treated as still active, so a transient blip never ends the wait
    early and kills a live run.

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
