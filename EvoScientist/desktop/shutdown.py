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


def backend_has_active_runs(url: str, *, timeout: float = 3.0) -> bool:
    """Return True if the backend at ``url`` has any busy thread.

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

    Fails OPEN (returns False) when the backend is unreachable or the probe
    errors: an unreachable backend has no reachable runs to protect, and a close
    must never hang or be vetoed by a flaky probe.
    """
    import httpx

    from ..langgraph_dev.manager import is_langgraph_dev_running
    from ..langgraph_dev.sdk import langgraph_dev_headers

    if not is_langgraph_dev_running(base_url=url):
        return False
    try:
        resp = httpx.post(
            f"{url}/threads/search",
            json={"status": "busy", "limit": 1},
            headers=langgraph_dev_headers(),
            timeout=timeout,
            trust_env=False,
        )
        resp.raise_for_status()
        return bool(resp.json())
    except Exception as exc:  # never trap the user in an unclosable window
        logger.warning("Active-run probe failed for %s: %s", url, exc)
        return False
