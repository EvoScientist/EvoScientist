"""Ask a running langgraph dev server whether it has work in flight.

Anything that is about to stop or restart a backend (closing an app, ``EvoSci
update``) asks first, because stopping the server kills its in-flight runs and
the background jobs it spawned. These helpers answer that question over HTTP,
with no UI and no dependency on a live server in tests.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Literal

logger = logging.getLogger(__name__)

ActiveState = Literal["idle", "active", "unknown"]


def _search_threads(url: str, payload: dict, *, timeout: float) -> list:
    """Raw ``POST /threads/search`` returning the matched threads (may be empty).

    A raw ``httpx`` call with a short timeout, NOT the langgraph SDK client: the
    SDK's client (``get_sync_client``) defaults to a 300s read timeout over a
    5-retry transport, which could stall the caller long after the user asked
    to stop. (The SDK is not exposed to proxies — it always passes an
    explicit transport, which disables httpx's env/registry proxy mounts.) A raw
    ``httpx.post`` DOES honour the system/registry proxy on Windows (present even
    with no ``*_PROXY`` env vars), and a VPN/corporate proxy would then swallow
    the loopback request, making a genuinely active backend look idle; hence
    ``trust_env=False``, exactly as ``is_langgraph_dev_running`` does for the
    health check. Raises on any transport/HTTP error (the caller decides how to
    treat "couldn't tell").
    """
    import httpx

    from ..langgraph_dev.sdk import langgraph_dev_headers

    resp = httpx.post(
        f"{url}/threads/search",
        json=payload,
        headers=langgraph_dev_headers(),
        timeout=timeout,
        trust_env=False,
    )
    resp.raise_for_status()
    return resp.json()


def _running_bg_processes(url: str, *, timeout: float) -> list[dict]:
    """Raw ``GET /api/bg_processes/running`` returning the running bg jobs (may be empty).

    Same raw-``httpx`` + ``trust_env=False`` rationale as :func:`_search_threads`
    (the SDK client would route this loopback call through a system/registry
    proxy on Windows and stall or swallow it). Each item is ``{process_id, name}``.
    Raises on any transport/HTTP error; the caller decides how to treat
    "couldn't tell".
    """
    import httpx

    from ..langgraph_dev.sdk import langgraph_dev_headers

    resp = httpx.get(
        f"{url}/api/bg_processes/running",
        headers=langgraph_dev_headers(),
        timeout=timeout,
        trust_env=False,
    )
    resp.raise_for_status()
    return resp.json().get("running", [])


def running_bg_process_names(url: str, *, timeout: float = 3.0) -> list[str]:
    """Names of the running background jobs, for a confirmation prompt to list.

    Best-effort display helper: returns ``[]`` on any error (a failed name
    lookup must never block a stop — the gating decision is made by
    :func:`_probe_active_state`, which treats the same error as ``"unknown"``).
    """
    try:
        return [
            p.get("name", "task") for p in _running_bg_processes(url, timeout=timeout)
        ]
    except Exception as exc:
        logger.warning(f"Listing running bg processes failed for {url}: {exc}")
        return []


def _probe_active_state(
    url: str, *, watched_thread_id: str | None = None, timeout: float = 3.0
) -> ActiveState:
    """Tri-state active-work probe for the backend at ``url``.

    Work is "active" when either:

    - ANY thread is ``busy`` — a run is executing (including background
      sub-agents, which run on their own threads). Busy work is finite and a
      restart kills it, so it counts regardless of which thread the user is on.
    - the WATCHED thread (``watched_thread_id``, the one the user has open) is
      ``interrupted`` — a turn paused awaiting human input (ask_user / HITL).
      Interrupted threads are checkpointed to ``.langgraph_api`` and survive a
      restart (resumable, not lost), and they accumulate, so a stale interrupted
      thread the user is NOT watching must not count; only the one they are
      actively deciding does. With no ``watched_thread_id`` the interrupted check
      is skipped entirely.
    - ANY background process (``run_in_background``) is still running. These are
      children of the langgraph dev server, so a restart tree-kills them, and
      unlike interrupted turns they are not checkpointed — losing them loses the
      work. So they count regardless of thread.

    The run checks are each a ``POST /threads/search`` (limit 1); the endpoint
    filters one ``status`` at a time and ANDs an ``ids`` filter, so the watched
    check is ``{status: "interrupted", ids: [watched], limit: 1}``. The
    background-job check is ``GET /api/bg_processes/running``.

    Returns:
        - ``"idle"`` — backend unreachable (its runs are already gone), or every
          check ran and none matched.
        - ``"active"`` — a check confirmed busy work or a watched interrupt.
        - ``"unknown"`` — the backend is up but a check errored (timeout,
          refused, bad response) and no other check confirmed activity, so the
          caller cannot tell whether work is in flight.

    The two public callers collapse ``"unknown"`` differently, which is the whole
    point of the tri-state: :func:`backend_has_active_runs` folds it to *not
    active* so a flaky probe never blocks a stop the user asked for;
    :func:`wait_for_backend_idle` folds it to *keep waiting* so a transient error
    never ends the wait and kills a live run.
    """
    from ..langgraph_dev.manager import is_langgraph_dev_running

    if not is_langgraph_dev_running(base_url=url):
        return "idle"
    saw_unknown = False
    # Any busy thread anywhere counts (finite executing work a restart kills).
    try:
        if _search_threads(url, {"status": "busy", "limit": 1}, timeout=timeout):
            return "active"
    except Exception as exc:
        logger.warning(f"Active-work probe (busy) failed for {url}: {exc}")
        saw_unknown = True
    # An interrupted thread counts only when it is the one the user is watching.
    if watched_thread_id:
        try:
            if _search_threads(
                url,
                {"status": "interrupted", "ids": [watched_thread_id], "limit": 1},
                timeout=timeout,
            ):
                return "active"
        except Exception as exc:
            logger.warning(
                f"Active-work probe (interrupted {watched_thread_id}) failed "
                f"for {url}: {exc}"
            )
            saw_unknown = True
    # A running background job counts too (killed by the restart, not resumable).
    try:
        if _running_bg_processes(url, timeout=timeout):
            return "active"
    except Exception as exc:
        logger.warning(f"Active-work probe (bg processes) failed for {url}: {exc}")
        saw_unknown = True
    return "unknown" if saw_unknown else "idle"


def backend_has_active_runs(
    url: str, *, watched_thread_id: str | None = None, timeout: float = 3.0
) -> bool:
    """Return True if the backend at ``url`` has active work.

    Active means any thread ``busy``, the WATCHED thread (``watched_thread_id``)
    ``interrupted``, or any background job still running. So a caller about to
    stop the backend asks when stopping would kill a running turn, abandon the
    interrupt the user is on, or stop a background job — but not for a stale
    interrupted thread they are not looking at.

    Fails OPEN (returns False) when the backend is unreachable or the probe
    errors: an unreachable backend has no reachable runs to protect, and a stop
    must never hang or be vetoed by a flaky probe. Thin bool view over
    :func:`_probe_active_state` — only a confirmed ``"active"`` counts.
    """
    return (
        _probe_active_state(url, watched_thread_id=watched_thread_id, timeout=timeout)
        == "active"
    )


def wait_for_backend_idle(
    url: str,
    *,
    probe: Callable[[str], ActiveState] = _probe_active_state,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 1.0,
    should_cancel: Callable[[], bool] | None = None,
    should_proceed_now: Callable[[], bool] | None = None,
) -> bool:
    """Block until the backend at ``url`` has no active work, then return True.

    Waits INDEFINITELY, for a caller that restarts the backend only once all
    active work — runs, background sub-agents, background jobs, and turns
    paused awaiting human input — has finished. Proceeds only on a
    CONFIRMED-idle state; an ``"unknown"`` result (backend up but the probe
    errored) is treated as still active, so a transient blip never ends the
    wait early and kills a live run.

    Two callbacks are polled each iteration (cancel takes precedence):

    - ``should_cancel`` True -> abort, return False, restart nothing (the user
      gave up on the restart, or the caller is shutting down).
    - ``should_proceed_now`` True -> stop waiting and return True even if work is
      still active (the user chose to kill the running work and restart now).

    Returns True once the backend is idle (or the user forced it), False if
    cancelled.
    """
    while True:
        if should_cancel is not None and should_cancel():
            return False
        if should_proceed_now is not None and should_proceed_now():
            return True
        if probe(url) == "idle":
            return True
        sleep(poll_interval)
