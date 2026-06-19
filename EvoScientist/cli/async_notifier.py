"""Async sub-agent auto-notification.

When a sub-agent on langgraph dev reaches a terminal state, a watcher coroutine
pushes a lightweight notification onto a thread-safe queue. The CLI loop drains
the queue, dedups against deepagents' async_tasks state, batches survivors,
and injects a synthetic user message that triggers one LLM turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, TypeAlias, TypedDict

if TYPE_CHECKING:
    from ..gateway import GraphGateway, GraphTarget

TERMINAL_STATUSES: Final = frozenset({"success", "error", "timeout", "interrupted"})
"""Aligned with langgraph_sdk.schema.RunStatus terminal values.

Cancel operations transition runs into ``interrupted`` (not ``cancelled``).
"""

# How many times the watcher will re-join the SSE stream when it closes
# cleanly but ``runs.get`` reports the run is still alive (typical cause:
# HTTP keep-alive timeout on long static periods). Bounded to prevent an
# unbounded loop if the server permanently misreports status.
_MAX_RECONNECT_ATTEMPTS: Final = 10


class AsyncTaskState(TypedDict, total=False):
    status: str
    last_checked_at: str
    last_updated_at: str


AsyncTasksState: TypeAlias = dict[str, AsyncTaskState]


@dataclass(frozen=True)
class AsyncTaskNotification:
    """A completed-async-task signal pushed by a watcher."""

    task_id: str
    agent_name: str
    status: str  # one of TERMINAL_STATUSES
    received_at: str  # ISO-8601 UTC timestamp
    prompt: str = ""  # original task description sent to the sub-agent
    kind: str = "agent"  # "agent" (sub-agent) | "bg-process" (background shell)
    # The CLI/main-agent thread_id under which the watcher was spawned. Used
    # to route the notification back to the originating CLI session so a
    # /new between launch and completion does not inject the synthetic
    # message into an unrelated thread (where ``check_async_task`` cannot
    # find the task_id). ``None`` means "unrouted" — the notification
    # drains for any current_thread_id (back-compat for direct callers).
    origin_cli_thread_id: str | None = None


# Per-thread routing: notifications with ``origin_cli_thread_id`` land in
# the matching sub-queue. Notifications without one go to ``_unrouted_queue``
# and drain regardless of current thread (back-compat for legacy callers
# and direct-put test paths).
_notifications_by_thread: dict[str, queue.Queue[AsyncTaskNotification]] = {}
_notifications_lock = threading.Lock()
_unrouted_queue: queue.Queue[AsyncTaskNotification] = queue.Queue()
# Public alias for the unrouted bucket — preserved so legacy tests and any
# external direct callers that did ``_notification_queue.put(...)`` keep
# working unchanged. New code should call ``_enqueue`` instead.
_notification_queue = _unrouted_queue

# Track active watcher tasks/futures for clean shutdown.
# dict[handle, origin_cli_thread_id] so the consumer's batching grace loop
# can filter for watchers tied to the current CLI thread (or unrouted)
# without being delayed by sibling-thread watchers.
_active_watchers: dict[object, str | None] = {}
# Map thread_id (sub-agent thread) → current watcher handle (supports
# replacement on update_async_task).
_watcher_by_thread: dict[str, asyncio.Task[None]] = {}


def _has_relevant_active_watchers(current_thread_id: str | None) -> bool:
    """Are there any in-flight watchers whose notifications would drain on
    a ``consume_notifications`` call for ``current_thread_id``?

    A watcher is relevant if its ``origin_cli_thread_id`` matches the
    current CLI thread or is ``None`` (unrouted bucket drains for any
    consumer). Sibling-thread watchers are ignored.
    """
    if current_thread_id is None:
        return bool(_active_watchers)
    return any(
        origin == current_thread_id or origin is None
        for origin in _active_watchers.values()
    )


logger = logging.getLogger(__name__)


def _enqueue(notification: AsyncTaskNotification) -> None:
    """Route a notification to its origin-thread queue or the unrouted bucket."""
    tid = notification.origin_cli_thread_id
    if not tid:
        _unrouted_queue.put(notification)
        return
    with _notifications_lock:
        q = _notifications_by_thread.get(tid)
        if q is None:
            q = queue.Queue()
            _notifications_by_thread[tid] = q
    q.put(notification)


def has_pending_notifications(current_thread_id: str | None = None) -> bool:
    """Cheap predicate for poller idle paths — true iff there's anything to consume.

    If ``current_thread_id`` is given, only the matching thread queue and
    the unrouted bucket count. With no argument, only the unrouted bucket
    counts (legacy behavior).
    """
    if not _unrouted_queue.empty():
        return True
    if current_thread_id is None:
        return False
    with _notifications_lock:
        q = _notifications_by_thread.get(current_thread_id)
    return q is not None and not q.empty()


def pending_thread_ids() -> set[str]:
    """Return the set of thread_ids with pending routed notifications."""
    with _notifications_lock:
        return {tid for tid, q in _notifications_by_thread.items() if not q.empty()}


async def read_async_tasks_from_gateway(
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
) -> AsyncTasksState:
    """Read async_tasks state through the active graph gateway."""
    try:
        values = await gateway.get_state_values(target, thread_id)
    except Exception:
        return {}
    return values.get("async_tasks", {})


async def watch_run_and_notify(
    client,
    thread_id: str,
    run_id: str,
    agent_name: str,
    prompt: str = "",
    origin_cli_thread_id: str | None = None,
) -> None:
    """Subscribe to a run's event stream; enqueue notification when it terminates.

    Status detection strategy (priority order):

      1. **In-band ``event="error"`` SSE part** — authoritative error signal
         from langgraph dev, no race against server-side state writeback.
      2. **Server-side state via ``runs.get``** — invoked after the stream
         closes (cleanly or with exception) to verify the run is actually
         done. Required because SSE long-poll can close on HTTP keep-alive
         timeout while the run is still running, which would otherwise be
         misread as ``"success"`` (observed in production with long-running
         literature search tasks under concurrency).
      3. **Re-join loop** — if ``runs.get`` reports ``pending`` / ``running``,
         the run is alive but we lost the stream; re-join up to
         ``_MAX_RECONNECT_ATTEMPTS`` times before giving up.

    The previous implementation trusted clean stream exits as success
    without any verification, which produced false-positive notifications
    when SSE keep-alive timeouts closed the stream early.

    Race-safety note: ``runs.get`` returning ``"error"`` immediately after
    a clean stream close can be a transient state for an actually-successful
    run (server hasn't finalized the writeback). We trust the absence of
    in-band error event over a stale ``runs.get="error"`` — see the
    ``status == "error" and not saw_error_event`` branch below.
    """
    for attempt in range(_MAX_RECONNECT_ATTEMPTS + 1):
        stream_failed = False
        saw_error_event = False
        try:
            async for chunk in client.runs.join_stream(
                thread_id=thread_id, run_id=run_id, stream_mode="values"
            ):
                ev = getattr(chunk, "event", None)
                data = getattr(chunk, "data", None)
                if ev == "error":
                    saw_error_event = True
                    logger.info(
                        "Watcher saw error event for task %s: %r", thread_id, data
                    )
        except Exception:
            stream_failed = True
            logger.warning(
                "Watcher stream failed for task %s", thread_id, exc_info=True
            )

        if saw_error_event:
            status = "error"
            break

        # Verify with server before deciding the run is done — clean stream
        # close does NOT guarantee terminal state.
        try:
            run = await client.runs.get(thread_id=thread_id, run_id=run_id)
            raw = run.get("status", "")
        except Exception:
            # Cannot verify terminal state. Defaulting to "success" here would
            # reintroduce the false-positive class this watcher exists to
            # prevent (clean stream + transient runs.get failure → unverified
            # success). Retry within the reconnect budget; on exhaustion drop
            # the notification rather than guess.
            if attempt >= _MAX_RECONNECT_ATTEMPTS:
                logger.warning(
                    "Watcher runs.get failed for task %s after %d reconnects; "
                    "unable to verify terminal state, skipping notification",
                    thread_id,
                    _MAX_RECONNECT_ATTEMPTS,
                    exc_info=True,
                )
                return
            logger.warning(
                "Watcher runs.get failed for task %s; retrying after backoff "
                "(attempt %d)",
                thread_id,
                attempt + 1,
                exc_info=True,
            )
            await asyncio.sleep(min(0.25 * (attempt + 1), 2.0))
            continue

        if raw not in TERMINAL_STATUSES:
            # Non-terminal status — includes the documented ``pending`` /
            # ``running`` values AND any future / unknown status the SDK may
            # introduce. Stream closed early but run is not done; re-join
            # unless we've exhausted attempts. Treating unknown statuses as
            # non-terminal is the safe default — better to retry once more
            # than to enqueue a false-positive on an unrecognized state.
            if attempt >= _MAX_RECONNECT_ATTEMPTS:
                logger.warning(
                    "Watcher gave up on task %s after %d reconnects "
                    "(server still reports %r); skipping notification",
                    thread_id,
                    _MAX_RECONNECT_ATTEMPTS,
                    raw,
                )
                return
            logger.info(
                "Watcher SSE closed for task %s but run reports %r; "
                "re-joining (attempt %d)",
                thread_id,
                raw,
                attempt + 1,
            )
            continue

        if raw == "error":
            # Race-safe interpretation: no in-band error event → trust the
            # absence over the server-side ``error`` (likely transient
            # writeback state for a successful run). Stream-failure path
            # is the one case where we DO trust ``error`` — the stream
            # blowing up usually means something genuinely went wrong.
            status = "error" if stream_failed else "success"
            break

        # success / timeout / interrupted — trust authoritative terminal status.
        status = raw
        break
    else:
        # Loop exhausted without a break — should be unreachable because the
        # re-join branch returns explicitly when attempts are exhausted, but
        # guard against future refactors.
        return

    notification = AsyncTaskNotification(
        task_id=thread_id,
        agent_name=agent_name,
        status=status,
        received_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        prompt=prompt,
        origin_cli_thread_id=origin_cli_thread_id,
    )
    _enqueue(notification)
    logger.info(
        "Enqueued async notification: task=%s agent=%s status=%s origin_thread=%s",
        thread_id,
        agent_name,
        status,
        origin_cli_thread_id or "<unrouted>",
    )


def spawn_watcher(
    client,
    thread_id: str,
    run_id: str,
    agent_name: str,
    prompt: str = "",
    origin_cli_thread_id: str | None = None,
) -> asyncio.Task[None]:
    """Spawn a watcher on the caller's asyncio loop.

    Replacement semantics support ``update_async_task`` which creates a new
    run_id on the same thread_id — we want the new watcher to take over
    without the old (now obsolete) watcher firing a stale notification.
    Cancellation propagates ``CancelledError`` (a BaseException), which the
    watcher's ``except Exception:`` does NOT catch — so ``_enqueue(...)``
    never executes for the cancelled watcher (no stale notification).

    ``origin_cli_thread_id`` tags the resulting notification so the consumer
    only injects it back into the originating CLI session.

    Caller must already be in a running asyncio event loop. Serve mode's
    ephemeral per-turn loop kills watchers spawned during a turn — that
    limitation is tracked separately.
    """
    old_task = _watcher_by_thread.get(thread_id)
    if old_task is not None and not old_task.done():
        old_task.cancel()

    task = asyncio.create_task(
        watch_run_and_notify(
            client,
            thread_id,
            run_id,
            agent_name,
            prompt,
            origin_cli_thread_id=origin_cli_thread_id,
        )
    )
    _watcher_by_thread[thread_id] = task
    _active_watchers[task] = origin_cli_thread_id

    def _cleanup(t: asyncio.Task[None]) -> None:
        _active_watchers.pop(t, None)
        # Only remove if THIS task is still the registered one — could
        # have been replaced by a newer spawn_watcher call already.
        if _watcher_by_thread.get(thread_id) is t:
            del _watcher_by_thread[thread_id]

    task.add_done_callback(_cleanup)
    return task


def _drain_one_queue(q: queue.Queue) -> list[AsyncTaskNotification]:
    items: list[AsyncTaskNotification] = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def drain_notifications(
    current_thread_id: str | None = None,
) -> list[AsyncTaskNotification]:
    """Pull pending notifications off the queue (non-blocking).

    With ``current_thread_id``: drains the matching per-thread queue plus
    the unrouted bucket. Without it: drains EVERY queue (legacy behavior;
    used by tests and diagnostics).
    """
    if current_thread_id is None:
        items: list[AsyncTaskNotification] = _drain_one_queue(_unrouted_queue)
        with _notifications_lock:
            queues = list(_notifications_by_thread.values())
        for q in queues:
            items.extend(_drain_one_queue(q))
        return items

    items = _drain_one_queue(_unrouted_queue)
    with _notifications_lock:
        q = _notifications_by_thread.get(current_thread_id)
    if q is not None:
        items.extend(_drain_one_queue(q))
    return items


def dedup_notifications(
    notifs: list[AsyncTaskNotification],
    async_tasks: AsyncTasksState | None,
) -> list[AsyncTaskNotification]:
    """Filter notifications the agent has already 'seen' via prior check.

    Logic: skip a notification if `async_tasks[task_id]` exists with a TERMINAL
    status and `last_checked_at >= last_updated_at` (timestamps are ISO-8601
    so lexicographic comparison is correct). Also skip if `last_checked_at`
    is empty (brand-new task where agent hasn't checked yet).
    """
    from .. import background  # cli -> core import; lazy to avoid import-order issues

    async_tasks = async_tasks or {}
    survivors: list[AsyncTaskNotification] = []
    for n in notifs:
        if n.kind == "bg-process":
            # Background process: skip if the launching session already inspected it
            # after it finished (check_process / list_processes) — mirrors the task
            # dedup below. Per-thread: another session's check doesn't suppress this.
            if background.was_observed_done(n.task_id, n.origin_cli_thread_id):
                logger.debug("Dedup: skipping shell notification for %s", n.task_id)
                continue
            survivors.append(n)
            continue
        task = async_tasks.get(n.task_id)
        if (
            task
            and task.get("status") in TERMINAL_STATUSES
            and task.get("last_checked_at", "") >= task.get("last_updated_at", "")
            and task.get("last_checked_at", "") != ""
        ):
            logger.debug(
                "Dedup: skipping notification for already-checked task %s", n.task_id
            )
            continue
        survivors.append(n)
    return survivors


def _render_notification_group(
    notifs: list[AsyncTaskNotification], title: str, label: str
) -> list[tuple[str, str]]:
    """Render one group of notifications inside a titled open-right frame.

    Open-right compact frame; bottom matches the top's width:
        ╭──  ✦ Agent Teams ✦  ────
             ✔ writing  Task: ...  success
        ╰─────────────────────────
    """
    top_divider = "╭──" + title + "────"  # 4 dashes on the right (2x of left)
    bottom_divider = "╰" + "─" * (len(top_divider) - 1)
    lines: list[tuple[str, str]] = [(top_divider, "dim")]
    for n in notifs:
        # `writing-agent` → `writing`.
        name = n.agent_name.removesuffix("-agent")
        if n.status == "success":
            icon, color = "✔", "#e67e22"  # carrot orange (CSS hex; Rich+Textual)
        elif n.status == "error":
            icon, color = "✗", "red"
        else:  # cancelled, timeout, interrupted
            icon, color = "⚠", "yellow"
        # Collapse newlines, truncate prompt/command preview to 60 chars.
        prompt_preview = (n.prompt or "").replace("\n", " ").strip()
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:60] + "…"
        if prompt_preview:
            text = f"     {icon} {name:18s}  {label}: {prompt_preview}  {n.status}"
        else:
            # Fallback: short task_id when no prompt is available
            short_tid = (
                f"{n.task_id[:8]}…{n.task_id[-4:]}"
                if len(n.task_id) > 12
                else n.task_id
            )
            text = f"     {icon} {name:18s}  ({short_tid})  {n.status}"
        lines.append((text, color))
    lines.append((bottom_divider, "dim"))
    return lines


def format_notification_lines(
    notifs: list[AsyncTaskNotification],
) -> list[tuple[str, str]]:
    """Render notifications as compact tool-result-style lines for screen display.

    Async sub-agents and background processes get SEPARATE titled frames so a shell
    background process is never mislabeled as an "Agent Team". Returns (text, rich_style)
    tuples. The LLM still receives the full ``format_batch_message`` text; this is purely
    the visual representation for the human operator.
    """
    if not notifs:
        return []
    tasks = [n for n in notifs if n.kind != "bg-process"]
    shell = [n for n in notifs if n.kind == "bg-process"]
    lines: list[tuple[str, str]] = []
    if tasks:
        lines += _render_notification_group(tasks, " ✦ Agent Teams ✦ ", "Task")
    if shell:
        lines += _render_notification_group(shell, " ✦ Background ✦ ", "Cmd")
    return lines


def format_batch_message(notifs: list[AsyncTaskNotification]) -> str:
    """Compose the synthetic user message that wakes the supervisor.

    Each task is rendered as a compact JSON object (one per line) so the LLM
    can reliably parse agent name, status, and task_id without ambiguity.
    ``ensure_ascii=False`` lets non-ASCII agent names pass through unchanged.
    Visual decoration lives in ``format_notification_lines``.
    """
    if not notifs:
        return ""
    lines = ["[Async tasks update]"]
    for n in notifs:
        lines.append(
            json.dumps(
                {
                    "agent": n.agent_name,
                    "kind": n.kind,
                    "status": n.status,
                    "task_id": n.task_id,
                },
                ensure_ascii=False,
            )
        )
    # bg-process is inspected with check_process; sub-agents with check_async_task.
    hints: list[str] = []
    if any(n.kind != "bg-process" for n in notifs):
        hints.append("check_async_task (sub-agents)")
    if any(n.kind == "bg-process" for n in notifs):
        hints.append("check_process (background processes)")
    lines.append(
        f"(Signal only — fetch full result via {' or '.join(hints)} if relevant to "
        "the current step, else acknowledge & continue.)"
    )
    return "\n".join(lines)


# Brief grace window after the last drain: catch one final burst of arrivals
NOTIFICATION_BATCH_GRACE_SECONDS = 0.3
# Max time we'll wait for in-flight watchers to settle before triggering the
# agent turn — bounds latency for long-running tasks while still batching
# co-completing ones.
NOTIFICATION_ACTIVE_WATCHER_WAIT_SECONDS = 3.0


async def consume_notifications(
    run_message: Callable[[str, list[AsyncTaskNotification]], Awaitable[None]],
    read_async_tasks_state: Callable[[], Awaitable[AsyncTasksState]],
    current_thread_id: str | None = None,
) -> None:
    """Drain queue, dedup, batch, and inject as a synthetic user message.

    Args:
        run_message: async callable receiving (llm_text, notifs_list).
            ``llm_text`` is the full structured message for the LLM
            (from ``format_batch_message``).  ``notifs_list`` is the
            survivors list so callers can render per-task visual lines
            without re-parsing the text.
        read_async_tasks_state: async callable returning current ``async_tasks``
                                from the agent's state for dedup.
        current_thread_id: the active CLI thread id. When given, only
            notifications whose ``origin_cli_thread_id`` matches (or that
            were enqueued unrouted) are drained — notifications belonging
            to other threads stay queued and naturally drain on the next
            poller tick after the user ``/resume``s back into them. When
            omitted (legacy callers / tests), every queue drains.
    """
    notifs = drain_notifications(current_thread_id)
    if not notifs:
        return
    # Adaptive grace: if other watchers tied to THIS thread (or unrouted) are
    # still in flight, wait briefly for them to settle so co-completing tasks
    # batch into a single agent turn. Sibling-thread watchers don't count —
    # their notifications wouldn't drain on this tick anyway.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + NOTIFICATION_ACTIVE_WATCHER_WAIT_SECONDS
    while _has_relevant_active_watchers(current_thread_id) and loop.time() < deadline:
        await asyncio.sleep(0.2)
        notifs.extend(drain_notifications(current_thread_id))
    # Final brief grace to catch arrivals enqueued just before this tick
    await asyncio.sleep(NOTIFICATION_BATCH_GRACE_SECONDS)
    notifs.extend(drain_notifications(current_thread_id))

    try:
        async_tasks = await read_async_tasks_state()
    except Exception:
        logger.warning("Failed to read async_tasks state for dedup", exc_info=True)
        async_tasks = {}

    survivors = dedup_notifications(notifs, async_tasks)
    if not survivors:
        logger.info(
            "All %d notifications deduped (already known to agent)", len(notifs)
        )
        return

    text = format_batch_message(survivors)
    await run_message(text, survivors)
