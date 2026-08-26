"""Async sub-agent auto-notification.

Completions are detected from thread state by the client-side reader
(:func:`enqueue_completions_from_state`): it reads deepagents' ``async_tasks``
registry through the graph gateway, checks each active task's live run status,
and enqueues a lightweight notification onto a thread-safe queue. The CLI loop
drains the queue, dedups against ``async_tasks`` state, batches survivors, and
injects a synthetic user message that triggers one LLM turn. The reader runs at
every turn/stream-close boundary and, throttled, on the idle poll tick
(:func:`enqueue_completions_from_state_throttled`) so a completion still surfaces
while the user sits idle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
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


class AsyncTaskState(TypedDict, total=False):
    status: str
    last_checked_at: str
    last_updated_at: str
    # Registry fields carried by the deepagents ``async_tasks`` channel; the
    # state-based reader needs these to look up and label a task's live run.
    agent_name: str
    run_id: str


AsyncTasksState: TypeAlias = dict[str, AsyncTaskState]


@dataclass(frozen=True)
class AsyncTaskNotification:
    """A completed-async-task signal enqueued by the state reader."""

    task_id: str
    agent_name: str
    status: str  # one of TERMINAL_STATUSES
    received_at: str  # ISO-8601 UTC timestamp
    prompt: str = ""  # original task description sent to the sub-agent
    kind: str = "agent"  # "agent" (sub-agent) | "bg-process" (background shell)
    # The CLI/main-agent thread_id the task was launched under. Used
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


def enqueue_task_notification(notification: AsyncTaskNotification) -> None:
    """Route a completed-task notification onto the consumer queue.

    Thin wrapper over :func:`_enqueue` so the state reader can enqueue without
    reaching into the module's private symbols.
    """
    _enqueue(notification)


def enqueue_bg_process_notification(
    *,
    task_id: str,
    agent_name: str,
    status: str,
    prompt: str = "",
    origin_cli_thread_id: str | None = None,
) -> None:
    """Build and enqueue a background-process completion notification.

    Called by the ``bg_processes`` state reader
    (:func:`enqueue_bg_process_completions_from_state`) so it never constructs the
    ``AsyncTaskNotification`` itself — the ``kind="bg-process"`` tag and the UTC
    ``received_at`` timestamp are filled in here.
    """
    _enqueue(
        AsyncTaskNotification(
            task_id=task_id,
            agent_name=agent_name,
            status=status,
            received_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            prompt=prompt,
            kind="bg-process",
            origin_cli_thread_id=origin_cli_thread_id,
        )
    )


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


# Task ids the reader has already enqueued a completion for. The persisted
# ``async_tasks[*].status`` lags the live run (it only advances when the agent
# calls ``check_async_task``), so without this a fast reader would re-enqueue
# the same completion every poll until the agent checks. Combined with the
# terminal-in-state skip below, one completion yields exactly one enqueue.
_reader_enqueued_task_ids: set[str] = set()

# Idle-tick throttle state for ``enqueue_completions_from_state_throttled``:
# the last monotonic time the reader polled per thread_id, and whether that
# poll still saw an active (not-yet-terminal) task worth re-polling. The active
# flag is refreshed on EVERY reader call (turn-boundary and idle), so a freshly
# launched task re-arms idle polling and a fully-terminal registry lets idle
# ticks skip the state read entirely.
IDLE_READER_MIN_INTERVAL_SECONDS: Final = 3.0
_idle_reader_last_poll: dict[str | None, float] = {}
_idle_reader_active_seen: dict[str | None, bool] = {}


async def enqueue_completions_from_state(
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
) -> int:
    """Detect async-task completions from thread state and enqueue them.

    Read the ``async_tasks`` registry through the gateway and, for each task not
    already known terminal, ask the gateway for the live run status. Newly
    terminal tasks are enqueued onto the shared consumer queue, so
    ``consume_notifications`` handles dedup/batching/injection unchanged. Both
    reads go through the gateway, so it behaves identically on either backend.
    This is the sole async-task completion mechanism on both backends.

    Called at every turn/stream-close boundary and, throttled, on the idle poll
    tick (via :func:`enqueue_completions_from_state_throttled`). Best-effort
    throughout: a failed status read leaves the task for the next poll (treated
    as not-yet-terminal), mirroring ``read_async_tasks_from_gateway``.

    Returns the number of still-active tasks (non-terminal, or not yet
    pollable / transiently unread) seen this pass — used to arm/disarm the idle
    throttle so idle polling stops once every launched task is terminal.
    """
    registry = await read_async_tasks_from_gateway(gateway, target, thread_id)
    still_active = 0
    for task_id, task in registry.items():
        if task_id in _reader_enqueued_task_ids:
            continue
        if task.get("status") in TERMINAL_STATUSES:
            # Already terminal in state → the agent saw it via a check-tool
            # writeback; nothing for the reader to surface.
            continue
        run_id = task.get("run_id")
        if not run_id:
            # Mid-launch: no run to poll yet, but it may gain one — keep idle
            # polling armed so we don't stop before the run exists.
            still_active += 1
            continue
        # task_id == the sub-agent thread_id (deepagents keys the registry by it).
        try:
            status = await gateway.get_run_status(target, task_id, run_id)
        except Exception:
            still_active += 1  # server unavailable / transient — retry next poll
            continue
        if status not in TERMINAL_STATUSES:
            still_active += 1
            continue
        enqueue_task_notification(
            AsyncTaskNotification(
                task_id=task_id,
                agent_name=task.get("agent_name", ""),
                status=status,
                received_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                origin_cli_thread_id=thread_id,
            )
        )
        _reader_enqueued_task_ids.add(task_id)
    _idle_reader_active_seen[thread_id] = still_active > 0
    return still_active


async def _run_throttled_reader(
    reader,
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
    *,
    min_interval_s: float,
    last_poll: dict[str | None, float],
    active_seen: dict[str | None, bool],
) -> None:
    """Idle-tick throttle shared by the async-task and bg-process readers.

    Runs ``reader`` at most once per ``min_interval_s`` per thread, and skips
    entirely once the last poll saw nothing active — a turn-boundary reader call
    re-arms it. ``reader`` sets ``active_seen[thread_id]`` itself. This bounds the
    idle state read (an HTTP round-trip on the server backend) to zero when
    nothing is pending, and to one poll per interval while work is in flight.
    Default-arm (``active_seen.get(..., True)``) so a thread with no prior
    observation still polls once, then disarms itself.
    """
    now = time.monotonic()
    last = last_poll.get(thread_id)
    if last is not None and (now - last) < min_interval_s:
        return
    if not active_seen.get(thread_id, True):
        return
    last_poll[thread_id] = now
    await reader(gateway, target, thread_id)


async def enqueue_completions_from_state_throttled(
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
    *,
    min_interval_s: float = IDLE_READER_MIN_INTERVAL_SECONDS,
) -> None:
    """Idle-poll counterpart to :func:`enqueue_completions_from_state`."""
    await _run_throttled_reader(
        enqueue_completions_from_state,
        gateway,
        target,
        thread_id,
        min_interval_s=min_interval_s,
        last_poll=_idle_reader_last_poll,
        active_seen=_idle_reader_active_seen,
    )


# --- Background-process reader (mirror of the async-task reader above) --------
# Process ids the bg-process reader has already surfaced an exit for. The mirrored
# ``bg_processes[*].status`` only goes terminal once the agent observes the exit
# via a tool (check/stop/list), so this idempotency set keeps one proactive exit →
# one enqueue for the agent-didn't-check path, mirroring ``_reader_enqueued_task_ids``.
_reader_enqueued_process_ids: set[str] = set()
_bg_idle_reader_last_poll: dict[str | None, float] = {}
_bg_idle_reader_active_seen: dict[str | None, bool] = {}


async def read_bg_processes_from_gateway(
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
) -> dict[str, dict]:
    """Read the ``bg_processes`` state channel through the active graph gateway."""
    try:
        values = await gateway.get_state_values(target, thread_id)
    except Exception:
        return {}
    return values.get("bg_processes", {})


async def enqueue_bg_process_completions_from_state(
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
) -> int:
    """Detect background-process exits from thread state and enqueue them.

    Mirror of :func:`enqueue_completions_from_state` for OS background processes.
    Reads the ``bg_processes`` registry the middleware mirrors into state and, for
    each process not already surfaced and not terminal-in-state (a terminal state
    means the agent already observed the exit via check/stop/list — nothing to
    proactively surface, which is the state-based dedup), polls
    ``gateway.get_process_status``. Newly-terminal processes are enqueued as
    ``kind="bg-process"`` notifications via the shared queue. Returns the
    still-active count for the idle throttle.
    """
    registry = await read_bg_processes_from_gateway(gateway, target, thread_id)
    still_active = 0
    for process_id, record in registry.items():
        if process_id in _reader_enqueued_process_ids:
            continue
        if record.get("status") in TERMINAL_STATUSES:
            continue
        try:
            status = await gateway.get_process_status(target, thread_id, process_id)
        except Exception:
            still_active += 1  # gateway unavailable / transient — retry next poll
            continue
        if status == "unknown":
            # Process gone from the registry (e.g. a server restart cleared it) —
            # its exit is unknowable, so stop polling it rather than spin forever.
            _reader_enqueued_process_ids.add(process_id)
            continue
        if status not in TERMINAL_STATUSES:
            still_active += 1
            continue
        enqueue_bg_process_notification(
            task_id=process_id,
            agent_name=record.get("name", ""),
            status=status,
            prompt=record.get("command", ""),
            origin_cli_thread_id=record.get("origin_thread_id") or thread_id,
        )
        _reader_enqueued_process_ids.add(process_id)
    _bg_idle_reader_active_seen[thread_id] = still_active > 0
    return still_active


async def enqueue_bg_process_completions_from_state_throttled(
    gateway: GraphGateway,
    target: GraphTarget,
    thread_id: str,
    *,
    min_interval_s: float = IDLE_READER_MIN_INTERVAL_SECONDS,
) -> None:
    """Idle-poll counterpart to :func:`enqueue_bg_process_completions_from_state`."""
    await _run_throttled_reader(
        enqueue_bg_process_completions_from_state,
        gateway,
        target,
        thread_id,
        min_interval_s=min_interval_s,
        last_poll=_bg_idle_reader_last_poll,
        active_seen=_bg_idle_reader_active_seen,
    )


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
    tasks = [n for n in notifs if n.kind == "agent"]
    shell = [n for n in notifs if n.kind == "bg-process"]
    unknown = [n for n in notifs if n.kind not in {"agent", "bg-process"}]
    lines: list[tuple[str, str]] = []
    if tasks:
        lines += _render_notification_group(tasks, " ✦ Agent Teams ✦ ", "Task")
    if shell:
        lines += _render_notification_group(shell, " ✦ Background ✦ ", "Cmd")
    if unknown:
        # Fallback so a future kind is never silently dropped from the display.
        lines += _render_notification_group(unknown, " ✦ Updates ✦ ", "Task")
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
    if any(n.kind == "agent" for n in notifs):
        hints.append("check_async_task (sub-agents)")
    if any(n.kind == "bg-process" for n in notifs):
        hints.append("check_process (background processes)")
    # Fallback when a batch has only unrecognized kinds (hints empty).
    hint_text = " or ".join(hints) if hints else "the appropriate status tool"
    lines.append(
        f"(Signal only — fetch full result via {hint_text} if relevant to "
        "the current step, else acknowledge & continue.)"
    )
    return "\n".join(lines)


# Brief grace window after the last drain: catch one final burst of arrivals
NOTIFICATION_BATCH_GRACE_SECONDS = 0.3


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
    # Brief fixed grace to catch reader enqueues arriving just before this tick,
    # so co-completing tasks batch into a single agent turn.
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
