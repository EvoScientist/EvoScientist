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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

TERMINAL_STATUSES: Final = frozenset({"success", "error", "timeout", "interrupted"})
"""Aligned with langgraph_sdk.schema.RunStatus terminal values.

Cancel operations transition runs into ``interrupted`` (not ``cancelled``).
"""


@dataclass(frozen=True)
class AsyncTaskNotification:
    """A completed-async-task signal pushed by a watcher."""

    task_id: str
    agent_name: str
    status: str  # one of TERMINAL_STATUSES
    summary: str  # last AI message, truncated to ≤500 chars; "" if none
    received_at: str  # ISO-8601 UTC timestamp
    prompt: str = ""  # original task description sent to the sub-agent


# Thread-safe (watcher coro pushes from asyncio loop; CLI consumer also asyncio)
_notification_queue: queue.Queue[AsyncTaskNotification] = queue.Queue()

# Track active watcher tasks for clean shutdown
_active_watchers: set[asyncio.Task] = set()
# Map thread_id → current watcher Task (supports replacement on update_async_task)
_watcher_by_thread: dict[str, asyncio.Task] = {}

logger = logging.getLogger(__name__)


def _extract_summary(final_state: dict | None) -> str:
    """Pull last AI message content (string or Anthropic blocks) → ≤500 chars."""
    if not isinstance(final_state, dict):
        return ""
    msgs = final_state.get("messages") or []
    last_ai = next(
        (m for m in reversed(msgs) if isinstance(m, dict) and m.get("type") == "ai"),
        None,
    )
    if not last_ai:
        return ""
    content = last_ai.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict)
        )
    return str(content)[:500]


async def watch_run_and_notify(
    client, thread_id: str, run_id: str, agent_name: str, prompt: str = ""
) -> None:
    """Subscribe to a run's event stream; enqueue notification when it terminates."""
    final_values: dict | None = None
    stream_failed = False
    try:
        async for chunk in client.runs.join_stream(
            thread_id=thread_id, run_id=run_id, stream_mode="values"
        ):
            ev = getattr(chunk, "event", None)
            data = getattr(chunk, "data", None)
            if ev == "values" and isinstance(data, dict):
                final_values = data
    except Exception:
        stream_failed = True
        logger.warning("Watcher stream failed for task %s", thread_id, exc_info=True)

    # Determine terminal status — if stream errored, fall back to runs.get
    status = "success"
    if stream_failed:
        try:
            run = await client.runs.get(thread_id=thread_id, run_id=run_id)
            status = run.get("status", "error")
        except Exception:
            status = "error"

    notification = AsyncTaskNotification(
        task_id=thread_id,
        agent_name=agent_name,
        status=status,
        summary=_extract_summary(final_values),
        received_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        prompt=prompt,
    )
    _notification_queue.put(notification)
    logger.info(
        "Enqueued async notification: task=%s agent=%s status=%s",
        thread_id,
        agent_name,
        status,
    )


def spawn_watcher(
    client,
    thread_id: str,
    run_id: str,
    agent_name: str,
    prompt: str = "",
) -> asyncio.Task:
    """Spawn a watcher; if one already exists for this thread_id, cancel it first.

    Replacement semantics support ``update_async_task`` which creates a new run_id
    on the same thread_id — we want the new watcher to take over without the
    old (now obsolete) watcher firing a stale notification. Cancellation
    propagates ``CancelledError`` (a BaseException), which the watcher's
    ``except Exception:`` does NOT catch — so ``_notification_queue.put(...)``
    never executes for the cancelled watcher (no stale notification).

    Caller must already be in a running asyncio event loop.
    """
    old_task = _watcher_by_thread.get(thread_id)
    if old_task is not None and not old_task.done():
        old_task.cancel()

    task = asyncio.create_task(
        watch_run_and_notify(client, thread_id, run_id, agent_name, prompt)
    )
    _watcher_by_thread[thread_id] = task

    def _cleanup(t: asyncio.Task) -> None:
        _active_watchers.discard(t)
        # Only remove from dict if THIS task is still the registered one
        # (could have been replaced by a newer spawn already)
        if _watcher_by_thread.get(thread_id) is t:
            del _watcher_by_thread[thread_id]

    task.add_done_callback(_cleanup)
    _active_watchers.add(task)
    return task


def drain_notifications() -> list[AsyncTaskNotification]:
    """Pull every pending notification off the queue (non-blocking).

    Returns a list of all notifications currently in the queue, emptying it.
    If the queue is empty, returns an empty list.
    """
    items: list[AsyncTaskNotification] = []
    while True:
        try:
            items.append(_notification_queue.get_nowait())
        except queue.Empty:
            return items


def dedup_notifications(
    notifs: list[AsyncTaskNotification],
    async_tasks: dict[str, dict] | None,
) -> list[AsyncTaskNotification]:
    """Filter notifications the agent has already 'seen' via prior check.

    Logic: skip a notification if `async_tasks[task_id]` exists with a TERMINAL
    status and `last_checked_at >= last_updated_at` (timestamps are ISO-8601
    so lexicographic comparison is correct). Also skip if `last_checked_at`
    is empty (brand-new task where agent hasn't checked yet).
    """
    if not async_tasks:
        return notifs
    survivors: list[AsyncTaskNotification] = []
    for n in notifs:
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


def format_notification_lines(
    notifs: list[AsyncTaskNotification],
) -> list[tuple[str, str]]:
    """Render notifications as compact tool-result-style lines for screen display.

    Returns a list of (text, rich_style) tuples — one per notification.
    Used by both Rich CLI (console.print) and TUI (_append_system).
    The LLM still receives the full format_batch_message text; this is
    purely a visual representation for the human operator.
    """
    if not notifs:
        return []
    # Open-right compact frame: short symmetric dashes around the title.
    # Bottom matches the top's width so the visual is balanced.
    #   ╭── ✦ Agent Teams ✦ ──
    #        ✔ writing  Task: ...  success
    #   ╰─────────────────────
    title = " ✦ Agent Teams ✦ "
    top_divider = "╭──" + title + "────"  # 4 dashes on the right (2x of left)
    bottom_divider = "╰" + "─" * (len(top_divider) - 1)
    lines: list[tuple[str, str]] = [(top_divider, "dim")]
    for n in notifs:
        # Strip the "-agent" suffix so it doesn't redundantly echo the header.
        # `writing-agent` → `writing`, `data-analysis-agent` → `data-analysis`.
        name = n.agent_name.removesuffix("-agent")
        if n.status == "success":
            icon, color = "✔", "#e67e22"  # carrot orange (CSS hex; Rich+Textual)
        elif n.status == "error":
            icon, color = "✗", "red"
        else:  # cancelled, timeout, interrupted
            icon, color = "⚠", "yellow"
        # Body format (5-space indent under "Agent:" header):
        #   ✔ writing             Task: <prompt preview>  success
        # Collapse newlines, truncate prompt to 60 chars.
        prompt_preview = (n.prompt or "").replace("\n", " ").strip()
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:60] + "…"
        if prompt_preview:
            text = f"     {icon} {name:18s}  Task: {prompt_preview}  {n.status}"
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
                {"agent": n.agent_name, "status": n.status, "task_id": n.task_id},
                ensure_ascii=False,
            )
        )
    lines.append("(Use check_async_task(task_id=...) to fetch full results.)")
    return "\n".join(lines)


# Brief grace window after the last drain: catch one final burst of arrivals
NOTIFICATION_BATCH_GRACE_SECONDS = 0.3
# Max time we'll wait for in-flight watchers to settle before triggering the
# agent turn — bounds latency for long-running tasks while still batching
# co-completing ones.
NOTIFICATION_ACTIVE_WATCHER_WAIT_SECONDS = 3.0


async def consume_notifications(
    run_message: Callable[[str, list[AsyncTaskNotification]], Awaitable[None]],
    read_async_tasks_state: Callable[[], Awaitable[dict[str, dict]]],
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
    """
    notifs = drain_notifications()
    if not notifs:
        return
    # Adaptive grace: if other watchers are still in flight, wait briefly for
    # them to settle so co-completing tasks batch into a single agent turn.
    deadline = (
        asyncio.get_event_loop().time() + NOTIFICATION_ACTIVE_WATCHER_WAIT_SECONDS
    )
    while _active_watchers and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.2)
        notifs.extend(drain_notifications())
    # Final brief grace to catch arrivals enqueued just before this tick
    await asyncio.sleep(NOTIFICATION_BATCH_GRACE_SECONDS)
    notifs.extend(drain_notifications())

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
