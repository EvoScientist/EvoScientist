"""Serve-side delivery watcher for committed proactive pushes.

The cron commits the proactive ``AIMessage`` into the source thread (server-side,
in langgraph-dev) and does NOT deliver it there — the channel origins live only in
the serve process's memory (``_thread_channel_origins``). This watcher runs in serve:
it notices committed proactive pushes and delivers them via
``publish_to_channel_origin`` (same pattern as serve's async-notifier drain). No
persisted outbox — delivery runs in the process that already holds the origins;
restart-recovery is out of scope for the MVP.

The core here is pure/DI'd (unit-testable without a server or bus): the serve loop
that polls periodically with the real SDK read + ``publish_to_channel_origin`` + a
persistent ``seen`` set is a thin wrapper wired at assembly.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .commit import read_proactive_tag

logger = logging.getLogger(__name__)


def extract_proactive_push(messages: list[Any]) -> tuple[str, str] | None:
    """Return ``(proactive_id, content)`` for the most recent proactive push.

    Scans from the newest message and returns the first one carrying the
    proactive-push tag (see ``commit.build_proactive_tag``) with a non-empty
    ``proactive_id`` and content. ``None`` when the thread carries no proactive
    push. Handles both ``BaseMessage`` objects and plain dicts (server state may
    return either).
    """
    for msg in reversed(messages or []):
        tag = read_proactive_tag(_additional_kwargs(msg))
        if tag is None:
            continue
        proactive_id = tag.get("proactive_id")
        content = _content(msg)
        if proactive_id and content:
            return str(proactive_id), content
    return None


async def deliver_pending_pushes(
    thread_ids: list[str],
    *,
    read_messages: Callable[[str], Awaitable[list[Any]]],
    publish: Callable[[str, str], bool],
    seen: set[str],
) -> list[str]:
    """Deliver each thread's undelivered proactive push; return delivered ids.

    For every thread, reads its messages (``read_messages``), finds the latest
    proactive push, and — if its ``proactive_id`` is not already in ``seen`` —
    delivers it via ``publish`` (``publish_to_channel_origin``). A ``proactive_id``
    is added to ``seen`` ONLY on a successful publish, so a failed publish (no
    origin yet, bus down) is retried on the next poll rather than lost. ``seen`` is
    the watcher's cross-poll idempotency set; the caller owns its lifetime.
    """
    delivered: list[str] = []
    for thread_id in thread_ids:
        push = extract_proactive_push(await read_messages(thread_id))
        if push is None:
            continue
        proactive_id, content = push
        if proactive_id in seen:
            continue
        if publish(thread_id, content):
            seen.add(proactive_id)
            delivered.append(proactive_id)
            logger.info(
                "[proactive delivery] pushed %s to thread %s",
                proactive_id,
                thread_id,
            )
        else:
            logger.warning(
                "[proactive delivery] publish returned False for thread %s "
                "(push %s) — no origin/bus; will retry next poll",
                thread_id,
                proactive_id,
            )
    return delivered


def _additional_kwargs(msg: Any) -> dict[str, Any]:
    kwargs = (
        getattr(msg, "additional_kwargs", None)
        if not isinstance(msg, dict)
        else msg.get("additional_kwargs")
    )
    return kwargs if isinstance(kwargs, dict) else {}


def _content(msg: Any) -> str | None:
    content = (
        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    )
    if isinstance(content, str) and content.strip():
        return content
    return None
