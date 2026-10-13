"""Session side of ``/rubric``: operations waiting for the next run, and the badge.

A command never writes thread state. It records an immutable operation here;
the run of the next message on that thread carries every unacknowledged operation in
its config, the grader applies them in order and remembers, per session, the
highest sequence number it applied; the client drops an operation once the
thread reports that number.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RubricSession:
    """Operations recorded by ``/rubric`` in this process, keyed by thread."""

    # The id and the counter share one object, so a reused id never restarts
    # the sequence (the grader would then skip the new operations).
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _seq: int = 0
    _pending: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, thread_id: str, **fields: str | None) -> dict[str, Any]:
        """Append one operation (``sticky=`` / ``next=``; ``None`` clears) for *thread_id*."""
        self._seq += 1
        operation: dict[str, Any] = {
            "session": self.session_id,
            "seq": self._seq,
            **fields,
        }
        self._pending.setdefault(thread_id, []).append(operation)
        return dict(operation)

    def changes_for(self, thread_id: str) -> list[dict[str, Any]]:
        """The thread's unacknowledged operations, oldest first (copies)."""
        return [dict(op) for op in self._pending.get(thread_id, [])]

    def acknowledge(self, thread_id: str, applied: Mapping[str, int] | None) -> None:
        """Drop the operations the thread reports as applied (``_rubric_applied``)."""
        watermark = (applied or {}).get(self.session_id, 0)
        remaining = [
            op for op in self._pending.get(thread_id, []) if op["seq"] > watermark
        ]
        if remaining:
            self._pending[thread_id] = remaining
        else:
            self._pending.pop(thread_id, None)


def rubric_configurable_extra(
    session: RubricSession, thread_id: str, mode: str
) -> dict[str, Any]:
    """The ``configurable`` entry that marks a run; a message's run carries the operations."""
    from .backends import RUBRIC_RUN_KEY

    # A notifier turn finishes the task it reports on, so changes wait for the next message.
    changes = session.changes_for(thread_id) if mode == "thread" else []
    return {RUBRIC_RUN_KEY: {"mode": mode, "changes": changes}}


def merge_configurable_extra(*parts: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Merge ``configurable_extra`` parts, skipping empty ones; ``None`` when nothing is left."""
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged or None


def values_for_badge(badge: str | None) -> dict[str, Any]:
    """Thread values that would show *badge*; used when the state cannot be read."""
    on = badge in ("on", "on+next")
    next_ = badge in ("next", "on+next")
    return {
        "_sticky_rubric": {"text": "", "id": ""} if on else None,
        "_next_rubric": {"text": "", "id": ""} if next_ else None,
    }


def rubric_badge(
    values: Mapping[str, Any] | None, pending: list[Mapping[str, Any]]
) -> str | None:
    """``on`` / ``next`` / ``on+next`` for the status bar, pending operations applied on top."""
    values = values or {}
    sticky = bool(values.get("_sticky_rubric"))
    next_ = bool(values.get("_next_rubric"))
    for operation in pending:
        if "sticky" in operation:
            sticky = bool(operation["sticky"])
        if "next" in operation:
            next_ = bool(operation["next"])
    if sticky and next_:
        return "on+next"
    if sticky:
        return "on"
    if next_:
        return "next"
    return None


# -- shared by the interactive CLI and the TUI ---------------------------------


def runs_in_process(graph_gateway: Any) -> bool:
    """Whether a session's runs execute in-process (the local gateway)."""
    from .gateway.server import LangGraphServerGateway

    execute_gateway = getattr(graph_gateway, "execute_gateway", graph_gateway)
    return not isinstance(execute_gateway, LangGraphServerGateway)


def thread_run_extra(
    session: RubricSession,
    graph_gateway: Any,
    *,
    thread_id: str | None,
    current_thread_id: str | None,
    mode: str | None,
) -> dict[str, Any] | None:
    """The run marker and pending operations, for the session's own thread only.

    A notifier turn for another thread runs with the current thread's
    workspace, and a run on the server gateway is not graded at all.
    """
    if (
        not mode
        or thread_id is None
        or thread_id != current_thread_id
        or not runs_in_process(graph_gateway)
    ):
        return None
    return rubric_configurable_extra(session, thread_id, mode)


async def acknowledge_applied(
    session: RubricSession,
    badges: dict[str, str | None],
    graph_gateway: Any,
    *,
    agent: Any,
    dirs: Any,
    thread_id: str | None,
) -> None:
    """Drop operations the thread applied; refresh its status-bar badge in *badges*."""
    if not thread_id:
        return
    # Without a readable state keep what this thread's badge last showed.
    values: dict[str, Any] = values_for_badge(badges.get(thread_id))
    if agent is not None:
        from .gateway import GraphTarget

        try:
            values = (
                await graph_gateway.get_state_values(
                    GraphTarget(local_graph=agent, **dirs.metadata()), thread_id
                )
                or {}
            )
        except Exception:
            logger.debug("rubric: thread state unavailable", exc_info=True)
    session.acknowledge(thread_id, values.get("_rubric_applied"))
    badges[thread_id] = rubric_badge(values, session.changes_for(thread_id))
