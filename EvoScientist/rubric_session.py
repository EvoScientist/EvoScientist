"""Session side of ``/rubric``: operations waiting for the next run, and the badge.

A command never writes thread state. It records an immutable operation here;
the next fresh run of that thread carries every unacknowledged operation in
its config, the grader applies them in order and remembers, per session, the
highest sequence number it applied; the client drops an operation once the
thread reports that number.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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
    """The ``configurable`` entry that marks a run and carries the thread's operations."""
    from .backends import RUBRIC_RUN_KEY

    return {RUBRIC_RUN_KEY: {"mode": mode, "changes": session.changes_for(thread_id)}}


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
