"""Proactive gate: decide whether a thread is worth a shadow turn.

Pure logic, no I/O and no model call. Every rejection carries a reason, so a
tick can record why a thread was skipped. The caller supplies a
:class:`ProactiveGateSettings` built from the config, which keeps the logic free
of ``EvoScientistConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ProactiveGateSettings:
    """Gate configuration for one tick."""

    idle_minutes: int
    # "HH:MM-HH:MM" (in ``timezone``, may wrap midnight); empty for none.
    quiet_hours: str
    # IANA zone name the quiet hours are evaluated in.
    timezone: str


@dataclass(frozen=True)
class GateDecision:
    should_run: bool
    reason: str


def evaluate_gate(
    settings: ProactiveGateSettings,
    *,
    now: datetime,
    armed: bool,
    has_origin: bool,
    in_flight: bool,
    last_activity: datetime | None,
    last_message_id: str | None,
    last_evaluated_id: str | None,
) -> GateDecision:
    """Return whether a shadow turn should run for one thread, with the reason.

    Checks run cheap-first, so the reason names the first one that fails:

    - ``not_armed``: the conversation is not armed with ``/proactive on``;
    - ``no_origin``: serve cannot deliver to it;
    - ``in_flight``: a run is active or the thread waits on an interrupt;
    - ``no_activity``: the thread has no message to reason about;
    - ``no_new_information``: its last non-proactive message is the one the
      previous evaluation saw;
    - ``idle_below_threshold``: it has not been quiet for long enough;
    - ``quiet_hours``: now falls inside the quiet-hours window.

    ``now`` and ``last_activity`` are compared in UTC (naive values count as
    UTC); quiet hours are evaluated in ``settings.timezone``.
    """
    if not armed:
        return GateDecision(False, "not_armed")
    if not has_origin:
        return GateDecision(False, "no_origin")
    if in_flight:
        return GateDecision(False, "in_flight")
    if last_activity is None or last_message_id is None:
        return GateDecision(False, "no_activity")
    if last_message_id == last_evaluated_id:
        return GateDecision(False, "no_new_information")
    now = _as_utc(now)
    idle_minutes = (now - _as_utc(last_activity)).total_seconds() / 60.0
    if idle_minutes < settings.idle_minutes:
        return GateDecision(False, "idle_below_threshold")
    if _in_quiet_hours(now, settings.quiet_hours, settings.timezone):
        return GateDecision(False, "quiet_hours")
    return GateDecision(True, "ok")


def _in_quiet_hours(now: datetime, quiet_hours: str, timezone: str) -> bool:
    window = _parse_window(quiet_hours)
    if window is None:
        return False
    start, end = window
    if start == end:
        # An empty window, not a 24-hour one.
        return False
    local = now.astimezone(ZoneInfo(timezone))
    minutes = local.hour * 60 + local.minute
    if start < end:
        return start <= minutes < end
    # The window wraps midnight, e.g. 22:00-08:00.
    return minutes >= start or minutes < end


def _parse_window(quiet_hours: str) -> tuple[int, int] | None:
    """``(start, end)`` in minutes for a normalised ``HH:MM-HH:MM``, else None."""
    if not quiet_hours:
        return None
    try:
        start, end = quiet_hours.split("-")
        return _to_minutes(start), _to_minutes(end)
    except ValueError:
        return None


def _to_minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
