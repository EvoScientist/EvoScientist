"""Proactive gate: decide whether a check should proceed.

Pure logic over the disabled flag, origin presence, an in-flight lock, the
idle-minutes threshold, and a quiet-hours window. Every rejection carries a
reason (logged); no model call happens before the gate returns
``should_run=True``. Kept decoupled from ``EvoScientistConfig`` — the caller
supplies a ``ProactiveGateSettings`` so the logic stays pure and testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from ..config.settings import _normalize_hhmm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProactiveGateSettings:
    """Static gate configuration for a proactive check."""

    enabled: bool
    idle_minutes: int
    quiet_hours: str | None  # "HH:MM-HH:MM" (local to ``timezone``) or None
    timezone: str  # IANA name, e.g. "America/New_York"


@dataclass(frozen=True)
class GateDecision:
    should_run: bool
    reason: str


def evaluate_gate(
    settings: ProactiveGateSettings,
    *,
    now: datetime,
    last_activity: datetime | None,
    has_origin: bool,
    in_flight: bool,
) -> GateDecision:
    """Return whether a proactive check should run, with a reason for the record.

    Checks are ordered cheap-first so the recorded reason names the first failing
    condition. ``now`` and ``last_activity`` are compared in UTC (naive values are
    treated as UTC); quiet hours are evaluated in ``settings.timezone``.
    """
    now = _as_utc(now)
    decision = _evaluate(
        settings,
        now=now,
        last_activity=last_activity,
        has_origin=has_origin,
        in_flight=in_flight,
    )
    logger.debug("proactive gate decision: %s", decision.reason)
    return decision


def _evaluate(
    settings: ProactiveGateSettings,
    *,
    now: datetime,
    last_activity: datetime | None,
    has_origin: bool,
    in_flight: bool,
) -> GateDecision:
    if not settings.enabled:
        return GateDecision(False, "disabled")
    if not has_origin:
        return GateDecision(False, "no_origin")
    if in_flight:
        return GateDecision(False, "in_flight")
    if last_activity is None:
        return GateDecision(False, "no_activity")
    idle_minutes = (now - _as_utc(last_activity)).total_seconds() / 60.0
    if idle_minutes < settings.idle_minutes:
        return GateDecision(False, "idle_below_threshold")
    if _in_quiet_hours(now, settings.quiet_hours, settings.timezone):
        return GateDecision(False, "quiet_hours")
    return GateDecision(True, "ok")


def _in_quiet_hours(now: datetime, quiet_hours: str | None, timezone: str) -> bool:
    window = _parse_window(quiet_hours)
    if window is None:
        return False
    start, end = window
    if start == end:
        # Degenerate window — treat as "never quiet" rather than "always".
        return False
    local = now.astimezone(ZoneInfo(timezone))
    minutes = local.hour * 60 + local.minute
    if start < end:
        return start <= minutes < end
    # Wraparound window, e.g. 22:00-08:00.
    return minutes >= start or minutes < end


def _parse_window(quiet_hours: str | None) -> tuple[int, int] | None:
    if not quiet_hours:
        return None
    parts = quiet_hours.split("-")
    if len(parts) != 2:
        return None
    start = _normalize_hhmm(parts[0])
    end = _normalize_hhmm(parts[1])
    if start is None or end is None:
        return None
    return _to_minutes(start), _to_minutes(end)


def _to_minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
