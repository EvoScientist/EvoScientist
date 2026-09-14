"""Tests for EvoScientist.proactive.gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from EvoScientist.proactive.gate import (
    ProactiveGateSettings,
    evaluate_gate,
)


def _settings(**over):
    base = {
        "enabled": True,
        "idle_minutes": 120,
        "quiet_hours": "22:00-08:00",
        "timezone": "UTC",
    }
    base.update(over)
    return ProactiveGateSettings(**base)


# Midday UTC, well outside the 22:00-08:00 quiet window.
_NOON = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _idle(minutes: int) -> datetime:
    return _NOON - timedelta(minutes=minutes)


def test_gate_passes_when_all_conditions_met():
    d = evaluate_gate(
        _settings(),
        now=_NOON,
        last_activity=_idle(180),
        has_origin=True,
        in_flight=False,
    )
    assert d.should_run is True
    assert d.reason == "ok"


def test_gate_disabled():
    d = evaluate_gate(
        _settings(enabled=False),
        now=_NOON,
        last_activity=_idle(180),
        has_origin=True,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "disabled")


def test_gate_no_origin():
    d = evaluate_gate(
        _settings(),
        now=_NOON,
        last_activity=_idle(180),
        has_origin=False,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "no_origin")


def test_gate_in_flight():
    d = evaluate_gate(
        _settings(),
        now=_NOON,
        last_activity=_idle(180),
        has_origin=True,
        in_flight=True,
    )
    assert (d.should_run, d.reason) == (False, "in_flight")


def test_gate_no_activity():
    d = evaluate_gate(
        _settings(),
        now=_NOON,
        last_activity=None,
        has_origin=True,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "no_activity")


def test_gate_idle_just_under_threshold():
    d = evaluate_gate(
        _settings(idle_minutes=120),
        now=_NOON,
        last_activity=_idle(119),
        has_origin=True,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "idle_below_threshold")


def test_gate_idle_exactly_at_threshold_passes():
    d = evaluate_gate(
        _settings(idle_minutes=120),
        now=_NOON,
        last_activity=_idle(120),
        has_origin=True,
        in_flight=False,
    )
    assert d.should_run is True


def test_gate_quiet_hours_wraparound_night():
    # 23:00 UTC is inside the 22:00-08:00 window.
    now = datetime(2026, 8, 20, 23, 0, tzinfo=UTC)
    d = evaluate_gate(
        _settings(),
        now=now,
        last_activity=now - timedelta(minutes=300),
        has_origin=True,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "quiet_hours")


def test_gate_quiet_hours_wraparound_early_morning():
    # 03:00 UTC is inside the 22:00-08:00 window.
    now = datetime(2026, 8, 20, 3, 0, tzinfo=UTC)
    d = evaluate_gate(
        _settings(),
        now=now,
        last_activity=now - timedelta(minutes=300),
        has_origin=True,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "quiet_hours")


def test_gate_quiet_hours_respects_timezone():
    # 12:00 UTC == 07:00 America/New_York (UTC-5, still inside 22:00-08:00 local).
    d = evaluate_gate(
        _settings(timezone="America/New_York"),
        now=datetime(2026, 1, 20, 12, 0, tzinfo=UTC),
        last_activity=datetime(2026, 1, 20, 6, 0, tzinfo=UTC),
        has_origin=True,
        in_flight=False,
    )
    assert (d.should_run, d.reason) == (False, "quiet_hours")


def test_gate_no_quiet_hours_configured():
    d = evaluate_gate(
        _settings(quiet_hours=None),
        now=datetime(2026, 8, 20, 23, 0, tzinfo=UTC),
        last_activity=datetime(2026, 8, 20, 20, 0, tzinfo=UTC),
        has_origin=True,
        in_flight=False,
    )
    assert d.should_run is True


def test_gate_naive_now_treated_as_utc():
    d = evaluate_gate(
        _settings(),
        now=datetime(2026, 8, 20, 12, 0),  # naive
        last_activity=datetime(2026, 8, 20, 8, 0),  # naive
        has_origin=True,
        in_flight=False,
    )
    assert d.should_run is True
