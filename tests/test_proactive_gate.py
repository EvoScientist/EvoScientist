"""Tests for EvoScientist.proactive.gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from EvoScientist.proactive.gate import ProactiveGateSettings, evaluate_gate

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
SETTINGS = ProactiveGateSettings(idle_minutes=120, quiet_hours="", timezone="UTC")


def _gate(settings=SETTINGS, **overrides):
    kwargs = {
        "now": NOW,
        "armed": True,
        "has_origin": True,
        "in_flight": False,
        "last_activity": NOW - timedelta(minutes=180),
        "last_message_id": "m2",
        "last_evaluated_id": "m1",
    }
    kwargs.update(overrides)
    return evaluate_gate(settings, **kwargs)


def test_passes_an_armed_quiet_thread_with_new_information():
    assert _gate().should_run is True
    assert _gate().reason == "ok"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"armed": False}, "not_armed"),
        ({"has_origin": False}, "no_origin"),
        ({"in_flight": True}, "in_flight"),
        ({"last_activity": None}, "no_activity"),
        ({"last_message_id": None}, "no_activity"),
        ({"last_evaluated_id": "m2"}, "no_new_information"),
        ({"last_activity": NOW - timedelta(minutes=30)}, "idle_below_threshold"),
    ],
)
def test_rejects_with_reason(overrides, reason):
    decision = _gate(**overrides)
    assert decision.should_run is False
    assert decision.reason == reason


def test_first_check_of_a_thread_has_new_information():
    assert _gate(last_evaluated_id=None).should_run is True


def test_reason_names_the_first_failing_check():
    assert _gate(armed=False, in_flight=True).reason == "not_armed"


def test_naive_timestamps_are_utc():
    naive_now = NOW.replace(tzinfo=None)
    assert _gate(now=naive_now, last_activity=naive_now - timedelta(hours=3)).should_run


@pytest.mark.parametrize(
    ("window", "hour", "quiet"),
    [
        ("22:00-08:00", 23, True),
        ("22:00-08:00", 3, True),
        ("22:00-08:00", 12, False),
        ("09:00-17:00", 12, True),
        ("09:00-17:00", 17, False),
        ("10:00-10:00", 10, False),
        ("", 3, False),
    ],
)
def test_quiet_hours(window, hour, quiet):
    now = NOW.replace(hour=hour)
    settings = ProactiveGateSettings(idle_minutes=0, quiet_hours=window, timezone="UTC")
    decision = _gate(settings, now=now, last_activity=now - timedelta(hours=1))
    assert (decision.reason == "quiet_hours") is quiet


def test_quiet_hours_use_the_configured_zone():
    # 12:00 UTC is 21:00 in Tokyo: outside 22:00-08:00 there, inside in UTC+11.
    settings = ProactiveGateSettings(
        idle_minutes=0, quiet_hours="22:00-08:00", timezone="Asia/Tokyo"
    )
    assert _gate(settings).should_run is True
    settings = ProactiveGateSettings(
        idle_minutes=0, quiet_hours="22:00-08:00", timezone="Pacific/Noumea"
    )
    assert _gate(settings).reason == "quiet_hours"
