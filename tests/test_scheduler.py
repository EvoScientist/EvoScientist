"""Tests for SchedulerMiddleware (scheduling guide + <scheduled_tasks> injection)."""

from unittest.mock import MagicMock


def _mw():
    from EvoScientist.middleware.scheduler import SchedulerMiddleware

    m = SchedulerMiddleware()
    m._cache = None
    m._cache_at = 0.0
    return m


def test_schedules_block_lists_active_crons(monkeypatch):
    from EvoScientist.cron import schedule as crons

    monkeypatch.setattr(crons, "is_available", lambda: True)
    monkeypatch.setattr(
        crons,
        "list_schedules",
        lambda: [
            {
                "cron_id": "abc12345-xyz",
                "schedule": "*/10 * * * *",
                "enabled": True,
                "metadata": {"name": "weather", "prompt": "search uk weather"},
            }
        ],
    )
    block = _mw()._schedules_block()
    assert "<scheduled_tasks>" in block
    assert "</scheduled_tasks>" in block
    assert "weather" in block
    assert "*/10 * * * *" in block
    assert "abc12345" in block


def test_schedules_block_empty_when_unavailable(monkeypatch):
    from EvoScientist.cron import schedule as crons

    monkeypatch.setattr(crons, "is_available", lambda: False)
    assert _mw()._schedules_block() == ""


def test_schedules_block_empty_on_error(monkeypatch):
    from EvoScientist.cron import schedule as crons

    monkeypatch.setattr(crons, "is_available", lambda: True)

    def _boom():
        raise RuntimeError("backend died")

    monkeypatch.setattr(crons, "list_schedules", _boom)
    assert _mw()._schedules_block() == ""


def test_modify_request_injects_guide_even_with_no_schedules(monkeypatch):
    # The static scheduling guide is injected on every call; the dynamic
    # <scheduled_tasks> block is added only when there are active schedules.
    m = _mw()
    monkeypatch.setattr(m, "_cached_schedules_block", lambda: "")
    req = MagicMock()
    req.system_message = None
    m.modify_request(req)
    req.override.assert_called_once()
    sysmsg = req.override.call_args.kwargs["system_message"]
    text = " ".join(str(b) for b in sysmsg.content_blocks)
    assert "scheduling_instructions" in text
    # The guide mentions <scheduled_tasks> in prose; the actual dynamic block
    # (identified by its closing tag) must be absent when there are no schedules.
    assert "</scheduled_tasks>" not in text


def test_modify_request_injects_guide_and_schedules(monkeypatch):
    m = _mw()
    monkeypatch.setattr(
        m, "_cached_schedules_block", lambda: "<scheduled_tasks>\nx\n</scheduled_tasks>"
    )
    req = MagicMock()
    req.system_message = None
    m.modify_request(req)
    req.override.assert_called_once()
    sysmsg = req.override.call_args.kwargs["system_message"]
    text = " ".join(str(b) for b in sysmsg.content_blocks)
    assert "scheduling_instructions" in text
    assert "</scheduled_tasks>" in text


def test_middleware_exposes_three_tools():
    """SchedulerMiddleware.tools must contain the three scheduling tools."""
    m = _mw()
    tool_names = [t.name for t in m.tools]
    assert tool_names == [
        "schedule_task",
        "list_scheduled_tasks",
        "cancel_scheduled_task",
    ]
