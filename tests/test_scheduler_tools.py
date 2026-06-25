"""Tests for the natural-language scheduling tools."""

from unittest.mock import patch


def test_schedule_task_translates_and_creates():
    from EvoScientist.middleware.scheduler import schedule_task

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.create_schedule",
            return_value={"cron_id": "c-7"},
        ) as mk,
    ):
        out = schedule_task.invoke(
            {
                "name": "weather",
                "cron": "*/10 * * * *",
                "prompt": "search uk weather and summarize",
                "timezone": "",
            }
        )
    assert "c-7" in out
    assert mk.call_args.kwargs["schedule"] == "*/10 * * * *"
    assert mk.call_args.kwargs["name"] == "weather"


def test_schedule_task_reports_backend_down():
    from EvoScientist.middleware.scheduler import schedule_task

    with patch("EvoScientist.cron.schedule.is_available", return_value=False):
        out = schedule_task.invoke(
            {"name": "x", "cron": "* * * * *", "prompt": "do x", "timezone": ""}
        )
    assert "unavailable" in out.lower()


def test_cancel_scheduled_task():
    from EvoScientist.middleware.scheduler import cancel_scheduled_task

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=[]),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        out = cancel_scheduled_task.invoke({"cron_id": "c-7"})
    mk.assert_not_called()
    assert "No scheduled task matching" in out


def test_cancel_scheduled_task_prefix_match():
    from EvoScientist.middleware.scheduler import cancel_scheduled_task

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.list_schedules",
            return_value=[{"cron_id": "c-7-abc"}],
        ),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        out = cancel_scheduled_task.invoke({"cron_id": "c-7"})
    mk.assert_called_once_with("c-7-abc")
    assert "c-7-abc" in out


def test_list_scheduled_tasks_formats_rows():
    from EvoScientist.middleware.scheduler import list_scheduled_tasks

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.list_schedules",
            return_value=[
                {
                    "cron_id": "c-1-xyz",
                    "schedule": "0 9 * * *",
                    "enabled": True,
                    "metadata": {"name": "daily"},
                }
            ],
        ),
    ):
        out = list_scheduled_tasks.invoke({})
    assert "daily" in out
    assert "0 9 * * *" in out


# ---------------------------------------------------------------------------
# B2: ambiguous prefix in cancel_scheduled_task tool
# ---------------------------------------------------------------------------


def test_cancel_ambiguous_prefix_aborts_without_deleting():
    """B2: two crons sharing a prefix → returns ambiguity message, delete NOT called."""
    from EvoScientist.middleware.scheduler import cancel_scheduled_task

    rows = [{"cron_id": "abc-111"}, {"cron_id": "abc-222"}]
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=rows),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        out = cancel_scheduled_task.invoke({"cron_id": "abc"})
    mk.assert_not_called()
    assert "Multiple" in out


def test_cancel_empty_cron_id_refuses_without_deleting():
    """Empty cron_id would match (and delete) the only cron — must refuse early."""
    from EvoScientist.middleware.scheduler import cancel_scheduled_task

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.list_schedules",
            return_value=[{"cron_id": "only-one"}],
        ),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        out = cancel_scheduled_task.invoke({"cron_id": "   "})
    mk.assert_not_called()
    assert "Provide" in out
