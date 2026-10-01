"""Tests for the natural-language scheduling tools."""

from unittest.mock import patch

from tests.fakes import TEST_WORKSPACE


def _tool(name: str):
    """The scheduling tool *name* built for ``TEST_WORKSPACE``."""
    from EvoScientist.middleware.scheduler import make_scheduling_tools

    return {t.name: t for t in make_scheduling_tools(TEST_WORKSPACE)}[name]


def test_schedule_task_translates_and_creates():
    schedule_task = _tool("schedule_task")

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
    assert mk.call_args.kwargs["workspace"] == TEST_WORKSPACE


def test_schedule_task_reports_backend_down():
    schedule_task = _tool("schedule_task")

    with patch("EvoScientist.cron.schedule.is_available", return_value=False):
        out = schedule_task.invoke(
            {"name": "x", "cron": "* * * * *", "prompt": "do x", "timezone": ""}
        )
    assert "unavailable" in out.lower()


def test_cancel_scheduled_task():
    cancel_scheduled_task = _tool("cancel_scheduled_task")

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=[]),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        out = cancel_scheduled_task.invoke({"cron_id": "c-7"})
    mk.assert_not_called()
    assert "No scheduled task matching" in out


def test_cancel_scheduled_task_prefix_match():
    cancel_scheduled_task = _tool("cancel_scheduled_task")

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.list_schedules",
            return_value=[{"cron_id": "c-7-abc"}],
        ),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        out = cancel_scheduled_task.invoke({"cron_id": "c-7"})
    mk.assert_called_once_with("c-7-abc", workspace=TEST_WORKSPACE)
    assert "c-7-abc" in out


def test_list_scheduled_tasks_formats_rows():
    list_scheduled_tasks = _tool("list_scheduled_tasks")

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
        ) as lister,
    ):
        out = list_scheduled_tasks.invoke({})
    lister.assert_called_once_with(workspace=TEST_WORKSPACE)
    assert "daily" in out
    assert "0 9 * * *" in out


# ---------------------------------------------------------------------------
# B2: ambiguous prefix in cancel_scheduled_task tool
# ---------------------------------------------------------------------------


def test_cancel_ambiguous_prefix_aborts_without_deleting():
    """B2: two crons sharing a prefix → returns ambiguity message, delete NOT called."""
    cancel_scheduled_task = _tool("cancel_scheduled_task")

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
    cancel_scheduled_task = _tool("cancel_scheduled_task")

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


# ---------------------------------------------------------------------------
# Optional rubric on schedule_task / list_scheduled_tasks
# ---------------------------------------------------------------------------


def test_schedule_task_forwards_rubric():
    schedule_task = _tool("schedule_task")

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.create_schedule",
            return_value={"cron_id": "c-8"},
        ) as mk,
    ):
        schedule_task.invoke(
            {
                "name": "digest",
                "cron": "0 8 * * 1-5",
                "prompt": "write scheduled/digest.md",
                "timezone": "",
                "rubric": "- scheduled/digest.md has today's date",
            }
        )
    assert mk.call_args.kwargs["rubric"] == "- scheduled/digest.md has today's date"


def test_schedule_task_without_rubric_forwards_none():
    schedule_task = _tool("schedule_task")

    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.create_schedule",
            return_value={"cron_id": "c-8"},
        ) as mk,
    ):
        schedule_task.invoke(
            {"name": "ping", "cron": "0 * * * *", "prompt": "ping", "timezone": ""}
        )
    assert mk.call_args.kwargs["rubric"] is None


def test_list_scheduled_tasks_marks_graded_rows():
    list_scheduled_tasks = _tool("list_scheduled_tasks")

    rows = [
        {
            "cron_id": "c-1-xyz",
            "schedule": "0 9 * * *",
            "enabled": True,
            "metadata": {"name": "graded", "rubric": "- out.md exists"},
        },
        {
            "cron_id": "c-2-xyz",
            "schedule": "0 9 * * *",
            "enabled": True,
            "metadata": {"name": "plain"},
        },
    ]
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=rows),
    ):
        out = list_scheduled_tasks.invoke({})
    graded, plain = out.splitlines()
    assert "rubric" in graded
    assert "rubric" not in plain
