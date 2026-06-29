"""Tests for the /schedule command."""

from unittest.mock import MagicMock, patch

from tests.conftest import run_async as _run


def _ctx():
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    return CommandContext(agent=None, thread_id="tid", ui=ui), ui


def test_list_when_backend_down():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, ui = _ctx()
    with patch("EvoScientist.cron.schedule.is_available", return_value=False):
        _run(ScheduleCommand().execute(ctx, ["list"]))
    msgs = [c.args[0] for c in ui.append_system.call_args_list]
    assert any("unavailable" in m.lower() for m in msgs)


def test_add_parses_five_field_cron_and_prompt():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, _ui = _ctx()
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.create_schedule",
            return_value={"cron_id": "c-9"},
        ) as mk,
    ):
        _run(
            ScheduleCommand().execute(
                ctx, ["add", "*/10", "*", "*", "*", "*", "search", "uk", "weather"]
            )
        )
    kw = mk.call_args.kwargs
    assert kw["schedule"] == "*/10 * * * *"
    assert kw["prompt"] == "search uk weather"


def test_list_renders_table():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, ui = _ctx()
    rows = [
        {
            "cron_id": "c-1",
            "schedule": "0 9 * * *",
            "enabled": True,
            "next_run_date": "2026-06-25T09:00:00+00:00",
            "metadata": {"name": "daily"},
        }
    ]
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=rows),
    ):
        _run(ScheduleCommand().execute(ctx, ["list"]))
    ui.mount_renderable.assert_called_once()


def test_add_parses_quoted_cron_and_prompt():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, _ui = _ctx()
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.create_schedule",
            return_value={"cron_id": "c-9"},
        ) as mk,
    ):
        _run(
            ScheduleCommand().execute(ctx, ["add", "*/10 * * * *", "search uk weather"])
        )
    kw = mk.call_args.kwargs
    assert kw["schedule"] == "*/10 * * * *"
    assert kw["prompt"] == "search uk weather"


def test_run_with_matching_prefix_fires_matched_prompt():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, _ui = _ctx()
    rows = [{"cron_id": "c-12345", "metadata": {"prompt": "do the thing"}}]
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=rows),
        patch(
            "EvoScientist.cron.schedule.run_now",
            return_value={"run_id": "r-1"},
        ) as rn,
    ):
        _run(ScheduleCommand().execute(ctx, ["run", "c-123"]))
    rn.assert_called_once_with("do the thing")


def test_run_with_no_match_reports():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, ui = _ctx()
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=[]),
        patch("EvoScientist.cron.schedule.run_now") as rn,
    ):
        _run(ScheduleCommand().execute(ctx, ["run", "nope"]))
    rn.assert_not_called()
    msgs = [c.args[0] for c in ui.append_system.call_args_list]
    assert any("No schedule matching" in m for m in msgs)


def test_pause_resume_set_enabled_with_resolved_id():
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    rows = [{"cron_id": "c-abcdef", "metadata": {"name": "t"}}]
    for sub, expected in (("pause", False), ("resume", True)):
        ctx, _ui = _ctx()
        with (
            patch("EvoScientist.cron.schedule.is_available", return_value=True),
            patch("EvoScientist.cron.schedule.list_schedules", return_value=rows),
            patch("EvoScientist.cron.schedule.set_enabled") as se,
        ):
            _run(ScheduleCommand().execute(ctx, [sub, "c-abc"]))
        se.assert_called_once_with("c-abcdef", expected)


# ---------------------------------------------------------------------------
# B1: _list error handling when backend dies after is_available()
# ---------------------------------------------------------------------------


def test_list_error_shows_red_message_no_exception():
    """B1: list_schedules raising after is_available() shows a red error, not a traceback."""
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, ui = _ctx()
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.list_schedules",
            side_effect=RuntimeError("backend gone"),
        ),
    ):
        _run(ScheduleCommand().execute(ctx, ["list"]))
    msgs = [c.args[0] for c in ui.append_system.call_args_list]
    assert any("Error:" in m for m in msgs)
    # Verify no exception escaped (test would have raised above otherwise)


# ---------------------------------------------------------------------------
# B2: ambiguous prefix detection in /schedule command
# ---------------------------------------------------------------------------


def test_remove_ambiguous_prefix_aborts_without_deleting():
    """B2: two crons sharing a prefix → ambiguity message, delete NOT called."""
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, ui = _ctx()
    rows = [
        {"cron_id": "abc-111", "metadata": {}},
        {"cron_id": "abc-222", "metadata": {}},
    ]
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch("EvoScientist.cron.schedule.list_schedules", return_value=rows),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        _run(ScheduleCommand().execute(ctx, ["remove", "abc"]))
    mk.assert_not_called()
    msgs = [c.args[0] for c in ui.append_system.call_args_list]
    assert any("Multiple" in m for m in msgs)


# ---------------------------------------------------------------------------
# B3: sanitized name from prompt
# ---------------------------------------------------------------------------


def test_remove_backend_error_shows_red_error_not_no_match():
    """FIX 1: list_schedules() crashing in _resolve → red 'Error:' message, not 'No schedule matching'."""
    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, ui = _ctx()
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.list_schedules",
            side_effect=RuntimeError("boom"),
        ),
        patch("EvoScientist.cron.schedule.delete_schedule") as mk,
    ):
        _run(ScheduleCommand().execute(ctx, ["remove", "abc"]))
    mk.assert_not_called()
    msgs = [c.args[0] for c in ui.append_system.call_args_list]
    assert any("Error:" in m for m in msgs), f"Expected red Error: message, got: {msgs}"
    assert not any("No schedule matching" in m for m in msgs), (
        f"Backend crash must not look like a name miss, got: {msgs}"
    )
    # The error message must mention unavailable/backend to distinguish from other errors.
    assert any("unavailable" in m.lower() or "backend" in m.lower() for m in msgs), (
        f"Error message should mention backend/unavailable, got: {msgs}"
    )


def test_add_name_sanitized_from_nasty_prompt():
    """B3: prompt with newline / slashes / special chars → clean kebab-case name."""
    import re

    from EvoScientist.commands.implementation.schedule import ScheduleCommand

    ctx, _ui = _ctx()
    # prompt with newline, slashes, and dots
    nasty_prompt = "Search /tmp/foo\nand summarize! Latest.Papers."
    with (
        patch("EvoScientist.cron.schedule.is_available", return_value=True),
        patch(
            "EvoScientist.cron.schedule.create_schedule",
            return_value={"cron_id": "c-x"},
        ) as mk,
    ):
        _run(ScheduleCommand().execute(ctx, ["add", "*/5 * * * *", nasty_prompt]))
    name = mk.call_args.kwargs["name"]
    # Must be non-empty, no spaces, no newlines, no slashes
    assert name
    assert " " not in name
    assert "\n" not in name
    assert "/" not in name
    # Only safe chars: lowercase alphanumeric and hyphens
    assert re.fullmatch(r"[a-z0-9][a-z0-9\-]*", name), f"Unexpected name: {name!r}"
