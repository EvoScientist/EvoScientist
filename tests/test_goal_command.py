"""The ``/goal`` command: it reads the thread and hands the surface a request."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import EvoScientist.commands.implementation  # noqa: F401  (registers the commands)
from EvoScientist.commands.base import ChannelRuntime, CommandContext
from EvoScientist.commands.manager import manager
from EvoScientist.paths import SessionDirs, Workspace
from tests.fakes import FakeGraphGateway

_LOADED_AGENT = object()
GOAL_ID = "goal:abc"


def _thread(status="active", *, note="", objective="ship the report", **extra):
    return {
        "_goal": {
            "objective": objective,
            "status": status,
            "note": note,
            "evidence": None,
            "rubric_id": GOAL_ID,
            "start": "m1",
        },
        "_sticky_rubric": {
            "text": "- the report exists\n- it has a title",
            "id": GOAL_ID,
        },
        **extra,
    }


def _ctx(
    tmp_path, *, values=None, agent=_LOADED_AGENT, ui=None, gateway=None, config=None
):
    ui = ui if ui is not None else MagicMock()
    ui.flush = AsyncMock()  # the manager awaits it after every command
    runtime = ChannelRuntime()
    ctx = CommandContext(
        agent=agent,
        thread_id="t",
        ui=ui,
        dirs=SessionDirs(Workspace(tmp_path)),
        channel_runtime=runtime,
        graph_gateway=gateway
        if gateway is not None
        else FakeGraphGateway(state_values=values or {}),
        config=config,
    )
    return ctx, ui, runtime


def _lines(ui) -> list[str]:
    lines = [str(c.args[0]) for c in ui.append_system.call_args_list]
    for c in ui.mount_renderable.call_args_list:
        lines.append(getattr(c.args[0], "plain", str(c.args[0])))
    return lines


def _text(ui) -> str:
    return "\n".join(_lines(ui))


class TestCreate:
    async def test_the_objective_is_everything_after_the_command(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path)
        assert await manager.execute("/goal ship the 'report'\nwith a title", ctx)
        request = ctx.goal_request
        assert request["kind"] == "create"
        assert request["objective"] == "ship the 'report'\nwith a title"
        assert request["review"] is True
        assert "target" not in request
        assert request["id"]
        assert _lines(ui) == []

    @pytest.mark.parametrize(
        "typed",
        [
            "show me the money",
            "status report by Friday",
            "pause the pipeline",
            "clear the backlog",
            "resume writing",
            "amendment 5 summary",
        ],
    )
    async def test_a_subcommand_word_inside_a_sentence_is_part_of_the_objective(
        self, tmp_path, typed
    ):
        ctx, _ui, _ = _ctx(tmp_path, values=_thread())
        assert await manager.execute(f"/goal {typed}", ctx)
        assert ctx.goal_request["kind"] == "create"
        assert ctx.goal_request["objective"] == typed

    async def test_a_thread_that_has_a_goal_can_get_a_new_one(self, tmp_path):
        ctx, _ui, _ = _ctx(tmp_path, values=_thread("complete"))
        assert await manager.execute("/goal another one", ctx)
        assert ctx.goal_request["kind"] == "create"

    async def test_an_oversized_objective_is_refused_before_any_run(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path)
        assert await manager.execute("/goal " + "x" * 8_001, ctx)
        assert ctx.goal_request is None
        assert "8,000" in _text(ui)

    async def test_unattended_sessions_skip_the_review(self, tmp_path):
        ctx, _ui, _ = _ctx(tmp_path, config=SimpleNamespace(auto_mode=True))
        assert await manager.execute("/goal ship it", ctx)
        assert ctx.goal_request["review"] is False

    async def test_without_a_loaded_agent_nothing_is_started(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path, agent=None)
        assert await manager.execute("/goal ship it", ctx)
        assert ctx.goal_request is None
        assert "not loaded" in _text(ui)


class TestAmend:
    @pytest.mark.parametrize("status", ["active", "blocked"])
    async def test_it_targets_the_goal_it_saw(self, tmp_path, status):
        ctx, _ui, _ = _ctx(tmp_path, values=_thread(status))
        assert await manager.execute('/goal amend also add a "summary"', ctx)
        request = ctx.goal_request
        assert request["kind"] == "amend"
        assert request["feedback"] == 'also add a "summary"'
        assert request["target"] == GOAL_ID
        assert request["review"] is True

    async def test_the_word_is_matched_without_regard_to_case(self, tmp_path):
        ctx, _ui, _ = _ctx(tmp_path, values=_thread())
        assert await manager.execute("/goal AMEND\nmore tests", ctx)
        assert ctx.goal_request["kind"] == "amend"
        assert ctx.goal_request["feedback"] == "more tests"

    async def test_without_feedback_it_shows_the_usage(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path, values=_thread())
        assert await manager.execute("/goal amend", ctx)
        assert ctx.goal_request is None
        assert "Usage: /goal" in _text(ui)

    async def test_without_a_goal(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path)
        assert await manager.execute("/goal amend more tests", ctx)
        assert ctx.goal_request is None
        assert "No goal" in _text(ui)

    @pytest.mark.parametrize(
        ("status", "hint"),
        [("paused", "/goal resume"), ("complete", "/goal <objective>")],
    )
    async def test_a_goal_that_cannot_be_amended_says_what_to_do(
        self, tmp_path, status, hint
    ):
        ctx, ui, _ = _ctx(tmp_path, values=_thread(status))
        assert await manager.execute("/goal amend more tests", ctx)
        assert ctx.goal_request is None
        assert hint in _text(ui)


class TestStatusChanges:
    @pytest.mark.parametrize(
        ("typed", "status"),
        [
            ("pause", "active"),
            ("resume", "paused"),
            ("clear", "active"),
            ("clear", "paused"),
            ("clear", "blocked"),
            ("clear", "complete"),
            ("  PAUSE  ", "active"),
        ],
    )
    async def test_an_allowed_change_becomes_a_request(self, tmp_path, typed, status):
        ctx, ui, _ = _ctx(tmp_path, values=_thread(status))
        assert await manager.execute(f"/goal {typed}", ctx)
        assert ctx.goal_request["kind"] == typed.strip().lower()
        assert ctx.goal_request["target"] == GOAL_ID
        assert _lines(ui) == []

    @pytest.mark.parametrize(
        ("typed", "status"),
        [
            ("pause", "paused"),
            ("pause", "blocked"),
            ("pause", "complete"),
            ("resume", "active"),
            ("resume", "blocked"),
            ("resume", "complete"),
        ],
    )
    async def test_a_change_the_status_does_not_allow_is_explained(
        self, tmp_path, typed, status
    ):
        ctx, ui, _ = _ctx(tmp_path, values=_thread(status))
        assert await manager.execute(f"/goal {typed}", ctx)
        assert ctx.goal_request is None
        assert f"The goal is {status}" in _text(ui)

    @pytest.mark.parametrize("typed", ["pause", "resume", "clear"])
    async def test_without_a_goal(self, tmp_path, typed):
        ctx, ui, _ = _ctx(tmp_path)
        assert await manager.execute(f"/goal {typed}", ctx)
        assert ctx.goal_request is None
        assert "No goal" in _text(ui)

    async def test_a_goal_whose_checklist_was_replaced_is_no_goal(self, tmp_path):
        values = _thread()
        values["_sticky_rubric"] = {"text": "- other", "id": "s1:1"}
        ctx, ui, _ = _ctx(tmp_path, values=values)
        assert await manager.execute("/goal pause", ctx)
        assert ctx.goal_request is None
        assert "No goal" in _text(ui)

    async def test_an_unreadable_thread_changes_nothing(self, tmp_path):
        gateway = FakeGraphGateway(state_error=RuntimeError("db locked"))
        ctx, ui, _ = _ctx(tmp_path, gateway=gateway)
        assert await manager.execute("/goal clear", ctx)
        assert ctx.goal_request is None
        assert "could not be read" in _text(ui)


class TestShow:
    @pytest.mark.parametrize("typed", ["", "show", "status", " Show "])
    async def test_it_prints_the_goal_and_its_criteria(self, tmp_path, typed):
        ctx, ui, _ = _ctx(tmp_path, values=_thread())
        assert await manager.execute(
            f"/goal {typed}".rstrip() if typed else "/goal", ctx
        )
        assert ctx.goal_request is None
        text = _text(ui)
        assert "Goal (active)" in text
        assert "ship the report" in text
        assert "- it has a title" in text

    async def test_the_users_text_is_never_markup(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path, values=_thread(objective="[bold red]boom[/]"))
        assert await manager.execute("/goal", ctx)
        plain = [
            getattr(c.args[0], "plain", None)
            for c in ui.mount_renderable.call_args_list
        ]
        assert "[bold red]boom[/]" in plain
        assert not any(
            "boom" in str(c.args[0]) for c in ui.append_system.call_args_list
        )

    @pytest.mark.parametrize(
        ("status", "note", "label"),
        [
            ("blocked", "need an API key", "Blocked: need an API key"),
            ("complete", "all tests pass", "Result: all tests pass"),
        ],
    )
    async def test_the_note_is_shown_with_what_it_means(
        self, tmp_path, status, note, label
    ):
        ctx, ui, _ = _ctx(tmp_path, values=_thread(status, note=note))
        assert await manager.execute("/goal status", ctx)
        assert label in _text(ui)

    async def test_without_a_goal_it_shows_the_usage(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path)
        assert await manager.execute("/goal", ctx)
        text = _text(ui)
        assert "No goal set for this thread." in text
        assert "Usage: /goal <objective>" in text

    async def test_a_pending_checklist_command_is_announced(self, tmp_path):
        ctx, ui, runtime = _ctx(tmp_path, values=_thread())
        runtime.rubric.record("t", sticky="- mine")
        assert await manager.execute("/goal", ctx)
        assert "cancelled with your next message" in _text(ui)

    async def test_an_applied_checklist_command_is_not_announced(self, tmp_path):
        ctx, ui, runtime = _ctx(tmp_path)
        operation = runtime.rubric.record("t", next="- once")
        ctx.graph_gateway.state_values = _thread(
            _rubric_applied={runtime.rubric.session_id: operation["seq"]}
        )
        assert await manager.execute("/goal", ctx)
        assert "cancelled" not in _text(ui)
        assert runtime.rubric.changes_for("t") == []

    async def test_a_pending_one_shot_does_not_cancel_the_goal(self, tmp_path):
        ctx, ui, runtime = _ctx(tmp_path, values=_thread())
        runtime.rubric.record("t", next="- once")
        assert await manager.execute("/goal", ctx)
        assert "cancelled" not in _text(ui)

    async def test_without_a_loaded_agent(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path, agent=None)
        assert await manager.execute("/goal", ctx)
        assert "not loaded" in _text(ui)

    async def test_an_unreadable_thread_is_not_blamed_on_the_agent(self, tmp_path):
        gateway = FakeGraphGateway(state_error=RuntimeError("db locked"))
        ctx, ui, _ = _ctx(tmp_path, gateway=gateway)
        assert await manager.execute("/goal", ctx)
        assert "could not be read" in _text(ui)
        assert "not loaded" not in _text(ui)


class TestRefusals:
    @pytest.mark.parametrize("typed", ["ship it", "", "pause", "amend more"])
    async def test_on_a_channel(self, tmp_path, typed):
        ui = MagicMock()
        ui.supports_interactive = False
        ctx, ui, _ = _ctx(tmp_path, ui=ui, values=_thread())
        assert await manager.execute(f"/goal {typed}".rstrip(), ctx)
        assert ctx.goal_request is None
        assert "terminal session" in _text(ui)

    async def test_when_the_session_executes_on_the_server(self, tmp_path):
        from EvoScientist.gateway.server import LangGraphServerGateway

        server = MagicMock(spec=LangGraphServerGateway)
        ctx, ui, _ = _ctx(tmp_path, gateway=server)
        assert await manager.execute("/goal ship it", ctx)
        assert ctx.goal_request is None
        assert "local gateway" in _text(ui)


class TestRegistration:
    def test_the_agent_is_awaited_for_everything_but_show(self):
        cmd, _ = manager.resolve("/goal")
        assert cmd.needs_agent([]) is False
        assert cmd.needs_agent(["show"]) is False
        assert cmd.needs_agent(["status"]) is False
        assert cmd.needs_agent(["show", "me"]) is True
        assert cmd.needs_agent(["pause"]) is True
        assert cmd.needs_agent(["ship", "it"]) is True

    def test_completion_lists_the_subcommands(self):
        cmd, _ = manager.resolve("/goal")
        names = [name for name, _description in cmd.get_completions([""])]
        assert names == ["show", "amend", "pause", "resume", "clear"]


class TestRubricCommandNote:
    @pytest.mark.parametrize("typed", ["set - mine", "clear"])
    async def test_set_and_clear_say_that_they_cancel_the_goal(self, tmp_path, typed):
        ctx, ui, _ = _ctx(tmp_path, values=_thread())
        assert await manager.execute(f"/rubric {typed}", ctx)
        assert "cancels the thread's goal" in _text(ui)

    async def test_next_does_not(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path, values=_thread())
        assert await manager.execute("/rubric next - once", ctx)
        assert "goal" not in _text(ui)

    async def test_without_a_goal_nothing_is_added(self, tmp_path):
        ctx, ui, _ = _ctx(tmp_path)
        assert await manager.execute("/rubric set - mine", ctx)
        assert "goal" not in _text(ui)


class TestBadge:
    @pytest.mark.parametrize(
        ("status", "badge"),
        [
            ("active", "goal: active"),
            ("paused", "goal: paused"),
            ("blocked", "goal: blocked"),
            ("complete", "goal: done"),
        ],
    )
    def test_a_goal_takes_the_place_of_the_checklist_badge(self, status, badge):
        from EvoScientist.rubric_session import rubric_badge, values_for_badge

        assert rubric_badge(_thread(status), []) == badge
        assert rubric_badge(values_for_badge(badge), []) == badge

    def test_a_waiting_one_shot_is_still_shown(self):
        from EvoScientist.rubric_session import rubric_badge, values_for_badge

        values = _thread(_next_rubric={"text": "- once", "id": "s:1"})
        assert rubric_badge(values, []) == "goal: active+next"
        assert rubric_badge(_thread(), [{"next": "- once"}]) == "goal: active+next"
        assert (
            rubric_badge(values_for_badge("goal: active+next"), [])
            == "goal: active+next"
        )

    def test_a_pending_checklist_command_shows_the_goal_as_ending(self):
        from EvoScientist.rubric_session import rubric_badge, values_for_badge

        pending = [{"sticky": "- mine"}]
        assert rubric_badge(_thread(), pending) == "goal: ending"
        assert rubric_badge(_thread(), [{"sticky": None}]) == "goal: ending"
        # Unreadable state: the badge keeps saying so while the command waits.
        assert rubric_badge(values_for_badge("goal: ending"), pending) == "goal: ending"

    def test_a_goal_without_its_checklist_is_no_goal(self):
        from EvoScientist.rubric_session import rubric_badge

        values = _thread()
        values["_sticky_rubric"] = {"text": "- other", "id": "s1:1"}
        assert rubric_badge(values, []) == "on"

    def test_the_status_bar_shows_it_under_its_own_name(self):
        from datetime import datetime

        from EvoScientist.cli.status_bar import build_status_fragments
        from tests.test_status_bar import TestRubricBadge

        snapshot = TestRubricBadge()._snapshot()
        frags = build_status_fragments(
            snapshot, datetime.now(), 120, rubric_badge="goal: active+next"
        )
        assert ("class:status-bar-strong", "goal: active+next") in frags
        assert not any("rubric" in text for _style, text in frags)
