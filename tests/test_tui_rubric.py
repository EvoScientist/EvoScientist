"""TUI side of ``/rubric``: the run marker, acknowledgement, badge and grading lines.

Boots the real TUI class through Textual's pilot (see
``test_tui_banner_position.capture_app``) and drives it with a fake gateway.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from tests.tui_fakes import (
    GatedStream,
    capture_app,
    chat_sequence,
    finish_turns,
    isolated_skills,  # noqa: F401  (autouse fixture)
    prepare_app,
    rubric_gateway,
    run_gated,
    submit,
    system_lines,
)


async def test_typed_turn_carries_the_thread_marker_and_acknowledges(
    monkeypatch, workspace
):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        session = app._channel_runtime.rubric
        tid = app._conversation_tid
        op = session.record(tid, sticky="- s")
        gateway = rubric_gateway(
            post_state={
                "_rubric_applied": {session.session_id: 1},
                "_sticky_rubric": {"text": "- s", "id": f"{session.session_id}:1"},
            }
        )
        prepare_app(app, workspace, gateway)
        await submit(app, "hello")
        await finish_turns(app, pilot)
        request = gateway.requests[-1]
        assert request.configurable_extra["rubric_run"] == {
            "mode": "thread",
            "changes": [op],
        }
        assert session.changes_for(tid) == []
        assert app._rubric_badges.get(tid) == "on"


async def test_notifier_turn_is_a_followup_for_the_current_thread_only(
    monkeypatch, workspace
):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test():
        gateway = rubric_gateway()
        prepare_app(app, workspace, gateway)
        tid = app._conversation_tid
        await app._run_turn(
            "[Async tasks update]",
            skip_user_message=True,
            resolve_mentions=False,
            thread_id_override=tid,
            rubric_mode="followup",
        )
        assert gateway.requests[-1].configurable_extra["rubric_run"]["mode"] == (
            "followup"
        )
        await app._run_turn(
            "[Async tasks update]",
            skip_user_message=True,
            resolve_mentions=False,
            thread_id_override="another-thread",
            rubric_mode="followup",
        )
        extra = gateway.requests[-1].configurable_extra or {}
        assert "rubric_run" not in extra


async def test_no_marker_when_the_session_executes_on_the_server(
    monkeypatch, workspace
):
    from unittest.mock import MagicMock

    from EvoScientist.gateway.server import LangGraphServerGateway

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway()
        gateway.execute_gateway = MagicMock(spec=LangGraphServerGateway)
        prepare_app(app, workspace, gateway)
        await submit(app, "hello")
        await finish_turns(app, pilot)
        extra = gateway.requests[-1].configurable_extra or {}
        assert "rubric_run" not in extra


async def test_grading_lines_sit_between_the_draft_and_the_revision(
    monkeypatch, workspace
):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway(
            events=[
                {"type": "text", "content": "draft"},
                {"type": "rubric_evaluation_start", "iteration": 0},
                {
                    "type": "rubric_evaluation",
                    "iteration": 0,
                    "result": "needs_revision",
                    "explanation": "one gap",
                    "criteria": [{"name": "c1", "passed": False, "gap": "missing"}],
                    "unverified": False,
                },
                {"type": "text", "content": "fixed"},
                {"type": "rubric_evaluation_start", "iteration": 1},
                {
                    "type": "rubric_evaluation",
                    "iteration": 1,
                    "result": "satisfied",
                    "explanation": "ok",
                    "criteria": [{"name": "c1", "passed": True}],
                    "unverified": False,
                },
                {"type": "done", "response": "draft\n\nfixed"},
            ]
        )
        prepare_app(app, workspace, gateway)
        await submit(app, "task")
        await finish_turns(app, pilot)
        seq = chat_sequence(app)
        verdicts = [
            i
            for i, (k, t) in enumerate(seq)
            if k == "system" and "not yet satisfied" in t
        ]
        assert len(verdicts) == 1
        verdict = verdicts[0]
        assert any(k == "assistant" and "draft" in t for k, t in seq[:verdict])
        assert any(k == "assistant" and "fixed" in t for k, t in seq[verdict:])
        assert any(k == "system" and "c1: missing" in t for k, t in seq)
        assert any(
            k == "system" and "satisfied" in t and "not" not in t for k, t in seq
        )
        assert sum(1 for k, t in seq if k == "system" and "checking" in t) == 2


async def test_deferred_event_mounts_one_line(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway(
            events=[
                {"type": "text", "content": "started the job"},
                {"type": "rubric_evaluation_deferred", "tasks": 0, "processes": 1},
                {"type": "done", "response": "started the job"},
            ]
        )
        prepare_app(app, workspace, gateway)
        await submit(app, "task")
        await finish_turns(app, pilot)
        lines = system_lines(app)
        assert any(
            "Acceptance check deferred: 1 background process still running" in line
            for line in lines
        )


async def test_rubric_command_updates_the_badge_before_the_next_message(
    monkeypatch, workspace
):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        prepare_app(app, workspace, rubric_gateway())
        await submit(app, "/rubric set - a")
        await pilot.pause()
        for _ in range(20):
            if app._rubric_badges.get(app._conversation_tid) == "on":
                break
            await pilot.pause()
        assert app._rubric_badges.get(app._conversation_tid) == "on"
        assert app._channel_runtime.rubric.changes_for(app._conversation_tid)


async def test_show_uses_the_loaded_agent_without_waiting(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway(
            state_values={"_sticky_rubric": {"text": "- saved here", "id": "s:1"}}
        )
        prepare_app(app, workspace, gateway)
        app._agent_loader.agent = object()
        await submit(app, "/rubric show")
        shown: list[str] = []
        for _ in range(20):
            await pilot.pause()
            # The checklist text is a plain renderable, the labels are system lines.
            shown = [str(w.render()) for w in app.query_one("#chat").children]
            if any("- saved here" in line for line in shown):
                break
        assert any("- saved here" in line for line in shown), shown
        assert not any("not loaded" in line for line in shown)


async def test_badge_appears_once_the_agent_has_loaded(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway(
            state_values={"_sticky_rubric": {"text": "- saved", "id": "s:1"}}
        )
        prepare_app(app, workspace, gateway)
        app._agent_loader.agent = None
        await app._refresh_status_snapshot()
        assert app._rubric_badges.get(app._conversation_tid) is None
        loaded = object()
        app._agent_loader.agent = loaded
        app._on_agent_load_success(loaded)
        for _ in range(20):
            await pilot.pause()
            if app._rubric_badges.get(app._conversation_tid) == "on":
                break
        assert app._rubric_badges.get(app._conversation_tid) == "on"


async def test_a_late_subagent_end_event_does_not_repeat_the_draft(
    monkeypatch, workspace
):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway(
            events=[
                {"type": "text", "content": "draft"},
                {
                    "type": "subagent_start",
                    "instance_id": "h1",
                    "name": "helper",
                    "description": "",
                    "tool_call_id": "tc-h1",
                },
                {"type": "rubric_evaluation_start", "iteration": 0},
                {
                    "type": "rubric_evaluation",
                    "iteration": 0,
                    "result": "needs_revision",
                    "explanation": "one gap",
                    "criteria": [{"name": "c1", "passed": False, "gap": "missing"}],
                    "unverified": False,
                },
                {"type": "text", "content": "fixed"},
                # Sent by the sub-agent's own task, so it can trail later events.
                {"type": "subagent_end", "instance_id": "h1", "name": "helper"},
                {"type": "done", "response": "draft\n\nfixed"},
            ]
        )
        prepare_app(app, workspace, gateway)
        await submit(app, "task")
        await finish_turns(app, pilot)
        seq = chat_sequence(app)
        verdict = next(
            i
            for i, (k, t) in enumerate(seq)
            if k == "system" and "not yet satisfied" in t
        )
        after = [t for k, t in seq[verdict:] if k == "assistant"]
        assert after, seq
        assert all("draft" not in t for t in after), after


_CHECK = {"type": "rubric_evaluation_start", "iteration": 0}
_CHECKING_LINE = "rubric: checking acceptance criteria (pass 1)"
_SATISFIED = {
    "type": "rubric_evaluation",
    "iteration": 0,
    "result": "satisfied",
    "explanation": "ok",
    "criteria": [],
    "unverified": False,
}


async def test_a_spinner_runs_under_the_checking_line_until_the_verdict(
    monkeypatch, workspace
):
    from EvoScientist.cli.widgets.loading_widget import LoadingWidget
    from EvoScientist.cli.widgets.system_message import SystemMessage

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gated = GatedStream(
            {"type": "text", "content": "draft"},
            _CHECK,
            "grading",
            _SATISFIED,
            "verdict",
            {"type": "done", "response": "draft"},
        )
        await run_gated(app, workspace, gated)

        # The grader's own work is not shown: the spinner says it is running.
        *_, line, last = await gated.chat_at(app, pilot, "grading")
        assert isinstance(line, SystemMessage)
        assert str(line.render()) == _CHECKING_LINE
        assert isinstance(last, LoadingWidget)
        gated.go_on("grading")

        # The verdict takes the spinner's place while the run still goes on.
        *_, line, last = await gated.chat_at(app, pilot, "verdict")
        assert str(line.render()) == _CHECKING_LINE
        assert isinstance(last, SystemMessage)
        assert str(last.render()) == "Acceptance criteria satisfied"
        assert not list(app.query(LoadingWidget))
        gated.go_on("verdict")
        await finish_turns(app, pilot)


async def test_a_check_that_opens_the_turn_keeps_one_spinner(monkeypatch, workspace):
    from EvoScientist.cli.widgets.loading_widget import LoadingWidget

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        # The answer had no text, so the turn's first spinner is still up.
        gated = GatedStream(
            _CHECK, "grading", _SATISFIED, {"type": "done", "response": ""}
        )
        await run_gated(app, workspace, gated)

        chat = await gated.chat_at(app, pilot, "grading")
        assert [isinstance(w, LoadingWidget) for w in chat].count(True) == 1
        assert isinstance(chat[-1], LoadingWidget)
        gated.go_on("grading")
        await finish_turns(app, pilot)
        assert not list(app.query(LoadingWidget))


async def test_a_grader_retry_notice_goes_above_the_spinner(monkeypatch, workspace):
    from EvoScientist.cli.widgets.loading_widget import LoadingWidget

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        retry = {
            "type": "model_retry",
            "attempt": 2,
            "wait_seconds": 8.0,
            "reason": "503: starting up",
            "budget_left_seconds": 50.0,
            "cause": "exception",
            "subagent": "rubric_grader",
            "instance_id": "EvoRubricMiddleware.after_agent:1",
            "discarded": {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []},
        }
        gated = GatedStream(
            {"type": "text", "content": "draft"},
            _CHECK,
            retry,
            "waiting",
            _SATISFIED,
            {"type": "done", "response": "draft"},
        )
        await run_gated(app, workspace, gated)

        *_, line, notice, last = await gated.chat_at(app, pilot, "waiting")
        assert str(line.render()) == _CHECKING_LINE
        assert str(notice.render()).startswith("rubric_grader: Model request failed")
        assert isinstance(last, LoadingWidget)
        gated.go_on("waiting")
        await finish_turns(app, pilot)
        assert not list(app.query(LoadingWidget))


async def test_checklist_text_is_shown_literally_in_the_tui(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        text = "- [x] report.md exists\n- cite [docs] and keep [/tmp/out] intact"
        gateway = rubric_gateway(
            state_values={"_sticky_rubric": {"text": text, "id": "s:1"}}
        )
        prepare_app(app, workspace, gateway)
        app._agent_loader.agent = object()
        await submit(app, "/rubric show")
        shown: list[str] = []
        for _ in range(20):
            await pilot.pause()
            shown = [str(w.render()) for w in app.query_one("#chat").children]
            if any(text == line for line in shown):
                break
        assert text in shown, shown
        assert not any("\\[" in line for line in shown), shown


@pytest.mark.parametrize("command", ["/new", "/resume other"])
async def test_a_session_change_keeps_its_own_agent_load(
    monkeypatch, workspace, command
):
    """``/new`` and ``/resume`` restart the load; the agent the command was
    handed must not be put back in its place."""
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = rubric_gateway()
        gateway.thread_store.resolved_thread_id = "other"
        gateway.thread_store.metadata = {"workspace_dir": str(workspace.root)}
        prepare_app(app, workspace, gateway)
        previous = object()
        app._agent_loader.agent = previous
        adopted: list[object] = []
        monkeypatch.setattr(app._agent_loader, "adopt", adopted.append)
        restarts: list[object] = []
        start = app._start_background_agent_load

        def _restart(dirs):
            restarts.append(dirs)
            start(dirs)

        monkeypatch.setattr(app, "_start_background_agent_load", _restart)
        await submit(app, command)
        for _ in range(40):
            await pilot.pause()
            if restarts and not app._busy:
                break
        assert restarts, system_lines(app)
        assert adopted == []
        assert app._agent_loader.agent is not previous


async def test_a_command_that_builds_a_new_agent_still_has_it_adopted(
    monkeypatch, workspace
):
    """The other side of the rule above: ``/model`` hands a new agent back."""
    import EvoScientist.cli.tui_interactive as tui_mod

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        prepare_app(app, workspace, rubric_gateway())
        previous, built = object(), object()
        app._agent_loader.agent = previous
        adopted: list[object] = []
        monkeypatch.setattr(app._agent_loader, "adopt", adopted.append)

        class _Swap:
            name = "/swap"

            def needs_agent(self, _args):
                return False

        async def _execute(_command, ctx):
            ctx.agent = built
            return True

        monkeypatch.setattr(tui_mod.cmd_manager, "resolve", lambda _c: (_Swap(), []))
        monkeypatch.setattr(tui_mod.cmd_manager, "execute", _execute)
        await submit(app, "/swap")
        for _ in range(40):
            await pilot.pause()
            if adopted:
                break
        assert adopted == [built]
