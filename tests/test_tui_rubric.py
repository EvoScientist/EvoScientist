"""TUI side of ``/rubric``: the run marker, acknowledgement, badge and grading lines.

Boots the real TUI class through Textual's pilot (see
``test_tui_banner_position._capture_app``) and drives it with a fake gateway.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from tests.test_tui_banner_position import _capture_app
from tests.test_tui_skill_pin import (
    _finish_turns,
    _isolated_skills,  # noqa: F401  (autouse fixture)
    _prepare,
    _submit,
    _system_lines,
)


def _gateway(events=None, state_values=None, post_state=None):
    """A fake gateway; ``post_state`` becomes the thread state once a run finished."""
    from tests.fakes import FakeGraphGateway, FakeThreadStore

    class _Gateway(FakeGraphGateway):
        async def stream_events(self, request):
            async for event in super().stream_events(request):
                yield event
            if post_state is not None:
                self.state_values = dict(post_state)

    return _Gateway(
        events=events
        or [{"type": "text", "content": "ok"}, {"type": "done", "response": "ok"}],
        thread_store=FakeThreadStore(messages=[]),
        state_values=state_values or {},
    )


def _assistant_texts(app) -> list[str]:
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage

    out = []
    for w in app.query_one("#chat").children:
        if isinstance(w, AssistantMessage):
            out.append(str(getattr(w, "_content", None) or w.render()))
    return out


def _chat_sequence(app) -> list[tuple[str, str]]:
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage
    from EvoScientist.cli.widgets.system_message import SystemMessage

    seq = []
    for w in app.query_one("#chat").children:
        if isinstance(w, AssistantMessage):
            seq.append(("assistant", str(getattr(w, "_content", None) or w.render())))
        elif isinstance(w, SystemMessage):
            seq.append(("system", str(w.render())))
    return seq


async def test_typed_turn_carries_the_thread_marker_and_acknowledges(
    monkeypatch, workspace
):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        session = app._channel_runtime.rubric
        tid = app._conversation_tid
        op = session.record(tid, sticky="- s")
        gateway = _gateway(
            post_state={
                "_rubric_applied": {session.session_id: 1},
                "_sticky_rubric": {"text": "- s", "id": f"{session.session_id}:1"},
            }
        )
        _prepare(app, workspace, gateway)
        await _submit(app, "hello")
        await _finish_turns(app, pilot)
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
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test():
        gateway = _gateway()
        _prepare(app, workspace, gateway)
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

    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = _gateway()
        gateway.execute_gateway = MagicMock(spec=LangGraphServerGateway)
        _prepare(app, workspace, gateway)
        await _submit(app, "hello")
        await _finish_turns(app, pilot)
        extra = gateway.requests[-1].configurable_extra or {}
        assert "rubric_run" not in extra


async def test_grading_lines_sit_between_the_draft_and_the_revision(
    monkeypatch, workspace
):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = _gateway(
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
        _prepare(app, workspace, gateway)
        await _submit(app, "task")
        await _finish_turns(app, pilot)
        seq = _chat_sequence(app)
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
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = _gateway(
            events=[
                {"type": "text", "content": "started the job"},
                {"type": "rubric_evaluation_deferred", "tasks": 0, "processes": 1},
                {"type": "done", "response": "started the job"},
            ]
        )
        _prepare(app, workspace, gateway)
        await _submit(app, "task")
        await _finish_turns(app, pilot)
        lines = _system_lines(app)
        assert any(
            "Acceptance check deferred: 1 background process still running" in line
            for line in lines
        )


async def test_rubric_command_updates_the_badge_before_the_next_message(
    monkeypatch, workspace
):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        _prepare(app, workspace, _gateway())
        await _submit(app, "/rubric set - a")
        await pilot.pause()
        for _ in range(20):
            if app._rubric_badges.get(app._conversation_tid) == "on":
                break
            await pilot.pause()
        assert app._rubric_badges.get(app._conversation_tid) == "on"
        assert app._channel_runtime.rubric.changes_for(app._conversation_tid)


async def test_show_uses_the_loaded_agent_without_waiting(monkeypatch, workspace):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = _gateway(
            state_values={"_sticky_rubric": {"text": "- saved here", "id": "s:1"}}
        )
        _prepare(app, workspace, gateway)
        app._agent_loader.agent = object()
        await _submit(app, "/rubric show")
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
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = _gateway(
            state_values={"_sticky_rubric": {"text": "- saved", "id": "s:1"}}
        )
        _prepare(app, workspace, gateway)
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


async def test_a_late_grader_end_event_does_not_repeat_the_draft(
    monkeypatch, workspace
):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        gateway = _gateway(
            events=[
                {"type": "text", "content": "draft"},
                {
                    "type": "subagent_start",
                    "instance_id": "g1",
                    "name": "rubric_grader",
                    "description": "",
                    "tool_call_id": "tc-g1",
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
                {"type": "subagent_end", "instance_id": "g1", "name": "rubric_grader"},
                {"type": "done", "response": "draft\n\nfixed"},
            ]
        )
        _prepare(app, workspace, gateway)
        await _submit(app, "task")
        await _finish_turns(app, pilot)
        seq = _chat_sequence(app)
        verdict = next(
            i
            for i, (k, t) in enumerate(seq)
            if k == "system" and "not yet satisfied" in t
        )
        after = [t for k, t in seq[verdict:] if k == "assistant"]
        assert after, seq
        assert all("draft" not in t for t in after), after


async def test_checklist_text_is_shown_literally_in_the_tui(monkeypatch, workspace):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        text = "- [x] report.md exists\n- cite [docs] and keep [/tmp/out] intact"
        gateway = _gateway(state_values={"_sticky_rubric": {"text": text, "id": "s:1"}})
        _prepare(app, workspace, gateway)
        app._agent_loader.agent = object()
        await _submit(app, "/rubric show")
        shown: list[str] = []
        for _ in range(20):
            await pilot.pause()
            shown = [str(w.render()) for w in app.query_one("#chat").children]
            if any(text == line for line in shown):
                break
        assert text in shown, shown
        assert not any("\\[" in line for line in shown), shown
