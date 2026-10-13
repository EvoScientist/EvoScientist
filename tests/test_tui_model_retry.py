"""TUI side of a retried model call: the failed attempt's widgets go, one line says why."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from tests.test_tui_banner_position import _capture_app
from tests.test_tui_rubric import (
    _assistant_texts,
    _chat_sequence,
    _Gated,
    _gateway,
    _run_gated,
)
from tests.test_tui_skill_pin import (
    _finish_turns,
    _isolated_skills,  # noqa: F401  (autouse fixture)
    _prepare,
    _submit,
)


def _retry(discarded, instance_id=""):
    return {
        "type": "model_retry",
        "attempt": 2,
        "wait_seconds": 8.0,
        "reason": "503: starting up",
        "budget_left_seconds": 50.0,
        "cause": "exception",
        "subagent": "",
        "instance_id": instance_id,
        "discarded": discarded,
    }


async def test_discarded_calls_leave_the_box_bookkeeping():
    """A dropped call must not keep a slot among the rows shown, be counted
    in the summary of the hidden ones or in the closed box's header."""
    from textual.app import App
    from textual.widgets import Static

    from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget
    from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

    class _Host(App):
        def compose(self):
            yield SubAgentWidget("code-agent")

    app = _Host()
    async with app.run_test() as pilot:
        box = app.query_one(SubAgentWidget)
        for call in ("done-1", "done-2"):
            await box.add_tool_call("read_file", {"file_path": f"/{call}"}, call)
            box.complete_tool("read_file", "ok", True, call)
        for call in ("a", "b", "c"):
            await box.add_tool_call("execute", {"command": call}, call)
        await pilot.pause()

        await box.discard_tool_calls(["b", "done-1"])
        await pilot.pause()

        assert sorted(box._tool_widgets) == ["a", "c", "done-2"]
        assert (box._running_ids, box._completed_ids) == (["a", "c"], ["done-2"])
        shown = {w._tool_id: bool(w.display) for w in app.query(ToolCallWidget)}
        assert shown == {"a": True, "c": True, "done-2": True}
        assert not box.query_one(".sa-collapse-summary", Static).has_class("--visible")

        # The same id emitted again is a new call, listed once.
        await box.add_tool_call("execute", {"command": "b"}, "b")
        assert box._running_ids == ["a", "c", "b"]

        box.finalize()
        assert "(4 tools)" in str(box.query_one(".sa-header", Static).render())


async def test_retry_removes_the_failed_attempt_and_mounts_the_notice(
    monkeypatch, workspace
):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        events = [
            {"type": "thinking", "content": "plan "},
            {"type": "text", "content": "Half a sen"},
            {"type": "tool_call", "id": "dead", "name": "ls", "args": {}},
            _retry(
                {
                    "text_chars": len("Half a sen"),
                    "thinking_chars": 0,
                    "tool_call_ids": ["dead"],
                }
            ),
            {"type": "text", "content": "Full answer."},
            {"type": "done", "response": "Full answer."},
        ]
        _prepare(app, workspace, _gateway(events=events))
        await _submit(app, "go")
        await _finish_turns(app, pilot)
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        assert not [
            w for w in app.query_one("#chat").children if isinstance(w, ToolCallWidget)
        ]
        assert _assistant_texts(app)[-1] == "Full answer."
        assert "Half a sen" not in "".join(_assistant_texts(app))
        assert (
            "system",
            "Model request failed (503: starting up); retrying in 8 s (attempt 2)",
        ) in _chat_sequence(app)


async def test_retry_trims_thinking_instead_of_removing_it(monkeypatch, workspace):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        events = [
            {"type": "thinking", "content": "keep this "},
            {"type": "thinking", "content": "drop this"},
            _retry(
                {
                    "text_chars": 0,
                    "thinking_chars": len("drop this"),
                    "tool_call_ids": [],
                }
            ),
            {"type": "text", "content": "ok"},
            {"type": "done", "response": "ok"},
        ]
        _prepare(app, workspace, _gateway(events=events))
        await _submit(app, "go")
        await _finish_turns(app, pilot)
        from EvoScientist.cli.widgets.thinking_widget import ThinkingWidget

        thinking = [
            w for w in app.query_one("#chat").children if isinstance(w, ThinkingWidget)
        ]
        assert len(thinking) == 1
        assert thinking[0]._content == "keep this "


async def test_discard_without_retry_mounts_no_line(monkeypatch, workspace):
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        events = [
            {"type": "text", "content": "partial "},
            {
                "type": "model_output_discarded",
                "instance_id": "",
                "discarded": {
                    "text_chars": 8,
                    "thinking_chars": 0,
                    "tool_call_ids": [],
                },
            },
            {"type": "text", "content": "complete"},
            {"type": "done", "response": "complete"},
        ]
        _prepare(app, workspace, _gateway(events=events))
        await _submit(app, "go")
        await _finish_turns(app, pilot)
        assert _assistant_texts(app)[-1] == "complete"
        assert not [
            s for s in _chat_sequence(app) if s[0] == "system" and "retrying" in s[1]
        ]


async def test_a_retry_before_any_content_keeps_the_spinner_last(
    monkeypatch, workspace
):
    from EvoScientist.cli.widgets.loading_widget import LoadingWidget

    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        nothing = {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []}
        gated = _Gated(
            _retry(nothing),
            "waiting",
            {"type": "text", "content": "Answer."},
            {"type": "done", "response": "Answer."},
        )
        await _run_gated(app, workspace, gated)

        *_, notice, last = await gated.chat_at(app, pilot, "waiting")
        assert str(notice.render()).startswith("Model request failed")
        assert isinstance(last, LoadingWidget)
        gated.go_on("waiting")
        await _finish_turns(app, pilot)
        assert not list(app.query(LoadingWidget))


async def test_a_subagent_retry_notice_follows_its_box(monkeypatch, workspace):
    from EvoScientist.cli.widgets.loading_widget import LoadingWidget
    from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        nothing = {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []}
        gated = _Gated(
            # No root content yet, so the turn's first spinner is still up.
            {
                "type": "subagent_start",
                "instance_id": "h1",
                "name": "helper",
                "description": "",
                "tool_call_id": "tc-h1",
            },
            {**_retry(nothing, "h1"), "subagent": "helper"},
            "waiting",
            {"type": "subagent_end", "instance_id": "h1", "name": "helper"},
            {"type": "text", "content": "Answer."},
            {"type": "done", "response": "Answer."},
        )
        await _run_gated(app, workspace, gated)

        *_, box, notice, last = await gated.chat_at(app, pilot, "waiting")
        assert isinstance(box, SubAgentWidget)
        assert str(notice.render()).startswith("helper: Model request failed")
        assert isinstance(last, LoadingWidget)
        gated.go_on("waiting")
        await _finish_turns(app, pilot)
        assert not list(app.query(LoadingWidget))


async def test_a_retry_notice_follows_the_tool_selection_line(monkeypatch, workspace):
    from EvoScientist.cli.widgets.loading_widget import LoadingWidget
    from EvoScientist.cli.widgets.tool_selection_widget import ToolSelectionWidget

    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test() as pilot:
        nothing = {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []}
        gated = _Gated(
            # The selection line leaves the turn's first spinner up, above it.
            {"type": "tool_selection", "tools": ["ls", "read_file"]},
            _retry(nothing),
            "waiting",
            {"type": "text", "content": "Answer."},
            {"type": "done", "response": "Answer."},
        )
        await _run_gated(app, workspace, gated)

        *_, selection, notice, last = await gated.chat_at(app, pilot, "waiting")
        assert isinstance(selection, ToolSelectionWidget)
        assert str(notice.render()).startswith("Model request failed")
        assert isinstance(last, LoadingWidget)
        gated.go_on("waiting")
        await _finish_turns(app, pilot)
        assert not list(app.query(LoadingWidget))


async def test_a_retry_after_a_pause_that_found_the_spinner_up(monkeypatch, workspace):
    from tests.test_tui_approve_all import DONE, _turn
    from tests.tui_fakes import approval_interrupt as _interrupt

    # The paused round's cleanup removed the spinner; the resumed round has
    # none to keep last. ``_turn`` asserts that the answer still arrives.
    nothing = {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []}
    await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [_retry(nothing), *DONE]],
        prepare=lambda app: setattr(app, "_hitl_auto_approve", True),
    )
