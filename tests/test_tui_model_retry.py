"""TUI side of a retried model call: the failed attempt's widgets go, one line says why."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from tests.test_tui_banner_position import _capture_app
from tests.test_tui_rubric import _assistant_texts, _chat_sequence, _gateway
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
