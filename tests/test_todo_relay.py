"""The todo list a channel gets: only a list ``write_todos`` accepted, sent
once per turn at its successful result, from every path that relays it."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway

_END = [
    {"type": "text", "content": "finished"},
    {"type": "done", "response": "finished"},
]


def _write(call_id: str, content: str, success: bool = True) -> list[dict]:
    return [
        {
            "type": "tool_call",
            "id": call_id,
            "name": "write_todos",
            "args": {"todos": [{"status": "pending", "content": content}]},
        },
        {
            "type": "tool_result",
            "id": call_id,
            "name": "write_todos",
            "content": "Updated todo list" if success else "Error: invalid status",
            "success": success,
        },
    ]


# (writes as (content, success), the lists the channel gets)
CASES = [
    pytest.param([("A", True)], [["A"]], id="one_write"),
    pytest.param([("A", False)], [], id="rejected_only"),
    pytest.param([("A", True), ("B", False)], [["A"]], id="ok_then_rejected"),
    pytest.param([("A", False), ("B", True)], [["B"]], id="rejected_then_ok"),
    pytest.param([("A", True), ("B", True)], [["A"]], id="first_list_only"),
]


def _turn(writes) -> list[dict]:
    events: list[dict] = []
    for i, (content, success) in enumerate(writes):
        events += _write(f"t{i}", content, success)
    return [*events, *_END]


def _record(sent: list):
    return lambda items: sent.append([item["content"] for item in items])


class TestRichLoop:
    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

    @pytest.mark.parametrize(("writes", "expected"), CASES)
    def test_the_accepted_list_is_sent(self, writes, expected):
        sent: list = []
        display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=False,
            on_todo=_record(sent),
            gateway=FakeGraphGateway(events=_turn(writes)),
            _state=display_mod.StreamState(),
        )
        assert sent == expected


class TestTui:
    @pytest.fixture(autouse=True)
    def _textual(self):
        pytest.importorskip("textual")

    @pytest.mark.parametrize(("writes", "expected"), CASES)
    async def test_the_accepted_list_is_sent(
        self, monkeypatch, workspace, writes, expected
    ):
        from tests.fakes.tui import run_turn

        sent: list = []
        await run_turn(
            monkeypatch, workspace, [_turn(writes)], on_todo_cb=_record(sent)
        )
        assert sent == expected


async def test_tui_task_list_goes_above_the_call_that_wrote_it(monkeypatch, workspace):
    pytest.importorskip("textual")
    from EvoScientist.cli.widgets.todo_widget import TodoWidget
    from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget
    from tests.fakes.tui import run_turn

    orders: list = []

    def watch_chat(app):
        chat = app.query_one("#chat")
        mount = chat.mount

        async def mount_and_record(*widgets, **kwargs):
            await mount(*widgets, **kwargs)
            orders.append(
                [
                    type(w)
                    for w in chat.children
                    if isinstance(w, (TodoWidget, ToolCallWidget))
                ]
            )

        monkeypatch.setattr(chat, "mount", mount_and_record)

    await run_turn(monkeypatch, workspace, [_turn([("A", True)])], prepare=watch_chat)

    assert orders[-1] == [TodoWidget, ToolCallWidget]
