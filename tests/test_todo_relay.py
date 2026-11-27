"""The todo list a channel gets: only a list ``write_todos`` accepted, sent
once per approval round at its successful result, from every path that
relays it."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway, stub_consumer

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


def _clear(call_id: str) -> list[dict]:
    """An accepted write that empties the list."""
    return [
        {
            "type": "tool_call",
            "id": call_id,
            "name": "write_todos",
            "args": {"todos": []},
        },
        {
            "type": "tool_result",
            "id": call_id,
            "name": "write_todos",
            "content": "Updated todo list to []",
            "success": True,
        },
    ]


# (the round after A and a pause, the lists the channel gets in the turn)
SECOND_ROUNDS = [
    pytest.param(_write("t2", "B"), [["A"], ["B"]], id="accepted"),
    pytest.param(_write("t2", "B", success=False), [["A"]], id="rejected"),
    pytest.param(_clear("t2"), [["A"]], id="cleared"),
    pytest.param(
        [*_clear("t2"), *_write("t3", "B")], [["A"], ["B"]], id="cleared_then_accepted"
    ),
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


def _two_rounds(pause: dict, second: list[dict]):
    """A turn whose first round writes A and then pauses (*pause*), and whose
    second round streams *second*."""
    calls = 0

    async def _stream(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            for event in [*_write("t1", "A"), pause]:
                yield event
            return
        for event in [*second, *_END]:
            yield event

    return _stream


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

    @pytest.mark.parametrize(("second", "expected"), SECOND_ROUNDS)
    def test_each_round_sends_its_first_accepted_list(
        self, monkeypatch, second, expected
    ):
        """The turn's state spans the approval pause, while the sent flag
        resets per round."""
        from tests.fakes.common import interrupt_event

        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        sent: list = []
        display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            on_todo=_record(sent),
            gateway=FakeGraphGateway(stream=_two_rounds(interrupt_event(1), second)),
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

    @pytest.mark.parametrize(("second", "expected"), SECOND_ROUNDS)
    async def test_each_round_sends_its_first_accepted_list(
        self, monkeypatch, workspace, second, expected
    ):
        from tests.fakes.tui import approval_interrupt, run_turn

        sent: list = []
        await run_turn(
            monkeypatch,
            workspace,
            [[*_write("t1", "A"), approval_interrupt(1)], [*second, *_END]],
            prepare=lambda app: setattr(app, "_hitl_auto_approve", True),
            on_todo_cb=_record(sent),
        )
        assert sent == expected


async def test_tui_task_list_empties_when_the_list_is_cleared(monkeypatch, workspace):
    pytest.importorskip("textual")
    from EvoScientist.cli.widgets.todo_widget import TodoWidget
    from tests.fakes.tui import run_turn

    shown: list = []
    update_items = TodoWidget.update_items

    def record(self, items):
        shown.append(list(items))
        update_items(self, items)

    monkeypatch.setattr(TodoWidget, "update_items", record)
    await run_turn(monkeypatch, workspace, [[*_write("t1", "A"), *_clear("t2"), *_END]])

    assert shown == [[]]


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


class TestStandaloneConsumer:
    """``EvoSci channel --agent`` relays through ``InboundConsumer``."""

    @pytest.mark.parametrize(("writes", "expected"), CASES)
    async def test_the_accepted_list_is_sent(
        self, monkeypatch, workspace, writes, expected
    ):
        sent = await self._run(monkeypatch, workspace, events=_turn(writes))
        assert sent == [_listed(c) for [c] in expected]

    @pytest.mark.parametrize(("second", "expected"), SECOND_ROUNDS)
    async def test_each_round_sends_its_first_accepted_list(
        self, monkeypatch, workspace, second, expected
    ):
        import EvoScientist.channels.consumer as consumer_mod

        async def _answer(_questions, _io, timeout=0):
            return {"answers": ["yes"], "status": "answered"}

        monkeypatch.setattr(consumer_mod, "resolve_ask_user", _answer)
        ask = {
            "type": "ask_user",
            "interrupt_id": "ask-1",
            "questions": [{"question": "Continue?"}],
        }
        sent = await self._run(monkeypatch, workspace, stream=_two_rounds(ask, second))
        assert sent == [_listed(c) for [c] in expected]

    @staticmethod
    async def _run(monkeypatch, workspace, **gateway) -> list[str]:
        from EvoScientist.channels.bus.events import InboundMessage

        sent: list[str] = []
        consumer = stub_consumer(workspace, FakeGraphGateway(**gateway))
        channel = consumer.manager.get_channel("stub")

        async def record(_sender, content, _metadata=None):
            sent.append(content)

        monkeypatch.setattr(channel, "send_todo_message", record)
        msg = InboundMessage(channel="stub", sender_id="u1", chat_id="c1", content="go")
        await consumer._stream_with_hitl(msg, channel, "t1", msg.session_key)
        return sent


def _listed(content: str) -> str:
    from EvoScientist.channels.consumer import _format_todo_list

    return _format_todo_list([{"status": "pending", "content": content}])
