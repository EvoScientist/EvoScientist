"""The todo list a channel gets: only a list ``write_todos`` accepted, sent
once per approval round at its successful result, from every path that
relays it."""

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


def _two_rounds(pause: dict):
    """A turn whose first round writes A and then pauses (*pause*), and whose
    second round has a write of B the tool rejects."""
    calls = 0

    async def _stream(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            for event in [*_write("t1", "A"), pause]:
                yield event
            return
        for event in [*_write("t2", "B", success=False), *_END]:
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

    def test_a_rejected_write_after_a_pause_resends_nothing(self, monkeypatch):
        """The turn's state keeps the first list across the approval pause,
        while the sent flag resets per round."""
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
            gateway=FakeGraphGateway(stream=_two_rounds(interrupt_event(1))),
            _state=display_mod.StreamState(),
        )
        assert sent == [["A"]]


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


class TestStandaloneConsumer:
    """``EvoSci channel --agent`` relays through ``InboundConsumer``."""

    @pytest.mark.parametrize(("writes", "expected"), CASES)
    async def test_the_accepted_list_is_sent(
        self, monkeypatch, workspace, writes, expected
    ):
        sent = await self._run(monkeypatch, workspace, events=_turn(writes))
        assert sent == [_listed(c) for [c] in expected]

    async def test_a_rejected_write_after_a_pause_resends_nothing(
        self, monkeypatch, workspace
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
        sent = await self._run(monkeypatch, workspace, stream=_two_rounds(ask))
        assert sent == [_listed("A")]

    @staticmethod
    async def _run(monkeypatch, workspace, **gateway) -> list[str]:
        from EvoScientist.channels.bus.events import InboundMessage
        from tests.test_channel_media_forwarding import _consumer

        sent: list[str] = []
        consumer = _consumer(workspace, FakeGraphGateway(**gateway))
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
