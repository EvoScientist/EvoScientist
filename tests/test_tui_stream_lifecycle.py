"""Regression tests for Textual TUI stream ownership."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from EvoScientist.gateway import RuntimeGateways
from EvoScientist.middleware import events as mw_events
from tests.fakes import FakeGraphGateway
from tests.test_tui_banner_position import _capture_app


@pytest.mark.parametrize("pause", ["interrupt", "ask_user"])
async def test_pause_closes_stream_in_tui_turn_context(monkeypatch, workspace, pause):
    """A TUI pause must release the run sink before the resumed round."""
    stream_calls = 0
    started_in: list[asyncio.Task | None] = []
    closed_in: list[asyncio.Task | None] = []
    sink_at_start: list[object] = []
    loop_errors: list[dict] = []

    async def _fake_stream(_request):
        """Bind the sink until the owning TUI task closes this stream."""
        nonlocal stream_calls
        stream_calls += 1
        started_in.append(asyncio.current_task())
        sink_at_start.append(mw_events._current_run_event_sink.get())
        token = mw_events.bind_run_event_sink(MagicMock())
        try:
            if stream_calls == 1:
                if pause == "interrupt":
                    yield {
                        "type": "interrupt",
                        "interrupt_id": "approval-1",
                        "action_requests": [
                            {"name": "execute", "args": {"command": "echo ok"}}
                        ],
                    }
                else:
                    yield {
                        "type": "ask_user",
                        "interrupt_id": "question-1",
                        "questions": [{"question": "Continue?"}],
                    }
                await asyncio.Event().wait()
            else:
                yield {"type": "text", "content": "final answer"}
                yield {"type": "done", "response": "final answer"}
        finally:
            closed_in.append(asyncio.current_task())
            mw_events.reset_run_event_sink(token)

    app = await _capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_fake_stream)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store,
            graph_gateway=gateway,
        )
        monkeypatch.setattr(
            app,
            "_await_agent_ready",
            AsyncMock(return_value=MagicMock()),
        )
        if pause == "interrupt":
            app._hitl_auto_approve = True

        loop = asyncio.get_running_loop()
        previous_exception_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            response = await app._stream_with_widgets(
                "hello",
                channel_ask_user_fn=(
                    lambda _event: (
                        {"answers": ["yes"], "status": "answered"}
                        if pause == "ask_user"
                        else None
                    )
                ),
            )
        finally:
            loop.set_exception_handler(previous_exception_handler)

    assert response == "final answer"
    assert stream_calls == 2
    assert closed_in == started_in
    assert sink_at_start == [None, None]
    assert loop_errors == []
