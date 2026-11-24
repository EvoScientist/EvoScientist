"""Regression tests for Textual TUI stream ownership."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from EvoScientist.cli.widgets.loading_widget import LoadingWidget
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.middleware import events as mw_events
from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway
from tests.fakes.tui import capture_app


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

    app = await capture_app(monkeypatch, workspace)
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


async def test_a_stop_before_the_run_removes_the_spinner(monkeypatch, workspace):
    """A /stop that lands before the run's first event leaves no spinner."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway()  # must never be asked for a stream
    scope = "test-channel:chat-1"
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._runtime_gateways = RuntimeGateways(
                thread_store=gateway.thread_store, graph_gateway=gateway
            )
            monkeypatch.setattr(
                app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
            )
            display_mod.request_stream_cancel(scope)
            response = await app._stream_with_widgets("hello", cancel_scope=scope)
            await pilot.pause()
            assert response == "[Stopped.]"
            assert gateway.requests == []
            assert not list(app.query(LoadingWidget))
    finally:
        display_mod.discard_stream_cancel(scope)


async def test_an_agent_wait_error_removes_the_spinner(monkeypatch, workspace):
    """A turn whose agent wait fails still takes its spinner down."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(
            app,
            "_await_agent_ready",
            AsyncMock(side_effect=RuntimeError("load failed")),
        )
        with pytest.raises(RuntimeError, match="load failed"):
            await app._stream_with_widgets("hello")
        await pilot.pause()
        assert gateway.requests == []
        assert not list(app.query(LoadingWidget))


async def test_an_agent_wait_cancelled_removes_the_spinner(monkeypatch, workspace):
    """A turn cancelled in the agent wait still takes its spinner down."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(
            app,
            "_await_agent_ready",
            AsyncMock(side_effect=asyncio.CancelledError()),
        )
        with pytest.raises(asyncio.CancelledError):
            await app._stream_with_widgets("hello")
        await pilot.pause()
        assert gateway.requests == []
        assert not list(app.query(LoadingWidget))


async def test_a_status_refresh_during_teardown_is_ignored(monkeypatch, workspace):
    """The 1 s status refresh can fire while the app is being torn down."""
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
    app._render_status()
