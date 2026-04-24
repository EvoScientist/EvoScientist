from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from EvoScientist.commands.channel_ui import ChannelCommandUI
from tests.conftest import run_async as _run


def _make_ui(callback=None, bus_ref=None):
    captured: list[str] = []
    ui = ChannelCommandUI(
        SimpleNamespace(
            channel_type="fake",
            chat_id="chat-1",
            message_id="msg-1",
            metadata={},
            bus_ref=bus_ref,
            channel_ref=None,
        ),
        append_system_callback=lambda text, style="dim": captured.append(text),
        handle_session_resume_callback=callback,
    )
    return ui, captured


async def _run_resume(ui, thread_id: str, workspace_dir: str):
    loop = asyncio.get_running_loop()
    scheduled: list[asyncio.Task] = []

    def _schedule(coro, _loop):
        task = loop.create_task(coro)
        scheduled.append(task)
        return task

    with (
        patch("EvoScientist.cli.channel._bus_loop", new=loop),
        patch(
            "EvoScientist.commands.channel_ui.asyncio.run_coroutine_threadsafe",
            side_effect=_schedule,
        ),
    ):
        await ui.handle_session_resume(thread_id, workspace_dir)
        if scheduled:
            await asyncio.gather(*scheduled)


def _sent_text(bus_ref) -> str:
    return "\n".join(
        call.args[0].content for call in bus_ref.publish_outbound.await_args_list
    )


def test_handle_session_resume_sends_history_back_to_channel_without_local_duplicate():
    callback = AsyncMock()
    bus_ref = SimpleNamespace(publish_outbound=AsyncMock())
    ui, captured = _make_ui(callback=callback, bus_ref=bus_ref)

    messages = [
        SimpleNamespace(type="human", content="How does this work?"),
        SimpleNamespace(type="ai", content="Here is the saved answer."),
    ]

    with patch(
        "EvoScientist.sessions.get_thread_messages",
        new=AsyncMock(return_value=messages),
    ):
        _run(_run_resume(ui, "thread-42", "/workspace"))

    callback.assert_awaited_once_with("thread-42", "/workspace")
    assert captured == []
    text = _sent_text(bus_ref)
    assert "Resumed session: thread-42" in text
    assert "Conversation history:" in text
    assert "User: How does this work?" in text
    assert "EvoScientist: Here is the saved answer." in text


def test_handle_session_resume_reports_history_load_error():
    callback = AsyncMock()
    bus_ref = SimpleNamespace(publish_outbound=AsyncMock())
    ui, captured = _make_ui(callback=callback, bus_ref=bus_ref)

    with patch(
        "EvoScientist.sessions.get_thread_messages",
        new=AsyncMock(side_effect=RuntimeError("db locked")),
    ):
        _run(_run_resume(ui, "thread-42", "/workspace"))

    callback.assert_awaited_once_with("thread-42", "/workspace")
    assert captured == []
    text = _sent_text(bus_ref)
    assert "Resumed session: thread-42" in text
    assert "history unavailable: db locked" in text


def test_handle_session_resume_distinguishes_non_displayable_messages():
    bus_ref = SimpleNamespace(publish_outbound=AsyncMock())
    ui, captured = _make_ui(bus_ref=bus_ref)

    with patch(
        "EvoScientist.sessions.get_thread_messages",
        new=AsyncMock(return_value=[SimpleNamespace(type="tool", content="hidden")]),
    ):
        _run(_run_resume(ui, "thread-42", "/workspace"))

    assert captured == [
        "Resumed session: thread-42\nNo displayable messages in this session."
    ]
    text = _sent_text(bus_ref)
    assert "No displayable messages in this session." in text
