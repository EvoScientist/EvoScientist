from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from EvoScientist.commands.channel_ui import ChannelCommandUI
from tests.conftest import run_async as _run


def _make_ui(callback=None):
    captured: list[str] = []
    ui = ChannelCommandUI(
        SimpleNamespace(
            channel_type="fake",
            chat_id="chat-1",
            message_id="msg-1",
            metadata={},
            bus_ref=None,
            channel_ref=None,
        ),
        append_system_callback=lambda text, style="dim": captured.append(text),
        handle_session_resume_callback=callback,
    )
    return ui, captured


def test_wait_for_thread_pick_lists_thread_ids_for_channel_users():
    ui, captured = _make_ui()

    result = _run(
        ui.wait_for_thread_pick(
            [
                {
                    "thread_id": "abc12345",
                    "preview": "First prompt",
                    "message_count": 3,
                    "updated_at": None,
                },
                {
                    "thread_id": "def67890",
                    "preview": "Second prompt",
                    "message_count": 1,
                    "updated_at": None,
                },
            ],
            current_thread="abc12345",
            title=">>> Select session to resume <<<",
        )
    )

    assert result is None
    text = "\n".join(captured)
    assert "abc12345" in text
    assert "def67890" in text
    assert "/resume <id>" in text


def test_handle_session_resume_sends_history_back_to_channel():
    callback = AsyncMock()
    ui, captured = _make_ui(callback=callback)

    messages = [
        SimpleNamespace(type="human", content="How does this work?"),
        SimpleNamespace(type="ai", content="Here is the saved answer."),
    ]

    with patch(
        "EvoScientist.sessions.get_thread_messages",
        new=AsyncMock(return_value=messages),
    ):
        _run(ui.handle_session_resume("thread-42", "/workspace"))

    callback.assert_awaited_once_with("thread-42", "/workspace")
    text = "\n".join(captured)
    assert "Resumed session: thread-42" in text
    assert "Conversation history:" in text
    assert "User: How does this work?" in text
    assert "EvoScientist: Here is the saved answer." in text
