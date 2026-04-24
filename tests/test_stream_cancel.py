"""Tests for /stop channel-initiated stream cancellation.

Covers:
- ``_consume`` observes the cancel event and terminates with ``[Stopped.]``
- ``_run_streaming`` clears a stale set event on fresh entry
- Bus handler intercepts ``/stop`` (and ``/cancel``) without enqueueing
- ``StopCommand`` slash command sets the event and is idempotent
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.channels.bus.events import InboundMessage
from EvoScientist.stream import display as display_mod
from tests.conftest import run_async as _run


@pytest.fixture(autouse=True)
def _clean_cancel_event():
    """Ensure all stream-cancel scopes start clear for every test."""
    with display_mod._stream_cancel_lock:
        display_mod._stream_cancel_event.clear()
        display_mod._stream_cancel_events.clear()
        display_mod._stream_cancel_events[display_mod._DEFAULT_STREAM_CANCEL_SCOPE] = (
            display_mod._stream_cancel_event
        )
    yield
    with display_mod._stream_cancel_lock:
        display_mod._stream_cancel_event.clear()
        display_mod._stream_cancel_events.clear()
        display_mod._stream_cancel_events[display_mod._DEFAULT_STREAM_CANCEL_SCOPE] = (
            display_mod._stream_cancel_event
        )


@pytest.fixture(autouse=True)
def _clean_channel_queue():
    """Drain the channel message queue before and after each test."""
    from EvoScientist.cli import channel as channel_mod

    def _drain():
        while not channel_mod._message_queue.empty():
            try:
                channel_mod._message_queue.get_nowait()
            except Exception:
                break

    _drain()
    with channel_mod._response_lock:
        channel_mod._pending_responses.clear()
    with channel_mod._channel_request_lock:
        channel_mod._channel_requests.clear()
        channel_mod._session_requests.clear()
        channel_mod._cancelled_channel_messages.clear()
    yield
    _drain()
    with channel_mod._response_lock:
        channel_mod._pending_responses.clear()
    with channel_mod._channel_request_lock:
        channel_mod._channel_requests.clear()
        channel_mod._session_requests.clear()
        channel_mod._cancelled_channel_messages.clear()


# ---------------------------------------------------------------------------
# 1. _consume breaks on cancel event
# ---------------------------------------------------------------------------


def test_consume_breaks_on_cancel_event():
    """After set(), ``_consume`` should stop pulling events and mark
    ``state.response_text`` with the ``[Stopped.]`` suffix."""
    seen_events: list[int] = []
    cancel_scope = "scope:consume"

    async def _fake_stream(agent, message, thread_id, **kwargs):
        for i in range(100):
            if i == 3:
                # Set during iteration — next loop iter should bail.
                display_mod.request_stream_cancel(cancel_scope)
            seen_events.append(i)
            yield {"type": "text", "content": f"chunk-{i}"}

    with patch(
        "EvoScientist.stream.display.stream_agent_events",
        new=_fake_stream,
    ):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope=cancel_scope,
        )

    # We set the flag during event index 3; the cancel check runs at the
    # top of the NEXT iteration (index 4), so indices 0-3 are pulled from
    # the generator before exit.
    assert len(seen_events) <= 5
    assert "[Stopped.]" in result


# ---------------------------------------------------------------------------
# 2. fresh _run_streaming clears stale set event
# ---------------------------------------------------------------------------


def test_run_streaming_short_circuits_when_scope_already_cancelled():
    """A queued request that is cancelled before start should stop immediately."""
    seen_event = False
    cancel_scope = "scope:queued"

    async def _fake_stream(agent, message, thread_id, **kwargs):
        nonlocal seen_event
        seen_event = True
        yield {"type": "text", "content": "ok"}

    display_mod.request_stream_cancel(cancel_scope)

    with patch(
        "EvoScientist.stream.display.stream_agent_events",
        new=_fake_stream,
    ):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope=cancel_scope,
        )

    assert result == "[Stopped.]"
    assert seen_event is False


def test_run_streaming_ignores_other_scope_cancel():
    """Cancelling one scope must not bleed into a different stream."""
    display_mod.request_stream_cancel("scope:other")

    async def _fake_stream(agent, message, thread_id, **kwargs):
        yield {"type": "text", "content": "ok"}

    with patch(
        "EvoScientist.stream.display.stream_agent_events",
        new=_fake_stream,
    ):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope="scope:self",
        )

    assert "[Stopped.]" not in result


# ---------------------------------------------------------------------------
# 3. bus /stop intercept — sets event, sends ack, does NOT enqueue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", ["/stop", "/cancel", "  /STOP  "])
def test_bus_handle_message_stop_intercept(cmd):
    """`_handle_bus_message` should short-circuit on /stop and /cancel
    (case-insensitive, whitespace-tolerant): set the cancel event, call
    bus.publish_outbound once with 'Stopped.', and leave the main-thread
    message queue untouched."""
    from EvoScientist.cli.channel import _handle_bus_message, _message_queue

    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    manager = MagicMock()
    manager.get_channel = MagicMock(return_value=MagicMock())

    inbound = InboundMessage(
        channel="fake",
        sender_id="user1",
        chat_id="chat1",
        content=cmd,
        message_id="m-1",
    )

    _run(_handle_bus_message(bus, manager, inbound))

    # Stop ack is immediate, but unrelated default streams stay untouched.
    assert not display_mod.is_stream_cancel_requested()

    # Ack sent, exactly once, content = "Stopped.".
    assert bus.publish_outbound.call_count == 1
    ack = bus.publish_outbound.call_args.args[0]
    assert ack.channel == "fake"
    assert ack.chat_id == "chat1"
    assert ack.content == "Stopped."
    assert ack.reply_to == "m-1"

    # Nothing was pushed to the main-thread queue.
    assert _message_queue.empty()


def test_bus_handle_message_non_stop_falls_through():
    """Plain content should NOT trigger the stop fast-path: the cancel
    event stays clear and the message is enqueued normally."""
    from EvoScientist.cli.channel import _handle_bus_message, _message_queue

    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    manager = MagicMock()
    channel_mock = MagicMock()
    channel_mock.start_typing = AsyncMock()
    channel_mock.stop_typing = AsyncMock()
    manager.get_channel = MagicMock(return_value=channel_mock)

    async def _run_with_cancel():
        task = asyncio.create_task(
            _handle_bus_message(
                bus,
                manager,
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="hello",
                ),
            )
        )
        # Wait for enqueue to happen, then cancel to tear down the wait.
        for _ in range(20):
            if not _message_queue.empty():
                break
            await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _run(_run_with_cancel())

    assert not display_mod.is_stream_cancel_requested()
    # Non-stop path reaches the enqueue.
    assert not _message_queue.empty()


# ---------------------------------------------------------------------------
# 4 & 5. StopCommand — execute sets event; second call is idempotent
# ---------------------------------------------------------------------------


def test_stop_command_execute_sets_event():
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.stop import StopCommand

    ui = MagicMock()
    ctx = CommandContext(agent=None, thread_id="t", ui=ui)

    _run(StopCommand().execute(ctx, []))

    assert display_mod.is_stream_cancel_requested()
    assert ui.append_system.call_count == 1
    msg = ui.append_system.call_args.args[0]
    assert "Stop requested" in msg


def test_stop_command_idempotent_when_already_requested():
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.stop import StopCommand

    display_mod.request_stream_cancel()
    ui = MagicMock()
    ctx = CommandContext(agent=None, thread_id="t", ui=ui)

    _run(StopCommand().execute(ctx, []))

    assert display_mod.is_stream_cancel_requested()
    msg = ui.append_system.call_args.args[0]
    assert "already requested" in msg.lower()


# ---------------------------------------------------------------------------
# 6. pending HITL/ask_user branches short-circuit when stop is requested
# ---------------------------------------------------------------------------


def test_run_streaming_pending_interrupt_short_circuits_on_cancel():
    """If cancel is already set, pending HITL prompt should not run."""

    async def _empty_stream(agent, message, thread_id, **kwargs):
        if False:
            yield {}

    state = display_mod.StreamState()
    state.pending_interrupt = {
        "action_requests": [{"name": "execute", "args": {"command": "echo hi"}}]
    }
    display_mod.request_stream_cancel("scope:hitl")

    prompt_called = False

    def _prompt(_requests):
        nonlocal prompt_called
        prompt_called = True
        return None

    with patch("EvoScientist.stream.display.stream_agent_events", new=_empty_stream):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            hitl_prompt_fn=_prompt,
            cancel_scope="scope:hitl",
            _state=state,
        )

    assert result == "[Stopped.]"
    assert prompt_called is False
