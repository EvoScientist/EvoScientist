"""Tests for bus-mode queue bridge (_bus_inbound_consumer).

The consumer no longer calls the agent directly.  Instead it enqueues a
``ChannelMessage`` on a thread-safe queue and waits for the main CLI
thread to set a response via ``_set_channel_response()``.
"""

import asyncio

import pytest

from EvoScientist.channels.base import Channel, OutgoingMessage
from EvoScientist.channels.bus.events import InboundMessage
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from tests.conftest import run_async as _run


def _drain_queue(q):
    """Drain a queue.Queue before a test to avoid cross-test leaks."""
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break


@pytest.fixture(autouse=True)
def clean_channel_state():
    """Reset shared channel bridge state before and after each test."""
    from EvoScientist.cli import channel as channel_mod
    from EvoScientist.cli.channel import _message_queue
    from EvoScientist.stream import display as display_mod

    def _reset() -> None:
        _drain_queue(_message_queue)
        with channel_mod._response_lock:
            channel_mod._pending_responses.clear()
        with channel_mod._channel_request_lock:
            channel_mod._channel_requests.clear()
            channel_mod._session_requests.clear()
            channel_mod._cancelled_channel_messages.clear()
        with channel_mod._hitl_lock:
            channel_mod._pending_hitl.clear()
            channel_mod._hitl_auto_approve.clear()
        with display_mod._stream_cancel_lock:
            display_mod._stream_cancel_event.clear()
            display_mod._stream_cancel_events.clear()
            display_mod._stream_cancel_events[
                display_mod._DEFAULT_STREAM_CANCEL_SCOPE
            ] = display_mod._stream_cancel_event

    _reset()
    yield
    _reset()


class _FakeConfig:
    text_chunk_limit = 4096
    allowed_senders = None


class FakeChannel(Channel):
    """Minimal channel for bus integration testing."""

    name = "fake"

    def __init__(self):
        super().__init__(_FakeConfig())
        self._started = False
        self._stopped = False
        self._sent: list[OutgoingMessage] = []

    async def start(self):
        self._started = True

    async def stop(self):
        self._stopped = True

    async def receive(self):
        while True:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield msg
            except TimeoutError:
                return

    async def send(self, message: OutgoingMessage) -> bool:
        self._sent.append(message)
        return True

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        pass


class TestBusInboundConsumer:
    """Test the _bus_inbound_consumer queue bridge."""

    def test_processes_inbound_and_publishes_outbound(self):
        """InboundMessage -> queue -> response -> OutboundMessage flow."""
        from EvoScientist.cli.channel import (
            _bus_inbound_consumer,
            _message_queue,
            _set_channel_response,
        )

        _drain_queue(_message_queue)

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="hello agent",
                )
            )

            # Wait for consumer to enqueue the message
            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            msg = _message_queue.get_nowait()
            assert msg.content == "hello agent"
            assert msg.sender == "user1"
            assert msg.channel_type == "fake"

            # Simulate main-thread response
            _set_channel_response(msg.msg_id, "Reply to: hello agent")

            outbound = await asyncio.wait_for(
                bus.consume_outbound(),
                timeout=2.0,
            )
            assert outbound.channel == "fake"
            assert outbound.chat_id == "chat1"
            assert "Reply to: hello agent" in outbound.content

            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

        _run(_test())

    def test_no_response_fallback(self):
        """Empty response is replaced with 'No response' fallback."""
        from EvoScientist.cli.channel import (
            _bus_inbound_consumer,
            _message_queue,
            _set_channel_response,
        )

        _drain_queue(_message_queue)

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="test",
                )
            )

            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            msg = _message_queue.get_nowait()
            # Set empty response — falsy, so consumer falls back to "No response"
            _set_channel_response(msg.msg_id, "")

            outbound = await asyncio.wait_for(
                bus.consume_outbound(),
                timeout=2.0,
            )
            assert outbound.content == "No response"

            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

        _run(_test())

    def test_late_response_after_timeout_still_publishes(self, monkeypatch):
        """A response that arrives after the bridge timeout is still forwarded."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import (
            _bus_inbound_consumer,
            _message_queue,
            _set_channel_response,
        )

        monkeypatch.setattr(channel_mod, "_RESPONSE_TIMEOUT", 0.05)
        monkeypatch.setattr(channel_mod, "_LATE_RESPONSE_TIMEOUT", 1.0)

        _drain_queue(_message_queue)

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="slow request",
                    message_id="msg-123",
                )
            )

            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            msg = _message_queue.get_nowait()

            notice = await asyncio.wait_for(
                bus.consume_outbound(),
                timeout=1.0,
            )
            assert "Still working on it" in notice.content
            assert notice.reply_to == "msg-123"

            _set_channel_response(msg.msg_id, "final answer")

            outbound = await asyncio.wait_for(
                bus.consume_outbound(),
                timeout=1.0,
            )
            assert outbound.content == "final answer"
            assert outbound.reply_to == "msg-123"

            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

        _run(_test())

    def test_late_timeout_keeps_active_request_cancellable(self, monkeypatch):
        """Late timeout must not discard an active request's cancel scope."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import (
            _channel_message_cancel_scope,
            _channel_request_state,
            _claim_channel_request,
            _handle_bus_message,
            _message_queue,
        )
        from EvoScientist.stream import display as display_mod

        monkeypatch.setattr(channel_mod, "_RESPONSE_TIMEOUT", 0.05)
        monkeypatch.setattr(channel_mod, "_LATE_RESPONSE_TIMEOUT", 0.05)

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            task = asyncio.create_task(
                _handle_bus_message(
                    bus,
                    manager,
                    InboundMessage(
                        channel="fake",
                        sender_id="user1",
                        chat_id="chat1",
                        content="still running",
                        message_id="msg-active",
                    ),
                )
            )

            queued = None
            for _ in range(20):
                with _message_queue.mutex:
                    queued = _message_queue.queue[0] if _message_queue.queue else None
                if queued is not None:
                    break
                await asyncio.sleep(0.05)

            assert queued is not None
            assert _claim_channel_request(queued) is True

            notice = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Still working on it" in notice.content

            await task

            assert _channel_request_state(queued.msg_id) == "active"
            cancel_scope = _channel_message_cancel_scope(queued)
            assert not display_mod.is_stream_cancel_requested(cancel_scope)

            await _handle_bus_message(
                bus,
                manager,
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="/stop",
                    message_id="msg-stop-active",
                ),
            )

            ack = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert ack.content == "Stopped."
            assert ack.reply_to == "msg-stop-active"
            assert display_mod.is_stream_cancel_requested(cancel_scope)

        _run(_test())

    def test_cancelled_wait_cleans_pending_response(self):
        """Cancelling a pending bus message should not leak its response slot."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import _handle_bus_message, _message_queue

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            task = asyncio.create_task(
                _handle_bus_message(
                    bus,
                    manager,
                    InboundMessage(
                        channel="fake",
                        sender_id="user1",
                        chat_id="chat1",
                        content="cancel me",
                    ),
                )
            )

            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            queued = _message_queue.get_nowait()
            with channel_mod._response_lock:
                assert queued.msg_id in channel_mod._pending_responses

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            with channel_mod._response_lock:
                assert queued.msg_id not in channel_mod._pending_responses

        _run(_test())

    def test_consumer_shutdown_cleans_pending_response(self):
        """Stopping the consumer should cancel late waits and clear state."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import _bus_inbound_consumer, _message_queue

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="slow shutdown",
                )
            )

            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            queued = _message_queue.get_nowait()
            with channel_mod._response_lock:
                assert queued.msg_id in channel_mod._pending_responses

            consumer.cancel()
            await consumer

            with channel_mod._response_lock:
                assert queued.msg_id not in channel_mod._pending_responses

        _run(_test())

    def test_stop_during_hitl_wait_releases_wait_and_acks(self):
        """`/stop` should wake pending HITL wait and publish immediate ack."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import _bus_inbound_consumer, _message_queue

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            hitl_event = channel_mod._register_hitl_wait("fake", "chat1")
            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="/stop",
                    message_id="m-stop-1",
                )
            )

            for _ in range(20):
                if hitl_event.is_set():
                    break
                await asyncio.sleep(0.05)
            assert hitl_event.is_set()
            assert channel_mod._pop_hitl_reply("fake", "chat1") == "/stop"

            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
            assert outbound.content == "Stopped."
            assert outbound.reply_to == "m-stop-1"
            assert _message_queue.empty()

            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

        _run(_test())

    def test_stop_cancels_queued_request_before_main_thread_processes_it(self):
        """`/stop` should cancel a queued request instead of only acking."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import (
            _claim_or_complete_channel_request,
            _handle_bus_message,
            _message_queue,
        )

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            task = asyncio.create_task(
                _handle_bus_message(
                    bus,
                    manager,
                    InboundMessage(
                        channel="fake",
                        sender_id="user1",
                        chat_id="chat1",
                        content="please work",
                        message_id="m-work-1",
                    ),
                )
            )

            queued = None
            for _ in range(20):
                with _message_queue.mutex:
                    queued = _message_queue.queue[0] if _message_queue.queue else None
                if queued is not None:
                    break
                await asyncio.sleep(0.05)

            assert queued is not None
            with channel_mod._response_lock:
                assert queued.msg_id in channel_mod._pending_responses

            await _handle_bus_message(
                bus,
                manager,
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="/stop",
                    message_id="m-stop-2",
                ),
            )

            with pytest.raises(asyncio.CancelledError):
                await task

            skipped = _message_queue.get_nowait()
            assert skipped.msg_id == queued.msg_id
            assert _claim_or_complete_channel_request(skipped) is False

            with channel_mod._response_lock:
                assert queued.msg_id not in channel_mod._pending_responses
            with channel_mod._channel_request_lock:
                assert queued.msg_id not in channel_mod._channel_requests
                assert queued.msg_id not in channel_mod._cancelled_channel_messages
                assert "fake:chat1" not in channel_mod._session_requests

            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
            assert outbound.content == "Stopped."
            assert outbound.reply_to == "m-stop-2"
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(bus.consume_outbound(), timeout=0.2)

        _run(_test())

    def test_stop_leaves_resolved_response_available_for_delivery(self):
        """`/stop` must not steal a response whose waiter already resolved."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import (
            ChannelMessage,
            _cancel_channel_session,
            _claim_channel_request,
            _complete_channel_request,
            _enqueue_channel_message,
            _pop_channel_response,
            _set_channel_response,
        )

        async def _test():
            msg = ChannelMessage(
                msg_id="msg-resolved",
                content="already answered",
                sender="user1",
                channel_type="fake",
                metadata={},
                channel_ref=None,
                bus_ref=None,
                chat_id="chat1",
                message_id="m-resolved",
            )

            waiter = _enqueue_channel_message(msg)
            assert _claim_channel_request(msg) is True

            _set_channel_response(msg.msg_id, "final answer")
            assert await asyncio.wait_for(asyncio.shield(waiter), timeout=1.0) == (
                "final answer"
            )

            cancelled_count, active_count = _cancel_channel_session("fake", "chat1")
            assert cancelled_count == 0
            assert active_count == 0

            with channel_mod._response_lock:
                assert msg.msg_id in channel_mod._pending_responses
            with channel_mod._channel_request_lock:
                assert msg.msg_id not in channel_mod._cancelled_channel_messages

            assert _pop_channel_response(msg.msg_id) == "final answer"
            _complete_channel_request(msg.msg_id)

        _run(_test())

    def test_message_counting(self):
        """Messages are counted via record_message."""
        from EvoScientist.cli.channel import (
            _bus_inbound_consumer,
            _message_queue,
            _set_channel_response,
        )

        _drain_queue(_message_queue)

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
            )

            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            msg = _message_queue.get_nowait()
            _set_channel_response(msg.msg_id, "ok")

            await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)

            assert manager._message_counts["fake"]["received"] == 1
            assert manager._message_counts["fake"]["sent"] == 1

            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

        _run(_test())

    def test_channel_message_carries_metadata(self):
        """ChannelMessage carries metadata, chat_id, and message_id."""
        from EvoScientist.cli.channel import (
            _bus_inbound_consumer,
            _message_queue,
            _set_channel_response,
        )

        _drain_queue(_message_queue)

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))

            await bus.publish_inbound(
                InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="with metadata",
                    metadata={"key": "value"},
                    message_id="msg-123",
                )
            )

            for _ in range(20):
                if not _message_queue.empty():
                    break
                await asyncio.sleep(0.05)

            msg = _message_queue.get_nowait()
            assert msg.content == "with metadata"
            assert msg.metadata == {"key": "value"}
            assert msg.chat_id == "chat1"
            assert msg.message_id == "msg-123"
            assert msg.channel_ref is ch

            _set_channel_response(msg.msg_id, "done")

            outbound = await asyncio.wait_for(
                bus.consume_outbound(),
                timeout=2.0,
            )
            assert outbound.reply_to == "msg-123"

            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

        _run(_test())
