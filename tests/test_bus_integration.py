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

    def test_cancelled_wait_cleans_pending_response(self):
        """Cancelling a pending bus message should not leak its response slot."""
        from EvoScientist.cli import channel as channel_mod
        from EvoScientist.cli.channel import _handle_bus_message, _message_queue

        _drain_queue(_message_queue)
        with channel_mod._response_lock:
            channel_mod._pending_responses.clear()

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
