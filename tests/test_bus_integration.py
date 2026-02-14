"""Tests for bus-mode agent integration (_bus_inbound_consumer)."""

import asyncio

import pytest

from EvoScientist.channels.bus.events import InboundMessage, OutboundMessage
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.base import Channel, IncomingMessage, OutgoingMessage


def _run(coro):
    """Run an async coroutine safely, creating a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
            except asyncio.TimeoutError:
                return

    async def send(self, message: OutgoingMessage) -> bool:
        self._sent.append(message)
        return True

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        pass


def _mock_stream_events(content, reply):
    """Create a mock stream_agent_events that yields text then done."""
    async def _stream(agent, message, thread_id):
        yield {"type": "text", "content": reply}
        yield {"type": "done", "response": reply}
    return _stream


def _mock_stream_events_error(error_msg):
    """Create a mock stream_agent_events that raises."""
    async def _stream(agent, message, thread_id):
        raise RuntimeError(error_msg)
        yield  # make it an async generator  # pragma: no cover
    return _stream


def _mock_stream_events_with_thinking(thinking_text, reply):
    """Create a mock stream_agent_events that yields thinking then done."""
    async def _stream(agent, message, thread_id):
        yield {"type": "thinking", "content": thinking_text}
        yield {"type": "text", "content": reply}
        yield {"type": "done", "content": reply}
    return _stream


class TestBusInboundConsumer:
    """Test the _bus_inbound_consumer bridge function."""

    def test_processes_inbound_and_publishes_outbound(self):
        """InboundMessage -> agent -> OutboundMessage flow."""
        from EvoScientist.cli.channel import _bus_inbound_consumer

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            mock_stream = _mock_stream_events(
                "hello agent", "Reply to: hello agent",
            )

            import EvoScientist.stream.events as events_mod
            original = events_mod.stream_agent_events
            events_mod.stream_agent_events = mock_stream

            try:
                consumer = asyncio.create_task(
                    _bus_inbound_consumer(bus, manager, None, "test-thread", False)
                )

                await bus.publish_inbound(InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="hello agent",
                ))

                await asyncio.sleep(0.5)

                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=2.0,
                )
                assert outbound.channel == "fake"
                assert outbound.chat_id == "chat1"
                assert "Reply to: hello agent" in outbound.content

                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass
            finally:
                events_mod.stream_agent_events = original

        _run(_test())

    def test_agent_error_publishes_error_outbound(self):
        """When agent raises, an error message is published outbound."""
        from EvoScientist.cli.channel import _bus_inbound_consumer

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            mock_stream = _mock_stream_events_error("agent crashed")

            import EvoScientist.stream.events as events_mod
            original = events_mod.stream_agent_events
            events_mod.stream_agent_events = mock_stream

            try:
                consumer = asyncio.create_task(
                    _bus_inbound_consumer(bus, manager, None, "test-thread", False)
                )

                await bus.publish_inbound(InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="crash me",
                ))

                await asyncio.sleep(0.5)

                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=2.0,
                )
                assert outbound.channel == "fake"
                assert "Error" in outbound.content or "error" in outbound.content.lower()
                assert "agent crashed" in outbound.content

                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass
            finally:
                events_mod.stream_agent_events = original

        _run(_test())

    def test_message_counting(self):
        """Messages are counted via record_message."""
        from EvoScientist.cli.channel import _bus_inbound_consumer

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            manager.register(ch)

            mock_stream = _mock_stream_events("test", "ok")

            import EvoScientist.stream.events as events_mod
            original = events_mod.stream_agent_events
            events_mod.stream_agent_events = mock_stream

            try:
                consumer = asyncio.create_task(
                    _bus_inbound_consumer(bus, manager, None, "test-thread", False)
                )

                await bus.publish_inbound(InboundMessage(
                    channel="fake",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                ))

                await asyncio.sleep(0.5)
                await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)

                assert manager._message_counts["fake"]["received"] == 1
                assert manager._message_counts["fake"]["sent"] == 1

                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass
            finally:
                events_mod.stream_agent_events = original

        _run(_test())

    def test_thinking_sent_to_channel(self):
        """Thinking messages are sent to the channel when show_thinking=True."""
        from EvoScientist.cli.channel import _bus_inbound_consumer

        async def _test():
            bus = MessageBus()
            manager = ChannelManager(bus)
            ch = FakeChannel()
            server = manager.register(ch)
            server.send_thinking = True

            long_thinking = "A" * 250  # >= _MIN_THINKING_LEN (200)
            mock_stream = _mock_stream_events_with_thinking(
                long_thinking, "final answer",
            )

            import EvoScientist.stream.events as events_mod
            original = events_mod.stream_agent_events
            events_mod.stream_agent_events = mock_stream

            try:
                consumer = asyncio.create_task(
                    _bus_inbound_consumer(
                        bus, manager, None, "test-thread", True,
                    )
                )

                await bus.publish_inbound(InboundMessage(
                    channel="fake",
                    sender_id="user1",
                    chat_id="chat1",
                    content="think about this",
                    metadata={"chat_id": "chat1"},
                ))

                await asyncio.sleep(0.5)

                # Drain outbound (final answer)
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=2.0,
                )
                assert "final answer" in outbound.content

                # Check that thinking was sent via channel.send
                thinking_msgs = [
                    m for m in ch._sent
                    if "\U0001f9e0" in m.content
                ]
                assert len(thinking_msgs) == 1
                assert long_thinking in thinking_msgs[0].content

                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass
            finally:
                events_mod.stream_agent_events = original

        _run(_test())
