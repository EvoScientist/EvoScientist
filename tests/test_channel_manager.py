"""Tests for ChannelManager."""

import asyncio

import pytest

from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.base import Channel, OutboundMessage


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
    """Minimal channel for testing."""

    name = "fake"

    def __init__(self):
        super().__init__(_FakeConfig())
        self._started = False
        self._stopped = False
        self._sent: list[OutboundMessage] = []

    async def start(self):
        self._started = True

    async def stop(self):
        self._stopped = True

    async def receive(self):
        while True:
            try:
                msg = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5,
                )
                yield msg
            except asyncio.TimeoutError:
                return

    async def send(self, message: OutboundMessage) -> bool:
        self._sent.append(message)
        return True

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        pass


class TestChannelManagerRegister:
    def test_register_channel(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)
        ch = FakeChannel()
        result = mgr.register(ch)
        assert "fake" in mgr.enabled_channels
        assert mgr.get_channel("fake") is ch
        assert result is ch

    def test_duplicate_register_raises(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(FakeChannel())
        with pytest.raises(ValueError, match="already registered"):
            mgr.register(FakeChannel())

    def test_get_status(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(FakeChannel())
        status = mgr.get_status()
        assert "fake" in status
        assert status["fake"]["registered"] is True


class TestChannelManagerDispatch:
    def test_outbound_dispatch_routes_to_channel(self):
        async def _test():
            bus = MessageBus()
            mgr = ChannelManager(bus)
            ch = FakeChannel()
            mgr.register(ch)

            # Start only the dispatcher (not full start_all)
            dispatch = asyncio.create_task(
                mgr._dispatch_outbound()
            )

            # Publish an outbound message
            await bus.publish_outbound(OutboundMessage(
                channel="fake", chat_id="u1",
                content="hello from agent",
            ))

            await asyncio.sleep(0.1)
            dispatch.cancel()
            try:
                await dispatch
            except asyncio.CancelledError:
                pass

            assert len(ch._sent) == 1
            assert ch._sent[0].content == "hello from agent"
            assert ch._sent[0].chat_id == "u1"

        _run(_test())


class TestChannelManagerTracking:
    def test_record_message(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(FakeChannel())

        mgr.record_message("fake", "received")
        mgr.record_message("fake", "received")
        mgr.record_message("fake", "sent")

        assert mgr._message_counts["fake"]["received"] == 2
        assert mgr._message_counts["fake"]["sent"] == 1

    def test_record_message_unknown_channel(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)

        # Should not raise, auto-creates entry
        mgr.record_message("unknown", "received")
        assert mgr._message_counts["unknown"]["received"] == 1

    def test_get_detailed_status(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(FakeChannel())

        # Simulate start_all setting start_times
        from datetime import datetime
        mgr._start_times["fake"] = datetime.now()
        mgr._message_counts["fake"] = {"received": 5, "sent": 3}

        status = mgr.get_detailed_status()
        assert "fake" in status
        assert status["fake"]["registered"] is True
        assert status["fake"]["received"] == 5
        assert status["fake"]["sent"] == 3
        assert status["fake"]["uptime_seconds"] >= 0
        assert status["fake"]["start_time"] is not None

    def test_get_detailed_status_no_start_time(self):
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(FakeChannel())

        status = mgr.get_detailed_status()
        assert status["fake"]["uptime_seconds"] == 0
        assert status["fake"]["start_time"] is None
