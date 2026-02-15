"""Tests for the Message Bus decoupling layer."""

import asyncio


from EvoScientist.channels.bus.events import InboundMessage, OutboundMessage
from EvoScientist.channels.bus.message_bus import MessageBus


def _run(coro):
    """Run an async coroutine safely, creating a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Event tests ──


class TestInboundMessage:
    def test_session_key(self):
        msg = InboundMessage(
            channel="telegram", sender_id="u1",
            chat_id="c1", content="hi",
        )
        assert msg.session_key == "telegram:c1"

    def test_defaults(self):
        msg = InboundMessage(
            channel="discord", sender_id="u2",
            chat_id="c2", content="hello",
        )
        assert msg.media == []
        assert msg.metadata == {}
        assert msg.message_id == ""


class TestOutboundMessage:
    def test_fields(self):
        msg = OutboundMessage(
            channel="telegram", chat_id="c1", content="reply",
        )
        assert msg.channel == "telegram"
        assert msg.chat_id == "c1"
        assert msg.reply_to is None
        assert msg.media == []


# ── MessageBus tests ──


class TestMessageBus:
    def test_inbound_publish_consume(self):
        async def _test():
            bus = MessageBus()
            msg = InboundMessage(
                channel="telegram", sender_id="u1",
                chat_id="c1", content="hello",
            )
            await bus.publish_inbound(msg)
            assert bus.inbound_size == 1
            got = await bus.consume_inbound()
            assert got is msg
            assert bus.inbound_size == 0
        _run(_test())

    def test_outbound_publish_consume(self):
        async def _test():
            bus = MessageBus()
            msg = OutboundMessage(
                channel="discord", chat_id="c1", content="reply",
            )
            await bus.publish_outbound(msg)
            assert bus.outbound_size == 1
            got = await bus.consume_outbound()
            assert got is msg
            assert bus.outbound_size == 0
        _run(_test())

    def test_subscribe_and_dispatch(self):
        async def _test():
            bus = MessageBus()
            received = []

            async def callback(msg):
                received.append(msg)

            bus.subscribe_outbound("telegram", callback)

            msg = OutboundMessage(
                channel="telegram", chat_id="c1", content="hi",
            )
            await bus.publish_outbound(msg)

            dispatch = asyncio.create_task(bus.dispatch_outbound())
            await asyncio.sleep(0.05)
            bus.stop()
            await asyncio.sleep(0.05)
            dispatch.cancel()

            assert len(received) == 1
            assert received[0] is msg
        _run(_test())

    def test_stop(self):
        bus = MessageBus()
        assert bus._running is False
        bus.stop()
        assert bus._running is False
