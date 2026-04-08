"""Tests for shared channel debug logging helpers."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from EvoScientist.channels.debug import (
    TraceMixin,
    debug_trace_enabled,
    emit_debug_event,
    emit_debug_event_if,
)

from .conftest import run_async


def test_debug_trace_enabled_from_bool():
    assert debug_trace_enabled(True) is True
    assert debug_trace_enabled(False) is False


def test_emit_debug_event_structured(caplog):
    logger = logging.getLogger("tests.channel.debug")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        emit_debug_event(
            logger,
            "inbound_raw",
            channel="telegram",
            enabled=True,
            message_id="123",
            chat_id="-1001",
            has_text=True,
        )
    assert "event=inbound_raw" in caplog.text
    assert "channel=telegram" in caplog.text
    assert "message_id=123" in caplog.text
    assert "has_text=true" in caplog.text


# ── emit_debug_event_if tests ────────────────────────────────────────


def test_emit_debug_event_if_disabled_does_not_log(caplog):
    logger = logging.getLogger("tests.channel.event_if_disabled")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        emit_debug_event_if(
            logger,
            "should_not_appear",
            False,
            channel="test",
            key="value",
        )
    assert "should_not_appear" not in caplog.text


# ── Middleware structured event tests ────────────────────────────────


def _make_raw(**overrides):
    """Create a minimal RawIncoming for testing."""
    from EvoScientist.channels.base import RawIncoming

    defaults = {"sender_id": "user1", "chat_id": "chat1", "text": "hello"}
    defaults.update(overrides)
    return RawIncoming(**defaults)


def _make_channel_context(*, debug_trace=True, name="test_channel"):
    """Create a mock channel context dict for middleware testing."""
    channel = MagicMock()
    channel.name = name
    channel.is_debug_trace_enabled.return_value = debug_trace
    channel.config = MagicMock()
    channel.config.debug_trace = debug_trace
    return {"channel": channel}


def test_middleware_dedup_emits_structured_event(caplog):
    from EvoScientist.channels.middleware import DedupMiddleware

    async def _run():
        mw = DedupMiddleware()
        ctx = _make_channel_context()
        raw = _make_raw(message_id="dup1")

        # First call — not a duplicate
        result = await mw.process_inbound(raw, ctx)
        assert result is not None

        # Second call — duplicate, should emit structured event
        caplog.clear()
        result = await mw.process_inbound(raw, ctx)
        assert result is None

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "middleware_dedup_drop" in caplog.text
    assert "message_id=dup1" in caplog.text


def test_middleware_allowlist_emits_structured_event(caplog):
    from EvoScientist.channels.middleware import AllowListMiddleware

    async def _run():
        mw = AllowListMiddleware(allowed_senders={"allowed_user"})
        ctx = _make_channel_context()
        raw = _make_raw(sender_id="blocked_user")
        result = await mw.process_inbound(raw, ctx)
        assert result is None

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "middleware_allowlist_drop" in caplog.text
    assert "reason=sender_not_allowed" in caplog.text


def test_middleware_mention_gating_emits_structured_event(caplog):
    from EvoScientist.channels.middleware import MentionGatingMiddleware

    async def _run():
        mw = MentionGatingMiddleware(require_mention="group")
        ctx = _make_channel_context()
        raw = _make_raw(is_group=True, was_mentioned=False)
        result = await mw.process_inbound(raw, ctx)
        assert result is None

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "middleware_mention_drop" in caplog.text
    assert "policy=group" in caplog.text


def test_typing_manager_emits_trace_events(caplog):
    from EvoScientist.channels.middleware import TypingManager

    async def _run():
        send_action = AsyncMock(side_effect=RuntimeError("typing api down"))
        mgr = TypingManager(
            send_action,
            interval=100.0,
            debug_trace=True,
            channel_name="test",
        )
        await mgr.start("chat1")
        await asyncio.sleep(0)
        await mgr.stop("chat1")

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "typing_error" in caplog.text
    assert "chat_id=chat1" in caplog.text


def test_ack_reaction_emits_error_traces(caplog):
    from EvoScientist.channels.middleware import AckReactionMiddleware

    async def _run():
        send_fn = AsyncMock()
        remove_fn = AsyncMock(side_effect=RuntimeError("remove failed"))
        ack = AckReactionMiddleware(
            scope="all",
            send_fn=send_fn,
            remove_fn=remove_fn,
            remove_after_reply=True,
            debug_trace=True,
            channel_name="test",
        )

        # Successful send
        await ack.send_ack("chat1", "msg1")
        send_fn.assert_awaited_once()

        # Error on remove
        await ack.remove_ack("chat1")
        remove_fn.assert_awaited_once()

        # Error on send
        send_fn.reset_mock()
        send_fn.side_effect = RuntimeError("api down")
        await ack.send_ack("chat2", "msg2")

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "ack_send_error" in caplog.text
    assert "ack_remove_error" in caplog.text
    assert "api down" in caplog.text
    assert "remove failed" in caplog.text


def test_inbound_raw_event_emitted(caplog):
    """Integration-style: _enqueue_raw emits inbound_raw at the top."""
    from EvoScientist.channels.base import Channel, RawIncoming

    # Create a minimal concrete channel
    class _TestChannel(Channel):
        name = "test_inbound"
        capabilities = MagicMock()
        capabilities.max_text_length = 4096
        capabilities.format_type = "markdown"

        async def start(self):
            pass

        async def _send_chunk(self, chat_id, formatted, raw, reply_to, metadata):
            pass

    config = MagicMock()
    config.debug_trace = True
    config.debug_payloads = False
    config.text_chunk_limit = 0
    config.stt_enabled = False
    config.inbound_middlewares = []
    config.name = "test_inbound"
    config.require_mention = "off"
    config.allowed_senders = None
    config.allowed_channels = None
    config.dm_policy = "open"
    config.ack_scope = "off"
    config.dedup_ttl = 3600

    async def _run():
        with patch.object(Channel, "__abstractmethods__", set()):
            ch = _TestChannel(config)
        raw = RawIncoming(sender_id="u1", chat_id="c1", text="hi", message_id="m1")
        await ch._enqueue_raw(raw)

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "inbound_raw" in caplog.text
    assert "sender_id=u1" in caplog.text


def test_format_fallback_emits_event(caplog):
    """_send_with_format_fallback emits outbound_format_fallback on fallback."""
    from EvoScientist.channels.base import Channel

    class _TestChannel(Channel):
        name = "test_fallback"
        capabilities = MagicMock()
        capabilities.max_text_length = 4096
        capabilities.format_type = "markdown"

        async def start(self):
            pass

        async def _send_chunk(self, chat_id, formatted, raw, reply_to, metadata):
            pass

    config = MagicMock()
    config.debug_trace = True
    config.debug_payloads = False
    config.text_chunk_limit = 0
    config.stt_enabled = False
    config.inbound_middlewares = []
    config.name = "test_fallback"
    config.require_mention = "off"
    config.allowed_senders = None
    config.allowed_channels = None
    config.dm_policy = "open"
    config.ack_scope = "off"
    config.dedup_ttl = 3600

    call_count = 0

    async def _failing_send(text):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("parse error in formatted text")

    async def _run():
        with patch.object(Channel, "__abstractmethods__", set()):
            ch = _TestChannel(config)
        await ch._send_with_format_fallback(_failing_send, "<b>hi</b>", "hi")

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "outbound_format_fallback" in caplog.text
    assert call_count == 2


# ── TraceMixin tests ─────────────────────────────────────────────────


def _make_trace_stub(logger_name: str = "tests.mixin") -> TraceMixin:
    """Create a minimal TraceMixin instance for testing."""

    class _Stub(TraceMixin):
        name = "stub"

        def __init__(self):
            self._debug_trace = True
            self._trace_logger = logging.getLogger(logger_name)

    return _Stub()


def test_trace_mixin_trace_event(caplog):
    stub = _make_trace_stub("tests.mixin")
    with caplog.at_level(logging.DEBUG, logger="tests.mixin"):
        stub._trace_event("some_event", key="val")
    assert "event=some_event" in caplog.text
    assert "channel=stub" in caplog.text
    assert "key=val" in caplog.text


def test_standalone_dispatcher_treats_false_send_as_error(caplog):
    from EvoScientist.channels.bus import MessageBus
    from EvoScientist.channels.bus.events import OutboundMessage
    from EvoScientist.channels.standalone import standalone_outbound_dispatcher

    channel = MagicMock()
    channel.name = "test"
    channel.is_debug_trace_enabled.return_value = True
    channel.send = AsyncMock(return_value=False)
    bus = MessageBus()

    async def _run():
        task = asyncio.create_task(standalone_outbound_dispatcher(bus, channel))
        await bus.publish_outbound(
            OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    with caplog.at_level(logging.DEBUG):
        run_async(_run())
    assert "standalone_dispatch_error" in caplog.text
    assert "send() returned False" in caplog.text


def test_standalone_dispatcher_sends_media():
    from EvoScientist.channels.bus import MessageBus
    from EvoScientist.channels.bus.events import OutboundMessage
    from EvoScientist.channels.standalone import standalone_outbound_dispatcher

    channel = MagicMock()
    channel.name = "test"
    channel.is_debug_trace_enabled.return_value = True
    channel.send = AsyncMock(return_value=True)
    channel.send_media = AsyncMock(return_value=True)
    bus = MessageBus()

    async def _run():
        task = asyncio.create_task(standalone_outbound_dispatcher(bus, channel))
        await bus.publish_outbound(
            OutboundMessage(
                channel="test", chat_id="c1", content="", media=["/tmp/a.png"]
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run_async(_run())
    channel.send_media.assert_awaited_once_with(
        recipient="c1",
        file_path="/tmp/a.png",
        metadata={},
    )


# ── One-time warning test ────────────────────────────────────────────


def test_emit_debug_event_warns_on_level_mismatch(caplog):
    import EvoScientist.channels.debug as dbg

    # Reset the global warning flag
    dbg._warned_debug_level_mismatch = False
    logger = logging.getLogger("tests.level_mismatch")
    # Logger at WARNING — higher than DEBUG
    with caplog.at_level(logging.WARNING, logger=logger.name):
        emit_debug_event(logger, "should_warn", channel="test", enabled=True, x=1)
        emit_debug_event(logger, "should_not_warn_again", channel="test", enabled=True)

    # Should see the one-time mismatch warning
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    mismatch_warnings = [r for r in warnings if "debug tracing is enabled" in r.message]
    assert len(mismatch_warnings) == 1
    # Should NOT see the actual debug events
    assert "should_warn" not in caplog.text or "event=should_warn" not in caplog.text

    # Reset for other tests
    dbg._warned_debug_level_mismatch = False
