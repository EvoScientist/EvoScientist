"""Shared standalone runner for channel servers.

Provides the channel-agnostic agent loop that any channel can use to
run headless — consuming inbound messages from the bus, streaming
agent events, and dispatching outbound replies.

Usage from a channel's ``main()``::

    from EvoScientist.channels.standalone import run_standalone

    channel = SomeChannel(config)
    bus = MessageBus()
    run_standalone(channel, bus, use_agent=True, send_thinking=True)
"""

import asyncio
import logging
import signal

from .base import Channel
from .bus import MessageBus
from .bus.events import OutboundMessage
from .consumer import InboundConsumer
from .debug import emit_debug_event

logger = logging.getLogger(__name__)


def _channel_trace_enabled(channel: Channel) -> bool:
    """Check if debug tracing is enabled on the channel."""
    try:
        return channel.is_debug_trace_enabled()
    except Exception:
        return False


async def _deliver_outbound(channel: Channel, msg: OutboundMessage) -> None:
    """Deliver an outbound message, including any media attachments."""
    if msg.content:
        sent = await channel.send(msg)
        if not sent:
            raise RuntimeError("send() returned False")
    for media_path in msg.media:
        media_ok = await channel.send_media(
            recipient=msg.chat_id,
            file_path=media_path,
            metadata=msg.metadata,
        )
        if not media_ok:
            raise RuntimeError(f"send_media() returned False for {media_path}")


async def standalone_outbound_dispatcher(
    bus: MessageBus,
    channel: Channel,
) -> None:
    """Consume outbound messages from the bus and send via channel."""
    while True:
        try:
            msg: OutboundMessage = await asyncio.wait_for(
                bus.consume_outbound(),
                timeout=1.0,
            )
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        try:
            await _deliver_outbound(channel, msg)
        except Exception as e:
            emit_debug_event(
                logger,
                "standalone_dispatch_error",
                channel=channel.name,
                enabled=_channel_trace_enabled(channel),
                recipient=msg.recipient,
                error=str(e),
            )
            logger.error(f"Error sending outbound: {e}")


async def _async_main(
    channel: Channel,
    bus: MessageBus,
    use_agent: bool,
    send_thinking: bool,
) -> None:
    """Async entry point — gather channel, dispatcher and optional consumer."""
    from .channel_manager import ChannelManager

    channel.set_bus(bus)
    if send_thinking:
        channel.send_thinking = True

    # Create a lightweight manager for the consumer to use
    manager = ChannelManager(bus)
    manager._channels[channel.name] = channel

    await manager.start_health()

    tasks = [channel.run()]

    dispatcher = standalone_outbound_dispatcher(bus, channel)
    tasks.append(dispatcher)

    consumer: InboundConsumer | None = None
    if use_agent:
        logger.info("Loading EvoScientist agent...")
        from ..EvoScientist import create_cli_agent

        agent = create_cli_agent()
        logger.info("Agent loaded")

        consumer = InboundConsumer(
            bus=bus,
            manager=manager,
            agent=agent,
            thread_id="",
            send_thinking=send_thinking,
        )
        manager.register_health_provider("consumer", lambda: consumer.metrics)
        tasks.append(consumer.run())
        if send_thinking:
            logger.info("Thinking messages enabled")

    async def _graceful_shutdown() -> None:
        """Graceful shutdown: drain consumer, flush outbound, stop channel."""
        logger.info("Graceful shutdown initiated...")
        if consumer is not None:
            await consumer.stop()
        # Drain outbound queue before stopping the channel
        drained = 0
        while True:
            try:
                msg = bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await asyncio.wait_for(_deliver_outbound(channel, msg), timeout=5.0)
                if msg.content or msg.media:
                    drained += 1
            except Exception:
                pass
        if drained:
            logger.info(f"Outbound drain: {drained} sent")
        channel._running = False
        await channel.stop()
        await manager.stop_health()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_graceful_shutdown()),
        )

    await asyncio.gather(*tasks)


def run_standalone(
    channel: Channel,
    bus: MessageBus,
    *,
    use_agent: bool = False,
    send_thinking: bool = False,
) -> None:
    """Synchronous entry point that spins up the standalone runner.

    Parameters
    ----------
    channel:
        A fully-configured :class:`Channel` instance.
    bus:
        The :class:`MessageBus` shared with *channel*.
    use_agent:
        When ``True``, load the EvoScientist agent and process inbound
        messages through it.
    send_thinking:
        When ``True`` **and** *use_agent* is set, forward intermediate
        thinking messages to the channel.
    """
    asyncio.run(_async_main(channel, bus, use_agent, send_thinking))
