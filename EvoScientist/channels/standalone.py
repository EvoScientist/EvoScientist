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

logger = logging.getLogger(__name__)


async def standalone_outbound_dispatcher(
    bus: MessageBus, channel: Channel,
) -> None:
    """Consume outbound messages from the bus and send via channel."""
    while True:
        try:
            msg: OutboundMessage = await asyncio.wait_for(
                bus.consume_outbound(), timeout=1.0,
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        try:
            if msg.content:
                await channel.send(msg)
        except Exception as e:
            logger.error(f"Error sending outbound: {e}")


async def _async_main(
    channel: Channel, bus: MessageBus,
    use_agent: bool, send_thinking: bool,
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
                if msg.content:
                    await asyncio.wait_for(channel.send(msg), timeout=5.0)
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
            sig, lambda s=sig: asyncio.create_task(_graceful_shutdown()),
        )

    await asyncio.gather(*tasks)


def run_standalone(
    channel: Channel, bus: MessageBus, *,
    use_agent: bool = False, send_thinking: bool = False,
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
