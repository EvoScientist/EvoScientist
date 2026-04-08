"""Async message bus that decouples chat channels from the agent core.

Channels push messages to the inbound queue; the agent (or any consumer)
reads from inbound, processes, and pushes responses to the outbound queue.
A background dispatcher routes outbound messages to the correct channel
via subscriber callbacks.

Deduplication is handled at the Channel level (single dedup point).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ..debug import TraceMixin, debug_trace_enabled
from .events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

OutboundCallback = Callable[[OutboundMessage], Awaitable[None]]


class MessageBus(TraceMixin):
    """Async message bus that decouples chat channels from the agent core."""

    name = "bus"

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=5000)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=5000)
        self._outbound_subscribers: dict[str, list[OutboundCallback]] = {}
        self._running = False
        self._debug_trace = debug_trace_enabled()
        self._trace_logger = logger

    # ── inbound (channel → agent) ──

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    # ── outbound (agent → channel) ──

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    # ── subscriber routing ──

    def subscribe_outbound(
        self,
        channel: str,
        callback: OutboundCallback,
    ) -> None:
        """Register a callback for outbound messages targeting *channel*."""
        if channel not in self._outbound_subscribers:
            self._outbound_subscribers[channel] = []
        self._outbound_subscribers[channel].append(callback)

    async def dispatch_outbound(self) -> None:
        """Route outbound messages to subscribed channels.

        Run as a background task — loops until :meth:`stop` is called.
        """
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.outbound.get(),
                    timeout=1.0,
                )
            except TimeoutError:
                continue
            subscribers = self._outbound_subscribers.get(msg.channel, [])
            if not subscribers:
                self._trace_event(
                    "bus_dispatch_drop",
                    target_channel=msg.channel,
                    reason="no_subscriber",
                    chat_id=msg.chat_id,
                )
                logger.warning(f"No subscriber for channel: {msg.channel}")
                continue
            for callback in subscribers:
                try:
                    await callback(msg)
                except Exception as e:
                    self._trace_event(
                        "bus_dispatch_error",
                        target_channel=msg.channel,
                        chat_id=msg.chat_id,
                        error_type=type(e).__name__,
                    )
                    logger.error(f"Error dispatching to {msg.channel}: {e}")

    def stop(self) -> None:
        """Stop the dispatcher loop."""
        self._running = False

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
