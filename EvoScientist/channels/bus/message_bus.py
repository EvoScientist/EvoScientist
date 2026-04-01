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

from ..debug import debug_trace_enabled, emit_debug_event
from .events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

OutboundCallback = Callable[[OutboundMessage], Awaitable[None]]


class MessageBus:
    """Async message bus that decouples chat channels from the agent core."""

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=5000)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=5000)
        self._outbound_subscribers: dict[str, list[OutboundCallback]] = {}
        self._running = False
        self._debug_trace = debug_trace_enabled()

    def _trace_event(self, event: str, channel: str, **fields) -> None:
        emit_debug_event(
            logger,
            event,
            channel=channel,
            enabled=self._debug_trace,
            **fields,
        )

    # ── inbound (channel → agent) ──

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)
        self._trace_event(
            "bus_publish_inbound",
            msg.channel,
            sender_id=msg.sender_id,
            chat_id=msg.chat_id,
            message_id=msg.message_id or "-",
            queue_size=self.inbound.qsize(),
        )

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    # ── outbound (agent → channel) ──

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)
        self._trace_event(
            "bus_publish_outbound",
            msg.channel,
            chat_id=msg.chat_id,
            reply_to=msg.reply_to,
            content_len=len(msg.content or ""),
            media_count=len(msg.media),
            queue_size=self.outbound.qsize(),
        )

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
        self._trace_event(
            "bus_subscribe_outbound",
            channel,
            subscriber_count=len(self._outbound_subscribers[channel]),
        )

    async def dispatch_outbound(self) -> None:
        """Route outbound messages to subscribed channels.

        Run as a background task — loops until :meth:`stop` is called.
        """
        self._running = True
        self._trace_event("bus_dispatcher_start", "bus")
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
                    msg.channel,
                    reason="no_subscriber",
                    chat_id=msg.chat_id,
                )
                logger.warning(f"No subscriber for channel: {msg.channel}")
                continue
            for callback in subscribers:
                try:
                    await callback(msg)
                    self._trace_event(
                        "bus_dispatch_ok",
                        msg.channel,
                        chat_id=msg.chat_id,
                        subscriber_count=len(subscribers),
                    )
                except Exception as e:
                    self._trace_event(
                        "bus_dispatch_error",
                        msg.channel,
                        chat_id=msg.chat_id,
                        error_type=type(e).__name__,
                    )
                    logger.error(f"Error dispatching to {msg.channel}: {e}")

    def stop(self) -> None:
        """Stop the dispatcher loop."""
        self._running = False
        self._trace_event("bus_dispatcher_stop", "bus")

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
