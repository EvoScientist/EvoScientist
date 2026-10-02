"""Async message bus that decouples chat channels from the agent core.

Channels push messages to the inbound queue; the agent (or any consumer)
reads from inbound, processes, and pushes responses to the outbound queue.
``ChannelManager._dispatch_outbound`` routes outbound messages to the
correct channel by looking up its registered :class:`Channel` instance.

Deduplication is handled at the Channel level (single dedup point).
"""

import asyncio
import logging

from ..debug import TraceMixin, debug_trace_enabled
from .events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)


class MessageBus(TraceMixin):
    """Async message bus that decouples chat channels from the agent core."""

    name = "bus"

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=5000)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=5000)
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

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
