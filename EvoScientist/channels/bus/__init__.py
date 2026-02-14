"""Message bus for decoupled channel-agent communication."""

from .events import InboundMessage, OutboundMessage
from .message_bus import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
