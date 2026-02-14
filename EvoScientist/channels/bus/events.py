"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel.

    Carries enough context for the bus to route and for the agent
    to build a session: which channel, who sent it, which chat.
    """

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = ""
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sender(self) -> str:
        """Alias for ``sender_id`` (compatibility with IncomingMessage)."""
        return self.sender_id

    @property
    def session_key(self) -> str:
        """Unique key for session identification: ``channel:chat_id``."""
        return f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def recipient(self) -> str:
        """Alias for ``chat_id`` (compatibility with OutgoingMessage)."""
        return self.chat_id
