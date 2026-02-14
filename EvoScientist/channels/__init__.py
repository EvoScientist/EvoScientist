"""Communication channels for EvoScientist.

This module provides an extensible interface for different messaging channels
(iMessage, Telegram, Discord) to communicate with the EvoScientist agent.
"""

from .base import Channel, RawIncoming, IncomingMessage, OutgoingMessage
from .bus import MessageBus, InboundMessage, OutboundMessage
from .channel_manager import ChannelManager, register_channel, create_channel, available_channels
from .consumer import InboundConsumer
from .standalone import run_standalone

# Backward compat: ChannelServer is now Channel itself
ChannelServer = Channel

__all__ = [
    "Channel",
    "ChannelServer",
    "ChannelManager",
    "MessageBus",
    "RawIncoming",
    "IncomingMessage",
    "OutgoingMessage",
    "InboundMessage",
    "OutboundMessage",
    "InboundConsumer",
    "run_standalone",
    "register_channel",
    "create_channel",
    # New modules
    "ChannelCapabilities",
    "UnifiedFormatter",
    "TypingManager",
    "chunk_text",
    # Plugin architecture
    "ChannelPlugin",
    "ChannelMeta",
    "ReloadPolicy",
]

from .capabilities import ChannelCapabilities
from .formatter import UnifiedFormatter
from .middleware import TypingManager
from .base import chunk_text

# Plugin architecture
from .plugin import ChannelPlugin, ChannelMeta, ReloadPolicy
