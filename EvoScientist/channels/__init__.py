"""Communication channels for EvoScientist.

This module provides an extensible interface for different messaging channels
(iMessage, Telegram, Discord, Slack, WeChat, DingTalk, Feishu, Email, QQ, Signal) to communicate with the EvoScientist agent.
"""

from .base import Channel, IncomingMessage, OutgoingMessage, RawIncoming, chunk_text
from .bus import InboundMessage, MessageBus, OutboundMessage
from .capabilities import ChannelCapabilities
from .channel_manager import (
    ChannelManager,
    available_channels,
    create_channel,
    register_channel,
)
from .consumer import InboundConsumer
from .formatter import UnifiedFormatter
from .middleware import TypingManager
from .plugin import ChannelMeta, ChannelPlugin, ReloadPolicy
from .standalone import run_standalone

# Backward compat: ChannelServer is now Channel itself
ChannelServer = Channel

__all__ = [
    "Channel",
    # New modules
    "ChannelCapabilities",
    "ChannelManager",
    "ChannelMeta",
    # Plugin architecture
    "ChannelPlugin",
    "ChannelServer",
    "InboundConsumer",
    "InboundMessage",
    "IncomingMessage",
    "MessageBus",
    "OutboundMessage",
    "OutgoingMessage",
    "RawIncoming",
    "ReloadPolicy",
    "TypingManager",
    "UnifiedFormatter",
    "available_channels",
    "chunk_text",
    "create_channel",
    "register_channel",
    "run_standalone",
]
