"""Channel capabilities declaration system.

Each channel declares its capabilities via a ChannelCapabilities dataclass,
enabling the framework to adapt behavior automatically (formatting, reactions,
streaming, threading, etc.) without per-channel branching in core logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FormatType = Literal["html", "markdown", "slack_mrkdwn", "discord", "plain"]


@dataclass(frozen=True)
class ChannelCapabilities:
    """Immutable declaration of what a channel supports.

    Set once as a class attribute on each Channel subclass.
    The framework inspects these at runtime to auto-configure behavior.
    """

    # ── Messaging features ──────────────────────────────────────────
    format_type: FormatType = "plain"
    max_text_length: int = 4096
    max_file_size: int = 20 * 1024 * 1024  # 20 MB

    # ── Interaction capabilities ────────────────────────────────────
    streaming: bool = False       # edit-in-place streaming output
    threading: bool = False       # message threads / topics
    reactions: bool = False       # emoji reactions on messages
    typing: bool = False          # typing indicator API
    inline_buttons: bool = False  # inline keyboard / action buttons

    # ── Media capabilities ──────────────────────────────────────────
    media_send: bool = False      # can send files/images
    media_receive: bool = False   # can receive files/images
    voice: bool = False           # platform has voice/audio messages that arrive as downloadable files (receive only, not bot sending)
    stickers: bool = False        # supports sticker receive (not bot sending)
    location: bool = False        # supports location message receive (not bot sending)
    video: bool = False           # video messages

    # ── Group features ──────────────────────────────────────────────
    groups: bool = False          # group chat support
    mentions: bool = False        # @mention detection

    # ── Rich text ───────────────────────────────────────────────────
    markdown: bool = False        # supports Markdown rendering
    html: bool = False            # supports HTML rendering

    # ── Extended capabilities ────────────────────────────────────────
    chat_types: tuple[str, ...] = ()   # ("direct", "group", "channel", "thread")
    edit: bool = False                  # message editing after send
    unsend: bool = False                # message recall / unsend
    block_streaming: bool = False       # block edit-in-place streaming
    native_commands: bool = False       # platform-native slash commands
    polls: bool = False                 # poll / vote messages

    def supports(self, feature: str) -> bool:
        """Check if a feature is supported by name."""
        return getattr(self, feature, False)


# ═════════════════════════════════════════════════════════════════════
# Pre-built capability profiles for each channel
# ═════════════════════════════════════════════════════════════════════

TELEGRAM = ChannelCapabilities(
    format_type="html",
    max_text_length=4000,
    streaming=False,  # could edit messages, but not implemented yet
    threading=False,   # topics exist but not used yet
    reactions=True,
    typing=True,
    media_send=True,
    media_receive=True,
    voice=True,
    stickers=True,
    location=True,
    groups=True,
    mentions=True,
    html=True,
    chat_types=("direct", "group", "channel"),
    edit=True,
    unsend=True,
    native_commands=True,
    polls=True,
)

DISCORD = ChannelCapabilities(
    format_type="discord",
    max_text_length=2000,
    streaming=False,
    threading=True,
    reactions=True,
    typing=True,
    media_send=True,
    media_receive=True,
    voice=False,  # no distinct voice message type in Discord bot API
    groups=True,
    mentions=True,
    markdown=True,
    chat_types=("direct", "group", "thread"),
    edit=True,
    unsend=True,
    native_commands=True,
    polls=True,
)

SLACK = ChannelCapabilities(
    format_type="slack_mrkdwn",
    max_text_length=4000,
    streaming=False,
    threading=True,
    reactions=True,
    typing=True,
    media_send=True,
    media_receive=True,
    voice=False,  # no distinct voice message type in Slack bot API
    groups=True,
    mentions=True,
    chat_types=("direct", "group", "thread"),
    edit=True,
    unsend=True,
    native_commands=True,
)

FEISHU = ChannelCapabilities(
    format_type="markdown",
    max_text_length=4096,
    reactions=True,
    typing=False,  # no typing API
    media_send=True,
    media_receive=True,
    voice=True,
    stickers=True,
    groups=True,
    mentions=True,
    markdown=True,
    chat_types=("direct", "group"),
    edit=True,
    unsend=True,
)

DINGTALK = ChannelCapabilities(
    format_type="markdown",
    max_text_length=4096,
    typing=False,  # no typing API for bots
    media_send=True,
    media_receive=True,
    voice=True,
    groups=True,
    mentions=True,
    markdown=True,
    chat_types=("direct", "group"),
)

QQ = ChannelCapabilities(
    format_type="plain",
    max_text_length=4096,
    typing=False,  # no typing API for QQ bots
    media_send=True,
    media_receive=True,
    voice=False,  # qq-botpy does not expose voice as a distinct message type
    groups=True,
    mentions=True,
    chat_types=("direct", "group", "channel"),
    unsend=True,
)

WECHAT = ChannelCapabilities(
    format_type="markdown",  # WeCom supports markdown
    max_text_length=4096,
    typing=False,  # no typing API
    media_send=True,
    media_receive=True,
    voice=True,
    location=True,
    groups=True,
    mentions=True,
    markdown=True,
    chat_types=("direct", "group"),
    unsend=True,
)

SIGNAL = ChannelCapabilities(
    format_type="plain",
    max_text_length=4096,
    reactions=True,
    typing=True,
    media_send=True,
    media_receive=True,
    voice=True,
    groups=True,
    mentions=True,
    chat_types=("direct", "group"),
)

EMAIL = ChannelCapabilities(
    format_type="html",
    max_text_length=999_999,  # no practical limit
    media_send=True,
    media_receive=True,
    html=True,
    chat_types=("direct",),
)

IMESSAGE = ChannelCapabilities(
    format_type="plain",
    max_text_length=999_999,
    typing=False,      # Apple does not expose typing indicator API
    media_send=True,
    media_receive=True,
    voice=True,
    groups=True,
    mentions=False,    # iMessage has no @mention concept
    reactions=False,   # imsg CLI cannot send tapback reactions
    chat_types=("direct", "group"),
)
