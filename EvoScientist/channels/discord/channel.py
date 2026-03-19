"""Discord channel implementation using discord.py."""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from ..base import Channel, RawIncoming, ChannelError
from ..capabilities import DISCORD as DISCORD_CAPS
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig(BaseChannelConfig):
    bot_token: str = ""
    text_chunk_limit: int = 2000


class DiscordChannel(Channel):
    """Discord channel using discord.py."""

    name = "discord"

    capabilities = DISCORD_CAPS
    _typing_interval: float = 8.0
    _ready_attrs = ("_client",)
    _mention_pattern = r"<@!?{bot_id}>\s*"

    def __init__(self, config: DiscordConfig):
        super().__init__(config)
        self._client = None
        self._ready = asyncio.Event()
        # Cache message objects for ACK reactions
        self._message_cache: dict[str, object] = {}
        self._MESSAGE_CACHE_MAX = 200

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            raise ChannelError(
                "discord.py not installed. "
                "Install with: pip install evoscientist[discord]"
            )

        if not self.config.bot_token:
            raise ChannelError("Discord bot token is required")

        proxy = (
            self.config.proxy
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("http_proxy")
            or os.environ.get("HTTP_PROXY")
            or None
        )

        logger.info(
            "Discord connect: token=%s...%s proxy=%s",
            self.config.bot_token[:8],
            self.config.bot_token[-4:],
            proxy or "(none)",
        )

        intents = discord.Intents.default()
        intents.message_content = True
        client_kwargs = {"intents": intents}
        if proxy:
            client_kwargs["proxy"] = proxy
        self._client = discord.Client(**client_kwargs)

        self._start_task_error: BaseException | None = None

        @self._client.event
        async def on_ready():
            logger.info(f"Discord bot ready: {self._client.user}")
            self._ready.set()

        @self._client.event
        async def on_message(message):
            await self._on_message(message)

        async def _guarded_start():
            try:
                logger.info("Discord gateway: starting client.start()...")
                await self._client.start(self.config.bot_token)
            except Exception as exc:
                logger.error("Discord gateway error: %s: %s", type(exc).__name__, exc)
                self._start_task_error = exc
                self._ready.set()  # unblock the waiter so it doesn't hang

        logger.info("Discord connect: launching gateway task")
        asyncio.create_task(_guarded_start())

        try:
            await asyncio.wait_for(self._ready.wait(), timeout=60)
        except asyncio.TimeoutError:
            raise ChannelError(
                "Discord bot failed to connect within 60s. "
                "Check network/proxy connectivity to gateway.discord.gg"
            )

        if self._start_task_error:
            raise ChannelError(
                f"Discord bot failed to connect: {self._start_task_error}"
            )

        self._running = True
        logger.info("Discord channel started")

    async def _cleanup(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Discord channel stopped")

    # ── Typing indicator ────────────────────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        if not self._client:
            return
        ch = self._client.get_channel(int(chat_id))
        if ch:
            await ch.trigger_typing()

    # ── ACK Reactions ───────────────────────────────────────────────

    async def _send_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "👀"
    ) -> None:
        msg = self._message_cache.get(message_id)
        if msg:
            try:
                await msg.add_reaction(emoji)
            except Exception as e:
                logger.debug(f"Discord ACK reaction failed: {e}")

    async def _remove_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "👀"
    ) -> None:
        msg = self._message_cache.get(message_id)
        if msg and self._client and self._client.user:
            try:
                await msg.remove_reaction(emoji, self._client.user)
            except Exception as e:
                logger.debug(f"Discord remove ACK reaction failed: {e}")

    def _cache_message(self, message) -> None:
        """Cache a discord message object for later reaction use."""
        mid = str(message.id)
        self._message_cache[mid] = message
        # Evict oldest entries if cache is too large
        if len(self._message_cache) > self._MESSAGE_CACHE_MAX:
            oldest = list(self._message_cache.keys())[: self._MESSAGE_CACHE_MAX // 2]
            for k in oldest:
                self._message_cache.pop(k, None)

    # ── Send ────────────────────────────────────────────────────────

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        import discord

        thread_id = (metadata or {}).get("thread_id", "")
        target_id = int(thread_id) if thread_id else int(chat_id)
        ch = self._client.get_channel(target_id)
        if not ch:
            raise RuntimeError(f"Discord channel {target_id} not found")
        ref = None
        if reply_to:
            try:
                ref = discord.MessageReference(
                    message_id=int(reply_to),
                    channel_id=target_id,
                )
            except (ValueError, TypeError):
                pass

        async def _send(text):
            await ch.send(text, reference=ref)

        await self._send_with_format_fallback(_send, formatted_text, raw_text)

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        import discord

        channel_id = self._resolve_media_chat_id(recipient, metadata)
        ch = self._client.get_channel(int(channel_id))
        if not ch:
            logger.error(f"Discord channel {channel_id} not found")
            return False
        file = discord.File(file_path)
        await ch.send(content=caption or None, file=file)
        return True

    def _get_bot_identifier(self) -> str | None:
        if self._client and self._client.user:
            return str(self._client.user.id)
        return None

    # ── Inbound ─────────────────────────────────────────────────────

    async def _on_message(self, message) -> None:
        import discord

        if message.author == self._client.user:
            return

        # Cache for ACK reactions
        self._cache_message(message)

        user_id = str(message.author.id)
        channel_id = str(message.channel.id)

        is_dm = isinstance(message.channel, discord.DMChannel)
        was_mentioned = is_dm or (self._client.user in message.mentions)

        text = message.content or ""
        annotations: list[str] = []
        media_paths: list[str] = []

        if self.config.include_attachments and message.attachments:
            for attachment in message.attachments:
                too_large = self._check_attachment_size(
                    attachment.size or 0,
                    attachment.filename,
                )
                if too_large:
                    annotations.append(too_large)
                    continue
                try:
                    safe_name = attachment.filename.replace("/", "_")
                    file_path = self._media_path(f"{attachment.id}_{safe_name}")
                    await attachment.save(file_path)
                    media_paths.append(str(file_path))
                    annotations.append(f"[attachment: {file_path}]")
                except Exception as e:
                    logger.warning(f"Failed to download Discord attachment: {e}")
                    annotations.append(
                        f"[attachment: {attachment.filename} - download failed]"
                    )

        # Detect thread context
        thread_id = ""
        parent_channel_id = channel_id
        if hasattr(message.channel, "parent") and message.channel.parent:
            # Message is inside a Thread — store thread info
            thread_id = channel_id  # the thread IS the channel
            parent_channel_id = str(message.channel.parent.id)

        await self._enqueue_raw(
            RawIncoming(
                sender_id=user_id,
                chat_id=parent_channel_id,
                text=text,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=message.created_at or datetime.now(),
                message_id=str(message.id),
                metadata={"chat_id": parent_channel_id, "thread_id": thread_id},
                is_group=not is_dm,
                was_mentioned=was_mentioned,
            )
        )
