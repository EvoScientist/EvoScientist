"""Telegram channel implementation using python-telegram-bot."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..base import Channel, RawIncoming, ChannelError, IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS
from ..capabilities import TELEGRAM as TELEGRAM_CAPS
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig(BaseChannelConfig):
    bot_token: str = ""
    text_chunk_limit: int = 4096


class TelegramChannel(Channel):
    """Telegram channel using python-telegram-bot with long polling."""

    name = "telegram"

    capabilities = TELEGRAM_CAPS
    _typing_interval: float = 4.0
    _ready_attrs = ("_app",)
    _non_retryable_patterns = ("parse", "can't parse")
    _mention_pattern = r"(?i)@{bot_id}\s*"

    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self._app = None
        self._bot_username: str = ""

    async def start(self) -> None:
        if not self.config.bot_token:
            raise ChannelError("Telegram bot token is required")

        try:
            from telegram.ext import (
                ApplicationBuilder,
                MessageHandler,
                filters,
            )
        except ImportError:
            raise ChannelError(
                "python-telegram-bot not installed. "
                "Install with: pip install evoscientist[telegram]"
            )

        builder = ApplicationBuilder().token(self.config.bot_token)
        if self.config.proxy:
            builder = builder.proxy(self.config.proxy).get_updates_proxy(self.config.proxy)
        self._app = builder.build()

        # Accept text and media message types
        media_filter = filters.TEXT
        if self.config.include_attachments:
            media_filter = (
                filters.TEXT
                | filters.PHOTO
                | filters.VOICE
                | filters.AUDIO
                | filters.Document.ALL
                | filters.VIDEO
                | filters.Sticker.ALL
                | filters.LOCATION
            )

        self._app.add_handler(
            MessageHandler(media_filter & ~filters.COMMAND, self._on_message)
        )

        await self._app.initialize()
        # Cache bot username for @mention detection in groups
        bot_info = await self._app.bot.get_me()
        self._bot_username = (bot_info.username or "").lower()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        logger.info("Telegram channel started (polling)")

    async def _cleanup(self) -> None:
        if self._app:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram channel stopped")

    # ── Typing indicator (override base) ────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        """Send typing action via Telegram Bot API."""
        if self._app:
            await self._app.bot.send_chat_action(
                chat_id=int(chat_id), action="typing",
            )

    # ── Send (template method overrides) ──────────────────────────

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        reply_id = int(reply_to) if reply_to else None

        async def _send(text):
            await self._app.bot.send_message(
                chat_id=int(chat_id), text=text,
                parse_mode="HTML" if text == formatted_text else None,
                reply_to_message_id=reply_id,
            )

        await self._send_with_format_fallback(_send, formatted_text, raw_text)

    _MEDIA_SENDERS = {
        IMAGE_EXTS: ("send_photo", "photo"),
        VIDEO_EXTS: ("send_video", "video"),
        AUDIO_EXTS: ("send_audio", "audio"),
    }

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file through Telegram."""
        chat_id = int(self._resolve_media_chat_id(recipient, metadata))
        cap = caption or None
        ext = Path(file_path).suffix.lower()
        for exts, (method, param) in self._MEDIA_SENDERS.items():
            if ext in exts:
                await getattr(self._app.bot, method)(
                    chat_id=chat_id, caption=cap, **{param: file_path},
                )
                return True
        await self._app.bot.send_document(
            chat_id=chat_id, document=file_path, caption=cap,
        )
        return True

    def _get_bot_identifier(self) -> str | None:
        return self._bot_username or None

    async def _send_ack_reaction(self, chat_id: str, message_id: str, emoji: str = "👀") -> None:
        """Send an acknowledgment reaction via Telegram."""
        if self._app:
            try:
                from telegram import ReactionTypeEmoji
                await self._app.bot.set_message_reaction(
                    chat_id=int(chat_id),
                    message_id=int(message_id),
                    reaction=[ReactionTypeEmoji(emoji)],
                )
            except Exception as e:
                logger.debug(f"Telegram ACK reaction failed: {e}")

    async def _remove_ack_reaction(self, chat_id: str, message_id: str, emoji: str = "👀") -> None:
        """Remove the ack reaction by setting empty reaction list."""
        if self._app:
            try:
                await self._app.bot.set_message_reaction(
                    chat_id=int(chat_id),
                    message_id=int(message_id),
                    reaction=[],
                )
            except Exception as e:
                logger.debug(f"Telegram remove ACK reaction failed: {e}")

    async def _on_message(self, update, context) -> None:
        """Handler callback for text, photos, voice, audio, documents, video."""
        if not update.message:
            return

        message = update.message
        user_id = str(message.from_user.id)
        chat_id = str(message.chat_id)

        # Detect group and mention status for centralized gating
        is_group = message.chat.type in ("group", "supergroup")
        was_mentioned = True  # DM default
        if is_group and self._bot_username:
            text_check = (message.text or message.caption or "").lower()
            was_mentioned = f"@{self._bot_username}" in text_check

        content_parts: list[str] = []
        media_paths: list[str] = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Handle media files
        annotations: list[str] = []
        if self.config.include_attachments:
            media_file = None
            media_type = None

            if message.photo:
                media_file = message.photo[-1]  # Largest size
                media_type = "image"
            elif message.voice:
                media_file = message.voice
                media_type = "voice"
            elif message.audio:
                media_file = message.audio
                media_type = "audio"
            elif message.video:
                media_file = message.video
                media_type = "video"
            elif message.document:
                media_file = message.document
                media_type = "file"
            elif message.sticker:
                media_file = message.sticker
                media_type = "sticker"

            # Location is not a downloadable file — handle separately
            if message.location and not media_file:
                loc = message.location
                annotations.append(
                    f"[位置] ({loc.latitude}, {loc.longitude})"
                )

            if media_file and self._app:
                file_size = getattr(media_file, 'file_size', 0) or 0
                too_large = self._check_attachment_size(file_size, media_type)
                if too_large:
                    annotations.append(too_large)
                else:
                    try:
                        file = await self._app.bot.get_file(
                            media_file.file_id,
                        )
                        ext = self._get_extension(
                            media_type,
                            getattr(media_file, 'mime_type', None),
                        )
                        file_path = self._media_path(
                            f"{media_file.file_id[:16]}{ext}"
                        )
                        await file.download_to_drive(str(file_path))

                        media_paths.append(str(file_path))
                        annotations.append(f"[{media_type}: {file_path}]")
                        logger.debug(
                            f"Downloaded {media_type} to {file_path}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to download media: {e}")
                        annotations.append(
                            f"[{media_type}: download failed]"
                        )

        text_content = "\n".join(content_parts) if content_parts else ""

        await self._enqueue_raw(RawIncoming(
            sender_id=user_id,
            chat_id=chat_id,
            text=text_content,
            media_files=media_paths,
            content_annotations=annotations,
            timestamp=message.date or datetime.now(),
            message_id=str(message.message_id),
            metadata={"chat_id": chat_id},
            is_group=is_group,
            was_mentioned=was_mentioned,
        ))

    _MIME_TO_EXT = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "audio/ogg": ".ogg", "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a", "video/mp4": ".mp4", "video/quicktime": ".mov",
    }
    _TYPE_TO_EXT = {
        "image": ".jpg", "voice": ".ogg", "audio": ".mp3",
        "video": ".mp4", "file": "", "sticker": ".webp",
    }

    @staticmethod
    def _get_extension(media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type and MIME type."""
        if mime_type and mime_type in TelegramChannel._MIME_TO_EXT:
            return TelegramChannel._MIME_TO_EXT[mime_type]
        return TelegramChannel._TYPE_TO_EXT.get(media_type, "")
