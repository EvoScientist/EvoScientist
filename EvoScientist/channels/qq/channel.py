"""QQ Bot channel — powered by botpy SDK.

Uses the official qq-botpy SDK for WebSocket connection and message handling.
No manual WebSocket protocol implementation needed.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from ..base import Channel, RawIncoming, ChannelError
from ..capabilities import QQ as QQ_CAPS
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)

try:
    import botpy
    from botpy.message import C2CMessage, GroupMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None
    GroupMessage = None


@dataclass
class QQConfig(BaseChannelConfig):
    app_id: str = ""
    app_secret: str = ""
    text_chunk_limit: int = 4096


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            super().__init__(intents=intents)

        async def on_ready(self):
            logger.info(f"QQ bot ready: {self.robot.name}")

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_msg(message, "c2c")

        async def on_group_at_message_create(self, message: "GroupMessage"):
            await channel._on_msg(message, "group")

    return _Bot


class QQChannel(Channel):
    capabilities = QQ_CAPS
    name = "qq"
    _ready_attrs = ("_client", "_running")
    _mention_pattern = r"@\S+\s*"
    _mention_strip_count = 1

    def __init__(self, config: QQConfig):
        super().__init__(config)
        self._client: "botpy.Client | None" = None
        self._bot_task: asyncio.Task | None = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._msg_seq: dict[str, int] = {}  # msg_id -> next seq number
        self._msg_seq_order: deque = deque(maxlen=500)

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        if not QQ_AVAILABLE:
            raise ChannelError("QQ SDK not installed. Run: pip install qq-botpy")
        if not self.config.app_id or not self.config.app_secret:
            raise ChannelError("QQ app_id and app_secret are required")
        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()
        self._bot_task = asyncio.create_task(self._run_bot())
        logger.info("QQ channel starting...")

    async def _run_bot(self) -> None:
        try:
            await self._client.start(appid=self.config.app_id, secret=self.config.app_secret)
        except Exception as e:
            logger.error(f"QQ auth failed: {e}")
            self._running = False

    # ── Incoming ──────────────────────────────────────────────────

    async def _on_msg(self, message, msg_type: str) -> None:
        try:
            if message.id in self._processed_ids:
                return
            self._processed_ids.append(message.id)

            author = message.author
            content = (message.content or "").strip()

            if msg_type == "c2c":
                sender_id = str(getattr(author, "user_openid", ""))
                chat_id = sender_id
            else:
                sender_id = str(getattr(author, "member_openid", ""))
                chat_id = str(getattr(message, "group_openid", ""))

            # Handle attachments (images, files, audio, video)
            annotations: list[str] = []
            media_paths: list[str] = []
            attachments = getattr(message, "attachments", None) or []
            for att in attachments:
                url = getattr(att, "url", "") or ""
                filename = getattr(att, "filename", "attachment") or "attachment"
                content_type = getattr(att, "content_type", "") or ""
                if url:
                    local, ann = await self._download_attachment(
                        url, f"qq_{filename}",
                    )
                    if local:
                        media_paths.append(local)
                    if ann:
                        annotations.append(ann)
                else:
                    annotations.append(f"[{content_type or 'attachment'}: {filename}]")

            if not content and not media_paths and not annotations:
                return

            await self._enqueue_raw(RawIncoming(
                sender_id=sender_id,
                chat_id=chat_id,
                text=content,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=datetime.now(),
                message_id=message.id,
                is_group=(msg_type == "group"),
                was_mentioned=True,
                metadata={
                    "chat_id": chat_id,
                    "msg_type": msg_type,
                    "event_id": message.id,
                    "backend": "qq",
                },
            ))
        except Exception as e:
            logger.error(f"Error handling QQ message: {e}")

    # ── Send ──────────────────────────────────────────────────────

    def _next_msg_seq(self, msg_id: str) -> int:
        """Return the next msg_seq for *msg_id* and increment the counter."""
        seq = self._msg_seq.get(msg_id, 1)
        self._msg_seq[msg_id] = seq + 1
        if msg_id not in set(self._msg_seq_order):
            self._msg_seq_order.append(msg_id)
            if len(self._msg_seq_order) > 500:
                oldest = self._msg_seq_order.popleft()
                self._msg_seq.pop(oldest, None)
        return seq

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        if not self._client:
            raise ChannelError("QQ client not initialized")
        msg_type = (metadata or {}).get("msg_type", "c2c")
        msg_id = (metadata or {}).get("event_id", "")
        seq = self._next_msg_seq(msg_id)
        if msg_type == "group":
            await self._client.api.post_group_message(
                group_openid=chat_id, msg_type=0,
                content=raw_text, msg_id=msg_id, msg_seq=seq,
            )
        else:
            await self._client.api.post_c2c_message(
                openid=chat_id, msg_type=0,
                content=raw_text, msg_id=msg_id, msg_seq=seq,
            )

    # _send_typing_action: inherited no-op (QQ Bot API has no typing indicator)

    # ── Media send ────────────────────────────────────────────────

    # qq-botpy file_type constants: 1=image, 2=video, 3=audio
    _FILE_TYPE_MAP = {
        ".jpg": 1, ".jpeg": 1, ".png": 1, ".gif": 1, ".webp": 1, ".bmp": 1,
        ".mp4": 2, ".mov": 2, ".avi": 2,
        ".mp3": 3, ".ogg": 3, ".m4a": 3, ".wav": 3, ".silk": 3,
    }

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file through QQ Bot API.

        Uses post_group_file / post_c2c_file with a URL.  Local files
        without a public URL are not supported — falls back to a text hint.
        """
        if not self._client:
            raise ChannelError("QQ client not initialized")

        from pathlib import Path
        chat_id = self._resolve_media_chat_id(recipient, metadata)
        msg_type = (metadata or {}).get("msg_type", "c2c")
        msg_id = (metadata or {}).get("event_id", "")
        ext = Path(file_path).suffix.lower()
        file_type = self._FILE_TYPE_MAP.get(ext, 1)  # default to image

        # qq-botpy file API requires a URL, not a local path
        is_url = file_path.startswith("http://") or file_path.startswith("https://")
        if not is_url:
            # Fallback: send text hint for local files
            name = Path(file_path).name
            hint = f"[文件] {name}" + (f"\n{caption}" if caption else "")
            await self._send_chunk(chat_id, hint, hint, None, metadata or {})
            return True

        try:
            if msg_type == "group":
                await self._client.api.post_group_file(
                    group_openid=chat_id,
                    file_type=file_type,
                    url=file_path,
                    srv_send_msg=True,
                )
            else:
                await self._client.api.post_c2c_file(
                    openid=chat_id,
                    file_type=file_type,
                    url=file_path,
                    srv_send_msg=True,
                )
        except Exception as e:
            logger.warning(f"QQ media send failed: {e}")
            return False

        if caption:
            await self._send_chunk(chat_id, caption, caption, None, metadata or {})
        return True

    # ── Cleanup ───────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        self._running = False
        if self._bot_task:
            self._bot_task.cancel()
            try:
                await self._bot_task
            except asyncio.CancelledError:
                pass
        self._client = None
        logger.info("QQ channel stopped")
