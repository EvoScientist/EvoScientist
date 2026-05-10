"""QQ channel implementation using botpy SDK."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from ..base import Channel, ChannelError, RawIncoming
from ..capabilities import QQ as QQ_CAPS
from ..config import BaseChannelConfig
from ..formatter import UnifiedFormatter

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


# ── Inline keyboard (button) helpers ─────────────────────────────────


def _normalize_button(btn: dict) -> tuple[str, str] | None:
    """Return ``(label, value)`` for a button, or ``None`` if no label."""
    label = (btn.get("text") or "").strip()
    if not label:
        return None
    raw = btn.get("value")
    return label, str(raw) if raw is not None else label


def _build_qq_keyboard(buttons: list[dict]) -> dict | None:
    """Build a QQ Bot keyboard payload (one button per row).

    Render style: 1 = primary (blue), 0 = secondary (grey) — QQ has no danger.
    ``action.permission`` is required by the schema; ``type=2`` is harmless for
    C2C (the click always comes from the DM peer). Returns ``None`` if no
    button has a usable label.
    """
    rows: list[dict] = []
    for idx, btn in enumerate(buttons):
        norm = _normalize_button(btn)
        if norm is None:
            continue
        label, value = norm
        style = 1 if btn.get("type") == "primary" else 0
        rows.append(
            {
                "buttons": [
                    {
                        "id": btn.get("id") or f"btn_{idx}",
                        "render_data": {
                            "label": label,
                            "visited_label": label,
                            "style": style,
                        },
                        "action": {
                            "type": 1,  # callback (server pushes interaction event)
                            "permission": {"type": 2},
                            "data": value,
                        },
                    }
                ]
            }
        )
    return {"content": {"rows": rows}} if rows else None


@dataclass
class QQConfig(BaseChannelConfig):
    app_id: str = ""
    app_secret: str = ""
    text_chunk_limit: int = 4096


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(
        public_messages=True,
        direct_message=True,
        interaction=True,  # button clicks → on_interaction_create
    )

    class _Bot(botpy.Client):
        def __init__(self):
            super().__init__(intents=intents)

        async def on_ready(self):
            logger.info(f"QQ bot ready: {self.robot.name}")

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_msg(message, "c2c")

        async def on_group_at_message_create(self, message: "GroupMessage"):
            await channel._on_msg(message, "group")

        async def on_interaction_create(self, interaction):
            await channel._on_interaction(interaction)

    return _Bot


class QQChannel(Channel):
    """QQ channel using botpy SDK."""

    name = "qq"

    capabilities = QQ_CAPS
    _ready_attrs = ("_client", "_running")
    _non_retryable_patterns = ()
    _mention_pattern = r"@\S+\s*"
    _mention_strip_count = 1
    _markdown_fallback_exc_types: ClassVar[tuple[type[Exception], ...]] = (
        TypeError,
        ValueError,
    )

    def __init__(self, config: QQConfig):
        super().__init__(config)
        self._client: botpy.Client | None = None
        self._bot_task: asyncio.Task | None = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._msg_seq: dict[str, int] = {}  # msg_id -> next seq number
        self._msg_seq_order: deque = deque(maxlen=500)
        self._msg_seq_ids: set[str] = set()  # companion set for O(1) lookup
        self._plain_formatter = UnifiedFormatter.for_channel("plain")

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
            await self._client.start(
                appid=self.config.app_id, secret=self.config.app_secret
            )
        except Exception as e:
            logger.error(f"QQ auth failed: {e}")
            self._running = False

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
                        url,
                        f"qq_{filename}",
                    )
                    if local:
                        media_paths.append(local)
                    if ann:
                        annotations.append(ann)
                else:
                    annotations.append(f"[{content_type or 'attachment'}: {filename}]")

            if not content and not media_paths and not annotations:
                return

            await self._enqueue_raw(
                RawIncoming(
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
                )
            )
        except Exception as e:
            logger.error(f"Error handling QQ message: {e}")

    async def _on_interaction(self, interaction) -> None:
        """Handle ``on_interaction_create`` (button click).

        Surfaces the click as an :class:`InboundMessage` whose ``content`` is
        the button's ``data`` verbatim — so a "1"/"approve"/… click flows
        through ``_parse_approval_reply`` exactly like a typed reply.

        The click runs through inbound middleware (Dedup suppresses QQ
        retries) but is published directly to the bus so the per-sender
        debounce buffer doesn't merge the click value with subsequent text.
        Group-scope clicks are ignored (DM-only by design).
        """
        # ACK first — QQ requires a response within ~5s or the button UI
        # shows "expired".  Code 0 just means "received"; downstream still
        # decides the actual approval/rejection.
        interaction_id = getattr(interaction, "id", "") or ""
        if interaction_id and self._client:
            try:
                await self._client.api.on_interaction_result(interaction_id, 0)
            except Exception as ack_exc:
                logger.debug("QQ interaction ack failed: %s", ack_exc)

        try:
            user_openid = getattr(interaction, "user_openid", "") or ""
            if not user_openid:
                logger.debug("QQ interaction ignored (no user_openid; not C2C)")
                return

            resolved = getattr(getattr(interaction, "data", None), "resolved", None)
            button_data = getattr(resolved, "button_data", "") or ""
            button_id = getattr(resolved, "button_id", "") or ""
            triggering_msg_id = getattr(resolved, "message_id", "") or ""

            # QQ may serialize non-str values; coerce.  Fall back to button id
            # when no data — same path as a typed reply via _parse_approval_reply.
            button_value = str(button_data) if button_data != "" else ""
            text = button_value or button_id

            # Stable id so DedupMiddleware suppresses any QQ retry callbacks.
            message_id = (
                f"{triggering_msg_id}:action:{interaction_id}"
                if interaction_id
                else f"qq_action:{datetime.now().timestamp()}"
            )

            raw = RawIncoming(
                sender_id=user_openid,
                chat_id=user_openid,  # C2C: chat_id == user_openid
                text=text,
                timestamp=datetime.now(),
                message_id=message_id,
                metadata={
                    "chat_id": user_openid,
                    "msg_type": "c2c",
                    "event_id": triggering_msg_id,
                    "backend": "qq",
                    "button_click": True,
                    "button_id": button_id,
                    "button_value": button_value,
                },
                is_group=False,
                was_mentioned=True,
            )

            inbound = await self._build_inbound_async(raw)
            if inbound is not None and self._bus:
                await self._bus.publish_inbound(inbound)
        except Exception:
            logger.exception("QQ interaction handler error")

    # ── Send ──────────────────────────────────────────────────────

    def _next_msg_seq(self, msg_id: str) -> int:
        """Return the next msg_seq for *msg_id* and increment the counter."""
        seq = self._msg_seq.get(msg_id, 1)
        self._msg_seq[msg_id] = seq + 1
        if msg_id not in self._msg_seq_ids:
            self._msg_seq_order.append(msg_id)
            self._msg_seq_ids.add(msg_id)
            if len(self._msg_seq_order) > 500:
                oldest = self._msg_seq_order.popleft()
                self._msg_seq_ids.discard(oldest)
                self._msg_seq.pop(oldest, None)
        return seq

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        if not self._client:
            raise ChannelError("QQ client not initialized")
        msg_type = (metadata or {}).get("msg_type", "c2c")
        msg_id = (metadata or {}).get("event_id", "")
        seq = self._next_msg_seq(msg_id)

        # Inline keyboard is C2C-only here — group keyboards have stricter
        # permission semantics and are out of scope for now.
        buttons = (metadata or {}).get("buttons") if msg_type == "c2c" else None
        keyboard = _build_qq_keyboard(buttons) if buttons else None

        try:
            await self._post_markdown_message(
                chat_id, raw_text, msg_type, msg_id, seq, keyboard=keyboard
            )
            return
        except Exception as exc:
            if not self._should_fallback_to_plain_text(exc):
                logger.error(
                    "QQ markdown send failed with non-fallbackable error "
                    "(chat_id=%s, msg_id=%s, seq=%s): %r",
                    chat_id,
                    msg_id,
                    seq,
                    exc,
                )
                raise
            self._record_markdown_fallback(chat_id, raw_text, exc)
            logger.warning(
                "QQ markdown send failed, falling back to plain text "
                "(chat_id=%s, msg_id=%s, seq=%s): %r",
                chat_id,
                msg_id,
                seq,
                exc,
            )

        # QQ may have already consumed `seq` server-side even on failure.
        # Reusing it for the plain retry triggers "duplicate msg_seq", so
        # always advance to a fresh seq before the fallback send.
        fallback_seq = self._next_msg_seq(msg_id)
        plain_text = self._plain_formatter.format(raw_text)
        # Plain-text fallback can't carry a keyboard.  Append `value=label`
        # pairs so the user can still type "1"/"approve"/… instead of
        # tapping (`_parse_approval_reply` accepts the same values).
        if buttons:
            pairs = []
            for btn in buttons:
                norm = _normalize_button(btn)
                if norm is not None:
                    label, value = norm
                    pairs.append(f"{value}={label}")
            if pairs:
                plain_text = f"{plain_text}\n\nReply: {', '.join(pairs)}"
        try:
            await self._post_plain_message(
                chat_id, plain_text, msg_type, msg_id, fallback_seq
            )
        except Exception as plain_exc:
            logger.error(
                "QQ plain fallback also failed (chat_id=%s, msg_id=%s, seq=%s): %r",
                chat_id,
                msg_id,
                fallback_seq,
                plain_exc,
            )
            raise

    # QQ server-side error codes / fragments that indicate the markdown
    # request itself is invalid (template not configured, format rejected,
    # content audit, etc.). Seeing any of these means we should retry with
    # plain text rather than re-raise.
    _QQ_MARKDOWN_ERROR_MARKERS: ClassVar[tuple[str, ...]] = (
        "304014",  # markdown template not configured
        "304003",  # invalid markdown params
        "40034059",  # generic send message failed (often markdown-related)
        "模板",  # CN: template (standard form)
        "模版",  # CN: template (variant form)
        "审核",  # CN: audit
    )

    def _should_fallback_to_plain_text(self, exc: Exception) -> bool:
        """Return True only for markdown compatibility/validation failures."""
        if isinstance(exc, self._markdown_fallback_exc_types):
            return True

        msg = str(exc).lower()
        compatibility_tokens = ("unsupported", "unexpected", "unknown", "invalid")

        if "unexpected keyword argument" in msg:
            return True
        if "markdown" in msg and any(token in msg for token in compatibility_tokens):
            return True
        if "msg_type" in msg and any(token in msg for token in compatibility_tokens):
            return True

        # QQ-specific server error codes returned by qq-botpy as strings.
        raw = str(exc)
        for marker in self._QQ_MARKDOWN_ERROR_MARKERS:
            if marker in raw or marker.lower() in msg:
                return True
        return False

    def _record_markdown_fallback(
        self,
        chat_id: str,
        raw_text: str,
        exc: Exception,
    ) -> None:
        """Emit optional debug trace for markdown fallback without blocking send."""
        trace_event = getattr(self, "_trace_event", None)
        if not callable(trace_event):
            return
        try:
            trace_event(
                "outbound_format_fallback",
                chat_id=chat_id,
                error=str(exc),
                formatted_len=len(raw_text),
                raw_len=len(raw_text),
            )
        except Exception as trace_exc:
            logger.debug("QQ fallback trace failed: %s", trace_exc)

    async def _post_markdown_message(
        self,
        chat_id: str,
        text: str,
        msg_type: str,
        msg_id: str,
        seq: int,
        keyboard: dict | None = None,
    ) -> None:
        payload = {
            "msg_type": 2,
            "markdown": {"content": text},
            "msg_id": msg_id,
            "msg_seq": seq,
        }
        if keyboard is not None:
            payload["keyboard"] = keyboard
        if msg_type == "group":
            await self._client.api.post_group_message(
                group_openid=chat_id,
                **payload,
            )
        else:
            await self._client.api.post_c2c_message(
                openid=chat_id,
                **payload,
            )

    async def _post_plain_message(
        self,
        chat_id: str,
        text: str,
        msg_type: str,
        msg_id: str,
        seq: int,
    ) -> None:
        payload = {
            "msg_type": 0,
            "content": text,
            "msg_id": msg_id,
            "msg_seq": seq,
        }
        if msg_type == "group":
            await self._client.api.post_group_message(
                group_openid=chat_id,
                **payload,
            )
        else:
            await self._client.api.post_c2c_message(
                openid=chat_id,
                **payload,
            )

    # _send_typing_action: inherited no-op (QQ Bot API has no typing indicator)

    # ── Media send ────────────────────────────────────────────────

    # qq-botpy file_type constants: 1=image, 2=video, 3=audio
    _FILE_TYPE_MAP: ClassVar[dict[str, int]] = {
        ".jpg": 1,
        ".jpeg": 1,
        ".png": 1,
        ".gif": 1,
        ".webp": 1,
        ".bmp": 1,
        ".mp4": 2,
        ".mov": 2,
        ".avi": 2,
        ".mp3": 3,
        ".ogg": 3,
        ".m4a": 3,
        ".wav": 3,
        ".silk": 3,
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
