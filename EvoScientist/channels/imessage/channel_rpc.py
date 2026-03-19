"""iMessage channel using imsg JSON-RPC.

This is an improved implementation that uses the imsg CLI
via JSON-RPC, similar to OpenClaw's approach.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..base import Channel, ChannelError, RawIncoming
from ..config import BaseChannelConfig
from .rpc_client import ImsgRpcClient, RpcNotification
from .targets import (
    ChatGuidTarget,
    ChatIdentifierTarget,
    ChatIdTarget,
    normalize_handle,
    parse_target,
)

logger = logging.getLogger(__name__)


class _IMessageAllowListMiddleware:
    """Custom allow-list middleware for iMessage's rich sender filtering.

    Supports chat_id/chat_guid matching, wildcard, and normalized
    phone/email matching — logic that the generic AllowListMiddleware
    does not cover.
    """

    def __init__(self, channel: "IMessageChannelRpc"):
        self._channel = channel

    async def process_inbound(self, raw, context):
        chat_id = raw.metadata.get("chat_id")
        chat_guid = raw.metadata.get("chat_guid")
        if not self._channel._is_sender_allowed(raw.sender_id, chat_id, chat_guid):
            return None
        return raw


@dataclass
class IMessageConfig(BaseChannelConfig):
    """Configuration for iMessage channel."""

    cli_path: str = "imsg"
    db_path: str | None = None
    text_chunk_limit: int = 4096
    service: str = "auto"  # imessage, sms, or auto
    region: str = "US"


class IMessageChannelRpc(Channel):
    """iMessage channel using imsg JSON-RPC.

    This implementation uses the imsg CLI via JSON-RPC over stdio,
    providing real-time message streaming instead of polling.

    Args:
        config: Channel configuration
    """

    name = "imessage"
    _ready_attrs = ("_client",)

    def __init__(self, config: IMessageConfig | None = None):
        super().__init__(config or IMessageConfig())
        self._client: ImsgRpcClient | None = None
        self._subscription_id: int | None = None
        self._background_tasks: set[asyncio.Task] = set()

    # ── Pipeline overrides ────────────────────────────────────────

    def _build_inbound_middlewares(self):
        """Use iMessage-specific allow-list middleware.

        iMessage doesn't need MentionGating (always sets was_mentioned=True).
        """
        from ..middleware import DedupMiddleware, GroupHistoryMiddleware

        middlewares = []
        middlewares.append(DedupMiddleware())
        middlewares.append(_IMessageAllowListMiddleware(self))
        if self.capabilities.groups:
            middlewares.append(GroupHistoryMiddleware())
        return middlewares

    # ── Incoming message handling ─────────────────────────────────

    def _handle_notification(self, notification: RpcNotification) -> None:
        """Handle incoming RPC notifications."""
        if notification.method == "message":
            _task = asyncio.create_task(self._handle_message(notification.params))
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
        elif notification.method == "error":
            logger.error(f"imsg error: {notification.params}")

    async def _handle_message(self, params: dict | None) -> None:
        """Process incoming message notification."""
        if not params:
            return

        message = params.get("message", {})
        if not message:
            return

        # Skip messages from self
        if message.get("is_from_me"):
            return

        sender = message.get("sender", "").strip()
        if not sender:
            return

        text = message.get("text", "").strip()

        # Parse timestamp
        timestamp = datetime.now()
        if created_at := message.get("created_at"):
            try:
                timestamp = datetime.fromisoformat(created_at)
            except ValueError:
                pass

        # Build metadata
        metadata = {
            "chat_id": message.get("chat_id"),
            "chat_guid": message.get("chat_guid"),
            "is_group": message.get("is_group", False),
            "chat_name": message.get("chat_name"),
        }

        # Handle attachments if enabled
        annotations: list[str] = []
        media_paths: list[str] = []
        _VOICE_EXTS = {".caf", ".m4a", ".aac", ".ogg", ".opus", ".mp3", ".amr"}
        if self.config.include_attachments:
            attachments = message.get("attachments", [])
            for att in attachments:
                # imsg CLI provides local file paths for attachments
                file_path = att if isinstance(att, str) else att.get("path", "")
                if not file_path:
                    annotations.append("[attachment: missing path]")
                    continue
                att_path = Path(file_path)
                is_voice = att_path.suffix.lower() in _VOICE_EXTS
                media_label = "voice" if is_voice else "attachment"
                if att_path.exists():
                    fname = att_path.name
                    # Check file size before copying
                    from ..base import MAX_ATTACHMENT_BYTES

                    if att_path.stat().st_size > MAX_ATTACHMENT_BYTES:
                        annotations.append(
                            f"[{media_label}: {fname} - too large "
                            f"({att_path.stat().st_size} bytes)]"
                        )
                    else:
                        local = self._media_path(f"imsg_{fname}")
                        try:
                            import shutil

                            shutil.copy2(str(att_path), str(local))
                            media_paths.append(str(local))
                            annotations.append(f"[{media_label}: {local}]")
                        except Exception as e:
                            logger.warning(f"Failed to copy iMessage attachment: {e}")
                            annotations.append(
                                f"[{media_label}: {fname} - copy failed]"
                            )
                else:
                    annotations.append(f"[{media_label}: {file_path} - not found]")

        if not text and not media_paths and not annotations:
            return

        is_group = message.get("is_group", False)

        await self._enqueue_raw(
            RawIncoming(
                sender_id=sender,
                chat_id=str(metadata.get("chat_id", sender)),
                text=text,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=timestamp,
                message_id=str(message.get("id", "")),
                metadata=metadata,
                is_group=is_group,
                was_mentioned=True,  # iMessage has no mention concept
            )
        )

    # ── Sender filtering ──────────────────────────────────────────

    def _is_sender_allowed(
        self,
        sender: str,
        chat_id: int | None = None,
        chat_guid: str | None = None,
    ) -> bool:
        """Check if sender is in allowed list.

        Supports:
        - Wildcard "*" to allow all
        - chat_id:123 to match by chat ID
        - chat_guid:abc to match by chat GUID
        - Normalized phone/email matching
        """
        if not self.config.allowed_senders:
            return True

        # Wildcard allows all
        if "*" in self.config.allowed_senders:
            return True

        sender_normalized = normalize_handle(sender)

        for entry in self.config.allowed_senders:
            entry = entry.strip()
            if not entry:
                continue

            lower = entry.lower()

            # Check chat_id match
            if lower.startswith("chat_id:") or lower.startswith("chatid:"):
                if chat_id is not None:
                    try:
                        allowed_id = int(entry.split(":", 1)[1].strip())
                        if allowed_id == chat_id:
                            return True
                    except ValueError:
                        pass
                continue

            # Check chat_guid match
            if lower.startswith("chat_guid:") or lower.startswith("chatguid:"):
                if chat_guid:
                    allowed_guid = entry.split(":", 1)[1].strip()
                    if allowed_guid == chat_guid:
                        return True
                continue

            # Normalize and compare handle
            entry_normalized = normalize_handle(entry)
            if entry_normalized == sender_normalized:
                return True

        return False

    def _normalize_sender(self, sender: str) -> str:
        """Normalize a sender identifier."""
        return sender if sender.startswith("chat") else normalize_handle(sender)

    def add_allowed_sender(self, sender: str) -> None:
        """Add a sender to the allowed list."""
        normalized = self._normalize_sender(sender)
        if self.config.allowed_senders is None:
            self.config.allowed_senders = set()
        self.config.allowed_senders.add(normalized)
        logger.info(f"Added allowed sender: {normalized}")

    def remove_allowed_sender(self, sender: str) -> None:
        """Remove a sender from the allowed list."""
        normalized = self._normalize_sender(sender)
        if self.config.allowed_senders:
            self.config.allowed_senders.discard(normalized)
            logger.info(f"Removed allowed sender: {normalized}")

    def clear_allowed_senders(self) -> None:
        """Clear allowed list (allow all)."""
        self.config.allowed_senders = None
        logger.info("Cleared allowed senders (allowing all)")

    def list_allowed_senders(self) -> list[str]:
        """Get current allowed senders."""
        return list(self.config.allowed_senders) if self.config.allowed_senders else []

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize and start the channel."""
        logger.info("Starting iMessage channel (RPC)...")

        self._client = ImsgRpcClient(
            cli_path=self.config.cli_path,
            db_path=self.config.db_path,
            on_notification=self._handle_notification,
        )

        try:
            await self._client.start()
        except Exception as e:
            raise ChannelError(f"Failed to start imsg: {e}") from e

        # Subscribe to message events
        try:
            result = await self._client.request(
                "watch.subscribe",
                {"attachments": self.config.include_attachments},
            )
            self._subscription_id = result.get("subscription")
        except Exception as e:
            await self._client.stop()
            raise ChannelError(f"Failed to subscribe: {e}") from e

        self._running = True
        logger.info("iMessage channel started")

    async def _cleanup(self) -> None:
        if self._client and self._subscription_id:
            try:
                await self._client.request(
                    "watch.unsubscribe",
                    {"subscription": self._subscription_id},
                )
            except Exception:
                pass
        if self._client:
            await self._client.stop()
            self._client = None
        logger.info("iMessage channel stopped")

    # ── Send (template method overrides) ──────────────────────────

    def _resolve_target(self, chat_id: str | None, metadata: dict | None) -> dict:
        """Resolve send target from metadata or chat_id string."""
        meta = metadata or {}
        for key in ("chat_id", "chat_guid", "chat_identifier"):
            if meta.get(key):
                return {key: meta[key]}
        if chat_id:
            try:
                target = parse_target(chat_id)
                if isinstance(target, ChatIdTarget):
                    return {"chat_id": target.chat_id}
                elif isinstance(target, ChatGuidTarget):
                    return {"chat_guid": target.chat_guid}
                elif isinstance(target, ChatIdentifierTarget):
                    return {"chat_identifier": target.chat_identifier}
                else:
                    return {"to": target.to, "service": target.service.value}
            except ValueError:
                return {"to": chat_id}
        return {}

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        """Send a single text chunk via iMessage RPC."""
        if not self._client:
            raise RuntimeError("iMessage client not running")

        params: dict = {
            "text": formatted_text,
            "service": self.config.service,
            "region": self.config.region,
        }
        params.update(self._resolve_target(chat_id, metadata))

        if reply_to:
            params["reply_to"] = reply_to

        await self._client.request("send", params)

    # ── Retry logic (override base) ───────────────────────────────

    def _format_chunk(self, text: str) -> str:
        """iMessage uses plain text; no formatting conversion needed."""
        return text

    def _extract_retry_after(self, exc: Exception) -> float | None:
        """iMessage-specific retry logic.

        RPC errors (e.g. AppleScript failures) are generally not
        retryable.  Transient connection issues get a short retry.
        """
        msg = str(exc).lower()
        if "not found" in msg or "applescript" in msg or "permission" in msg:
            return None  # not retryable
        if "timeout" in msg or "connection" in msg:
            return 1.0
        return None  # default: don't retry RPC errors

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file via iMessage."""
        if not self._client:
            return False

        params: dict = {
            "file": file_path,
            "service": self.config.service,
            "region": self.config.region,
        }

        if caption:
            params["text"] = caption

        target = self._resolve_target(recipient, metadata)
        if not target:
            logger.error("Cannot send media: no recipient")
            return False
        params.update(target)

        await self._client.request("send", params)
        return True
