"""Signal channel implementation via signal-cli JSON RPC.

Pure Python — communicates with signal-cli daemon over TCP socket.

Architecture:
1. signal-cli must be running in JSON RPC mode:
   signal-cli -u +NUMBER daemon --socket localhost:7583
2. We connect via TCP, send JSON RPC requests, receive events
3. Inbound: listen for "receive" method notifications
4. Outbound: call "send" method via JSON RPC
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..base import Channel, RawIncoming, ChannelError
from ..capabilities import SIGNAL as SIGNAL_CAPS
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig(BaseChannelConfig):
    """Configuration for Signal channel."""
    phone_number: str = ""
    cli_path: str = "signal-cli"
    config_dir: str | None = None
    rpc_port: int = 7583
    text_chunk_limit: int = 4096


class SignalChannel(Channel):
    capabilities = SIGNAL_CAPS
    """Signal channel using signal-cli JSON RPC.

    No public IP needed — local TCP socket connection.
    Requires signal-cli to be installed and registered.
    """

    name = "signal"
    _non_retryable_patterns = ("unregistered", "auth")

    def __init__(self, config: SignalConfig):
        super().__init__(config)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._rpc_id = 0
        self._daemon_proc = None
        # Cache message_id → sender for reaction targetAuthor (bounded)
        self._msg_senders: dict[str, str] = {}
        self._msg_senders_order: deque = deque(maxlen=200)

    async def start(self) -> None:
        if not self.config.phone_number:
            raise ChannelError("Signal phone_number is required")

        # Try to start signal-cli daemon if not already running
        await self._ensure_daemon()

        # Connect to JSON RPC socket
        await self._connect()

        self._running = True
        logger.info(f"Signal channel started (phone: {self.config.phone_number})")

        # Listen for incoming messages in background task
        # (start() must return so that run() can iterate receive())
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def _ensure_daemon(self) -> None:
        """Start signal-cli daemon if not already running."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", self.config.rpc_port),
                timeout=2,
            )
            writer.close()
            await writer.wait_closed()
            logger.info("signal-cli daemon already running")
            return
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            pass

        # Start daemon
        cmd = [self.config.cli_path, "-u", self.config.phone_number]
        if self.config.config_dir:
            cmd.extend(["--config", self.config.config_dir])
        cmd.extend(["daemon", "--tcp",
                     f"localhost:{self.config.rpc_port}", "--no-receive-stdout"])

        logger.info(f"Starting signal-cli daemon: {' '.join(cmd)}")
        try:
            self._daemon_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise ChannelError(
                f"signal-cli not found at '{self.config.cli_path}'. "
                "Install: https://github.com/AsamK/signal-cli"
            )

        # Wait for daemon to be ready
        for _ in range(30):
            await asyncio.sleep(1)
            try:
                reader, writer = await asyncio.open_connection(
                    "localhost", self.config.rpc_port,
                )
                writer.close()
                await writer.wait_closed()
                logger.info("signal-cli daemon started")
                return
            except (ConnectionRefusedError, OSError):
                continue

        raise ChannelError("signal-cli daemon failed to start within 30s")

    async def _connect(self) -> None:
        """Connect to signal-cli JSON RPC socket."""
        try:
            self._reader, self._writer = await asyncio.open_connection(
                "localhost", self.config.rpc_port,
            )
        except Exception as e:
            raise ChannelError(f"Cannot connect to signal-cli: {e}")

    async def _listen_loop(self) -> None:
        """Listen for incoming JSON RPC notifications."""
        while self._running and self._reader:
            try:
                line = await self._reader.readline()
                if not line:
                    break
                data = json.loads(line.decode())
                await self._handle_rpc(data)
            except asyncio.CancelledError:
                break
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Signal listen error: {e}")
                # Reconnect
                if self._running:
                    await asyncio.sleep(2)
                    try:
                        await self._connect()
                    except Exception:
                        pass

    async def _handle_rpc(self, data: dict) -> None:
        """Handle a JSON RPC message from signal-cli."""
        method = data.get("method", "")

        if method != "receive":
            return

        params = data.get("params", {})
        envelope = params.get("envelope", {})
        source = envelope.get("source") or envelope.get("sourceUuid") or ""
        source_number = envelope.get("sourceNumber") or source
        source_name = envelope.get("sourceName") or ""
        timestamp = envelope.get("timestamp", 0)

        # Ignore messages from self
        if source_number == self.config.phone_number or source == self.config.phone_number:
            logger.debug("Ignoring message from self")
            return

        # Data message (text)
        data_msg = envelope.get("dataMessage", {})
        if data_msg:
            text = data_msg.get("message", "")
            group_info = data_msg.get("groupInfo", {})
            is_group = bool(group_info)
            chat_id = group_info.get("groupId", source_number) if is_group else source_number
            msg_ts = data_msg.get("timestamp", timestamp)

            media_paths: list[str] = []
            annotations: list[str] = []
            _VOICE_TYPES = {"audio/aac", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/opus"}
            attachments = data_msg.get("attachments", [])
            for att in attachments:
                att_size = att.get("size", 0)
                att_name = att.get("filename", "attachment")
                att_file = att.get("file")  # signal-cli provides local path
                content_type = att.get("contentType", "")
                is_voice = content_type in _VOICE_TYPES or att.get("voiceNote", False)
                media_label = "voice" if is_voice else "attachment"
                if att_file:
                    from pathlib import Path as _Path
                    att_path = _Path(att_file)
                    if att_path.exists():
                        from ..base import MAX_ATTACHMENT_BYTES
                        if att_path.stat().st_size > MAX_ATTACHMENT_BYTES:
                            annotations.append(f"[{media_label}: {att_name} - too large ({att_path.stat().st_size} bytes)]")
                        else:
                            local = self._media_path(f"signal_{att_name}")
                            import shutil
                            shutil.copy2(str(att_path), str(local))
                            media_paths.append(str(local))
                            annotations.append(f"[{media_label}: {local}]")
                    else:
                        annotations.append(f"[{media_label}: {att_name} - file not found]")
                elif att_size:
                    too_large = self._check_attachment_size(att_size, att_name)
                    if too_large:
                        annotations.append(too_large)
                    else:
                        annotations.append(f"[{media_label}: {att_name}]")

            if not text and not media_paths and not annotations:
                if not attachments:
                    return
                # Had attachments but none downloaded successfully
                if not annotations:
                    text = "[attachment]"

            try:
                ts = datetime.fromtimestamp(msg_ts / 1000) if msg_ts else datetime.now()
            except (ValueError, TypeError, OSError):
                ts = datetime.now()

            was_mentioned = not is_group  # DMs always pass
            if is_group:
                mentions = data_msg.get("mentions", [])
                for m in mentions:
                    if m.get("uuid") == self.config.phone_number or m.get("number") == self.config.phone_number:
                        was_mentioned = True
                        break

            # Cache message_id → sender for reaction targetAuthor
            self._cache_msg_sender(str(msg_ts), source_number)

            logger.info("Signal message from %s: %s", source_number, text[:50] if text else "[media]")
            await self._enqueue_raw(RawIncoming(
                sender_id=source_number,
                chat_id=chat_id,
                text=text,
                content_annotations=annotations,
                media_files=media_paths,
                timestamp=ts,
                message_id=str(msg_ts),
                is_group=is_group,
                was_mentioned=was_mentioned,
                metadata={
                    "chat_id": chat_id,
                    "source_name": source_name,
                    "sender_id": source_number,
                    "backend": "signal",
                },
            ))

    # ── Typing indicator ────────────────────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        """Send typing indicator via signal-cli JSON RPC."""
        params: dict[str, Any] = {
            "account": self.config.phone_number,
        }
        if self._is_group_id(chat_id):
            params["groupId"] = chat_id
        else:
            params["recipient"] = [chat_id]
        try:
            await self._rpc_call("sendTyping", params)
        except Exception:
            pass  # typing indicator is best-effort

    # ── ACK reaction ─────────────────────────────────────────────

    def _cache_msg_sender(self, message_id: str, sender: str) -> None:
        """Store message_id → sender mapping for reaction targetAuthor."""
        if len(self._msg_senders) >= 200:
            oldest = self._msg_senders_order.popleft()
            self._msg_senders.pop(oldest, None)
        self._msg_senders[message_id] = sender
        self._msg_senders_order.append(message_id)

    async def _send_ack_reaction(self, chat_id: str, message_id: str, emoji: str = "👀") -> None:
        """Send an acknowledgment reaction via signal-cli sendReaction."""
        target_author = self._msg_senders.get(message_id, "")
        if not target_author:
            return  # cannot send reaction without knowing the original sender
        try:
            params: dict[str, Any] = {
                "account": self.config.phone_number,
                "emoji": emoji,
                "targetAuthor": target_author,
                "targetTimestamp": int(message_id),
            }
            if self._is_group_id(chat_id):
                params["groupId"] = chat_id
            else:
                params["recipient"] = [chat_id]
            await self._rpc_call("sendReaction", params)
        except Exception as e:
            logger.debug(f"Signal ack reaction failed: {e}")

    async def _remove_ack_reaction(self, chat_id: str, message_id: str, emoji: str = "👀") -> None:
        """Remove ACK reaction via signal-cli sendReaction --remove."""
        target_author = self._msg_senders.get(message_id, "")
        if not target_author:
            return
        try:
            params: dict[str, Any] = {
                "account": self.config.phone_number,
                "emoji": emoji,
                "targetAuthor": target_author,
                "targetTimestamp": int(message_id),
                "remove": True,
            }
            if self._is_group_id(chat_id):
                params["groupId"] = chat_id
            else:
                params["recipient"] = [chat_id]
            await self._rpc_call("sendReaction", params)
        except Exception as e:
            logger.debug(f"Signal remove ACK reaction failed: {e}")

    # ── Send ──────────────────────────────────────────────────────

    @staticmethod
    def _is_group_id(chat_id: str) -> bool:
        """Return True if *chat_id* looks like a Signal group ID.

        Group IDs are base64-encoded strings (e.g. ``"aB3d...=="``).
        Individual recipients are either phone numbers (``"+1234..."``)
        or UUIDs (``"817ab5e9-..."``) — neither of which is a group.
        """
        return not chat_id.startswith("+") and "-" not in chat_id

    def _is_ready(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def _rpc_call(self, method: str, params: dict) -> dict | None:
        """Send a JSON RPC call to signal-cli."""
        if not self._writer:
            return None

        self._rpc_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(request) + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()
        return None  # We don't wait for response in this simple impl

    async def _send_chunk(
        self, chat_id, formatted_text, raw_text, reply_to, metadata,
    ):
        # Determine if group or individual
        params: dict[str, Any] = {
            "message": raw_text,
            "account": self.config.phone_number,
        }

        if self._is_group_id(chat_id):
            params["groupId"] = chat_id
        else:
            params["recipient"] = [chat_id]

        await self._rpc_call("send", params)

    # ── Formatting ────────────────────────────────────────────────


    # ── Mention stripping ────────────────────────────────────────────

    def _strip_mention(self, text: str) -> str:
        """Strip bot mention from Signal messages.

        Signal mentions are embedded as special objects that reference
        the phone number. The text contains a placeholder character (U+FFFC)
        at the mention position.
        """
        phone = self.config.phone_number
        if phone:
            # Remove phone number if directly mentioned as text
            text = re.sub(rf"@?{re.escape(phone)}\s*", "", text).strip()
        # Remove Unicode Object Replacement Character used as mention placeholder
        text = text.replace("\uFFFC", "").strip()
        return text

    # ── Media send ────────────────────────────────────────────────

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file via signal-cli JSON RPC.

        Uses the "send" RPC method with the attachments parameter.
        """
        chat_id = self._resolve_media_chat_id(recipient, metadata)
        params: dict[str, Any] = {
            "account": self.config.phone_number,
            "attachments": [file_path],
        }
        if caption:
            params["message"] = caption

        if self._is_group_id(chat_id):
            params["groupId"] = chat_id
        else:
            params["recipient"] = [chat_id]

        await self._rpc_call("send", params)
        return True

    # ── Cleanup ───────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        if hasattr(self, "_listen_task") and self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
        if self._daemon_proc:
            self._daemon_proc.terminate()
            self._daemon_proc = None
        logger.info("Signal channel stopped")
