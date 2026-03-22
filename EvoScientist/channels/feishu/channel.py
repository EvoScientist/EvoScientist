"""Feishu (飞书/Lark) channel implementation.

Receives messages via HTTP event subscription webhook (aiohttp) or
WebSocket long connection (lark-oapi SDK), sends replies via Feishu
Open API REST endpoints.

Feishu Open API docs: https://open.feishu.cn/document

Authentication:
  - App ID + App Secret → tenant_access_token (2-hour TTL, auto-refreshed)

Event subscription (two modes):
  - **Webhook**: URL verification challenge on first request,
    ``im.message.receive_v1`` events via HTTP POST callback.
    Requires a publicly reachable URL.
  - **WebSocket**: outbound long connection via ``lark-oapi`` SDK.
    No public IP required.

Send API:
  - ``POST /open-apis/im/v1/messages?receive_id_type=chat_id``
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from aiohttp import web

from ..base import Channel, ChannelError, RawIncoming
from ..capabilities import FEISHU as FEISHU_CAPS
from ..config import BaseChannelConfig
from ..mixins import TokenMixin, WebhookMixin

logger = logging.getLogger(__name__)


# ── Markdown → Feishu Post conversion ────────────────────────────


def _parse_inline_text(text: str) -> list[dict]:
    """Parse inline Markdown elements into Feishu post tag dicts.

    Handles: `code`, **bold**, ~~strikethrough~~, [link](url), _italic_.
    """
    elements: list[dict] = []
    # Pattern order matters: code first (protect content), then bold, strikethrough, link, italic
    pattern = re.compile(
        r"`([^`]+)`"  # inline code
        r"|\*\*(.+?)\*\*"  # bold
        r"|~~(.+?)~~"  # strikethrough
        r"|\[([^\]]+)\]\(([^)]+)\)"  # link
        r"|_(.+?)_"  # italic
    )
    pos = 0
    for m in pattern.finditer(text):
        # Plain text before this match
        if m.start() > pos:
            elements.append({"tag": "text", "text": text[pos : m.start()]})

        if m.group(1) is not None:
            # inline code → code_block would be block-level; use text with style
            elements.append(
                {
                    "tag": "text",
                    "text": m.group(1),
                    "style": ["code_block"],
                }
            )
        elif m.group(2) is not None:
            elements.append(
                {
                    "tag": "text",
                    "text": m.group(2),
                    "style": ["bold"],
                }
            )
        elif m.group(3) is not None:
            elements.append(
                {
                    "tag": "text",
                    "text": m.group(3),
                    "style": ["strikethrough"],
                }
            )
        elif m.group(4) is not None:
            elements.append(
                {
                    "tag": "a",
                    "text": m.group(4),
                    "href": m.group(5),
                }
            )
        elif m.group(6) is not None:
            elements.append(
                {
                    "tag": "text",
                    "text": m.group(6),
                    "style": ["italic"],
                }
            )
        pos = m.end()

    # Remaining plain text
    if pos < len(text):
        elements.append({"tag": "text", "text": text[pos:]})
    return elements


def _parse_inline_elements(line: str) -> list[dict]:
    """Parse a single Markdown line into a list of Feishu post elements.

    Handles headings (→ bold), blockquotes (→ italic with prefix),
    list items (→ bullet prefix), and plain lines.
    """
    # Heading: # Title → bold text
    heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if heading_match:
        return [{"tag": "text", "text": heading_match.group(2), "style": ["bold"]}]

    # Blockquote: > text → italic with "▎" prefix
    quote_match = re.match(r"^>\s*(.*)$", line)
    if quote_match:
        inner = quote_match.group(1)
        elements = [{"tag": "text", "text": "▎", "style": ["italic"]}]
        elements.extend(_parse_inline_text(inner))
        return elements

    # Unordered list: - item or * item → "• " prefix
    list_match = re.match(r"^[\-\*]\s+(.+)$", line)
    if list_match:
        elements = [{"tag": "text", "text": "• "}]
        elements.extend(_parse_inline_text(list_match.group(1)))
        return elements

    # Ordered list: 1. item → keep number prefix
    ol_match = re.match(r"^(\d+)\.\s+(.+)$", line)
    if ol_match:
        elements = [{"tag": "text", "text": f"{ol_match.group(1)}. "}]
        elements.extend(_parse_inline_text(ol_match.group(2)))
        return elements

    # Plain line
    return _parse_inline_text(line)


def _markdown_to_feishu_post(text: str) -> dict | None:
    """Convert Markdown text to Feishu post (rich text) JSON structure.

    Returns a dict like {"zh_cn": {"content": [[...]]}} suitable for
    Feishu msg_type="post", or None if the text is empty.
    """
    if not text or not text.strip():
        return None

    paragraphs: list[list[dict]] = []
    current_paragraph: list[dict] = []
    in_code_block = False
    code_lines: list[str] = []
    code_lang = ""

    for line in text.split("\n"):
        # Code block fences
        if line.startswith("```"):
            if not in_code_block:
                # Flush any pending paragraph
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
                in_code_block = True
                code_lang = line[3:].strip()
                code_lines = []
            else:
                # End of code block
                code_text = "\n".join(code_lines)
                paragraphs.append(
                    [
                        {
                            "tag": "code_block",
                            "language": code_lang or "plain",
                            "text": code_text,
                        }
                    ]
                )
                in_code_block = False
                code_lines = []
                code_lang = ""
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Empty line → new paragraph
        if not line.strip():
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
            continue

        # Non-empty line
        elements = _parse_inline_elements(line)
        if elements:
            # Each visual line becomes its own paragraph in Feishu post
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = elements

    # Flush remaining
    if in_code_block and code_lines:
        code_text = "\n".join(code_lines)
        paragraphs.append(
            [
                {
                    "tag": "code_block",
                    "language": code_lang or "plain",
                    "text": code_text,
                }
            ]
        )
    elif current_paragraph:
        paragraphs.append(current_paragraph)

    if not paragraphs:
        return None

    return {"zh_cn": {"content": paragraphs}}


@dataclass
class FeishuConfig(BaseChannelConfig):
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encrypt_key: str = ""
    webhook_port: int = 9000
    text_chunk_limit: int = 4096
    feishu_domain: str = "https://open.feishu.cn"
    subscription_mode: str = "webhook"  # "webhook" | "websocket"


class FeishuChannel(Channel, WebhookMixin, TokenMixin):
    capabilities = FEISHU_CAPS
    """Feishu channel using Open API + event subscription webhook."""

    name = "feishu"
    _ready_attrs = ("_http_client", "_access_token")
    _non_retryable_patterns = (
        "app_access_token is empty",  # invalid credentials
        "10003",  # invalid app_id
        "10014",  # invalid app_secret
        "99991401",  # permission denied
        "99991663",  # no permission
        "99991672",  # feature not enabled
    )
    _rate_limit_patterns = ("99991400", "rate limit", "频率限制")
    _rate_limit_delay = 2.0

    def __init__(self, config: FeishuConfig):
        super().__init__(config)
        self._mention_names: list[str] = []  # bot mention keys from events
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._lark_ws_thread: threading.Thread | None = None
        self._ws_event_queue: queue.Queue | None = None
        self._ws_consumer_task: asyncio.Task | None = None

    # ── WebhookMixin overrides ────────────────────────────────────

    def _get_webhook_port(self) -> int:
        return self.config.webhook_port

    def _webhook_routes(self) -> list[tuple[str, Any]]:
        return [("POST", "/webhook/event", self._handle_event)]

    # ── TokenMixin overrides ──────────────────────────────────────

    async def _fetch_token(self) -> tuple[str, int]:
        """Fetch Feishu tenant_access_token."""
        url = f"{self.config.feishu_domain}/open-apis/auth/v3/tenant_access_token/internal"
        body = {
            "app_id": self.config.app_id,
            "app_secret": self.config.app_secret,
        }
        try:
            resp = await self._http_client.post(url, json=body)
            data = resp.json()
        except Exception as e:
            raise ChannelError(f"Failed to get Feishu access token: {e}") from e

        if data.get("code") != 0:
            raise ChannelError(f"Feishu auth error: {data.get('msg', 'unknown')}")
        return data["tenant_access_token"], data.get("expire", 7200)

    # ── Lifecycle ─────────────────────────────────────────────────

    _VALID_SUBSCRIPTION_MODES = ("webhook", "websocket")

    async def start(self) -> None:
        if not self.config.app_id:
            raise ChannelError("Feishu app_id is required")
        if not self.config.app_secret:
            raise ChannelError("Feishu app_secret is required")
        if self.config.subscription_mode not in self._VALID_SUBSCRIPTION_MODES:
            raise ChannelError(
                f"Invalid feishu_subscription_mode: {self.config.subscription_mode!r}. "
                f"Must be one of {self._VALID_SUBSCRIPTION_MODES}"
            )

        if self.config.subscription_mode == "websocket":
            await self._start_websocket_mode()
        else:
            await self._start_webhook_mode()

    async def _start_webhook_mode(self) -> None:
        try:
            import httpx  # noqa: F401
            from aiohttp import web  # noqa: F401
        except ImportError:
            raise ChannelError(
                "aiohttp or httpx not installed. "
                "Install with: pip install aiohttp httpx"
            ) from None

        # Start webhook server (sets up self._http_client)
        await self._start_webhook_server()

        # Verify credentials by fetching initial token
        await self._refresh_token()

        self._running = True
        logger.info(
            f"Feishu channel started (webhook on port {self.config.webhook_port})"
        )

    async def _start_websocket_mode(self) -> None:
        try:
            import lark_oapi as lark
        except ImportError:
            raise ChannelError(
                "lark-oapi not installed. Install with: pip install 'lark-oapi>=1.4.0'"
            ) from None

        import httpx

        proxy = getattr(self.config, "proxy", None) or None
        self._http_client = httpx.AsyncClient(timeout=15, proxy=proxy)

        # Verify credentials by fetching initial token
        await self._refresh_token()

        self._main_loop = asyncio.get_running_loop()

        # Thread-safe queue: SDK thread puts events, main loop consumes
        self._ws_event_queue = queue.Queue()

        # Set _running BEFORE creating consumer task — the task checks
        # `while self._running` and would exit immediately otherwise.
        self._running = True
        self._ws_consumer_task = asyncio.create_task(self._consume_ws_events())

        # Build SDK event handler
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_lark_sdk_message)
            .build()
        )

        ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.WARNING,
        )

        def _run_ws():
            # Workaround: nest_asyncio (used by the main process) patches
            # the event loop policy globally but does NOT patch Handle._run
            # for context re-entry (versions < 1.7).  The SDK creates its
            # own event loop in this thread; its transport callbacks hit
            #   RuntimeError: cannot enter context: … is already entered
            # Fix: patch Handle._run to retry with a context copy.
            _orig_handle_run = asyncio.Handle._run

            def _safe_handle_run(self):
                try:
                    self._context.run(self._callback, *self._args)
                except RuntimeError as exc:
                    if "cannot enter context" not in str(exc):
                        raise
                    ctx = self._context.copy()
                    ctx.run(self._callback, *self._args)

            asyncio.Handle._run = _safe_handle_run
            try:
                ws_client.start()
            except Exception:
                logger.exception(
                    "Feishu WebSocket SDK thread exited unexpectedly. "
                    "The channel will no longer receive messages. "
                    "Check app_id/app_secret and connection limits."
                )
            finally:
                asyncio.Handle._run = _orig_handle_run

        self._lark_ws_thread = threading.Thread(target=_run_ws, daemon=True)
        self._lark_ws_thread.start()

        logger.info("Feishu channel started (WebSocket long connection mode)")

    def _on_lark_sdk_message(self, data) -> None:
        """Sync callback invoked in lark-oapi SDK thread.

        Converts the SDK event object to a dict and puts it on a
        thread-safe queue.  The ``_consume_ws_events`` task on the main
        asyncio loop picks it up — no asyncio cross-thread calls needed,
        avoiding the nest_asyncio + Python 3.11 contextvars conflict.
        """
        try:
            event = data.event
            msg = event.message
            sender = event.sender

            # Rebuild mentions list from SDK objects
            mentions_list = []
            if msg.mentions:
                for m in msg.mentions:
                    mention_dict: dict[str, Any] = {"key": m.key, "id": {}}
                    if m.id:
                        mention_dict["id"] = {
                            "open_id": getattr(m.id, "open_id", ""),
                            "user_id": getattr(m.id, "user_id", ""),
                        }
                    mentions_list.append(mention_dict)

            event_dict = {
                "sender": {
                    "sender_id": {
                        "open_id": sender.sender_id.open_id if sender.sender_id else "",
                        "user_id": getattr(sender.sender_id, "user_id", "")
                        if sender.sender_id
                        else "",
                    },
                    "sender_type": sender.sender_type or "",
                },
                "message": {
                    "chat_id": msg.chat_id or "",
                    "message_type": msg.message_type or "",
                    "message_id": msg.message_id or "",
                    "chat_type": msg.chat_type or "",
                    "content": msg.content or "{}",
                    "create_time": msg.create_time or "",
                    "mentions": mentions_list,
                },
            }

            self._ws_event_queue.put(event_dict)
        except Exception:
            logger.exception("Feishu SDK message handler error")

    async def _consume_ws_events(self) -> None:
        """Main-loop task that drains the thread-safe event queue."""
        while self._running:
            try:
                event_dict = self._ws_event_queue.get_nowait()
                try:
                    await self._on_message(event_dict)
                except Exception:
                    logger.exception("Feishu WS event processing error")
            except queue.Empty:
                await asyncio.sleep(0.05)

    async def _cleanup(self) -> None:
        if self.config.subscription_mode == "websocket":
            if self._ws_consumer_task:
                self._ws_consumer_task.cancel()
                try:
                    await self._ws_consumer_task
                except asyncio.CancelledError:
                    pass
                self._ws_consumer_task = None
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            # Daemon thread exits with the process; no explicit stop needed
            self._lark_ws_thread = None
            self._main_loop = None
            self._ws_event_queue = None
        else:
            await self._stop_webhook_server()
        self._access_token = None
        logger.info("Feishu channel stopped")

    # ── Token helpers (adapt old API to mixin) ────────────────────

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        return await TokenMixin._ensure_token(self)

    # ── Send (template method overrides) ──────────────────────────

    async def _feishu_send(self, url: str, body: dict, headers: dict) -> bool:
        """POST to Feishu API and return True if code==0."""
        try:
            resp = await self._http_client.post(url, json=body, headers=headers)
            return resp.json().get("code") == 0
        except Exception as e:
            logger.warning(f"Feishu send error: {e}")
            return False

    async def _send_chunk(
        self,
        chat_id,
        formatted_text,
        raw_text,
        reply_to,
        metadata,
    ):
        token = await self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        post_content = _markdown_to_feishu_post(raw_text)

        # If reply_to is set, try the reply API first
        if reply_to:
            reply_url = (
                f"{self.config.feishu_domain}/open-apis/im/v1/messages/{reply_to}/reply"
            )
            if post_content is not None:
                body = {"msg_type": "post", "content": json.dumps(post_content)}
            else:
                body = {
                    "msg_type": "text",
                    "content": json.dumps({"text": formatted_text}),
                }
            if await self._feishu_send(reply_url, body, headers):
                return

        # Normal send (non-reply or reply fallback)
        url = (
            f"{self.config.feishu_domain}"
            f"/open-apis/im/v1/messages?receive_id_type=chat_id"
        )

        # Try post format first
        if post_content is not None:
            body = {
                "receive_id": chat_id,
                "msg_type": "post",
                "content": json.dumps(post_content),
            }
            if await self._feishu_send(url, body, headers):
                return

        # Fallback: plain text
        body = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": formatted_text}),
        }
        if not await self._feishu_send(url, body, headers):
            raise RuntimeError("Feishu send failed")

    # ── Media helpers ──────────────────────────────────────────────

    _IMAGE_EXTENSIONS: ClassVar[set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
    }

    async def _download_media(
        self,
        message_id: str,
        file_key: str,
        msg_type: str,
    ) -> str | None:
        """Download an image or file attachment from Feishu.

        Returns the local file path on success, or None on failure.
        """
        token = await self._ensure_token()
        resource_type = "image" if msg_type == "image" else "file"
        url = (
            f"{self.config.feishu_domain}"
            f"/open-apis/im/v1/messages/{message_id}"
            f"/resources/{file_key}?type={resource_type}"
        )
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = await self._http_client.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Feishu media download failed: HTTP {resp.status_code}")
                return None

            # Check attachment size before writing to disk
            cl = resp.headers.get("content-length")
            if cl:
                try:
                    too_large = self._check_attachment_size(int(cl), file_key)
                    if too_large:
                        logger.warning(too_large)
                        return None
                except (ValueError, TypeError):
                    pass
            from ..base import MAX_ATTACHMENT_BYTES

            if len(resp.content) > MAX_ATTACHMENT_BYTES:
                logger.warning(f"Feishu media too large: {len(resp.content)} bytes")
                return None

            # Determine extension from Content-Type or default
            content_type = resp.headers.get("content-type", "")
            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/bmp": ".bmp",
            }
            ext = ext_map.get(content_type, ".bin")
            local_path = self._media_path(f"feishu_{message_id}_{file_key}{ext}")
            local_path.write_bytes(resp.content)
            return str(local_path)
        except Exception as e:
            logger.warning(f"Failed to download Feishu media: {e}")
            return None

    async def _upload_feishu_resource(
        self,
        url: str,
        headers: dict,
        file_path: str,
        field_name: str,
        extra_data: dict,
    ) -> dict | None:
        """Upload a file to Feishu API. Returns response data or None on failure."""
        with open(file_path, "rb") as f:
            resp = await self._http_client.post(
                url,
                headers=headers,
                data=extra_data,
                files={field_name: (Path(file_path).name, f)},
            )
        data = resp.json()
        if data.get("code") != 0:
            logger.error(f"Feishu upload failed: {data.get('msg')}")
            return None
        return data["data"]

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file through Feishu."""
        token = await self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        chat_id = self._resolve_media_chat_id(recipient, metadata)

        path = Path(file_path)
        ext = path.suffix.lower()
        is_image = ext in self._IMAGE_EXTENSIONS

        send_url = (
            f"{self.config.feishu_domain}"
            f"/open-apis/im/v1/messages?receive_id_type=chat_id"
        )

        if is_image:
            upload_url = f"{self.config.feishu_domain}/open-apis/im/v1/images"
            data = await self._upload_feishu_resource(
                upload_url,
                headers,
                file_path,
                "image",
                {"image_type": "message"},
            )
            if not data:
                return False
            body = {
                "receive_id": chat_id,
                "msg_type": "image",
                "content": json.dumps({"image_key": data["image_key"]}),
            }
        else:
            upload_url = f"{self.config.feishu_domain}/open-apis/im/v1/files"
            data = await self._upload_feishu_resource(
                upload_url,
                headers,
                file_path,
                "file",
                {"file_type": "stream", "file_name": path.name},
            )
            if not data:
                return False
            body = {
                "receive_id": chat_id,
                "msg_type": "file",
                "content": json.dumps({"file_key": data["file_key"]}),
            }

        if not await self._feishu_send(send_url, body, headers):
            return False

        # Send caption as a separate text message if provided
        if caption:
            cap_body = {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": caption}),
            }
            await self._feishu_send(send_url, cap_body, headers)

        return True

    # ── ACK reaction ───────────────────────────────────────────────

    async def _send_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "THUMBSUP"
    ) -> None:
        """Send an acknowledgment reaction via Feishu Open API."""
        try:
            token = await self._ensure_token()
            url = f"{self.config.feishu_domain}/open-apis/im/v1/messages/{message_id}/reactions"
            await self._http_client.post(
                url,
                json={"reaction_type": {"emoji_type": emoji}},
                headers={"Authorization": f"Bearer {token}"},
            )
        except Exception as e:
            logger.debug(f"Feishu ack reaction failed: {e}")

    async def _remove_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "THUMBSUP"
    ) -> None:
        """Remove ACK reaction via Feishu Open API.

        Feishu's DELETE /reactions endpoint requires the reaction_id, which
        we don't track. No-op for now.
        """
        pass

    # ── Mention stripping ─────────────────────────────────────────

    def _strip_mention(self, text: str) -> str:
        """Strip bot @mention placeholders from Feishu text.

        In Feishu v2 events the text contains placeholders like ``@_user_1``
        for each mention. ``_mention_names`` caches the placeholder keys that
        belong to the bot (identified during ``_on_message``).
        """
        result = text
        for key in self._mention_names:
            result = result.replace(key, "")
        # Clean up extra whitespace left behind
        return re.sub(r"  +", " ", result).strip()

    # ── Event decryption ─────────────────────────────────────────

    def _decrypt_event(self, encrypted: str) -> dict:
        """Decrypt a Feishu encrypted event payload (AES-256-CBC).

        Feishu encryption spec:
          key   = SHA256(encrypt_key)
          data  = base64_decode(encrypted)
          iv    = data[:16]
          plain = AES_CBC_decrypt(data[16:], key, iv)  # PKCS7 padded
        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        key = hashlib.sha256(self.config.encrypt_key.encode()).digest()
        data = base64.b64decode(encrypted)
        iv, ciphertext = data[:16], data[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        # Remove PKCS7 padding
        pad_len = padded[-1]
        plaintext = padded[:-pad_len].decode()
        return json.loads(plaintext)

    # ── Webhook event handler ─────────────────────────────────────

    async def _handle_event(self, request) -> web.Response:
        """Handle POST /webhook/event from Feishu."""
        from aiohttp import web

        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400)

        # ── Decrypt if encrypt_key is configured ──
        if self.config.encrypt_key and "encrypt" in body:
            try:
                body = self._decrypt_event(body["encrypt"])
            except Exception:
                logger.exception("Feishu event decryption failed")
                return web.Response(status=400)

        # ── URL verification challenge ──
        if body.get("type") == "url_verification":
            challenge = body.get("challenge", "")
            return web.json_response({"challenge": challenge})

        # ── v2 event schema ──
        schema = body.get("schema")
        if schema == "2.0":
            header = body.get("header", {})

            # Verify token if configured
            if self.config.verification_token:
                token = header.get("token", "")
                if token != self.config.verification_token:
                    logger.warning("Feishu event token mismatch")
                    return web.Response(status=403)

            event_type = header.get("event_type", "")
            logger.info(f"Feishu v2 event received: {event_type}")
            if event_type == "im.message.receive_v1":
                try:
                    await self._on_message(body.get("event", {}))
                except Exception:
                    logger.exception("Feishu _on_message failed")

        # ── v1 event schema (legacy) ──
        elif "event" in body:
            if self.config.verification_token:
                token = body.get("token", "")
                if token != self.config.verification_token:
                    logger.warning("Feishu event token mismatch (v1)")
                    return web.Response(status=403)

            event = body["event"]
            msg_type = event.get("type", "")
            logger.info(f"Feishu v1 event received: type={msg_type}")
            if msg_type == "message":
                try:
                    await self._on_message_v1(event)
                except Exception:
                    logger.exception("Feishu _on_message_v1 failed")
        else:
            logger.info(f"Feishu event ignored: schema={schema}")

        return web.Response(status=200)

    async def _on_message(self, event: dict) -> None:
        """Handle im.message.receive_v1 event (v2 schema)."""
        sender_info = event.get("sender", {})
        sender_id_info = sender_info.get("sender_id", {})
        sender_id = sender_id_info.get("open_id") or sender_id_info.get("user_id") or ""
        sender_type = sender_info.get("sender_type", "")

        # Skip bot's own messages
        if sender_type == "app":
            return

        message = event.get("message", {})
        chat_id = message.get("chat_id", "")
        msg_type = message.get("message_type", "")
        message_id = message.get("message_id", "")

        # In group chats, detect mention status for centralized gating
        chat_type = message.get("chat_type", "")
        is_group = chat_type == "group"
        was_mentioned = True
        if is_group:
            mentions = message.get("mentions", [])
            was_mentioned = bool(mentions)
            # Cache bot mention keys — bot mentions have empty user IDs
            bot_keys = []
            for m in mentions:
                m_id = m.get("id", {})
                # Bot/app mentions have no open_id / user_id
                if not m_id.get("open_id") and not m_id.get("user_id"):
                    key = m.get("key", "")
                    if key:
                        bot_keys.append(key)
            if bot_keys:
                self._mention_names = bot_keys

        # Parse content JSON
        content_str = message.get("content", "{}")
        try:
            content_data = json.loads(content_str)
        except json.JSONDecodeError:
            content_data = {}

        text = ""
        annotations: list[str] = []
        media_paths: list[str] = []

        if msg_type == "text":
            text = content_data.get("text", "")
        elif msg_type == "post":
            text = self._extract_post_text(content_data)
        elif msg_type == "image" and self.config.include_attachments:
            image_key = content_data.get("image_key", "")
            if image_key:
                local = await self._download_media(message_id, image_key, "image")
                if local:
                    media_paths.append(local)
                    annotations.append(f"[attachment: {local}]")
                else:
                    annotations.append("[image message - download failed]")
            else:
                annotations.append("[image message]")
        elif msg_type == "file" and self.config.include_attachments:
            file_key = content_data.get("file_key", "")
            file_name = content_data.get("file_name", "unknown")
            if file_key:
                local = await self._download_media(message_id, file_key, "file")
                if local:
                    media_paths.append(local)
                    annotations.append(f"[attachment: {local}]")
                else:
                    annotations.append(f"[file: {file_name} - download failed]")
            else:
                annotations.append(f"[file message: {file_name}]")
        elif msg_type in ("audio", "media") and self.config.include_attachments:
            # Feishu audio messages are voice recordings
            media_label = "voice" if msg_type == "audio" else msg_type
            file_key = content_data.get("file_key", "")
            if file_key:
                local = await self._download_media(message_id, file_key, "file")
                if local:
                    media_paths.append(local)
                    annotations.append(f"[{media_label}: {local}]")
                else:
                    annotations.append(f"[{media_label} message - download failed]")
            else:
                annotations.append(f"[{media_label} message]")
        elif msg_type == "sticker":
            sticker_key = content_data.get("file_key", "")
            if sticker_key and self.config.include_attachments:
                local = await self._download_media(message_id, sticker_key, "image")
                if local:
                    media_paths.append(local)
                    annotations.append(f"[sticker: {local}]")
                else:
                    annotations.append("[sticker message]")
            else:
                annotations.append("[sticker message]")
        else:
            text = f"[{msg_type} message]"

        if not text and not media_paths and not annotations:
            return

        # Parse timestamp (milliseconds)
        create_time = message.get("create_time", "")
        try:
            timestamp = (
                datetime.fromtimestamp(int(create_time) / 1000)
                if create_time
                else datetime.now()
            )
        except (ValueError, TypeError, OSError):
            timestamp = datetime.now()

        await self._enqueue_raw(
            RawIncoming(
                sender_id=sender_id,
                chat_id=chat_id,
                text=text,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=timestamp,
                message_id=message_id,
                metadata={
                    "chat_id": chat_id,
                    "chat_type": message.get("chat_type", ""),
                },
                is_group=is_group,
                was_mentioned=was_mentioned,
            )
        )

    async def _on_message_v1(self, event: dict) -> None:
        """Handle v1 schema message event (legacy)."""
        sender_id = event.get("open_id", "")
        if not sender_id:
            return

        # Detect group and mention status for centralized gating
        chat_type = event.get("chat_type", "")
        is_group = chat_type == "group"
        was_mentioned = True
        if is_group:
            text_without_at = event.get("text_without_at_bot", "")
            was_mentioned = bool(text_without_at)

        text = event.get("text_without_at_bot", "") or event.get("text", "")
        if not text:
            return

        chat_id = event.get("open_chat_id", "")
        message_id = event.get("open_message_id", "")

        await self._enqueue_raw(
            RawIncoming(
                sender_id=sender_id,
                chat_id=chat_id,
                text=text,
                timestamp=datetime.now(),
                message_id=message_id,
                metadata={
                    "chat_id": chat_id,
                    "chat_type": event.get("chat_type", ""),
                },
                is_group=is_group,
                was_mentioned=was_mentioned,
            )
        )

    @staticmethod
    def _extract_post_text(content: dict) -> str:
        """Extract plain text from Feishu post (rich text) content."""
        parts: list[str] = []
        # Post content has locale keys like "zh_cn", "en_us"
        for locale_key in ("zh_cn", "en_us", "ja_jp"):
            locale_content = content.get(locale_key)
            if locale_content:
                title = locale_content.get("title", "")
                if title:
                    parts.append(title)
                for paragraph in locale_content.get("content", []):
                    line_parts: list[str] = []
                    for element in paragraph:
                        tag = element.get("tag", "")
                        if tag == "text":
                            line_parts.append(element.get("text", ""))
                        elif tag == "a":
                            line_parts.append(element.get("text", ""))
                        elif tag == "at":
                            # Skip @mentions of the bot
                            pass
                    line = "".join(line_parts).strip()
                    if line:
                        parts.append(line)
                break  # Use first available locale
        return "\n".join(parts)
