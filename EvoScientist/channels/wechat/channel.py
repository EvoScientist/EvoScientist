"""WeChat channel implementation.

Supports two backends via a unified Channel interface:

1. **wecom** (企业微信应用): Corporate WeChat official API
   - Receives messages via HTTP callback (XML + optional AES encryption)
   - Sends replies via REST API (POST /cgi-bin/message/send)
   - Supports text, image, file, markdown messages
   - Token auto-refresh with 2-hour TTL

2. **wechatmp** (微信公众号): WeChat Official Account API
   - Receives messages via HTTP callback (XML + optional AES encryption)
   - Sends replies via REST API (POST /cgi-bin/message/custom/send)
   - Supports text, image, news messages

Both backends use httpx (already a core dependency) and aiohttp for
webhook server — matching the Feishu channel pattern.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

from ..base import Channel, ChannelError, RawIncoming
from ..capabilities import WECHAT as WECHAT_CAPS
from ..config import BaseChannelConfig
from ..mixins import TokenMixin, WebhookMixin

logger = logging.getLogger(__name__)


# ── Markdown → plain text (fallback for WeChat text messages) ────


def _strip_markdown(text: str) -> str:
    """Strip Markdown formatting for plain-text WeChat messages."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`").strip(), text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Remove italic
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"\1", text)
    # Remove strikethrough
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Convert links
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1(\2)", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Convert list items
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)
    return text


# ── Config dataclasses ───────────────────────────────────────────


@dataclass
class WeComConfig(BaseChannelConfig):
    """Configuration for WeCom (企业微信) backend."""

    corp_id: str = ""
    agent_id: str = ""
    secret: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    webhook_port: int = 9001


@dataclass
class WeChatMPConfig(BaseChannelConfig):
    """Configuration for WeChat Official Account (公众号) backend."""

    app_id: str = ""
    app_secret: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    webhook_port: int = 9001


# ── Unified WeChat Channel ───────────────────────────────────────


class WeChatChannel(Channel, WebhookMixin, TokenMixin):
    capabilities = WECHAT_CAPS
    """Unified WeChat channel supporting WeCom and Official Account backends.

    Architecture follows the same pattern as FeishuChannel:
    - HTTP webhook server (aiohttp) for inbound messages
    - REST API calls (httpx) for outbound messages
    - Token auto-refresh
    """

    name = "wechat"
    _typing_interval: float = 5.0  # WeChat has no typing API, but keep for interface
    _ready_attrs = ("_http_client", "_access_token")
    _rate_limit_patterns = ("45009", "frequency", "freq")
    _rate_limit_delay = 2.0
    _mention_pattern = r"@\S+\s*"
    _mention_strip_count = 1

    def __init__(
        self,
        config: WeComConfig | WeChatMPConfig,
        backend: str = "wecom",
    ):
        super().__init__(config)
        self._backend = backend
        self._access_token: str | None = None
        self._token_expires: float = 0
        self._runner = None
        self._site = None
        self._http_client = None
        self._crypto = None  # WeChatCrypto instance (optional)
        self._typing_message_ids: dict[
            str, list[str]
        ] = {}  # chat_id → [msgid, ...] for typing recall
        self._background_tasks: set[asyncio.Task] = set()

    # ── Lifecycle ─────────────────────────────────────────────────

    def _webhook_routes(self) -> list[tuple[str, str, Any]]:
        """Return HTTP routes for the shared webhook server."""
        return [
            ("GET", "/wechat/callback", self._handle_verify),
            ("POST", "/wechat/callback", self._handle_message),
        ]

    async def start(self) -> None:
        try:
            import httpx
            from aiohttp import web
        except ImportError:
            raise ChannelError(
                "aiohttp or httpx not installed. "
                "Install with: pip install aiohttp httpx"
            ) from None

        self._validate_config()

        import httpx

        self._http_client = httpx.AsyncClient(
            timeout=15,
            proxy=self._get_proxy(),
        )

        # Set up message encryption if configured
        if self.config.encoding_aes_key and self.config.token:
            from .crypto import WeChatCrypto

            app_id = self._get_app_id()
            self._crypto = WeChatCrypto(
                token=self.config.token,
                encoding_aes_key=self.config.encoding_aes_key,
                app_id=app_id,
            )

        # Verify credentials by fetching initial token
        await self._refresh_token()

        if not getattr(self, "_shared_webhook_server", None):
            app = web.Application()
            app.router.add_get("/wechat/callback", self._handle_verify)
            app.router.add_post("/wechat/callback", self._handle_message)

            self._runner = web.AppRunner(app)
            await self._runner.setup()
            self._site = web.TCPSite(
                self._runner,
                "0.0.0.0",
                self.config.webhook_port,
            )
            await self._site.start()

        self._running = True
        logger.info(
            f"WeChat channel started "
            f"(backend={self._backend}, "
            f"webhook on port {self.config.webhook_port})"
        )

    async def _cleanup(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._access_token = None
        logger.info("WeChat channel stopped")

    def _validate_config(self) -> None:
        """Validate required config fields based on backend."""
        if self._backend == "wecom":
            cfg = self.config
            if not cfg.corp_id:
                raise ChannelError("WeCom corp_id is required")
            if not cfg.secret:
                raise ChannelError("WeCom secret is required")
            if not cfg.agent_id:
                raise ChannelError("WeCom agent_id is required")
        elif self._backend == "wechatmp":
            cfg = self.config
            if not cfg.app_id:
                raise ChannelError("WeChat MP app_id is required")
            if not cfg.app_secret:
                raise ChannelError("WeChat MP app_secret is required")

    def _get_app_id(self) -> str:
        """Return the app identifier for crypto operations."""
        if self._backend == "wecom":
            return self.config.corp_id
        return self.config.app_id

    # ── Token management ──────────────────────────────────────────

    async def _refresh_token(self) -> None:
        """Fetch or refresh the access_token."""
        if self._backend == "wecom":
            url = (
                f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
                f"?corpid={self.config.corp_id}"
                f"&corpsecret={self.config.secret}"
            )
        else:
            url = (
                f"https://api.weixin.qq.com/cgi-bin/token"
                f"?grant_type=client_credential"
                f"&appid={self.config.app_id}"
                f"&secret={self.config.app_secret}"
            )

        try:
            resp = await self._http_client.get(url)
            data = resp.json()
        except Exception as e:
            if not self._running:
                raise ChannelError(f"Failed to get WeChat access token: {e}") from e
            raise RuntimeError(f"Failed to get WeChat access token: {e}") from e

        if data.get("errcode", 0) != 0:
            err_msg = (
                f"WeChat auth error ({data.get('errcode')}): "
                f"{data.get('errmsg', 'unknown')}"
            )
            if not self._running:
                raise ChannelError(err_msg)
            raise RuntimeError(err_msg)

        self._access_token = data["access_token"]
        expire = data.get("expires_in", 7200)
        # Refresh 5 minutes before expiry
        self._token_expires = time.monotonic() + expire - 300
        logger.debug(f"WeChat token refreshed, expires in {expire}s")

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if not self._access_token or time.monotonic() >= self._token_expires:
            await self._refresh_token()
        return self._access_token

    # ── Signature verification (GET callback) ─────────────────────

    async def _handle_verify(self, request) -> web.Response:
        """Handle GET /wechat/callback for URL verification.

        WeChat/WeCom sends: msg_signature, timestamp, nonce, echostr
        We decrypt echostr (encrypted mode) or verify signature (plain mode)
        and return the plain echostr.
        """
        from aiohttp import web

        signature = request.query.get("msg_signature") or request.query.get(
            "signature", ""
        )
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")
        echostr = request.query.get("echostr", "")

        logger.info(f"Verify request received: timestamp={timestamp}")

        if not echostr:
            return web.Response(status=400, text="missing echostr")

        # Encrypted mode: WeCom sends msg_signature and encrypted echostr
        if self._crypto and request.query.get("msg_signature"):
            # Verify signature first
            sig_ok = self._crypto.verify_signature(signature, timestamp, nonce, echostr)
            if not sig_ok:
                logger.warning("WeChat verify: signature mismatch")
            # Try to decrypt regardless — the decrypted echostr must be returned
            try:
                plain_echostr, _ = self._crypto.decrypt(echostr)
                logger.info("WeChat verify: echostr decrypted successfully")
                return web.Response(text=plain_echostr)
            except Exception as e:
                logger.error(f"WeChat verify: echostr decrypt failed: {e}")
                return web.Response(status=500)
        else:
            # Plain mode verification
            token = self.config.token
            if token:
                parts = sorted([token, timestamp, nonce])
                expected = hashlib.sha1("".join(parts).encode()).hexdigest()
                if expected != signature:
                    logger.warning("WeChat verify: signature mismatch (plain)")
                    return web.Response(status=403)
            return web.Response(text=echostr)

    # ── Inbound message handling (POST callback) ──────────────────

    async def _handle_message(self, request) -> web.Response:
        """Handle POST /wechat/callback for incoming messages."""
        from aiohttp import web

        from .crypto import parse_xml

        try:
            body = await request.text()
        except Exception:
            return web.Response(status=400)

        logger.info(f"WeChat callback POST received, body length={len(body)}")
        xml_data = parse_xml(body)

        # If encrypted, decrypt first
        encrypt = xml_data.get("Encrypt", "")
        if encrypt and self._crypto:
            signature = request.query.get("msg_signature", "")
            timestamp = request.query.get("timestamp", "")
            nonce = request.query.get("nonce", "")

            if not self._crypto.verify_signature(signature, timestamp, nonce, encrypt):
                logger.warning("WeChat message signature mismatch")
                return web.Response(status=403)

            try:
                decrypted_xml, _from_id = self._crypto.decrypt(encrypt)
                xml_data = parse_xml(decrypted_xml)
            except Exception as e:
                logger.error(f"WeChat decrypt failed: {e}")
                return web.Response(status=500)

        # Process message asynchronously — WeCom requires a response within
        # 5 seconds, but media downloads can take much longer.  Return
        # "success" immediately and handle the message in the background.
        _task = asyncio.create_task(self._safe_process_message(xml_data))
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        return web.Response(text="success")

    async def _safe_process_message(self, xml_data: dict[str, str]) -> None:
        """Wrapper that catches exceptions so fire-and-forget tasks don't leak."""
        try:
            await self._process_message(xml_data)
        except Exception:
            logger.exception("Error processing WeChat message")

    async def _process_message(self, xml_data: dict[str, str]) -> None:
        """Process a parsed XML message from WeChat/WeCom callback."""
        msg_type = xml_data.get("MsgType", "")
        from_user = xml_data.get("FromUserName", "")
        to_user = xml_data.get("ToUserName", "")
        content = xml_data.get("Content", "")
        msg_id = xml_data.get("MsgId", "")
        create_time = xml_data.get("CreateTime", "")

        logger.info(
            f"WeChat message received: type={msg_type}, from={from_user}, id={msg_id}, keys={list(xml_data.keys())}"
        )

        if not from_user:
            return

        # Determine chat_id
        # For WeCom: FromUserName is the user's UserID
        # For MP: FromUserName is the user's OpenID
        chat_id = from_user

        # Group chat detection
        is_group = False
        was_mentioned = True  # Default: treat as mentioned (DMs)

        # WeCom group detection: ChatId field indicates a group message
        if self._backend == "wecom":
            group_chat_id = xml_data.get("ChatId", "")
            if group_chat_id:
                is_group = True
                chat_id = group_chat_id
                # WeCom sets MsgType=event with Event=sys when bot is @mentioned,
                # but for text messages we check the XML AtUserList field
                at_user_list = xml_data.get("AtUserList", "")
                was_mentioned = bool(at_user_list)

        # Handle different message types
        text = ""
        annotations: list[str] = []
        media_paths: list[str] = []

        if msg_type == "text":
            text = content
        elif msg_type == "image":
            pic_url = xml_data.get("PicUrl", "")
            media_id = xml_data.get("MediaId", "")
            if pic_url:
                local, ann = await self._download_attachment(
                    pic_url,
                    f"wechat_{msg_id}.jpg",
                )
                if local:
                    media_paths.append(local)
                if ann:
                    annotations.append(ann)
            elif media_id:
                local, ann = await self._download_wechat_media(
                    media_id,
                    f"wechat_image_{msg_id}",
                )
                if local:
                    media_paths.append(local)
                if ann:
                    annotations.append(ann)
            else:
                annotations.append("[image: no download source]")
        elif msg_type == "voice":
            recognition = xml_data.get("Recognition", "")
            media_id = xml_data.get("MediaId", "")
            if media_id:
                local, ann = await self._download_wechat_media(
                    media_id, f"wechat_voice_{msg_id}"
                )
                if local:
                    media_paths.append(local)
                if ann:
                    ann = ann.replace("[attachment:", "[voice:")
                    annotations.append(ann)
            if recognition:
                text = f"[语音识别] {recognition}"
            elif not media_paths:
                annotations.append("[voice message]")
        elif msg_type in ("video", "shortvideo"):
            media_id = xml_data.get("MediaId", "")
            if media_id:
                local, ann = await self._download_wechat_media(
                    media_id, f"wechat_{msg_type}_{msg_id}"
                )
                if local:
                    media_paths.append(local)
                if ann:
                    annotations.append(ann)
            if not media_paths:
                annotations.append(f"[{msg_type} message]")
        elif msg_type == "location":
            label = xml_data.get("Label", "")
            lat = xml_data.get("Location_X", "")
            lon = xml_data.get("Location_Y", "")
            text = f"[位置] {label} ({lat}, {lon})"
        elif msg_type == "file":
            media_id = xml_data.get("MediaId", "")
            file_name = xml_data.get("FileName", "") or xml_data.get(
                "Title", f"wechat_file_{msg_id}"
            )
            logger.info(
                f"WeChat file message: name={file_name}, media_id={media_id!r}, keys={list(xml_data.keys())}"
            )
            if media_id:
                local, ann = await self._download_wechat_media(
                    media_id,
                    f"wechat_file_{msg_id}_{file_name}",
                )
                logger.info(f"WeChat file download result: local={local}, ann={ann}")
                if local:
                    media_paths.append(local)
                if ann:
                    annotations.append(ann)
            if not media_paths:
                annotations.append(f"[file: {file_name}]")
        elif msg_type == "link":
            title = xml_data.get("Title", "")
            description = xml_data.get("Description", "")
            url = xml_data.get("Url", "")
            text = f"[链接] {title}\n{description}\n{url}"
        elif msg_type == "event":
            event_type = xml_data.get("Event", "")
            if event_type == "subscribe":
                text = "[用户关注]"
            elif event_type == "unsubscribe":
                logger.info(f"User {from_user} unsubscribed")
                return  # Don't process
            elif event_type == "CLICK":
                event_key = xml_data.get("EventKey", "")
                text = f"[菜单点击] {event_key}"
            elif event_type in ("LOCATION", "VIEW"):
                # Periodic location reports and menu-link clicks — ignore
                return
            else:
                logger.debug(f"Ignoring WeChat event: {event_type}")
                return
        else:
            text = f"[{msg_type} message]"

        if not text and not media_paths and not annotations:
            return

        # Parse timestamp
        try:
            timestamp = (
                datetime.fromtimestamp(int(create_time))
                if create_time
                else datetime.now()
            )
        except (ValueError, TypeError, OSError):
            timestamp = datetime.now()

        await self._enqueue_raw(
            RawIncoming(
                sender_id=from_user,
                chat_id=chat_id,
                text=text,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=timestamp,
                message_id=msg_id,
                is_group=is_group,
                was_mentioned=was_mentioned,
                metadata={
                    "chat_id": chat_id,
                    "to_user": to_user,
                    "backend": self._backend,
                },
            )
        )

    # ── Send (template method overrides) ──────────────────────────

    def _format_chunk(self, text: str) -> str:
        """WeCom uses markdown formatter; MP uses plain text."""
        if self._backend == "wecom":
            return self._formatter.format(text)  # markdown profile
        return _strip_markdown(text)

    async def _send_chunk(
        self,
        chat_id,
        formatted_text,
        raw_text,
        reply_to,
        metadata,
    ):
        token = await self._ensure_token()

        if self._backend == "wecom":
            # Group chat: use appchat/send endpoint
            if chat_id.startswith("wr"):
                try:
                    await self._wecom_send_group_markdown(token, chat_id, raw_text)
                    return
                except Exception:
                    pass
                await self._wecom_send_group_text(token, chat_id, raw_text)
            else:
                # DM: Try markdown first, fall back to plain text
                try:
                    await self._wecom_send_markdown(token, chat_id, raw_text)
                    return
                except Exception:
                    pass
                await self._wecom_send_text(token, chat_id, raw_text)
        else:
            await self._mp_send_text(token, chat_id, raw_text)

    # ── WeCom send ────────────────────────────────────────────────

    async def _wecom_send_text(
        self,
        token: str,
        user_id: str,
        text: str,
    ) -> None:
        """Send a text message via WeCom API."""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
        body = {
            "touser": user_id,
            "msgtype": "text",
            "agentid": int(self.config.agent_id),
            "text": {"content": _strip_markdown(text)},
        }
        await self._post_api(url, body)

    async def _wecom_send_markdown(
        self,
        token: str,
        user_id: str,
        text: str,
    ) -> None:
        """Send a markdown message via WeCom API.

        Note: WeCom markdown only supports a subset of Markdown
        (no code blocks, no images). Falls back to text if the
        message is too complex.
        """
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
        body = {
            "touser": user_id,
            "msgtype": "markdown",
            "agentid": int(self.config.agent_id),
            "markdown": {"content": text},
        }
        await self._post_api(url, body)

    # ── WeCom group send ────────────────────────────────────────────

    async def _wecom_send_group_text(
        self,
        token: str,
        chatid: str,
        text: str,
    ) -> None:
        """Send a text message to a WeCom group chat."""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/appchat/send?access_token={token}"
        body = {
            "chatid": chatid,
            "msgtype": "text",
            "text": {"content": _strip_markdown(text)},
        }
        await self._post_api(url, body)

    async def _wecom_send_group_markdown(
        self,
        token: str,
        chatid: str,
        text: str,
    ) -> None:
        """Send a markdown message to a WeCom group chat."""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/appchat/send?access_token={token}"
        body = {
            "chatid": chatid,
            "msgtype": "markdown",
            "markdown": {"content": text},
        }
        await self._post_api(url, body)

    # ── MP send ───────────────────────────────────────────────────

    async def _mp_send_text(
        self,
        token: str,
        openid: str,
        text: str,
    ) -> None:
        """Send a text message via WeChat MP customer service API."""
        url = (
            f"https://api.weixin.qq.com/cgi-bin/message/custom/send"
            f"?access_token={token}"
        )
        body = {
            "touser": openid,
            "msgtype": "text",
            "text": {"content": _strip_markdown(text)},
        }
        await self._post_api(url, body)

    # ── Media send ────────────────────────────────────────────────

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file via WeChat/WeCom."""
        token = await self._ensure_token()
        chat_id = self._resolve_media_chat_id(recipient, metadata)

        # Upload media to get media_id
        media_id = await self._upload_media(token, file_path)
        if not media_id:
            return False

        path = Path(file_path)
        ext = path.suffix.lower()
        is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

        if self._backend == "wecom":
            msg_type = "image" if is_image else "file"
            # Group chat: use appchat/send endpoint
            if chat_id.startswith("wr"):
                url = (
                    f"https://qyapi.weixin.qq.com/cgi-bin/appchat/send"
                    f"?access_token={token}"
                )
                body = {
                    "chatid": chat_id,
                    "msgtype": msg_type,
                    msg_type: {"media_id": media_id},
                }
            else:
                url = (
                    f"https://qyapi.weixin.qq.com/cgi-bin/message/send"
                    f"?access_token={token}"
                )
                body = {
                    "touser": chat_id,
                    "msgtype": msg_type,
                    "agentid": int(self.config.agent_id),
                    msg_type: {"media_id": media_id},
                }
        else:
            url = (
                f"https://api.weixin.qq.com/cgi-bin/message/custom/send"
                f"?access_token={token}"
            )
            msg_type = "image" if is_image else "file"  # MP only supports image
            if not is_image:
                # MP doesn't support file via customer service API;
                # send caption as text instead
                if caption:
                    await self._mp_send_text(
                        token, chat_id, f"[文件] {path.name}\n{caption}"
                    )
                return True
            body = {
                "touser": chat_id,
                "msgtype": "image",
                "image": {"media_id": media_id},
            }

        await self._post_api(url, body)

        # Send caption separately if provided
        if caption:
            if self._backend == "wecom":
                if chat_id.startswith("wr"):
                    await self._wecom_send_group_text(token, chat_id, caption)
                else:
                    await self._wecom_send_text(token, chat_id, caption)
            else:
                await self._mp_send_text(token, chat_id, caption)

        return True

    async def _upload_media(
        self,
        token: str,
        file_path: str,
    ) -> str | None:
        """Upload a media file and return the media_id."""
        path = Path(file_path)
        ext = path.suffix.lower()
        is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
        media_type = "image" if is_image else "file"

        if self._backend == "wecom":
            url = (
                f"https://qyapi.weixin.qq.com/cgi-bin/media/upload"
                f"?access_token={token}&type={media_type}"
            )
        else:
            url = (
                f"https://api.weixin.qq.com/cgi-bin/media/upload"
                f"?access_token={token}&type={media_type}"
            )

        try:
            with open(file_path, "rb") as f:
                resp = await self._http_client.post(
                    url,
                    files={"media": (path.name, f)},
                )
            data = resp.json()
            if data.get("errcode", 0) != 0 and "media_id" not in data:
                logger.error(f"WeChat media upload failed: {data.get('errmsg')}")
                return None
            return data.get("media_id")
        except Exception as e:
            logger.error(f"WeChat media upload error: {e}")
            return None

    # ── Media download helper ────────────────────────────────────

    async def _download_wechat_media(
        self,
        media_id: str,
        filename: str,
    ) -> tuple[str | None, str | None]:
        """Download media by media_id via WeChat/WeCom media API."""
        token = await self._ensure_token()
        if self._backend == "wecom":
            url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
        else:
            url = f"https://api.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
        return await self._download_attachment(url, filename)

    # ── Shared API helper ─────────────────────────────────────────

    async def _post_api(self, url: str, body: dict) -> dict:
        """POST to WeChat/WeCom API, check errcode, return response."""
        try:
            resp = await self._http_client.post(url, json=body)
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"WeChat API error: {e}") from e

        errcode = data.get("errcode", 0)
        if errcode != 0:
            errmsg = data.get("errmsg", "unknown")
            # Token expired — refresh and retry once
            if errcode in (40014, 42001):
                logger.warning("WeChat token expired, refreshing...")
                await self._refresh_token()
                token = self._access_token
                # Replace token in URL
                if "access_token=" in url:
                    url = re.sub(
                        r"access_token=[^&]+",
                        f"access_token={token}",
                        url,
                    )
                    resp = await self._http_client.post(url, json=body)
                    data = resp.json()
                    if data.get("errcode", 0) != 0:
                        raise RuntimeError(
                            f"WeChat API error after retry: {data.get('errmsg')}"
                        )
                    return data
            else:
                raise RuntimeError(f"WeChat API error ({errcode}): {errmsg}")

        return data

    # ── Typing indicator (WeCom only) ────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        """Send typing indicator via WeChat.

        WeChat has no native typing API.  For the WeCom backend we
        approximate the experience by posting a short-lived "…" message
        that is recalled once the real reply is sent (handled by
        ``stop_typing``).  WeChat MP has no recall API so we skip it.
        """
        if self._backend != "wecom" or not self._http_client:
            return
        try:
            token = await self._ensure_token()
            if chat_id.startswith("wr"):
                url = (
                    f"https://qyapi.weixin.qq.com/cgi-bin/appchat/send"
                    f"?access_token={token}"
                )
                body = {
                    "chatid": chat_id,
                    "msgtype": "text",
                    "text": {"content": "\u2026"},
                }
            else:
                url = (
                    f"https://qyapi.weixin.qq.com/cgi-bin/message/send"
                    f"?access_token={token}"
                )
                body = {
                    "touser": chat_id,
                    "msgtype": "text",
                    "agentid": int(self.config.agent_id),
                    "text": {"content": "\u2026"},
                }
            data = await self._post_api(url, body)
            msgid = data.get("msgid")
            if msgid:
                self._typing_message_ids.setdefault(chat_id, []).append(msgid)
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        """Cancel typing loop and recall all status messages."""
        msgids = self._typing_message_ids.pop(chat_id, [])
        if msgids and self._http_client and self._backend == "wecom":
            try:
                token = await self._ensure_token()
                url = (
                    f"https://qyapi.weixin.qq.com/cgi-bin/message/recall"
                    f"?access_token={token}"
                )
                for msgid in msgids:
                    try:
                        await self._http_client.post(url, json={"msgid": msgid})
                    except Exception:
                        pass
            except Exception:
                pass
        await super().stop_typing(chat_id)
