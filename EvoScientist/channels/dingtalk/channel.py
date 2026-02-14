"""DingTalk channel — refactored with WebSocketMixin + TokenMixin."""

import asyncio
import json
import logging
from urllib.parse import quote_plus
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..base import Channel, RawIncoming, ChannelError
from ..capabilities import DINGTALK as DINGTALK_CAPS
from ..mixins import WebSocketMixin, TokenMixin
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)

GATEWAY_URL = "https://api.dingtalk.com/v1.0/gateway/connections/open"
TOKEN_URL = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
SEND_URL = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"
MEDIA_SEND_URL = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"
MEDIA_UPLOAD_URL = "https://oapi.dingtalk.com/media/upload"


@dataclass
class DingTalkConfig(BaseChannelConfig):
    client_id: str = ""
    client_secret: str = ""
    text_chunk_limit: int = 4096


class DingTalkChannel(Channel, WebSocketMixin, TokenMixin):
    capabilities = DINGTALK_CAPS
    name = "dingtalk"
    _ready_attrs = ("_http_client", "_access_token")
    _non_retryable_patterns = ("invalidauthentication", "forbidden", "40014")
    _mention_pattern = r"@\S+\s*"
    _mention_strip_count = 1

    def __init__(self, config: DingTalkConfig):
        super().__init__(config)

    async def start(self) -> None:
        import httpx
        if not self.config.client_id or not self.config.client_secret:
            raise ChannelError("DingTalk client_id and client_secret are required")
        self._http_client = httpx.AsyncClient(timeout=15, proxy=self.config.proxy)
        await self._refresh_token()
        self._running = True
        logger.info("DingTalk channel starting (Stream Mode)...")
        self._ws_task = asyncio.create_task(self._ws_loop())

    # ── TokenMixin ────────────────────────────────────────────────

    async def _fetch_token(self) -> tuple[str, int]:
        data = await self._api_post(TOKEN_URL, {
            "appKey": self.config.client_id,
            "appSecret": self.config.client_secret,
        })
        token = data.get("accessToken")
        if not token:
            raise ChannelError(f"DingTalk auth error: {data}")
        return token, int(data.get("expireIn", 7200))

    async def _api_post(self, url, body, headers=None):
        resp = await self._http_client.post(url, json=body, headers=headers)
        return resp.json()

    # ── WebSocketMixin ────────────────────────────────────────────

    async def _get_ws_url(self) -> str:
        resp = await self._http_client.post(GATEWAY_URL, json={
            "clientId": self.config.client_id,
            "clientSecret": self.config.client_secret,
            "subscriptions": [{"type": "CALLBACK", "topic": "/v1.0/im/bot/messages/get"}],
            "ua": "dingtalk-sdk-python/v0.24.3-union",
        })
        data = resp.json()
        endpoint, ticket = data.get("endpoint"), data.get("ticket")
        if not endpoint or not ticket:
            raise ChannelError(f"DingTalk gateway failed: {data}")
        return f"{endpoint}?ticket={quote_plus(ticket)}"

    async def _on_ws_message(self, data) -> None:
        if not isinstance(data, dict):
            return
        headers = data.get("headers", {})
        msg_id = headers.get("messageId", "")

        # System ping
        if data.get("type") == "SYSTEM" and headers.get("topic") == "ping":
            await self._ws_send_json({"code": 200, "headers": headers, "message": "OK", "data": data.get("data", "")})
            return

        # ACK
        await self._ws_send_json({"code": 200, "headers": {"contentType": "application/json", "messageId": msg_id}, "message": "OK", "data": "{}"})

        if data.get("type") != "CALLBACK":
            return

        payload = data.get("data", "{}")
        payload = json.loads(payload) if isinstance(payload, str) else payload
        text_obj = payload.get("text", {})
        content = (text_obj.get("content", "") if isinstance(text_obj, dict) else str(text_obj)).strip()
        content = content or payload.get("content", "").strip()
        if not content:
            return

        # Download attachments if present
        annotations: list[str] = []
        media_paths: list[str] = []
        for att_key in ("imageContent", "fileContent", "videoContent", "audioContent"):
            att = payload.get(att_key)
            if att and isinstance(att, dict):
                file_size = att.get("fileSize") or att.get("downloadSize") or 0
                file_name = att.get("fileName", att_key)
                download_url = att.get("downloadCode") or att.get("downloadUrl") or ""
                # DingTalk audioContent is voice messages
                media_label = "voice" if att_key == "audioContent" else att_key
                if download_url and (self.config.include_attachments if hasattr(self.config, 'include_attachments') else True):
                    # DingTalk download URLs require access token
                    try:
                        dl_token = await self._ensure_token()
                        dl_headers = {"x-acs-dingtalk-access-token": dl_token}
                    except Exception:
                        dl_headers = None
                    local, ann = await self._download_attachment(
                        download_url, f"dingtalk_{file_name}",
                        headers=dl_headers,
                        file_size=int(file_size) if file_size else None,
                    )
                    if local:
                        media_paths.append(local)
                    if ann:
                        ann = ann.replace("[attachment:", f"[{media_label}:")
                        annotations.append(ann)
                elif file_size:
                    too_large = self._check_attachment_size(int(file_size), file_name)
                    if too_large:
                        annotations.append(too_large)
                    else:
                        annotations.append(f"[{media_label}: {file_name}]")

        sender_id = payload.get("senderStaffId") or payload.get("senderId", "")
        conv_id = payload.get("conversationId", "")
        is_group = payload.get("conversationType") == "2"
        # For send API (oToMessages/batchSend), userIds needs staffId, not conversationId
        chat_id = sender_id
        create_time = payload.get("createAt") or payload.get("createTime", "")

        # Mention gating: DMs always pass; groups require @bot
        was_mentioned = not is_group
        if is_group:
            # isInAtList is set by DingTalk when bot is @mentioned
            if payload.get("isInAtList"):
                was_mentioned = True
            else:
                # Fallback: check atUsers array
                at_users = payload.get("atUsers") or []
                for u in at_users:
                    if u.get("dingtalkId") == self.config.client_id:
                        was_mentioned = True
                        break

        try:
            ts = datetime.fromtimestamp(int(create_time) / 1000) if create_time else datetime.now()
        except (ValueError, TypeError, OSError):
            ts = datetime.now()

        await self._enqueue_raw(RawIncoming(
            sender_id=sender_id, chat_id=chat_id, text=content, timestamp=ts,
            message_id=msg_id, is_group=is_group, was_mentioned=was_mentioned,
            media_files=media_paths,
            content_annotations=annotations,
            metadata={"chat_id": chat_id, "sender_nick": payload.get("senderNick", ""), "backend": "dingtalk"},
        ))

    # _send_typing_action: inherited no-op (DingTalk has no typing API)
    # _format_chunk: inherited from base (UnifiedFormatter)

    # ── Send ──────────────────────────────────────────────────────

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        token = await self._ensure_token()
        data = await self._api_post(SEND_URL, {
            "robotCode": self.config.client_id,
            "userIds": [chat_id],
            "msgKey": "sampleMarkdown",
            "msgParam": json.dumps({"text": raw_text, "title": "EvoScientist"}),
        }, headers={"x-acs-dingtalk-access-token": token})
        return data

    # ── Media send ────────────────────────────────────────────────

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file through DingTalk.

        For images: uploads via /media/upload to get media_id, then sends
        as sampleImageMsg.  Non-image files are sent as markdown links
        (DingTalk robot API does not support arbitrary file uploads).
        """
        token = await self._ensure_token()
        chat_id = self._resolve_media_chat_id(recipient, metadata)
        headers = {"x-acs-dingtalk-access-token": token}
        ext = Path(file_path).suffix.lower()

        if ext in self._IMAGE_EXTS:
            # Try uploading image to get media_id for native image message
            media_id = await self._upload_dingtalk_media(token, file_path, "image")
            if media_id:
                await self._api_post(MEDIA_SEND_URL, {
                    "robotCode": self.config.client_id,
                    "userIds": [chat_id],
                    "msgKey": "sampleImageMsg",
                    "msgParam": json.dumps({"photoURL": media_id}),
                }, headers=headers)
            else:
                # Fallback to markdown with file path
                await self._api_post(MEDIA_SEND_URL, {
                    "robotCode": self.config.client_id,
                    "userIds": [chat_id],
                    "msgKey": "sampleMarkdown",
                    "msgParam": json.dumps({
                        "text": f"![image]({file_path})" + (f"\n{caption}" if caption else ""),
                        "title": caption or "Image",
                    }),
                }, headers=headers)
        else:
            # Non-image: send as markdown with filename
            name = Path(file_path).name
            text = f"[文件] {name}" + (f"\n{caption}" if caption else "")
            await self._api_post(MEDIA_SEND_URL, {
                "robotCode": self.config.client_id,
                "userIds": [chat_id],
                "msgKey": "sampleMarkdown",
                "msgParam": json.dumps({"text": text, "title": name}),
            }, headers=headers)

        if caption and ext in self._IMAGE_EXTS:
            # Send caption separately for image messages
            await self._api_post(MEDIA_SEND_URL, {
                "robotCode": self.config.client_id,
                "userIds": [chat_id],
                "msgKey": "sampleMarkdown",
                "msgParam": json.dumps({"text": caption, "title": "Caption"}),
            }, headers=headers)
        return True

    async def _upload_dingtalk_media(
        self, token: str, file_path: str, media_type: str = "image",
    ) -> str | None:
        """Upload a file to DingTalk media API and return the media_id."""
        try:
            url = f"{MEDIA_UPLOAD_URL}?access_token={token}&type={media_type}"
            with open(file_path, "rb") as f:
                resp = await self._http_client.post(
                    url, files={"media": (Path(file_path).name, f)},
                )
            data = resp.json()
            return data.get("media_id")
        except Exception as e:
            logger.warning(f"DingTalk media upload failed: {e}")
            return None

    async def _cleanup(self) -> None:
        if hasattr(self, "_ws_task") and self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None
        await self._stop_ws()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._access_token = None
        logger.info("DingTalk channel stopped")
