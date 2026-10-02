"""Personal WeChat (个人微信) channel via Tencent's iLink Bot API.

Adapted from ``hermes-agent/gateway/platforms/weixin.py``.

This backend connects a *personal* WeChat account using the same iLink Bot
protocol as Hermes Agent:

- **Inbound**: long-poll ``ilink/bot/getupdates`` returns ``msgs[]`` with
  typed ``item_list[]`` entries (text / image / voice / file / video).
- **Outbound**: POST ``ilink/bot/sendmessage`` with a per-peer
  ``context_token`` echoed from the latest inbound message.
- **Media**: AES-128-ECB encrypted CDN protocol against
  ``novac2c.cdn.weixin.qq.com``.
- **Auth**: bearer token obtained via QR-code login (``qr_login()``).
- **State**: account credentials + sync buffer + per-peer ``context_token``
  cache persisted under ``DATA_DIR/wechat_personal/accounts/``.

This is fundamentally different from the WeCom / MP webhook backends —
personal accounts have no official bot API, so we ride iLink's long-poll
gateway. Group delivery for QR-login bot identities is generally not
supported on the iLink side.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import re
import secrets
import struct
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from ..base import Channel, ChannelError, RawIncoming, media_path
from ..capabilities import WECHAT as WECHAT_CAPS
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)


# ── iLink endpoints / constants ────────────────────────────────────

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
WEIXIN_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
ILINK_APP_ID = "bot"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0

EP_GET_UPDATES = "ilink/bot/getupdates"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"
EP_SEND_TYPING = "ilink/bot/sendtyping"
EP_GET_CONFIG = "ilink/bot/getconfig"
EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"

LONG_POLL_TIMEOUT_MS = 35_000
API_TIMEOUT_MS = 15_000
CONFIG_TIMEOUT_MS = 10_000
QR_TIMEOUT_MS = 35_000

MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY_SECONDS = 2
BACKOFF_DELAY_SECONDS = 30
SESSION_EXPIRED_ERRCODE = -14
RATE_LIMIT_ERRCODE = -2
MESSAGE_DEDUP_TTL_SECONDS = 300

# Item types within message.item_list
ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

# Message types
MSG_TYPE_USER = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2

TYPING_START = 1
TYPING_STOP = 2

_WEIXIN_CDN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "novac2c.cdn.weixin.qq.com",
        "ilinkai.weixin.qq.com",
        "wx.qlogo.cn",
        "thirdwx.qlogo.cn",
        "res.wx.qq.com",
        "mmbiz.qpic.cn",
        "mmbiz.qlogo.cn",
    }
)


# ── Helpers ────────────────────────────────────────────────────────


def _safe_id(value: str | None, keep: int = 8) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "?"
    return raw[:keep] if len(raw) > keep else raw


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _random_wechat_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(value).encode()).decode("ascii")


def _base_info() -> dict:
    return {"channel_version": CHANNEL_VERSION}


def _headers(token: str | None, body: str) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode())),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _make_ssl_connector():
    """Return an aiohttp TCPConnector pinned to certifi's CA bundle.

    ``ilinkai.weixin.qq.com`` is not always verifiable against system CA
    stores (notably Homebrew OpenSSL on macOS Apple Silicon). When
    ``certifi`` is available, use Mozilla's bundle for verification.
    """
    try:
        import ssl

        import aiohttp
        import certifi
    except ImportError:
        return None
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return aiohttp.TCPConnector(ssl=ssl_ctx)


def _account_dir() -> Path:
    from ...paths import DATA_DIR

    path = DATA_DIR / "wechat_personal" / "accounts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _account_file(account_id: str) -> Path:
    return _account_dir() / f"{account_id}.json"


def save_account(
    account_id: str, *, token: str, base_url: str, user_id: str = ""
) -> None:
    """Persist iLink account credentials for later reuse."""
    payload = {
        "token": token,
        "base_url": base_url,
        "user_id": user_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _account_file(account_id)
    _atomic_json_write(path, payload)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_account(account_id: str) -> dict | None:
    """Load persisted account credentials, or ``None`` if not found."""
    path = _account_file(account_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sync_buf_path(account_id: str) -> Path:
    return _account_dir() / f"{account_id}.sync.json"


def _load_sync_buf(account_id: str) -> str:
    path = _sync_buf_path(account_id)
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("get_updates_buf", "")
    except Exception:
        return ""


def _save_sync_buf(account_id: str, sync_buf: str) -> None:
    _atomic_json_write(_sync_buf_path(account_id), {"get_updates_buf": sync_buf})


class ContextTokenStore:
    """Disk-backed ``context_token`` cache, keyed by peer user_id.

    iLink expects every outbound reply to echo the latest ``context_token``
    received from the peer. The token is stable across a session but rotates
    when iLink invalidates it; we persist the cache so cron-pushed messages
    can resume after process restarts.
    """

    def __init__(self, account_id: str) -> None:
        self._account_id = account_id
        self._cache: dict[str, str] = {}

    def _path(self) -> Path:
        return _account_dir() / f"{self._account_id}.context-tokens.json"

    def restore(self) -> None:
        path = self._path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "personal-wechat: failed to restore context tokens for %s: %s",
                _safe_id(self._account_id),
                exc,
            )
            return
        for user_id, token in data.items():
            if isinstance(token, str) and token:
                self._cache[user_id] = token

    def get(self, user_id: str) -> str | None:
        return self._cache.get(user_id)

    def set(self, user_id: str, token: str) -> None:
        self._cache[user_id] = token
        try:
            _atomic_json_write(self._path(), dict(self._cache))
        except Exception as exc:
            logger.warning(
                "personal-wechat: failed to persist context tokens for %s: %s",
                _safe_id(self._account_id),
                exc,
            )

    def pop(self, user_id: str) -> None:
        self._cache.pop(user_id, None)


class TypingTicketCache:
    """Short-lived ``typing_ticket`` cache from ``getconfig``."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[str, float]] = {}

    def get(self, user_id: str) -> str | None:
        entry = self._cache.get(user_id)
        if not entry:
            return None
        if time.time() - entry[1] >= self._ttl:
            self._cache.pop(user_id, None)
            return None
        return entry[0]

    def set(self, user_id: str, ticket: str) -> None:
        self._cache[user_id] = (ticket, time.time())


class _MessageDedup:
    """Simple TTL-based message-id deduplicator."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    def is_duplicate(self, key: str) -> bool:
        if not key:
            return False
        now = time.time()
        # Evict expired entries opportunistically
        if len(self._seen) > 1024:
            self._seen = {k: t for k, t in self._seen.items() if now - t < self._ttl}
        if key in self._seen and now - self._seen[key] < self._ttl:
            return True
        self._seen[key] = now
        return False


# ── CDN media (AES-128-ECB) ────────────────────────────────────────


def _cdn_download_url(cdn_base_url: str, encrypted_query_param: str) -> str:
    return (
        f"{cdn_base_url.rstrip('/')}/download"
        f"?encrypted_query_param={quote(encrypted_query_param, safe='')}"
    )


def _assert_weixin_cdn_url(url: str) -> None:
    """Reject URLs that don't point at a known WeChat CDN host (SSRF guard)."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
    except Exception as exc:
        raise ValueError(f"Unparseable media URL: {url!r}") from exc

    if scheme not in ("http", "https"):
        raise ValueError(f"Media URL has disallowed scheme {scheme!r}")
    if host not in _WEIXIN_CDN_ALLOWLIST:
        raise ValueError(f"Media URL host {host!r} is not in the WeChat CDN allowlist.")


def _media_reference(item: dict, key: str) -> dict:
    return (item.get(key) or {}).get("media") or {}


async def _download_bytes(session, url: str, timeout_seconds: float = 60.0) -> bytes:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with session.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.read()


async def _download_and_decrypt_media(
    session,
    *,
    cdn_base_url: str,
    encrypted_query_param: str | None,
    aes_key_b64: str | None,
    full_url: str | None,
    timeout_seconds: float,
) -> bytes:
    """Download (and AES-128-ECB decrypt if keyed) a CDN media payload."""
    from .crypto import aes128_ecb_decrypt, parse_ilink_aes_key

    if encrypted_query_param:
        url = _cdn_download_url(cdn_base_url, encrypted_query_param)
        raw = await _download_bytes(session, url, timeout_seconds)
    elif full_url:
        _assert_weixin_cdn_url(full_url)
        raw = await _download_bytes(session, full_url, timeout_seconds)
    else:
        raise RuntimeError("media item had neither encrypt_query_param nor full_url")

    if aes_key_b64:
        raw = aes128_ecb_decrypt(raw, parse_ilink_aes_key(aes_key_b64))
    return raw


def _cache_media_bytes(data: bytes, suffix: str, prefix: str = "wechat_") -> str:
    """Persist downloaded media to MEDIA_DIR; return the local path."""
    name = f"{prefix}{uuid.uuid4().hex}{suffix}"
    path = media_path(name)
    path.write_bytes(data)
    return str(path)


def _mime_from_filename(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


# ── Protocol primitives ─────────────────────────────────────────────


async def _api_post(
    session,
    *,
    base_url: str,
    endpoint: str,
    payload: dict,
    token: str | None,
    timeout_ms: int,
) -> dict:
    import aiohttp

    body = _json_dumps({**payload, "base_info": _base_info()})
    url = f"{base_url.rstrip('/')}/{endpoint}"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.post(
        url,
        data=body,
        headers=_headers(token, body),
        timeout=timeout,
    ) as resp:
        raw = await resp.text()
        if not resp.ok:
            raise RuntimeError(f"iLink POST {endpoint} HTTP {resp.status}: {raw[:200]}")
        return json.loads(raw)


async def _api_get(session, *, base_url: str, endpoint: str, timeout_ms: int) -> dict:
    import aiohttp

    url = f"{base_url.rstrip('/')}/{endpoint}"
    headers = {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        raw = await resp.text()
        if not resp.ok:
            raise RuntimeError(f"iLink GET {endpoint} HTTP {resp.status}: {raw[:200]}")
        return json.loads(raw)


async def _get_updates(
    session, *, base_url: str, token: str, sync_buf: str, timeout_ms: int
) -> dict:
    try:
        return await _api_post(
            session,
            base_url=base_url,
            endpoint=EP_GET_UPDATES,
            payload={"get_updates_buf": sync_buf},
            token=token,
            timeout_ms=timeout_ms,
        )
    except TimeoutError:
        return {"ret": 0, "msgs": [], "get_updates_buf": sync_buf}


async def _send_message(
    session,
    *,
    base_url: str,
    token: str,
    to: str,
    text: str,
    context_token: str | None,
    client_id: str,
) -> dict:
    if not text or not text.strip():
        raise ValueError("_send_message: text must not be empty")
    msg: dict[str, Any] = {
        "from_user_id": "",
        "to_user_id": to,
        "client_id": client_id,
        "message_type": MSG_TYPE_BOT,
        "message_state": MSG_STATE_FINISH,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    }
    if context_token:
        msg["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_MESSAGE,
        payload={"msg": msg},
        token=token,
        timeout_ms=API_TIMEOUT_MS,
    )


async def _send_typing(
    session,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    typing_ticket: str,
    status: int,
) -> None:
    await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_TYPING,
        payload={
            "ilink_user_id": to_user_id,
            "typing_ticket": typing_ticket,
            "status": status,
        },
        token=token,
        timeout_ms=CONFIG_TIMEOUT_MS,
    )


async def _get_config(
    session,
    *,
    base_url: str,
    token: str,
    user_id: str,
    context_token: str | None,
) -> dict:
    payload: dict[str, Any] = {"ilink_user_id": user_id}
    if context_token:
        payload["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_GET_CONFIG,
        payload=payload,
        token=token,
        timeout_ms=CONFIG_TIMEOUT_MS,
    )


def _is_stale_session(ret, errcode, errmsg) -> bool:
    if ret == SESSION_EXPIRED_ERRCODE or errcode == SESSION_EXPIRED_ERRCODE:
        return True
    if ret == RATE_LIMIT_ERRCODE or errcode == RATE_LIMIT_ERRCODE:
        return (errmsg or "").lower() == "unknown error"
    return False


def _extract_text(item_list: list[dict]) -> str:
    for item in item_list:
        if item.get("type") == ITEM_TEXT:
            text = str((item.get("text_item") or {}).get("text") or "")
            ref = item.get("ref_msg") or {}
            ref_item = ref.get("message_item") or {}
            ref_type = ref_item.get("type")
            if ref_type in (ITEM_IMAGE, ITEM_VIDEO, ITEM_FILE, ITEM_VOICE):
                title = ref.get("title") or ""
                prefix = f"[引用媒体: {title}]\n" if title else "[引用媒体]\n"
                return f"{prefix}{text}".strip()
            return text
    for item in item_list:
        if item.get("type") == ITEM_VOICE:
            voice_text = str((item.get("voice_item") or {}).get("text") or "")
            if voice_text:
                return voice_text
    return ""


def _guess_chat_type(message: dict, account_id: str) -> tuple[str, str]:
    room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
    to_user_id = str(message.get("to_user_id") or "").strip()
    is_group = bool(room_id) or (
        to_user_id
        and account_id
        and to_user_id != account_id
        and message.get("msg_type") == 1
    )
    if is_group:
        return (
            "group",
            room_id or to_user_id or str(message.get("from_user_id") or ""),
        )
    return "dm", str(message.get("from_user_id") or "")


# ── QR login flow ────────────────────────────────────────────────────


async def qr_login(
    *, bot_type: str = "3", timeout_seconds: int = 480
) -> dict[str, str] | None:
    """Run the interactive iLink QR login flow.

    Prints a QR code to the terminal; the user scans it in WeChat. On
    success, persists credentials to ``DATA_DIR/wechat_personal/accounts/``.
    Returns the credential dict or ``None`` on timeout/failure.
    """
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError(
            "aiohttp is required for personal WeChat QR login. "
            "Install with: pip install aiohttp"
        ) from exc

    async with aiohttp.ClientSession(
        trust_env=True, connector=_make_ssl_connector()
    ) as session:
        try:
            qr_resp = await _api_get(
                session,
                base_url=ILINK_BASE_URL,
                endpoint=f"{EP_GET_BOT_QR}?bot_type={bot_type}",
                timeout_ms=QR_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.error("personal-wechat: QR fetch failed: %s", exc)
            return None

        qrcode_value = str(qr_resp.get("qrcode") or "")
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
        if not qrcode_value:
            logger.error("personal-wechat: QR response missing qrcode")
            return None

        qr_scan = qrcode_url or qrcode_value
        print("\n请使用微信扫描以下二维码：")
        if qrcode_url:
            print(qrcode_url)
        try:
            import qrcode

            q = qrcode.QRCode()
            q.add_data(qr_scan)
            q.make(fit=True)
            q.print_ascii(invert=True)
        except Exception:
            print("（终端二维码渲染失败，请直接打开上面的链接扫码）")

        deadline = time.time() + timeout_seconds
        current_base = ILINK_BASE_URL
        refresh_count = 0

        while time.time() < deadline:
            try:
                status_resp = await _api_get(
                    session,
                    base_url=current_base,
                    endpoint=f"{EP_GET_QR_STATUS}?qrcode={qrcode_value}",
                    timeout_ms=QR_TIMEOUT_MS,
                )
            except (TimeoutError, Exception):
                await asyncio.sleep(1)
                continue

            status = str(status_resp.get("status") or "wait")
            if status == "wait":
                print(".", end="", flush=True)
            elif status == "scaned":
                print("\n已扫码，请在微信里确认...")
            elif status == "scaned_but_redirect":
                redirect = str(status_resp.get("redirect_host") or "")
                if redirect:
                    current_base = f"https://{redirect}"
            elif status == "expired":
                refresh_count += 1
                if refresh_count > 3:
                    print("\n二维码多次过期，请重新执行登录。")
                    return None
                print(f"\n二维码已过期，正在刷新... ({refresh_count}/3)")
                try:
                    qr_resp = await _api_get(
                        session,
                        base_url=ILINK_BASE_URL,
                        endpoint=f"{EP_GET_BOT_QR}?bot_type={bot_type}",
                        timeout_ms=QR_TIMEOUT_MS,
                    )
                    qrcode_value = str(qr_resp.get("qrcode") or "")
                    qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
                    qr_scan = qrcode_url or qrcode_value
                    if qrcode_url:
                        print(qrcode_url)
                    try:
                        import qrcode as _qr

                        q = _qr.QRCode()
                        q.add_data(qr_scan)
                        q.make(fit=True)
                        q.print_ascii(invert=True)
                    except Exception:
                        pass
                except Exception:
                    return None
            elif status == "confirmed":
                account_id = str(status_resp.get("ilink_bot_id") or "")
                token = str(status_resp.get("bot_token") or "")
                base_url = str(status_resp.get("baseurl") or ILINK_BASE_URL)
                user_id = str(status_resp.get("ilink_user_id") or "")
                if not account_id or not token:
                    return None
                save_account(
                    account_id, token=token, base_url=base_url, user_id=user_id
                )
                print(f"\n微信连接成功，account_id={account_id}")
                return {
                    "account_id": account_id,
                    "token": token,
                    "base_url": base_url,
                    "user_id": user_id,
                }

            await asyncio.sleep(1)

        print("\n微信登录超时。")
        return None


# ── Config dataclass ────────────────────────────────────────────────


@dataclass
class WeixinPersonalConfig(BaseChannelConfig):
    """Personal WeChat (iLink Bot) configuration."""

    account_id: str = ""
    token: str = ""
    base_url: str = ILINK_BASE_URL
    cdn_base_url: str = WEIXIN_CDN_BASE_URL
    send_chunk_delay_seconds: float = 1.5
    send_chunk_retries: int = 4
    send_chunk_retry_delay_seconds: float = 1.0
    dm_policy: str = "open"
    group_policy: str = "disabled"
    group_allowed_senders: set[str] | None = None
    text_chunk_limit: int = 2000


# ── Channel class ───────────────────────────────────────────────────


class WeixinPersonalChannel(Channel):
    """Personal WeChat channel via Tencent's iLink Bot API.

    Uses long-poll (not standard polling) for inbound, so it manages its
    own loop instead of relying on ``PollingMixin``. The pattern mirrors
    Hermes' ``WeixinAdapter`` but adapts to EvoScientist's Channel API
    (``RawIncoming`` → ``_enqueue_raw`` → ``InboundMessage``).
    """

    name = "wechat"
    capabilities = WECHAT_CAPS
    _ready_attrs = ("_send_session", "_token")
    _typing_interval: float = 5.0
    _rate_limit_patterns = ("rate", "limit", "freq")
    _rate_limit_delay = 2.0

    def __init__(self, config: WeixinPersonalConfig) -> None:
        super().__init__(config)
        self._account_id = config.account_id
        self._token = config.token
        self._base_url = (config.base_url or ILINK_BASE_URL).rstrip("/")
        self._cdn_base_url = (config.cdn_base_url or WEIXIN_CDN_BASE_URL).rstrip("/")
        self._send_chunk_delay = max(0.0, float(config.send_chunk_delay_seconds))
        self._send_retries = max(0, int(config.send_chunk_retries))
        self._send_retry_delay = max(0.0, float(config.send_chunk_retry_delay_seconds))
        self._group_policy = (config.group_policy or "disabled").lower()
        self._group_allowed = config.group_allowed_senders or set()
        self._dm_policy = (config.dm_policy or "open").lower()

        self._poll_session: Any = None
        self._send_session: Any = None
        self._poll_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._token_store: ContextTokenStore | None = (
            ContextTokenStore(self._account_id) if self._account_id else None
        )
        self._typing_cache = TypingTicketCache()
        self._dedup = _MessageDedup(MESSAGE_DEDUP_TTL_SECONDS)

        # Restore credentials from disk if not given inline
        if self._account_id and not self._token:
            persisted = load_account(self._account_id)
            if persisted:
                self._token = str(persisted.get("token") or "")
                if persisted.get("base_url"):
                    self._base_url = str(persisted["base_url"]).rstrip("/")

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        try:
            import aiohttp  # availability check
        except ImportError as exc:
            raise ChannelError(
                "aiohttp is required for personal WeChat. "
                "Install with: pip install aiohttp"
            ) from exc

        if not self._token:
            raise ChannelError(
                "Personal WeChat needs a token. Run `qr_login()` first or set "
                "wechat_personal_token / wechat_personal_account_id."
            )
        if not self._account_id:
            raise ChannelError("Personal WeChat needs an account_id (run qr_login).")

        import aiohttp

        self._poll_session = aiohttp.ClientSession(
            trust_env=True, connector=_make_ssl_connector()
        )
        self._send_session = aiohttp.ClientSession(
            trust_env=True, connector=_make_ssl_connector()
        )
        if self._token_store is not None:
            self._token_store.restore()

        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="wechat-personal-poll"
        )

        logger.info(
            "Personal WeChat channel started (account=%s, base=%s)",
            _safe_id(self._account_id),
            self._base_url,
        )
        if self._group_policy != "disabled":
            logger.warning(
                "wechat_personal_group_policy=%s is set, but iLink QR-login "
                "bot identities typically cannot receive ordinary group "
                "messages — this is an iLink-side limitation.",
                self._group_policy,
            )

    async def _cleanup(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

        if self._poll_session is not None:
            await self._poll_session.close()
            self._poll_session = None
        if self._send_session is not None:
            await self._send_session.close()
            self._send_session = None
        logger.info("Personal WeChat channel stopped")

    # ── Long-poll loop ────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        assert self._poll_session is not None
        sync_buf = _load_sync_buf(self._account_id)
        timeout_ms = LONG_POLL_TIMEOUT_MS
        consecutive_failures = 0

        while self._running:
            try:
                response = await _get_updates(
                    self._poll_session,
                    base_url=self._base_url,
                    token=self._token,
                    sync_buf=sync_buf,
                    timeout_ms=timeout_ms,
                )
                suggested = response.get("longpolling_timeout_ms")
                if isinstance(suggested, int) and suggested > 0:
                    timeout_ms = suggested

                ret = response.get("ret", 0)
                errcode = response.get("errcode", 0)
                if (ret not in (0, None)) or (errcode not in (0, None)):
                    if _is_stale_session(ret, errcode, response.get("errmsg")):
                        logger.error(
                            "Personal WeChat session expired; pausing 10 minutes"
                        )
                        await asyncio.sleep(600)
                        consecutive_failures = 0
                        continue
                    consecutive_failures += 1
                    logger.warning(
                        "getUpdates failed ret=%s errcode=%s errmsg=%s (%d/%d)",
                        ret,
                        errcode,
                        response.get("errmsg", ""),
                        consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES,
                    )
                    delay = (
                        BACKOFF_DELAY_SECONDS
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        else RETRY_DELAY_SECONDS
                    )
                    await asyncio.sleep(delay)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                    continue

                consecutive_failures = 0
                new_sync = str(response.get("get_updates_buf") or "")
                if new_sync:
                    sync_buf = new_sync
                    _save_sync_buf(self._account_id, sync_buf)

                for message in response.get("msgs") or []:
                    task = asyncio.create_task(self._safe_process(message))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_failures += 1
                logger.error(
                    "personal-wechat poll error (%d/%d): %s",
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                    exc,
                )
                delay = (
                    BACKOFF_DELAY_SECONDS
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                    else RETRY_DELAY_SECONDS
                )
                await asyncio.sleep(delay)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0

    async def _safe_process(self, message: dict) -> None:
        try:
            await self._process_message(message)
        except Exception:
            logger.exception("personal-wechat: error processing message")

    async def _process_message(self, message: dict) -> None:
        sender_id = str(message.get("from_user_id") or "").strip()
        if not sender_id or sender_id == self._account_id:
            return

        message_id = str(message.get("message_id") or "").strip()
        if message_id and self._dedup.is_duplicate(message_id):
            return

        item_list = message.get("item_list") or []
        text = _extract_text(item_list)

        if text:
            content_key = (
                f"content:{sender_id}:{hashlib.md5(text.encode()).hexdigest()}"
            )
            if self._dedup.is_duplicate(content_key):
                return

        chat_type, effective_chat_id = _guess_chat_type(message, self._account_id)
        is_group = chat_type == "group"
        if is_group:
            if self._group_policy == "disabled":
                return
            if (
                self._group_policy == "allowlist"
                and effective_chat_id not in self._group_allowed
            ):
                return
        else:
            if self._dm_policy == "disabled":
                return
            if self._dm_policy == "allowlist":
                allowed = getattr(self.config, "allowed_senders", None)
                if allowed and sender_id not in allowed:
                    return

        context_token = str(message.get("context_token") or "").strip()
        if context_token and self._token_store is not None:
            self._token_store.set(sender_id, context_token)
        ticket_task = asyncio.create_task(
            self._maybe_fetch_typing_ticket(sender_id, context_token or None)
        )
        self._background_tasks.add(ticket_task)
        ticket_task.add_done_callback(self._background_tasks.discard)

        media_paths: list[str] = []
        annotations: list[str] = []
        for item in item_list:
            await self._collect_media(item, media_paths, annotations)
            ref_msg = item.get("ref_msg") or {}
            ref_item = ref_msg.get("message_item")
            if isinstance(ref_item, dict):
                await self._collect_media(ref_item, media_paths, annotations)

        if not text and not media_paths and not annotations:
            return

        try:
            ts = datetime.fromtimestamp(int(message.get("create_time") or 0))
        except (ValueError, TypeError, OSError):
            ts = datetime.now()

        await self._enqueue_raw(
            RawIncoming(
                sender_id=sender_id,
                chat_id=effective_chat_id,
                text=text,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=ts,
                message_id=message_id,
                is_group=is_group,
                was_mentioned=True,  # iLink only delivers messages targeted at us
                metadata={
                    "chat_id": effective_chat_id,
                    "context_token": context_token,
                    "backend": "personal",
                },
            )
        )

    # ── Inbound media download ────────────────────────────────────

    async def _collect_media(
        self, item: dict, media_paths: list[str], annotations: list[str]
    ) -> None:
        item_type = item.get("type")
        if item_type == ITEM_IMAGE:
            local = await self._download_image(item)
            if local:
                media_paths.append(local)
            else:
                annotations.append("[image: download failed]")
        elif item_type == ITEM_VIDEO:
            local = await self._download_video(item)
            if local:
                media_paths.append(local)
            else:
                annotations.append("[video: download failed]")
        elif item_type == ITEM_FILE:
            local, ann = await self._download_file(item)
            if local:
                media_paths.append(local)
            elif ann:
                annotations.append(ann)
        elif item_type == ITEM_VOICE:
            local = await self._download_voice(item)
            if local:
                media_paths.append(local)

    async def _download_image(self, item: dict) -> str | None:
        media = _media_reference(item, "image_item")
        image_item = item.get("image_item") or {}
        # iLink puts the AES key directly on image_item as hex
        aes_hex = str(image_item.get("aeskey") or "")
        aes_b64 = (
            base64.b64encode(bytes.fromhex(aes_hex)).decode("ascii")
            if aes_hex
            else media.get("aes_key")
        )
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=aes_b64,
                full_url=media.get("full_url"),
                timeout_seconds=30.0,
            )
            return _cache_media_bytes(data, ".jpg", "wechat_img_")
        except Exception as exc:
            logger.warning("personal-wechat: image download failed: %s", exc)
            return None

    async def _download_video(self, item: dict) -> str | None:
        media = _media_reference(item, "video_item")
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=media.get("aes_key"),
                full_url=media.get("full_url"),
                timeout_seconds=120.0,
            )
            return _cache_media_bytes(data, ".mp4", "wechat_video_")
        except Exception as exc:
            logger.warning("personal-wechat: video download failed: %s", exc)
            return None

    async def _download_file(self, item: dict) -> tuple[str | None, str | None]:
        file_item = item.get("file_item") or {}
        media = file_item.get("media") or {}
        filename = str(file_item.get("file_name") or "document.bin")
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=media.get("aes_key"),
                full_url=media.get("full_url"),
                timeout_seconds=60.0,
            )
            safe = re.sub(r"[^\w.\-]", "_", filename)
            local = media_path(f"wechat_file_{uuid.uuid4().hex}_{safe}")
            local.write_bytes(data)
            return str(local), f"[attachment: {local}]"
        except Exception as exc:
            logger.warning("personal-wechat: file download failed: %s", exc)
            return None, f"[file: {filename} - download failed]"

    async def _download_voice(self, item: dict) -> str | None:
        voice_item = item.get("voice_item") or {}
        media = voice_item.get("media") or {}
        if voice_item.get("text"):
            return None
        try:
            data = await _download_and_decrypt_media(
                self._poll_session,
                cdn_base_url=self._cdn_base_url,
                encrypted_query_param=media.get("encrypt_query_param"),
                aes_key_b64=media.get("aes_key"),
                full_url=media.get("full_url"),
                timeout_seconds=60.0,
            )
            return _cache_media_bytes(data, ".silk", "wechat_voice_")
        except Exception as exc:
            logger.warning("personal-wechat: voice download failed: %s", exc)
            return None

    async def _maybe_fetch_typing_ticket(
        self, user_id: str, context_token: str | None
    ) -> None:
        if not self._poll_session or not self._token:
            return
        if self._typing_cache.get(user_id):
            return
        try:
            response = await _get_config(
                self._poll_session,
                base_url=self._base_url,
                token=self._token,
                user_id=user_id,
                context_token=context_token,
            )
            ticket = str(response.get("typing_ticket") or "")
            if ticket:
                self._typing_cache.set(user_id, ticket)
        except Exception as exc:
            logger.debug("personal-wechat getConfig failed for %s: %s", user_id, exc)

    # ── Outbound: text ────────────────────────────────────────────

    async def _send_chunk(
        self,
        chat_id: str,
        formatted_text: str,
        raw_text: str,
        reply_to: str | None,
        metadata: dict,
    ) -> None:
        """Send a single text chunk via iLink sendmessage with retries.

        On session-expired errors (errcode -14), automatically retries once
        without ``context_token`` — iLink accepts tokenless sends as a
        degraded fallback, keeping cron-pushed messages working when no
        user message has refreshed the session recently.
        """
        if not self._send_session:
            raise RuntimeError("personal-wechat send session not initialized")

        context_token = self._token_store.get(chat_id) if self._token_store else None
        text = formatted_text or raw_text
        last_error: Exception | None = None
        retried_without_token = False

        for attempt in range(self._send_retries + 1):
            try:
                resp = await _send_message(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=text,
                    context_token=context_token,
                    client_id=f"evosci-wechat-{uuid.uuid4().hex}",
                )
                if resp and isinstance(resp, dict):
                    ret = resp.get("ret")
                    errcode = resp.get("errcode")
                    if (ret not in (0, None)) or (errcode not in (0, None)):
                        errmsg = resp.get("errmsg") or resp.get("msg") or "unknown"
                        if (
                            _is_stale_session(ret, errcode, errmsg)
                            and not retried_without_token
                            and context_token
                        ):
                            retried_without_token = True
                            context_token = None
                            if self._token_store is not None:
                                self._token_store.pop(chat_id)
                            logger.warning(
                                "personal-wechat: session expired for %s; "
                                "retrying without context_token",
                                _safe_id(chat_id),
                            )
                            continue
                        is_rate_limited = (
                            ret == RATE_LIMIT_ERRCODE or errcode == RATE_LIMIT_ERRCODE
                        )
                        if is_rate_limited:
                            last_error = RuntimeError(
                                f"iLink sendmessage rate limited: "
                                f"ret={ret} errcode={errcode} errmsg={errmsg}"
                            )
                            if attempt >= self._send_retries:
                                break
                            wait = self._send_retry_delay * 3
                            logger.warning(
                                "personal-wechat: rate limited for %s; "
                                "backing off %.1fs",
                                _safe_id(chat_id),
                                wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise RuntimeError(
                            f"iLink sendmessage error: "
                            f"ret={ret} errcode={errcode} errmsg={errmsg}"
                        )
                # delay between sequential chunks for chat-friendly pacing
                if self._send_chunk_delay > 0:
                    await asyncio.sleep(self._send_chunk_delay)
                return
            except Exception as exc:
                last_error = exc
                if attempt >= self._send_retries:
                    break
                wait = self._send_retry_delay * (attempt + 1)
                logger.warning(
                    "personal-wechat send chunk failed to=%s attempt=%d/%d, "
                    "retrying in %.2fs: %s",
                    _safe_id(chat_id),
                    attempt + 1,
                    self._send_retries + 1,
                    wait,
                    exc,
                )
                if wait > 0:
                    await asyncio.sleep(wait)

        if last_error is not None:
            raise last_error

    # ── Outbound: media ───────────────────────────────────────────

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Personal WeChat outbound media.

        iLink media upload requires a multi-step flow (getuploadurl →
        AES-encrypt → POST to CDN → sendmessage with media reference).
        That flow is non-trivial and was deliberately not ported in the
        first cut; for now we send a placeholder text reference so the
        agent's reply is delivered while media support is added.
        """
        chat_id = self._resolve_media_chat_id(recipient, metadata)
        path = Path(file_path)
        note = f"[media: {path.name}]"
        if caption:
            note = f"{note}\n{caption}"
        try:
            await self._send_chunk(chat_id, note, note, None, metadata or {})
            logger.warning(
                "personal-wechat: media upload not yet implemented; "
                "sent placeholder text for %s",
                path.name,
            )
            return True
        except Exception as exc:
            logger.error("personal-wechat send_media error: %s", exc)
            return False

    # ── Typing indicator ──────────────────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        if not self._send_session or not self._token:
            return
        ticket = self._typing_cache.get(chat_id)
        if not ticket:
            return
        try:
            await _send_typing(
                self._send_session,
                base_url=self._base_url,
                token=self._token,
                to_user_id=chat_id,
                typing_ticket=ticket,
                status=TYPING_START,
            )
        except Exception as exc:
            logger.debug(
                "personal-wechat typing start failed for %s: %s",
                _safe_id(chat_id),
                exc,
            )

    async def stop_typing(self, chat_id: str) -> None:
        if self._send_session and self._token:
            ticket = self._typing_cache.get(chat_id)
            if ticket:
                try:
                    await _send_typing(
                        self._send_session,
                        base_url=self._base_url,
                        token=self._token,
                        to_user_id=chat_id,
                        typing_ticket=ticket,
                        status=TYPING_STOP,
                    )
                except Exception as exc:
                    logger.debug(
                        "personal-wechat typing stop failed for %s: %s",
                        _safe_id(chat_id),
                        exc,
                    )
        await super().stop_typing(chat_id)
