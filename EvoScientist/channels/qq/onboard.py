"""QQ Bot scan-to-configure (QR code onboard) flow.

Ported from hermes-agent/gateway/platforms/qqbot/onboard.py.

Calls the ``q.qq.com`` ``create_bind_task`` / ``poll_bind_result`` APIs to
generate a QR code URL and poll for scan completion. On success the caller
receives the bot's *app_id*, *client_secret* (decrypted locally), and the
scanner's *user_openid* — enough to fully configure the QQ channel.

The bot must already be registered at https://q.qq.com — scanning binds
the QQ user (developer / admin) to the existing application; it does not
create a new one.

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from enum import IntEnum
from urllib.parse import quote

from .crypto import decrypt_secret, generate_bind_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoints / timing
# ---------------------------------------------------------------------------

# The portal domain is configurable for corporate proxies / sandbox routing.
PORTAL_HOST = os.getenv("QQ_PORTAL_HOST", "q.qq.com")

ONBOARD_CREATE_PATH = "/lite/create_bind_task"
ONBOARD_POLL_PATH = "/lite/poll_bind_result"
QR_URL_TEMPLATE = (
    "https://q.qq.com/qqbot/openclaw/connect.html"
    "?task_id={task_id}&_wv=2&source=evoscientist"
)

ONBOARD_API_TIMEOUT = 10.0
ONBOARD_POLL_INTERVAL = 2.0

_MAX_REFRESHES = 3


# ---------------------------------------------------------------------------
# Bind status
# ---------------------------------------------------------------------------


class BindStatus(IntEnum):
    """Status codes returned by ``poll_bind_result``."""

    NONE = 0
    PENDING = 1
    COMPLETED = 2
    EXPIRED = 3


# ---------------------------------------------------------------------------
# HTTP headers
# ---------------------------------------------------------------------------


def _get_evoscientist_version() -> str:
    try:
        from importlib.metadata import version

        return version("evoscientist")
    except Exception:
        return "dev"


def _build_user_agent() -> str:
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    os_name = platform.system().lower()
    return (
        f"EvoScientistQQ/1.0.0 (Python/{py_version}; {os_name}; "
        f"EvoScientist/{_get_evoscientist_version()})"
    )


def _api_headers() -> dict[str, str]:
    """Standard HTTP headers for q.qq.com onboard API requests.

    ``q.qq.com`` requires ``Accept: application/json`` — without it,
    the server returns a JavaScript anti-bot challenge page.
    """
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _build_user_agent(),
    }


# ---------------------------------------------------------------------------
# QR rendering
# ---------------------------------------------------------------------------

try:
    import qrcode as _qrcode_mod
except (ImportError, TypeError):
    _qrcode_mod = None  # type: ignore[assignment]


def _render_qr(url: str) -> bool:
    """Render a QR code to the terminal. Returns True on success."""
    if _qrcode_mod is None:
        return False
    try:
        qr = _qrcode_mod.QRCode(
            error_correction=_qrcode_mod.constants.ERROR_CORRECT_M,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _create_bind_task(timeout: float = ONBOARD_API_TIMEOUT) -> tuple[str, str]:
    """Create a bind task and return *(task_id, aes_key_base64)*.

    Raises:
        RuntimeError: if the API returns a non-zero ``retcode``.
    """
    import httpx

    url = f"https://{PORTAL_HOST}{ONBOARD_CREATE_PATH}"
    key = generate_bind_key()

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(url, json={"key": key}, headers=_api_headers())
        resp.raise_for_status()
        data = resp.json()

    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", "create_bind_task failed"))

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError("create_bind_task: missing task_id in response")

    logger.debug("create_bind_task ok: task_id=%s", task_id)
    return task_id, key


def _poll_bind_result(
    task_id: str,
    timeout: float = ONBOARD_API_TIMEOUT,
) -> tuple[BindStatus, str, str, str]:
    """Poll the bind result for *task_id*.

    Returns:
        ``(status, bot_appid, bot_encrypt_secret, user_openid)``.

    Raises:
        RuntimeError: if the API returns a non-zero ``retcode``.
    """
    import httpx

    url = f"https://{PORTAL_HOST}{ONBOARD_POLL_PATH}"

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(url, json={"task_id": task_id}, headers=_api_headers())
        resp.raise_for_status()
        data = resp.json()

    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", "poll_bind_result failed"))

    d = data.get("data", {})
    return (
        BindStatus(d.get("status", 0)),
        str(d.get("bot_appid", "")),
        d.get("bot_encrypt_secret", ""),
        d.get("user_openid", ""),
    )


def build_connect_url(task_id: str) -> str:
    """Build the QR-code target URL for a given *task_id*."""
    return QR_URL_TEMPLATE.format(task_id=quote(task_id))


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def qr_register(timeout_seconds: int = 600) -> dict | None:
    """Run the QQ Bot scan-to-configure QR registration flow.

    Handles create → display → poll → decrypt in one call. The QR
    auto-refreshes up to ``_MAX_REFRESHES`` times if the user takes
    too long to scan.

    Args:
        timeout_seconds: Total wall-clock budget across all refreshes.

    Returns:
        ``{"app_id": ..., "client_secret": ..., "user_openid": ...}`` on
        success, or ``None`` on failure / expiry / cancellation.
    """
    deadline = time.monotonic() + timeout_seconds

    for refresh_count in range(_MAX_REFRESHES + 1):
        # ── Create bind task ──
        try:
            task_id, aes_key = _create_bind_task()
        except Exception as exc:
            logger.warning("[QQ onboard] Failed to create bind task: %s", exc)
            return None

        url = build_connect_url(task_id)

        # ── Display QR code + URL ──
        print()
        if _render_qr(url):
            print(f"  Scan the QR code above, or open this URL on your phone:\n  {url}")
        else:
            print(f"  Open this URL in QQ on your phone:\n  {url}")
            print("  Tip: pip install qrcode  to display a scannable QR code here")
        print()

        # ── Poll loop ──
        consecutive_errors = 0
        while time.monotonic() < deadline:
            try:
                status, app_id, encrypted_secret, user_openid = _poll_bind_result(
                    task_id
                )
            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    "[QQ onboard] poll_bind_result failed (%d consecutive): %s",
                    consecutive_errors,
                    exc,
                )
                if consecutive_errors >= 5:
                    print(
                        "\n  Repeated polling failures — aborting."
                        " See logs for details."
                    )
                    return None
                time.sleep(ONBOARD_POLL_INTERVAL)
                continue
            consecutive_errors = 0

            if status == BindStatus.COMPLETED:
                try:
                    client_secret = decrypt_secret(encrypted_secret, aes_key)
                except Exception as exc:
                    logger.warning("[QQ onboard] decrypt_secret failed: %s", exc)
                    return None
                print()
                print(f"  QR scan complete! (App ID: {app_id})")
                if user_openid:
                    print(f"  Scanner's OpenID: {user_openid}")
                return {
                    "app_id": app_id,
                    "client_secret": client_secret,
                    "user_openid": user_openid,
                }

            if status == BindStatus.EXPIRED:
                if refresh_count >= _MAX_REFRESHES:
                    logger.warning(
                        "[QQ onboard] QR code expired %d times — giving up",
                        _MAX_REFRESHES,
                    )
                    return None
                print(
                    f"\n  QR code expired, refreshing... "
                    f"({refresh_count + 1}/{_MAX_REFRESHES})"
                )
                break  # next outer iteration creates a new task

            time.sleep(ONBOARD_POLL_INTERVAL)
        else:
            # deadline reached without completing
            logger.warning("[QQ onboard] Poll timed out after %ds", timeout_seconds)
            return None

    return None
