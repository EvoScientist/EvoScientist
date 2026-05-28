"""Feishu / Lark scan-to-create (QR code onboard) flow.

Drives the Feishu open-platform device-code flow at
``accounts.feishu.cn/oauth/v1/app/registration`` (and the Lark equivalent
at ``accounts.larksuite.com``).  The user scans a terminal QR code with
Feishu / Lark mobile, the platform provisions a ``PersonalAgent``-archetype
bot application with the required IM permissions pre-attached, and the
poll endpoint returns ``client_id`` / ``client_secret`` — enough to fully
configure :class:`FeishuChannel`.

Domain auto-switches from ``feishu`` to ``lark`` if the poll response's
``user_info.tenant_brand`` reports a Lark tenant.

The HTTP shape mirrors RFC 8628 (OAuth Device Authorization Grant) with
a vendor-specific ``action`` form field selecting init / begin / poll.

Style follows :mod:`EvoScientist.channels.qq.onboard` — httpx, plain
``print`` for progress, and an optional ``qrcode`` dependency for ASCII
rendering.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_ACCOUNTS_URLS: dict[str, str] = {
    "feishu": "https://accounts.feishu.cn",
    "lark": "https://accounts.larksuite.com",
}
_OPEN_URLS: dict[str, str] = {
    "feishu": "https://open.feishu.cn",
    "lark": "https://open.larksuite.com",
}
_REGISTRATION_PATH = "/oauth/v1/app/registration"

_REQUEST_TIMEOUT_S = 10.0
_DEFAULT_POLL_INTERVAL_S = 5
_DEFAULT_EXPIRE_S = 600


def _accounts_base_url(domain: str) -> str:
    return _ACCOUNTS_URLS.get(domain, _ACCOUNTS_URLS["feishu"])


def _open_base_url(domain: str) -> str:
    return _OPEN_URLS.get(domain, _OPEN_URLS["feishu"])


# ---------------------------------------------------------------------------
# QR rendering
# ---------------------------------------------------------------------------

try:
    import qrcode as _qrcode_mod
except (ImportError, TypeError):
    _qrcode_mod = None  # type: ignore[assignment]


def _render_qr(url: str) -> bool:
    """Render *url* as an ASCII QR in the terminal.  Returns True on success."""
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
# Registration HTTP
# ---------------------------------------------------------------------------


def _post_registration(base_url: str, body: dict[str, str]) -> dict:
    """POST form-encoded *body* to the registration endpoint.

    The endpoint replies with JSON even on 4xx responses (``authorization_pending``
    comes back as HTTP 400 with a parseable body), so we always read the body
    and only fall back to raising if the bytes are missing or not JSON.
    """
    import httpx

    url = f"{base_url}{_REGISTRATION_PATH}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    with httpx.Client(timeout=_REQUEST_TIMEOUT_S, follow_redirects=True) as client:
        resp = client.post(url, data=body, headers=headers)
    # Don't raise_for_status — 4xx may still carry a usable JSON body.
    try:
        return resp.json()
    except ValueError:
        resp.raise_for_status()  # re-raise underlying HTTP error
        raise  # pragma: no cover — raise_for_status already raised


def _init_registration(domain: str) -> None:
    """Probe the registration environment.  Raises if client_secret auth is unavailable."""
    res = _post_registration(_accounts_base_url(domain), {"action": "init"})
    methods = res.get("supported_auth_methods") or []
    if "client_secret" not in methods:
        raise RuntimeError(
            f"Feishu / Lark registration environment does not support "
            f"client_secret auth (got: {methods})"
        )


def _begin_registration(domain: str) -> dict:
    """Start the device-code flow.

    Returns a dict with ``device_code``, ``qr_url``, ``user_code``,
    ``interval``, and ``expire_in``.
    """
    res = _post_registration(
        _accounts_base_url(domain),
        {
            "action": "begin",
            "archetype": "PersonalAgent",
            "auth_method": "client_secret",
            "request_user_info": "open_id",
        },
    )
    device_code = res.get("device_code")
    if not device_code:
        raise RuntimeError(
            f"Feishu / Lark registration did not return a device_code: {res}"
        )
    qr_url = res.get("verification_uri_complete") or ""
    sep = "&" if "?" in qr_url else "?"
    qr_url = f"{qr_url}{sep}from=evoscientist&tp=evoscientist"
    return {
        "device_code": device_code,
        "qr_url": qr_url,
        "user_code": res.get("user_code", ""),
        "interval": int(res.get("interval") or _DEFAULT_POLL_INTERVAL_S),
        "expire_in": int(res.get("expire_in") or _DEFAULT_EXPIRE_S),
    }


def _poll_registration(
    *,
    device_code: str,
    interval: int,
    expire_in: int,
    domain: str,
) -> dict | None:
    """Poll until the user scans, or the device_code expires / is denied.

    Auto-switches the polling domain to ``lark`` if the server reports
    ``user_info.tenant_brand == "lark"`` — the credentials only resolve
    against the matching open-platform host.

    Returns a dict with ``app_id``, ``app_secret``, ``domain``, ``open_id``
    on success, or ``None`` on timeout / explicit denial.
    """
    deadline = time.monotonic() + expire_in
    current_domain = domain
    domain_switched = False
    poll_count = 0

    while time.monotonic() < deadline:
        try:
            res = _post_registration(
                _accounts_base_url(current_domain),
                {
                    "action": "poll",
                    "device_code": device_code,
                    "tp": "ob_app",
                },
            )
        except Exception as exc:
            logger.debug("[Feishu onboard] poll request error: %s", exc)
            time.sleep(interval)
            continue

        poll_count += 1
        if poll_count == 1:
            print("  Waiting for scan…", end="", flush=True)
        elif poll_count % 6 == 0:
            print(".", end="", flush=True)

        # Domain auto-detection — the server may still return creds in
        # this same poll, so we fall through rather than restarting.
        user_info = res.get("user_info") or {}
        if (
            user_info.get("tenant_brand") == "lark"
            and not domain_switched
            and current_domain != "lark"
        ):
            current_domain = "lark"
            domain_switched = True

        if res.get("client_id") and res.get("client_secret"):
            print()  # newline after the dots
            return {
                "app_id": res["client_id"],
                "app_secret": res["client_secret"],
                "domain": current_domain,
                "open_id": user_info.get("open_id"),
            }

        error = res.get("error", "")
        if error in {"access_denied", "expired_token"}:
            print()
            logger.warning("[Feishu onboard] Registration %s", error)
            return None

        # authorization_pending / slow_down / unknown — keep polling
        time.sleep(interval)

    print()
    logger.warning("[Feishu onboard] Poll timed out after %ds", expire_in)
    return None


# ---------------------------------------------------------------------------
# Bot probe (best-effort, uses tenant_access_token + /bot/v3/info)
# ---------------------------------------------------------------------------


def _probe_bot(app_id: str, app_secret: str, domain: str) -> dict | None:
    """Fetch bot name / bot_open_id via the open-platform REST API.

    Best-effort: failures return ``None`` and the caller proceeds without
    a friendly bot name.  Uses raw HTTP so we don't require ``lark-oapi``
    to be installed at onboard time (it's only needed for WebSocket mode).
    """
    import httpx

    base = _open_base_url(domain)
    token_url = f"{base}/open-apis/auth/v3/tenant_access_token/internal"
    info_url = f"{base}/open-apis/bot/v3/info"

    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT_S, follow_redirects=True) as client:
            tok_resp = client.post(
                token_url,
                json={"app_id": app_id, "app_secret": app_secret},
            )
            tok_data = tok_resp.json()
            if tok_data.get("code") != 0:
                logger.debug("[Feishu onboard] token fetch failed: %s", tok_data)
                return None
            token = tok_data.get("tenant_access_token")
            if not token:
                return None

            info_resp = client.get(
                info_url,
                headers={"Authorization": f"Bearer {token}"},
            )
            info_data = info_resp.json()
    except Exception as exc:
        logger.debug("[Feishu onboard] bot probe failed: %s", exc)
        return None

    if info_data.get("code") != 0:
        return None
    bot = info_data.get("bot") or info_data.get("data", {}).get("bot") or {}
    return {
        "bot_name": bot.get("app_name") or bot.get("bot_name"),
        "bot_open_id": bot.get("open_id"),
    }


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def qr_register(
    *,
    initial_domain: str = "feishu",
    timeout_seconds: int = 600,
) -> dict[str, Any] | None:
    """Run the Feishu / Lark scan-to-create QR registration flow.

    Args:
        initial_domain: ``"feishu"`` (default, mainland) or ``"lark"`` (overseas).
            Auto-switches mid-flow if the scanning user is on the other tenant.
        timeout_seconds: Wall-clock budget for the whole flow.

    Returns on success::

        {
            "app_id": str,
            "app_secret": str,
            "domain": "feishu" | "lark",
            "open_id": str | None,
            "bot_name": str | None,
            "bot_open_id": str | None,
        }

    Returns ``None`` on expected failures (network, denial, timeout).
    """
    try:
        return _qr_register_inner(
            initial_domain=initial_domain,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.warning("[Feishu onboard] Registration failed: %s", exc)
        return None


def _qr_register_inner(
    *,
    initial_domain: str,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    print("  Connecting to Feishu / Lark…", end="", flush=True)
    _init_registration(initial_domain)
    begin = _begin_registration(initial_domain)
    print(" done.")

    print()
    qr_url = begin["qr_url"]
    if _render_qr(qr_url):
        print(
            f"\n  Scan the QR code above with Feishu / Lark on your phone,\n"
            f"  or open this URL directly:\n  {qr_url}"
        )
    else:
        print(f"  Open this URL in Feishu / Lark on your phone:\n\n  {qr_url}\n")
        print(
            "  Tip: pip install qrcode  to display a scannable QR code here next time"
        )
    print()

    result = _poll_registration(
        device_code=begin["device_code"],
        interval=begin["interval"],
        expire_in=min(begin["expire_in"], timeout_seconds),
        domain=initial_domain,
    )
    if not result:
        return None

    bot_info = _probe_bot(result["app_id"], result["app_secret"], result["domain"])
    if bot_info:
        result["bot_name"] = bot_info.get("bot_name")
        result["bot_open_id"] = bot_info.get("bot_open_id")
    else:
        result["bot_name"] = None
        result["bot_open_id"] = None

    return result
