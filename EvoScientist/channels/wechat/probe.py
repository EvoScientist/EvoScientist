"""WeChat/WeCom credential validation."""

import logging

logger = logging.getLogger(__name__)


async def validate_wecom(
    corp_id: str,
    secret: str,
    proxy: str | None = None,
) -> tuple[bool, str]:
    """Validate WeCom credentials by fetching an access token.

    Returns:
        Tuple of (is_valid, message).
    """
    if not corp_id or not secret:
        return False, "corp_id and secret are required"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    url = (
        f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        f"?corpid={corp_id}&corpsecret={secret}"
    )
    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.get(url, timeout=10)
        data = resp.json()
        if data.get("errcode", 0) == 0:
            return True, "WeCom credentials valid"
        return False, f"Error ({data.get('errcode')}): {data.get('errmsg')}"
    except Exception as e:
        return False, f"Error: {e}"


async def validate_wechat_mp(
    app_id: str,
    app_secret: str,
    proxy: str | None = None,
) -> tuple[bool, str]:
    """Validate WeChat Official Account credentials.

    Returns:
        Tuple of (is_valid, message).
    """
    if not app_id or not app_secret:
        return False, "app_id and app_secret are required"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    url = (
        f"https://api.weixin.qq.com/cgi-bin/token"
        f"?grant_type=client_credential"
        f"&appid={app_id}&secret={app_secret}"
    )
    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.get(url, timeout=10)
        data = resp.json()
        if "access_token" in data:
            return True, "WeChat MP credentials valid"
        return False, f"Error ({data.get('errcode')}): {data.get('errmsg')}"
    except Exception as e:
        return False, f"Error: {e}"


async def validate_wechat_personal(
    account_id: str,
    token: str = "",
) -> tuple[bool, str]:
    """Validate that a personal WeChat (iLink) account has been logged in.

    Personal-WeChat credentials are obtained via QR-code scan and persisted
    on disk; there is no offline credential format the user can paste in.
    This probe simply checks that:

    - the account_id is set, and
    - either *token* is supplied inline, or a saved-account file exists for
      *account_id* under ``DATA_DIR/wechat_personal/accounts/``.

    Online liveness is not checked because the iLink long-poll endpoint is
    not designed for cheap probes.
    """
    if not account_id:
        return False, (
            "account_id is required. Run "
            "`python -m EvoScientist.channels.wechat.serve --qr-login` first."
        )

    if token:
        return True, f"Personal WeChat account {account_id[:8]}… token provided"

    from .personal import load_account

    persisted = load_account(account_id)
    if not persisted or not persisted.get("token"):
        return False, (
            f"No saved credentials for account_id={account_id[:8]}…. "
            "Run `python -m EvoScientist.channels.wechat.serve --qr-login`."
        )
    return True, f"Personal WeChat account {account_id[:8]}… loaded from disk"
