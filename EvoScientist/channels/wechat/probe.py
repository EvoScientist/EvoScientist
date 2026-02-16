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
