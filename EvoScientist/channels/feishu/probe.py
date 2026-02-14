"""Feishu (飞书/Lark) app credential validation."""

import logging

logger = logging.getLogger(__name__)


async def validate_feishu_credentials(
    app_id: str,
    app_secret: str,
    domain: str = "https://open.feishu.cn",
) -> tuple[bool, str]:
    """Validate Feishu app credentials by requesting a tenant_access_token.

    Returns:
        Tuple of (is_valid, message).
    """
    if not app_id:
        return False, "No app_id provided"
    if not app_secret:
        return False, "No app_secret provided"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    url = f"{domain}/open-apis/auth/v3/tenant_access_token/internal"
    body = {"app_id": app_id, "app_secret": app_secret}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, timeout=10)
        data = resp.json()
        if data.get("code") == 0:
            return True, f"App: {app_id}"
        msg = data.get("msg", "unknown error")
        return False, f"Auth failed: {msg}"
    except Exception as e:
        return False, f"Error: {e}"
