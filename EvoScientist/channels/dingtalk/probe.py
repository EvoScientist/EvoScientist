"""DingTalk credential validation."""

import logging

logger = logging.getLogger(__name__)


async def validate_dingtalk(
    client_id: str,
    client_secret: str,
    proxy: str | None = None,
) -> tuple[bool, str]:
    """Validate DingTalk credentials by fetching an access token."""
    if not client_id or not client_secret:
        return False, "client_id and client_secret are required"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    body = {"appKey": client_id, "appSecret": client_secret}

    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.post(url, json=body, timeout=10)
        data = resp.json()
        if data.get("accessToken"):
            return True, "DingTalk credentials valid"
        return False, f"Error: {data.get('message', data)}"
    except Exception as e:
        return False, f"Error: {e}"
