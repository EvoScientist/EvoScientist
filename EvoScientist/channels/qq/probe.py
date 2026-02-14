"""QQ Bot credential validation."""

import logging

logger = logging.getLogger(__name__)

QQ_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"


async def validate_qq(
    app_id: str,
    app_secret: str,
) -> tuple[bool, str]:
    """Validate QQ Bot credentials by fetching an access token."""
    if not app_id or not app_secret:
        return False, "app_id and app_secret are required"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    body = {"appId": app_id, "clientSecret": app_secret}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(QQ_TOKEN_URL, json=body, timeout=10)
        data = resp.json()
        if data.get("access_token"):
            return True, "QQ Bot credentials valid"
        return False, f"Error: {data.get('message', data)}"
    except Exception as e:
        return False, f"Error: {e}"
