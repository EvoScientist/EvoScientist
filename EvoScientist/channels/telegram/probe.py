"""Telegram bot token validation."""

import logging

logger = logging.getLogger(__name__)


async def validate_telegram_token(token: str, proxy: str | None = None) -> tuple[bool, str]:
    """Validate a Telegram bot token via the getMe API.

    Returns:
        Tuple of (is_valid, message).
    """
    if not token:
        return False, "No token provided"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.get(url, timeout=10)
        data = resp.json()
        if data.get("ok"):
            username = data["result"].get("username", "unknown")
            return True, f"Bot: @{username}"
        return False, "Invalid token"
    except Exception as e:
        return False, f"Error: {e}"
