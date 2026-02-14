"""Discord bot token validation."""

import logging

logger = logging.getLogger(__name__)


async def validate_discord_token(token: str, proxy: str | None = None) -> tuple[bool, str]:
    """Validate a Discord bot token via the REST API.

    Returns:
        Tuple of (is_valid, message).
    """
    if not token:
        return False, "No token provided"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    url = "https://discord.com/api/v10/users/@me"
    headers = {"Authorization": f"Bot {token}"}
    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            username = data.get("username", "unknown")
            return True, f"Bot: {username}"
        return False, "Invalid token"
    except Exception as e:
        return False, f"Error: {e}"
