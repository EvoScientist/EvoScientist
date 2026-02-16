"""Slack bot token validation."""

import logging

logger = logging.getLogger(__name__)


async def validate_slack_tokens(
    bot_token: str,
    app_token: str | None = None,
    proxy: str | None = None,
) -> tuple[bool, str]:
    """Validate Slack bot token via the auth.test API.

    Optionally checks the app-level token format (must start with ``xapp-``).

    Returns:
        Tuple of (is_valid, message).
    """
    if not bot_token:
        return False, "No bot token provided"

    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"

    # Validate bot token via auth.test
    url = "https://slack.com/api/auth.test"
    headers = {"Authorization": f"Bearer {bot_token}"}
    try:
        async with httpx.AsyncClient(proxy=proxy) as client:
            resp = await client.post(url, headers=headers, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            error = data.get("error", "unknown error")
            return False, f"Invalid bot token: {error}"
        bot_name = data.get("user", "unknown")
        team = data.get("team", "unknown")
    except Exception as e:
        return False, f"Error: {e}"

    # Optionally validate app token format
    if app_token:
        if not app_token.startswith("xapp-"):
            return False, "App token must start with 'xapp-'"

    return True, f"Bot: {bot_name} (team: {team})"
