"""QQ channel for EvoScientist.

Uses the official qq-botpy SDK for WebSocket connection.

Usage in config:
    channel_enabled = "qq"
    qq_app_id = "your_app_id"
    qq_app_secret = "your_app_secret"
"""

from ..channel_manager import _parse_csv, register_channel
from .channel import QQChannel, QQConfig

__all__ = ["QQChannel", "QQConfig"]


def create_from_config(config) -> QQChannel:
    allowed = _parse_csv(config.qq_allowed_senders)
    return QQChannel(
        QQConfig(
            app_id=config.qq_app_id,
            app_secret=config.qq_app_secret,
            allowed_senders=allowed,
        )
    )


register_channel("qq", create_from_config)
