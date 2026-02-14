"""DingTalk (钉钉) channel for EvoScientist.

Uses Stream Mode (WebSocket) for receiving messages — no public IP needed.
Sends replies via HTTP API.

Usage in config:
    channel_enabled = "dingtalk"
    dingtalk_client_id = "your_app_key"
    dingtalk_client_secret = "your_app_secret"
"""

from .channel import DingTalkChannel, DingTalkConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["DingTalkChannel", "DingTalkConfig"]


def create_from_config(config) -> DingTalkChannel:
    allowed = _parse_csv(getattr(config, "dingtalk_allowed_senders", ""))
    proxy = getattr(config, "dingtalk_proxy", "") or None
    return DingTalkChannel(DingTalkConfig(
        client_id=getattr(config, "dingtalk_client_id", ""),
        client_secret=getattr(config, "dingtalk_client_secret", ""),
        allowed_senders=allowed,
        proxy=proxy,
    ))


register_channel("dingtalk", create_from_config)
