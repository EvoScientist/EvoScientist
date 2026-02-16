"""WeChat channel implementations for EvoScientist.

Supports multiple WeChat backends:
  - **wecom**: 企业微信应用 (WeCom / WeChat Work) via official API
    — Most stable, pure HTTP, no third-party dependencies
  - **wechatmp**: 微信公众号 (WeChat Official Account) via official API
    — Pure HTTP webhook, suitable for public-facing bots

Both backends use httpx (already a core dependency) and receive messages
via HTTP webhook, send replies via REST API.

Usage in config:
    channel_enabled = "wechat"
    wechat_backend = "wecom"       # or "wechatmp"

    # WeCom settings
    wechat_wecom_corp_id = "..."
    wechat_wecom_agent_id = "..."
    wechat_wecom_secret = "..."
    wechat_wecom_token = "..."
    wechat_wecom_encoding_aes_key = "..."
    wechat_webhook_port = 9001

    # OR: Official Account settings
    wechat_mp_app_id = "..."
    wechat_mp_app_secret = "..."
    wechat_mp_token = "..."
    wechat_mp_encoding_aes_key = "..."
    wechat_webhook_port = 9001
"""

from .channel import WeChatChannel, WeComConfig, WeChatMPConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["WeChatChannel", "WeComConfig", "WeChatMPConfig"]


def create_from_config(config) -> WeChatChannel:
    backend = config.wechat_backend or "wecom"
    allowed = _parse_csv(config.wechat_allowed_senders)
    proxy = config.wechat_proxy if config.wechat_proxy else None
    port = int(config.wechat_webhook_port or 9001)

    if backend == "wechatmp":
        mp_config = WeChatMPConfig(
            app_id=config.wechat_mp_app_id,
            app_secret=config.wechat_mp_app_secret,
            token=config.wechat_mp_token,
            encoding_aes_key=config.wechat_mp_encoding_aes_key,
            webhook_port=port,
            allowed_senders=allowed,
            proxy=proxy,
        )
        return WeChatChannel(mp_config, backend="wechatmp")
    else:
        wecom_config = WeComConfig(
            corp_id=config.wechat_wecom_corp_id,
            agent_id=config.wechat_wecom_agent_id,
            secret=config.wechat_wecom_secret,
            token=config.wechat_wecom_token,
            encoding_aes_key=config.wechat_wecom_encoding_aes_key,
            webhook_port=port,
            allowed_senders=allowed,
            proxy=proxy,
        )
        return WeChatChannel(wecom_config, backend="wecom")


register_channel("wechat", create_from_config)
