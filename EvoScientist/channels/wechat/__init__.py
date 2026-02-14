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
    backend = getattr(config, "wechat_backend", "wecom") or "wecom"
    allowed = _parse_csv(getattr(config, "wechat_allowed_senders", ""))
    proxy = getattr(config, "wechat_proxy", "") or None
    port = int(getattr(config, "wechat_webhook_port", 9001) or 9001)

    if backend == "wechatmp":
        mp_config = WeChatMPConfig(
            app_id=getattr(config, "wechat_mp_app_id", ""),
            app_secret=getattr(config, "wechat_mp_app_secret", ""),
            token=getattr(config, "wechat_mp_token", ""),
            encoding_aes_key=getattr(config, "wechat_mp_encoding_aes_key", ""),
            webhook_port=port,
            allowed_senders=allowed,
            proxy=proxy,
        )
        return WeChatChannel(mp_config, backend="wechatmp")
    else:
        wecom_config = WeComConfig(
            corp_id=getattr(config, "wechat_wecom_corp_id", ""),
            agent_id=getattr(config, "wechat_wecom_agent_id", ""),
            secret=getattr(config, "wechat_wecom_secret", ""),
            token=getattr(config, "wechat_wecom_token", ""),
            encoding_aes_key=getattr(config, "wechat_wecom_encoding_aes_key", ""),
            webhook_port=port,
            allowed_senders=allowed,
            proxy=proxy,
        )
        return WeChatChannel(wecom_config, backend="wecom")


register_channel("wechat", create_from_config)
