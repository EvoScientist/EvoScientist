"""WeChat channel implementations for EvoScientist.

Supports multiple WeChat backends:
  - **wecom**: 企业微信应用 (WeCom / WeChat Work) via official API
    — Most stable, pure HTTP, no third-party dependencies
  - **wechatmp**: 微信公众号 (WeChat Official Account) via official API
    — Pure HTTP webhook, suitable for public-facing bots
  - **personal**: 个人微信 via Tencent's iLink Bot API
    — Long-poll + AES-128-ECB CDN media protocol; QR-code login required.
    Adapted from hermes-agent.

Backends 1+2 share the HTTP-webhook ``WeChatChannel``; backend 3 uses the
long-poll ``WeixinPersonalChannel``.

Usage in config:
    channel_enabled = "wechat"
    wechat_backend = "wecom"       # or "wechatmp" or "personal"

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

    # OR: Personal WeChat (iLink Bot)
    # First run `python -m EvoScientist.channels.wechat.serve --qr-login`
    # to obtain an account_id + token via QR-code scan.
    wechat_personal_account_id = "..."
    wechat_personal_token = "..."           # optional if persisted on disk
    wechat_personal_dm_policy = "open"      # open | allowlist
    wechat_personal_group_policy = "disabled"
"""

from ..channel_manager import _parse_csv, register_channel
from .channel import WeChatChannel, WeChatMPConfig, WeComConfig
from .personal import WeixinPersonalChannel, WeixinPersonalConfig, qr_login

__all__ = [
    "WeChatChannel",
    "WeChatMPConfig",
    "WeComConfig",
    "WeixinPersonalChannel",
    "WeixinPersonalConfig",
    "qr_login",
]


def create_from_config(config):
    """Factory dispatched on ``config.wechat_backend``."""
    backend = (getattr(config, "wechat_backend", "") or "wecom").lower()
    allowed = _parse_csv(getattr(config, "wechat_allowed_senders", ""))
    proxy = getattr(config, "wechat_proxy", "") or None
    port = int(getattr(config, "wechat_webhook_port", 9001) or 9001)

    if backend == "personal":
        group_allowed = _parse_csv(getattr(config, "wechat_personal_group_allowed", ""))
        cfg = WeixinPersonalConfig(
            account_id=getattr(config, "wechat_personal_account_id", ""),
            token=getattr(config, "wechat_personal_token", ""),
            base_url=getattr(config, "wechat_personal_base_url", "")
            or "https://ilinkai.weixin.qq.com",
            cdn_base_url=getattr(config, "wechat_personal_cdn_base_url", "")
            or "https://novac2c.cdn.weixin.qq.com/c2c",
            dm_policy=getattr(config, "wechat_personal_dm_policy", "open") or "open",
            group_policy=getattr(config, "wechat_personal_group_policy", "disabled")
            or "disabled",
            group_allowed_senders=group_allowed,
            allowed_senders=allowed,
            proxy=proxy,
        )
        return WeixinPersonalChannel(cfg)

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
