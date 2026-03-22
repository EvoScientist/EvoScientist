from ..channel_manager import _parse_csv, register_channel
from .channel import FeishuChannel, FeishuConfig

__all__ = ["FeishuChannel", "FeishuConfig"]


def create_from_config(config) -> FeishuChannel:
    allowed = _parse_csv(config.feishu_allowed_senders)
    proxy = config.feishu_proxy or None
    return FeishuChannel(
        FeishuConfig(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            verification_token=config.feishu_verification_token,
            encrypt_key=config.feishu_encrypt_key,
            webhook_port=config.feishu_webhook_port,
            allowed_senders=allowed,
            feishu_domain=config.feishu_domain,
            proxy=proxy,
            subscription_mode=getattr(config, "feishu_subscription_mode", "webhook"),
        )
    )


register_channel("feishu", create_from_config)
