from .channel import FeishuChannel, FeishuConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["FeishuChannel", "FeishuConfig"]


def create_from_config(config) -> FeishuChannel:
    allowed = _parse_csv(config.feishu_allowed_senders)
    return FeishuChannel(FeishuConfig(
        app_id=config.feishu_app_id,
        app_secret=config.feishu_app_secret,
        verification_token=config.feishu_verification_token,
        encrypt_key=config.feishu_encrypt_key,
        webhook_port=config.feishu_webhook_port,
        allowed_senders=allowed,
        feishu_domain=config.feishu_domain,
        proxy=getattr(config, 'feishu_proxy', '') or None,
    ))


register_channel("feishu", create_from_config)
