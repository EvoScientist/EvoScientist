from .channel import TelegramChannel, TelegramConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["TelegramChannel", "TelegramConfig"]


def create_from_config(config) -> TelegramChannel:
    allowed = _parse_csv(config.telegram_allowed_senders)
    proxy = config.telegram_proxy if config.telegram_proxy else None
    return TelegramChannel(TelegramConfig(
        bot_token=config.telegram_bot_token,
        allowed_senders=allowed,
        proxy=proxy,
    ))


register_channel("telegram", create_from_config)
