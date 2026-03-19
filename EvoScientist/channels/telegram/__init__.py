from ..channel_manager import _parse_csv, register_channel
from .channel import TelegramChannel, TelegramConfig

__all__ = ["TelegramChannel", "TelegramConfig"]


def create_from_config(config) -> TelegramChannel:
    allowed = _parse_csv(config.telegram_allowed_senders)
    proxy = config.telegram_proxy or None
    return TelegramChannel(
        TelegramConfig(
            bot_token=config.telegram_bot_token,
            allowed_senders=allowed,
            proxy=proxy,
        )
    )


register_channel("telegram", create_from_config)
