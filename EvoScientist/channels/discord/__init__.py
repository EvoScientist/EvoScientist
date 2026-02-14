from .channel import DiscordChannel, DiscordConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["DiscordChannel", "DiscordConfig"]


def create_from_config(config) -> DiscordChannel:
    allowed = _parse_csv(config.discord_allowed_senders)
    channels = _parse_csv(config.discord_allowed_channels)
    proxy = config.discord_proxy if config.discord_proxy else None
    return DiscordChannel(DiscordConfig(
        bot_token=config.discord_bot_token,
        allowed_senders=allowed,
        allowed_channels=channels,
        proxy=proxy,
    ))


register_channel("discord", create_from_config)
