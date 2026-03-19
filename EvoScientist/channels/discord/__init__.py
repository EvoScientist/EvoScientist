from ..channel_manager import _parse_csv, register_channel
from .channel import DiscordChannel, DiscordConfig

__all__ = ["DiscordChannel", "DiscordConfig"]


def create_from_config(config) -> DiscordChannel:
    allowed = _parse_csv(config.discord_allowed_senders)
    channels = _parse_csv(config.discord_allowed_channels)
    proxy = config.discord_proxy or None
    return DiscordChannel(
        DiscordConfig(
            bot_token=config.discord_bot_token,
            allowed_senders=allowed,
            allowed_channels=channels,
            proxy=proxy,
        )
    )


register_channel("discord", create_from_config)
