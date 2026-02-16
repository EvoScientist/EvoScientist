from .channel import SlackChannel, SlackConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["SlackChannel", "SlackConfig"]


def create_from_config(config) -> SlackChannel:
    allowed = _parse_csv(config.slack_allowed_senders)
    channels = _parse_csv(config.slack_allowed_channels)
    proxy = config.slack_proxy if config.slack_proxy else None
    return SlackChannel(SlackConfig(
        bot_token=config.slack_bot_token,
        app_token=config.slack_app_token,
        allowed_senders=allowed,
        allowed_channels=channels,
        proxy=proxy,
    ))


register_channel("slack", create_from_config)
