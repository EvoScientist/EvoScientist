from ..channel_manager import _parse_csv, register_channel
from .channel import SlackChannel, SlackConfig

__all__ = ["SlackChannel", "SlackConfig"]


def create_from_config(config) -> SlackChannel:
    allowed = _parse_csv(config.slack_allowed_senders)
    channels = _parse_csv(config.slack_allowed_channels)
    proxy = config.slack_proxy or None
    return SlackChannel(
        SlackConfig(
            bot_token=config.slack_bot_token,
            app_token=config.slack_app_token,
            allowed_senders=allowed,
            allowed_channels=channels,
            proxy=proxy,
        )
    )


register_channel("slack", create_from_config)
