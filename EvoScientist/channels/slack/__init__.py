from .channel import SlackChannel, SlackConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["SlackChannel", "SlackConfig"]


def create_from_config(config) -> SlackChannel:
    allowed = _parse_csv(config.slack_allowed_senders)
    channels = _parse_csv(config.slack_allowed_channels)
    return SlackChannel(SlackConfig(
        bot_token=config.slack_bot_token,
        app_token=config.slack_app_token,
        allowed_senders=allowed,
        allowed_channels=channels,
        proxy=getattr(config, 'slack_proxy', '') or None,
    ))


register_channel("slack", create_from_config)
