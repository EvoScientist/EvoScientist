"""Slack channel server.

Standalone script to run the Slack channel with CLI options.

Usage:
    python -m EvoScientist.channels.slack.serve --bot-token TOKEN --app-token TOKEN [OPTIONS]

Examples:
    # Allow all senders (default)
    python -m EvoScientist.channels.slack.serve --bot-token xoxb-... --app-token xapp-...

    # Only allow specific senders and channels
    python -m EvoScientist.channels.slack.serve --bot-token xoxb-... --app-token xapp-... --allow U123 --allow-channel C456

    # With agent and thinking
    python -m EvoScientist.channels.slack.serve --bot-token xoxb-... --app-token xapp-... --agent --thinking
"""

import argparse
import logging

from .channel import SlackChannel, SlackConfig
from ..bus import MessageBus
from ..standalone import run_standalone

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Slack channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bot-token",
        required=True,
        help="Slack bot token (xoxb-...)",
    )
    parser.add_argument(
        "--app-token",
        required=True,
        help="Slack app-level token for Socket Mode (xapp-...)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (Slack user ID). Can be used multiple times.",
    )
    parser.add_argument(
        "--allow-channel",
        action="append",
        dest="allowed_channels",
        help="Allowed channel ID. Can be used multiple times.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use EvoScientist agent as handler (default: echo)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Send thinking content as intermediate messages (requires --agent)",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    config = SlackConfig(
        bot_token=args.bot_token,
        app_token=args.app_token,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
        allowed_channels=set(args.allowed_channels) if args.allowed_channels else None,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = SlackChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
