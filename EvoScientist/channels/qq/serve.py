"""QQ channel server.

Standalone script to run the QQ channel with CLI options.

Usage:
    python -m EvoScientist.channels.qq.serve --app-id ID --app-secret SECRET [OPTIONS]

Examples:
    # Basic usage
    python -m EvoScientist.channels.qq.serve --app-id ID --app-secret SECRET

    # Sandbox mode with allowed senders
    python -m EvoScientist.channels.qq.serve --app-id ID --app-secret SECRET --allow user123

    # With agent and thinking
    python -m EvoScientist.channels.qq.serve --app-id ID --app-secret SECRET --agent --thinking
"""

import argparse
import logging

from .channel import QQChannel, QQConfig
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
        description="QQ channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--app-id",
        required=True,
        help="QQ bot app ID",
    )
    parser.add_argument(
        "--app-secret",
        required=True,
        help="QQ bot app secret",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (QQ user ID). Can be used multiple times.",
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

    config = QQConfig(
        app_id=args.app_id,
        app_secret=args.app_secret,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = QQChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
