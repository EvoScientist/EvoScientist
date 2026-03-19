"""DingTalk channel server.

Standalone script to run the DingTalk channel with CLI options.

Usage:
    python -m EvoScientist.channels.dingtalk.serve --client-id ID --client-secret SECRET [OPTIONS]

Examples:
    # Basic usage
    python -m EvoScientist.channels.dingtalk.serve --client-id ID --client-secret SECRET

    # With proxy and allowed senders
    python -m EvoScientist.channels.dingtalk.serve --client-id ID --client-secret SECRET --proxy http://proxy:8080 --allow user123

    # With agent and thinking
    python -m EvoScientist.channels.dingtalk.serve --client-id ID --client-secret SECRET --agent --thinking
"""

import argparse
import logging

from ..bus import MessageBus
from ..standalone import run_standalone
from .channel import DingTalkChannel, DingTalkConfig

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DingTalk channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--client-id",
        required=True,
        help="DingTalk app client ID",
    )
    parser.add_argument(
        "--client-secret",
        required=True,
        help="DingTalk app client secret",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (DingTalk user ID). Can be used multiple times.",
    )
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL",
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

    config = DingTalkConfig(
        client_id=args.client_id,
        client_secret=args.client_secret,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
        proxy=args.proxy,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = DingTalkChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
