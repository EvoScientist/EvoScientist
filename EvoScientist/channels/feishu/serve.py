"""Feishu (飞书/Lark) channel server.

Standalone script to run the Feishu channel with CLI options.

Usage:
    python -m EvoScientist.channels.feishu.serve --app-id ID --app-secret SECRET [OPTIONS]

Examples:
    # Basic setup
    python -m EvoScientist.channels.feishu.serve --app-id ID --app-secret SECRET

    # With verification token and custom port
    python -m EvoScientist.channels.feishu.serve --app-id ID --app-secret SECRET \\
        --verification-token TOKEN --webhook-port 9000

    # With agent and thinking
    python -m EvoScientist.channels.feishu.serve --app-id ID --app-secret SECRET --agent --thinking
"""

import argparse
import logging

from .channel import FeishuChannel, FeishuConfig
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
        description="Feishu (飞书/Lark) channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--app-id",
        required=True,
        help="Feishu App ID",
    )
    parser.add_argument(
        "--app-secret",
        required=True,
        help="Feishu App Secret",
    )
    parser.add_argument(
        "--verification-token",
        default="",
        help="Feishu event verification token",
    )
    parser.add_argument(
        "--encrypt-key",
        default="",
        help="Feishu event encrypt key",
    )
    parser.add_argument(
        "--webhook-port",
        type=int,
        default=9000,
        help="Port for webhook HTTP server (default: 9000)",
    )
    parser.add_argument(
        "--domain",
        default="https://open.feishu.cn",
        help="Feishu API domain (use https://open.larksuite.com for Lark)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (Feishu open_id). Can be used multiple times.",
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

    config = FeishuConfig(
        app_id=args.app_id,
        app_secret=args.app_secret,
        verification_token=args.verification_token,
        encrypt_key=args.encrypt_key,
        webhook_port=args.webhook_port,
        feishu_domain=args.domain,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = FeishuChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
