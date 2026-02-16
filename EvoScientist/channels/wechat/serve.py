"""WeChat channel server.

Standalone script to run the WeChat channel with CLI options.

Usage:
    # WeCom (企业微信应用)
    python -m EvoScientist.channels.wechat.serve \\
        --backend wecom \\
        --corp-id CORP_ID \\
        --agent-id AGENT_ID \\
        --secret SECRET \\
        --token TOKEN \\
        --aes-key AES_KEY

    # WeChat Official Account (公众号)
    python -m EvoScientist.channels.wechat.serve \\
        --backend wechatmp \\
        --app-id APP_ID \\
        --app-secret APP_SECRET \\
        --token TOKEN \\
        --aes-key AES_KEY

Options:
    --port PORT          Webhook listen port (default: 9001)
    --allow USER_ID      Allowed sender (repeatable)
    --agent              Use EvoScientist agent as handler
    --thinking           Send thinking content as intermediate messages
"""

import argparse
import logging

from .channel import WeChatChannel, WeComConfig, WeChatMPConfig
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
        description="WeChat channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["wecom", "wechatmp"],
        default="wecom",
        help="WeChat backend type (default: wecom)",
    )
    parser.add_argument("--port", type=int, default=9001, help="Webhook port")
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender ID (repeatable)",
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
        help="Use EvoScientist agent as handler",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Send thinking content (requires --agent)",
    )

    # WeCom settings
    wecom = parser.add_argument_group("WeCom (企业微信)")
    wecom.add_argument("--corp-id", default="", help="WeCom Corp ID")
    wecom.add_argument("--agent-id", default="", help="WeCom Agent ID")
    wecom.add_argument("--secret", default="", help="WeCom Secret")

    # MP settings
    mp = parser.add_argument_group("WeChat Official Account (公众号)")
    mp.add_argument("--app-id", default="", help="MP App ID")
    mp.add_argument("--app-secret", default="", help="MP App Secret")

    # Shared settings
    parser.add_argument("--token", default="", help="Callback verification token")
    parser.add_argument("--aes-key", default="", help="EncodingAESKey")
    parser.add_argument("--proxy", default="", help="HTTP proxy URL")

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    allowed = set(args.allowed_senders) if args.allowed_senders else None
    allowed_channels = set(args.allowed_channels) if args.allowed_channels else None
    proxy = args.proxy or None

    if args.backend == "wecom":
        config = WeComConfig(
            corp_id=args.corp_id,
            agent_id=args.agent_id,
            secret=args.secret,
            token=args.token,
            encoding_aes_key=args.aes_key,
            webhook_port=args.port,
            allowed_senders=allowed,
            allowed_channels=allowed_channels,
            proxy=proxy,
        )
    else:
        config = WeChatMPConfig(
            app_id=args.app_id,
            app_secret=args.app_secret,
            token=args.token,
            encoding_aes_key=args.aes_key,
            webhook_port=args.port,
            allowed_senders=allowed,
            allowed_channels=allowed_channels,
            proxy=proxy,
        )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = WeChatChannel(config, backend=args.backend)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
