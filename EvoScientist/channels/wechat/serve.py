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

    # Personal WeChat (个人微信 via iLink Bot)
    # First, log in via QR scan to obtain credentials:
    python -m EvoScientist.channels.wechat.serve --qr-login
    # Then run with the saved account_id:
    python -m EvoScientist.channels.wechat.serve \\
        --backend personal --account-id <id>

Options:
    --port PORT          Webhook listen port (default: 9001)
    --allow USER_ID      Allowed sender (repeatable)
    --agent              Use EvoScientist agent as handler
    --thinking           Send thinking content as intermediate messages
"""

import argparse
import asyncio
import logging

from ..bus import MessageBus
from ..standalone import run_standalone
from .channel import WeChatChannel, WeChatMPConfig, WeComConfig
from .personal import (
    WeixinPersonalChannel,
    WeixinPersonalConfig,
    load_account,
    qr_login,
)

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
        choices=["wecom", "wechatmp", "personal"],
        default="wecom",
        help="WeChat backend type (default: wecom)",
    )
    parser.add_argument(
        "--qr-login",
        action="store_true",
        help="Run interactive QR-code login for personal WeChat and exit",
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

    # Personal-WeChat settings
    personal = parser.add_argument_group("Personal WeChat (iLink Bot)")
    personal.add_argument(
        "--account-id",
        default="",
        help="iLink account_id (obtained via --qr-login)",
    )
    personal.add_argument(
        "--bot-token",
        default="",
        help="iLink bearer token; if omitted, loaded from disk via account-id",
    )
    personal.add_argument(
        "--dm-policy",
        choices=["open", "allowlist"],
        default="open",
        help="Direct-message policy (default: open)",
    )
    personal.add_argument(
        "--group-policy",
        choices=["open", "allowlist", "disabled"],
        default="disabled",
        help="Group-message policy (default: disabled — iLink rarely delivers)",
    )

    # Shared settings
    parser.add_argument("--token", default="", help="Callback verification token")
    parser.add_argument("--aes-key", default="", help="EncodingAESKey")
    parser.add_argument("--proxy", default="", help="HTTP proxy URL")

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    if args.qr_login:
        result = asyncio.run(qr_login())
        if not result:
            raise SystemExit(1)
        return

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
        channel = WeChatChannel(config, backend=args.backend)
    elif args.backend == "wechatmp":
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
        channel = WeChatChannel(config, backend=args.backend)
    else:  # personal
        token = args.bot_token
        if not token and args.account_id:
            persisted = load_account(args.account_id)
            if persisted:
                token = persisted.get("token", "")
        if not args.account_id or not token:
            raise SystemExit(
                "Personal WeChat requires --account-id (and a saved token, "
                "obtained via --qr-login)."
            )
        personal_config = WeixinPersonalConfig(
            account_id=args.account_id,
            token=token,
            allowed_senders=allowed,
            allowed_channels=allowed_channels,
            dm_policy=args.dm_policy,
            group_policy=args.group_policy,
            proxy=proxy,
        )
        channel = WeixinPersonalChannel(personal_config)

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
