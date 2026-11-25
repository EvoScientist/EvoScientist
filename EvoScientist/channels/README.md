# Channels

Connect EvoScientist to messaging platforms so they share the same agent core, memory, and workspace as the CLI. All settings live in `~/.config/evoscientist/config.yaml` (or environment variables prefixed with `EVOSCIENTIST_`).

## Quick Start

1. **Install the extra for your platform** (quote brackets in zsh):
   ```bash
   pip install 'EvoScientist[telegram]'
   # Optional extras in pyproject.toml:
   #   telegram, discord, slack, feishu, wechat, qq, all-channels
   # DingTalk, Signal, Email, and iMessage need no extra Python packages.
   ```

2. **Configure via wizard, CLI, or env vars:**
   ```bash
   EvoSci channel setup telegram   # or `EvoSci onboard`

   # Or directly:
   EvoSci config set channel_enabled telegram
   EvoSci config set telegram_bot_token "123456:ABC-xxx"
   EvoSci config set telegram_allowed_senders "YOUR_USER_ID"
   ```

3. **Run:**
   ```bash
   EvoSci serve   # Headless mode: agent + all enabled channels
   ```
   You can also start channels inside an interactive CLI session via `/channel`, or run multiple platforms concurrently with `channel_enabled: "telegram,discord,slack"`.

---

## How Messages Are Handled

- **Deduplication & debounce:** Duplicate platform message IDs are ignored, and rapid consecutive messages from the same sender are merged before reaching the agent.
- **Access control (`*_allowed_senders`, `*_allowed_channels`):** Empty allows everyone—**always set an allowlist on any persistent deployment.** Tokens are never logged at `INFO` level, and webhook channels verify request signatures when configured.
- **Group mention gating (`require_mention`):** Controls when the bot replies in groups: `"group"` *(default: `@mention` required in groups, not DMs)*, `"always"`, or `"off"`. Recent group messages are buffered so the agent has context when mentioned.
- **Formatting & chunking:** Outbound Markdown is converted to each platform's native format, split cleanly without breaking code blocks when possible, and retried on transient network errors.

---

## Platform Matrix

| Channel | Extra needed | Transport | Public IP? | Format | Media / Voice |
|---|---|---|:-:|---|---|
| **Telegram** | `telegram` | Long polling | No | HTML | S/R · Voice R |
| **Discord** | `discord` | WebSocket (`discord.py`) | No | Discord MD | S/R (threads) |
| **Slack** | `slack` | Socket Mode WS | No | mrkdwn | S/R (threads) |
| **Feishu (Lark)** | `feishu` | WebSocket *or* Webhook (`:9000`) | WS: No · Webhook: Yes | Post JSON | S/R · Voice R |
| **WeChat** | `wechat` | Personal QR / WeCom / MP Webhook (`:9001`) | Personal: No · Webhook: Yes | MD / Plain | S/R · Voice R |
| **DingTalk** | *(none)* | Stream Mode WS | No | MD | S/R · Voice R |
| **QQ** | `qq` | Bot Gateway WS | No | MD / Plain | S/R |
| **Signal** | *(needs `signal-cli`)* | TCP JSON-RPC (`:7583`) | No | Plain | S/R · Voice R |
| **Email** | *(none)* | IMAP (`993`) + SMTP (`587`) | No | HTML + Plain | S/R |
| **iMessage** | *(macOS `imsg` CLI)* | stdio JSON-RPC | No | Plain | S/R · Voice R |

*(S = send, R = receive. Telegram, Discord, Slack, Feishu, WeChat, and DingTalk also support `*_proxy`.)*

---

## Per-Channel Setup

### Telegram
- **Install:** `pip install 'EvoScientist[telegram]'`
- **Setup:** Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) and get your numeric ID from [@userinfobot](https://t.me/userinfobot). For group chats, send `/setprivacy` → `Disable` in BotFather.
- **Config (`EVOSCIENTIST_TELEGRAM_*`):**
  ```yaml
  channel_enabled: "telegram"
  telegram_bot_token: "123456789:ABC..."
  telegram_allowed_senders: "123456789"
  telegram_proxy: ""
  ```

### Discord
- **Install:** `pip install 'EvoScientist[discord]'`
- **Setup:** In [Discord Developer Portal](https://discord.com/developers/applications), create an app → **Bot** → copy token and enable **Message Content Intent** → **OAuth2 URL Generator** (`bot`; `Send Messages`, `Read Message History`, `Attach Files`, `Add Reactions`) → invite to your server. Enable Developer Mode in Discord to copy User/Channel IDs.
- **Config (`EVOSCIENTIST_DISCORD_*`):**
  ```yaml
  channel_enabled: "discord"
  discord_bot_token: "MTIz..."
  discord_allowed_senders: ""
  discord_allowed_channels: ""
  discord_proxy: ""
  ```

### Slack
- **Install:** `pip install 'EvoScientist[slack]'`
- **Setup:** At [Slack API](https://api.slack.com/apps), create an app → enable **Socket Mode** and generate an App Token (`xapp-...`, `connections:write`) → under **OAuth & Permissions** add bot scopes (`chat:write`, `channels:history`, `groups:history`, `im:history`, `files:read`, `files:write`, `reactions:write`) → install to workspace (`xoxb-...`) → under **Event Subscriptions** subscribe to `message.channels`, `message.groups`, `message.im`, `app_mention`.
- **Config (`EVOSCIENTIST_SLACK_*`):**
  ```yaml
  channel_enabled: "slack"
  slack_bot_token: "xoxb-..."
  slack_app_token: "xapp-1-..."
  slack_allowed_senders: ""    # Member ID (U...)
  slack_allowed_channels: ""   # Channel ID (C...)
  slack_proxy: ""
  ```

### Feishu (Lark)
- **Install:** `pip install 'EvoScientist[feishu]'`
- **Setup:** Create a custom app on [Feishu Open Platform](https://open.feishu.cn/app) (or [Lark](https://open.larksuite.com/app)), copy **App ID** and **App Secret**, enable event `im.message.receive_v1` and permission `im:message:send_as_bot`, and publish.
  - **WebSocket mode (`websocket`, recommended):** Select **Long Connection (长连接)** in Event Subscriptions—no public IP or tunnel needed.
  - **Webhook mode (`webhook`):** Point callback URL to `http://<host>:9000/webhook/event` (use `ngrok`/`natapp` locally) and set `feishu_verification_token` / `feishu_encrypt_key`.
- **Config (`EVOSCIENTIST_FEISHU_*`):**
  ```yaml
  channel_enabled: "feishu"
  feishu_app_id: "cli_..."
  feishu_app_secret: "..."
  feishu_subscription_mode: "websocket"   # or "webhook"
  feishu_webhook_port: 9000               # webhook mode only
  feishu_verification_token: ""           # webhook mode only
  feishu_encrypt_key: ""                  # webhook mode only (optional)
  feishu_domain: "https://open.feishu.cn" # https://open.larksuite.com for Lark
  feishu_allowed_senders: ""              # open_id
  feishu_proxy: ""
  ```

### WeChat (Personal, WeCom, or Official Account)
- **Install:** `pip install 'EvoScientist[wechat]'`
- **Personal WeChat (`wechat_backend: "personal"`):** Sign in by scanning a QR code (iLink Bot API—no public IP needed):
  ```bash
  python -m EvoScientist.channels.wechat.serve --qr-login
  ```
  ```yaml
  channel_enabled: "wechat"
  wechat_backend: "personal"
  wechat_personal_dm_policy: "open"          # open | allowlist
  wechat_personal_group_policy: "disabled"   # disabled | allowlist | open
  wechat_personal_group_allowed: ""
  wechat_allowed_senders: ""
  ```
- **WeCom (`wechat_backend: "wecom"`):** Create an app in [WeCom Admin Console](https://work.weixin.qq.com), set Receive Messages URL to `http://<host>:9001/wechat/callback`, and **add your server's public IP to Trusted IP** (otherwise calls fail with error `60020`).
  ```yaml
  channel_enabled: "wechat"
  wechat_backend: "wecom"
  wechat_webhook_port: 9001
  wechat_wecom_corp_id: "ww..."
  wechat_wecom_agent_id: "1000002"
  wechat_wecom_secret: "..."
  wechat_wecom_token: "..."
  wechat_wecom_encoding_aes_key: "..."
  wechat_allowed_senders: ""
  wechat_proxy: ""
  ```
- **Official Account (`wechat_backend: "wechatmp"`):** Requires a verified service account; set `wechat_mp_app_id`, `wechat_mp_app_secret`, `wechat_mp_token`, and `wechat_mp_encoding_aes_key` (plain-text replies only).

### DingTalk
- **Install:** No extra Python packages needed.
- **Setup:** Create a bot app at [DingTalk Open Platform](https://open-dev.dingtalk.com), copy **AppKey** (`client_id`) and **AppSecret**, and enable **Stream Mode** (no public IP needed).
- **Config (`EVOSCIENTIST_DINGTALK_*`):**
  ```yaml
  channel_enabled: "dingtalk"
  dingtalk_client_id: "ding..."
  dingtalk_client_secret: "..."
  dingtalk_allowed_senders: ""
  dingtalk_proxy: ""
  ```

### QQ
- **Install:** `pip install 'EvoScientist[qq]'`
- **Setup:** Create a bot on [QQ Open Platform](https://q.qq.com), copy **AppID** and **AppSecret**, and add it as a friend or to a group.
- **Config (`EVOSCIENTIST_QQ_*`):**
  ```yaml
  channel_enabled: "qq"
  qq_app_id: "..."
  qq_app_secret: "..."
  qq_allowed_senders: ""
  ```

### Signal
- **Install:** No extra Python packages; install [`signal-cli`](https://github.com/AsamK/signal-cli#installation) separately.
- **Setup:** Register (`signal-cli -u +1234567890 register` / `verify`) or link a device (`signal-cli link -n "EvoScientist"`). EvoScientist auto-starts the `signal-cli` daemon if needed.
- **Config (`EVOSCIENTIST_SIGNAL_*`):**
  ```yaml
  channel_enabled: "signal"
  signal_phone_number: "+1234567890"   # E.164 format
  signal_cli_path: "signal-cli"
  signal_config_dir: ""
  signal_allowed_senders: ""
  signal_rpc_port: 7583
  ```

### Email (IMAP + SMTP)
- **Install:** Built-in (no extra required).
- **Setup:** Use an app password (for Gmail/Outlook: IMAP `993` SSL, SMTP `587` STARTTLS).
- **Config (`EVOSCIENTIST_EMAIL_*`):**
  ```yaml
  channel_enabled: "email"
  email_imap_host: "imap.gmail.com"
  email_imap_port: 993
  email_imap_username: "bot@gmail.com"
  email_imap_password: "app-password"
  email_imap_mailbox: "INBOX"
  email_imap_use_ssl: true
  email_smtp_host: "smtp.gmail.com"
  email_smtp_port: 587
  email_smtp_username: "bot@gmail.com"
  email_smtp_password: "app-password"
  email_smtp_use_tls: true
  email_from_address: "bot@gmail.com"   # defaults to smtp_username
  email_poll_interval: 30               # seconds
  email_mark_seen: true
  email_max_body_chars: 12000
  email_subject_prefix: "Re: "
  email_allowed_senders: ""
  ```

### iMessage *(macOS only)*
- **Install:** No Python extra; requires [`imsg`](https://github.com/anthropics/imsg) (`brew install imsg`), signed-in Messages.app, and **Full Disk Access** for your terminal. Not usable inside Docker.
- **Config (`EVOSCIENTIST_IMESSAGE_*`):**
  ```yaml
  channel_enabled: "imessage"
  imessage_allowed_senders: "+1234567890,user@icloud.com"  # phone, email, chat_id:123, chat_guid:..., or *
  ```

---

## Docker Quick Reference

Polling and WebSocket channels need no exposed ports; webhook channels (Feishu webhook, WeChat WeCom/MP) map ports `9000`/`9001`:

```bash
# Polling / WebSocket example
docker run -d \
  -e EVOSCIENTIST_CHANNEL_ENABLED="telegram" \
  -e EVOSCIENTIST_TELEGRAM_BOT_TOKEN="123456:ABC-xxx" \
  ghcr.io/evoscientist/evoscientist:latest serve

# Webhook example (WeChat)
docker run -d -p 9001:9001 \
  -e EVOSCIENTIST_CHANNEL_ENABLED="wechat" \
  -e EVOSCIENTIST_WECHAT_BACKEND="wecom" \
  -e EVOSCIENTIST_WECHAT_WECOM_CORP_ID="ww..." \
  -e EVOSCIENTIST_WECHAT_WECOM_SECRET="..." \
  ghcr.io/evoscientist/evoscientist:latest serve
```

## Troubleshooting

- **Bot doesn't reply:** Check `channel_enabled`, verify credentials (`EvoSci config get ...`), verify your ID is in `*_allowed_senders`, and confirm you `@mentioned` the bot in group chats (or set `require_mention: "off"`).
- **Missing dependency error:** Install the platform extra (`pip install 'EvoScientist[<channel>]'`, or `all-channels`).
- **Webhook not receiving events:** Confirm public reachability (or `ngrok`/`natapp`), exact callback path (`/webhook/event` for Feishu, `/wechat/callback` for WeChat), and matching signing tokens.
- **WeCom error `60020`:** Add your server's public IP to the WeCom app's **Trusted IP** list.
- **Debug logs:** Run with `EVOSCIENTIST_LOG_LEVEL=DEBUG` (or `EvoSci config set log_level debug`) and optionally `EvoSci config set channel_debug_tracing true` to see inbound filter decisions.
