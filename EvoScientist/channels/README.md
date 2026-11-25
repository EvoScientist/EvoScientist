# Channels

Connect EvoScientist to up to **10 messaging platforms** so they share the same agent core, memory, and workspace as the CLI. All channel settings live in `~/.config/evoscientist/config.yaml` (or env vars prefixed with `EVOSCIENTIST_`).

## Quick Start

1. **Install the extra for your platform** (quote brackets in zsh):
   ```bash
   pip install 'EvoScientist[telegram]'
   # Extras: telegram, discord, slack, feishu, wechat, dingtalk, qq, signal, all-channels
   # Email and iMessage require no extra Python packages.
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
   EvoSci serve            # Agent + all enabled channels (headless)
   EvoSci channel status   # Check health (or `curl http://localhost:8080/healthz`)
   ```
   You can also start channels inside an interactive CLI session via `/channel`, or run multiple platforms at once with `channel_enabled: "telegram,discord,slack"`.

---

## How Messages Flow

```txt
Platform (x10) ──▶ Inbound Middleware ──▶ MessageBus ──▶ InboundConsumer (8 workers, per-chat lock) ──▶ Agent
                                              │
Platform ◀── Format & Code-Aware Chunking ◀── Dispatcher ◀── OutboundMessage ◀───────────────────────────┘
```

1. **Inbound middleware** runs in order on every message:
   - **Dedup** — drops duplicate platform `message_id`s (60s LRU cache, 1,000 entries).
   - **Allowlist & Pairing (`dm_policy`)** — `"open"` (everyone), `"allowlist"` (`allowed_senders` / `allowed_channels`), or `"pairing"` (unknown DM senders must enter a pairing code first).
   - **Group History** — buffers up to 50 recent group messages (5-min window) and injects them as context once the bot is mentioned.
   - **Mention Gating (`channel_require_mention`)** — `"group"` *(default: @mention required in groups, not DMs)*, `"none"` (reply to everything), or `"always"`.
2. **Debounce (500ms)** merges rapid consecutive messages from the same sender before queuing onto `MessageBus`.
3. **Outbound delivery** converts Markdown to each platform's native format (`UnifiedFormatter`), splits long replies without breaking code blocks whenever possible, uploads media attachments, and retries transient errors with exponential backoff.

> **Security:** Leave `allowed_senders` empty *only* for local testing. On any persistent deployment, always set `*_allowed_senders` (and `*_allowed_channels` for Discord/Slack). Tokens are never logged at `INFO` level, and webhook channels enforce signature verification when configured.

---

## Platform Matrix

| Channel | Transport | Public IP needed? | Format | Max chunk | Media | Voice | Groups / @Mention |
|---|---|:-:|---|---:|:-:|:-:|:-:|
| **Telegram** | Long polling | No | HTML | 4,000 | S/R | R | Yes / `@` |
| **Discord** | WebSocket (`discord.py`) | No | Discord MD | 2,000 | S/R | — | Yes / `@` (threads) |
| **Slack** | Socket Mode WS | No | mrkdwn | 4,000 | S/R | — | Yes / `@` (threads) |
| **Feishu (Lark)** | WebSocket *or* Webhook (`:9000`) | WS: No · Webhook: Yes | Post JSON | 4,096 | S/R | R | Yes / `@` |
| **WeChat** | Webhook (`:9001`) | Yes | MD (WeCom) / Plain (MP) | 4,096 | S/R | R | Yes / `@` |
| **DingTalk** | Stream Mode WS | No | MD | 4,096 | S/R | R | Yes / `@` |
| **QQ** | Bot Gateway WS | No | MD / Plain | 4,096 | S/R | — | Yes / `@` |
| **Signal** | TCP JSON-RPC (`signal-cli`, `:7583`) | No | Plain | 4,096 | S/R | R | Yes / UUID |
| **Email** | IMAP (`993`) + SMTP (`587`) | No | HTML + Plain | None | S/R | — | — |
| **iMessage** | stdio JSON-RPC (`imsg`, macOS) | No | Plain | 4,000 | S/R | R | Yes |

*(S = send, R = receive. All channels support `allowed_senders`; Telegram, Discord, Slack, Feishu, WeChat, and DingTalk also support `*_proxy`.)*

---

## Per-Channel Setup Guides

### Telegram
- **Install:** `pip install 'EvoScientist[telegram]'`
- **Setup:** Create a bot via [@BotFather](https://t.me/BotFather) (`/newbot`), get your numeric ID from [@userinfobot](https://t.me/userinfobot). For group chats, disable privacy mode in BotFather (`/setprivacy` → `Disable`).
- **Config (`EVOSCIENTIST_TELEGRAM_*`):**
  ```yaml
  channel_enabled: "telegram"
  telegram_bot_token: "123456789:ABC..."
  telegram_allowed_senders: "123456789"
  telegram_proxy: ""
  ```
- **Behavior:** Long polling (drops backlog on startup), HTML formatting with plain-text fallback, typing indicator (4s) + 👀 ACK reaction, routes images/video/audio/files automatically.

### Discord
- **Install:** `pip install 'EvoScientist[discord]'`
- **Setup:** In [Discord Developer Portal](https://discord.com/developers/applications), create an app → **Bot** → copy token and enable **Message Content Intent** → **OAuth2 URL Generator** (`bot` scope; `Send Messages`, `Read Message History`, `Attach Files`, `Add Reactions`) → invite to server. Enable Developer Mode in Discord to copy User/Channel IDs.
- **Config (`EVOSCIENTIST_DISCORD_*`):**
  ```yaml
  channel_enabled: "discord"
  discord_bot_token: "MTIz..."
  discord_allowed_senders: ""
  discord_allowed_channels: ""
  discord_proxy: ""
  ```
- **Behavior:** Thread-aware replies, 20 MB attachment downloads, typing indicator (8s), 👀 ACK reaction, respects `429 Retry-After`.

### Slack
- **Install:** `pip install 'EvoScientist[slack]'`
- **Setup:** At [Slack API](https://api.slack.com/apps), create an app → enable **Socket Mode** and generate an App Token (`xapp-...`, scope `connections:write`) → under **OAuth & Permissions** add bot scopes (`chat:write`, `channels:history`, `groups:history`, `im:history`, `files:read`, `files:write`, `reactions:write`) → **Install to Workspace** (`xoxb-...`) → under **Event Subscriptions** subscribe to `message.channels`, `message.groups`, `message.im`, `app_mention`.
- **Config (`EVOSCIENTIST_SLACK_*`):**
  ```yaml
  channel_enabled: "slack"
  slack_bot_token: "xoxb-..."
  slack_app_token: "xapp-1-..."
  slack_allowed_senders: ""    # Member ID (U...)
  slack_allowed_channels: ""   # Channel ID (C...)
  slack_proxy: ""
  ```
- **Behavior:** Socket Mode (no public IP), always replies in thread (`thread_ts`), 👀 ACK reaction, temporary `"..."` typing placeholder, file uploads via `files_upload_v2`.

### Feishu (Lark)
- **Install:** `pip install 'EvoScientist[feishu]'`
- **Setup:** Create a custom app on [Feishu Open Platform](https://open.feishu.cn/app) (or [Lark](https://open.larksuite.com/app)), copy **App ID** and **App Secret**, enable event `im.message.receive_v1` and permission `im:message:send_as_bot`, then publish.
  - **Recommended — WebSocket mode (`websocket`):** Choose **Long Connection (长连接)** in Event Subscriptions. No public IP, tunnel, or verification token needed.
  - **Webhook mode (`webhook`):** Point callback URL to `http://<host>:9000/webhook/event` (use `ngrok`/`natapp` locally) and copy Verification Token / Encrypt Key.
- **Config (`EVOSCIENTIST_FEISHU_*`):**
  ```yaml
  channel_enabled: "feishu"
  feishu_app_id: "cli_..."
  feishu_app_secret: "..."
  feishu_subscription_mode: "websocket"   # or "webhook"
  feishu_webhook_port: 9000               # webhook mode only
  feishu_domain: "https://open.feishu.cn" # use https://open.larksuite.com for Lark
  feishu_allowed_senders: ""              # open_id
  feishu_proxy: ""
  ```
- **Behavior:** Auto-refreshes 2h `tenant_access_token`, converts Markdown to Feishu Post rich text, supports images/files and ACK reactions.

### WeChat (WeCom & Official Account)
- **Install:** `pip install 'EvoScientist[wechat]'`
- **WeCom *(recommended — free, no verification needed)*:** Create an app in [WeCom Admin Console](https://work.weixin.qq.com), set Receive Messages URL to `http://<host>:9001/wechat/callback`, and **add your server's public IP to Trusted IP** (missing this causes error `60020`).
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
- **Official Account (`wechatmp`):** Set `wechat_backend: "wechatmp"` with `wechat_mp_app_id`, `wechat_mp_app_secret`, `wechat_mp_token`, `wechat_mp_encoding_aes_key` (requires verified service account; plain-text replies only).
- **Behavior:** Async webhook responses (returns `"success"` within 5s), optional AES crypto, 2h token auto-refresh, supports text/image/voice/video/location/file/link.

### DingTalk
- **Install:** `pip install 'EvoScientist[dingtalk]'`
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
- **Behavior:** WebSocket gateway (`qq-botpy`), prefers native QQ Markdown (`msg_type=2`) with structured plain-text fallback.

### Signal
- **Install:** `pip install 'EvoScientist[signal]'` plus [`signal-cli`](https://github.com/AsamK/signal-cli#installation).
- **Setup:** Register (`signal-cli -u +1234567890 register` / `verify`) or link an existing device (`signal-cli link -n "EvoScientist"`). EvoScientist auto-starts the `signal-cli` JSON-RPC daemon if needed.
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
- **Behavior:** Polls up to 20 unread messages per cycle, preserves threads via `In-Reply-To`/`References`, sends multipart HTML + plain text, downloads attachments.

### iMessage *(macOS only)*
- **Install:** No Python extra; requires [`imsg`](https://github.com/anthropics/imsg) (`brew install imsg`), signed-in Messages.app, and **Full Disk Access** for your terminal.
- **Config (`EVOSCIENTIST_IMESSAGE_*`):**
  ```yaml
  channel_enabled: "imessage"
  imessage_cli_path: "imsg"
  imessage_db_path: ""
  imessage_service: "auto"   # imessage | sms | auto
  imessage_region: "US"
  imessage_allowed_senders: ""  # phone, email, chat_id:123, chat_guid:..., or *
  ```
- **Behavior:** Real-time `watch.subscribe` over stdio JSON-RPC (not usable inside Docker).

---

## Docker Quick Reference

Polling/WebSocket channels need no exposed ports; webhook channels (Feishu webhook, WeChat) map ports `9000`/`9001`:

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

- **Bot doesn't reply:** Check `channel_enabled`, verify credentials (`EvoSci config get ...`), verify your ID is in `allowed_senders`, and confirm you `@mentioned` the bot if testing in a group.
- **Import error / channel not found:** Install the platform extra (`pip install 'EvoScientist[<channel>]'`).
- **Webhook not receiving events:** Confirm public reachability (or `ngrok`/`natapp`), exact callback path (`/webhook/event` for Feishu, `/wechat/callback` for WeChat), and matching signing tokens.
- **WeCom error `60020`:** Whitelist your server's public IP in WeCom's **Trusted IP** settings.
- **Debug logs:** Run with `EVOSCIENTIST_LOG_LEVEL=DEBUG` (or `EvoSci config set log_level debug`) to see middleware drops (`DedupMiddleware`, `AllowListMiddleware`, `MentionGatingMiddleware`) and per-channel logs (`EvoScientist.channels.<name>.channel`).
