# Channels

EvoScientist provides unified integration with 11 messaging platforms. This document covers the architecture overview, capability matrix, and detailed deployment guide for each channel.

Configuration file: `~/.config/evoscientist/config.yaml` (or use environment variables with the `EVOSCIENTIST_` prefix).

## Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Telegram │  │ Discord  │  │  Slack   │  ... (×11)
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   ▼
           ┌──────────────┐
           │  MessageBus  │   async queue, 5000 cap
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │InboundConsumer│  → Agent → OutboundMessage
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │  Dispatcher  │   routes replies to origin channel
           └──────────────┘
```

**Core modules:**

| Module | Responsibility |
|--------|---------------|
| `base.py` | Abstract `Channel` base class — declarative readiness checks, retry strategy, mention stripping, send fallback, media handling |
| `capabilities.py` | `ChannelCapabilities` frozen dataclass — each channel declares its capabilities, framework adapts automatically |
| `mixins.py` | Reusable patterns: `WebhookMixin` (aiohttp + httpx), `WebSocketMixin` (connect/reconnect/heartbeat), `PollingMixin` (async polling), `TokenMixin` (OAuth token refresh) |
| `config.py` | `BaseChannelConfig` — shared config fields (allowed_senders, proxy, text_chunk_limit, etc.) |
| `bus/` | `MessageBus` async message queue + `InboundMessage`/`OutboundMessage` dataclasses |
| `channel_manager.py` | Lifecycle management (start/stop), health checks, channel registry |
| `consumer.py` | `InboundConsumer` — dequeue messages, invoke Agent, publish replies |
| `retry.py` | Configurable exponential backoff retry (`RetryConfig`: attempts, min/max delay, jitter) |
| `markdown_utils.py` | Universal Markdown converter with per-platform formatting plugins |

## Capability Matrix

| Channel | Format | Max Len | Media | Voice | Sticker | Location | Video | Typing | Reaction | Thread | Group | @Mention | No Public IP | Token Refresh | Proxy | Allowlist |
|:--------|:------:|:-------:|:-----:|:-----:|:-------:|:--------:|:-----:|:------:|:--------:|:------:|:-----:|:--------:|:------------:|:-------------:|:-----:|:---------:|
| Telegram | HTML | 4000 | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | | ✓ | ✓ | ✓ | | ✓ | ✓ |
| Discord | Discord | 2000 | ✓ | | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ |
| Slack | Mrkdwn | 4000 | ✓ | | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ |
| Feishu | MD | 4096 | ✓ | ✓ | ✓ | | | | ✓ | | ✓ | ✓ | | ✓ | ✓ | ✓ |
| WeChat | MD | 4096 | ✓ | ✓ | | ✓ | | | | | ✓ | ✓ | | ✓ | ✓ | ✓ |
| DingTalk | MD | 4096 | ✓ | ✓ | | | | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| QQ | Plain | 4096 | ✓ | | | | | | | | ✓ | ✓ | ✓ | | | ✓ |
| Signal | Plain | 4096 | ✓ | ✓ | | | | ✓ | ✓ | | ✓ | ✓ | ✓ | | | ✓ |
| iMessage | Plain | ∞ | ✓ | ✓ | | | | | | | ✓ | | ✓ | | | ✓ |
| Email | HTML | ∞ | ✓ | | | | | | | | | | ✓ | | | ✓ |

### Connection Types

| Channel     | Transport | Connection Mode                        | Default Port |
|-------------|-----------|----------------------------------------|:------------:|
| Telegram    | HTTPS     | Long polling (`getUpdates`)            | —            |
| Discord     | WebSocket | Gateway events (`discord.py`)          | —            |
| Slack       | WebSocket | Socket Mode (`slack-sdk`)              | —            |
| Feishu      | HTTP      | Webhook `POST /webhook/event`          | 9000         |
| WeChat      | HTTP      | Webhook `POST /wechat/callback`        | 9001         |
| DingTalk    | WebSocket | Stream Mode (DingTalk gateway)         | —            |
| QQ          | WebSocket | Bot Gateway (`qq-botpy`)               | —            |
| Signal      | TCP       | JSON-RPC (`signal-cli` daemon)         | 7583         |
| iMessage    | stdio     | JSON-RPC (`imsg` CLI)                  | —            |
| Email       | TCP       | IMAP polling + SMTP send               | 993/587      |

> **"—"** means no listening port is required — no public IP or port forwarding needed.

## Quick Start

### 1. Install channel dependencies

```bash
pip install evoscientist[telegram]
# Available extras: telegram, discord, slack, feishu, wechat,
#   dingtalk, qq, email, signal
# iMessage requires no extra Python dependencies
```

### 2. Configure

```bash
# Option A: Interactive wizard
EvoSci onboard

# Option B: CLI commands
EvoSci config set channel_enabled telegram
EvoSci config set telegram_bot_token "123456:ABC-xxx"

# Option C: Environment variables (EVOSCIENTIST_ prefix, uppercase)
export EVOSCIENTIST_CHANNEL_ENABLED=telegram
export EVOSCIENTIST_TELEGRAM_BOT_TOKEN="123456:ABC-xxx"
```

### 3. Start

```bash
EvoSci serve                # Start agent + all enabled channels
# or
EvoSci channel start        # Standalone channel mode (message loop only)
```

### 4. Health check

```bash
curl http://localhost:8080/healthz
```

```json
{
  "status": "healthy",
  "channels": { "enabled": ["telegram"], "running": ["telegram"] }
}
```

### Running multiple channels

Comma-separate channel names in the config to enable multiple channels simultaneously:

```yaml
channel_enabled: "telegram,discord,imessage"
```

All enabled channels run concurrently via the internal message bus.

---

## Channel Deployment Guides

---

### Telegram

**Install:** `pip install evoscientist[telegram]`

**Prerequisites:**

1. Search for [@BotFather](https://t.me/BotFather) in Telegram, send `/newbot`, and follow the prompts to create a bot.
2. BotFather will return a Bot Token (format: `123456789:ABCdefGHI...`) — save it securely.
3. Get your user ID: send any message to [@userinfobot](https://t.me/userinfobot), it will reply with your numeric ID.
4. (Optional) For group use: add the bot to a group, then in BotFather send `/setprivacy` → `Disable` so the bot can read group messages.

**Configuration:**

```yaml
channel_enabled: "telegram"
telegram_bot_token: "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
telegram_allowed_senders: ""       # Comma-separated user IDs; empty = no restriction
telegram_proxy: ""                 # Optional HTTPS proxy (e.g. http://proxy:8080)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `telegram_bot_token` | `str` | `""` | **Required.** Bot API Token from BotFather |
| `telegram_allowed_senders` | `str` | `""` | Comma-separated user IDs, empty = allow all |
| `telegram_proxy` | `str` | `""` | HTTPS proxy URL |

**Env vars:** `EVOSCIENTIST_TELEGRAM_BOT_TOKEN`, `EVOSCIENTIST_TELEGRAM_ALLOWED_SENDERS`, `EVOSCIENTIST_TELEGRAM_PROXY`

**Technical details:** Long polling mode, `drop_pending_updates=True` on startup to skip backlog. Markdown→Telegram HTML auto-conversion (bold, italic, strikethrough, links, code blocks, headings, lists). Falls back to plain text on HTML parse failure. Media routed by extension to `send_photo`/`send_video`/`send_audio`/`send_document`. In groups, only responds when @mentioned; auto-strips @mention. Typing indicator refreshes every 4s. Retry: 3 attempts, min delay 0.4s, parse errors not retried. Text chunk limit: 4000 chars.

---

### Discord

**Install:** `pip install evoscientist[discord]`

**Prerequisites:**

1. Go to [Discord Developer Portal](https://discord.com/developers/applications) → New Application → enter a name.
2. Left menu **Bot** → Reset Token → copy the Bot Token.
3. Under **Privileged Gateway Intents**, enable **Message Content Intent** (required to read message content).
4. Left menu **OAuth2** → URL Generator:
   - Scopes: check `bot`
   - Bot Permissions: check `Send Messages`, `Read Message History`, `Attach Files`, `Add Reactions`
   - Copy the generated URL, open in browser, select a server to invite the bot.
5. Get user ID: Discord Settings → Advanced → enable Developer Mode → right-click username → Copy User ID.

**Configuration:**

```yaml
channel_enabled: "discord"
discord_bot_token: "MTIzNDU2Nzg5.xxxx.xxxxx"
discord_allowed_senders: ""        # Comma-separated user IDs
discord_allowed_channels: ""       # Comma-separated channel IDs
discord_proxy: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `discord_bot_token` | `str` | `""` | **Required.** Bot Token |
| `discord_allowed_senders` | `str` | `""` | Comma-separated user IDs, empty = allow all |
| `discord_allowed_channels` | `str` | `""` | Comma-separated channel IDs, empty = allow all |
| `discord_proxy` | `str` | `""` | HTTPS proxy URL |

**Env vars:** `EVOSCIENTIST_DISCORD_BOT_TOKEN`, `EVOSCIENTIST_DISCORD_ALLOWED_SENDERS`, `EVOSCIENTIST_DISCORD_ALLOWED_CHANNELS`, `EVOSCIENTIST_DISCORD_PROXY`

**Technical details:** WebSocket Gateway (`discord.py`). In server channels, only responds when @mentioned; DMs respond directly. Replies via `MessageReference`. Attachment download (max 20 MB) with safe filename sanitization. Media sent via `discord.File`. Typing indicator refreshes every 8s. Retry: 3 attempts, parses `Retry-After` header for 429s. Text chunk limit: 2000 chars.

---

### Slack

**Install:** `pip install evoscientist[slack]`

**Prerequisites:**

1. Go to [Slack API](https://api.slack.com/apps) → Create New App → From scratch → select workspace.
2. Left menu **Socket Mode** → enable → Generate App-Level Token, scope `connections:write` → copy App Token (`xapp-...`).
3. Left menu **OAuth & Permissions** → add Bot Token Scopes:
   - `chat:write`, `channels:history`, `groups:history`, `im:history`, `files:read`, `files:write`, `reactions:write`
4. Click **Install to Workspace** → copy Bot User OAuth Token (`xoxb-...`).
5. Left menu **Event Subscriptions** → enable → Subscribe to bot events: `message.channels`, `message.groups`, `message.im`, `app_mention`.
6. Get Member ID: click user avatar → profile → **⋮** → Copy member ID.

**Configuration:**

```yaml
channel_enabled: "slack"
slack_bot_token: "xoxb-xxxx-xxxx-xxxx"
slack_app_token: "xapp-1-xxxx-xxxx"
slack_allowed_senders: ""          # Member ID (U...)
slack_allowed_channels: ""         # Channel ID (C...)
slack_proxy: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `slack_bot_token` | `str` | `""` | **Required.** Bot User OAuth Token (`xoxb-`) |
| `slack_app_token` | `str` | `""` | **Required.** Socket Mode App Token (`xapp-`) |
| `slack_allowed_senders` | `str` | `""` | Comma-separated Member IDs |
| `slack_allowed_channels` | `str` | `""` | Comma-separated Channel IDs |
| `slack_proxy` | `str` | `""` | HTTPS proxy URL |

**Env vars:** `EVOSCIENTIST_SLACK_BOT_TOKEN`, `EVOSCIENTIST_SLACK_APP_TOKEN`, `EVOSCIENTIST_SLACK_ALLOWED_SENDERS`, `EVOSCIENTIST_SLACK_ALLOWED_CHANNELS`, `EVOSCIENTIST_SLACK_PROXY`

**Technical details:** Socket Mode (no public URL needed). Markdown→mrkdwn conversion. DMs respond directly; channels only respond to `app_mention` events. Thread replies via `thread_ts`. Attachments downloaded with Bearer auth. Media sent via `files_upload_v2`. Runs `auth_test()` on startup to verify credentials. Retry: 3 attempts, exponential backoff + jitter. Text chunk limit: 4000 chars.

---

### Feishu (Lark)

**Install:** `pip install evoscientist[feishu]`

**Prerequisites:**

1. Go to [Feishu Open Platform](https://open.feishu.cn/app) (international: [Lark Developer](https://open.larksuite.com/app)) → create a custom app.
2. Copy the **App ID** and **App Secret**.
3. Left menu **Event Subscriptions** → set request URL to `http://your-host:9000/webhook/event` → copy **Verification Token** and **Encrypt Key**.
4. Add event: `im.message.receive_v1` (receive messages).
5. Left menu **Permissions** → enable `im:message:send_as_bot`.
6. Create a version and publish.

> Webhook must be publicly reachable. For local dev, use `ngrok http 9000`.

**Configuration:**

```yaml
channel_enabled: "feishu"
feishu_app_id: "cli_xxxxxxx"
feishu_app_secret: "xxxxxxxxxxxxxxxxxx"
feishu_webhook_port: 9000
feishu_allowed_senders: ""         # open_id
feishu_domain: "https://open.feishu.cn"
feishu_proxy: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `feishu_app_id` | `str` | `""` | **Required.** App ID |
| `feishu_app_secret` | `str` | `""` | **Required.** App Secret |
| `feishu_webhook_port` | `int` | `9000` | Webhook HTTP port |
| `feishu_allowed_senders` | `str` | `""` | Comma-separated open_ids |
| `feishu_domain` | `str` | `"https://open.feishu.cn"` | API domain (use `https://open.larksuite.com` for Lark) |
| `feishu_proxy` | `str` | `""` | HTTPS proxy URL |

**Env vars:** `EVOSCIENTIST_FEISHU_APP_ID`, `EVOSCIENTIST_FEISHU_APP_SECRET`, `EVOSCIENTIST_FEISHU_WEBHOOK_PORT`, `EVOSCIENTIST_FEISHU_DOMAIN`

**Technical details:** Webhook on `POST /webhook/event` with URL verification challenge-response. `tenant_access_token` auto-refresh (2h TTL, refreshes 5 min before expiry). Markdown→Post rich text conversion (code blocks, bold, italic, strikethrough, links, headings, quotes, lists). Plain text fallback. Group @mention filtering. Media: images via `/im/v1/images`, files via `/im/v1/files`. Replies via `/messages/{id}/reply`. Retry: 3 attempts, rate limit delay 2.0s, matches `99991400`/`rate limit`. Text chunk limit: 4096 chars.

---

### WeChat

**Install:** `pip install evoscientist[wechat]`

Two backends supported: **WeCom** (recommended, free, no certification needed) and **WeChat Official Account** (requires verified service account).

#### WeCom

**Prerequisites:**

1. Log in to [WeCom Admin Console](https://work.weixin.qq.com) → App Management → create a custom app.
2. Copy the **Corp ID**, **AgentId**, and **Secret**.
3. In app details → Receive Messages → Set API Receive → URL: `http://your-host:9001/wechat/callback` → copy **Token** and **EncodingAESKey**.

```yaml
channel_enabled: "wechat"
wechat_backend: "wecom"
wechat_webhook_port: 9001
wechat_wecom_corp_id: "ww..."
wechat_wecom_agent_id: "1000002"
wechat_wecom_secret: "xxxxxxxxxxxxxxxxxx"
wechat_wecom_token: "xxxxxxxxxxxxxxxxxx"
wechat_wecom_encoding_aes_key: "xxxxxxxxxxxxxxxxxx"
wechat_allowed_senders: ""
wechat_proxy: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wechat_backend` | `str` | `"wecom"` | `"wecom"` or `"wechatmp"` |
| `wechat_webhook_port` | `int` | `9001` | Callback HTTP port |
| `wechat_wecom_corp_id` | `str` | `""` | **Required (WeCom).** Corp ID |
| `wechat_wecom_agent_id` | `str` | `""` | **Required (WeCom).** App AgentId |
| `wechat_wecom_secret` | `str` | `""` | **Required (WeCom).** App Secret |
| `wechat_wecom_token` | `str` | `""` | **Required (WeCom).** Callback Token |
| `wechat_wecom_encoding_aes_key` | `str` | `""` | **Required (WeCom).** Callback EncodingAESKey |

#### WeChat Official Account

**Prerequisites:**

1. Log in to [WeChat Official Account Platform](https://mp.weixin.qq.com) → Settings & Development → Basic Configuration.
2. Copy the **AppID** and **AppSecret**.
3. Server Configuration → URL: `http://your-host:9001/wechat/callback` → set **Token** and **EncodingAESKey**.

```yaml
wechat_backend: "wechatmp"
wechat_mp_app_id: "wx..."
wechat_mp_app_secret: "xxxxxxxxxxxxxxxxxx"
wechat_mp_token: "xxxxxxxxxxxxxxxxxx"
wechat_mp_encoding_aes_key: "xxxxxxxxxxxxxxxxxx"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wechat_mp_app_id` | `str` | `""` | **Required (MP).** AppID |
| `wechat_mp_app_secret` | `str` | `""` | **Required (MP).** AppSecret |
| `wechat_mp_token` | `str` | `""` | **Required (MP).** Server Token |
| `wechat_mp_encoding_aes_key` | `str` | `""` | **Required (MP).** Server EncodingAESKey |

**Technical details:** Webhook HTTP server. XML message parsing. Signature verification. `access_token` auto-refresh. Optional AES encryption/decryption. WeCom supports Markdown message format; Official Account uses plain text. Media send/receive. Retry + backoff. Text chunk limit: 2048 chars.

---

### DingTalk

**Install:** `pip install evoscientist[dingtalk]`

**Prerequisites:**

1. Go to [DingTalk Open Platform](https://open-dev.dingtalk.com) → App Development → create a bot app.
2. Copy the **AppKey** (Client ID) and **AppSecret** (Client Secret).
3. Enable **Stream Mode** in the app configuration — no public IP needed.
4. Publish the app and add the bot to a group, or test via direct message.

**Configuration:**

```yaml
channel_enabled: "dingtalk"
dingtalk_client_id: "ding..."
dingtalk_client_secret: "xxxxxxxxxxxxxxxxxx"
dingtalk_allowed_senders: ""
dingtalk_proxy: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dingtalk_client_id` | `str` | `""` | **Required.** AppKey |
| `dingtalk_client_secret` | `str` | `""` | **Required.** AppSecret |
| `dingtalk_allowed_senders` | `str` | `""` | Comma-separated user IDs |
| `dingtalk_proxy` | `str` | `""` | HTTPS proxy URL |

**Env vars:** `EVOSCIENTIST_DINGTALK_CLIENT_ID`, `EVOSCIENTIST_DINGTALK_CLIENT_SECRET`

**Technical details:** Stream Mode (WebSocket, no public IP needed). Connects via DingTalk gateway with automatic ping/pong heartbeat and message ACK. `access_token` auto-refresh. Group @mention filtering (strips first `@bot` mention). Supports image, file, video, audio attachment download. Sends in Markdown format (`sampleMarkdown`). Auth errors (`invalidauthentication`/`forbidden`/`40014`) not retried. Text chunk limit: 4096 chars.

---

### QQ

**Install:** `pip install evoscientist[qq]`

**Prerequisites:**

1. Go to [QQ Open Platform](https://q.qq.com) → create a bot application.
2. Complete developer verification, create a sandbox or production bot.
3. Copy the **AppID** and **AppSecret**.
4. Search for and add the bot as a friend in QQ, or add it to a group.

**Configuration:**

```yaml
channel_enabled: "qq"
qq_app_id: "xxxxxxxxxx"
qq_app_secret: "xxxxxxxxxxxxxxxxxx"
qq_allowed_senders: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `qq_app_id` | `str` | `""` | **Required.** AppID |
| `qq_app_secret` | `str` | `""` | **Required.** AppSecret |
| `qq_allowed_senders` | `str` | `""` | Comma-separated user IDs |

**Env vars:** `EVOSCIENTIST_QQ_APP_ID`, `EVOSCIENTIST_QQ_APP_SECRET`

**Technical details:** Uses `qq-botpy` SDK via WebSocket to connect to QQ Bot Gateway. Supports C2C (direct) and group messages. Message deduplication (1000-entry LRU cache). Group @mention filtering (strips first `@bot`). Intents: `public_messages=True`, `direct_message=True`. Text chunk limit: 2048 chars.

---

### Signal

**Install:** `pip install evoscientist[signal]` (also requires [signal-cli](https://github.com/AsamK/signal-cli) installed separately)

**Prerequisites:**

1. Install signal-cli: see [signal-cli installation guide](https://github.com/AsamK/signal-cli#installation).
2. Register or link a phone number:
   - Register: `signal-cli -u +1234567890 register`, then `signal-cli -u +1234567890 verify CODE`
   - Link existing device: `signal-cli link -n "EvoScientist"`
3. EvoScientist will auto-start the signal-cli daemon if it's not already running.

**Configuration:**

```yaml
channel_enabled: "signal"
signal_phone_number: "+1234567890"
signal_cli_path: "signal-cli"
signal_config_dir: ""
signal_allowed_senders: ""
signal_rpc_port: 7583
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `signal_phone_number` | `str` | `""` | **Required.** Signal phone number (E.164 format) |
| `signal_cli_path` | `str` | `"signal-cli"` | Path to signal-cli binary |
| `signal_config_dir` | `str` | `""` | signal-cli config directory (optional) |
| `signal_allowed_senders` | `str` | `""` | Comma-separated phone numbers |
| `signal_rpc_port` | `int` | `7583` | JSON RPC socket port |

**Env vars:** `EVOSCIENTIST_SIGNAL_PHONE_NUMBER`, `EVOSCIENTIST_SIGNAL_CLI_PATH`, `EVOSCIENTIST_SIGNAL_RPC_PORT`

**Technical details:** JSON RPC over TCP socket to signal-cli daemon. Auto-starts daemon if not running (`signal-cli -u +NUMBER daemon --socket localhost:PORT`). Listens for `receive` notifications. Sends via `send` RPC method. Group detection via `groupInfo`. Mention detection via UUID matching. No public IP needed. Text chunk limit: 4096 chars.

---

### Email

**Install:** `pip install evoscientist[email]` (core dependencies included, no extras needed)

**Prerequisites:**

1. Prepare an email account with IMAP + SMTP support (Gmail, Outlook, self-hosted, etc.).
2. **Gmail:** Enable 2FA → generate an App Password. IMAP: `imap.gmail.com:993` (SSL), SMTP: `smtp.gmail.com:587` (STARTTLS).
3. **Outlook/Office 365:** IMAP: `outlook.office365.com:993` (SSL), SMTP: `smtp.office365.com:587` (STARTTLS).
4. Ensure IMAP access is enabled in your email settings.

**Configuration:**

```yaml
channel_enabled: "email"
email_imap_host: "imap.gmail.com"
email_imap_port: 993
email_imap_username: "bot@gmail.com"
email_imap_password: "xxxx-xxxx-xxxx-xxxx"
email_imap_mailbox: "INBOX"
email_imap_use_ssl: true
email_smtp_host: "smtp.gmail.com"
email_smtp_port: 587
email_smtp_username: "bot@gmail.com"
email_smtp_password: "xxxx-xxxx-xxxx-xxxx"
email_smtp_use_tls: true
email_from_address: "bot@gmail.com"
email_poll_interval: 30
email_mark_seen: true
email_allowed_senders: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `email_imap_host` | `str` | `""` | **Required.** IMAP server address |
| `email_imap_port` | `int` | `993` | IMAP port |
| `email_imap_username` | `str` | `""` | **Required.** IMAP login username |
| `email_imap_password` | `str` | `""` | **Required.** IMAP login password (or app password) |
| `email_imap_mailbox` | `str` | `"INBOX"` | Mailbox folder to monitor |
| `email_imap_use_ssl` | `bool` | `true` | Use SSL for IMAP connection |
| `email_smtp_host` | `str` | `""` | **Required.** SMTP server address |
| `email_smtp_port` | `int` | `587` | SMTP port |
| `email_smtp_username` | `str` | `""` | **Required.** SMTP login username |
| `email_smtp_password` | `str` | `""` | **Required.** SMTP login password |
| `email_smtp_use_tls` | `bool` | `true` | Use STARTTLS (`true`) or SSL (`false`) |
| `email_from_address` | `str` | `""` | Sender address (defaults to smtp_username) |
| `email_poll_interval` | `int` | `30` | IMAP poll interval in seconds |
| `email_mark_seen` | `bool` | `true` | Mark emails as read after processing |
| `email_max_body_chars` | `int` | `12000` | Max email body chars (truncated beyond) |
| `email_subject_prefix` | `str` | `"Re: "` | Reply subject prefix |
| `email_allowed_senders` | `str` | `""` | Comma-separated sender email addresses |

**Env vars:** `EVOSCIENTIST_EMAIL_IMAP_HOST`, `EVOSCIENTIST_EMAIL_IMAP_USERNAME`, `EVOSCIENTIST_EMAIL_IMAP_PASSWORD`, `EVOSCIENTIST_EMAIL_SMTP_HOST`, `EVOSCIENTIST_EMAIL_SMTP_USERNAME`, `EVOSCIENTIST_EMAIL_SMTP_PASSWORD`

**Technical details:** IMAP polling mode, checks for UNSEEN emails periodically (max 20 per cycle). Supports SSL and STARTTLS. Auto-parses multipart emails (prefers text/plain, falls back text/html → plain text). Attachments auto-downloaded. Replies set `In-Reply-To` and `References` headers to maintain email threads. Sends HTML + plain text dual format (multipart/alternative), falls back to plain text on HTML failure. IMAP auto-reconnects on disconnect. Auth errors (auth/login/credential) not retried. No public IP needed. Text chunk limit: no limit.

---

### iMessage

**Install:** No extra Python dependencies. Requires the [imsg](https://github.com/anthropics/imsg) CLI tool.

**Requirements:** macOS only (iMessage is Apple-proprietary). Requires a signed-in Apple ID with iMessage and Full Disk Access permission for the terminal app.

**Prerequisites:**

1. Install imsg CLI:
   ```bash
   brew install imsg
   ```
2. Verify: `imsg --version`
3. Ensure Messages.app is signed in and working on macOS.

**Configuration:**

```yaml
channel_enabled: "imessage"
imessage_cli_path: "imsg"
imessage_db_path: ""
imessage_service: "auto"
imessage_region: "US"
imessage_allowed_senders: ""
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `imessage_cli_path` | `str` | `"imsg"` | Path to imsg CLI binary |
| `imessage_db_path` | `str` | `""` | iMessage database path (empty = default) |
| `imessage_service` | `str` | `"auto"` | Send service: `imessage`, `sms`, or `auto` |
| `imessage_region` | `str` | `"US"` | Phone number region code |
| `imessage_allowed_senders` | `str` | `""` | Comma-separated allowlist (see below) |

**Allowlist formats:** phone (`+1234567890`), email (`user@example.com`), `chat_id:123`, `chat_guid:iMessage;-;+1234567890`, wildcard `*`.

**Env vars:** `EVOSCIENTIST_IMESSAGE_CLI_PATH`, `EVOSCIENTIST_IMESSAGE_SERVICE`, `EVOSCIENTIST_IMESSAGE_ALLOWED_SENDERS`

**Technical details:** JSON-RPC over stdio with imsg CLI. Creates `watch.subscribe` on startup for real-time message streaming (not polling). Supports iMessage + SMS dual channel (`service: auto`). Target resolution supports chat_id, chat_guid, chat_identifier, and phone/email. Attachments read from local paths provided by imsg. Group detection via `is_group` field. RPC errors (AppleScript/permission/not found) not retried; only connection timeouts retried. Plain text format (no Markdown). No public IP needed. Text chunk limit: 4000 chars.
