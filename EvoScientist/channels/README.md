# Channels

EvoScientist provides unified integration with 10 messaging platforms. This document covers the architecture overview, message processing pipeline, capability matrix, security model, deployment guides, and troubleshooting.

Configuration file: `~/.config/evoscientist/config.yaml` (or use environment variables with the `EVOSCIENTIST_` prefix).

## Table of Contents

- [Architecture](#architecture)
- [Message Processing Pipeline](#message-processing-pipeline)
- [Middleware Pipeline](#middleware-pipeline)
- [Capability Matrix](#capability-matrix)
- [Security and Access Control](#security-and-access-control)
- [Quick Start](#quick-start)
- [Channel Deployment Guides](#channel-deployment-guides)
  - [Telegram](#telegram) | [Discord](#discord) | [Slack](#slack) | [Feishu (Lark)](#feishu-lark) | [WeChat](#wechat)
  - [DingTalk](#dingtalk) | [QQ](#qq) | [Signal](#signal) | [Email](#email) | [iMessage](#imessage)
- [Running Multiple Channels](#running-multiple-channels)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │            Messaging Platforms              │
                        │                                             │
                        │  ┌────────┐ ┌────────┐ ┌────────┐         │
                        │  │Telegram│ │Discord │ │ Slack  │  ...x10 │
                        │  └───┬────┘ └───┬────┘ └───┬────┘         │
                        └──────┼──────────┼──────────┼───────────────┘
                               │          │          │
                        ┌──────┴──────────┴──────────┴───────────────┐
                        │           Inbound Middleware               │
                        │                                            │
                        │  Dedup → AllowList → Pairing → GroupHist   │
                        │                                → Mention   │
                        └──────────────────┬─────────────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────────┐
                        │            MessageBus                │
                        │                                      │
                        │  inbound queue ──► outbound queue    │
                        │  (asyncio.Queue, capacity 5000)      │
                        └──────────┬───────────────┬───────────┘
                                   │               │
                                   ▼               ▼
                        ┌──────────────────┐ ┌─────────────────┐
                        │ InboundConsumer   │ │   Dispatcher    │
                        │                  │ │                 │
                        │ Worker pool (8)  │ │ Routes replies  │
                        │ Per-chat locks   │ │ to origin       │
                        │ Session dedup    │ │ channel          │
                        │ Timeout handling │ │                 │
                        │       │          │ └─────────────────┘
                        │       ▼          │
                        │   Agent Core     │
                        └──────────────────┘
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `base.py` | Abstract `Channel` base class — readiness checks, retry strategy, mention stripping, format fallback, media handling, debounce, send locks |
| `capabilities.py` | `ChannelCapabilities` frozen dataclass — each channel declares features, framework adapts automatically |
| `plugin.py` | `ChannelPlugin` base with adapter slots — `ConfigAdapter`, `SecurityAdapter`, `GroupAdapter`, `MentionAdapter`, `OutboundAdapter`, `ThreadingAdapter`, etc. |
| `mixins.py` | Reusable async patterns: `WebhookMixin` (aiohttp server + httpx client), `WebSocketMixin` (connect/reconnect/heartbeat), `PollingMixin` (async polling loop), `TokenMixin` (OAuth token auto-refresh) |
| `config.py` | `BaseChannelConfig` — shared config fields (allowed_senders, proxy, text_chunk_limit, etc.) + `SingleAccountConfigAdapter` / `MultiAccountConfigAdapter` |
| `bus/` | `MessageBus` async event bus with `InboundMessage` / `OutboundMessage` dataclasses, decoupling channels from agent core |
| `channel_manager.py` | `ChannelManager` — lifecycle management (start/stop), health monitoring, channel registry, account management, outbound dispatch |
| `consumer.py` | `InboundConsumer` — worker pool, per-chat serial locks, session deduplication, timeout handling |
| `retry.py` | `RetryConfig` — exponential backoff retry with per-channel presets (attempts, min/max delay, jitter) |
| `formatter.py` | `UnifiedFormatter` — Markdown to platform-specific format conversion (HTML, Slack mrkdwn, Discord, plain text) |
| `standalone.py` | Headless channel runner (`run_standalone`) for running channels without the CLI |

## Message Processing Pipeline

### Inbound (User Message → Agent)

```
1. Platform SDK/Webhook receives raw message
       │
2. Channel._on_message() parses into RawIncoming
       │
3. Channel._enqueue_raw() runs middleware pipeline:
   ├── DedupMiddleware      — drop duplicates (LRU cache, 60s TTL)
   ├── AllowListMiddleware  — enforce sender/channel restrictions
   ├── PairingMiddleware    — handle DM pairing flow (if dm_policy="pairing")
   ├── GroupHistoryMiddleware — buffer group context for injection
   └── MentionGatingMiddleware — filter by @mention policy in groups
       │
4. InboundMessage queued on Channel._queue
       │
5. Channel.run() → receive() → queue_message() with debounce
       │
       │  (500ms debounce window: rapid messages from same sender merged)
       │
6. MessageBus.publish_inbound()
       │
7. InboundConsumer acquires per-chat lock → invokes Agent
       │
8. Agent response → OutboundMessage → MessageBus.publish_outbound()
```

### Outbound (Agent Response → User)

```
1. OutboundMessage arrives on MessageBus outbound queue
       │
2. Dispatcher routes to origin channel by name
       │
3. Channel.send() processes the response:
   ├── Stop typing indicator
   ├── Format text (Markdown → platform format)
   ├── Chunk text to platform limit (code-block-aware splitting)
   ├── Send each chunk via _send_chunk() with format fallback
   ├── Send media attachments via _send_media_impl()
   └── Retry on transient errors (exponential backoff)
```

### Text Chunking

Long responses are split intelligently with this priority:

1. Markdown code block fence boundaries
2. Double newlines (paragraph breaks)
3. Single newlines
4. Space characters
5. Hard cut at limit (last resort)

Code blocks are never split mid-block when possible. Each chunk is sent as a separate message.

## Middleware Pipeline

Middleware runs sequentially on each inbound message. Each middleware can pass, modify, or drop the message.

### 1. DedupMiddleware

Prevents duplicate message processing using a bounded LRU cache with TTL.

- Cache size: 1000 entries (configurable)
- TTL: 60 seconds
- Key: `message_id` from the platform
- Messages with the same ID within the TTL window are silently dropped

### 2. AllowListMiddleware

Enforces sender and channel restrictions based on the `dm_policy` config.

| Policy | Behavior |
|--------|----------|
| `"open"` | Accept messages from anyone |
| `"allowlist"` | Only accept from `allowed_senders` / `allowed_channels` |
| `"pairing"` | Require DM pairing before accepting (see PairingMiddleware) |

When `allowed_senders` is set (non-empty), only messages from listed sender IDs pass through. Same for `allowed_channels`.

### 3. PairingMiddleware

Handles an interactive DM pairing flow for the `"pairing"` dm_policy.

- First message from an unknown sender triggers a pairing request
- The sender must provide a valid pairing code
- Once paired, the sender is added to the allowlist for future messages

### 4. GroupHistoryMiddleware

Buffers recent group chat messages to provide conversation context.

- Only active when `capabilities.groups = True`
- Maintains a per-chat rolling buffer (default: 50 messages, 5-minute max age)
- When the bot is mentioned in a group, recent history is injected into the message metadata so the agent can see prior context
- Non-mentioned group messages are buffered but not forwarded (see MentionGating)

### 5. MentionGatingMiddleware

Controls whether the bot responds in group chats.

| `require_mention` | Behavior |
|-------------------|----------|
| `True` / `"group"` | Only respond to @mentions in groups; always respond in DMs |
| `False` / `"none"` | Respond to all messages in all contexts |
| `"always"` | Require @mention even in DMs |

Default: `"group"` — the bot ignores group messages unless explicitly @mentioned.

Mention detection is platform-specific:
- **Telegram**: checks for `@bot_username` in text
- **Discord**: checks `message.mentions` for bot user
- **Slack**: handled via separate `app_mention` event type
- **Feishu**: checks `mentions` array in event payload
- **DingTalk**: checks `isInAtList` flag or `atUsers` array
- **WeChat (WeCom)**: checks `AtUserList` XML field

## Capability Matrix

| Channel | Format | Max Len | Media | Voice | Sticker | Location | Video | Typing | Reaction | Thread | Group | @Mention | No Public IP | Token Refresh | Proxy | Allowlist |
|:--------|:------:|:-------:|:-----:|:-----:|:-------:|:--------:|:-----:|:------:|:--------:|:------:|:-----:|:--------:|:------------:|:-------------:|:-----:|:---------:|
| Telegram | HTML | 4000 | S/R | R | R | R | | 4s | emoji | | G | @ | yes | | yes | yes |
| Discord | Discord | 2000 | S/R | | | | | 8s | emoji | yes | G | @ | yes | | yes | yes |
| Slack | Mrkdwn | 4000 | S/R | | | | | post | emoji | yes | G | @ | yes | | yes | yes |
| Feishu      | Post | 4096 | S/R | R | R | | | | emoji | | G | @ | ws mode | 2h | yes | yes |
| WeChat | MD | 4096 | S/R | R | | R | R | recall | | | G | @ | no | 2h | yes | yes |
| DingTalk | MD | 4096 | S/R | R | | | R | | | | G | @ | yes | 2h | yes | yes |
| QQ | Plain | 4096 | S/R | | | | | | | | G | @ | yes | | | yes |
| Signal | Plain | 4096 | S/R | R | | | | api | emoji | | G | UUID | yes | | | yes |
| iMessage | Plain | - | S/R | R | | | | | | | G | | yes | | | yes |
| Email | HTML | - | S/R | | | | | | | | | | yes | | | yes |

Legend: **S** = send, **R** = receive, **G** = group chat, **@** = @mention detection, **-** = no practical limit

### Connection Types

| Channel     | Transport | Connection Mode                        | Default Port |
|-------------|-----------|----------------------------------------|:------------:|
| Telegram    | HTTPS     | Long polling (`getUpdates`)            | --            |
| Discord     | WebSocket | Gateway events (`discord.py`)          | --            |
| Slack       | WebSocket | Socket Mode (`slack-sdk`)              | --            |
| Feishu      | HTTP/WS   | Webhook or WebSocket long connection    | 9000/--      |
| WeChat      | HTTP      | Webhook `POST /wechat/callback`        | 9001         |
| DingTalk    | WebSocket | Stream Mode (DingTalk gateway)         | --            |
| QQ          | WebSocket | Bot Gateway (`qq-botpy`)               | --            |
| Signal      | TCP       | JSON-RPC (`signal-cli` daemon)         | 7583         |
| iMessage    | stdio     | JSON-RPC (`imsg` CLI)                  | --            |
| Email       | TCP       | IMAP polling + SMTP send               | 993/587      |

> **"--"** means no listening port is required -- no public IP or port forwarding needed.

### Format Conversion

The `UnifiedFormatter` converts Markdown output from the agent into platform-native formats:

| Target Format | Conversion |
|:-------------|:-----------|
| HTML (Telegram, Email) | `**bold**` → `<b>bold</b>`, `` `code` `` → `<code>code</code>`, code blocks → `<pre>`, special chars escaped |
| Slack mrkdwn | `**bold**` → `*bold*`, `_italic_` → `_italic_`, code blocks preserved, `<>&` escaped |
| Discord Markdown | Mostly passthrough, minor adjustments for Discord-specific rendering |
| Feishu Post | Markdown → Feishu rich text JSON (code blocks, bold, italic, strikethrough, links, headings, quotes, lists) |
| Plain text | All formatting stripped, structure preserved via indentation |

## Security and Access Control

### Sender Allowlist

Every channel supports `allowed_senders` to restrict who can interact with the bot:

```yaml
telegram_allowed_senders: "123456789,987654321"     # Telegram user IDs
discord_allowed_senders: "111222333444555666"        # Discord user IDs
slack_allowed_senders: "U0123ABCDEF"                # Slack Member IDs
feishu_allowed_senders: "ou_xxxxxxxxxxxx"            # Feishu open_ids
signal_allowed_senders: "+1234567890"                # Phone numbers
email_allowed_senders: "alice@example.com"           # Email addresses
imessage_allowed_senders: "+1234567890,user@icloud.com"  # Phone or email
```

When `allowed_senders` is empty, the channel accepts messages from anyone. **For production deployments, always set an allowlist.**

### Channel Allowlist

For platforms with multiple channels/groups (Discord, Slack), restrict which channels the bot operates in:

```yaml
discord_allowed_channels: "111222333444555666,777888999000111222"
slack_allowed_channels: "C0123ABCDEF,C0456GHIJKL"
```

### Token and Secret Handling

- API tokens are stored in the config file or environment variables, never logged at INFO level
- Discord logs only the first 8 and last 4 characters of the bot token for debugging
- WeChat/Feishu tokens are auto-refreshed before expiry (5-minute margin on 2-hour TTL)
- Webhook signature verification is enforced when `token`/`encoding_aes_key` is configured (WeChat, Feishu)

### Group Chat Behavior

By default, the bot only responds in group chats when explicitly @mentioned. This prevents the bot from responding to every message in a busy group. Configure via:

```yaml
# Default: only respond when mentioned in groups
channel_require_mention: "group"

# Respond to all messages (including groups)
channel_require_mention: "none"
```

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

### Built-in slash command: `/gpu`

All channels that run through the shared `MessageBus` support a built-in
`/gpu` command. If a user sends the exact message `/gpu`, the message is
handled before the normal agent pipeline:

- EvoScientist runs `nvidia-smi` on the host machine
- the raw command output is returned directly to the originating channel
- the normal agent / HITL / ask_user flow is skipped for that message

This works across the standard channel integrations, including Telegram,
Discord, Slack, Feishu, WeChat, DingTalk, QQ, Signal, Email, and iMessage.
The host must have `nvidia-smi` available.

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

---

## Channel Deployment Guides

---

### Telegram

**Install:** `pip install evoscientist[telegram]`

**Prerequisites:**

1. Search for [@BotFather](https://t.me/BotFather) in Telegram, send `/newbot`, and follow the prompts to create a bot.
2. BotFather will return a Bot Token (format: `123456789:ABCdefGHI...`) -- save it securely.
3. Get your user ID: send any message to [@userinfobot](https://t.me/userinfobot), it will reply with your numeric ID.
4. (Optional) For group use: add the bot to a group, then in BotFather send `/setprivacy` -> `Disable` so the bot can read group messages.

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

**Technical details:** Long polling mode, `drop_pending_updates=True` on startup to skip backlog. Markdown to Telegram HTML auto-conversion (bold, italic, strikethrough, links, code blocks, headings, lists). Falls back to plain text on HTML parse failure. Media routed by extension to `send_photo`/`send_video`/`send_audio`/`send_document`. In groups, only responds when @mentioned; auto-strips @mention. Typing indicator refreshes every 4s. ACK reaction (eyes emoji) on message receipt, removed after reply. Retry: 3 attempts, min delay 0.4s, parse errors not retried. Text chunk limit: 4000 chars.

---

### Discord

**Install:** `pip install evoscientist[discord]`

**Prerequisites:**

1. Go to [Discord Developer Portal](https://discord.com/developers/applications) -> New Application -> enter a name.
2. Left menu **Bot** -> Reset Token -> copy the Bot Token.
3. Under **Privileged Gateway Intents**, enable **Message Content Intent** (required to read message content).
4. Left menu **OAuth2** -> URL Generator:
   - Scopes: check `bot`
   - Bot Permissions: check `Send Messages`, `Read Message History`, `Attach Files`, `Add Reactions`
   - Copy the generated URL, open in browser, select a server to invite the bot.
5. Get user ID: Discord Settings -> Advanced -> enable Developer Mode -> right-click username -> Copy User ID.

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

**Technical details:** WebSocket Gateway (`discord.py`). In server channels, only responds when @mentioned; DMs respond directly. Thread-aware: messages in threads are tracked with `parent_channel_id` and `thread_id`. Replies via `MessageReference`. Message cache (200 entries) for ACK emoji reactions. Attachment download (max 20 MB) with safe filename sanitization. Media sent via `discord.File`. Typing indicator refreshes every 8s. Retry: 3 attempts, parses `Retry-After` header for 429s. Text chunk limit: 2000 chars.

---

### Slack

**Install:** `pip install evoscientist[slack]`

**Prerequisites:**

1. Go to [Slack API](https://api.slack.com/apps) -> Create New App -> From scratch -> select workspace.
2. Left menu **Socket Mode** -> enable -> Generate App-Level Token, scope `connections:write` -> copy App Token (`xapp-...`).
3. Left menu **OAuth & Permissions** -> add Bot Token Scopes:
   - `chat:write`, `channels:history`, `groups:history`, `im:history`, `files:read`, `files:write`, `reactions:write`
4. Click **Install to Workspace** -> copy Bot User OAuth Token (`xoxb-...`).
5. Left menu **Event Subscriptions** -> enable -> Subscribe to bot events: `message.channels`, `message.groups`, `message.im`, `app_mention`.
6. Get Member ID: click user avatar -> profile -> **...** -> Copy member ID.

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

**Technical details:** Socket Mode (no public URL needed). Markdown to mrkdwn conversion. DMs respond directly; channels respond to `app_mention` events. Thread replies via `thread_ts` -- all replies are threaded to the original message. Typing indicator approximated by posting/deleting a "..." message (Slack has no bot typing API). ACK reaction (eyes emoji) on message receipt. Attachments downloaded with Bearer auth. Media sent via `files_upload_v2`. Runs `auth_test()` on startup to verify credentials and cache bot user ID. Retry: 3 attempts, exponential backoff + jitter. Text chunk limit: 4000 chars.

---

### Feishu (Lark)

**Install:** `pip install evoscientist[feishu]`

**Prerequisites:**

1. Go to [Feishu Open Platform](https://open.feishu.cn/app) (international: [Lark Developer](https://open.larksuite.com/app)) -> create a custom app.
2. Copy the **App ID** and **App Secret**.
3. Left menu **Event Subscriptions**:
   - **Webhook mode**: set request URL to `http://your-host:9000/webhook/event` -> copy **Verification Token** and **Encrypt Key**.
   - **WebSocket mode**: select **长连接** (Long Connection) as the subscription method. No URL needed.
4. Add event: `im.message.receive_v1` (receive messages).
5. Left menu **Permissions** -> enable `im:message:send_as_bot`.
6. Create a version and publish.

> **Webhook mode** requires a publicly reachable URL. For local dev, use `ngrok http 9000` or [natapp](https://natapp.cn/) (recommended for China).

#### Subscription Modes

Feishu supports two subscription modes:

| Mode | Transport | Public IP? | Best For |
|------|-----------|:----------:|----------|
| `webhook` (default) | HTTP POST callback | Yes | Servers with public IP / cloud deployment |
| `websocket` | WebSocket long connection | **No** | Local dev, behind NAT/firewall, China (no ngrok needed) |

**WebSocket mode** uses the official `lark-oapi` SDK to maintain an outbound WebSocket connection to Feishu servers. No public IP, port forwarding, or tunnel is required.

To use WebSocket mode:

```bash
# Install the SDK
pip install 'evoscientist[feishu]'

# Via config file
feishu_subscription_mode: "websocket"

# Via CLI (standalone)
python -m EvoScientist.channels.feishu.serve --app-id ID --app-secret SECRET --mode websocket
```

> **Note:** In WebSocket mode, `feishu_verification_token`, `feishu_encrypt_key`, and `feishu_webhook_port` are not used — the SDK handles authentication and encryption internally.

**Configuration:**

```yaml
channel_enabled: "feishu"
feishu_app_id: "cli_xxxxxxx"
feishu_app_secret: "xxxxxxxxxxxxxxxxxx"
feishu_subscription_mode: "webhook"   # or "websocket"
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
| `feishu_subscription_mode` | `str` | `"webhook"` | `"webhook"` or `"websocket"` (WebSocket long connection, no public IP) |

**Env vars:** `EVOSCIENTIST_FEISHU_APP_ID`, `EVOSCIENTIST_FEISHU_APP_SECRET`, `EVOSCIENTIST_FEISHU_WEBHOOK_PORT`, `EVOSCIENTIST_FEISHU_DOMAIN`

**Technical details:** Webhook on `POST /webhook/event` with URL verification challenge-response. Supports both v1 (legacy) and v2 event schemas. Optional AES-256-CBC event decryption (when `encrypt_key` configured). `tenant_access_token` auto-refresh (2h TTL, refreshes 5 min before expiry). Markdown to Feishu Post rich text conversion (code blocks, bold, italic, strikethrough, links, headings, quotes, ordered/unordered lists). Plain text fallback. Group @mention filtering with mention key caching. Media: images via `/im/v1/images`, files via `/im/v1/files`. Replies via `/messages/{id}/reply` API. ACK reaction via `/messages/{id}/reactions`. Retry: 3 attempts, rate limit delay 2.0s, matches `99991400`/`rate limit`. Non-retryable: permission denied (`99991401`), invalid credentials. Text chunk limit: 4096 chars.

---

### WeChat

**Install:** `pip install evoscientist[wechat]`

Two backends supported: **WeCom** (recommended, free, no certification needed) and **WeChat Official Account** (requires verified service account).

#### WeCom

**Prerequisites:**

1. Log in to [WeCom Admin Console](https://work.weixin.qq.com) -> App Management -> create a custom app.
2. Copy the **Corp ID**, **AgentId**, and **Secret**.
3. In app details -> Receive Messages -> Set API Receive -> URL: `http://your-host:9001/wechat/callback` -> copy **Token** and **EncodingAESKey**.
4. In app details -> **Trusted IP** -> add your server's public IP address. Without this, all API calls will fail with error `60020`.

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

1. Log in to [WeChat Official Account Platform](https://mp.weixin.qq.com) -> Settings & Development -> Basic Configuration.
2. Copy the **AppID** and **AppSecret**.
3. Server Configuration -> URL: `http://your-host:9001/wechat/callback` -> set **Token** and **EncodingAESKey**.

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

**Technical details:** Webhook HTTP server for inbound (XML message parsing). GET callback for URL verification (SHA1 signature check). POST callback for message handling -- returns `"success"` within 5s and processes asynchronously. Optional AES encryption/decryption via `WeChatCrypto`. `access_token` auto-refresh (2h TTL, 5-min margin). Token-expired errors (40014, 42001) trigger automatic retry with refreshed token. WeCom supports Markdown message format with plain text fallback; Official Account uses plain text only (customer service API). WeCom group messages sent via `/appchat/send` endpoint (group IDs start with `wr`). Typing indicator approximated by posting/recalling a "..." message (WeCom only). Supports text, image, voice, video, location, file, and link message types. Media upload via `/media/upload`. Text chunk limit: 4096 chars.

---

### DingTalk

**Install:** `pip install evoscientist[dingtalk]`

**Prerequisites:**

1. Go to [DingTalk Open Platform](https://open-dev.dingtalk.com) -> App Development -> create a bot app.
2. Copy the **AppKey** (Client ID) and **AppSecret** (Client Secret).
3. Enable **Stream Mode** in the app configuration -- no public IP needed.
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

**Technical details:** Stream Mode via WebSocket -- connects to DingTalk gateway (`/v1.0/gateway/connections/open`) with automatic ticket-based auth. Ping/pong heartbeat with system topic handling. Message ACK via JSON response. `accessToken` auto-refresh. Sends via robot `oToMessages/batchSend` API in Markdown format (`sampleMarkdown`). Image uploads via `/media/upload` API with `sampleImageMsg`. Group @mention detection via `isInAtList` flag with `atUsers` array fallback. `downloadCode` resolution via `/robot/messageFiles/download` API for file/image/video/audio attachments. Auth errors (`invalidauthentication`/`forbidden`/`40014`) not retried. Text chunk limit: 4096 chars.

---

### QQ

**Install:** `pip install evoscientist[qq]`

**Prerequisites:**

1. Go to [QQ Open Platform](https://q.qq.com) -> create a bot application.
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
2. **Gmail:** Enable 2FA -> generate an App Password. IMAP: `imap.gmail.com:993` (SSL), SMTP: `smtp.gmail.com:587` (STARTTLS).
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

**Technical details:** IMAP polling mode, checks for UNSEEN emails periodically (max 20 per cycle). Supports SSL and STARTTLS. Auto-parses multipart emails (prefers text/plain, falls back text/html -> plain text). Attachments auto-downloaded. Replies set `In-Reply-To` and `References` headers to maintain email threads. Sends HTML + plain text dual format (multipart/alternative), falls back to plain text on HTML failure. IMAP auto-reconnects on disconnect. Auth errors (auth/login/credential) not retried. No public IP needed. Text chunk limit: no limit.

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

---

## Running Multiple Channels

Comma-separate channel names in the config to enable multiple channels simultaneously:

```yaml
channel_enabled: "telegram,discord,slack"
```

All enabled channels run concurrently via the internal `MessageBus`. Each channel:
- Has its own connection lifecycle (connect, reconnect, health check)
- Shares the same `InboundConsumer` worker pool and agent instance
- Routes outbound replies back to the originating channel automatically
- Shares the same built-in commands such as `/gpu`

### Multi-Channel Architecture

```
                    ChannelManager
                    ├── TelegramChannel  ──┐
                    ├── DiscordChannel   ──┤
                    ├── SlackChannel     ──┤──► MessageBus ──► InboundConsumer ──► Agent
                    ├── FeishuChannel    ──┤                         │
                    └── ...              ──┘                         ▼
                                                              OutboundMessage
                                                                    │
                                                              Dispatcher routes
                                                              to origin channel
```

### Health Monitoring

The `ChannelManager` runs a background health check task that monitors all active channels. Access health status via:

```bash
# CLI
EvoSci channel status

# HTTP (if health endpoint is enabled)
curl http://localhost:8080/healthz
```

## Docker Deployment

For webhook-based channels (Feishu, WeChat), Docker simplifies port mapping and process management:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install evoscientist[feishu,wechat]

# Expose webhook ports
EXPOSE 9000 9001

CMD ["EvoSci", "serve"]
```

```bash
docker build -t evoscientist .
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  -e EVOSCIENTIST_CHANNEL_ENABLED="feishu,wechat" \
  -e EVOSCIENTIST_FEISHU_APP_ID="cli_xxx" \
  -e EVOSCIENTIST_FEISHU_APP_SECRET="xxx" \
  -e EVOSCIENTIST_WECHAT_BACKEND="wecom" \
  -e EVOSCIENTIST_WECHAT_WECOM_CORP_ID="ww..." \
  -e EVOSCIENTIST_WECHAT_WECOM_SECRET="xxx" \
  evoscientist
```

For polling/WebSocket channels (Telegram, Discord, Slack, DingTalk, QQ), no port mapping is needed:

```bash
docker run -d \
  -e EVOSCIENTIST_CHANNEL_ENABLED="telegram" \
  -e EVOSCIENTIST_TELEGRAM_BOT_TOKEN="123456:ABC-xxx" \
  evoscientist
```

## Troubleshooting

### Common Issues

**Bot not responding to messages**

1. Check that `channel_enabled` includes your channel name
2. Verify the bot token/credentials are correct: `EvoSci config get telegram_bot_token`
3. If using `allowed_senders`, ensure your user ID is listed
4. For group chats, ensure the bot is @mentioned (default behavior)
5. Check logs for middleware drops: `DedupMiddleware`, `AllowListMiddleware`, or `MentionGatingMiddleware`

**"channel X not found" or import errors**

Install the channel-specific dependencies:
```bash
pip install evoscientist[telegram]  # or discord, slack, feishu, etc.
```

**Webhook channels (Feishu, WeChat) not receiving messages**

1. Ensure the webhook URL is publicly reachable (not behind NAT without port forwarding)
2. For local development, use a tunnel: `ngrok http 9000` or [natapp](https://natapp.cn/) for China
3. Verify the callback URL matches exactly (including path: `/webhook/event` for Feishu, `/wechat/callback` for WeChat)
4. Check that signature verification tokens match between the platform config and your local config

**WeChat API error `60020`**

Add your server's public IP to the WeCom app's **Trusted IP** list in the admin console.

**Token refresh failures**

- Feishu/WeChat/DingTalk tokens auto-refresh with a 5-minute safety margin before expiry
- If the refresh endpoint is unreachable (network issues), messages will fail until the next successful refresh
- Check proxy settings if your server requires a proxy to reach external APIs

**Duplicate responses**

- The `DedupMiddleware` prevents most duplicates using a 60-second LRU cache
- If you see duplicates, check if the platform is sending the same message with different IDs (some platforms retry delivery)

**Messages truncated**

- Each platform has a max text length (see Capability Matrix)
- Long responses are automatically chunked at paragraph/code-block boundaries
- Adjust `text_chunk_limit` in the channel config if needed

### Debug Logging

Enable debug logs for the channel subsystem:

```bash
export EVOSCIENTIST_LOG_LEVEL=DEBUG
# or
EvoSci config set log_level debug
```

Channel-specific log output is prefixed with the module path (e.g., `EvoScientist.channels.telegram.channel`).
