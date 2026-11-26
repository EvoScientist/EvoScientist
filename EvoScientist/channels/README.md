# Channels

Connect EvoScientist to messaging platforms so they share the same agent core, memory, and workspace as the CLI. All settings live in `~/.config/evoscientist/config.yaml`; set them with the `EvoSci channel setup` wizard or `EvoSci config set <key> <value>`.

## Quick Start

1. **Configure with the wizard:**
   ```bash
   EvoSci channel setup   # pick platforms, enter credentials (or `EvoSci onboard`)
   ```
   The wizard offers to install any missing platform packages into EvoScientist's own environment. To configure by hand instead:
   ```bash
   EvoSci config set channel_enabled telegram
   EvoSci config set telegram_bot_token "123456:ABC-xxx"
   EvoSci config set telegram_allowed_senders "YOUR_USER_ID"
   ```
   *(Working from a source checkout? Install a platform's extra with `uv sync --extra telegram`, or `--extra all-channels`.)*

2. **Run:**
   ```bash
   EvoSci serve   # Headless mode: agent + all enabled channels
   ```
   You can also start channels inside an interactive CLI session via `/channel`, or run multiple platforms concurrently with `channel_enabled: "telegram,discord,slack"`.

---

## How Messages Are Handled

- **Deduplication & debounce:** Duplicate platform message IDs are ignored, and rapid consecutive messages from the same sender are merged before reaching the agent.
- **Access control (`*_allowed_senders`, `*_allowed_channels`):** Empty allows everyone—**always set an allowlist on any persistent deployment.** Tokens are never logged at `INFO` level, and webhook channels verify request signatures when configured.
- **Direct messages (`dm_policy`):** `allowlist` (default) applies `*_allowed_senders` to DMs too. `open` lets anyone DM the bot even when an allowlist is set; group messages are still checked.
- **Group mention gating (`require_mention`):** With `group` (default), the bot replies in group chats only when `@mentioned`; DMs need no mention. `always` requires a mention in DMs too; `off` replies to every message. Recent group messages are buffered so the agent has context when mentioned.
- **Formatting & chunking:** Outbound Markdown is converted to each platform's native format, split cleanly without breaking code blocks when possible, and retried on transient network errors.
- **Voice messages:** Platforms marked *Voice R* below are transcribed to text when the `stt` extra is installed and `stt_enabled` is `true`.
- **Webhook port:** Webhook channels (Feishu webhook mode, WeChat WeCom / Official Account) share one HTTP server on `shared_webhook_port` (default `9000`). Set it to `0` to give each channel its own `feishu_webhook_port` / `wechat_webhook_port`.

---

## Platform Matrix

| Channel | Extra needed | Transport | Public IP? | Format | Media / Voice |
|---|---|---|:-:|---|---|
| **Telegram** | `telegram` | Long polling | No | HTML | S/R · Voice R |
| **Discord** | `discord` | WebSocket (`discord.py`) | No | Discord MD | S/R (threads) |
| **Slack** | `slack` | Socket Mode WS | No | mrkdwn | S/R (threads) |
| **Feishu (Lark)** | `feishu` | Webhook (`:9000`) *or* WebSocket | Webhook: Yes · WS: No | Post JSON | S/R · Voice R |
| **WeChat** | `wechat` | Personal QR / WeCom / MP Webhook (`:9000`) | Personal: No · Webhook: Yes | MD / Plain | S/R · Voice R |
| **DingTalk** | *(none)* | Stream Mode WS | No | MD | S/R · Voice R |
| **QQ** | `qq` | Bot Gateway WS | No | MD / Plain | S/R |
| **Signal** | *(needs `signal-cli`)* | TCP JSON-RPC (`:7583`) | No | Plain | S/R · Voice R |
| **Email** | *(none)* | IMAP + SMTP | No | HTML + Plain | S/R |
| **iMessage** | *(macOS `imsg` CLI)* | stdio JSON-RPC | No | Plain | S/R · Voice R |

*(S = send, R = receive. Telegram, Discord, Slack, Feishu, WeChat, and DingTalk also support `*_proxy`.)*

---

## Per-Channel Setup

Each block below shows the `config.yaml` keys. `EvoSci channel setup` asks for each platform's main credentials and allowed senders; set the other keys with `EvoSci config set`, such as `*_allowed_channels`, `*_proxy`, and the WeCom / Official Account Token and EncodingAESKey (`wechat_wecom_token` / `wechat_wecom_encoding_aes_key`, `wechat_mp_token` / `wechat_mp_encoding_aes_key`).

### Telegram
- **Extra:** `telegram`
- **Setup:** Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) and get your numeric ID from [@userinfobot](https://t.me/userinfobot). For group chats, send `/setprivacy` → `Disable` in BotFather.
  ```yaml
  channel_enabled: "telegram"
  telegram_bot_token: "123456789:ABC..."
  telegram_allowed_senders: "123456789"
  telegram_proxy: ""
  ```

### Discord
- **Extra:** `discord`
- **Setup:** In [Discord Developer Portal](https://discord.com/developers/applications), create an app → **Bot** → copy token and enable **Message Content Intent** → **OAuth2 URL Generator** (`bot`; `Send Messages`, `Read Message History`, `Attach Files`, `Add Reactions`) → invite to your server. Enable Developer Mode in Discord to copy User/Channel IDs.
  ```yaml
  channel_enabled: "discord"
  discord_bot_token: "MTIz..."
  discord_allowed_senders: ""
  discord_allowed_channels: ""
  discord_proxy: ""
  ```

### Slack
- **Extra:** `slack`
- **Setup:** At [Slack API](https://api.slack.com/apps), create an app → enable **Socket Mode** and generate an App Token (`xapp-...`, `connections:write`) → under **OAuth & Permissions** add bot scopes (`chat:write`, `channels:history`, `groups:history`, `im:history`, `files:read`, `files:write`, `reactions:write`) → install to workspace (`xoxb-...`) → under **Event Subscriptions** subscribe to `message.channels`, `message.groups`, `message.im`, `app_mention`.
  ```yaml
  channel_enabled: "slack"
  slack_bot_token: "xoxb-..."
  slack_app_token: "xapp-1-..."
  slack_allowed_senders: ""    # Member ID (U...)
  slack_allowed_channels: ""   # Channel ID (C...)
  slack_proxy: ""
  ```

### Feishu (Lark)
- **Extra:** `feishu`
- **Setup:** Create a custom app on [Feishu Open Platform](https://open.feishu.cn/app) (or [Lark](https://open.larksuite.com/app)), copy **App ID** and **App Secret**, enable event `im.message.receive_v1` and permission `im:message:send_as_bot`, and publish. Pick a subscription mode (the default is `webhook`):
  - **WebSocket mode (`feishu_subscription_mode: "websocket"`):** Select **Long Connection (长连接)** in Event Subscriptions—no public IP or tunnel needed. Easiest for most setups.
  - **Webhook mode (`webhook`, default):** Point the callback URL to `http://<host>:9000/webhook/event` (use `ngrok`/`natapp` locally) and set `feishu_verification_token` / `feishu_encrypt_key`.
  ```yaml
  channel_enabled: "feishu"
  feishu_app_id: "cli_..."
  feishu_app_secret: "..."
  feishu_subscription_mode: "websocket"   # default: "webhook"
  feishu_verification_token: ""           # webhook mode only
  feishu_encrypt_key: ""                  # webhook mode only (optional)
  feishu_domain: "https://open.feishu.cn" # https://open.larksuite.com for Lark
  feishu_allowed_senders: ""              # open_id
  feishu_proxy: ""
  ```

### WeChat (Personal, WeCom, or Official Account)
- **Extra:** `wechat`
- **Personal WeChat (`wechat_backend: "personal"`):** Sign in by scanning a QR code (iLink Bot API—no public IP needed). `EvoSci channel setup` shows the QR code and saves the account for you (its manual option takes an `account_id` from an earlier login). From a source checkout you can also log in by hand, then put the `account_id` it prints into config:
  ```bash
  uv run python -m EvoScientist.channels.wechat.serve --qr-login
  ```
  ```yaml
  channel_enabled: "wechat"
  wechat_backend: "personal"
  wechat_personal_account_id: "..."          # printed by --qr-login
  wechat_personal_dm_policy: "open"          # open | allowlist
  wechat_personal_group_policy: "disabled"   # disabled | allowlist | open
  wechat_personal_group_allowed: ""
  wechat_allowed_senders: ""
  ```
- **WeCom (`wechat_backend: "wecom"`):** In the [WeCom Admin Console](https://work.weixin.qq.com) → **App Management**, create a custom app and copy the **Corp ID**, **AgentId**, and **Secret**. In the app's **Receive Messages → Set API Receive**, set the URL to `http://<host>:9000/wechat/callback` and copy the **Token** and **EncodingAESKey**. **Add your server's public IP to Trusted IP** (otherwise calls fail with error `60020`).
  ```yaml
  channel_enabled: "wechat"
  wechat_backend: "wecom"
  wechat_wecom_corp_id: "ww..."
  wechat_wecom_agent_id: "1000002"
  wechat_wecom_secret: "..."
  wechat_wecom_token: "..."
  wechat_wecom_encoding_aes_key: "..."
  wechat_allowed_senders: ""
  wechat_proxy: ""
  ```
- **Official Account (`wechat_backend: "wechatmp"`):** Requires a verified service account (plain-text replies only). On [mp.weixin.qq.com](https://mp.weixin.qq.com) → **Settings & Development → Basic Configuration**, copy the AppID and AppSecret into `wechat_mp_app_id` / `wechat_mp_app_secret`. Under **Server Configuration**, set the URL to `http://<host>:9000/wechat/callback` and choose a Token and EncodingAESKey for `wechat_mp_token` / `wechat_mp_encoding_aes_key`.

### DingTalk
- **Extra:** None.
- **Setup:** Create a bot app at [DingTalk Open Platform](https://open-dev.dingtalk.com), copy **AppKey** (`client_id`) and **AppSecret**, and enable **Stream Mode** (no public IP needed).
  ```yaml
  channel_enabled: "dingtalk"
  dingtalk_client_id: "ding..."
  dingtalk_client_secret: "..."
  dingtalk_allowed_senders: ""
  dingtalk_proxy: ""
  ```

### QQ
- **Extra:** `qq`
- **Setup:** Create a bot on [QQ Open Platform](https://q.qq.com), copy **AppID** and **AppSecret**, and add it as a friend or to a group.
  ```yaml
  channel_enabled: "qq"
  qq_app_id: "..."
  qq_app_secret: "..."
  qq_allowed_senders: ""
  ```

### Signal
- **Extra:** None; install [`signal-cli`](https://github.com/AsamK/signal-cli#installation) separately.
- **Setup:** Register (`signal-cli -u +1234567890 register` / `verify`) or link a device (`signal-cli link -n "EvoScientist"`). EvoScientist auto-starts the `signal-cli` daemon if needed.
  ```yaml
  channel_enabled: "signal"
  signal_phone_number: "+1234567890"   # E.164 format
  signal_cli_path: "signal-cli"
  signal_config_dir: ""
  signal_allowed_senders: ""
  signal_rpc_port: 7583
  ```

### Email (IMAP + SMTP)
- **Extra:** None.
- **Setup:** Use an account with IMAP enabled and an app password.
  - **Gmail:** app passwords require 2-Step Verification. IMAP `imap.gmail.com:993`, SMTP `smtp.gmail.com:587`.
  - **Outlook / Office 365:** IMAP `outlook.office365.com:993`, SMTP `smtp.office365.com:587`.
  - **SMTP encryption:** `email_smtp_use_tls: true` uses STARTTLS (usually port `587`); `false` uses implicit SSL (usually port `465`)—there is no unencrypted mode.
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
  email_smtp_use_tls: true              # STARTTLS; false = implicit SSL (465)
  email_from_address: "bot@gmail.com"   # defaults to smtp_username
  email_poll_interval: 30               # seconds
  email_mark_seen: true
  email_max_body_chars: 12000
  email_subject_prefix: "Re: "
  email_allowed_senders: ""
  ```

### iMessage *(macOS only)*
- **Extra:** None; requires [`imsg`](https://github.com/steipete/imsg) (`brew install steipete/tap/imsg`), signed-in Messages.app, and **Full Disk Access** for your terminal. Not usable inside Docker.
  ```yaml
  channel_enabled: "imessage"
  imessage_allowed_senders: "+1234567890,user@icloud.com"  # phone, email, chat_id:123, chat_guid:..., or *
  ```

---

## Docker Quick Reference

The image keeps `config.yaml` inside its data volume, so configure channels by running `config set` against that volume, then start `serve` with the same volume:

```bash
IMAGE=ghcr.io/evoscientist/evoscientist:latest
VOL=evosci-data:/home/evosci/.evoscientist

docker run --rm -v $VOL $IMAGE config set channel_enabled telegram
docker run --rm -v $VOL $IMAGE config set telegram_bot_token "123456:ABC-xxx"
docker run --rm -v $VOL $IMAGE config set telegram_allowed_senders "YOUR_USER_ID"

docker run --rm -v $VOL $IMAGE setup   # once
mkdir -p workspace   # create first on Linux so Docker doesn't make it root-owned
docker run -d --env-file .env -v "$(pwd)/workspace:/workspace" -v $VOL $IMAGE serve
```

Polling and WebSocket channels need no exposed ports. Webhook channels (Feishu webhook mode, WeChat WeCom/MP) also need the shared webhook port published: `docker run -d -p 9000:9000 --env-file .env -v "$(pwd)/workspace:/workspace" -v $VOL $IMAGE serve`. `--env-file .env` passes your API keys (one `NAME=value` per line) from a host file you create first to the container as its environment; leave it out of both commands if the keys are already saved in the data volume. See the [Docker section of the Installation Guide](../../docs/guides/install.md#docker).

## Troubleshooting

- **Bot doesn't reply:** Check `channel_enabled`, verify credentials (`EvoSci config get <key>`), verify your ID is in `*_allowed_senders`, and confirm you `@mentioned` the bot in group chats (or `EvoSci config set require_mention off`).
- **Missing dependency error:** Re-run `EvoSci channel setup` and accept the install prompt (from a source checkout: `uv sync --extra <channel>` or `--extra all-channels`).
- **Webhook not receiving events:** Confirm the shared webhook port (`shared_webhook_port`, default `9000`) is publicly reachable (or tunnelled with `ngrok`/`natapp`), the exact callback path (`/webhook/event` for Feishu, `/wechat/callback` for WeChat), and matching signing tokens.
- **WeCom error `60020`:** Add your server's public IP to the WeCom app's **Trusted IP** list.
- **Debug logs:** Run with `EVOSCIENTIST_LOG_LEVEL=DEBUG` (or `EvoSci config set log_level debug`) and optionally `EvoSci config set channel_debug_tracing true` to see inbound filter decisions.
- **Health check:** While channels run, `curl http://localhost:8080/healthz` returns each channel's success and failure counts and account states (in Docker, run `docker exec <container> curl http://localhost:8080/healthz`, or publish the port with `-p 127.0.0.1:8080:8080`). The endpoint listens on port `8080` on all interfaces; if that port is taken, the log says so and channels still start.
