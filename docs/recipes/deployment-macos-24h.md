# Running EvoScientist 24/7 on macOS (Telegram + Voice STT + Claude OAuth)

Deploy EvoScientist as an always-on background service on an Apple silicon Mac (e.g., Mac Mini) using `launchd`, with Telegram messaging, offline voice transcription (`faster-whisper`), and Claude Pro/Max subscription routing (`ccproxy` managed automatically by `EvoSci server`).

```txt
Telegram (text / voice) ──▶ EvoSci serve (Telegram + faster-whisper STT)
                                │
Anthropic API ◀── ccproxy ◀── EvoSci background server
```

## Prerequisites

- macOS 13+ on Apple silicon, [uv](https://docs.astral.sh/uv), and an active Claude Pro/Max subscription (or an API key)
- A Telegram bot token from [@BotFather](https://t.me/BotFather) and your numeric user ID from [@userinfobot](https://t.me/userinfobot)

## 1. Install EvoScientist With Extras

Always quote brackets in `zsh`:

```bash
uv tool install 'EvoScientist[telegram,stt,oauth]'
which EvoSci && EvoSci --version
```

*(Working from a source checkout instead? Run `uv sync --extra telegram --extra stt --extra oauth` inside your clone.)*

## 2. Authenticate Claude OAuth (Optional — Skip If Using an API Key)

Log in once per machine—`EvoSci server` starts and stops `ccproxy` automatically when `anthropic_auth_mode` is `oauth` (see [Subscription OAuth](subscription-oauth.md)):

```bash
ccproxy auth login claude_api
ccproxy auth status claude_api
```

## 3. Pre-Download the Whisper STT Model

Avoids a ~250 MB download delay on your first voice note:

```bash
uv run --with faster-whisper python -c "from faster_whisper import WhisperModel; WhisperModel('Systran/faster-whisper-small', device='cpu', compute_type='int8'); print('STT model ready')"
```

## 4. Configure & Smoke Test

```bash
EvoSci config set provider anthropic
EvoSci config set anthropic_auth_mode oauth
EvoSci config set channel_enabled telegram
EvoSci config set telegram_bot_token       "YOUR_BOT_TOKEN"
EvoSci config set telegram_allowed_senders "YOUR_TELEGRAM_USER_ID"
EvoSci config set stt_enabled  true
EvoSci config set stt_language auto   # auto | en | zh
EvoSci config set default_workdir "/absolute/path/to/your/workspace"
```

Test interactively first:

```bash
EvoSci serve --auto-approve
```

Send both a text message and a voice note to your Telegram bot. Once both work, stop `EvoSci serve` (`Ctrl+C`) and register `launchd`.

## 5. Register a `launchd` Auto-Start Service

```bash
EVOSCI_BIN=$(which EvoSci)
WORKDIR="/absolute/path/to/your/workspace"   # adjust to your workspace

cat > ~/Library/LaunchAgents/com.evosci.serve.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.evosci.serve</string>
    <key>ProgramArguments</key>
    <array>
        <string>${EVOSCI_BIN}</string>
        <string>serve</string>
        <string>--auto-approve</string>
    </array>
    <key>WorkingDirectory</key><string>${WORKDIR}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/evosci.log</string>
    <key>StandardErrorPath</key><string>/tmp/evosci.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.evosci.serve.plist
launchctl list | grep evosci
```

Watch live logs anytime with `tail -f /tmp/evosci.log`.

## Day-to-Day Management

| Action | Command |
|---|---|
| Restart `EvoSci serve` | `launchctl kickstart -k gui/$(id -u)/com.evosci.serve` |
| Stop `EvoSci serve` | `launchctl unload ~/Library/LaunchAgents/com.evosci.serve.plist` |
| Stop the background agent server (and its managed `ccproxy`) | `EvoSci server stop` *(outlives `EvoSci serve`; run this before restarting after an update)* |
| Reload after editing the plist | `launchctl unload ~/Library/LaunchAgents/com.evosci.serve.plist && launchctl load ~/Library/LaunchAgents/com.evosci.serve.plist` |

## Troubleshooting

- **`zsh: no matches found`:** Quote package extras (`'EvoScientist[telegram,stt,oauth]'`).
- **Claude OAuth token expired/missing:** Run `ccproxy auth login claude_api` and check `ccproxy auth status claude_api`.
- **Voice messages arrive as `[voice: ...]` instead of text:** Confirm `stt_enabled` is `true` and `faster-whisper` is installed in the same environment as `EvoSci`.
- **Agent asks for shell approval in Telegram:** Make sure `--auto-approve` is included in `ProgramArguments`.
- **Workspace on an external `/Volumes/...` drive:** `KeepAlive: true` automatically retries until the drive mounts after reboot.
