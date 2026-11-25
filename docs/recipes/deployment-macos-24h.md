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

> **Set up with an older version of this recipe?** Stop and remove its LaunchAgents first. Otherwise the old bot keeps polling Telegram next to the new one, and the old `com.evosci.ccproxy` job keeps running (`EvoSci server stop` doesn't stop it; the server now manages `ccproxy` itself):
>
> ```bash
> for job in com.evosci.serve com.evosci.ccproxy; do
>   launchctl unload ~/Library/LaunchAgents/$job.plist 2>/dev/null
>   rm -f ~/Library/LaunchAgents/$job.plist
> done
> ```
>
> The old recipe also pointed `anthropic_base_url` at that proxy. If you're switching to an API key, clear it and restart the background server, or requests keep going to the stopped proxy: `EvoSci config set anthropic_base_url "" && EvoSci server stop`. (OAuth mode sets the proxy URL itself, so it doesn't matter there.)

## 1. Install EvoScientist With Extras

Install with the [one-line installer](../guides/install.md):

```bash
curl -fsSL https://evoscientist.ai/install.sh | bash
```

**Open a new terminal** so `EvoSci` is on your `PATH`, then add the extras pinned to the same release's tested dependencies (`EvoSci update` keeps them from then on):

```bash
V=$(EvoSci --version | awk '{print $2}')
uv=$(sed -E 's/.*"path": *"([^"]*)".*/\1/' "${EVOSCIENTIST_DATA_DIR:-$HOME/.evoscientist}/tools/uv.json")   # the installer's own uv
"$uv" tool install --force --python 3.12 --managed-python \
  --with-executables-from ccproxy-api \
  -c "https://raw.githubusercontent.com/EvoScientist/EvoScientist/v$V/constraints.txt" \
  "EvoScientist[telegram,stt,oauth]==$V"
which EvoSci ccproxy && EvoSci --version
```

*(`--with-executables-from` puts `ccproxy` on your `PATH` for step 2. Reinstalling over an existing install drops packages EvoScientist added for MCP servers or channels—re-add them afterwards.)*

*(Working from a source checkout instead? Run `uv sync --extra telegram --extra stt --extra oauth` inside your clone, and use `<clone>/.venv/bin/EvoSci` wherever this recipe says `EvoSci`.)*

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
EvoSci config set anthropic_auth_mode oauth            # Claude Pro/Max (step 2) only
# or, with an API key:
#   EvoSci config set anthropic_auth_mode api_key
#   EvoSci config set anthropic_api_key "sk-ant-..."
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
EVOSCI_BIN=$(which EvoSci)   # source checkout: EVOSCI_BIN=/path/to/clone/.venv/bin/EvoSci
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

A LaunchAgent starts when you log in, not at boot. To come back on its own after a power cut or update reboot, turn on automatic login (**System Settings → Users & Groups**). macOS doesn't offer this while FileVault is on. Anyone who can restart the Mac then gets into that account without a password, so only do this with a standard (non-admin) account on a machine in a place you trust.

## Day-to-Day Management

| Action | Command |
|---|---|
| Restart `EvoSci serve` | `launchctl kickstart -k gui/$(id -u)/com.evosci.serve` |
| Stop `EvoSci serve` | `launchctl unload ~/Library/LaunchAgents/com.evosci.serve.plist` |
| Stop the background agent server (and its managed `ccproxy`) | `EvoSci server stop` *(outlives `EvoSci serve`; run this before restarting after an update)* |
| Reload after editing the plist | `launchctl unload ~/Library/LaunchAgents/com.evosci.serve.plist && launchctl load ~/Library/LaunchAgents/com.evosci.serve.plist` |

## Troubleshooting

- **`zsh: no matches found`:** Quote package extras (`'EvoScientist[telegram,stt,oauth]'`).
- **Claude OAuth token expired/missing:** Run `ccproxy auth login claude_api` and check `ccproxy auth status claude_api`. If `ccproxy` isn't found (`EvoSci update` doesn't keep it on your `PATH`), call it from next to `EvoSci`: `E=$(command -v EvoSci); "$(dirname "$(readlink "$E" || echo "$E")")/ccproxy" auth login claude_api`.
- **Voice messages arrive as `[voice: ...]` instead of text:** Confirm `stt_enabled` is `true` and `faster-whisper` is installed in the same environment as `EvoSci`.
- **Agent asks for shell approval in Telegram:** Make sure `--auto-approve` is included in `ProgramArguments`.
- **Workspace on an external `/Volumes/...` drive:** `KeepAlive: true` automatically retries until the drive mounts after reboot.
