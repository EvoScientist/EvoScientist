# Running EvoScientist 24/7 on macOS (Telegram + Voice STT + ccproxy)

Deploy EvoScientist as an always-on background service on an Apple silicon Mac (e.g., Mac Mini) using `launchd`, with Telegram messaging, offline voice-to-text (`faster-whisper`), and Claude subscription routing via `ccproxy`.

```txt
Telegram (text / voice) ──▶ python-telegram-bot ──▶ STT (faster-whisper)
                                                          │
Anthropic Claude API ◀── ccproxy (:8000) ◀── EvoScientist agent
```

## Prerequisites

- macOS 13+ on Apple silicon, [uv](https://docs.astral.sh/uv), and an active Claude subscription
- A Telegram bot token from [@BotFather](https://t.me/BotFather) and your numeric user ID from [@userinfobot](https://t.me/userinfobot)

## 1. Clone & Install From Source

Clone both repos into the same parent directory (`~/Dev/tools`):

```bash
mkdir -p ~/Dev/tools && cd ~/Dev/tools
git clone git@github.com:jhfnetboy/EvoScientist.git
git clone git@github.com:jhfnetboy/ccproxy-api.git

# Install patched ccproxy fork (PyPI lacks Claude 4+ adaptive thinking / OAuth fixes)
cd ~/Dev/tools/ccproxy-api
git checkout jhf-research
pip install -e .
which ccproxy

# Authenticate Claude subscription once (saves ~/.claude/.credentials.json)
ccproxy auth login claude-api
ls ~/.claude/.credentials.json

# Install EvoScientist with Telegram + STT extras (always quote brackets in zsh)
cd ~/Dev/tools/EvoScientist
git checkout jhf-research
uv pip install -e '.[telegram,stt]'
which EvoSci && EvoSci -h
```

## 2. Pre-Download the Whisper STT Model

Avoids a ~250 MB download delay on your first voice message:

```bash
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('Systran/faster-whisper-small', device='cpu', compute_type='int8'); print('STT model ready')"
```

## 3. Configure & Smoke Test

```bash
cd ~/Dev/tools/EvoScientist
EvoSci config set anthropic_base_url "http://localhost:8000/claude"
EvoSci config set anthropic_api_key  "sk-dummy"
EvoSci config set model              "claude-sonnet-5-5"
EvoSci config set telegram_bot_token       "YOUR_BOT_TOKEN"
EvoSci config set telegram_allowed_senders "YOUR_TELEGRAM_USER_ID"
EvoSci config set stt_enabled  true
EvoSci config set stt_language zh   # zh | en | auto
EvoSci config set default_workdir "/absolute/path/to/your/workspace"
```

Verify interactively in two terminals before creating `launchd` plists:

```bash
# Terminal 1
ccproxy serve --port 8000

# Terminal 2
cd ~/Dev/tools/EvoScientist && EvoSci serve --auto-approve
```

Send both a text message and a voice note to your Telegram bot. Once both work, stop the terminals and continue.

## 4. Register `launchd` Auto-Start Services

```bash
CCPROXY=$(which ccproxy)
EVOSCI_DIR="$HOME/Dev/tools/EvoScientist"
EVOSCI_BIN="${EVOSCI_DIR}/.venv/bin/EvoSci"

cat > ~/Library/LaunchAgents/com.evosci.ccproxy.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.evosci.ccproxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>${CCPROXY}</string><string>serve</string><string>--port</string><string>8000</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/ccproxy.log</string>
    <key>StandardErrorPath</key><string>/tmp/ccproxy.log</string>
</dict>
</plist>
EOF

cat > ~/Library/LaunchAgents/com.evosci.serve.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.evosci.serve</string>
    <key>ProgramArguments</key>
    <array>
        <string>${EVOSCI_BIN}</string><string>serve</string><string>--auto-approve</string>
    </array>
    <key>WorkingDirectory</key><string>${EVOSCI_DIR}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/evosci.log</string>
    <key>StandardErrorPath</key><string>/tmp/evosci.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.evosci.ccproxy.plist
launchctl load ~/Library/LaunchAgents/com.evosci.serve.plist
launchctl list | grep evosci
```

Watch live logs anytime with `tail -f /tmp/evosci.log /tmp/ccproxy.log`.

## Day-to-Day Management

| Action | Command |
|---|---|
| Restart EvoSci / ccproxy | `launchctl kickstart -k gui/$(id -u)/com.evosci.serve` (or `com.evosci.ccproxy`) |
| Stop a service | `launchctl unload ~/Library/LaunchAgents/com.evosci.<name>.plist` |
| Stop the background agent server | `EvoSci server stop` *(outlives `EvoSci serve`; run this before restarting after an update)* |
| Reload after editing a plist | `launchctl unload <plist> && launchctl load <plist>` |

## Troubleshooting

- **`zsh: no matches found`:** Quote extras in zsh (`uv pip install -e '.[telegram,stt]'`).
- **Package installed into conda base instead of `.venv`:** Always run `uv pip install` inside `~/Dev/tools/EvoScientist`, never plain `pip install`.
- **`ValueError: No valid OAuth access token available`:** Run `ccproxy auth login claude-api` and confirm `~/.claude/.credentials.json` exists.
- **Voice messages arrive as `[voice: ...]` instead of text:** Set `EvoSci config set stt_enabled true` and install `faster-whisper` in `.venv`.
- **Whisper hallucinating on silence:** Make sure you're on a branch/release with VAD filtering (`no_speech_prob` threshold from PR #28).
- **Prompts for shell approval:** Confirm `--auto-approve` is in `com.evosci.serve.plist`.
- **Workspace on an external `/Volumes/...` drive:** `KeepAlive: true` automatically retries until the drive finishes mounting at boot.
