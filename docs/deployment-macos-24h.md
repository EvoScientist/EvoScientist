# Running EvoScientist 24/7 on macOS (Telegram Bot + STT + ccproxy)

This guide covers deploying EvoScientist as a fully automated 24/7 service on macOS
(tested on Mac Mini with Apple Silicon), using:

- **EvoScientist** — the AI research agent
- **ccproxy** — local proxy that routes API calls through your Claude Code subscription
- **Telegram channel** — messaging interface
- **STT** — automatic voice message transcription (faster-whisper)

---

## Prerequisites

- macOS 13+ (Apple Silicon or Intel)
- [Claude Code](https://claude.ai/code) installed and subscription active
- [uv](https://astral.sh/uv) installed
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Your Telegram user ID (from [@userinfobot](https://t.me/userinfobot))

---

## 1. Clone repositories

```bash
mkdir -p ~/Dev/tools
cd ~/Dev/tools

git clone git@github.com:jhfnetboy/EvoScientist.git
git clone git@github.com:jhfnetboy/ccproxy-api.git
```

> Both repos should sit as siblings in the same parent directory.

---

## 2. Install ccproxy from source

> **Do not** `pip install ccproxy-api` from PyPI — it lacks the adaptive thinking
> and OAuth header fixes required for Claude 4+ models. Install from the patched fork.

```bash
cd ~/Dev/tools/ccproxy-api
git checkout jhf-research
pip install -e .
which ccproxy   # verify: should print a path
```

---

## 3. Authenticate ccproxy with your Claude subscription

This step opens a browser window to log in with your Claude account.
It only needs to be done once per machine.

```bash
ccproxy auth login claude-api
```

Verify the token was saved:

```bash
ls ~/.claude/.credentials.json   # note the leading dot
```

---

## 4. Install EvoScientist from source

```bash
cd ~/Dev/tools/EvoScientist
git checkout jhf-research

# Install with Telegram and STT support
uv pip install -e '.[telegram,stt]'
```

> **Important:** always use `uv pip install`, not `pip install`, to ensure packages
> land in the correct `.venv` that the service will use.

Verify:

```bash
which EvoSci
EvoSci -h
```

---

## 5. Pre-download the STT model

The STT model (~250 MB) downloads from HuggingFace on first use.
Pre-download it now to avoid a delay on the first voice message:

```bash
uv run python -c "
from faster_whisper import WhisperModel
WhisperModel('Systran/faster-whisper-small', device='cpu', compute_type='int8')
print('STT model ready')
"
```

---

## 6. Configure EvoScientist

```bash
cd ~/Dev/tools/EvoScientist

EvoSci config set anthropic_base_url "http://localhost:8000/claude"
EvoSci config set anthropic_api_key  "sk-dummy"
EvoSci config set model              "claude-sonnet-4-6"

# Telegram
EvoSci config set telegram_bot_token      "YOUR_BOT_TOKEN"
EvoSci config set telegram_allowed_senders "YOUR_TELEGRAM_USER_ID"

# STT voice transcription
EvoSci config set stt_enabled  true
EvoSci config set stt_language zh     # zh / en / auto

# Optional: set a default workspace directory
EvoSci config set default_workdir "/absolute/path/to/your/workspace"
```

---

## 7. Smoke test before creating services

Open two terminals:

**Terminal 1 — ccproxy:**
```bash
ccproxy serve --port 8000
```

**Terminal 2 — EvoSci:**
```bash
cd ~/Dev/tools/EvoScientist
EvoSci serve --auto-approve
```

Send a text message and a voice message to your Telegram bot.
If both work, proceed to set up the background services.

---

## 8. Create launchd services (auto-start on login)

Run this script once to create both plist files:

```bash
CCPROXY=$(which ccproxy)
EVOSCI_DIR="$HOME/Dev/tools/EvoScientist"   # adjust if your path differs
EVOSCI_BIN="${EVOSCI_DIR}/.venv/bin/EvoSci"

# ── ccproxy service ──────────────────────────────────────────────────
cat > ~/Library/LaunchAgents/com.evosci.ccproxy.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.evosci.ccproxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>${CCPROXY}</string>
        <string>serve</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/ccproxy.log</string>
    <key>StandardErrorPath</key><string>/tmp/ccproxy.log</string>
</dict>
</plist>
EOF

# ── EvoSci serve service ─────────────────────────────────────────────
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
    <key>WorkingDirectory</key><string>${EVOSCI_DIR}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/evosci.log</string>
    <key>StandardErrorPath</key><string>/tmp/evosci.log</string>
</dict>
</plist>
EOF

# ── Load both services ───────────────────────────────────────────────
launchctl load ~/Library/LaunchAgents/com.evosci.ccproxy.plist
launchctl load ~/Library/LaunchAgents/com.evosci.serve.plist

echo "Services loaded:"
launchctl list | grep evosci
```

---

## 9. Check status and logs

```bash
# Are both services running?
launchctl list | grep evosci

# Live logs
tail -f /tmp/evosci.log
tail -f /tmp/ccproxy.log
```

---

## Managing services

| Action | Command |
|--------|---------|
| Restart EvoSci | `launchctl kickstart -k gui/$(id -u)/com.evosci.serve` |
| Restart ccproxy | `launchctl kickstart -k gui/$(id -u)/com.evosci.ccproxy` |
| Stop EvoSci | `launchctl unload ~/Library/LaunchAgents/com.evosci.serve.plist` |
| Stop ccproxy | `launchctl unload ~/Library/LaunchAgents/com.evosci.ccproxy.plist` |
| Reload after plist edit | `launchctl unload <plist> && launchctl load <plist>` |

---

## Troubleshooting

### `ccproxy not found`
ccproxy is not installed. Run:
```bash
cd ~/Dev/tools/ccproxy-api && pip install -e .
```

### `ValueError: No valid OAuth access token available`
ccproxy cannot find your Claude login token. Run:
```bash
ccproxy auth login claude-api
```
A browser window will open. Log in with your Claude subscription account.
Verify the token was saved at `~/.claude/.credentials.json` (hidden file, note the leading dot).

### `Channel telegram fatal error: python-telegram-bot not installed`
The package was installed in the wrong Python environment. Use `uv pip`:
```bash
cd ~/Dev/tools/EvoScientist
uv pip install 'python-telegram-bot>=21.0'
```

### `zsh: no matches found: evoscientist[telegram]`
Shell interprets `[` as a glob. Always quote the package name:
```bash
uv pip install 'EvoScientist[telegram,stt]'   # correct
uv pip install  EvoScientist[telegram,stt]    # wrong — shell eats the brackets
```

### Agent keeps asking to approve ffmpeg / shell commands
Add `--auto-approve` to the EvoSci serve command in the plist (see step 8).

### Voice messages not transcribed / agent receives `[voice: ...]` annotation
STT is disabled or not installed:
```bash
EvoSci config set stt_enabled true
uv pip install faster-whisper
```

### Whisper outputs gibberish (`字幕by索兰娅` etc.)
This is a known Whisper hallucination on silent/very short audio. It is filtered
automatically by the built-in VAD filter and `no_speech_prob` threshold introduced
in PR #28. Make sure you are on the `jhf-research` branch or a release that includes
that fix.

### `Could not find service "com.evosci.*" in domain`
The plist was never loaded. Run:
```bash
launchctl load ~/Library/LaunchAgents/com.evosci.ccproxy.plist
launchctl load ~/Library/LaunchAgents/com.evosci.serve.plist
```

### heredoc `cat > file << EOF` hangs in terminal
The shell variable inside the heredoc was not expanded because `EOF` was quoted
(`<< 'EOF'`). Use unquoted `<< EOF` when the content contains `$VARIABLES` that
should be expanded, or pre-set the variables before running the command.

### Services on external drive not starting at boot
`/Volumes/...` paths are only available after the external drive mounts.
With `KeepAlive: true`, launchd will keep retrying every few seconds and
the services will start automatically once the drive is available.
This is normal behaviour for a Mac Mini where the drive is always connected.

### `pip install` installs to conda base instead of `.venv`
Always use `uv pip install` inside the EvoScientist project directory,
or explicitly target the venv:
```bash
cd ~/Dev/tools/EvoScientist
uv pip install 'some-package'
```

---

## Architecture overview

```
Telegram app
    │  voice / text message
    ▼
python-telegram-bot (polling)
    │
    ▼
EvoScientist channel layer
    │  STT hook in _enqueue_raw()
    │  .ogg → faster-whisper → plain text
    ▼
EvoScientist agent
    │  tool calls / LLM requests
    ▼
ccproxy  (localhost:8000)
    │  OAuth token from ~/.claude/.credentials.json
    ▼
Anthropic Claude API  (claude.ai subscription)
```
