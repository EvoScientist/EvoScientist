# Using Your AI Subscriptions (Claude OAuth & ChatGPT Sign-In)

Run EvoScientist using your existing **Claude Pro/Max** and/or **ChatGPT Plus/Pro** subscriptions—no API keys and no per-token billing.

```txt
EvoScientist
 ├─ Claude   ──▶ ccproxy (127.0.0.1:8000, ~/.claude/.credentials.json) ──▶ api.anthropic.com
 └─ ChatGPT  ──▶ direct sign-in (~/.evoscientist/auth/chatgpt.json)   ──▶ chatgpt.com/backend-api/codex
```

- **Claude** uses [ccproxy](https://pypi.org/project/ccproxy-api/), which the background server starts and stops automatically on `ccproxy_port` (reusing an existing `ccproxy` if you already run one; check with `EvoSci server status`).
- **ChatGPT** needs no proxy and no Codex CLI—EvoScientist signs in directly and auto-refreshes tokens.

---

## Part 1: Claude Pro / Max (via `ccproxy`)

1. **Install the OAuth extra:** The easiest route is `EvoSci onboard` → Anthropic → Claude Pro/Max sign-in. It installs `ccproxy` into EvoScientist's own environment (keeping its pinned dependencies) and runs the login for you, covering steps 1–2. To do it by hand on a [one-line install](../guides/install.md), reinstall with the extra, pinned to your release's tested dependencies:
   ```bash
   V=$(EvoSci --version | awk '{print $2}')
   uv=$(sed -E 's/.*"path": *"([^"]*)".*/\1/' ~/.evoscientist/tools/uv.json)   # the installer's own uv
   "$uv" tool install --force --python 3.12 --managed-python \
     -c "https://raw.githubusercontent.com/EvoScientist/EvoScientist/v$V/constraints.txt" \
     "EvoScientist[oauth]==$V"
   which ccproxy
   ```
   *(The reinstall drops packages EvoScientist added for MCP servers or channels—re-add them afterwards, or use `EvoSci onboard` instead. Source checkout: `uv sync --extra oauth`.)*

2. **Authenticate once per machine:**
   ```bash
   ccproxy auth login claude_api
   ccproxy auth status claude_api
   ```
   ccproxy refreshes the token itself; re-run the login only if `status` shows it expired or not authenticated.

3. **Configure EvoScientist:**
   ```bash
   EvoSci config set anthropic_auth_mode oauth
   EvoSci config set provider anthropic
   EvoSci config set model claude-sonnet-5-5
   ```

Run `EvoSci` normally—no `ANTHROPIC_API_KEY` required. Any Anthropic model ID your plan serves works (unlisted IDs pass through verbatim).

> **Note:** Extended thinking is automatically disabled over `ccproxy` because proxied endpoints reject thinking blocks on multi-turn round-trips.

---

## Part 2: ChatGPT Plus / Pro (Direct Sign-In)

Uses the unofficial ChatGPT/Codex OAuth sign-in from `langchain-openai` (subject to OpenAI's terms for your plan). No extra packages needed.

1. **Sign in (once per machine):**
   ```bash
   EvoSci login chatgpt       # or choose "ChatGPT sign-in" inside `EvoSci onboard`
   ```
   Your browser opens OpenAI's login page and returns to `localhost:1455`. Sign out locally anytime with `EvoSci logout chatgpt` (doesn't revoke remotely). *(For GUI/programmatic callers, `EvoSci login chatgpt --json` prints JSONL `url` / `signed_in` / `error` events without opening a browser.)*

2. **Configure EvoScientist:**
   ```bash
   EvoSci config set openai_auth_mode oauth
   EvoSci config set provider openai
   EvoSci config set model gpt-6-sol
   EvoSci config set reasoning_effort high   # low | medium | high (default) | xhigh | max
   ```
   While `openai_auth_mode: oauth` is active, **every** `openai` model (main, auxiliary, and fallbacks) uses your ChatGPT plan, ignoring `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `EVOSCIENTIST_USE_RESPONSES_API`.

### Plan & Context Notes

- **Supported models depend on your plan** (and differ from the public OpenAI API). If your ChatGPT tier doesn't include a model ID, the backend returns `The '<model>' model is not supported when using Codex with a ChatGPT account.`
- **272K context budget:** EvoScientist summarizes conversations before 272K tokens on this route to match Codex defaults and protect subscription rate limits.
- **Reasoning effort:** Higher settings (`xhigh`/`max`) increase latency and consume more plan quota (`none`/`minimal` map to `low`).

### Headless SSH & Docker

- **Over SSH:** Forward the callback port before running `EvoSci login chatgpt`, then open the printed URL locally:
  ```bash
  ssh -L 1455:localhost:1455 <host>
  ```
- **In Docker:** Sign in on a machine with a browser, stop EvoScientist in the container, copy `~/.evoscientist/auth/chatgpt.json` into the container's `/home/evosci/.evoscientist/auth/chatgpt.json`, then **sign in again on your host machine**. OpenAI rotates tokens on refresh, so two machines sharing one token file will log each other out.
  - The file **and its `auth/` directory** must belong to the container user `evosci`: EvoScientist rewrites the file on every refresh.
  - If the container later reports `ChatGPT sign-in required`, repeat both steps (copy a fresh file in, then sign in again on the host).

---

## Quick Verification

```bash
# Claude: check ccproxy health + model call
curl -s http://127.0.0.1:8000/health | head -c 60
ANTHROPIC_BASE_URL=http://127.0.0.1:8000/claude ANTHROPIC_API_KEY=ccproxy-oauth \
python -c "from EvoScientist.llm import get_chat_model; print(get_chat_model('claude-sonnet-5-5', provider='anthropic').invoke('Reply with exactly: OK').content)"

# ChatGPT: model call (after `EvoSci login chatgpt`)
EVOSCIENTIST_OPENAI_AUTH_MODE=oauth \
python -c "from EvoScientist.llm import get_chat_model; print(get_chat_model('gpt-6-sol', provider='openai').invoke('Reply with exactly: OK').text)"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ChatGPT sign-in required` | Run `EvoSci login chatgpt` (or sign in from desktop Settings, or switch back via `EvoSci config set openai_auth_mode api_key`). |
| `Port 1455 is in use` | Close the other running sign-in or Codex CLI login and retry. |
| OpenAI connection error to `127.0.0.1` after upgrade | Run `EvoSci server stop` so a fresh server starts without legacy ccproxy routing. |
| `ccproxy is required ... not found` | Install the extra (Part 1, step 1). |
| `Run: ccproxy auth login claude_api` | Authenticate Claude OAuth once via that command. |
| `ccproxy_port 8000 is in use` | Free port `8000` or pick another via `EvoSci config set ccproxy_port <port>`. |
| `ccproxy` config changes ignored | Run `EvoSci server stop`, restart, and check for a shadowing `.ccproxy.toml` / `ccproxy.toml`. |

**Debugging Claude requests:** Run a second ccproxy on a spare port with `ccproxy serve --port 8001 --log-level debug` to see exactly what is forwarded upstream.

> **Security note:** Never paste `ccproxy --log-level debug` output or `~/.evoscientist/auth/chatgpt.json` anywhere—both contain live bearer tokens. Heavy agent loops also consume your subscription's shared quota faster than normal chat.
