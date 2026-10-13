# Using EvoScientist with Your AI Subscriptions (Claude OAuth + ChatGPT sign-in)

Run EvoScientist on the subscriptions you already pay for — **Claude Pro/Max** and/or **ChatGPT Plus/Pro** — with no API keys and no per-token billing.

The two providers connect differently: Claude requests go through [ccproxy](https://pypi.org/project/ccproxy-api/), a local proxy that reuses your Claude Code OAuth login; ChatGPT needs no proxy, because EvoScientist signs in to your ChatGPT account itself.

---

## How it works

```txt
EvoScientist (LangChain client)
    │
    ├─ Claude:   ANTHROPIC_BASE_URL=http://127.0.0.1:8000/claude
    │            ▼
    │        ccproxy (localhost:8000), OAuth token from ~/.claude/.credentials.json
    │            ▼
    │        api.anthropic.com
    │
    └─ ChatGPT:  token from ~/.evoscientist/auth/chatgpt.json
                 ▼
             chatgpt.com/backend-api/codex

    (both billed against your subscription quota, not API credits)
```

- **Claude** (`anthropic_auth_mode: oauth`): `EvoSci` starts ccproxy automatically (or reuses a running instance on `ccproxy_port`) and points the Anthropic base URL at it.
- **ChatGPT** (`openai_auth_mode: oauth`): EvoScientist opens OpenAI's sign-in page once, stores the token under `~/.evoscientist/auth/chatgpt.json`, and calls the Codex backend with it. No proxy and no Codex CLI are involved.

---

## Prerequisites

- An active **Claude Pro/Max** subscription (for Claude models) and/or an active **ChatGPT Plus/Pro** subscription (for GPT models)
- For Claude only, EvoScientist installed with the OAuth extra (ChatGPT sign-in needs nothing extra):

```bash
pip install 'evoscientist[oauth]'
# or, editable install:
uv sync --extra oauth
```

Verify the proxy binary is available:

```bash
which ccproxy
```

---

## Part 1 — Claude models via Claude subscription

### 1. Authenticate (once per machine)

```bash
ccproxy auth login claude_api
```

A browser window opens; log in with your Claude subscription account. Verify:

```bash
ccproxy auth status claude_api   # shows email, subscription tier, status
```

### 2. Configure EvoScientist

```bash
EvoSci config set anthropic_auth_mode oauth
EvoSci config set provider          anthropic
EvoSci config set model             claude-sonnet-4-6
```

Any Anthropic model ID your subscription serves works here — including IDs
not in EvoScientist's short-name registry (they pass through verbatim).

### 3. Done

Run `EvoSci` normally. No `ANTHROPIC_API_KEY` needed; EvoScientist sets a
placeholder key and routes through ccproxy's `/claude` endpoint.

> **Note:** extended thinking is automatically disabled on this route —
> proxied endpoints reject thinking blocks on conversation round-trips.

---

## Part 2 — GPT models via ChatGPT subscription (ChatGPT sign-in)

This uses the ChatGPT sign-in that ships with `langchain-openai`, the same kind of sign-in the Codex CLI uses. It is unofficial, and OpenAI's terms apply to how you use your plan.

### 1. Sign in (once per machine)

Either run the wizard:

```bash
EvoSci onboard
```

pick **OpenAI**, then **ChatGPT sign-in (no API key needed)**, and confirm `Log in to ChatGPT now?`.

Or set the mode yourself and start EvoScientist in a terminal, which asks `Sign in with ChatGPT now?`:

```bash
EvoSci config set openai_auth_mode oauth
EvoSci
```

Your browser opens OpenAI's sign-in page and returns to a local callback on `localhost:1455`. You stay signed in afterwards: the token refreshes itself before a request needs it.

### 2. Configure EvoScientist

```bash
EvoSci config set openai_auth_mode oauth
EvoSci config set provider         openai
EvoSci config set model            gpt-6-sol
EvoSci config set reasoning_effort high     # optional; see note below
```

In this mode every `openai` model uses your plan: the main model, the auxiliary model and any OpenAI entry in the fallback chain. `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `EVOSCIENTIST_USE_RESPONSES_API` are ignored while it is on.

### 3. Models depend on your plan

Which model IDs the backend serves depends on your ChatGPT plan, and the lineup does **not** match the public OpenAI API. Measured on a ChatGPT **Plus** account (October 2026): `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra`, the `gpt-5.6` family and `gpt-5.5` complete successfully. Other tiers differ. A model your plan does not include answers:

```text
The '<model>' model is not supported when using Codex with a ChatGPT account.
```

Community measurements also indicate this route enforces a smaller context window than the public API for the same model IDs (~272K for gpt-5.5 vs 1.05M on the API).

> **`reasoning_effort`:** an explicitly configured effort (`low`/`medium`/`high`/`xhigh`) is passed through; when unset, EvoScientist applies its own per-model default. Higher effort costs more latency and more of your subscription quota per request.

### 4. Machines without a browser

Over SSH, EvoScientist prints the sign-in link instead of opening a browser. Forward the callback port from your own computer, then open the link there:

```bash
ssh -L 1455:localhost:1455 <host>
```

### 5. Docker

Sign in on a machine with a browser, then copy `~/.evoscientist/auth/chatgpt.json` into the container's data volume at `/home/evosci/.evoscientist/auth/chatgpt.json`. Stop EvoScientist in the container before you replace the file, and make sure the file and its `auth/` directory belong to the container user (`evosci`): EvoScientist rewrites the file whenever the token refreshes. Each refresh replaces the stored token, so the copy you started from goes stale; if the container reports `ChatGPT sign-in required`, run `EvoSci onboard` on the machine with the browser, choose ChatGPT sign-in and re-authenticate, then copy the new file.

---

## Verification

Claude: ccproxy health check (default port 8000; change if you set `ccproxy_port`):

```bash
curl -s http://127.0.0.1:8000/health | head -c 60
```

End-to-end Claude probe:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000/claude ANTHROPIC_API_KEY=ccproxy-oauth \
python -c "
from EvoScientist.llm import get_chat_model
print(get_chat_model('claude-sonnet-4-6', provider='anthropic').invoke('Reply with exactly: OK').content)
"
```

End-to-end ChatGPT probe (after signing in):

```bash
EVOSCIENTIST_OPENAI_AUTH_MODE=oauth \
python -c "
from EvoScientist.llm import get_chat_model
print(get_chat_model('gpt-6-sol', provider='openai').invoke('Reply with exactly: OK').text)
"
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ChatGPT sign-in required (openai_auth_mode is "oauth"). Start EvoSci in a terminal to sign in.` | No ChatGPT token on this machine, or OpenAI no longer accepts the stored one | Start `EvoSci` in a terminal and confirm the sign-in prompt |
| `Port 1455 is in use (another sign-in may be running). Close it and try again.` | Another sign-in, or a Codex CLI login, holds the callback port | Finish or close it, then retry |
| `The '<model>' model is not supported when using Codex with a ChatGPT account.` | Your ChatGPT plan does not include that model | Pick a model your plan serves |
| OpenAI requests fail with a connection error to `127.0.0.1` right after upgrading | A background server started before the upgrade still routes OpenAI through ccproxy | `EvoSci server stop`, then start again |
| `Run: ccproxy auth login claude_api` on startup | No Claude OAuth credentials on this machine | Run the login command shown |
| `ccproxy is required for OAuth mode but not found` | OAuth extra not installed (Claude only) | `pip install 'evoscientist[oauth]'` |
| ccproxy config change has no effect | A long-running ccproxy instance predates the config, or a higher-precedence `.ccproxy.toml`/`ccproxy.toml` shadows it | Restart that ccproxy instance; check for a shadowing config file |

**Debugging tip (Claude):** `ccproxy serve --port 8001 --log-level debug` on a spare port shows exactly what is forwarded upstream. **Warning:** debug logs print `Authorization` bearer tokens — never paste raw log lines anywhere. The same goes for `chatgpt.json`: it holds your ChatGPT tokens.

---

## Quota and billing notes

- Subscription routing consumes your **subscription's** usage limits (the same pool as Claude Code / the Codex app). Nothing is billed to an API key, but heavy agent workloads can exhaust subscription rate limits faster than chat use.
- Tokens refresh on their own: Claude's via ccproxy (re-run `ccproxy auth login claude_api` only if status shows expired/not authenticated), ChatGPT's by EvoScientist before a request.
