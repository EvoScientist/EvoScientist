# Using EvoScientist with Your AI Subscriptions (Claude + ChatGPT/Codex OAuth)

Run EvoScientist on the subscriptions you already pay for — **Claude Pro/Max**
and/or **ChatGPT Plus/Pro** — with no API keys and no per-token billing.
Requests are routed through [ccproxy](https://pypi.org/project/ccproxy-api/),
a local proxy that authenticates with the same OAuth tokens your Claude Code
or Codex CLI login produced.

This recipe covers both providers, including two non-obvious pitfalls on the
ChatGPT/Codex path that produce misleading errors.

---

## How it works

```txt
EvoScientist (LangChain client)
    │  ANTHROPIC_BASE_URL=http://127.0.0.1:8000/claude      (Claude OAuth)
    │  OPENAI_BASE_URL=http://127.0.0.1:8000/codex/v1       (Codex OAuth)
    ▼
ccproxy  (localhost:8000)
    │  Claude:  OAuth token from ~/.claude/.credentials.json
    │  Codex:   OAuth token from ~/.codex/auth.json
    ▼
api.anthropic.com  /  chatgpt.com/backend-api/codex
    (billed against your subscription quota, not API credits)
```

EvoScientist manages all of this for you: when an `*_auth_mode` config key is
set to `oauth`, `EvoSci` starts ccproxy automatically (or reuses a running
instance on `ccproxy_port`) and points the provider's base URL at it.

---

## Prerequisites

- An active **Claude Pro/Max** subscription (for Claude models) and/or an
  active **ChatGPT Plus/Pro** subscription (for GPT models)
- EvoScientist installed with the OAuth extra:

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

## Part 2 — GPT models via ChatGPT subscription (Codex OAuth)

### 1. Authenticate (once per machine)

```bash
ccproxy auth login codex
ccproxy auth status codex   # shows email, subscription tier, status
```

### 2. Configure EvoScientist

```bash
EvoSci config set openai_auth_mode oauth
EvoSci config set provider         openai
EvoSci config set model            gpt-5.5
EvoSci config set reasoning_effort high     # optional; see note below
```

### 3. Know the two Codex-route pitfalls

The ChatGPT Codex backend is not a plain OpenAI-compatible API, and ccproxy's
defaults interact badly with it in two ways. **EvoScientist handles both
automatically as of [#324](https://github.com/EvoScientist/EvoScientist/pull/324)**
— read this section if you are on an older version, run your own ccproxy
instance (e.g. as a launchd/systemd service), or need to debug the errors.

#### Pitfall A — ccproxy silently rewrites your model

ccproxy ships default Codex *model mappings* that rewrite **any** model whose
name starts with `gpt-`, `o1-`, `o3-`, or `claude-` to `gpt-5.3-codex` before
forwarding. Your configured model never reaches the backend. On accounts
where `gpt-5.3-codex` is not served, every request fails with an error naming
a model you never asked for:

```text
The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account.
```

**Fix:** disable the mappings so requested models pass through unmodified.
EvoScientist (with #324) generates this config and passes it via
`ccproxy serve --config` when it starts ccproxy itself. For a ccproxy
instance you manage yourself, add this section to your global config
(`~/.config/ccproxy/config.toml`) and restart it:

```toml
[plugins.codex]
model_mappings = []
```

(Note: ccproxy loads only the first config file it finds — a `.ccproxy.toml`
in its working directory or a `ccproxy.toml` in a git repository root takes
precedence and silently overrides the global config.)

#### Pitfall B — the backend gates models on client identity

ccproxy forwards your client's own `User-Agent` upstream and only gap-fills
its Codex headers. The backend then sees a generic HTTP client instead of a
Codex client and rejects current models:

```text
The 'gpt-5.5' model requires a newer version of Codex. Please upgrade to the latest app or CLI and try again.
```

**Fix:** send Codex-CLI-shaped headers with every request. EvoScientist
(with #324) does this automatically whenever it detects the ccproxy Codex
route. It advertises the installed Codex CLI version, with
`EVOSCIENTIST_CODEX_CLIENT_VERSION` taking explicit precedence. Otherwise,
it advertises the newer of the installed CLI version and EvoScientist's
minimum fallback. If you probe the route manually (curl, scripts), supply
the headers yourself — see Verification below.

### 4. Model availability is account-specific

Which model IDs the Codex backend serves depends on your ChatGPT plan, and
the lineup does **not** match the public OpenAI API. Measured on a ChatGPT
**Plus** account (July 2026): plain `gpt-5.5` and `gpt-5.4` complete
successfully, while `gpt-5.3-codex` is rejected. Other tiers differ. If a
model errors, try the adjacent one (`gpt-5.4`) before assuming the setup is
broken. Community measurements also indicate the Codex route enforces a
smaller context window than the public API for the same model IDs
(~272K for gpt-5.5 vs 1.05M on the API).

> **`reasoning_effort`:** with #324, the ccproxy Codex route uses the
> Responses API and accepts reasoning configuration. With
> [#321](https://github.com/EvoScientist/EvoScientist/pull/321), EvoScientist
> passes an explicitly configured effort (`low`/`medium`/`high`/`xhigh`);
> when unset, EvoScientist applies its own per-model default. Older releases
> that used Chat Completions on this route ignored reasoning effort. Higher
> effort costs more latency and more of your subscription quota per request.

---

## Verification

Health check (default port 8000; change if you set `ccproxy_port`):

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

End-to-end Codex probe (headers required — Pitfall B).
Any recent Codex CLI version string works here; the backend only checks that the client is new enough:

```bash
CODEX_VERSION="<recent Codex CLI version>"   # e.g. what `codex --version` prints
curl -sN http://127.0.0.1:8000/codex/v1/responses \
  -H "content-type: application/json" \
  -H "originator: codex_cli_rs" \
  -H "version: ${CODEX_VERSION}" \
  -H "user-agent: codex_cli_rs/${CODEX_VERSION} (probe)" \
  -d '{"model":"gpt-5.5","stream":true,"store":false,
       "input":[{"role":"user","content":[{"type":"input_text","text":"Reply with exactly: OK"}]}]}'
```

Success is an SSE stream ending in `response.completed` whose payload shows
`"model":"gpt-5.5"` — confirming the model was not rewritten en route.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account.` (you requested a different model) | ccproxy's default model mappings rewrote your model (Pitfall A) | Disable `model_mappings` and restart ccproxy |
| `The '<model>' model requires a newer version of Codex. Please upgrade…` | Backend rejected the client identity (Pitfall B) | Use an EvoScientist build containing #324, or send Codex headers; `EVOSCIENTIST_CODEX_CLIENT_VERSION` is the explicit override |
| `The '<model>' model is not supported…` for the model you actually requested | Your ChatGPT tier does not serve that ID | Try `gpt-5.4`; check tier |
| `Run: ccproxy auth login codex` (or `claude_api`) on startup | No OAuth credentials on this machine | Run the login command shown |
| `ccproxy not found` | OAuth extra not installed | `pip install 'evoscientist[oauth]'` |
| Config change has no effect | A long-running ccproxy instance predates the config, or a higher-precedence `.ccproxy.toml`/`ccproxy.toml` shadows it | Restart that ccproxy instance; check for a shadowing config file |

**Debugging tip:** `ccproxy serve --port 8001 --log-level debug` on a spare
port shows exactly what is forwarded upstream. **Warning:** debug logs print
`Authorization` bearer tokens — never paste raw log lines anywhere.

---

## Quota and billing notes

- OAuth routing consumes your **subscription's** usage limits (the same pool
  as Claude Code / the Codex app). Nothing is billed to an API key, but heavy
  agent workloads can exhaust subscription rate limits faster than chat use.
- Tokens auto-refresh via ccproxy; re-run `ccproxy auth login …` only if
  status shows expired/not authenticated.
