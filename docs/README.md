# EvoScientist Documentation

Guides, recipes, and integration references for getting the most out of EvoScientist.

## Start Here

New to EvoScientist? Read in this order:

1. **[Main README](../README.md)** — What EvoScientist does, desktop app & quick install, onboarding, and everyday commands.
2. **[Installation Guide](guides/install.md)** — One-line installer internals, mainland China mirrors (`--cn`), proxies, corporate TLS certs, upgrading, and uninstalling.
3. **[Security Policy](../SECURITY.md)** — Trust model, approval modes, command guard, and safe LAN/SSH access.

## Guides

| Guide | What it covers |
|---|---|
| [Installation](guides/install.md) | One-line install options, `--cn` mirrors, proxies, custom TLS certs, `EvoSci update`, and clean uninstall |
| [Custom Endpoints](guides/custom-endpoints.md) | Pointing `custom-openai` / `custom-anthropic` at vLLM, LM Studio, LiteLLM, or internal gateways |
| [Working Toward a Goal](guides/goal.md) | `/goal` in the TUI & CLI: draft acceptance criteria, review them, and grade every turn until met |
| [Grading With a Rubric](guides/rubric.md) | Explicit checklists via `--rubric` (single-shot) or `/rubric` (interactive), exit codes, and deferral |
| [Transient Model Retries](guides/model-retry.md) | Retry budgets (`model_retry_wait_seconds`, `auto_mode_retry_wait_seconds`), fallback chains, and backoff |
| [`stream-json` Protocol](guides/stream-json.md) | JSONL event schema (`--output-format stream-json`) for driving EvoScientist headlessly from scripts/SDKs |

## Recipes & Examples

| Page | What it covers |
|---|---|
| [Using AI Subscriptions (Claude OAuth + ChatGPT)](recipes/subscription-oauth.md) | Run on Claude Pro/Max (`ccproxy`) and/or ChatGPT Plus/Pro (direct sign-in) with no API keys |
| [24/7 macOS Deployment (Telegram + STT + ccproxy)](recipes/deployment-macos-24h.md) | Run EvoScientist continuously on a Mac via `launchd` with voice transcription |
| [Example: Literature Survey](examples/survey-literature/README.md) | Produce a bilingual conference-grade survey (68 SIGIR 2026 papers) using `paper-navigator` |

## Integrations

| Integration | Where to look |
|---|---|
| **MCP Servers** | [MCP Integration Guide](../EvoScientist/mcp/README.md) — `EvoSci mcp add`, transports, wildcard tool filters, and per-agent routing |
| **Messaging Channels** | [Channels Guide](../EvoScientist/channels/README.md) — Telegram, Discord, Slack, Feishu, WeChat, DingTalk, QQ, Signal, Email, and iMessage |

## Contributing & Agents

- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — What belongs in a PR, verification expectations, and local dev setup.
- **[AGENTS.md](../AGENTS.md)** — Repo map, safety rules, multi-surface checklist, and verification commands for coding agents.

### Adding a New Recipe

Keep recipes focused on one real setup or workflow, use `EvoSci` CLI commands (`EvoSci onboard`, `EvoSci config set`, `EvoSci serve`), specify any optional extras cleanly (e.g. `'EvoScientist[telegram,stt]'`), and add a row to the table above.
