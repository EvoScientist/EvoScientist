# Custom OpenAI- and Anthropic-compatible endpoints

`custom-openai` and `custom-anthropic` reach any endpoint that speaks the OpenAI or the Anthropic protocol: a hosted gateway, vLLM, LM Studio, an internal proxy. Each keeps its own base URL and key and leaves every other provider's settings alone. One endpoint of each kind is configured at a time; named endpoints per provider are discussed in [#282](https://github.com/EvoScientist/EvoScientist/issues/282).

## Settings

Four settings per provider, saved with `EvoSci config set` (or the matching environment variable). An exported environment variable takes precedence over the value saved by `config set`: config values are only applied to the environment when the variable is unset, so an old value in your shell profile keeps being used and nothing reports it.

| Setting | Env var | Required |
|---|---|---|
| `provider` | — | `custom-openai` or `custom-anthropic` |
| `custom_openai_base_url` | `CUSTOM_OPENAI_BASE_URL` | yes (e.g. `https://example.com/v1`) |
| `custom_openai_api_key` | `CUSTOM_OPENAI_API_KEY` | only if the endpoint checks keys |
| `custom_anthropic_base_url` | `CUSTOM_ANTHROPIC_BASE_URL` | yes, for `custom-anthropic` (e.g. `https://example.com`) |
| `custom_anthropic_api_key` | `CUSTOM_ANTHROPIC_API_KEY` | only if the endpoint checks keys |
| `model` | — | yes (see below) |

Replace the placeholder values below with your own before running:

```bash
EvoSci config set provider custom-openai
EvoSci config set custom_openai_base_url https://example.com/v1
EvoSci config set custom_openai_api_key YOUR_API_KEY
EvoSci config set model YOUR_MODEL_ID
```

The same with `custom-anthropic`, `custom_anthropic_base_url` (e.g. `https://example.com`, no `/v1` — the client appends it) and `custom_anthropic_api_key`. A trailing `/` on either base URL is stripped before use. A missing base URL fails with an error naming `CUSTOM_OPENAI_BASE_URL` / `CUSTOM_ANTHROPIC_BASE_URL` (the environment variable, even if you configured everything with `config set`); a missing key does not (see below).

## Which model id is sent

`get_chat_model` looks the configured `model` up as EvoScientist's own short name first, matching name **and** provider. A name that equals one of the same provider's short names (`gpt-6.1-sol`, `claude-sonnet-5-5`, …) is sent as written, because the exact-match entry keeps it verbatim. When **no** provider is named, a name that equals one of **another** provider's short names is replaced by that entry's model id — for example `gpt-6.1-terra` is sent as `openai/gpt-6.1-terra` and `qwen3.5-27b` as `qwen/qwen3.5-27b`, which a LiteLLM-style proxy serving plain OpenAI names may not recognize. Any other name is sent to the endpoint exactly as typed, so a model only your endpoint knows is addressed by its own id verbatim.

Naming a provider keeps the id that provider serves: a name given **with** `custom-openai` is sent as written, even when it also happens to be another provider's short name. Without that, picking `qwen3.5-27b` from a picker that listed it would send `qwen/qwen3.5-27b`, which an endpoint serving the bare id does not recognize.

When `custom_openai_base_url` is set, `/model`, `/model-fallback add`, the WebUI model list (`GET /api/models`) and `EvoSci onboard` ask the endpoint for `GET {base_url}/models` and offer those ids under `custom-openai`, in place of the three defaults (`gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`). The key is sent as `Authorization: Bearer` only when `custom_openai_api_key` (or `CUSTOM_OPENAI_API_KEY`) is set, so a keyless local server is still listed. If the endpoint is unreachable, returns a non-200, or lists nothing, the pickers fall back to those three defaults. `custom-anthropic` is unchanged: its pickers still offer only EvoScientist's defaults, and an id only that endpoint knows has to be typed in `EvoSci onboard` ("Type a model name…") or saved with `EvoSci config set model <id>`.

## Keys

An endpoint without a key runs without one: instead of failing, a placeholder key (`not-needed`) is sent, with a one-time warning naming the setting to set if the endpoint does check keys. The key is passed explicitly to the client, so the SDK never falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and never sends your vendor key to another server.

`EvoSci onboard` sets a custom endpoint up through its base URL and does **not** validate the key for `custom-openai` / `custom-anthropic` — a wrong key is only surfaced by the endpoint itself on first use.
