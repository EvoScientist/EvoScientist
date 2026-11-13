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

`get_chat_model` looks the configured `model` up as EvoScientist's own short name first, matching name **and** provider. A name that equals one of the same provider's short names (`gpt-6.1-sol`, `claude-sonnet-5-5`, …) is sent as written, because the exact-match entry keeps it verbatim. A name that equals one of **another** provider's short names is replaced by that entry's model id — for example `gpt-6.1-terra` is sent as `openai/gpt-6.1-terra` and `qwen3.5-27b` as `qwen/qwen3.5-27b`, which a LiteLLM-style proxy serving plain OpenAI names may not recognize. Any other name is sent to the endpoint exactly as typed, so a model only your endpoint knows is addressed by its own id verbatim.

The pickers offer only EvoScientist's defaults for these providers. `/model` accepts any short name in the model registry — including other providers' names when you also give the provider, e.g. `/model qwen3.5-27b custom-openai` — but rejects ids outside the registry (listing an endpoint's own models is [#713](https://github.com/EvoScientist/EvoScientist/issues/713)), so an id only your endpoint knows has to be typed in `EvoSci onboard` ("Type a model name…") or saved with `EvoSci config set model <id>`; this applies to `custom-anthropic` just as to `custom-openai`.

## Keys

An endpoint without a key runs without one: instead of failing, a placeholder key (`not-needed`) is sent, with a one-time warning naming the setting to set if the endpoint does check keys. The key is passed explicitly to the client, so the SDK never falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and never sends your vendor key to another server.

`EvoSci onboard` sets a custom endpoint up through its base URL and does **not** validate the key for `custom-openai` / `custom-anthropic` — a wrong key is only surfaced by the endpoint itself on first use.
