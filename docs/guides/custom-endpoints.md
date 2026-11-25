# Custom OpenAI- & Anthropic-Compatible Endpoints

Use `custom-openai` or `custom-anthropic` to connect EvoScientist to any compatible endpoint—vLLM, LM Studio, LiteLLM, Ollama, or an internal company gateway—without disturbing your normal vendor provider settings.

*(One endpoint of each kind is supported at a time; multiple named endpoints are tracked in [#282](https://github.com/EvoScientist/EvoScientist/issues/282).)*

## Quick Setup

```bash
EvoSci config set provider custom-openai
EvoSci config set custom_openai_base_url https://example.com/v1
EvoSci config set custom_openai_api_key YOUR_API_KEY   # omit if your server needs no key
EvoSci config set model YOUR_MODEL_ID
```

For an Anthropic-compatible endpoint, use `custom-anthropic`, `custom_anthropic_base_url` (**without** `/v1`—the SDK appends it), and `custom_anthropic_api_key`. Trailing slashes are stripped automatically.

### All Settings & Environment Variables

| Config key | Environment variable | Required |
|---|---|---|
| `provider` | — | `custom-openai` or `custom-anthropic` |
| `custom_openai_base_url` | `CUSTOM_OPENAI_BASE_URL` | Yes for `custom-openai` (e.g. `https://example.com/v1`) |
| `custom_openai_api_key` | `CUSTOM_OPENAI_API_KEY` | Only if the endpoint enforces auth |
| `custom_anthropic_base_url` | `CUSTOM_ANTHROPIC_BASE_URL` | Yes for `custom-anthropic` (e.g. `https://example.com`) |
| `custom_anthropic_api_key` | `CUSTOM_ANTHROPIC_API_KEY` | Only if the endpoint enforces auth |
| `model` | — | Yes |

> **Precedence gotcha:** Exported environment variables always beat `config.yaml`. If an old `CUSTOM_OPENAI_BASE_URL` sits in your shell profile, it silently overrides `EvoSci config set`.

## How Model IDs Are Sent & Discovered

- **Verbatim model IDs:** Whenever `provider` is explicitly `custom-openai` or `custom-anthropic`, EvoScientist sends your configured `model` string **verbatim**—even if that name also exists in another provider's short-name table (like `qwen3.5-27b`). *(If no provider is named at all, short names from other providers resolve to their registry IDs, e.g. `openai/gpt-6.1-terra`.)*
- **Automatic model listing (`custom-openai`):** When `custom_openai_base_url` is set, `/model`, `/model-fallback add`, `EvoSci onboard`, and the WebUI picker (`GET /api/models`) query `GET {base_url}/models` (up to 500 IDs, bounded by a short timeout) and display what your server actually serves. If the probe fails or returns empty, pickers fall back to EvoScientist's built-in defaults.
- **`custom-anthropic`:** Pickers show the built-in defaults; enter custom model IDs via **"Type a model name…"** in `EvoSci onboard` or `EvoSci config set model <id>`.

## Keyless Local Servers (vLLM, LM Studio, etc.)

If your endpoint doesn't require authentication, leave the API key unset. EvoScientist passes a placeholder (`not-needed`) with a one-time warning and explicitly **never** falls back to `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, so your real vendor keys never leak to a third-party server.

*(Note: `EvoSci onboard` does not validate keys for custom endpoints—a bad key surfaces on the first model request.)*
