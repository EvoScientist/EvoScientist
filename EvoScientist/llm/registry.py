"""Model registry data — short names, model ids, providers, routing tables.

Pure data with no langchain/provider-SDK imports: the onboard wizard, the
``/model`` pickers, and provider validation read this registry without paying
for the chat-model construction stack in :mod:`.models` (~2000 modules).
"""

from __future__ import annotations

_MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"
_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
_ZHIPU_CODE_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
_VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
_VOLCENGINE_CODE_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_CODE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"

_ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
_KIMI_CODING_BASE_URL = "https://api.kimi.com/coding/"
_REQUESTY_BASE_URL = "https://router.requesty.ai/v1"

# Providers routed through the OpenAI provider with a custom base_url.
# Maps provider name → (base_url or None, env var for API key).
_OPENAI_ROUTED_PROVIDERS: dict[str, tuple[str | None, str]] = {
    "atlascloud": (_ATLASCLOUD_BASE_URL, "ATLASCLOUD_API_KEY"),
    "moonshot": (_MOONSHOT_BASE_URL, "MOONSHOT_API_KEY"),
    "siliconflow": (_SILICONFLOW_BASE_URL, "SILICONFLOW_API_KEY"),
    "zhipu": (_ZHIPU_BASE_URL, "ZHIPU_API_KEY"),
    "zhipu-code": (_ZHIPU_CODE_BASE_URL, "ZHIPU_API_KEY"),
    "volcengine": (_VOLCENGINE_BASE_URL, "VOLCENGINE_API_KEY"),
    "volcengine-code": (_VOLCENGINE_CODE_BASE_URL, "VOLCENGINE_API_KEY"),
    "dashscope": (_DASHSCOPE_BASE_URL, "DASHSCOPE_API_KEY"),
    "dashscope-code": (_DASHSCOPE_CODE_BASE_URL, "DASHSCOPE_API_KEY"),
    "requesty": (_REQUESTY_BASE_URL, "REQUESTY_API_KEY"),
    "custom-openai": (
        None,
        "CUSTOM_OPENAI_API_KEY",
    ),  # base_url from CUSTOM_OPENAI_BASE_URL env
}

# Providers routed through the Anthropic provider with a custom base_url.
# Maps provider name → (base_url or None, env var for API key).
_ANTHROPIC_ROUTED_PROVIDERS: dict[str, tuple[str | None, str]] = {
    "minimax": (_MINIMAX_ANTHROPIC_BASE_URL, "MINIMAX_API_KEY"),
    "kimi-coding": (_KIMI_CODING_BASE_URL, "KIMI_API_KEY"),
    "custom-anthropic": (None, "CUSTOM_ANTHROPIC_API_KEY"),
}

# Anthropic-routed providers that support extended thinking.
_THINKING_CAPABLE_PROVIDERS: set[str] = {"minimax"}

# Moonshot rejects a forced tool choice while thinking is enabled, and kimi-k3
# cannot disable thinking — structured output must use json_schema there.
# Moonshot-specific: do NOT widen to other mandatory-reasoning models.
_OPENROUTER_JSON_SCHEMA_STRUCTURED_OUTPUT_MODELS = frozenset(
    {"moonshotai/kimi-k3", "moonshotai/kimi-k3-20260715"}
)


def _is_mandatory_thinking_kimi(model_id: str) -> bool:
    """True for Kimi models whose thinking cannot be disabled (K3 family)."""
    short_id = model_id.split("/")[-1]
    return short_id.startswith("kimi-k3") or short_id == "kimi-for-coding"


# Model registry: list of (short_name, model_id, provider)
# Allows same short_name across different providers.
_MODEL_ENTRIES: list[tuple[str, str, str]] = [
    # Custom Anthropic (third-party Claude-compatible endpoints, current-gen defaults)
    # Listed BEFORE native anthropic so MODELS dict defaults to native provider
    ("claude-sonnet-4-6", "claude-sonnet-4-6", "custom-anthropic"),
    ("claude-haiku-4-5", "claude-haiku-4-5", "custom-anthropic"),
    # Custom OpenAI (third-party OpenAI-compatible endpoints, 3 defaults)
    # Listed BEFORE native openai so MODELS dict defaults to native provider
    ("gpt-5.5-pro", "gpt-5.5-pro", "custom-openai"),
    ("gpt-5.5", "gpt-5.5", "custom-openai"),
    ("gpt-5.4", "gpt-5.4", "custom-openai"),
    ("gpt-5.3-codex", "gpt-5.3-codex", "custom-openai"),
    ("gpt-5-mini", "gpt-5-mini", "custom-openai"),
    # Atlas Cloud (OpenAI-compatible)
    ("qwen3.5-27b", "qwen/qwen3.5-27b", "atlascloud"),
    # Anthropic (current generation)
    ("claude-fable-5", "claude-fable-5", "anthropic"),
    ("claude-opus-5", "claude-opus-5", "anthropic"),
    ("claude-opus-4-8", "claude-opus-4-8", "anthropic"),
    ("claude-sonnet-5", "claude-sonnet-5", "anthropic"),
    ("claude-sonnet-4-6", "claude-sonnet-4-6", "anthropic"),
    ("claude-haiku-4-5", "claude-haiku-4-5", "anthropic"),
    # OpenAI
    ("gpt-5.6-sol", "gpt-5.6-sol", "openai"),
    ("gpt-5.6-terra", "gpt-5.6-terra", "openai"),
    ("gpt-5.6-luna", "gpt-5.6-luna", "openai"),
    ("gpt-5.5-pro", "gpt-5.5-pro", "openai"),
    ("gpt-5.5", "gpt-5.5", "openai"),
    ("gpt-5.4", "gpt-5.4", "openai"),
    ("gpt-5.4-mini", "gpt-5.4-mini", "openai"),
    ("gpt-5.4-nano", "gpt-5.4-nano", "openai"),
    ("gpt-5.3-codex", "gpt-5.3-codex", "openai"),
    ("gpt-5.2-codex", "gpt-5.2-codex", "openai"),
    ("gpt-5.2", "gpt-5.2", "openai"),
    ("gpt-5.1", "gpt-5.1", "openai"),
    ("gpt-5", "gpt-5", "openai"),
    ("gpt-5-mini", "gpt-5-mini", "openai"),
    ("gpt-5-nano", "gpt-5-nano", "openai"),
    # Google GenAI
    ("gemini-3.6-flash", "gemini-3.6-flash", "google-genai"),
    ("gemini-3.5-flash", "gemini-3.5-flash", "google-genai"),
    ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite", "google-genai"),
    ("gemini-3.1-pro", "gemini-3.1-pro-preview", "google-genai"),
    (
        "gemini-3.1-pro-customtools",
        "gemini-3.1-pro-preview-customtools",
        "google-genai",
    ),
    ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "google-genai"),
    ("gemini-3-flash", "gemini-3-flash-preview", "google-genai"),
    ("gemini-2.5-flash", "gemini-2.5-flash", "google-genai"),
    ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "google-genai"),
    ("gemini-2.5-pro", "gemini-2.5-pro", "google-genai"),
    # MiniMax (direct API — Anthropic-compatible; default: api.minimaxi.com, global: api.minimax.io)
    ("minimax-m3", "MiniMax-M3", "minimax"),
    ("minimax-m2.7", "MiniMax-M2.7", "minimax"),
    ("minimax-m2.7-highspeed", "MiniMax-M2.7-highspeed", "minimax"),
    ("minimax-m2.5", "MiniMax-M2.5", "minimax"),
    ("minimax-m2.5-highspeed", "MiniMax-M2.5-highspeed", "minimax"),
    # NVIDIA
    ("nemotron-super", "nvidia/nemotron-3-super-120b-a12b", "nvidia"),
    ("nemotron-nano", "nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
    ("glm-5.2", "z-ai/glm-5.2", "nvidia"),
    ("glm4.7", "z-ai/glm4.7", "nvidia"),
    ("deepseek-v3.2", "deepseek-ai/deepseek-v3.2", "nvidia"),
    ("deepseek-v3.1", "deepseek-ai/deepseek-v3.1-terminus", "nvidia"),
    ("kimi-k2.5", "moonshotai/kimi-k2.5", "nvidia"),
    ("kimi-k2-thinking", "moonshotai/kimi-k2-thinking", "nvidia"),
    ("minimax-m2.5", "minimaxai/minimax-m2.5", "nvidia"),
    ("minimax-m2.1", "minimaxai/minimax-m2.1", "nvidia"),
    ("qwen3.5-397b", "qwen/qwen3.5-397b-a17b", "nvidia"),
    ("step-3.5-flash", "stepfun-ai/step-3.5-flash", "nvidia"),
    # SiliconFlow
    ("minimax-m2.5", "Pro/MiniMaxAI/MiniMax-M2.5", "siliconflow"),
    ("glm-5.2", "Pro/zai-org/GLM-5.2", "siliconflow"),
    ("glm-5", "Pro/zai-org/GLM-5", "siliconflow"),
    ("kimi-k2.5", "Pro/moonshotai/Kimi-K2.5", "siliconflow"),
    ("glm-4.7", "Pro/zai-org/GLM-4.7", "siliconflow"),
    # Requesty (aggregator — OpenAI-compatible router, provider/model IDs).
    # Listed before OpenRouter so that for model names shared with OpenRouter
    # or a native provider, Requesty does not override them (the dict below is
    # last-entry-wins); Requesty is selected explicitly via get_models_for_provider.
    ("claude-sonnet-4.6", "anthropic/claude-sonnet-4-6", "requesty"),
    ("claude-opus-4.8", "anthropic/claude-opus-4-8", "requesty"),
    ("gemini-3.5-flash", "google/gemini-3.5-flash", "requesty"),
    ("grok-4.3", "xai/grok-4.3", "requesty"),
    ("grok-build-0.1", "xai/grok-build-0.1", "requesty"),
    # OpenRouter
    ("claude-fable-5", "anthropic/claude-fable-5", "openrouter"),
    ("claude-opus-5", "anthropic/claude-opus-5", "openrouter"),
    ("claude-opus-5-fast", "anthropic/claude-opus-5-fast", "openrouter"),
    ("claude-opus-4.8", "anthropic/claude-opus-4.8", "openrouter"),
    ("claude-opus-4.8-fast", "anthropic/claude-opus-4.8-fast", "openrouter"),
    ("claude-sonnet-5", "anthropic/claude-sonnet-5", "openrouter"),
    ("claude-sonnet-4.6", "anthropic/claude-sonnet-4.6", "openrouter"),
    ("gpt-5.6-sol", "openai/gpt-5.6-sol", "openrouter"),
    ("gpt-5.6-terra", "openai/gpt-5.6-terra", "openrouter"),
    ("gpt-5.6-luna", "openai/gpt-5.6-luna", "openrouter"),
    ("gpt-5.5-pro", "openai/gpt-5.5-pro", "openrouter"),
    ("gpt-5.5", "openai/gpt-5.5", "openrouter"),
    ("gpt-5.4", "openai/gpt-5.4", "openrouter"),
    ("gpt-5.3-codex", "openai/gpt-5.3-codex", "openrouter"),
    ("gemini-3.6-flash", "google/gemini-3.6-flash", "openrouter"),
    ("gemini-3.5-flash", "google/gemini-3.5-flash", "openrouter"),
    ("gemini-3.5-flash-lite", "google/gemini-3.5-flash-lite", "openrouter"),
    ("gemini-3.1-pro", "google/gemini-3.1-pro-preview", "openrouter"),
    ("gemini-3-flash", "google/gemini-3-flash-preview", "openrouter"),
    ("kimi-k3", "moonshotai/kimi-k3", "openrouter"),
    ("kimi-k2.6", "moonshotai/kimi-k2.6", "openrouter"),
    ("glm-5.2", "z-ai/glm-5.2", "openrouter"),
    ("glm-5v-turbo", "z-ai/glm-5v-turbo", "openrouter"),
    ("minimax-m3", "minimax/minimax-m3", "openrouter"),
    ("mimo-v2.5-pro", "xiaomi/mimo-v2.5-pro", "openrouter"),
    ("mimo-v2.5", "xiaomi/mimo-v2.5", "openrouter"),
    ("grok-build-0.1", "x-ai/grok-build-0.1", "openrouter"),
    ("grok-4.5", "x-ai/grok-4.5", "openrouter"),
    ("hy3", "tencent/hy3", "openrouter"),
    ("qwen3.8-max", "qwen/qwen3.8-max", "openrouter"),
    ("qwen3.7-max", "qwen/qwen3.7-max", "openrouter"),
    ("qwen3.7-plus", "qwen/qwen3.7-plus", "openrouter"),
    ("qwen3.6-flash", "qwen/qwen3.6-flash", "openrouter"),
    ("qwen3.5-122b", "qwen/qwen3.5-122b-a10b", "openrouter"),
    ("deepseek-v4-pro", "deepseek/deepseek-v4-pro", "openrouter"),
    ("deepseek-v4-flash", "deepseek/deepseek-v4-flash", "openrouter"),
    # Volcengine Coding Plan (火山引擎代码计划 — coding-only endpoint)
    # Listed before Zhipu so simple GLM lookups keep their existing default.
    ("glm-5.2", "glm-5-2", "volcengine-code"),
    ("kimi-k2.5", "kimi-k2-5", "volcengine-code"),
    # Zhipu CodePlan (智谱代码计划 — coding-only endpoint)
    ("glm-5.2", "glm-5.2", "zhipu-code"),
    ("glm-5.1", "glm-5.1", "zhipu-code"),
    ("glm-5", "glm-5", "zhipu-code"),
    ("glm-5-turbo", "glm-5-turbo", "zhipu-code"),
    ("glm-5v-turbo", "glm-5v-turbo", "zhipu-code"),
    ("glm-4.7", "glm-4.7", "zhipu-code"),
    # Zhipu (智谱 — general endpoint, default for simple lookups)
    ("glm-5.2", "glm-5.2", "zhipu"),
    ("glm-5.1", "glm-5.1", "zhipu"),
    ("glm-5", "glm-5", "zhipu"),
    ("glm-5-turbo", "glm-5-turbo", "zhipu"),
    ("glm-5v-turbo", "glm-5v-turbo", "zhipu"),
    ("glm-4.7", "glm-4.7", "zhipu"),
    # Volcengine (火山引擎 — Doubao models)
    ("doubao-seed-2.0-pro", "doubao-seed-2-0-pro-260215", "volcengine"),
    ("doubao-seed-2.0-lite", "doubao-seed-2-0-lite-260215", "volcengine"),
    ("doubao-seed-2.0-mini", "doubao-seed-2-0-mini-260215", "volcengine"),
    ("doubao-seed-2.0-code", "doubao-seed-2-0-code-preview-260215", "volcengine"),
    ("doubao-seed-1.6", "doubao-seed-1.6", "volcengine"),
    ("doubao-1.5-pro", "doubao-1.5-pro-256k", "volcengine"),
    ("doubao-1.5-thinking-pro", "doubao-1.5-thinking-pro", "volcengine"),
    # DashScope Coding Plan (阿里云代码计划 — subscription sk-sp-* endpoint)
    ("qwen3.8-max", "qwen3.8-max", "dashscope-code"),
    ("qwen3.7-max", "qwen3.7-max", "dashscope-code"),
    ("qwen3.7-plus", "qwen3.7-plus", "dashscope-code"),
    ("qwen3.6-max", "qwen3.6-max-preview", "dashscope-code"),
    ("qwen3.6-plus", "qwen3.6-plus", "dashscope-code"),
    ("qwen3.6-flash", "qwen3.6-flash", "dashscope-code"),
    ("qwen3-coder", "qwen3-coder-plus", "dashscope-code"),
    ("qwen3-coder-next", "qwen3-coder-next", "dashscope-code"),
    ("qwen3-max", "qwen3-max", "dashscope-code"),
    ("qwen3.5-plus", "qwen3.5-plus", "dashscope-code"),
    # DashScope (阿里云 — Qwen models, default for simple lookups)
    ("qwen3.8-max", "qwen3.8-max", "dashscope"),
    ("qwen3.7-max", "qwen3.7-max", "dashscope"),
    ("qwen3.7-plus", "qwen3.7-plus", "dashscope"),
    ("qwen3.6-max", "qwen3.6-max-preview", "dashscope"),
    ("qwen3.6-plus", "qwen3.6-plus", "dashscope"),
    ("qwen3.6-flash", "qwen3.6-flash", "dashscope"),
    ("qwen3-coder", "qwen3-coder-plus", "dashscope"),
    ("qwen3-235b", "qwen3-235b-a22b", "dashscope"),
    ("qwen-max", "qwen-max", "dashscope"),
    ("qwq-plus", "qwq-plus", "dashscope"),
    # DeepSeek
    ("deepseek-v4-pro", "deepseek-v4-pro", "deepseek"),
    ("deepseek-v4-flash", "deepseek-v4-flash", "deepseek"),
    # Legacy aliases (deprecated 2026-07-24; route to v4-flash thinking/non-thinking)
    ("deepseek-r1", "deepseek-reasoner", "deepseek"),
    ("deepseek-v3", "deepseek-chat", "deepseek"),
    # Moonshot (OpenAI-compatible)
    ("kimi-k3", "kimi-k3", "moonshot"),
    ("kimi-k2.6", "kimi-k2.6", "moonshot"),
    ("kimi-k2.5", "kimi-k2.5", "moonshot"),
    ("kimi-k2-thinking", "kimi-k2-thinking", "moonshot"),
    ("kimi-k2-thinking-turbo", "kimi-k2-thinking-turbo", "moonshot"),
    ("moonshot-v1-auto", "moonshot-v1-auto", "moonshot"),
    ("moonshot-v1-128k", "moonshot-v1-128k", "moonshot"),
    ("moonshot-v1-32k", "moonshot-v1-32k", "moonshot"),
    ("moonshot-v1-8k", "moonshot-v1-8k", "moonshot"),
    # Kimi Coding Plan (Anthropic-compatible)
    ("kimi-for-coding", "kimi-for-coding", "kimi-coding"),
]

# Public dict for simple lookups (last entry wins for duplicate names).
# Use get_models_for_provider() for provider-aware lookups.
MODELS: dict[str, tuple[str, str]] = {
    name: (model_id, provider) for name, model_id, provider in _MODEL_ENTRIES
}

DEFAULT_MODEL = "claude-sonnet-4-6"


def get_models_for_provider(provider: str) -> list[tuple[str, str]]:
    """Get all models for a specific provider.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openrouter').

    Returns:
        List of (short_name, model_id) tuples for the provider.
    """
    return [(name, model_id) for name, model_id, p in _MODEL_ENTRIES if p == provider]


def list_models() -> list[str]:
    """List all available model short names.

    Returns:
        List of unique model short names that can be passed to get_chat_model().
    """
    seen = set()
    result = []
    for name, _, _ in _MODEL_ENTRIES:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def list_models_by_provider() -> list[tuple[str, str, str]]:
    """List all unique (short_name, model_id, provider) entries.

    Returns:
        De-duplicated list of model entries preserving registry order.
    """
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str, str]] = []
    for name, model_id, provider in _MODEL_ENTRIES:
        key = (name, provider)
        if key not in seen:
            seen.add(key)
            result.append((name, model_id, provider))
    return result


async def list_model_picker_entries(
    ollama_base_url: str | None,
    *,
    include_custom_ollama: bool,
) -> list[tuple[str, str, str]]:
    """Return model picker entries, optionally including local Ollama models."""
    entries = list_models_by_provider()
    if ollama_base_url:
        from .ollama_discovery import discover_ollama_models

        for detected_name in await discover_ollama_models(
            ollama_base_url,
            timeout=1.5,
        ):
            entries.append((detected_name, detected_name, "ollama"))
        if include_custom_ollama:
            entries.append(("Custom Ollama model...", "__custom_ollama__", "ollama"))
    return entries


def get_model_info(model: str) -> tuple[str, str] | None:
    """Get the (model_id, provider) tuple for a short name.

    Args:
        model: Short model name.

    Returns:
        Tuple of (model_id, provider) or None if not found.
    """
    return MODELS.get(model)
