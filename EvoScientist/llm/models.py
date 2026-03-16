"""LLM model configuration based on LangChain init_chat_model.

This module provides a unified interface for creating chat model instances
with support for multiple providers (Anthropic, OpenAI, Google GenAI, NVIDIA,
SiliconFlow, OpenRouter, ZhipuAI, Volcengine, DashScope, Ollama, and custom
OpenAI-compatible endpoints) and convenient short names for common models.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model

_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
_ZHIPU_CODE_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
_VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Third-party providers routed through the OpenAI provider with a custom base_url.
# Maps provider name → (base_url or None, env var for API key).
_THIRD_PARTY_PROVIDERS: dict[str, tuple[str | None, str]] = {
    "siliconflow": (_SILICONFLOW_BASE_URL, "SILICONFLOW_API_KEY"),
    "openrouter": (_OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"),
    "zhipu": (_ZHIPU_BASE_URL, "ZHIPU_API_KEY"),
    "zhipu-code": (_ZHIPU_CODE_BASE_URL, "ZHIPU_API_KEY"),
    "volcengine": (_VOLCENGINE_BASE_URL, "VOLCENGINE_API_KEY"),
    "dashscope": (_DASHSCOPE_BASE_URL, "DASHSCOPE_API_KEY"),
    "custom": (None, "CUSTOM_API_KEY"),  # base_url from CUSTOM_BASE_URL env
}

# Model registry: list of (short_name, model_id, provider)
# Allows same short_name across different providers.
_MODEL_ENTRIES: list[tuple[str, str, str]] = [
    # Anthropic (ordered by capability)
    ("claude-opus-4-6", "claude-opus-4-6", "anthropic"),
    ("claude-sonnet-4-6", "claude-sonnet-4-6", "anthropic"),
    ("claude-opus-4-5", "claude-opus-4-5-20251101", "anthropic"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929", "anthropic"),
    ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "anthropic"),
    # OpenAI
    ("gpt-5.4", "gpt-5.4-2026-03-05", "openai"),
    ("gpt-5.3-codex", "gpt-5.3-codex", "openai"),
    ("gpt-5.2-codex", "gpt-5.2-codex", "openai"),
    ("gpt-5.2", "gpt-5.2-2025-12-11", "openai"),
    ("gpt-5.1", "gpt-5.1-2025-11-13", "openai"),
    ("gpt-5", "gpt-5-2025-08-07", "openai"),
    ("gpt-5-mini", "gpt-5-mini-2025-08-07", "openai"),
    ("gpt-5-nano", "gpt-5-nano-2025-08-07", "openai"),
    # Google GenAI
    ("gemini-3.1-pro", "gemini-3.1-pro-preview", "google-genai"),
    ("gemini-3.1-pro-customtools", "gemini-3.1-pro-preview-customtools", "google-genai"),
    ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "google-genai"),
    ("gemini-3-flash", "gemini-3-flash-preview", "google-genai"),
    ("gemini-2.5-flash", "gemini-2.5-flash", "google-genai"),
    ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "google-genai"),
    ("gemini-2.5-pro", "gemini-2.5-pro", "google-genai"),
    # NVIDIA
    ("nemotron-super", "nvidia/nemotron-3-super-120b-a12b", "nvidia"),
    ("nemotron-nano", "nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
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
    ("glm-5", "Pro/zai-org/GLM-5", "siliconflow"),
    ("kimi-k2.5", "Pro/moonshotai/Kimi-K2.5", "siliconflow"),
    ("glm-4.7", "Pro/zai-org/GLM-4.7", "siliconflow"),
    # OpenRouter
    ("minimax-m2.5", "minimax/minimax-m2.5", "openrouter"),
    ("grok-4.1-fast", "x-ai/grok-4.1-fast", "openrouter"),
    ("qwen3.5-122b", "qwen/qwen3.5-122b-a10b", "openrouter"),
    ("gemini-3-flash", "google/gemini-3-flash-preview", "openrouter"),
    ("claude-sonnet-4.6", "anthropic/claude-sonnet-4.6", "openrouter"),
    # Zhipu CodePlan (智谱代码计划 — coding-only endpoint)
    ("glm-5", "glm-5", "zhipu-code"),
    ("glm-4.7", "glm-4.7", "zhipu-code"),
    # Zhipu (智谱 — general endpoint, default for simple lookups)
    ("glm-5", "glm-5", "zhipu"),
    ("glm-4.7", "glm-4.7", "zhipu"),
    # Volcengine (火山引擎 — Doubao models)
    ("doubao-seed-1.6", "doubao-seed-1.6", "volcengine"),
    ("doubao-1.5-pro", "doubao-1.5-pro-256k", "volcengine"),
    ("doubao-1.5-thinking-pro", "doubao-1.5-thinking-pro", "volcengine"),
    # DashScope (阿里云 — Qwen models)
    ("qwen3-coder", "qwen3-coder-plus", "dashscope"),
    ("qwen3-235b", "qwen3-235b-a22b", "dashscope"),
    ("qwen-max", "qwen-max", "dashscope"),
    ("qwq-plus", "qwq-plus", "dashscope"),
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
    return [
        (name, model_id)
        for name, model_id, p in _MODEL_ENTRIES
        if p == provider
    ]


def _apply_auto_config(
    provider: str,
    model_id: str,
    is_third_party: bool,
    kwargs: dict[str, Any],
) -> None:
    """Auto-enable provider-specific features (thinking, reasoning, etc.).

    Mutates *kwargs* in place.  Only sets keys that the caller hasn't already
    provided, so explicit user settings are never overridden.
    """
    # Anthropic: extended thinking
    if provider == "anthropic" and "thinking" not in kwargs:
        if model_id.endswith("4-6"):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs.setdefault("effort", "max")
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    # OpenAI (native, not third-party routed): reasoning
    if provider == "openai" and not is_third_party and "reasoning" not in kwargs:
        kwargs["reasoning"] = {"effort": "high", "summary": "auto"}

    # Google GenAI: surface thinking traces
    if provider == "google-genai":
        kwargs.setdefault("include_thoughts", True)

    # Ollama: separate reasoning content from response for thinking models
    if provider == "ollama" and "reasoning" not in kwargs:
        kwargs["reasoning"] = True


def get_chat_model(
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """Get a chat model instance.

    Args:
        model: Model name (short name like 'claude-sonnet-4-5' or full ID
               like 'claude-sonnet-4-5-20250929'). Defaults to DEFAULT_MODEL.
        provider: Override the provider (e.g., 'anthropic', 'openai').
                  If not specified, inferred from model name or defaults to 'anthropic'.
        **kwargs: Additional arguments passed to init_chat_model (e.g., temperature).

    Returns:
        A LangChain chat model instance.

    Examples:
        >>> model = get_chat_model()  # Uses default (claude-sonnet-4-5)
        >>> model = get_chat_model("claude-opus-4-5")  # Use short name
        >>> model = get_chat_model("gpt-4o")  # OpenAI model
        >>> model = get_chat_model("claude-3-opus-20240229", provider="anthropic")  # Full ID
    """
    model = model or DEFAULT_MODEL

    # Look up short name in registry (provider-aware)
    model_id = None
    if provider:
        # Try exact match with provider first
        for name, mid, p in _MODEL_ENTRIES:
            if name == model and p == provider:
                model_id = mid
                break
    if model_id is None and model in MODELS:
        model_id, default_provider = MODELS[model]
        provider = provider or default_provider

    if model_id is None:
        # Assume it's a full model ID
        model_id = model
        # Try to infer provider from model ID prefix
        if provider is None:
            if model_id.startswith(("claude-", "anthropic")):
                provider = "anthropic"
            elif model_id.startswith(("gpt-", "o1", "davinci", "text-")):
                provider = "openai"
            elif model_id.startswith("gemini"):
                provider = "google-genai"
            elif model_id.startswith("ollama:"):
                provider = "ollama"
                model_id = model_id.removeprefix("ollama:")
            else:
                provider = "anthropic"  # Default fallback

    # Third-party providers → route through OpenAI provider with base_url
    _is_third_party = provider in _THIRD_PARTY_PROVIDERS
    if provider in _THIRD_PARTY_PROVIDERS:
        base_url_default, api_key_env = _THIRD_PARTY_PROVIDERS[provider]
        if provider == "custom":
            base_url = os.environ.get("CUSTOM_BASE_URL", "")
        else:
            base_url = base_url_default
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get(api_key_env, "")
        if api_key:
            kwargs["api_key"] = api_key
        # SiliconFlow: disable thinking — LangChain drops reasoning_content
        # from history, causing error 20015 on multi-turn requests.
        if provider == "siliconflow":
            kwargs.setdefault("extra_body", {})["enable_thinking"] = False
        provider = "openai"
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url

    _apply_auto_config(provider, model_id, _is_third_party, kwargs)

    return init_chat_model(model=model_id, model_provider=provider, **kwargs)


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


def get_model_info(model: str) -> tuple[str, str] | None:
    """Get the (model_id, provider) tuple for a short name.

    Args:
        model: Short model name.

    Returns:
        Tuple of (model_id, provider) or None if not found.
    """
    return MODELS.get(model)
