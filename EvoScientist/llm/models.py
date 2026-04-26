"""LLM model configuration based on LangChain init_chat_model.

This module provides a unified interface for creating chat model instances
with support for multiple providers (Anthropic, OpenAI, Google GenAI, MiniMax
(Anthropic-compatible), NVIDIA, SiliconFlow, OpenRouter, ZhipuAI, Volcengine,
DashScope, DeepSeek, Ollama, and custom OpenAI/Anthropic-compatible endpoints) and
convenient short names for common models.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model

from .patches import (
    _is_ccproxy_codex,
    _patch_ccproxy_system_to_developer,
    _patch_deepseek_reasoning_passback,
    _patch_openai_compat_content,
    _patch_openrouter_reasoning_details,
)

_MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"
_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
_ZHIPU_CODE_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
_VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
_KIMI_CODING_BASE_URL = "https://api.kimi.com/coding/"

# Providers routed through the OpenAI provider with a custom base_url.
# Maps provider name → (base_url or None, env var for API key).
_OPENAI_ROUTED_PROVIDERS: dict[str, tuple[str | None, str]] = {
    "deepseek": (_DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY"),
    "moonshot": (_MOONSHOT_BASE_URL, "MOONSHOT_API_KEY"),
    "siliconflow": (_SILICONFLOW_BASE_URL, "SILICONFLOW_API_KEY"),
    "zhipu": (_ZHIPU_BASE_URL, "ZHIPU_API_KEY"),
    "zhipu-code": (_ZHIPU_CODE_BASE_URL, "ZHIPU_API_KEY"),
    "volcengine": (_VOLCENGINE_BASE_URL, "VOLCENGINE_API_KEY"),
    "dashscope": (_DASHSCOPE_BASE_URL, "DASHSCOPE_API_KEY"),
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

# Model registry: list of (short_name, model_id, provider)
# Allows same short_name across different providers.
_MODEL_ENTRIES: list[tuple[str, str, str]] = [
    # Custom Anthropic (third-party Claude-compatible endpoints, 3 defaults)
    # Listed BEFORE native anthropic so MODELS dict defaults to native provider
    ("claude-sonnet-4-6", "claude-sonnet-4-6", "custom-anthropic"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5", "custom-anthropic"),
    ("claude-haiku-4-5", "claude-haiku-4-5", "custom-anthropic"),
    # Custom OpenAI (third-party OpenAI-compatible endpoints, 3 defaults)
    # Listed BEFORE native openai so MODELS dict defaults to native provider
    ("gpt-5.5", "gpt-5.5", "custom-openai"),
    ("gpt-5.4", "gpt-5.4", "custom-openai"),
    ("gpt-5.3-codex", "gpt-5.3-codex", "custom-openai"),
    ("gpt-5-mini", "gpt-5-mini", "custom-openai"),
    # Anthropic (ordered by capability)
    ("claude-opus-4-7", "claude-opus-4-7", "anthropic"),
    ("claude-opus-4-6", "claude-opus-4-6", "anthropic"),
    ("claude-sonnet-4-6", "claude-sonnet-4-6", "anthropic"),
    ("claude-opus-4-5", "claude-opus-4-5", "anthropic"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5", "anthropic"),
    ("claude-haiku-4-5", "claude-haiku-4-5", "anthropic"),
    # OpenAI
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
    ("minimax-m2.7", "MiniMax-M2.7", "minimax"),
    ("minimax-m2.7-highspeed", "MiniMax-M2.7-highspeed", "minimax"),
    ("minimax-m2.5", "MiniMax-M2.5", "minimax"),
    ("minimax-m2.5-highspeed", "MiniMax-M2.5-highspeed", "minimax"),
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
    ("claude-opus-4.7", "anthropic/claude-opus-4.7", "openrouter"),
    ("claude-opus-4.6", "anthropic/claude-opus-4.6", "openrouter"),
    ("claude-sonnet-4.6", "anthropic/claude-sonnet-4.6", "openrouter"),
    ("gpt-5.5", "openai/gpt-5.5", "openrouter"),
    ("gpt-5.4", "openai/gpt-5.4", "openrouter"),
    ("gpt-5.3-codex", "openai/gpt-5.3-codex", "openrouter"),
    ("gemini-3.1-pro", "google/gemini-3.1-pro-preview", "openrouter"),
    ("gemini-3-flash", "google/gemini-3-flash-preview", "openrouter"),
    ("kimi-k2.5", "moonshotai/kimi-k2.5", "openrouter"),
    ("glm-5v-turbo", "z-ai/glm-5v-turbo", "openrouter"),
    ("minimax-m2.7", "minimax/minimax-m2.7", "openrouter"),
    ("mimo-v2-pro", "xiaomi/mimo-v2-pro", "openrouter"),
    ("grok-4.1-fast", "x-ai/grok-4.1-fast", "openrouter"),
    ("qwen3.5-122b", "qwen/qwen3.5-122b-a10b", "openrouter"),
    ("deepseek-v4-pro", "deepseek/deepseek-v4-pro", "openrouter"),
    ("deepseek-v4-flash", "deepseek/deepseek-v4-flash", "openrouter"),
    # Zhipu CodePlan (智谱代码计划 — coding-only endpoint)
    ("glm-5.1", "glm-5.1", "zhipu-code"),
    ("glm-5", "glm-5", "zhipu-code"),
    ("glm-5-turbo", "glm-5-turbo", "zhipu-code"),
    ("glm-5v-turbo", "glm-5v-turbo", "zhipu-code"),
    ("glm-4.7", "glm-4.7", "zhipu-code"),
    # Zhipu (智谱 — general endpoint, default for simple lookups)
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
    # DashScope (阿里云 — Qwen models)
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


def _apply_auto_config(
    provider: str,
    model_id: str,
    is_third_party: bool,
    kwargs: dict[str, Any],
    original_provider: str | None = None,
) -> None:
    """Auto-enable provider-specific features (thinking, reasoning, etc.).

    Mutates *kwargs* in place.  Only sets keys that the caller hasn't already
    provided, so explicit user settings are never overridden.
    """
    # Anthropic: extended thinking
    if provider == "anthropic" and "thinking" not in kwargs:
        _supports_thinking = original_provider in _THINKING_CAPABLE_PROVIDERS
        # Detect local proxy (e.g. ccproxy): thinking blocks in conversation
        # history cause 422 errors because the proxy doesn't accept 'thinking'
        # as a valid content block type on round-trip.
        if not is_third_party:
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
            _is_proxy = "127.0.0.1" in base_url or "localhost" in base_url
        else:
            _is_proxy = False
        if _is_proxy or (is_third_party and not _supports_thinking):
            pass
        elif model_id.endswith(("4-6", "4-7")):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs.setdefault("effort", "max")
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    # OpenAI (native, not third-party routed): reasoning
    if provider == "openai" and not is_third_party and "reasoning" not in kwargs:
        if _is_ccproxy_codex():
            # ccproxy uses Chat Completions which doesn't support reasoning.
            pass
        else:
            _eff = (
                "xhigh"
                if ("5.4" in model_id or "5.5" in model_id or "codex" in model_id)
                else "high"
            )
            kwargs["reasoning"] = {"effort": _eff, "summary": "auto"}

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

    # Anthropic base_url override (e.g. ccproxy at localhost:8000/api/v1)
    _is_third_party = (
        provider in _OPENAI_ROUTED_PROVIDERS or provider in _ANTHROPIC_ROUTED_PROVIDERS
    )
    _is_openai_proxy = False
    _original_provider: str | None = None
    if provider == "anthropic":
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    # Native OpenAI base_url override (e.g. ccproxy Codex at localhost:8000/codex/v1)
    elif provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
            _is_openai_proxy = _is_ccproxy_codex()
            if _is_openai_proxy:
                # Use Responses API for ccproxy: bypasses the format chain
                # converter (Chat→Responses→Chat) which returns 502 on
                # complex responses.  System messages are converted to
                # developer role by _patch_ccproxy_system_to_developer().
                kwargs.setdefault("use_responses_api", True)
                # Streaming must stay ON for Responses API: ccproxy's
                # StreamingBufferService loses output when assembling
                # non-streaming responses.  (The old streaming=False was
                # for Chat Completions tool_call duplication — not an issue
                # with the Responses API SSE format.)
                kwargs.pop("streaming", None)  # remove if set elsewhere
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    # OpenAI-routed providers → route through OpenAI provider with base_url
    elif provider in _OPENAI_ROUTED_PROVIDERS:
        _original_provider = provider
        base_url_default, api_key_env = _OPENAI_ROUTED_PROVIDERS[provider]
        if provider == "custom-openai":
            base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "CUSTOM_OPENAI_BASE_URL environment variable is required when using "
                    "the 'custom-openai' provider. Please set it to your "
                    "OpenAI-compatible API endpoint URL (e.g. https://api.openai.com/v1)."
                )
            base_url = base_url.rstrip("/")
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
        # Moonshot: disable thinking for all models to prevent LangChain from dropping
        # reasoning_content, which causes multi-turn conversation errors (error 20015).
        # Even native thinking models like kimi-k2-thinking operate in non-thinking mode.
        if provider == "moonshot":
            kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
        provider = "openai"

    # OpenRouter → native ChatOpenRouter via init_chat_model.
    elif provider == "openrouter":
        _patch_openrouter_reasoning_details()
        _is_third_party = True
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
        # Enable reasoning; disable summary to avoid multi-turn schema errors.
        effort = os.environ.get("EVOSCIENTIST_REASONING_EFFORT", "").strip() or "high"
        kwargs.setdefault("reasoning", {"effort": effort, "summary": "disabled"})

    # Anthropic-routed providers → route through Anthropic provider with base_url
    elif provider in _ANTHROPIC_ROUTED_PROVIDERS:
        _original_provider = provider
        base_url_default, api_key_env = _ANTHROPIC_ROUTED_PROVIDERS[provider]
        if provider == "custom-anthropic":
            base_url = os.environ.get("CUSTOM_ANTHROPIC_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "CUSTOM_ANTHROPIC_BASE_URL environment variable is required when using "
                    "the 'custom-anthropic' provider. Please set it to your "
                    "Anthropic-compatible API endpoint URL (e.g. https://api.anthropic.com)."
                )
            base_url = base_url.rstrip("/")
        elif provider == "minimax":
            base_url = os.environ.get("MINIMAX_BASE_URL", base_url_default).rstrip("/")
        else:
            base_url = base_url_default
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get(api_key_env, "")
        if api_key:
            kwargs["api_key"] = api_key
        # Kimi Coding Plan requires claude-code User-Agent header
        if provider == "kimi-coding":
            kwargs.setdefault("default_headers", {})["User-Agent"] = "claude-code/0.1.0"
        provider = "anthropic"

    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url

    _apply_auto_config(provider, model_id, _is_third_party, kwargs, _original_provider)

    # User-level override for the OpenAI Responses API vs Chat Completions.
    # When "false", force Chat Completions and drop reasoning (which triggers
    # the Responses API path in langchain-openai). Only applies to OpenAI.
    if provider == "openai":
        _responses_api_setting = (
            os.environ.get("EVOSCIENTIST_USE_RESPONSES_API", "").strip().lower()
        )
        if _responses_api_setting == "false":
            kwargs["use_responses_api"] = False
            kwargs.pop("reasoning", None)
        elif _responses_api_setting == "true":
            kwargs["use_responses_api"] = True

    chat_model = init_chat_model(model=model_id, model_provider=provider, **kwargs)

    # Flatten list content to strings for strict OpenAI-compatible providers
    # (DeepSeek, SiliconFlow, OpenRouter, custom-openai, etc.) and
    # native OpenAI through a proxy, to avoid "sequence expected string" errors.
    # Moonshot and Kimi Coding support standard format, no patch needed.
    _no_patch_providers = {"moonshot", "kimi-coding"}
    if (
        _is_third_party or _is_openai_proxy
    ) and _original_provider not in _no_patch_providers:
        _patch_openai_compat_content(chat_model)

    # DeepSeek thinking mode requires reasoning_content passback in multi-turn
    # + tool_use scenarios.
    if _original_provider == "deepseek":
        _patch_deepseek_reasoning_passback(chat_model)

    if _is_openai_proxy:
        _patch_ccproxy_system_to_developer(chat_model)

    return chat_model


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


def get_model_info(model: str) -> tuple[str, str] | None:
    """Get the (model_id, provider) tuple for a short name.

    Args:
        model: Short model name.

    Returns:
        Tuple of (model_id, provider) or None if not found.
    """
    return MODELS.get(model)
