"""LLM model configuration based on LangChain init_chat_model.

This module provides a unified interface for creating chat model instances
with support for multiple providers (Anthropic, OpenAI) and convenient
short names for common models.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model

# Model registry: short_name -> (model_id, provider)
MODELS: dict[str, tuple[str, str]] = {
    # Anthropic
    "claude-sonnet-4-5": ("claude-sonnet-4-5-20250929", "anthropic"),
    "claude-opus-4-5": ("claude-opus-4-5-20251101", "anthropic"),
    "claude-3-5-sonnet": ("claude-3-5-sonnet-20241022", "anthropic"),
    "claude-3-5-haiku": ("claude-3-5-haiku-20241022", "anthropic"),
    # OpenAI
    "gpt-4o": ("gpt-4o", "openai"),
    "gpt-4o-mini": ("gpt-4o-mini", "openai"),
    "o1": ("o1", "openai"),
    "o1-mini": ("o1-mini", "openai"),
    # Google GenAI
    "gemini-3-pro": ("gemini-3-pro-preview", "google-genai"),
    "gemini-3-flash": ("gemini-3-flash-preview", "google-genai"),
    "gemini-2.5-flash": ("gemini-2.5-flash", "google-genai"),
    "gemini-2.5-flash-lite": ("gemini-2.5-flash-lite", "google-genai"),
    "gemini-2.5-pro": ("gemini-2.5-pro", "google-genai"),
    # NVIDIA
    "glm4.7": ("z-ai/glm4.7", "nvidia"),
    "deepseek-v3.1": ("deepseek-ai/deepseek-v3.1-terminus", "nvidia"),
    "nemotron-nano": ("nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
}

DEFAULT_MODEL = "claude-sonnet-4-5"


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

    # Look up short name in registry
    if model in MODELS:
        model_id, default_provider = MODELS[model]
        provider = provider or default_provider
    else:
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
            elif "/" in model_id:
                provider = "nvidia"
            else:
                provider = "anthropic"  # Default fallback

    # Auto-enable thinking for Anthropic models
    if provider == "anthropic" and "thinking" not in kwargs:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}

    return init_chat_model(model=model_id, model_provider=provider, **kwargs)


def list_models() -> list[str]:
    """List all available model short names.

    Returns:
        List of model short names that can be passed to get_chat_model().
    """
    return list(MODELS.keys())


def get_model_info(model: str) -> tuple[str, str] | None:
    """Get the (model_id, provider) tuple for a short name.

    Args:
        model: Short model name.

    Returns:
        Tuple of (model_id, provider) or None if not found.
    """
    return MODELS.get(model)
