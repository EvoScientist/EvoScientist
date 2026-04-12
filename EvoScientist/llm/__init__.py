"""LLM module for EvoScientist.

Provides a unified interface for creating chat model instances
with support for multiple providers.
"""

from .context_window import (
    DEFAULT_CONTEXT_WINDOW_FALLBACK,
    get_context_window,
    resolve_context_window,
)
from .models import (
    DEFAULT_MODEL,
    MODELS,
    get_chat_model,
    get_model_info,
    get_models_for_provider,
    list_models,
)

__all__ = [
    "DEFAULT_CONTEXT_WINDOW_FALLBACK",
    "DEFAULT_MODEL",
    "MODELS",
    "get_chat_model",
    "get_context_window",
    "get_model_info",
    "get_models_for_provider",
    "list_models",
    "resolve_context_window",
]
