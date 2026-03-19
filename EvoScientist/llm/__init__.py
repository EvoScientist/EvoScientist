"""LLM module for EvoScientist.

Provides a unified interface for creating chat model instances
with support for multiple providers.
"""

from .models import (
    DEFAULT_MODEL,
    MODELS,
    get_chat_model,
    get_model_info,
    get_models_for_provider,
    list_models,
)

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "get_chat_model",
    "get_model_info",
    "get_models_for_provider",
    "list_models",
]
