"""LLM module for EvoScientist.

Provides a unified interface for creating chat model instances
with support for multiple providers.
"""

from .models import (
    MODELS,
    DEFAULT_MODEL,
    get_chat_model,
    list_models,
    get_model_info,
)

__all__ = [
    "MODELS",
    "DEFAULT_MODEL",
    "get_chat_model",
    "list_models",
    "get_model_info",
]
