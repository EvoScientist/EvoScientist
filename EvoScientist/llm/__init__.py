"""LLM module for EvoScientist.

Provides a unified interface for creating chat model instances
with support for multiple providers.

``models`` is attached lazily via :mod:`lazy_loader` (SPEC-1 / PEP 562) so
that importing ``EvoScientist.llm`` (or any of its submodules, like
``context_window``) does not eagerly drag in ``langchain.chat_models`` and
its transitive ``langchain_anthropic``/``langchain_openai`` stack — that's
roughly 1 s of wall time on every CLI invocation.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    submodules=["context_window", "models", "patches"],
    submod_attrs={
        "context_window": [
            "DEFAULT_CONTEXT_WINDOW_FALLBACK",
            "get_context_window",
            "resolve_context_window",
        ],
        "models": [
            "DEFAULT_MODEL",
            "MODELS",
            "get_chat_model",
            "get_model_info",
            "get_models_for_provider",
            "list_models",
        ],
    },
)
