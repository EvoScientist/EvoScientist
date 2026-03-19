"""EvoScientist Agent - AI-powered research and code execution.

This package exposes a convenience API at the package root while keeping
imports lazy, so lightweight modules (for example config helpers) can be used
without importing heavy runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    # Agent graph (lazy to avoid expensive initialization at import time)
    "EvoScientist_agent": (".EvoScientist", "EvoScientist_agent"),
    "create_cli_agent": (".EvoScientist", "create_cli_agent"),
    # Backends
    "CustomSandboxBackend": (".backends", "CustomSandboxBackend"),
    "ReadOnlyFilesystemBackend": (".backends", "ReadOnlyFilesystemBackend"),
    # Configuration
    "EvoScientistConfig": (".config", "EvoScientistConfig"),
    "load_config": (".config", "load_config"),
    "save_config": (".config", "save_config"),
    "get_effective_config": (".config", "get_effective_config"),
    "get_config_path": (".config", "get_config_path"),
    # LLM
    "get_chat_model": (".llm", "get_chat_model"),
    "MODELS": (".llm", "MODELS"),
    "list_models": (".llm", "list_models"),
    "DEFAULT_MODEL": (".llm", "DEFAULT_MODEL"),
    # Prompts
    "get_system_prompt": (".prompts", "get_system_prompt"),
    "RESEARCHER_INSTRUCTIONS": (".prompts", "RESEARCHER_INSTRUCTIONS"),
    # Tools
    "tavily_search": (".tools", "tavily_search"),
    "think_tool": (".tools", "think_tool"),
    # Sessions
    "get_checkpointer": (".sessions", "get_checkpointer"),
    "generate_thread_id": (".sessions", "generate_thread_id"),
    "list_threads": (".sessions", "list_threads"),
    "delete_thread": (".sessions", "delete_thread"),
}


def __getattr__(name: str):
    """Lazily import and cache package-level attributes.

    Args:
        name: The attribute name to look up.

    Returns:
        The resolved attribute value.

    Raises:
        AttributeError: If the name is not in _EXPORTS.
    """
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    # Cache after first load to avoid repeated import lookups.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List available public attributes including lazy exports."""
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = list(_EXPORTS)
