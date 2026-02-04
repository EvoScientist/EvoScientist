"""EvoScientist Agent — AI-powered research & code execution."""

from .backends import CustomSandboxBackend, ReadOnlyFilesystemBackend
from .config import (
    EvoScientistConfig,
    load_config,
    save_config,
    get_effective_config,
    get_config_path,
)
from .llm import get_chat_model, MODELS, list_models, DEFAULT_MODEL
from .middleware import create_skills_middleware
from .prompts import get_system_prompt, RESEARCHER_INSTRUCTIONS
from .tools import tavily_search, think_tool
from .EvoScientist import EvoScientist_agent, create_cli_agent

__all__ = [
    # Agent graph (main export)
    "EvoScientist_agent",
    "create_cli_agent",
    # Backends
    "CustomSandboxBackend",
    "ReadOnlyFilesystemBackend",
    # Configuration
    "EvoScientistConfig",
    "load_config",
    "save_config",
    "get_effective_config",
    "get_config_path",
    # LLM
    "get_chat_model",
    "MODELS",
    "list_models",
    "DEFAULT_MODEL",
    # Middleware
    "create_skills_middleware",
    # Prompts
    "get_system_prompt",
    "RESEARCHER_INSTRUCTIONS",
    # Tools
    "tavily_search",
    "think_tool",
]
