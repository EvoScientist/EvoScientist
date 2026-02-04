"""Configuration management for EvoScientist.

Handles loading, saving, and merging configuration from multiple sources
with the following priority (highest to lowest):
    CLI arguments > Environment variables > Config file > Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Literal

import yaml


# =============================================================================
# Configuration paths
# =============================================================================

def get_config_dir() -> Path:
    """Get the configuration directory path.

    Uses XDG_CONFIG_HOME if set, otherwise ~/.config/evoscientist/
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "evoscientist"
    return Path.home() / ".config" / "evoscientist"


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return get_config_dir() / "config.yaml"


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass
class EvoScientistConfig:
    """EvoScientist configuration settings.

    Attributes:
        anthropic_api_key: Anthropic API key for Claude models.
        openai_api_key: OpenAI API key for GPT models.
        tavily_api_key: Tavily API key for web search.
        provider: Default LLM provider ('anthropic' or 'openai').
        model: Default model name (short name or full ID).
        default_mode: Default workspace mode ('daemon' or 'run').
        default_workdir: Default workspace directory (empty = use ./workspace).
        max_concurrent: Maximum concurrent sub-agents.
        max_iterations: Maximum delegation iterations.
        show_thinking: Whether to show thinking panels in CLI.
    """

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    nvidia_api_key: str = ""
    tavily_api_key: str = ""

    # LLM Settings
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"

    # Workspace Settings
    default_mode: Literal["daemon", "run"] = "daemon"
    default_workdir: str = ""

    # Agent Parameters
    max_concurrent: int = 3
    max_iterations: int = 3

    # UI Settings
    show_thinking: bool = True


# =============================================================================
# Config file operations
# =============================================================================

def load_config() -> EvoScientistConfig:
    """Load configuration from file.

    Returns:
        EvoScientistConfig instance with values from file, or defaults if
        file doesn't exist.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return EvoScientistConfig()

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # Filter to only valid fields
        valid_fields = {f.name for f in fields(EvoScientistConfig)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return EvoScientistConfig(**filtered_data)
    except Exception:
        # On any error, return defaults
        return EvoScientistConfig()


def save_config(config: EvoScientistConfig) -> None:
    """Save configuration to file.

    Args:
        config: EvoScientistConfig instance to save.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)

    # Save all fields including empty API keys (users can set them via env vars instead)
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def reset_config() -> None:
    """Reset configuration to defaults by deleting the config file."""
    config_path = get_config_path()
    if config_path.exists():
        config_path.unlink()


# =============================================================================
# Config value operations
# =============================================================================

def get_config_value(key: str) -> Any:
    """Get a single configuration value.

    Args:
        key: Configuration key name.

    Returns:
        The value, or None if key doesn't exist.
    """
    config = load_config()
    return getattr(config, key, None)


def set_config_value(key: str, value: Any) -> bool:
    """Set a single configuration value.

    Args:
        key: Configuration key name.
        value: New value.

    Returns:
        True if successful, False if key is invalid.
    """
    valid_fields = {f.name for f in fields(EvoScientistConfig)}
    if key not in valid_fields:
        return False

    config = load_config()

    # Type coercion based on field type
    field_info = next(f for f in fields(EvoScientistConfig) if f.name == key)
    field_type = field_info.type

    try:
        if field_type == "bool" or field_type is bool:
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes", "on")
            else:
                value = bool(value)
        elif field_type == "int" or field_type is int:
            value = int(value)
        elif field_type == "str" or field_type is str:
            value = str(value)
    except (ValueError, TypeError):
        return False

    setattr(config, key, value)
    save_config(config)
    return True


def list_config() -> dict[str, Any]:
    """List all configuration values.

    Returns:
        Dictionary of all configuration key-value pairs.
    """
    return asdict(load_config())


# =============================================================================
# Effective configuration (merging sources)
# =============================================================================

# Environment variable mappings
_ENV_MAPPINGS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "nvidia_api_key": "NVIDIA_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
    "default_mode": "EVOSCIENTIST_DEFAULT_MODE",
    "default_workdir": "EVOSCIENTIST_WORKSPACE_DIR",
}


def get_effective_config(cli_overrides: dict[str, Any] | None = None) -> EvoScientistConfig:
    """Get effective configuration by merging all sources.

    Priority (highest to lowest):
        1. CLI arguments (cli_overrides)
        2. Environment variables
        3. Config file
        4. Defaults

    Args:
        cli_overrides: Dictionary of CLI argument overrides.

    Returns:
        EvoScientistConfig with merged values.
    """
    # Start with file config (includes defaults for missing values)
    config = load_config()
    data = asdict(config)

    # Apply environment variable overrides
    for config_key, env_key in _ENV_MAPPINGS.items():
        env_value = os.environ.get(env_key)
        if env_value:
            # Type coercion
            field_info = next(f for f in fields(EvoScientistConfig) if f.name == config_key)
            field_type = field_info.type
            if field_type == "bool" or field_type is bool:
                data[config_key] = env_value.lower() in ("true", "1", "yes", "on")
            elif field_type == "int" or field_type is int:
                try:
                    data[config_key] = int(env_value)
                except ValueError:
                    pass
            else:
                data[config_key] = env_value

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and key in data:
                data[key] = value

    return EvoScientistConfig(**data)


def apply_config_to_env(config: EvoScientistConfig) -> None:
    """Apply config API keys to environment variables if not already set.

    This allows the config file to provide API keys that downstream
    libraries (like langchain-anthropic) can pick up.

    Args:
        config: Configuration to apply.
    """
    if config.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key
    if config.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    if config.nvidia_api_key and not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = config.nvidia_api_key
    if config.tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key
