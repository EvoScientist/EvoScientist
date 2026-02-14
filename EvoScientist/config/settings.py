"""Configuration management for EvoScientist.

Handles loading, saving, and merging configuration from multiple sources
with the following priority (highest to lowest):
    CLI arguments > Environment variables > Config file > Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields
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
        nvidia_api_key: NVIDIA API key for NVIDIA models.
        google_api_key: Google API key for Gemini models.
        tavily_api_key: Tavily API key for web search.
        provider: Default LLM provider ('anthropic', 'openai', 'google-genai', or 'nvidia').
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
    google_api_key: str = ""
    siliconflow_api_key: str = ""
    openrouter_api_key: str = ""
    custom_api_key: str = ""
    custom_base_url: str = ""
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

    # Channel Settings
    channel_enabled: str = ""  # "imessage" | "telegram" | "discord" | "" (comma-separated for multiple)
    require_mention: str = "group"  # "always" | "group" | "off"
    text_chunk_limit: int = 0  # 0 = use capability default
    allowed_channels: str = ""  # comma-separated channel IDs, empty = allow all

    # iMessage Settings
    imessage_enabled: bool = False  # legacy compat
    imessage_allowed_senders: str = ""
    imessage_send_thinking: bool = True

    # Telegram Settings
    telegram_bot_token: str = ""
    telegram_allowed_senders: str = ""
    telegram_proxy: str = ""

    # Discord Settings
    discord_bot_token: str = ""
    discord_allowed_senders: str = ""
    discord_allowed_channels: str = ""
    discord_proxy: str = ""

    # Slack Settings
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_allowed_senders: str = ""
    slack_allowed_channels: str = ""
    slack_proxy: str = ""

    # Feishu Settings
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""
    feishu_webhook_port: int = 9000
    feishu_allowed_senders: str = ""
    feishu_domain: str = "https://open.feishu.cn"
    feishu_proxy: str = ""

    # WeChat Settings
    wechat_backend: str = "wecom"
    wechat_webhook_port: int = 9001
    wechat_allowed_senders: str = ""
    wechat_proxy: str = ""
    wechat_wecom_corp_id: str = ""
    wechat_wecom_agent_id: str = ""
    wechat_wecom_secret: str = ""
    wechat_wecom_token: str = ""
    wechat_wecom_encoding_aes_key: str = ""
    wechat_mp_app_id: str = ""
    wechat_mp_app_secret: str = ""
    wechat_mp_token: str = ""
    wechat_mp_encoding_aes_key: str = ""

    # DingTalk Settings
    dingtalk_client_id: str = ""
    dingtalk_client_secret: str = ""
    dingtalk_allowed_senders: str = ""
    dingtalk_proxy: str = ""

    # Email Settings
    email_imap_host: str = ""
    email_imap_port: int = 993
    email_imap_username: str = ""
    email_imap_password: str = ""
    email_imap_mailbox: str = "INBOX"
    email_imap_use_ssl: bool = True
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_username: str = ""
    email_smtp_password: str = ""
    email_smtp_use_tls: bool = True
    email_from_address: str = ""
    email_poll_interval: int = 30
    email_mark_seen: bool = True
    email_max_body_chars: int = 12000
    email_subject_prefix: str = "Re: "
    email_allowed_senders: str = ""

    # QQ Settings
    qq_app_id: str = ""
    qq_app_secret: str = ""
    qq_allowed_senders: str = ""

    # Signal Settings
    signal_phone_number: str = ""
    signal_cli_path: str = "signal-cli"
    signal_config_dir: str = ""
    signal_allowed_senders: str = ""
    signal_rpc_port: int = 7583

    # Shared webhook port (0 = disabled)
    shared_webhook_port: int = 9000

    # DM access control policy
    dm_policy: str = "allowlist"


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

def _coerce_value(value: Any, field_type: Any) -> Any:
    """Coerce a value to the expected field type.

    Args:
        value: The value to coerce.
        field_type: The target type (from dataclass field).

    Returns:
        The coerced value.

    Raises:
        ValueError: If the value cannot be coerced.
        TypeError: If the value cannot be coerced.
    """
    if field_type == "bool" or field_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    if field_type == "int" or field_type is int:
        return int(value)
    return str(value)


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
        value = _coerce_value(value, field_type)
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
    "google_api_key": "GOOGLE_API_KEY",
    "siliconflow_api_key": "SILICONFLOW_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "custom_api_key": "CUSTOM_API_KEY",
    "custom_base_url": "CUSTOM_BASE_URL",
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
            field_info = next(f for f in fields(EvoScientistConfig) if f.name == config_key)
            try:
                data[config_key] = _coerce_value(env_value, field_info.type)
            except (ValueError, TypeError):
                pass

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
    if config.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = config.google_api_key
    if config.siliconflow_api_key and not os.environ.get("SILICONFLOW_API_KEY"):
        os.environ["SILICONFLOW_API_KEY"] = config.siliconflow_api_key
    if config.openrouter_api_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key
    if config.custom_api_key and not os.environ.get("CUSTOM_API_KEY"):
        os.environ["CUSTOM_API_KEY"] = config.custom_api_key
    if config.custom_base_url and not os.environ.get("CUSTOM_BASE_URL"):
        os.environ["CUSTOM_BASE_URL"] = config.custom_base_url
    if config.tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key
