"""Tests for EvoScientist configuration management."""

import os
from pathlib import Path

import pytest
import yaml

from EvoScientist.config import (
    EvoScientistConfig,
    apply_config_to_env,
    get_config_dir,
    get_config_path,
    get_config_value,
    get_effective_config,
    list_config,
    load_config,
    reset_config,
    save_config,
    set_config_value,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Use a temporary directory for config during tests."""
    config_dir = tmp_path / "evoscientist"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Prevent load_dotenv from loading the project's real .env file
    monkeypatch.setattr(
        "EvoScientist.config.settings.find_dotenv",
        lambda *a, **k: str(tmp_path / ".env"),
    )
    # Also clear any API keys from environment
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "EVOSCIENTIST_DEFAULT_MODE",
        "EVOSCIENTIST_WORKSPACE_DIR",
        "EVOSCIENTIST_UI_BACKEND",
    ]:
        monkeypatch.delenv(key, raising=False)
    return config_dir


@pytest.fixture
def clean_env(monkeypatch):
    """Remove environment variables that affect config (but keep temp config dir)."""
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "EVOSCIENTIST_DEFAULT_MODE",
        "EVOSCIENTIST_WORKSPACE_DIR",
        "EVOSCIENTIST_UI_BACKEND",
    ]:
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# Test EvoScientistConfig dataclass
# =============================================================================


class TestEvoScientistConfig:
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = EvoScientistConfig()

        assert config.anthropic_api_key == ""
        assert config.openai_api_key == ""
        assert config.tavily_api_key == ""
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5"
        assert config.default_mode == "daemon"
        assert config.default_workdir == ""
        assert config.show_thinking is True
        assert config.ui_backend == "tui"
        assert config.ollama_base_url == ""
        assert config.imessage_enabled is False
        assert config.imessage_allowed_senders == ""

    def test_auth_mode_default(self):
        """Test that anthropic_auth_mode defaults to api_key."""
        config = EvoScientistConfig()
        assert config.anthropic_auth_mode == "api_key"

    def test_auth_mode_set(self):
        """Test that anthropic_auth_mode can be set."""
        config = EvoScientistConfig(anthropic_auth_mode="oauth")
        assert config.anthropic_auth_mode == "oauth"

    def test_openai_auth_mode_default(self):
        """Test that openai_auth_mode defaults to api_key."""
        config = EvoScientistConfig()
        assert config.openai_auth_mode == "api_key"

    def test_openai_auth_mode_set(self):
        """Test that openai_auth_mode can be set."""
        config = EvoScientistConfig(openai_auth_mode="oauth")
        assert config.openai_auth_mode == "oauth"

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = EvoScientistConfig(
            anthropic_api_key="sk-ant-test",
            provider="openai",
            model="gpt-4o",
            default_mode="run",
        )

        assert config.anthropic_api_key == "sk-ant-test"
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.default_mode == "run"


# =============================================================================
# Test config path functions
# =============================================================================


class TestConfigPaths:
    def test_get_config_dir_with_xdg(self, monkeypatch, tmp_path):
        """Test config dir uses XDG_CONFIG_HOME when set."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_dir = get_config_dir()
        assert config_dir == tmp_path / "evoscientist"

    def test_get_config_dir_default(self, monkeypatch):
        """Test config dir defaults to ~/.config/evoscientist."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        config_dir = get_config_dir()
        assert config_dir == Path.home() / ".config" / "evoscientist"

    def test_get_config_path(self, temp_config_dir):
        """Test config file path."""
        config_path = get_config_path()
        assert config_path == temp_config_dir / "config.yaml"


# =============================================================================
# Test load/save/reset
# =============================================================================


class TestLoadSaveReset:
    def test_load_returns_defaults_when_no_file(self, temp_config_dir, clean_env):
        """Test that load returns defaults when config file doesn't exist."""
        config = load_config()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5"

    def test_save_creates_file(self, temp_config_dir, clean_env):
        """Test that save creates the config file."""
        config = EvoScientistConfig(provider="openai", model="gpt-4o")
        save_config(config)

        config_path = get_config_path()
        assert config_path.exists()

        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"

    def test_load_reads_saved_config(self, temp_config_dir, clean_env):
        """Test that load reads previously saved config."""
        original = EvoScientistConfig(
            anthropic_api_key="test-key",
            provider="openai",
        )
        save_config(original)

        loaded = load_config()
        assert loaded.anthropic_api_key == "test-key"
        assert loaded.provider == "openai"

    def test_reset_deletes_config_file(self, temp_config_dir, clean_env):
        """Test that reset deletes the config file."""
        config = EvoScientistConfig(provider="openai")
        save_config(config)

        config_path = get_config_path()
        assert config_path.exists()

        reset_config()
        assert not config_path.exists()

    def test_reset_no_file_is_safe(self, temp_config_dir, clean_env):
        """Test that reset is safe when no config file exists."""
        reset_config()  # Should not raise

    def test_load_ignores_invalid_fields(self, temp_config_dir, clean_env):
        """Test that load ignores unknown fields in config file."""
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.safe_dump(
                {
                    "provider": "openai",
                    "unknown_field": "should_be_ignored",
                    "another_bad": 123,
                },
                f,
            )

        config = load_config()
        assert config.provider == "openai"
        assert not hasattr(config, "unknown_field")


# =============================================================================
# Test get/set single values
# =============================================================================


class TestGetSetValues:
    def test_get_config_value(self, temp_config_dir, clean_env):
        """Test getting a single config value."""
        config = EvoScientistConfig(model="gpt-4o-mini")
        save_config(config)

        assert get_config_value("model") == "gpt-4o-mini"

    def test_get_config_value_invalid_key(self, temp_config_dir, clean_env):
        """Test getting an invalid key returns None."""
        assert get_config_value("nonexistent_key") is None

    def test_set_config_value(self, temp_config_dir, clean_env):
        """Test setting a single config value."""
        save_config(EvoScientistConfig())

        result = set_config_value("model", "gpt-4o")
        assert result is True

        config = load_config()
        assert config.model == "gpt-4o"

    def test_set_config_value_invalid_key(self, temp_config_dir, clean_env):
        """Test setting an invalid key returns False."""
        result = set_config_value("nonexistent_key", "value")
        assert result is False

    def test_set_config_value_type_coercion_bool(self, temp_config_dir, clean_env):
        """Test that boolean values are coerced correctly."""
        save_config(EvoScientistConfig())

        set_config_value("show_thinking", "false")
        assert get_config_value("show_thinking") is False

        set_config_value("show_thinking", "1")
        assert get_config_value("show_thinking") is True

        set_config_value("show_thinking", "yes")
        assert get_config_value("show_thinking") is True

    def test_set_imessage_enabled_coercion(self, temp_config_dir, clean_env):
        """Test that imessage_enabled is coerced from string to bool."""
        save_config(EvoScientistConfig())

        set_config_value("imessage_enabled", "true")
        assert get_config_value("imessage_enabled") is True

        set_config_value("imessage_enabled", "false")
        assert get_config_value("imessage_enabled") is False

    def test_set_imessage_allowed_senders(self, temp_config_dir, clean_env):
        """Test that imessage_allowed_senders stores comma-separated string."""
        save_config(EvoScientistConfig())

        set_config_value("imessage_allowed_senders", "+1234567890,+0987654321")
        assert get_config_value("imessage_allowed_senders") == "+1234567890,+0987654321"

    def test_list_config(self, temp_config_dir, clean_env):
        """Test listing all config values."""
        config = EvoScientistConfig(provider="openai", model="gpt-4o")
        save_config(config)

        all_config = list_config()
        assert isinstance(all_config, dict)
        assert all_config["provider"] == "openai"
        assert all_config["model"] == "gpt-4o"


# =============================================================================
# Test priority chain (CLI > env > file > default)
# =============================================================================


class TestPriorityChain:
    def test_defaults_when_nothing_set(self, temp_config_dir, clean_env, monkeypatch):
        """Test defaults are used when nothing is configured."""
        # Ensure no env vars affect the test
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "TAVILY_API_KEY",
            "EVOSCIENTIST_DEFAULT_MODE",
            "EVOSCIENTIST_WORKSPACE_DIR",
            "EVOSCIENTIST_UI_BACKEND",
        ]:
            monkeypatch.delenv(key, raising=False)
        config = get_effective_config()
        assert config.provider == "anthropic"
        assert config.default_mode == "daemon"

    def test_file_overrides_defaults(self, temp_config_dir, clean_env):
        """Test file config overrides defaults."""
        save_config(EvoScientistConfig(provider="openai"))

        config = get_effective_config()
        assert config.provider == "openai"

    def test_env_overrides_file(self, temp_config_dir, monkeypatch):
        """Test environment variables override file config."""
        save_config(EvoScientistConfig(default_mode="run"))
        monkeypatch.setenv("EVOSCIENTIST_DEFAULT_MODE", "daemon")

        config = get_effective_config()
        assert config.default_mode == "daemon"

    def test_cli_overrides_env(self, temp_config_dir, monkeypatch):
        """Test CLI arguments override environment variables."""
        monkeypatch.setenv("EVOSCIENTIST_DEFAULT_MODE", "daemon")

        config = get_effective_config(cli_overrides={"default_mode": "run"})
        assert config.default_mode == "run"

    def test_cli_overrides_file(self, temp_config_dir, clean_env):
        """Test CLI arguments override file config."""
        save_config(EvoScientistConfig(model="gpt-4o"))

        config = get_effective_config(cli_overrides={"model": "claude-opus-4-5"})
        assert config.model == "claude-opus-4-5"

    def test_env_ui_backend_override(self, temp_config_dir, monkeypatch):
        """UI backend can be selected via environment variable."""
        save_config(EvoScientistConfig(ui_backend="cli"))
        monkeypatch.setenv("EVOSCIENTIST_UI_BACKEND", "tui")
        config = get_effective_config()
        assert config.ui_backend == "tui"

    def test_env_api_key_override(self, temp_config_dir, monkeypatch):
        """Test API keys from env override file."""
        save_config(EvoScientistConfig(anthropic_api_key="file-key"))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

        config = get_effective_config()
        assert config.anthropic_api_key == "env-key"

    def test_env_auth_mode_override(self, temp_config_dir, monkeypatch):
        """Test auth mode from env overrides file."""
        save_config(EvoScientistConfig(anthropic_auth_mode="api_key"))
        monkeypatch.setenv("EVOSCIENTIST_ANTHROPIC_AUTH_MODE", "oauth")

        config = get_effective_config()
        assert config.anthropic_auth_mode == "oauth"

    def test_env_openai_auth_mode_override(self, temp_config_dir, monkeypatch):
        """Test openai_auth_mode from env overrides file."""
        save_config(EvoScientistConfig(openai_auth_mode="api_key"))
        monkeypatch.setenv("EVOSCIENTIST_OPENAI_AUTH_MODE", "oauth")

        config = get_effective_config()
        assert config.openai_auth_mode == "oauth"


# =============================================================================
# Test apply_config_to_env
# =============================================================================


class TestApplyConfigToEnv:
    def test_applies_api_keys_when_not_set(self, clean_env):
        """Test that API keys are applied to env when not already set."""
        config = EvoScientistConfig(
            anthropic_api_key="config-ant-key",
            openai_api_key="config-oai-key",
            tavily_api_key="config-tav-key",
        )

        apply_config_to_env(config)

        assert os.environ.get("ANTHROPIC_API_KEY") == "config-ant-key"
        assert os.environ.get("OPENAI_API_KEY") == "config-oai-key"
        assert os.environ.get("TAVILY_API_KEY") == "config-tav-key"

    def test_does_not_override_existing_env(self, monkeypatch):
        """Test that existing env vars are not overridden."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "existing-key")

        config = EvoScientistConfig(anthropic_api_key="config-key")
        apply_config_to_env(config)

        assert os.environ.get("ANTHROPIC_API_KEY") == "existing-key"

    def test_empty_config_keys_not_applied(self, clean_env):
        """Test that empty config keys don't create env vars."""
        config = EvoScientistConfig()  # All empty
        apply_config_to_env(config)

        assert os.environ.get("ANTHROPIC_API_KEY") is None
        assert os.environ.get("OPENAI_API_KEY") is None

    def test_ollama_base_url_applied(self, clean_env, monkeypatch):
        """Test that ollama_base_url is applied to OLLAMA_BASE_URL env var."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        config = EvoScientistConfig(ollama_base_url="http://localhost:11434")
        apply_config_to_env(config)

        assert os.environ.get("OLLAMA_BASE_URL") == "http://localhost:11434"

    def test_ollama_base_url_not_overridden(self, monkeypatch):
        """Test that existing OLLAMA_BASE_URL env var is not overridden."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://existing:11434")
        config = EvoScientistConfig(ollama_base_url="http://new:11434")
        apply_config_to_env(config)

        assert os.environ.get("OLLAMA_BASE_URL") == "http://existing:11434"
