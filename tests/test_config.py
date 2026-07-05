"""Tests for EvoScientist configuration management."""

import os
from pathlib import Path

import pytest
import yaml

from EvoScientist.config import (
    EvoScientistConfig,
    MemoryControls,
    MemoryObservationTarget,
    MemoryObservationWriter,
    MemorySkillSynthesisCadence,
    MemorySkillSynthesisMode,
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


@pytest.fixture(autouse=True)
def _restore_dangerous_env():
    """Snapshot/restore EVOSCIENTIST_DANGEROUS_MODE around every test.

    apply_config_to_env writes this var via direct ``os.environ`` assignment, and
    monkeypatch's ``delenv`` of an originally-absent key records no undo — so
    without this, a test that turns dangerous mode on would leak the env var into
    later tests (an order-dependent landmine, e.g. under pytest-randomly).
    """
    _sentinel = object()
    _prev = os.environ.get("EVOSCIENTIST_DANGEROUS_MODE", _sentinel)
    yield
    if _prev is _sentinel:
        os.environ.pop("EVOSCIENTIST_DANGEROUS_MODE", None)
    else:
        os.environ["EVOSCIENTIST_DANGEROUS_MODE"] = _prev


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
        "EVOSCIENTIST_MEMORY_PROFILE_ENABLED",
        "EVOSCIENTIST_MEMORY_OBSERVATIONS_ENABLED",
        "EVOSCIENTIST_MEMORY_OBSERVATION_WRITER",
        "EVOSCIENTIST_MEMORY_WORKERS_ENABLED",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_ENABLED",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_MODE",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_CADENCE",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_TIME",
        "EVOSCIENTIST_AUXILIARY_MODEL",
        "EVOSCIENTIST_AUXILIARY_PROVIDER",
        "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE",
        "EVOSCIENTIST_DANGEROUS_MODE",
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
        "EVOSCIENTIST_MEMORY_PROFILE_ENABLED",
        "EVOSCIENTIST_MEMORY_OBSERVATIONS_ENABLED",
        "EVOSCIENTIST_MEMORY_OBSERVATION_WRITER",
        "EVOSCIENTIST_MEMORY_WORKERS_ENABLED",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_ENABLED",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_MODE",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_CADENCE",
        "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_TIME",
        "EVOSCIENTIST_AUXILIARY_MODEL",
        "EVOSCIENTIST_AUXILIARY_PROVIDER",
        "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE",
        "EVOSCIENTIST_DANGEROUS_MODE",
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
        assert config.model == "claude-sonnet-4-6"
        assert config.default_mode == "daemon"
        assert config.default_workdir == ""
        assert config.show_thinking is True
        assert config.ui_backend == "tui"
        assert config.log_level == "warning"
        assert config.reasoning_effort == "high"
        assert config.openrouter_anthropic_prompt_cache is True
        assert config.memory_profile_enabled is True
        assert config.memory_observations_enabled is True
        assert config.memory_observation_writer == MemoryObservationWriter.ALL
        assert config.memory_workers_enabled is True
        assert config.memory_skill_synthesis_enabled is True
        assert config.memory_skill_synthesis_mode == MemorySkillSynthesisMode.REVIEW
        assert (
            config.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.WEEKLY
        )
        assert config.memory_skill_synthesis_time == "03:00"
        assert config.ollama_base_url == ""
        assert config.channel_debug_tracing is False
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

    def test_dangerous_mode_default(self):
        """dangerous_mode defaults off and does not force auto_approve."""
        config = EvoScientistConfig()
        assert config.dangerous_mode is False
        assert config.auto_approve is False

    def test_dangerous_mode_implies_auto_approve(self):
        """Enabling dangerous_mode forces auto_approve via __post_init__."""
        config = EvoScientistConfig(dangerous_mode=True)
        assert config.dangerous_mode is True
        assert config.auto_approve is True


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
        assert config.model == "claude-sonnet-4-6"

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

    def test_set_channel_debug_tracing_coercion(self, temp_config_dir, clean_env):
        """Test that channel_debug_tracing is coerced from string to bool."""
        save_config(EvoScientistConfig())

        set_config_value("channel_debug_tracing", "true")
        assert get_config_value("channel_debug_tracing") is True

        set_config_value("channel_debug_tracing", "false")
        assert get_config_value("channel_debug_tracing") is False

    def test_set_memory_observation_writer_validates_value(
        self, temp_config_dir, clean_env
    ):
        """Observation writer mode accepts only the supported product controls."""
        save_config(
            EvoScientistConfig(memory_observation_writer=MemoryObservationWriter.ALL)
        )

        assert set_config_value("memory_observation_writer", "worker") is True
        assert get_config_value("memory_observation_writer") == "worker"
        assert set_config_value("memory_observation_writer", "AGENT") is True
        assert get_config_value("memory_observation_writer") == "agent"
        assert set_config_value("memory_observation_writer", "subagent") is False
        assert get_config_value("memory_observation_writer") == "agent"

    def test_memory_controls_observation_writer_targets(self):
        """MemoryControls centralizes observation writer target semantics."""
        worker_controls = MemoryControls.from_config(
            EvoScientistConfig(
                memory_profile_enabled=False,
                memory_observations_enabled=True,
                memory_observation_writer=MemoryObservationWriter.WORKER,
                memory_workers_enabled=True,
            )
        )
        all_controls = MemoryControls.from_config(
            EvoScientistConfig(
                memory_profile_enabled=False,
                memory_observations_enabled=True,
                memory_observation_writer=MemoryObservationWriter.ALL,
                memory_workers_enabled=True,
            )
        )

        assert worker_controls.observation_tool_enabled(
            MemoryObservationTarget.TURN_WORKER
        )
        assert worker_controls.worker_needed(MemoryObservationTarget.TURN_WORKER)
        assert worker_controls.observation_tool_enabled(
            MemoryObservationTarget.SUBAGENT_WORKER
        )
        assert not worker_controls.observation_tool_enabled(
            MemoryObservationTarget.AGENT
        )
        assert all_controls.worker_needed(MemoryObservationTarget.TURN_WORKER)
        assert all_controls.observation_tool_enabled(MemoryObservationTarget.AGENT)
        assert all_controls.observation_tool_enabled(
            MemoryObservationTarget.TURN_WORKER
        )
        assert all_controls.observation_tool_enabled(
            MemoryObservationTarget.SUBAGENT_WORKER
        )

    def test_set_memory_skill_synthesis_values_validate(
        self, temp_config_dir, clean_env
    ):
        save_config(EvoScientistConfig())

        assert set_config_value("memory_skill_synthesis_mode", "auto") is True
        assert get_config_value("memory_skill_synthesis_mode") == "auto"
        assert set_config_value("memory_skill_synthesis_mode", "always") is False
        assert get_config_value("memory_skill_synthesis_mode") == "auto"

        assert set_config_value("memory_skill_synthesis_cadence", "nightly") is True
        assert get_config_value("memory_skill_synthesis_cadence") == "nightly"
        assert set_config_value("memory_skill_synthesis_cadence", "hourly") is False
        assert get_config_value("memory_skill_synthesis_cadence") == "nightly"

        assert set_config_value("memory_skill_synthesis_time", "3:05") is True
        assert get_config_value("memory_skill_synthesis_time") == "03:05"
        assert set_config_value("memory_skill_synthesis_time", "24:00") is False
        assert get_config_value("memory_skill_synthesis_time") == "03:05"

        loaded = load_config()
        assert loaded.memory_skill_synthesis_mode == MemorySkillSynthesisMode.AUTO
        assert (
            loaded.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.NIGHTLY
        )

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

        config = get_effective_config(cli_overrides={"model": "claude-opus-4-8"})
        assert config.model == "claude-opus-4-8"

    def test_env_ui_backend_override(self, temp_config_dir, monkeypatch):
        """UI backend can be selected via environment variable."""
        save_config(EvoScientistConfig(ui_backend="cli"))
        monkeypatch.setenv("EVOSCIENTIST_UI_BACKEND", "tui")
        config = get_effective_config()
        assert config.ui_backend == "tui"

    def test_env_log_level_override(self, temp_config_dir, monkeypatch):
        """Log level can be selected via environment variable."""
        save_config(EvoScientistConfig(log_level="warning"))
        monkeypatch.setenv("EVOSCIENTIST_LOG_LEVEL", "DEBUG")
        config = get_effective_config()
        assert config.log_level == "DEBUG"

    def test_env_reasoning_effort_override(self, temp_config_dir, monkeypatch):
        """Reasoning effort can be selected via environment variable."""
        save_config(EvoScientistConfig(reasoning_effort="medium"))
        monkeypatch.setenv("EVOSCIENTIST_REASONING_EFFORT", "high")
        config = get_effective_config()
        assert config.reasoning_effort == "high"

    def test_env_channel_debug_tracing_override(self, temp_config_dir, monkeypatch):
        """Channel tracing can be enabled via environment variable."""
        save_config(EvoScientistConfig(channel_debug_tracing=False))
        monkeypatch.setenv("EVOSCIENTIST_CHANNEL_DEBUG_TRACING", "true")
        config = get_effective_config()
        assert config.channel_debug_tracing is True

    def test_env_memory_controls_override(self, temp_config_dir, monkeypatch):
        """Memory controls can be selected via environment variables."""
        save_config(
            EvoScientistConfig(
                memory_profile_enabled=True,
                memory_observations_enabled=True,
                memory_observation_writer=MemoryObservationWriter.ALL,
                memory_workers_enabled=True,
            )
        )
        monkeypatch.setenv("EVOSCIENTIST_MEMORY_PROFILE_ENABLED", "false")
        monkeypatch.setenv("EVOSCIENTIST_MEMORY_OBSERVATIONS_ENABLED", "false")
        monkeypatch.setenv("EVOSCIENTIST_MEMORY_OBSERVATION_WRITER", "worker")
        monkeypatch.setenv("EVOSCIENTIST_MEMORY_WORKERS_ENABLED", "false")

        config = get_effective_config()
        assert config.memory_profile_enabled is False
        assert config.memory_observations_enabled is False
        assert config.memory_observation_writer == MemoryObservationWriter.WORKER
        assert config.memory_workers_enabled is False

    def test_sandbox_execute_timeout_default(self, temp_config_dir, clean_env):
        """Sandbox execute timeout defaults to 300 seconds."""
        assert EvoScientistConfig().sandbox_execute_timeout == 300
        assert get_effective_config().sandbox_execute_timeout == 300

    def test_env_sandbox_execute_timeout_override(self, temp_config_dir, monkeypatch):
        """Sandbox execute timeout can be set via env var and coerces to int."""
        monkeypatch.setenv("EVOSCIENTIST_SANDBOX_EXECUTE_TIMEOUT", "600")
        config = get_effective_config()
        assert config.sandbox_execute_timeout == 600
        assert isinstance(config.sandbox_execute_timeout, int)

    def test_sandbox_execute_timeout_invalid_falls_back(self):
        """Non-positive / non-int values fall back to the default (would
        otherwise crash CustomSandboxBackend construction at startup)."""
        assert (
            EvoScientistConfig(sandbox_execute_timeout=0).sandbox_execute_timeout == 300
        )
        assert (
            EvoScientistConfig(sandbox_execute_timeout=-5).sandbox_execute_timeout
            == 300
        )
        assert (
            EvoScientistConfig(sandbox_execute_timeout="abc").sandbox_execute_timeout
            == 300
        )
        assert (
            EvoScientistConfig(sandbox_execute_timeout=True).sandbox_execute_timeout
            == 300
        )

    def test_set_sandbox_execute_timeout_rejects_invalid(
        self, temp_config_dir, clean_env
    ):
        """set_config_value must reject (not silently persist) a non-positive timeout."""
        save_config(EvoScientistConfig(sandbox_execute_timeout=120))
        assert set_config_value("sandbox_execute_timeout", 0) is False
        assert set_config_value("sandbox_execute_timeout", -5) is False
        # bool is an int subclass; reject it before coercion turns True into 1.
        assert set_config_value("sandbox_execute_timeout", True) is False
        # The earlier valid value is untouched on disk.
        assert get_config_value("sandbox_execute_timeout") == 120
        # A valid value still goes through.
        assert set_config_value("sandbox_execute_timeout", 600) is True
        assert get_config_value("sandbox_execute_timeout") == 600

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

    def test_env_openrouter_anthropic_prompt_cache_opt_out(
        self, temp_config_dir, monkeypatch
    ):
        """Test OpenRouter Anthropic prompt cache opt-out env overrides file."""
        save_config(EvoScientistConfig(openrouter_anthropic_prompt_cache=True))
        monkeypatch.setenv("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", "false")

        config = get_effective_config()
        assert config.openrouter_anthropic_prompt_cache is False

    def test_set_openrouter_anthropic_prompt_cache(self, temp_config_dir, clean_env):
        """Test OpenRouter Anthropic prompt cache can be set through config."""
        save_config(EvoScientistConfig())

        assert set_config_value("openrouter_anthropic_prompt_cache", "false") is True
        assert get_config_value("openrouter_anthropic_prompt_cache") is False


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
        assert os.environ.get("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE") is None

    def test_openrouter_anthropic_prompt_cache_opt_out_applied(
        self, clean_env, monkeypatch
    ):
        """Test OpenRouter Anthropic prompt cache opt-out config is applied to env."""
        monkeypatch.delenv(
            "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", raising=False
        )
        config = EvoScientistConfig(openrouter_anthropic_prompt_cache=False)
        apply_config_to_env(config)

        assert os.environ.get("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE") == (
            "false"
        )

    def test_dangerous_mode_round_trips_to_env(self, clean_env, monkeypatch):
        """dangerous_mode set via CLI override must survive a fresh re-read.

        Regression: a --dangerous CLI flag is never persisted to file/env, so a
        fresh get_effective_config() (warning banner, run_in_background, the
        langgraph dev subprocess) would read it back as False while the backend
        is already unconfined. apply_config_to_env round-trips it to env.
        """
        from EvoScientist.config import get_effective_config

        monkeypatch.delenv("EVOSCIENTIST_DANGEROUS_MODE", raising=False)
        cfg = get_effective_config({"dangerous_mode": True})
        assert cfg.dangerous_mode is True

        apply_config_to_env(cfg)
        assert os.environ.get("EVOSCIENTIST_DANGEROUS_MODE") == "true"
        # The fresh, no-override read now agrees with the backend.
        assert get_effective_config().dangerous_mode is True

    def test_dangerous_mode_not_applied_when_off(self, clean_env, monkeypatch):
        """dangerous_mode=False must not write the env var."""
        monkeypatch.delenv("EVOSCIENTIST_DANGEROUS_MODE", raising=False)
        apply_config_to_env(EvoScientistConfig())
        assert os.environ.get("EVOSCIENTIST_DANGEROUS_MODE") is None

    def test_dangerous_mode_off_clears_stale_env(self, clean_env, monkeypatch):
        """Re-applying a non-dangerous config clears a previously-set env var.

        Regression: the round-trip used to be set-only, so once dangerous mode
        was applied in a process it could never be lowered — leaking unconfined
        access into later non-dangerous reads.
        """
        monkeypatch.delenv("EVOSCIENTIST_DANGEROUS_MODE", raising=False)
        apply_config_to_env(EvoScientistConfig(dangerous_mode=True))
        assert os.environ.get("EVOSCIENTIST_DANGEROUS_MODE") == "true"
        apply_config_to_env(EvoScientistConfig())  # dangerous off
        assert os.environ.get("EVOSCIENTIST_DANGEROUS_MODE") is None

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


class TestAuxiliaryModelConfig:
    """auxiliary_model / auxiliary_provider config fields (plain str, optional)."""

    def test_defaults_empty(self):
        cfg = EvoScientistConfig()
        assert cfg.auxiliary_model == ""
        assert cfg.auxiliary_provider == ""

    def test_save_and_load_round_trip(self, temp_config_dir, clean_env):
        save_config(
            EvoScientistConfig(
                auxiliary_model="claude-haiku-4-5",
                auxiliary_provider="anthropic",
            )
        )
        loaded = load_config()
        assert loaded.auxiliary_model == "claude-haiku-4-5"
        assert loaded.auxiliary_provider == "anthropic"

    def test_get_set_value(self, temp_config_dir, clean_env):
        save_config(EvoScientistConfig())
        assert set_config_value("auxiliary_model", "qwen3.6-flash") is True
        assert set_config_value("auxiliary_provider", "dashscope") is True
        assert get_config_value("auxiliary_model") == "qwen3.6-flash"
        assert get_config_value("auxiliary_provider") == "dashscope"

    def test_env_overrides_file(self, temp_config_dir, monkeypatch):
        save_config(EvoScientistConfig(auxiliary_model="claude-haiku-4-5"))
        monkeypatch.setenv("EVOSCIENTIST_AUXILIARY_MODEL", "gpt-5.5")
        monkeypatch.setenv("EVOSCIENTIST_AUXILIARY_PROVIDER", "openai")

        config = get_effective_config()
        assert config.auxiliary_model == "gpt-5.5"
        assert config.auxiliary_provider == "openai"


def test_scheduler_config_defaults_and_env(monkeypatch):
    from EvoScientist.config.settings import EvoScientistConfig, get_effective_config

    c = EvoScientistConfig()
    assert c.enable_scheduler is True
    assert c.scheduler_default_timezone == ""
    assert c.memory_skill_synthesis_enabled is True
    assert c.memory_skill_synthesis_mode == MemorySkillSynthesisMode.REVIEW
    assert c.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.WEEKLY
    assert c.memory_skill_synthesis_time == "03:00"

    monkeypatch.setenv("EVOSCIENTIST_ENABLE_SCHEDULER", "false")
    eff = get_effective_config({})
    assert eff.enable_scheduler is False

    monkeypatch.setenv("EVOSCIENTIST_SCHEDULER_DEFAULT_TIMEZONE", "America/New_York")
    monkeypatch.setenv("EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_ENABLED", "false")
    monkeypatch.setenv("EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_MODE", "auto")
    monkeypatch.setenv("EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_CADENCE", "monthly")
    monkeypatch.setenv("EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_TIME", "4:30")
    eff2 = get_effective_config({})
    assert eff2.scheduler_default_timezone == "America/New_York"
    assert eff2.memory_skill_synthesis_enabled is False
    assert eff2.memory_skill_synthesis_mode == MemorySkillSynthesisMode.AUTO
    assert eff2.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.MONTHLY
    assert eff2.memory_skill_synthesis_time == "04:30"
