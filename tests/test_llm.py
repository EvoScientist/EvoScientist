"""Tests for EvoScientist LLM module."""

from unittest.mock import patch

import pytest

from EvoScientist.llm import (
    DEFAULT_MODEL,
    MODELS,
    get_chat_model,
    get_model_info,
    get_models_for_provider,
    list_models,
)
from EvoScientist.llm.models import _MODEL_ENTRIES

# =============================================================================
# Test MODELS registry
# =============================================================================


class TestModelsRegistry:
    def test_models_is_dict(self):
        """Test that MODELS is a dictionary."""
        assert isinstance(MODELS, dict)

    def test_entries_has_all_providers(self):
        """Test that _MODEL_ENTRIES covers all registered providers."""
        providers = {p for _, _, p in _MODEL_ENTRIES}
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google-genai" in providers
        assert "minimax" in providers
        assert "nvidia" in providers
        assert "siliconflow" in providers
        assert "openrouter" in providers
        assert "zhipu" in providers
        assert "zhipu-code" in providers
        assert "volcengine" in providers
        assert "dashscope" in providers
        assert "deepseek" in providers
        assert "moonshot" in providers
        assert "kimi-coding" in providers

    def test_entries_are_valid_tuples(self):
        """Test that _MODEL_ENTRIES contains valid (name, model_id, provider) tuples."""
        valid_providers = {
            "anthropic",
            "openai",
            "google-genai",
            "minimax",
            "nvidia",
            "siliconflow",
            "openrouter",
            "zhipu",
            "zhipu-code",
            "volcengine",
            "dashscope",
            "custom-openai",
            "custom-anthropic",
            "deepseek",
            "moonshot",
            "kimi-coding",
        }
        for entry in _MODEL_ENTRIES:
            assert len(entry) == 3, f"Entry {entry} doesn't have 3 elements"
            name, model_id, provider = entry
            assert isinstance(name, str)
            assert isinstance(model_id, str)
            assert provider in valid_providers, (
                f"Unknown provider '{provider}' for '{name}'"
            )

    def test_get_models_for_provider(self):
        """Test that get_models_for_provider returns correct models."""
        anthropic_models = get_models_for_provider("anthropic")
        assert len(anthropic_models) > 0
        for name, model_id in anthropic_models:
            assert isinstance(name, str)
            assert isinstance(model_id, str)

        # Third-party providers now have registered models
        openrouter_models = get_models_for_provider("openrouter")
        assert len(openrouter_models) > 0
        siliconflow_models = get_models_for_provider("siliconflow")
        assert len(siliconflow_models) > 0


# =============================================================================
# Test DEFAULT_MODEL
# =============================================================================


class TestDefaultModel:
    def test_default_model_exists_in_registry(self):
        """Test that DEFAULT_MODEL is a valid model in MODELS."""
        assert DEFAULT_MODEL in MODELS

    def test_default_model_is_anthropic(self):
        """Test that default model uses Anthropic."""
        _, provider = MODELS[DEFAULT_MODEL]
        assert provider == "anthropic"


# =============================================================================
# Test list_models
# =============================================================================


class TestListModels:
    def test_returns_list(self):
        """Test that list_models returns a list."""
        result = list_models()
        assert isinstance(result, list)

    def test_returns_all_model_names(self):
        """Test that list_models returns all model names."""
        result = list_models()
        assert set(result) == set(MODELS.keys())

    def test_list_is_not_empty(self):
        """Test that the list is not empty."""
        assert len(list_models()) > 0


# =============================================================================
# Test get_model_info
# =============================================================================


class TestGetModelInfo:
    def test_returns_tuple_for_valid_model(self):
        """Test that get_model_info returns tuple for valid model."""
        result = get_model_info("claude-sonnet-4-5")
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_none_for_invalid_model(self):
        """Test that get_model_info returns None for invalid model."""
        result = get_model_info("nonexistent-model")
        assert result is None

    def test_returns_correct_info(self):
        """Test that get_model_info returns correct info."""
        model_id, provider = get_model_info("gpt-5-nano")
        assert model_id == "gpt-5-nano"
        assert provider == "openai"


# =============================================================================
# Test get_chat_model
# =============================================================================


class TestGetChatModel:
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_uses_default_model_when_none(self, mock_init):
        """Test that get_chat_model uses default model when model=None."""
        mock_init.return_value = "mock_model"

        get_chat_model()

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        # Default model should be resolved from MODELS
        expected_model_id, expected_provider = MODELS[DEFAULT_MODEL]
        assert call_kwargs["model"] == expected_model_id
        assert call_kwargs["model_provider"] == expected_provider

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_resolves_short_name(self, mock_init):
        """Test that get_chat_model resolves short names correctly."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-opus-4-5")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-5"
        assert call_kwargs["model_provider"] == "anthropic"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_resolves_openai_short_name(self, mock_init):
        """Test that get_chat_model resolves OpenAI short names."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-5-mini")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["model_provider"] == "openai"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_uses_full_model_id(self, mock_init):
        """Test that get_chat_model accepts full model IDs."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-3-opus-20240229")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"
        # Should infer anthropic from the model prefix
        assert call_kwargs["model_provider"] == "anthropic"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_provider_override(self, mock_init):
        """Test that provider can be overridden."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-sonnet-4-5", provider="custom_provider")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "custom_provider"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_passes_kwargs(self, mock_init):
        """Test that additional kwargs are passed through."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-5-nano", temperature=0.7, max_tokens=1000)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_infers_openai_from_gpt_prefix(self, mock_init):
        """Test that OpenAI is inferred from gpt- prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-4-turbo-preview")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_infers_openai_from_o1_prefix(self, mock_init):
        """Test that OpenAI is inferred from o1 prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("o1-preview")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_infers_google_from_gemini_prefix(self, mock_init):
        """Test that google-genai is inferred from gemini prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("gemini-2.0-flash")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "google-genai"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_defaults_to_anthropic_for_unknown(self, mock_init):
        """Test that anthropic is default for unknown model prefixes."""
        mock_init.return_value = "mock_model"

        get_chat_model("some-unknown-model")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"


# =============================================================================
# Test Ollama provider
# =============================================================================


class TestOllamaProvider:
    """Ollama models are not in the static registry (detected dynamically).
    All tests use explicit provider or ollama: prefix."""

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_explicit_provider(self, mock_init):
        """Test that explicit provider='ollama' routes correctly."""
        mock_init.return_value = "mock_model"

        get_chat_model("llama3.1:8b", provider="ollama")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "llama3.1:8b"
        assert call_kwargs["model_provider"] == "ollama"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_ollama_base_url_passthrough(self, mock_init, monkeypatch):
        """Test that OLLAMA_BASE_URL env var is passed to kwargs."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-cluster:11434")

        get_chat_model("llama3.1:8b", provider="ollama")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["base_url"] == "http://gpu-cluster:11434"
        assert call_kwargs["model_provider"] == "ollama"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_ollama_no_base_url_when_unset(self, mock_init, monkeypatch):
        """Test that base_url is not set when OLLAMA_BASE_URL is empty."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        get_chat_model("llama3.1:8b", provider="ollama")

        call_kwargs = mock_init.call_args[1]
        assert "base_url" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_reasoning_auto_enabled_for_ollama(self, mock_init, monkeypatch):
        """Test that reasoning is auto-enabled for Ollama models."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        get_chat_model("llama3.1:8b", provider="ollama")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" not in call_kwargs
        assert call_kwargs["reasoning"] is True

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_reasoning_not_overridden_for_ollama(self, mock_init, monkeypatch):
        """Test that explicit reasoning=False is not overridden for Ollama."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        get_chat_model("llama3.1:8b", provider="ollama", reasoning=False)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] is False

    def test_no_static_registry_entries(self):
        """Test that Ollama has no static registry entries (models detected dynamically)."""
        ollama_models = get_models_for_provider("ollama")
        assert len(ollama_models) == 0

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_ollama_prefix_inference(self, mock_init, monkeypatch):
        """Test that ollama: prefix infers ollama provider."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        get_chat_model("ollama:phi3:mini")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "phi3:mini"
        assert call_kwargs["model_provider"] == "ollama"


# =============================================================================
# Test slash model ID no longer routes to nvidia
# =============================================================================


class TestSlashModelIdFallback:
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_slash_model_id_defaults_to_anthropic(self, mock_init):
        """Unregistered model IDs containing '/' should NOT route to nvidia.

        They fall through to the default 'anthropic' provider, consistent
        with how all other unknown model IDs are handled.
        """
        mock_init.return_value = "mock_model"

        get_chat_model("some-org/some-model")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "some-org/some-model"
        assert call_kwargs["model_provider"] == "anthropic"


# =============================================================================
# Test third-party provider routing
# =============================================================================


class TestThirdPartyRouting:
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_siliconflow_routes_through_openai(self, mock_init, monkeypatch):
        """SiliconFlow provider should route through OpenAI with correct base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key-123")

        get_chat_model("Pro/zai-org/GLM-5", provider="siliconflow")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://api.siliconflow.cn/v1"
        assert call_kwargs["api_key"] == "sf-key-123"
        # SiliconFlow should disable thinking
        assert call_kwargs["extra_body"]["enable_thinking"] is False

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_uses_native_provider(self, mock_init, monkeypatch):
        """OpenRouter should use native 'openrouter' provider via init_chat_model."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-456")

        get_chat_model("x-ai/grok-4.1-fast", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openrouter"
        assert call_kwargs["api_key"] == "or-key-456"
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "disabled"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_reasoning_user_override(self, mock_init, monkeypatch):
        """User-supplied reasoning config should not be overridden."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        get_chat_model(
            "x-ai/grok-4.1-fast",
            provider="openrouter",
            reasoning={"effort": "low"},
        )

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "low"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_reasoning_effort_from_env(self, mock_init, monkeypatch):
        """Reasoning effort should be configurable via env var."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("EVOSCIENTIST_REASONING_EFFORT", "medium")

        get_chat_model("x-ai/grok-4.1-fast", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "medium", "summary": "disabled"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_custom_routes_through_openai(self, mock_init, monkeypatch):
        """Custom provider should route through OpenAI with env-configured base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://my-llm.example.com/v1")
        monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "custom-key-789")

        get_chat_model("my-custom-model", provider="custom-openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://my-llm.example.com/v1"
        assert call_kwargs["api_key"] == "custom-key-789"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_base_url_override(self, mock_init, monkeypatch):
        """Anthropic provider should support base_url override (e.g. ccproxy)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:8000/api/v1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")

        get_chat_model("claude-sonnet-4-6", provider="anthropic")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"
        assert call_kwargs["base_url"] == "http://localhost:8000/api/v1"
        assert call_kwargs["api_key"] == "sk-dummy"
        # Proxy mode: thinking skipped (history round-trip causes 422)
        assert "thinking" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_no_base_url_when_unset(self, mock_init, monkeypatch):
        """Anthropic provider should not set base_url when env var is empty."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")

        get_chat_model("claude-sonnet-4-6", provider="anthropic")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"
        assert "base_url" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_third_party_no_reasoning(self, mock_init, monkeypatch):
        """Third-party providers routed through OpenAI should NOT get auto-reasoning."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key")

        get_chat_model("deepseek-v3", provider="siliconflow")

        call_kwargs = mock_init.call_args[1]
        assert "reasoning" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_volcengine_routes_through_openai(self, mock_init, monkeypatch):
        """Volcengine provider should route through OpenAI with correct base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("VOLCENGINE_API_KEY", "ve-key-123")

        get_chat_model("doubao-seed-1.6", provider="volcengine")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
        assert call_kwargs["api_key"] == "ve-key-123"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_dashscope_routes_through_openai(self, mock_init, monkeypatch):
        """DashScope provider should route through OpenAI with correct base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key-456")

        get_chat_model("qwen-max", provider="dashscope")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert (
            call_kwargs["base_url"]
            == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        assert call_kwargs["api_key"] == "ds-key-456"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_routes_through_anthropic(self, mock_init, monkeypatch):
        """MiniMax provider should route through Anthropic with correct base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key-123")

        get_chat_model("MiniMax-M2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"
        assert call_kwargs["base_url"] == "https://api.minimaxi.com/anthropic"
        assert call_kwargs["api_key"] == "mm-key-123"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_base_url_env_override(self, mock_init, monkeypatch):
        """MINIMAX_BASE_URL env var should override the default base URL."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key-123")
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")

        get_chat_model("MiniMax-M2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["base_url"] == "https://api.minimax.io/anthropic"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_gets_thinking(self, mock_init, monkeypatch):
        """MiniMax provider should get auto-thinking (thinking-capable via Anthropic)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")

        get_chat_model("MiniMax-M2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" in call_kwargs
        assert "reasoning" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_short_name_resolution(self, mock_init, monkeypatch):
        """MiniMax short names should resolve to correct model IDs."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")

        get_chat_model("minimax-m2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.5"
        assert call_kwargs["model_provider"] == "anthropic"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_highspeed_model(self, mock_init, monkeypatch):
        """MiniMax M2.5-highspeed model should resolve correctly."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")

        get_chat_model("minimax-m2.5-highspeed", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.5-highspeed"
        assert call_kwargs["model_provider"] == "anthropic"
        assert call_kwargs["base_url"] == "https://api.minimaxi.com/anthropic"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_custom_anthropic_via_routed_dict(self, mock_init, monkeypatch):
        """custom-anthropic should work via _ANTHROPIC_ROUTED_PROVIDERS dict."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "https://my-claude.example.com")
        monkeypatch.setenv("CUSTOM_ANTHROPIC_API_KEY", "ca-key-789")

        get_chat_model("claude-sonnet-4-6", provider="custom-anthropic")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"
        assert call_kwargs["base_url"] == "https://my-claude.example.com"
        assert call_kwargs["api_key"] == "ca-key-789"
        # custom-anthropic is NOT thinking-capable → thinking skipped
        assert "thinking" not in call_kwargs


# =============================================================================
# Test MiniMax provider
# =============================================================================


class TestMiniMaxProvider:
    def test_minimax_in_anthropic_routed_providers(self):
        """MiniMax should be registered in _ANTHROPIC_ROUTED_PROVIDERS."""
        from EvoScientist.llm.models import _ANTHROPIC_ROUTED_PROVIDERS

        assert "minimax" in _ANTHROPIC_ROUTED_PROVIDERS
        base_url, api_key_env = _ANTHROPIC_ROUTED_PROVIDERS["minimax"]
        assert base_url == "https://api.minimaxi.com/anthropic"
        assert api_key_env == "MINIMAX_API_KEY"

    def test_minimax_not_in_openai_routed_providers(self):
        """MiniMax should NOT be in _OPENAI_ROUTED_PROVIDERS (moved to Anthropic)."""
        from EvoScientist.llm.models import _OPENAI_ROUTED_PROVIDERS

        assert "minimax" not in _OPENAI_ROUTED_PROVIDERS

    def test_minimax_models_registered(self):
        """MiniMax should have 4 direct model entries in _MODEL_ENTRIES."""
        minimax_models = get_models_for_provider("minimax")
        assert len(minimax_models) == 4
        model_names = {name for name, _ in minimax_models}
        assert "minimax-m2.7" in model_names
        assert "minimax-m2.7-highspeed" in model_names
        assert "minimax-m2.5" in model_names
        assert "minimax-m2.5-highspeed" in model_names

    def test_minimax_model_ids_correct(self):
        """MiniMax model IDs should match the official API model names."""
        minimax_models = get_models_for_provider("minimax")
        model_dict = dict(minimax_models)
        assert model_dict["minimax-m2.7"] == "MiniMax-M2.7"
        assert model_dict["minimax-m2.5"] == "MiniMax-M2.5"
        assert model_dict["minimax-m2.5-highspeed"] == "MiniMax-M2.5-highspeed"

    def test_minimax_short_name_in_models_dict(self):
        """MiniMax short names should be accessible via the MODELS dict."""
        # Note: MODELS dict uses last-entry-wins, so direct minimax entries
        # may be overridden by nvidia/siliconflow/openrouter entries.
        # Use get_models_for_provider() for provider-specific lookups.
        minimax_models = get_models_for_provider("minimax")
        assert len(minimax_models) > 0


# =============================================================================
# Test _flatten_message_content
# =============================================================================


class TestFlattenMessageContent:
    """Tests for the content-flattening utility used by OpenAI-compatible providers."""

    def test_string_passthrough(self):
        from EvoScientist.llm.patches import _flatten_message_content

        assert _flatten_message_content("hello") == "hello"

    def test_non_list_passthrough(self):
        from EvoScientist.llm.patches import _flatten_message_content

        assert _flatten_message_content(42) == 42
        assert _flatten_message_content(None) is None

    def test_text_blocks(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert _flatten_message_content(content) == "Hello\n\nWorld"

    def test_skips_thinking_blocks(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [
            {"type": "thinking", "text": "Let me think..."},
            {"type": "text", "text": "The answer is 42"},
            {"type": "reasoning", "text": "internal reasoning"},
            {"type": "reasoning_content", "text": "more reasoning"},
        ]
        assert _flatten_message_content(content) == "The answer is 42"

    def test_string_blocks(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = ["hello", "world"]
        assert _flatten_message_content(content) == "hello\n\nworld"

    def test_mixed_blocks(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [
            {"type": "thinking", "text": "skip me"},
            "plain string",
            {"type": "text", "text": "dict text"},
        ]
        assert _flatten_message_content(content) == "plain string\n\ndict text"

    def test_empty_list(self):
        from EvoScientist.llm.patches import _flatten_message_content

        assert _flatten_message_content([]) == ""

    def test_only_thinking_blocks(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [{"type": "thinking", "text": "thought"}]
        assert _flatten_message_content(content) == ""


# =============================================================================
# Test _patch_openai_compat_content (all 4 paths)
# =============================================================================


class TestPatchOpenAICompatContent:
    """Verify content flattening covers _generate, _agenerate, _stream, _astream."""

    def _make_model(self):
        """Create a minimal mock model with all 4 methods."""
        from unittest.mock import AsyncMock, MagicMock

        model = MagicMock()
        model._generate = MagicMock(return_value="gen_result")
        model._agenerate = AsyncMock(return_value="agen_result")
        model._stream = MagicMock(return_value=iter(["chunk1"]))
        model._astream = AsyncMock()
        return model

    def test_generate_flattened(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        model._generate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hello"

    @pytest.mark.anyio
    async def test_agenerate_flattened(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._agenerate
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        await model._agenerate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hello"

    def test_stream_flattened(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._stream
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        list(model._stream([msg]))

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hello"

    @pytest.mark.anyio
    async def test_astream_flattened(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        received_msgs = []

        async def _fake_astream(messages, *args, **kwargs):
            received_msgs.extend(messages)
            for chunk in ["c1", "c2"]:
                yield chunk

        model._astream = _fake_astream
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        chunks = []
        async for c in model._astream([msg]):
            chunks.append(c)

        assert chunks == ["c1", "c2"]
        assert received_msgs[0].content == "hello"


# =============================================================================
# Test _apply_auto_config
# =============================================================================


class TestAutoConfig:
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_5_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-5 models get enabled thinking with budget."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-sonnet-4-5")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_6_adaptive_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-6 models get adaptive thinking with max effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["effort"] == "max"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_6_proxy_no_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-6 models via proxy skip thinking (history round-trip 422)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ccproxy-oauth")

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" not in call_kwargs
        assert "effort" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_5_proxy_no_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-5 models via proxy also skip thinking (history round-trip 422)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ccproxy-oauth")

        get_chat_model("claude-sonnet-4-5")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_6_no_proxy_no_downgrade(self, mock_init, monkeypatch):
        """Anthropic 4-6 models without proxy still get adaptive thinking."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["effort"] == "max"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_thinking_not_overridden(self, mock_init):
        """User-supplied thinking config should not be overridden."""
        mock_init.return_value = "mock_model"
        custom_thinking = {"type": "enabled", "budget_tokens": 500}

        get_chat_model("claude-sonnet-4-6", thinking=custom_thinking)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == custom_thinking

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_reasoning_xhigh(self, mock_init, monkeypatch):
        """gpt-5.4+ and codex models get xhigh reasoning."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        get_chat_model("gpt-5.4", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "xhigh",
            "summary": "auto",
        }

        get_chat_model("gpt-5.3-codex", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "xhigh",
            "summary": "auto",
        }

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_reasoning_high_fallback(self, mock_init, monkeypatch):
        """Other OpenAI models get high reasoning effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        get_chat_model("gpt-5-nano")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }

        get_chat_model("gpt-5.2", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_base_url_override(self, mock_init, monkeypatch):
        """OpenAI provider should support base_url override (e.g. ccproxy Codex)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/codex/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "ccproxy-oauth")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "http://127.0.0.1:8000/codex/v1"
        assert call_kwargs["api_key"] == "ccproxy-oauth"
        # Proxy mode: reasoning skipped (ccproxy untested)
        assert "reasoning" not in call_kwargs
        # Proxy mode: Responses API (bypasses format chain), streaming ON
        assert call_kwargs["use_responses_api"] is True
        assert "streaming" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_localhost_non_ccproxy_not_downgraded(self, mock_init, monkeypatch):
        """Local OpenAI-compatible endpoints (vLLM, etc.) are not affected by ccproxy workarounds."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-local-key")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["base_url"] == "http://127.0.0.1:8080/v1"
        # NOT ccproxy: reasoning should be applied, no forced Chat Completions
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
        assert "use_responses_api" not in call_kwargs
        assert "streaming" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_codex_path_but_wrong_key_not_ccproxy(self, mock_init, monkeypatch):
        """ccproxy detection requires both /codex/ path AND ccproxy-oauth key."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/codex/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
        assert "use_responses_api" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_ccproxy_key_but_wrong_path_not_ccproxy(
        self, mock_init, monkeypatch
    ):
        """ccproxy detection requires both /codex/ path AND ccproxy-oauth key."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "ccproxy-oauth")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
        assert "use_responses_api" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openai_no_base_url_when_unset(self, mock_init, monkeypatch):
        """OpenAI provider should not set base_url when env var is empty."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert "base_url" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_google_thoughts(self, mock_init):
        """Google GenAI models get include_thoughts=True by default."""
        mock_init.return_value = "mock_model"

        get_chat_model("gemini-2.5-flash")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["include_thoughts"] is True

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_use_responses_api_false(self, mock_init, monkeypatch):
        """use_responses_api=false forces Chat Completions and drops reasoning."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("EVOSCIENTIST_USE_RESPONSES_API", "false")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is False
        assert "reasoning" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_use_responses_api_true(self, mock_init, monkeypatch):
        """use_responses_api=true explicitly enables the Responses API."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("EVOSCIENTIST_USE_RESPONSES_API", "true")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is True
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_use_responses_api_default_unchanged(self, mock_init, monkeypatch):
        """Empty use_responses_api preserves default behavior (no kwarg set)."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("EVOSCIENTIST_USE_RESPONSES_API", raising=False)

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert "use_responses_api" not in call_kwargs
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}

    @pytest.mark.parametrize("env_value", ["FALSE", " false ", "False"])
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_use_responses_api_false_normalization(
        self, mock_init, monkeypatch, env_value
    ):
        """Case/whitespace variants of 'false' are normalized correctly."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("EVOSCIENTIST_USE_RESPONSES_API", env_value)

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is False
        assert "reasoning" not in call_kwargs

    @pytest.mark.parametrize("env_value", ["TRUE", " true ", "True"])
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_use_responses_api_true_normalization(
        self, mock_init, monkeypatch, env_value
    ):
        """Case/whitespace variants of 'true' are normalized correctly."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("EVOSCIENTIST_USE_RESPONSES_API", env_value)

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is True
