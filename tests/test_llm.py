"""Tests for EvoScientist LLM module."""

from unittest.mock import patch

from EvoScientist.llm import (
    MODELS,
    DEFAULT_MODEL,
    get_chat_model,
    get_models_for_provider,
    list_models,
    get_model_info,
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
        assert model_id == "gpt-5-nano-2025-08-07"
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
        assert call_kwargs["model"] == "gpt-5-mini-2025-08-07"
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
    def test_openrouter_routes_through_openai(self, mock_init, monkeypatch):
        """OpenRouter provider should route through OpenAI with correct base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-456")

        get_chat_model("x-ai/grok-4.1-fast", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["api_key"] == "or-key-456"

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
        # Proxy mode: thinking skipped for 4-6 models (ccproxy manages it)
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
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        get_chat_model("x-ai/grok-4.1-fast", provider="openrouter")

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
    def test_minimax_routes_through_openai(self, mock_init, monkeypatch):
        """MiniMax provider should route through OpenAI with correct base_url."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key-123")

        get_chat_model("MiniMax-M2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://api.minimax.io/v1"
        assert call_kwargs["api_key"] == "mm-key-123"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_no_reasoning(self, mock_init, monkeypatch):
        """MiniMax provider should NOT get auto-reasoning (routed via OpenAI)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")

        get_chat_model("MiniMax-M2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert "reasoning" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_short_name_resolution(self, mock_init, monkeypatch):
        """MiniMax short names should resolve to correct model IDs."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")

        get_chat_model("minimax-m2.5", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.5"
        assert call_kwargs["model_provider"] == "openai"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_minimax_highspeed_model(self, mock_init, monkeypatch):
        """MiniMax M2.5-highspeed model should resolve correctly."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")

        get_chat_model("minimax-m2.5-highspeed", provider="minimax")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.5-highspeed"
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://api.minimax.io/v1"


# =============================================================================
# Test MiniMax provider
# =============================================================================


class TestMiniMaxProvider:
    def test_minimax_in_third_party_providers(self):
        """MiniMax should be registered in _THIRD_PARTY_PROVIDERS."""
        from EvoScientist.llm.models import _THIRD_PARTY_PROVIDERS

        assert "minimax" in _THIRD_PARTY_PROVIDERS
        base_url, api_key_env = _THIRD_PARTY_PROVIDERS["minimax"]
        assert base_url == "https://api.minimax.io/v1"
        assert api_key_env == "MINIMAX_API_KEY"

    def test_minimax_models_registered(self):
        """MiniMax should have direct model entries in _MODEL_ENTRIES."""
        minimax_models = get_models_for_provider("minimax")
        assert len(minimax_models) >= 2
        model_names = {name for name, _ in minimax_models}
        assert "minimax-m2.5" in model_names
        assert "minimax-m2.5-highspeed" in model_names

    def test_minimax_model_ids_correct(self):
        """MiniMax model IDs should match the official API model names."""
        minimax_models = get_models_for_provider("minimax")
        model_dict = {name: mid for name, mid in minimax_models}
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
        """Anthropic 4-6 models via proxy skip thinking (ccproxy manages it)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ccproxy-oauth")

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" not in call_kwargs
        assert "effort" not in call_kwargs

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_5_proxy_no_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-5 models via proxy also skip thinking."""
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
    def test_openai_reasoning(self, mock_init, monkeypatch):
        """Native OpenAI models get auto-reasoning."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        get_chat_model("gpt-5-nano")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}

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
        # Proxy mode: reasoning skipped (triggers Responses API → rs_ 404)
        assert "reasoning" not in call_kwargs

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
