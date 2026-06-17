"""Tests for EvoScientist LLM module."""

from unittest.mock import patch

import pytest

# Side-effect import: applies module-level monkey-patches (e.g.,
# _patch_openai_capture_reasoning_content) before tests reference patched
# functions from langchain_openai.
import EvoScientist.llm.patches  # noqa: F401
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
        assert "dashscope-code" in providers
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
            "dashscope-code",
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
        result = get_model_info("claude-sonnet-4-6")
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

        get_chat_model("claude-opus-4-8")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-8"
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

        get_chat_model("claude-sonnet-4-6", provider="custom_provider")

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
        # Assert the DEFAULT effort, so isolate from any leaked env override.
        monkeypatch.delenv("EVOSCIENTIST_REASONING_EFFORT", raising=False)

        get_chat_model("x-ai/grok-4.3", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openrouter"
        assert call_kwargs["api_key"] == "or-key-456"
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_reasoning_user_override(self, mock_init, monkeypatch):
        """User-supplied reasoning config should not be overridden."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        get_chat_model(
            "x-ai/grok-4.3",
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

        get_chat_model("x-ai/grok-4.3", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_anthropic_prompt_cache_disabled_by_default(
        self, mock_init, monkeypatch
    ):
        """OpenRouter Anthropic prompt caching should be opt-in."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.delenv(
            "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", raising=False
        )

        get_chat_model("claude-sonnet-4.6", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert "cache_control" not in call_kwargs
        assert "cache_control" not in call_kwargs.get("model_kwargs", {})

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_anthropic_prompt_cache_opt_in(self, mock_init, monkeypatch):
        """The opt-in flag should declare caching for OpenRouter Claude models."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", "true")

        get_chat_model("claude-sonnet-4.6", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openrouter"
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4.6"
        assert call_kwargs["model_kwargs"]["cache_control"] == {"type": "ephemeral"}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_prompt_cache_opt_in_skips_non_anthropic_openrouter(
        self, mock_init, monkeypatch
    ):
        """OpenRouter models with implicit caching should be left alone."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", "true")

        get_chat_model("x-ai/grok-4.3", provider="openrouter")

        call_kwargs = mock_init.call_args[1]
        assert "cache_control" not in call_kwargs
        assert "cache_control" not in call_kwargs.get("model_kwargs", {})

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_anthropic_prompt_cache_preserves_top_level_override(
        self, mock_init, monkeypatch
    ):
        """The default should not duplicate a caller's cache_control kwarg."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", "true")
        override = {"type": "ephemeral", "ttl": "1h"}

        get_chat_model(
            "claude-sonnet-4.6",
            provider="openrouter",
            cache_control=override,
        )

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["cache_control"] == override
        assert "cache_control" not in call_kwargs.get("model_kwargs", {})

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_anthropic_prompt_cache_preserves_model_kwargs_override(
        self, mock_init, monkeypatch
    ):
        """The default should not duplicate model_kwargs cache_control."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", "true")
        override = {"type": "ephemeral", "ttl": "1h"}

        get_chat_model(
            "claude-sonnet-4.6",
            provider="openrouter",
            model_kwargs={"cache_control": override},
        )

        call_kwargs = mock_init.call_args[1]
        assert "cache_control" not in call_kwargs
        assert call_kwargs["model_kwargs"]["cache_control"] == override

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_openrouter_anthropic_prompt_cache_warns_on_invalid_model_kwargs(
        self, mock_init, monkeypatch
    ):
        """Invalid model_kwargs shape should warn and skip cache injection."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE", "true")

        with pytest.warns(UserWarning, match="model_kwargs` is not a dict"):
            get_chat_model(
                "claude-sonnet-4.6",
                provider="openrouter",
                model_kwargs="bad",
            )

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_kwargs"] == "bad"

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
    def test_dashscope_code_routes_through_openai(self, mock_init, monkeypatch):
        """DashScope-Code (sk-sp-* subscription keys) routes through OpenAI
        with the coding.dashscope.aliyuncs.com base URL, reusing DASHSCOPE_API_KEY.
        """
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-sp-key-789")

        get_chat_model("qwen3-coder", provider="dashscope-code")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "https://coding.dashscope.aliyuncs.com/v1"
        assert call_kwargs["api_key"] == "sk-sp-key-789"

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
        """MiniMax should have 5 direct model entries in _MODEL_ENTRIES."""
        minimax_models = get_models_for_provider("minimax")
        assert len(minimax_models) == 5
        model_names = {name for name, _ in minimax_models}
        assert "minimax-m3" in model_names
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

    def test_preserves_image_block(self):
        from EvoScientist.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        assert _flatten_message_content([img]) == [img]

    def test_preserves_image_url_block(self):
        from EvoScientist.llm.patches import _flatten_message_content

        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
        assert _flatten_message_content([img]) == [img]

    def test_preserves_file_block(self):
        # PDF/document files are preserved (capable models read them).
        from EvoScientist.llm.patches import _flatten_message_content

        f = {"type": "file", "base64": "FFF", "mime_type": "application/pdf"}
        assert _flatten_message_content([f]) == [f]

    def test_unsupported_media_dropped(self):
        # video/audio are NOT in the allowlist -> dropped, not crashing
        # (langchain-openai raises ValueError on `video`).
        from EvoScientist.llm.patches import _flatten_message_content

        for block in (
            {"type": "video", "base64": "VVV", "mime_type": "video/mp4"},
            {"type": "audio", "base64": "ZZZ", "mime_type": "audio/wav"},
        ):
            assert _flatten_message_content([block]) == ""

    def test_non_image_media_dropped_keeps_text(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [
            {"type": "text", "text": "hi"},
            {"type": "video", "base64": "VVV", "mime_type": "video/mp4"},
        ]
        # Video dropped, text kept -> plain string (no media list).
        assert _flatten_message_content(content) == "hi"

    def test_consolidates_text_and_image(self):
        from EvoScientist.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [{"type": "text", "text": "a photo"}, img]
        assert _flatten_message_content(content) == [
            {"type": "text", "text": "a photo"},
            img,
        ]

    def test_multiple_text_blocks_with_image(self):
        from EvoScientist.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            img,
        ]
        assert _flatten_message_content(content) == [
            {"type": "text", "text": "a\n\nb"},
            img,
        ]

    def test_preserves_text_media_ordering(self):
        # Text after an image must stay AFTER it (not consolidated to the front).
        from EvoScientist.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [
            {"type": "text", "text": "before"},
            img,
            {"type": "text", "text": "after"},
        ]
        assert _flatten_message_content(content) == [
            {"type": "text", "text": "before"},
            img,
            {"type": "text", "text": "after"},
        ]

    def test_thinking_dropped_image_kept(self):
        from EvoScientist.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [{"type": "thinking", "text": "hmm"}, img]
        assert _flatten_message_content(content) == [img]

    def test_pure_text_still_returns_string(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
        result = _flatten_message_content(content)
        assert result == "x\n\ny"
        assert isinstance(result, str)

    def test_unknown_nontext_block_still_dropped(self):
        from EvoScientist.llm.patches import _flatten_message_content

        content = [{"type": "tool_use", "id": "1", "name": "foo"}]
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

    def test_generate_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        model._generate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == [{"type": "text", "text": "see"}, img]

    @pytest.mark.anyio
    async def test_agenerate_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._agenerate
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        await model._agenerate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == [{"type": "text", "text": "see"}, img]

    def test_stream_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._stream
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        list(model._stream([msg]))

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == [{"type": "text", "text": "see"}, img]

    @pytest.mark.anyio
    async def test_astream_preserves_media(self):
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

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        chunks = []
        async for c in model._astream([msg]):
            chunks.append(c)

        assert chunks == ["c1", "c2"]
        assert received_msgs[0].content == [{"type": "text", "text": "see"}, img]

    def test_toolmessage_image_hoisted_to_human(self):
        from langchain_core.messages import ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)  # hoist_tool_media=True (OpenAI-compat)

        # deepagents read_file emits this exact shape for an image file.
        tm = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called_msgs = orig.call_args[0][0]
        # Tool content becomes a string placeholder (OpenAI-compat requirement) ...
        assert isinstance(called_msgs[0].content, str)
        # ... and the image is hoisted into a following HumanMessage.
        assert len(called_msgs) == 2
        hoisted = called_msgs[1]
        assert hoisted.type == "human"
        assert any(
            isinstance(b, dict) and b.get("type") == "image" for b in hoisted.content
        )

    def test_toolmessage_image_kept_inline_when_no_hoist(self):
        from langchain_core.messages import ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model, hoist_tool_media=False)  # Anthropic-routed

        tm = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called_msgs = orig.call_args[0][0]
        # No hoisting: image stays inline in the tool message content.
        assert len(called_msgs) == 1
        content = called_msgs[0].content
        assert isinstance(content, list)
        assert any(isinstance(b, dict) and b.get("type") == "image" for b in content)

    def test_parallel_tool_images_hoisted_after_tools(self):
        from langchain_core.messages import AIMessage, ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        ai = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "read_file", "args": {}},
                {"id": "c2", "name": "read_file", "args": {}},
            ],
        )
        t1 = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="c1",
            name="read_file",
        )
        t2 = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "BBB", "mime_type": "image/png"}
            ],
            tool_call_id="c2",
            name="read_file",
        )
        model._generate([ai, t1, t2])

        called_msgs = orig.call_args[0][0]
        # Tool results stay consecutive; one hoisted HumanMessage follows them.
        assert [m.type for m in called_msgs] == ["ai", "tool", "tool", "human"]
        assert isinstance(called_msgs[1].content, str)
        assert isinstance(called_msgs[2].content, str)
        imgs = [b for b in called_msgs[3].content if b.get("type") == "image"]
        assert len(imgs) == 2

    def test_assistant_text_still_flattened_to_string(self):
        from langchain_core.messages import AIMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        msg = AIMessage(
            content=[
                {"type": "text", "text": "hi"},
                {"type": "thinking", "text": "t"},
            ]
        )
        model._generate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hi"

    def test_tool_media_flushed_before_next_human(self):
        from langchain_core.messages import HumanMessage, ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        tm = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        nxt = HumanMessage(content="thanks")
        model._generate([tm, nxt])

        called = orig.call_args[0][0]
        # tool(placeholder), hoisted image (human), then the original human msg
        assert [m.type for m in called] == ["tool", "human", "human"]
        assert isinstance(called[0].content, str)
        assert any(b.get("type") == "image" for b in called[1].content)
        assert called[2].content == "thanks"

    def test_tool_message_text_and_image_split(self):
        from langchain_core.messages import ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        tm = ToolMessage(
            content=[
                {"type": "text", "text": "chart description"},
                {"type": "image", "base64": "AAA", "mime_type": "image/png"},
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called = orig.call_args[0][0]
        # Tool keeps the text as its string content; image hoisted to a human msg.
        assert called[0].content == "chart description"
        assert any(b.get("type") == "image" for b in called[1].content)

    def test_tool_message_interleaved_text_not_lost(self):
        # Interleaved [text, image, text] in a tool result: BOTH text runs must
        # survive the hoisting split (not just the first).
        from langchain_core.messages import ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        tm = ToolMessage(
            content=[
                {"type": "text", "text": "before"},
                {"type": "image", "base64": "AAA", "mime_type": "image/png"},
                {"type": "text", "text": "after"},
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called = orig.call_args[0][0]
        # both text runs preserved in the tool placeholder; image hoisted
        assert "before" in called[0].content
        assert "after" in called[0].content
        assert any(b.get("type") == "image" for b in called[1].content)


# =============================================================================
# Test no-vision fallback (models that reject image input)
# =============================================================================


class TestNoVisionFallback:
    """Verify image-rejecting models fall back to a text placeholder."""

    def _img_tool(self):
        from langchain_core.messages import ToolMessage

        return ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="t1",
            name="read_file",
        )

    def _make_model(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        model._agenerate = None
        model._stream = None
        model._astream = None
        return model

    def test_media_error_types(self):
        from EvoScientist.llm.patches import (
            _FILE_CONTENT_TYPES,
            _IMAGE_CONTENT_TYPES,
            _is_http_400,
            _media_error_types,
        )

        # marker identifies the specific modality
        assert (
            _media_error_types(Exception("No endpoints found that support image input"))
            >= _IMAGE_CONTENT_TYPES
        )
        assert (
            _media_error_types(Exception("file input is not supported"))
            == _FILE_CONTENT_TYPES
        )
        # DeepSeek-style maps to all media (generic "expected text")
        assert (
            _media_error_types(
                Exception("unknown variant `image_url`, expected `text`")
            )
            >= _IMAGE_CONTENT_TYPES
        )
        # non-media errors implicate nothing
        assert _media_error_types(Exception("rate limit exceeded")) == set()
        assert (
            _media_error_types(Exception("No endpoints found for some/model")) == set()
        )
        # bare "expected text" (non-media schema error) must NOT match
        assert (
            _media_error_types(
                Exception("tool schema validation failed: expected text")
            )
            == set()
        )

        class _E(Exception):
            status_code = 400

        assert _is_http_400(_E("bad request"))
        assert not _is_http_400(Exception("rate limit exceeded"))

    def test_media_types_in(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _media_types_in

        img = {"type": "image", "base64": "A", "mime_type": "image/png"}
        f = {"type": "file", "base64": "F", "mime_type": "application/pdf"}
        assert _media_types_in([HumanMessage(content=[img, f])]) == {"image", "file"}
        assert _media_types_in([HumanMessage(content="hi")]) == set()

    def test_strip_media_types_replaces_only_given(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _strip_media_types

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        f = {"type": "file", "base64": "FFF", "mime_type": "application/pdf"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img, f])
        # Strip only files -> image survives, file becomes a placeholder block.
        out = _strip_media_types([msg], {"file"})
        types = [b.get("type") for b in out[0].content if isinstance(b, dict)]
        assert "image" in types  # image preserved
        assert "file" not in types  # file stripped
        assert any(
            b.get("type") == "text" and "omitted" in b.get("text", "").lower()
            for b in out[0].content
        )

    def test_strip_media_types_preserves_position(self):
        # Stripped block is replaced IN PLACE; surrounding text/kept media keep
        # their order (placeholder where the image was, file stays last).
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _strip_media_types

        img = {"type": "image", "base64": "A", "mime_type": "image/png"}
        f = {"type": "file", "base64": "F", "mime_type": "application/pdf"}
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "t1"},
                img,
                {"type": "text", "text": "t2"},
                f,
            ]
        )
        out = _strip_media_types([msg], {"image"})  # block only image
        content = out[0].content
        assert all(b.get("type") != "image" for b in content)  # image gone
        # order preserved: t1, placeholder (where image was), t2, file
        assert content[0]["text"] == "t1"
        assert content[1]["type"] == "text"
        assert "omitted" in content[1]["text"].lower()
        assert content[2]["text"] == "t2"
        assert content[3]["type"] == "file"  # file kept at its original position

    def test_strip_media_types_dedups_consecutive(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _strip_media_types

        a = {"type": "image", "base64": "A", "mime_type": "image/png"}
        b = {"type": "image", "base64": "B", "mime_type": "image/png"}
        msg = HumanMessage(content=[a, b])
        out = _strip_media_types([msg], {"image"})
        # two adjacent stripped blocks collapse into ONE placeholder
        assert len(out[0].content) == 1
        assert "omitted" in out[0].content[0]["text"].lower()

    def test_profile_no_vision_strips_upfront(self):
        # Proactive: profile says image_inputs is False -> strip from the start,
        # no failing first request.
        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model.profile = {"image_inputs": False}
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        assert model._generate([self._img_tool()]) == "ok"
        assert len(calls) == 1  # no failed attempt
        assert all(isinstance(m.content, str) for m in calls[0])
        assert any("omitted" in m.content.lower() for m in calls[0])

    def test_profile_with_vision_does_not_strip(self):
        # Profile says image_inputs is True -> normal preserve path (no upfront strip).
        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model.profile = {"image_inputs": True}
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        assert model._generate([self._img_tool()]) == "ok"
        # Image preserved (hoisted), not replaced by a placeholder.
        assert any(
            isinstance(m.content, list)
            and any(b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_generate_falls_back_and_caches(self):
        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            if not state["raised"]:  # fail exactly once, ever
                state["raised"] = True
                raise Exception("unknown variant `image_url`, expected `text`")
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        tm = self._img_tool()
        # 1st turn: preserve attempt fails once -> strip -> ok
        assert model._generate([tm]) == "ok"
        assert len(calls) == 2
        retry = calls[1]
        assert all(isinstance(m.content, str) for m in retry)
        assert any("omitted" in m.content.lower() for m in retry)

        # 2nd turn: cached no-vision -> straight to stripped, single call (no failure)
        calls.clear()
        assert model._generate([tm]) == "ok"
        assert len(calls) == 1
        assert all(isinstance(m.content, str) for m in calls[0])

    def test_non_image_error_not_retried(self):
        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            raise Exception("rate limit exceeded")

        model._generate = _gen
        _patch_openai_compat_content(model)

        with pytest.raises(Exception, match="rate limit"):
            model._generate([self._img_tool()])
        assert len(calls) == 1

    def test_stream_falls_back(self):
        from unittest.mock import MagicMock

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model._generate = MagicMock(return_value="g")
        calls = []

        def _stream(msgs, *a, **k):
            calls.append(msgs)
            if len(calls) == 1:
                raise Exception("No endpoints found that support image input")
            yield from ["x", "y"]

        model._stream = _stream
        _patch_openai_compat_content(model)

        out = list(model._stream([self._img_tool()]))
        assert out == ["x", "y"]
        assert len(calls) == 2

    @pytest.mark.anyio
    async def test_astream_falls_back(self):
        from unittest.mock import MagicMock

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model._generate = MagicMock(return_value="g")
        calls = []

        async def _astream(msgs, *a, **k):
            calls.append(msgs)
            if len(calls) == 1:
                raise Exception("No endpoints found that support image input")
            for c in ["x", "y"]:
                yield c

        model._astream = _astream
        _patch_openai_compat_content(model)

        out = [c async for c in model._astream([self._img_tool()])]
        assert out == ["x", "y"]
        assert len(calls) == 2

    def test_unrelated_400_retry_fails_not_cached(self):
        # A non-media 400 (e.g. tool schema) whose stripped retry ALSO fails must
        # surface the original error and must NOT permanently flip to no-media.
        from EvoScientist.llm.patches import _patch_openai_compat_content

        class _E(Exception):
            status_code = 400

        model = self._make_model()
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            raise _E("invalid tool schema")  # 400, not media; fails every time

        model._generate = _gen
        _patch_openai_compat_content(model)

        tm = self._img_tool()
        with pytest.raises(_E):
            model._generate([tm])
        assert len(calls) == 2  # preserve attempt + stripped retry (both fail)

        # Not cached: the next call attempts preserve again (not straight-to-stripped)
        calls.clear()
        with pytest.raises(_E):
            model._generate([tm])
        assert len(calls) == 2

    def test_pdf_rejection_does_not_disable_images(self):
        # Per-modality: a PDF/file rejection caches only file types; a later
        # image must still be preserved (not stripped).
        from langchain_core.messages import HumanMessage, ToolMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            has_file = any(
                isinstance(m.content, list)
                and any(
                    isinstance(b, dict) and b.get("type") == "file" for b in m.content
                )
                for m in msgs
            )
            if has_file and not state["raised"]:
                state["raised"] = True
                raise Exception("file input is not supported")
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        pdf_tm = ToolMessage(
            content_blocks=[
                {"type": "file", "base64": "F", "mime_type": "application/pdf"}
            ],
            tool_call_id="t1",
            name="read_file",
        )
        assert model._generate([pdf_tm]) == "ok"  # file rejected -> stripped -> ok

        # Now an image: must still be preserved (images not blocked by a PDF reject)
        calls.clear()
        img_msg = HumanMessage(
            content=[{"type": "image", "base64": "A", "mime_type": "image/png"}]
        )
        assert model._generate([img_msg]) == "ok"
        assert len(calls) == 1  # single attempt, no failure
        assert any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_bare_400_recovers_but_not_cached(self):
        # A bare 400 with NO media marker recovers this request (stripped retry)
        # but must NOT cache (no permanent degradation) — High #1.
        from EvoScientist.llm.patches import _patch_openai_compat_content

        class _E(Exception):
            status_code = 400

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            if not state["raised"]:
                state["raised"] = True
                raise _E("transient bad request")  # 400, no media marker
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        tm = self._img_tool()
        assert model._generate([tm]) == "ok"  # bare 400 -> stripped retry -> ok
        assert len(calls) == 2

        # NOT cached: the next call still attempts preserve (image kept, not stripped)
        calls.clear()
        assert model._generate([tm]) == "ok"
        assert len(calls) == 1
        assert any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_mixed_modality_caches_only_culprit(self):
        # image+file message; provider rejects only the file -> cache file only,
        # images stay preserved on later turns — High #2.
        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            if not state["raised"]:
                state["raised"] = True
                raise Exception("file input is not supported")
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        mixed = HumanMessage(
            content=[
                {"type": "image", "base64": "A", "mime_type": "image/png"},
                {"type": "file", "base64": "F", "mime_type": "application/pdf"},
            ]
        )
        assert model._generate([mixed]) == "ok"  # file rejected -> retry -> cache file

        # later image-only request: image must still be preserved
        calls.clear()
        img = HumanMessage(
            content=[{"type": "image", "base64": "A", "mime_type": "image/png"}]
        )
        assert model._generate([img]) == "ok"
        assert len(calls) == 1
        assert any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_stream_empty_retry_raises_original(self):
        # If the stripped streaming retry yields ZERO chunks, surface the
        # original error instead of silently returning an empty stream.
        from unittest.mock import MagicMock

        from EvoScientist.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model._generate = MagicMock(return_value="g")
        calls = []

        def _stream(msgs, *a, **k):
            calls.append(msgs)
            if len(calls) == 1:
                raise Exception("No endpoints found that support image input")
            return  # retry yields nothing
            yield  # pragma: no cover  (makes this a generator)

        model._stream = _stream
        _patch_openai_compat_content(model)

        with pytest.raises(Exception, match="support image"):
            list(model._stream([self._img_tool()]))
        assert len(calls) == 2


# =============================================================================
# Test _patch_deepseek_reasoning_passback
# =============================================================================


class TestPatchDeepseekReasoningPassback:
    """Verify reasoning_content is injected into DeepSeek payload assistant messages.

    This patch fixes the 400 error from DeepSeek V4 thinking mode in multi-turn
    + tool_use scenarios.  See langchain PR #34516 for upstream reference.
    """

    def _make_model(self, model_name="deepseek-v4-pro", payload_messages=None):
        """Create a mock ChatOpenAI-like model for the DeepSeek base URL."""
        from unittest.mock import MagicMock

        if payload_messages is None:
            payload_messages = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "ok"},
            ]

        model = MagicMock()
        model.model_name = model_name

        class _Wrapped:
            def __init__(self, msgs):
                self._msgs = msgs

            def to_messages(self):
                return self._msgs

        model._convert_input = lambda x: _Wrapped(x)
        model._get_request_payload = MagicMock(
            return_value={"messages": payload_messages}
        )
        return model

    def test_injects_reasoning_content_from_additional_kwargs(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = self._make_model()
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("hi"),
            AIMessage(
                content="hello",
                additional_kwargs={"reasoning_content": "let me think..."},
            ),
            HumanMessage("ok"),
        ]
        payload = model._get_request_payload(messages)

        assert payload["messages"][1]["reasoning_content"] == "let me think..."

    def test_empty_reasoning_for_reasoner_model(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = self._make_model(model_name="deepseek-reasoner")
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("hi"),
            AIMessage(content="hello"),  # no reasoning_content
            HumanMessage("ok"),
        ]
        payload = model._get_request_payload(messages)

        assert payload["messages"][1]["reasoning_content"] == ""

    def test_empty_fallback_for_non_reasoner_without_rc(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = self._make_model(model_name="deepseek-v4-pro")
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("hi"),
            AIMessage(content="hello"),  # no reasoning_content
            HumanMessage("ok"),
        ]
        payload = model._get_request_payload(messages)

        # Empty-string fallback applies to ALL DeepSeek models (not just
        # reasoner) so cross-provider history doesn't trigger 400.
        assert payload["messages"][1]["reasoning_content"] == ""

    def test_handles_multiple_ai_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = self._make_model(
            payload_messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ]
        )
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("q1"),
            AIMessage(content="a1", additional_kwargs={"reasoning_content": "rc1"}),
            HumanMessage("q2"),
            AIMessage(content="a2", additional_kwargs={"reasoning_content": "rc2"}),
            HumanMessage("q3"),
        ]
        payload = model._get_request_payload(messages)

        assert payload["messages"][1]["reasoning_content"] == "rc1"
        assert payload["messages"][3]["reasoning_content"] == "rc2"

    def test_real_world_tool_use_flow(self):
        """The real scenario this patch was built for: AI thinks → tool_call →
        ToolMessage → next turn must carry reasoning_content from prior AI msg.

        This mirrors what happens in /tmp/verify_deepseek.py and what the user
        actually triggers via 'create file then read it' in EvoSci CLI.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        # Mock payload that mirrors what langchain-openai produces:
        # user → assistant (with tool_calls) → tool_result → user (next turn)
        model = self._make_model(
            payload_messages=[
                {"role": "user", "content": "Read hello.txt"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "file contents", "tool_call_id": "call_1"},
                {"role": "user", "content": "now what?"},
            ]
        )
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("Read hello.txt"),
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "I should call read_file"},
                tool_calls=[{"name": "read_file", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="file contents", tool_call_id="call_1"),
            HumanMessage("now what?"),
        ]
        payload = model._get_request_payload(messages)

        # The assistant message (index 1) must carry reasoning_content
        assistant_msg = payload["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["reasoning_content"] == "I should call read_file"
        # tool_calls preserved
        assert "tool_calls" in assistant_msg
        # ToolMessage (index 2) untouched
        assert "reasoning_content" not in payload["messages"][2]

    def test_mixed_ai_messages_with_and_without_rc(self):
        """Some AIMessages have reasoning_content, some don't (e.g., legacy turns
        before patch was deployed). Each should be handled independently."""
        from langchain_core.messages import AIMessage, HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = self._make_model(
            model_name="deepseek-v4-pro",
            payload_messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},  # no rc
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},  # has rc
                {"role": "user", "content": "q3"},
            ],
        )
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("q1"),
            AIMessage(content="a1"),  # no reasoning_content
            HumanMessage("q2"),
            AIMessage(
                content="a2",
                additional_kwargs={"reasoning_content": "rc2"},
            ),
            HumanMessage("q3"),
        ]
        payload = model._get_request_payload(messages)

        # First AI msg: no rc → empty-string fallback (covers cross-model switch)
        assert payload["messages"][1]["reasoning_content"] == ""
        # Second AI msg: has rc → injected
        assert payload["messages"][3]["reasoning_content"] == "rc2"

    def test_handles_responses_api_payload(self):
        """Payload without 'messages' key (e.g. Responses API) should not crash."""
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = MagicMock()
        model.model_name = "deepseek-v4-pro"

        class _Wrapped:
            def __init__(self, msgs):
                self._msgs = msgs

            def to_messages(self):
                return self._msgs

        model._convert_input = lambda x: _Wrapped(x)
        # Simulate Responses API payload (no 'messages' key)
        model._get_request_payload = MagicMock(
            return_value={"input": [{"role": "user", "content": "hi"}]}
        )

        _patch_deepseek_reasoning_passback(model)

        # Should not raise, just return the payload as-is
        payload = model._get_request_payload([HumanMessage("hi")])
        assert "input" in payload
        assert "messages" not in payload

    def test_cross_provider_switch_history(self):
        """User chats with Anthropic/OpenAI then switches to DeepSeek V4 Pro.

        Historical AI messages have no reasoning_content (the previous
        provider never produced it). The patch must inject an empty-string
        fallback so DeepSeek doesn't 400 on
        "reasoning_content must be passed back to the API".
        """
        from langchain_core.messages import AIMessage, HumanMessage

        from EvoScientist.llm.patches import _patch_deepseek_reasoning_passback

        model = self._make_model(
            model_name="deepseek-v4-pro",
            payload_messages=[
                {"role": "user", "content": "earlier question to anthropic"},
                {"role": "assistant", "content": "anthropic answer"},
                {"role": "user", "content": "now ask deepseek pro"},
            ],
        )
        _patch_deepseek_reasoning_passback(model)

        messages = [
            HumanMessage("earlier question to anthropic"),
            AIMessage(content="anthropic answer"),  # no reasoning_content
            HumanMessage("now ask deepseek pro"),
        ]
        payload = model._get_request_payload(messages)

        assert payload["messages"][1]["reasoning_content"] == ""


# =============================================================================
# Test _patch_openai_capture_reasoning_content (module-level monkey-patch)
# =============================================================================


class TestPatchOpenAICaptureReasoningContent:
    """Verify reasoning_content is captured into AIMessage.additional_kwargs.

    This patch is applied at import time and globally affects langchain-openai's
    _convert_dict_to_message and _convert_delta_to_message_chunk.
    """

    def test_capture_from_non_streaming_response(self):
        """reasoning_content in OpenAI response dict → AIMessage.additional_kwargs."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message(
            {
                "role": "assistant",
                "content": "hi",
                "reasoning_content": "let me think...",
            }
        )
        assert msg.additional_kwargs.get("reasoning_content") == "let me think..."

    def test_capture_absent_when_field_missing(self):
        """No reasoning_content in response → not added to additional_kwargs."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message({"role": "assistant", "content": "hi"})
        assert "reasoning_content" not in msg.additional_kwargs

    def test_capture_from_streaming_chunk(self):
        """reasoning_content delta is captured onto the chunk's additional_kwargs."""
        from langchain_core.messages import AIMessageChunk
        from langchain_openai.chat_models.base import (
            _convert_delta_to_message_chunk,
        )

        chunk = _convert_delta_to_message_chunk(
            {"role": "assistant", "content": "", "reasoning_content": "thinking"},
            AIMessageChunk,
        )
        assert chunk.additional_kwargs.get("reasoning_content") == "thinking"

    def test_capture_does_not_affect_other_fields(self):
        """Existing tool_calls / function_call extraction unaffected."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message(
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "use the tool",
            }
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.additional_kwargs.get("reasoning_content") == "use the tool"


class TestIsResponsesReasoningItem:
    """_is_responses_reasoning_item flags encrypted OpenAI-Responses items."""

    def test_rs_id_is_responses_item(self):
        from EvoScientist.llm.patches import _is_responses_reasoning_item

        assert _is_responses_reasoning_item({"id": "rs_09363d42", "type": "x"})

    def test_encrypted_data_is_responses_item(self):
        from EvoScientist.llm.patches import _is_responses_reasoning_item

        assert _is_responses_reasoning_item({"data": "gAAAAAB...", "type": "x"})

    def test_plain_text_reasoning_is_not_responses_item(self):
        from EvoScientist.llm.patches import _is_responses_reasoning_item

        assert not _is_responses_reasoning_item(
            {"type": "reasoning.text", "text": "thinking", "index": 0}
        )
        assert not _is_responses_reasoning_item("not a dict")


class TestPatchOpenrouterStripResponsesReasoning:
    """OpenAI-Responses encrypted reasoning items (`rs_` id / encrypted data)
    are stripped from outgoing OpenRouter assistant messages, preventing the
    multi-turn "Item with id 'rs_...' not found" 400 (store=false; #37777).
    """

    def _apply(self):
        import langchain_openrouter.chat_models as mod

        import EvoScientist.llm.patches as patches

        orig = mod._convert_message_to_dict
        orig_flag = patches._openrouter_reasoning_strip_patched
        patches._openrouter_reasoning_strip_patched = False
        patches._patch_openrouter_strip_responses_reasoning()
        return patches, mod, orig, orig_flag

    @staticmethod
    def _restore(patches, mod, orig, orig_flag):
        mod._convert_message_to_dict = orig
        patches._openrouter_reasoning_strip_patched = orig_flag

    def test_strips_encrypted_item_drops_key_when_empty(self):
        from langchain_core.messages import AIMessage

        patches, mod, orig, orig_flag = self._apply()
        try:
            msg = AIMessage(
                content="done",
                additional_kwargs={
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "format": "openai-responses-v1",
                            "id": "rs_09363d42b054",
                            "data": "gAAAAAB...",
                            "summary": "real reasoning text",
                            "index": 0,
                        }
                    ],
                },
            )
            result = mod._convert_message_to_dict(msg)
            # sole entry was an rs_ item → reasoning_details removed entirely.
            assert "reasoning_details" not in result
        finally:
            self._restore(patches, mod, orig, orig_flag)

    def test_keeps_plain_text_reasoning(self):
        from langchain_core.messages import AIMessage

        patches, mod, orig, orig_flag = self._apply()
        try:
            msg = AIMessage(
                content="done",
                additional_kwargs={
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "thinking", "index": 0},
                        {"id": "rs_abc", "data": "blob", "index": 1},
                    ],
                },
            )
            result = mod._convert_message_to_dict(msg)
            kept = result["reasoning_details"]
            assert len(kept) == 1
            assert kept[0]["type"] == "reasoning.text"
        finally:
            self._restore(patches, mod, orig, orig_flag)

    def test_does_not_mutate_original_message(self):
        from langchain_core.messages import AIMessage

        patches, mod, orig, orig_flag = self._apply()
        try:
            details = [{"id": "rs_abc", "data": "blob"}]
            msg = AIMessage(
                content="x", additional_kwargs={"reasoning_details": details}
            )
            mod._convert_message_to_dict(msg)
            # stored history untouched — we filter a fresh list, not in place.
            assert details == [{"id": "rs_abc", "data": "blob"}]
        finally:
            self._restore(patches, mod, orig, orig_flag)

    def test_patch_is_idempotent(self):
        patches, mod, orig, orig_flag = self._apply()
        try:
            wrapper = mod._convert_message_to_dict
            # Second call is guarded by the flag → must not re-wrap.
            patches._patch_openrouter_strip_responses_reasoning()
            assert mod._convert_message_to_dict is wrapper
        finally:
            self._restore(patches, mod, orig, orig_flag)

    def test_non_dict_entry_is_kept(self):
        from langchain_core.messages import AIMessage

        patches, mod, orig, orig_flag = self._apply()
        try:
            msg = AIMessage(
                content="done",
                additional_kwargs={
                    "reasoning_details": [
                        "opaque",  # non-dict slipped in → kept, not crashed on
                        {"id": "rs_abc", "data": "blob", "index": 1},
                    ],
                },
            )
            result = mod._convert_message_to_dict(msg)
            assert result["reasoning_details"] == ["opaque"]
        finally:
            self._restore(patches, mod, orig, orig_flag)


# =============================================================================
# Test _apply_auto_config
# =============================================================================


class TestAutoConfig:
    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_5_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-5 models get enabled thinking with budget."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-haiku-4-5")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_6_adaptive_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-6 models get adaptive thinking with max effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call_kwargs["effort"] == "max"

    @patch("EvoScientist.llm.models.init_chat_model")
    def test_anthropic_4_8_adaptive_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-8 models get adaptive thinking with max effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-opus-4-8")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
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

        get_chat_model("claude-haiku-4-5")

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
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
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

        get_chat_model("gpt-5.5", provider="openai")
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
