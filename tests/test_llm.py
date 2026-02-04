"""Tests for EvoScientist LLM module."""

from unittest import mock

import pytest

from EvoScientist.llm import (
    MODELS,
    DEFAULT_MODEL,
    get_chat_model,
    list_models,
    get_model_info,
)


# =============================================================================
# Test MODELS registry
# =============================================================================


class TestModelsRegistry:
    def test_models_is_dict(self):
        """Test that MODELS is a dictionary."""
        assert isinstance(MODELS, dict)

    def test_models_has_expected_keys(self):
        """Test that MODELS contains expected model short names."""
        expected_keys = [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
        ]
        for key in expected_keys:
            assert key in MODELS, f"Expected model '{key}' not in MODELS"

    def test_models_values_are_tuples(self):
        """Test that MODELS values are (model_id, provider) tuples."""
        for name, value in MODELS.items():
            assert isinstance(value, tuple), f"MODELS['{name}'] is not a tuple"
            assert len(value) == 2, f"MODELS['{name}'] doesn't have 2 elements"
            model_id, provider = value
            assert isinstance(model_id, str), f"model_id for '{name}' is not a string"
            assert isinstance(provider, str), f"provider for '{name}' is not a string"
            assert provider in ("anthropic", "openai", "nvidia"), f"Unknown provider for '{name}': {provider}"

    def test_anthropic_models_have_anthropic_provider(self):
        """Test that claude models use anthropic provider."""
        for name, (model_id, provider) in MODELS.items():
            if name.startswith("claude"):
                assert provider == "anthropic", f"Claude model '{name}' doesn't use anthropic provider"

    def test_openai_models_have_openai_provider(self):
        """Test that gpt/o1 models use openai provider."""
        for name, (model_id, provider) in MODELS.items():
            if name.startswith(("gpt", "o1")):
                assert provider == "openai", f"OpenAI model '{name}' doesn't use openai provider"


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
        model_id, provider = get_model_info("gpt-4o")
        assert model_id == "gpt-4o"
        assert provider == "openai"


# =============================================================================
# Test get_chat_model
# =============================================================================


class TestGetChatModel:
    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_uses_default_model_when_none(self, mock_init):
        """Test that get_chat_model uses default model when model=None."""
        mock_init.return_value = "mock_model"

        result = get_chat_model()

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        # Default model should be resolved from MODELS
        expected_model_id, expected_provider = MODELS[DEFAULT_MODEL]
        assert call_kwargs["model"] == expected_model_id
        assert call_kwargs["model_provider"] == expected_provider

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_resolves_short_name(self, mock_init):
        """Test that get_chat_model resolves short names correctly."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-opus-4-5")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-5-20251101"
        assert call_kwargs["model_provider"] == "anthropic"

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_resolves_openai_short_name(self, mock_init):
        """Test that get_chat_model resolves OpenAI short names."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-4o-mini")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["model_provider"] == "openai"

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_uses_full_model_id(self, mock_init):
        """Test that get_chat_model accepts full model IDs."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-3-opus-20240229")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"
        # Should infer anthropic from the model prefix
        assert call_kwargs["model_provider"] == "anthropic"

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_provider_override(self, mock_init):
        """Test that provider can be overridden."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-sonnet-4-5", provider="custom_provider")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "custom_provider"

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_passes_kwargs(self, mock_init):
        """Test that additional kwargs are passed through."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-4o", temperature=0.7, max_tokens=1000)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_infers_openai_from_gpt_prefix(self, mock_init):
        """Test that OpenAI is inferred from gpt- prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-4-turbo-preview")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_infers_openai_from_o1_prefix(self, mock_init):
        """Test that OpenAI is inferred from o1 prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("o1-preview")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"

    @mock.patch("EvoScientist.llm.models.init_chat_model")
    def test_defaults_to_anthropic_for_unknown(self, mock_init):
        """Test that anthropic is default for unknown model prefixes."""
        mock_init.return_value = "mock_model"

        get_chat_model("some-unknown-model")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"
