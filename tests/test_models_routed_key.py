"""Tests for get_chat_model routed-provider API key handling (issue #498).

Routed providers (minimax / kimi-coding / custom-anthropic via ChatAnthropic;
novita / siliconflow / moonshot / custom-openai via ChatOpenAI) must fail fast
when their own API key is missing instead of silently falling back to
ANTHROPIC_API_KEY / OPENAI_API_KEY and sending that key to the third-party
endpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.llm.models import get_chat_model

# Env vars that must be cleared so the routed provider's own key is genuinely
# absent, and so the default-vendor fallback keys don't mask the bug.
_ANTHROPIC_ROUTED_ENVS = [
    "MINIMAX_API_KEY",
    "KIMI_API_KEY",
    "CUSTOM_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "MINIMAX_BASE_URL",
    "CUSTOM_ANTHROPIC_BASE_URL",
]
_OPENAI_ROUTED_ENVS = [
    "NOVITA_API_KEY",
    "SILICONFLOW_API_KEY",
    "MOONSHOT_API_KEY",
    "CUSTOM_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "CUSTOM_OPENAI_BASE_URL",
]


@pytest.fixture(autouse=True)
def _clean_routed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every routed-provider and default-vendor key for each test."""
    for name in _ANTHROPIC_ROUTED_ENVS + _OPENAI_ROUTED_ENVS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Anthropic-routed providers
# ---------------------------------------------------------------------------


def test_minimax_without_own_key_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """minimax must not silently use ANTHROPIC_API_KEY when MINIMAX_API_KEY is unset."""
    # Simulate the bug scenario: ANTHROPIC_API_KEY is set but MINIMAX_API_KEY is not.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")

    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        get_chat_model("minimax-m3", provider="minimax")


def test_minimax_with_own_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MINIMAX_API_KEY is set the model constructs normally."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-minimax-test")

    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("minimax-m3", provider="minimax")

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["model_provider"] == "anthropic"
    assert call_kwargs["api_key"] == "sk-minimax-test"
    assert call_kwargs["base_url"] == "https://api.minimaxi.com/anthropic"


def test_minimax_explicit_api_key_kwarg_succeeds() -> None:
    """An explicit api_key= kwarg bypasses the env-var requirement."""
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model(
            "minimax-m3", provider="minimax", api_key="sk-explicit"
        )

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["api_key"] == "sk-explicit"


def test_minimax_explicit_none_api_key_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key=None must not bypass the routed-provider key requirement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")

    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        get_chat_model("minimax-m3", provider="minimax", api_key=None)


def test_minimax_explicit_empty_api_key_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key='' must not bypass the routed-provider key requirement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")

    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        get_chat_model("minimax-m3", provider="minimax", api_key="")


def test_kimi_coding_without_own_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """kimi-coding must raise KIMI_API_KEY error when its own key is unset."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
    with pytest.raises(ValueError, match="KIMI_API_KEY"):
        get_chat_model("kimi-for-coding", provider="kimi-coding")


# ---------------------------------------------------------------------------
# OpenAI-routed providers
# ---------------------------------------------------------------------------


def test_novita_without_own_key_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """novita must not silently use OPENAI_API_KEY when NOVITA_API_KEY is unset."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")

    with pytest.raises(ValueError, match="NOVITA_API_KEY"):
        get_chat_model("moonshotai/kimi-k3", provider="novita")


def test_novita_with_own_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """When NOVITA_API_KEY is set the model constructs normally via OpenAI."""
    monkeypatch.setenv("NOVITA_API_KEY", "sk-novita-test")

    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("moonshotai/kimi-k3", provider="novita")

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["model_provider"] == "openai"
    assert call_kwargs["api_key"] == "sk-novita-test"


def test_novita_explicit_api_key_kwarg_succeeds() -> None:
    """An explicit api_key= kwarg bypasses the novita env-var requirement."""
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model(
            "moonshotai/kimi-k3", provider="novita", api_key="sk-explicit"
        )

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["api_key"] == "sk-explicit"


def test_novita_explicit_none_api_key_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key=None must not bypass the novita key requirement."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")

    with pytest.raises(ValueError, match="NOVITA_API_KEY"):
        get_chat_model("moonshotai/kimi-k3", provider="novita", api_key=None)


def test_novita_explicit_empty_api_key_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key='' must not bypass the novita key requirement."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")

    with pytest.raises(ValueError, match="NOVITA_API_KEY"):
        get_chat_model("moonshotai/kimi-k3", provider="novita", api_key="")


def test_siliconflow_without_own_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """siliconflow must raise SILICONFLOW_API_KEY error when its own key is unset."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        get_chat_model("Pro/zai-org/GLM-5.2", provider="siliconflow")


def test_moonshot_without_own_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """moonshot must raise MOONSHOT_API_KEY error when its own key is unset."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    with pytest.raises(ValueError, match="MOONSHOT_API_KEY"):
        get_chat_model("kimi-k3", provider="moonshot")


# ---------------------------------------------------------------------------
# custom-* routed providers (base_url from env, key from env or kwarg)
# ---------------------------------------------------------------------------


def test_custom_anthropic_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """custom-anthropic must raise CUSTOM_ANTHROPIC_API_KEY error when key is unset."""
    monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "https://my-proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
    with pytest.raises(ValueError, match="CUSTOM_ANTHROPIC_API_KEY"):
        get_chat_model("claude-sonnet-4-6", provider="custom-anthropic")


def test_custom_openai_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """custom-openai must raise CUSTOM_OPENAI_API_KEY error when key is unset."""
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://my-proxy.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    with pytest.raises(ValueError, match="CUSTOM_OPENAI_API_KEY"):
        get_chat_model("gpt-5.5", provider="custom-openai")


def test_custom_openai_explicit_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """custom-openai with explicit api_key= kwarg constructs normally."""
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://my-proxy.example.com/v1")
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model(
            "gpt-5.5", provider="custom-openai", api_key="sk-explicit"
        )

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["api_key"] == "sk-explicit"
    assert call_kwargs["base_url"] == "https://my-proxy.example.com/v1"


# ---------------------------------------------------------------------------
# Regression: native providers are NOT affected by the routed key check
# ---------------------------------------------------------------------------


def test_native_anthropic_without_key_does_not_raise_routed_error() -> None:
    """Native anthropic should not hit the routed-provider key check."""
    # No ANTHROPIC_API_KEY set — init_chat_model may raise its own error, but
    # it must NOT be our routed-key ValueError.
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("claude-sonnet-4-6", provider="anthropic")

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["model_provider"] == "anthropic"
    assert "api_key" not in call_kwargs


def test_native_openai_without_key_does_not_raise_routed_error() -> None:
    """Native openai should not hit the routed-provider key check."""
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("gpt-5.5", provider="openai")

    assert model is not None
    _, call_kwargs = mock_init.call_args
    assert call_kwargs["model_provider"] == "openai"
    assert "api_key" not in call_kwargs
