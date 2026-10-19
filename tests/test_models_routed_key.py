"""Tests for get_chat_model routed-provider API key handling (issue #498).

Routed providers share the OpenAI / Anthropic vendor SDK but point at a
different base_url. They must require their own API key and never silently
fall back to the vendor key, which would send the routed provider's
requests to the wrong host with the user's vendor credentials.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.llm.models import get_chat_model
from EvoScientist.llm.registry import (
    _ANTHROPIC_ROUTED_PROVIDERS,
    _OPENAI_ROUTED_PROVIDERS,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no provider keys or base URLs in the env.

    Earlier test files can leak provider keys into os.environ, which would
    make these tests pass only because of test order.
    """
    for _, api_key_env in (
        *_OPENAI_ROUTED_PROVIDERS.values(),
        *_ANTHROPIC_ROUTED_PROVIDERS.values(),
    ):
        monkeypatch.delenv(api_key_env, raising=False)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Registry-driven cases: every routed provider apart from custom-* must fail
# fast without its own key. Built from the registry so newly added providers
# are covered automatically. There are 16 entries apart from custom-*.
# ---------------------------------------------------------------------------

_ROUTED_CASES = [
    (provider, api_key_env, vendor_key_env)
    for table, vendor_key_env in (
        (_OPENAI_ROUTED_PROVIDERS, "OPENAI_API_KEY"),
        (_ANTHROPIC_ROUTED_PROVIDERS, "ANTHROPIC_API_KEY"),
    )
    for provider, (_, api_key_env) in table.items()
    if not provider.startswith("custom-")
]


@pytest.mark.parametrize(
    ("provider", "api_key_env", "vendor_key_env"),
    _ROUTED_CASES,
    ids=[c[0] for c in _ROUTED_CASES],
)
def test_routed_provider_without_own_key_raises(
    provider: str,
    api_key_env: str,
    vendor_key_env: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A routed provider must raise its own key error even when the vendor key
    is present, rather than silently using it."""
    monkeypatch.setenv(vendor_key_env, "sk-vendor-fallback")

    with pytest.raises(ValueError, match=api_key_env):
        get_chat_model("some-model", provider=provider)


# ---------------------------------------------------------------------------
# Explicit api_key=None / "" still trigger the missing-key check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("explicit", [None, ""])
def test_explicit_blank_key_raises(
    explicit: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-vendor-fallback")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        get_chat_model("minimax-m3", provider="minimax", api_key=explicit)


def test_explicit_key_used_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit api_key= kwarg is used as-is even when the provider's env
    var is unset and the vendor fallback key is present."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-vendor-fallback")
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        get_chat_model("minimax-m3", provider="minimax", api_key="sk-explicit")
    assert mock_init.call_args[1]["api_key"] == "sk-explicit"


# ---------------------------------------------------------------------------
# With the provider's own key: construction succeeds and the vendor key is
# never used.
# ---------------------------------------------------------------------------


def test_minimax_with_own_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MINIMAX_API_KEY is set the model constructs normally and the
    vendor key is not leaked."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-minimax-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-vendor-must-not-be-used")

    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("minimax-m3", provider="minimax")

    assert model is not None
    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["model_provider"] == "anthropic"
    assert call_kwargs["api_key"] == "sk-minimax-test"
    assert call_kwargs["api_key"] != "sk-ant-vendor-must-not-be-used"
    assert call_kwargs["base_url"] == "https://api.minimaxi.com/anthropic"


def test_novita_with_own_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """When NOVITA_API_KEY is set the model constructs via OpenAI and the
    vendor key is not leaked."""
    monkeypatch.setenv("NOVITA_API_KEY", "sk-novita-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-vendor-must-not-be-used")

    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("moonshotai/kimi-k3", provider="novita")

    assert model is not None
    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["model_provider"] == "openai"
    assert call_kwargs["api_key"] == "sk-novita-test"
    assert call_kwargs["api_key"] != "sk-openai-vendor-must-not-be-used"


# ---------------------------------------------------------------------------
# custom-* providers: placeholder key is allowed and warned about once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model", "prefix"),
    [
        ("custom-openai", "gpt-5.5", "CUSTOM_OPENAI"),
        ("custom-anthropic", "claude-sonnet-4-6", "CUSTOM_ANTHROPIC"),
    ],
)
def test_custom_placeholder_warns_once(
    provider: str,
    model: str,
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom-* endpoint with no key gets a placeholder; the warning is
    emitted on the first build only and not repeated on the second."""
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.delenv(f"{prefix}_API_KEY", raising=False)
    from EvoScientist.llm import models

    # Temporarily replace the cache for this test; monkeypatch restores the
    # original (possibly populated) cache afterward so other providers keep
    # their once-per-provider state.
    monkeypatch.setattr(models, "_PLACEHOLDER_WARNED", set())

    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        # First build: warn.
        with pytest.warns(UserWarning, match=f"{prefix.lower()}_api_key"):
            get_chat_model(model, provider=provider)
        assert mock_init.call_args[1]["api_key"] == "not-needed"
        # Second build: must not warn again.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            get_chat_model(model, provider=provider)
        assert mock_init.call_args[1]["api_key"] == "not-needed"


# ---------------------------------------------------------------------------
# Native (non-routed) providers are unaffected
# ---------------------------------------------------------------------------


def test_native_openai_without_key_does_not_raise_routed_error() -> None:
    """Native openai should not hit the routed-provider key check."""
    with patch("EvoScientist.llm.models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = get_chat_model("gpt-5.5", provider="openai")

    assert model is not None
    assert "api_key" not in mock_init.call_args[1]
