"""Tests for provider-agnostic context-window resolution."""

from types import SimpleNamespace

from EvoScientist.llm.context_window import (
    DEFAULT_CONTEXT_WINDOW_FALLBACK,
    get_context_window,
    resolve_context_window,
)


def test_prefers_direct_context_window_over_profile():
    model = SimpleNamespace(
        context_window=512_000,
        profile={"max_input_tokens": 200_000},
    )

    assert get_context_window(model) == 512_000


def test_uses_direct_context_length_attribute():
    model = SimpleNamespace(context_length=1_000_000)

    assert get_context_window(model) == 1_000_000


def test_uses_ollama_num_ctx():
    model = SimpleNamespace(model="llama3.2", num_ctx=65_536, profile=None)

    assert get_context_window(model) == 65_536


def test_uses_profile_context_length_before_max_input_tokens():
    model = SimpleNamespace(
        profile={
            "context_length": 1_000_000,
            "max_input_tokens": 128_000,
        }
    )

    assert get_context_window(model) == 1_000_000


def test_uses_profile_max_input_tokens_when_needed():
    model = SimpleNamespace(profile={"max_input_tokens": 200_000})

    assert get_context_window(model) == 200_000


def test_accepts_numeric_string_values():
    model = SimpleNamespace(profile={"context_length": "1_048_576"})

    assert get_context_window(model) == 1_048_576


def test_resolve_context_window_falls_back_when_missing():
    model = SimpleNamespace(profile=None)

    assert get_context_window(model) is None
    assert resolve_context_window(model) == DEFAULT_CONTEXT_WINDOW_FALLBACK
    assert resolve_context_window(model, fallback=42_000) == 42_000
