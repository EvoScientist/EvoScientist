"""Regression tests for the review findings on #713.

The picker offers an id the endpoint listed, but ``get_chat_model`` used to
rewrite that id into another provider's model id when the names collided, so
the request carried an id the endpoint does not serve. A discovered id was
also offered twice when the endpoint listed it twice, and a blank id was
offered at all.
"""

from __future__ import annotations

import pytest

from EvoScientist.llm.custom_openai_discovery import _names_from_body
from EvoScientist.llm.models import get_chat_model


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "sk-test")


def test_a_discovered_id_that_collides_with_another_provider_is_sent_as_listed(
    endpoint,
):
    """The endpoint listed "qwen3.5-27b"; that is the id to send."""
    model = get_chat_model("qwen3.5-27b", provider="custom-openai")

    assert model.model_name == "qwen3.5-27b", model.model_name


def test_a_discovered_id_that_collides_with_a_builtin_is_sent_as_listed(endpoint):
    model = get_chat_model("gpt-6.1-terra", provider="custom-openai")

    assert model.model_name == "gpt-6.1-terra", model.model_name


def test_an_unknown_id_is_sent_verbatim(endpoint):
    model = get_chat_model("totally-unknown-model", provider="custom-openai")

    assert model.model_name == "totally-unknown-model", model.model_name


def test_a_name_without_a_provider_still_resolves_through_the_registry():
    """Without a provider the registry fallback is what names the model."""
    from EvoScientist.llm.models import MODELS

    assert MODELS["qwen3.5-27b"][0] == "qwen/qwen3.5-27b"


def test_a_discovered_id_is_offered_once_even_when_listed_twice():
    body = {"data": [{"id": "dup"}, {"id": "dup"}, {"id": "other"}]}

    assert _names_from_body(body) == ["dup", "other"]


def test_a_blank_discovered_id_is_not_offered():
    body = {"data": [{"id": ""}, {"id": "  "}, {"id": "real-model"}]}

    assert _names_from_body(body) == ["real-model"]


def test_a_discovered_id_is_stripped():
    body = {"data": [{"id": "  spaced-model  "}]}

    assert _names_from_body(body) == ["spaced-model"]


def test_a_non_model_body_lists_nothing():
    assert _names_from_body({"error": "not a model list"}) == []
    assert _names_from_body({"data": "nope"}) == []
