"""Regression tests for the review findings on #713.

The picker offers an id the endpoint listed, but ``get_chat_model`` used to
rewrite that id into another provider's model id when the names collided, so
the request carried an id the endpoint does not serve. A discovered id was
also offered twice when the endpoint listed it twice, and a blank id was
offered at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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


def _picker_stub():
    """A questionary stub whose picker returns the first choice."""

    class _Answer:
        def execute(self):
            return ""

        def ask(self):
            return ""

    class _Choice:
        @staticmethod
        def select(*args, **kwargs):
            return _Answer()

    class _Text:
        @staticmethod
        def text(*args, **kwargs):
            return _Answer()

    return {"select": _Choice.select, "text": _Text.text}


def test_a_custom_anthropic_id_keeps_the_globals_mapping(endpoint):
    """The other providers keep the fallback the registry relies on.

    Only a CUSTOM endpoint serves the ids it was asked for; the other
    providers' short names are their own registry entries, so rewriting one
    into another provider's id is what their configuration relies on.
    """
    from EvoScientist.llm.models import MODELS

    assert MODELS["qwen3.5-27b"][0] == "qwen/qwen3.5-27b"


def test_a_provider_without_a_name_keeps_the_registry_mapping(endpoint):
    """A provider-less name still resolves through the registry."""
    from EvoScientist.llm.models import MODELS

    assert MODELS["qwen3.5-27b"][1] == "atlascloud"


def test_a_discovered_listing_is_capped():
    """A huge listing would fill the picker with one row per model."""
    body = {"data": [{"id": f"model-{i}"} for i in range(5000)]}

    names = _names_from_body(body)

    assert len(names) <= 500
    assert names[0] == "model-0"


def test_the_probe_uses_the_url_and_the_key_from_the_same_source(monkeypatch):
    """A key that belongs to one endpoint is never sent to another.

    The runtime gives the environment precedence over the saved config. The
    probe must use the same source for BOTH the URL and the key, or it would
    send the environment key to a stale saved URL.
    """
    from EvoScientist.config.onboard import steps as steps_module

    seen: dict[str, str] = {}

    def fake_list(base_url, api_key, **kwargs):
        seen["base_url"] = base_url or ""
        seen["api_key"] = api_key or ""
        return []

    monkeypatch.setattr(steps_module, "list_custom_openai_models", fake_list)
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "http://env-endpoint/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "env-key")
    config = SimpleNamespace(
        model="",
        custom_openai_base_url="http://saved-endpoint/v1",
        custom_openai_api_key="saved-key",
    )

    with patch(
        "EvoScientist.config.onboard.steps.questionary",
        **_picker_stub(),
    ):
        steps_module._step_model(config, "custom-openai")

    assert seen["base_url"] == "http://env-endpoint/v1", seen
    assert seen["api_key"] == "env-key", seen
