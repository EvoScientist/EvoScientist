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


def _probe_seen(monkeypatch, config, env_url=None, env_key=None):
    """Run the onboarding probe and report the URL and key it used."""
    from EvoScientist.config.onboard import steps as steps_module

    seen: dict[str, str] = {}

    def fake_list(base_url, api_key, **kwargs):
        seen["base_url"] = base_url or ""
        seen["api_key"] = api_key or ""
        return []

    monkeypatch.setattr(steps_module, "list_custom_openai_models", fake_list)
    if env_url is None:
        monkeypatch.delenv("CUSTOM_OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", env_url)
    if env_key is None:
        monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", env_key)

    with patch(
        "EvoScientist.config.onboard.steps.questionary",
        **_picker_stub(),
    ):
        steps_module._step_model(config, "custom-openai")
    return seen


def _config(url="http://saved-endpoint/v1", key="saved-key"):
    return SimpleNamespace(
        model="",
        custom_openai_base_url=url,
        custom_openai_api_key=key,
    )


def test_an_environment_url_is_never_probed_with_the_saved_key(monkeypatch):
    """A key is never sent to a URL it was not configured for."""
    seen = _probe_seen(
        monkeypatch,
        _config(),
        env_url="http://env-endpoint/v1",
        env_key=None,
    )

    assert seen["base_url"] == "http://env-endpoint/v1", seen
    assert seen["api_key"] == "", seen


def test_an_environment_key_is_probed_with_the_saved_url(monkeypatch):
    """The saved URL keeps the environment key when it has no key of its own.

    This is the onboarding case the upstream contract describes: the wizard
    has written the URL but may not have written the key back, so the live
    credential comes from the environment.
    """
    seen = _probe_seen(
        monkeypatch,
        _config(url="http://saved-endpoint/v1", key=""),
        env_url=None,
        env_key="env-key",
    )

    assert seen["base_url"] == "http://saved-endpoint/v1", seen
    assert seen["api_key"] == "env-key", seen


def test_both_from_the_environment_are_paired(monkeypatch):
    seen = _probe_seen(
        monkeypatch,
        _config(),
        env_url="http://env-endpoint/v1",
        env_key="env-key",
    )

    assert seen["base_url"] == "http://env-endpoint/v1", seen
    assert seen["api_key"] == "env-key", seen


def test_both_from_the_saved_config_are_paired(monkeypatch):
    seen = _probe_seen(monkeypatch, _config(), env_url=None, env_key=None)

    assert seen["base_url"] == "http://saved-endpoint/v1", seen
    assert seen["api_key"] == "saved-key", seen


@pytest.mark.asyncio
async def test_the_probe_has_a_total_deadline():
    """A slow endpoint cannot keep the picker from opening.

    httpx's timeout is an inactivity timeout: a server that trickles a byte
    at a time keeps it alive. The probe needs a total deadline as well.
    """
    import asyncio
    import time

    from EvoScientist.llm import custom_openai_discovery as discovery

    class _SlowClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError("the deadline did not fire")

    import httpx

    original = httpx.AsyncClient
    httpx.AsyncClient = _SlowClient
    try:
        started = time.monotonic()
        names = await discovery.discover_custom_openai_models(
            "http://slow-endpoint/v1", timeout=0.3
        )
        elapsed = time.monotonic() - started
    finally:
        httpx.AsyncClient = original

    assert names == []
    assert elapsed < 2.0, elapsed


def test_the_saved_key_is_used_when_the_environment_names_the_same_endpoint(
    monkeypatch,
):
    """The same endpoint named twice keeps the key that belongs to it.

    The user sets ``CUSTOM_OPENAI_BASE_URL``, keeps that URL in onboarding
    and enters its key: the wizard stores the key in the config without
    exporting it. The saved key then belongs to the environment's URL, so it
    must be used.
    """
    seen = _probe_seen(
        monkeypatch,
        _config(url="http://same-endpoint/v1", key="saved-key"),
        env_url="http://same-endpoint/v1",
        env_key=None,
    )

    assert seen["base_url"] == "http://same-endpoint/v1", seen
    assert seen["api_key"] == "saved-key", seen


def test_the_sync_probe_has_a_total_deadline(monkeypatch):
    """A trickling endpoint cannot keep the wizard waiting.

    HTTPX's timeout is an inactivity timeout, so an endpoint that sends a
    chunk every few seconds without finishing would never let the call
    return. The probe needs a total deadline as well.
    """
    import time

    import httpx

    from EvoScientist.llm import custom_openai_discovery as discovery

    def slow_get(*args, **kwargs):
        time.sleep(5)
        raise AssertionError("the deadline did not fire")

    monkeypatch.setattr(httpx, "get", slow_get)

    started = time.monotonic()
    names = discovery.list_custom_openai_models("http://slow-endpoint/v1", timeout=0.3)
    elapsed = time.monotonic() - started

    assert names == []
    assert elapsed < 2.0, elapsed


def test_the_same_endpoint_written_two_ways_keeps_the_saved_key(monkeypatch):
    """A trailing slash does not make the endpoint a different one.

    "http://x/v1" and "http://x/v1/" name the same endpoint, so the saved
    key belongs to it and is used. Treating them as different would withhold
    the key and probe keyless.
    """
    seen = _probe_seen(
        monkeypatch,
        _config(url="http://same-endpoint/v1/", key="saved-key"),
        env_url="http://same-endpoint/v1",
        env_key=None,
    )

    assert seen["base_url"] == "http://same-endpoint/v1/", seen
    assert seen["api_key"] == "saved-key", seen
