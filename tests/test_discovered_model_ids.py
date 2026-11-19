"""Discovered model ids: how they are listed and sent, and the onboarding probe."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from EvoScientist.llm.custom_openai_discovery import _names_from_body, _picker_names
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


def test_a_name_without_a_provider_still_resolves_through_the_registry(monkeypatch):
    """Without a provider the registry fallback is what names the model."""
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "sk-test")

    model = get_chat_model("qwen3.5-27b")

    assert model.model_name == "qwen/qwen3.5-27b", model.model_name


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
    assert _names_from_body({"error": "not a model list"}) is None
    assert _names_from_body({"data": "nope"}) is None


def test_a_custom_anthropic_id_is_sent_as_written(monkeypatch):
    """A CUSTOM endpoint serves the ids it was asked for.

    Only a CUSTOM endpoint serves the ids it was asked for; the other
    providers' short names are their own registry entries, so rewriting one
    into another provider's id is what their configuration relies on.
    """
    monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("CUSTOM_ANTHROPIC_API_KEY", "sk-test")

    model = get_chat_model("qwen3.5-27b", provider="custom-anthropic")

    assert model.model == "qwen3.5-27b", model.model


def test_another_provider_keeps_the_registry_mapping(monkeypatch):
    """The other providers keep the mapping their configuration relies on."""
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")

    model = get_chat_model("qwen3.5-27b", provider="siliconflow")

    assert model.model_name == "qwen/qwen3.5-27b", model.model_name


def test_a_discovered_listing_is_capped():
    """A huge listing would fill the picker with one row per model."""
    body = {"data": [{"id": f"model-{i}"} for i in range(5000)]}

    names = _picker_names(body)

    assert names == [f"model-{i}" for i in range(500)]


def _capture_onboarding_probe_credentials(
    monkeypatch, config, env_url=None, env_key=None
):
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

    with patch.object(steps_module, "questionary") as picker:
        picker.select.return_value.ask.return_value = "gpt-6-luna"
        steps_module._step_model(config, "custom-openai")
    return seen


def _saved_endpoint_config(url="http://saved-endpoint/v1", key="saved-key"):
    return SimpleNamespace(
        model="",
        custom_openai_base_url=url,
        custom_openai_api_key=key,
    )


def test_an_environment_url_is_probed_with_the_saved_key(monkeypatch):
    """Each setting takes its environment variable over its saved value.

    The environment names the URL and the config holds the key, so the
    probe sends both -- the same pair the model calls will send.
    """
    seen = _capture_onboarding_probe_credentials(
        monkeypatch,
        _saved_endpoint_config(),
        env_url="http://env-endpoint/v1",
        env_key=None,
    )

    assert seen["base_url"] == "http://env-endpoint/v1", seen
    assert seen["api_key"] == "saved-key", seen


def test_an_environment_key_is_probed_with_the_saved_url(monkeypatch):
    """The probe prefers the environment key even when the URL comes from config."""
    seen = _capture_onboarding_probe_credentials(
        monkeypatch,
        _saved_endpoint_config(),
        env_url=None,
        env_key="env-key",
    )

    assert seen["base_url"] == "http://saved-endpoint/v1", seen
    assert seen["api_key"] == "env-key", seen


def test_the_model_calls_keep_the_saved_key_when_the_environment_names_the_url(
    monkeypatch,
):
    """The key every model call sends must not change because of the probe.

    Each setting takes its environment variable over its saved value, so a
    container that reaches the host at another address still authenticates
    with the key the user saved. A probe that paired the two would answer
    401 on the very calls it was meant to describe.
    """
    from EvoScientist.config.settings import apply_config_to_env, get_effective_config
    from EvoScientist.llm.models import get_chat_model

    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "EvoScientist.config.settings.load_config",
        lambda *args, **kwargs: _saved_config(
            custom_openai_base_url="http://localhost:8000/v1",
            custom_openai_api_key="sk-saved",
        ),
    )

    apply_config_to_env(get_effective_config())
    model = get_chat_model("fake-agent", "custom-openai")

    assert model.openai_api_key.get_secret_value() == "sk-saved"


def test_probe_prefers_both_environment_values_over_saved_values(monkeypatch):
    seen = _capture_onboarding_probe_credentials(
        monkeypatch,
        _saved_endpoint_config(),
        env_url="http://env-endpoint/v1",
        env_key="env-key",
    )

    assert seen["base_url"] == "http://env-endpoint/v1", seen
    assert seen["api_key"] == "env-key", seen


def test_probe_uses_saved_url_and_key_when_environment_values_are_unset(monkeypatch):
    seen = _capture_onboarding_probe_credentials(
        monkeypatch, _saved_endpoint_config(), env_url=None, env_key=None
    )

    assert seen["base_url"] == "http://saved-endpoint/v1", seen
    assert seen["api_key"] == "saved-key", seen


@pytest.mark.asyncio
async def test_the_probe_has_a_total_deadline(monkeypatch):
    """A slow endpoint cannot keep the picker from opening.

    httpx's timeout is an inactivity timeout: a server that trickles a byte
    at a time keeps it alive. The probe needs a total deadline as well.
    """
    import asyncio

    from EvoScientist.llm import custom_openai_discovery as discovery

    class _TricklingBody(httpx.AsyncByteStream):
        started = False
        closed = False

        async def __aiter__(self):
            self.started = True
            for _ in range(20):
                await asyncio.sleep(0.05)
                yield b" "
            yield b'{"data": [{"id": "late"}]}'

        async def aclose(self):
            self.closed = True

    body = _TricklingBody()
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs)
    )
    names = await discovery.discover_custom_openai_models(
        "http://slow-endpoint/v1", timeout=0.3
    )

    assert body.started
    assert body.closed
    assert names == []


def test_the_environment_key_wins_over_an_expired_saved_key(monkeypatch):
    """The runtime gives the environment key precedence, and so does the probe.

    The saved key can have expired while the environment key is live. The
    probe must use the same effective credentials the runtime will use, or
    discovery fails against an endpoint that would have answered.
    """
    seen = _capture_onboarding_probe_credentials(
        monkeypatch,
        _saved_endpoint_config(url="http://same-endpoint/v1", key="expired-key"),
        env_url="http://same-endpoint/v1",
        env_key="live-key",
    )

    assert seen["base_url"] == "http://same-endpoint/v1", seen
    assert seen["api_key"] == "live-key", seen


def test_the_timed_out_request_is_aborted(monkeypatch):
    """A timed-out probe must not leave the request running.

    The deadline cancels the request, so the worker ends with it instead of
    keeping its connection and buffering into it. A probe that only stopped
    waiting would leave the request running.
    """
    import asyncio
    import time

    from EvoScientist.llm import custom_openai_discovery as discovery

    cancelled: list[bool] = []

    async def never_answers(request):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        raise AssertionError("the deadline did not fire")

    client = httpx.AsyncClient
    transport = httpx.MockTransport(never_answers)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs)
    )
    started = time.monotonic()
    names = discovery.list_custom_openai_models("http://slow-endpoint/v1", timeout=0.3)
    elapsed = time.monotonic() - started

    assert names == []
    assert elapsed < 2.0, elapsed
    assert cancelled, "the request was not cancelled, so it kept running"


def _saved_config(**fields: str):
    """A config object carrying only the fields a test sets."""
    from EvoScientist.config.settings import EvoScientistConfig

    return EvoScientistConfig(**fields)
