"""The explicitly composed client proxy forwards live per-run model config."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from EvoScientist.llm import patches


@pytest.mark.parametrize("is_async", [False, True])
async def test_client_cache_injects_live_config_only_into_create(monkeypatch, is_async):
    config = {"model": "first", "model_provider": "provider"}
    monkeypatch.setattr(patches, "_read_cfg_configurable", lambda: config)
    create = AsyncMock() if is_async else MagicMock()
    real = SimpleNamespace(
        runs=SimpleNamespace(create=create, get=MagicMock(), cancel=MagicMock()),
        threads=object(),
    )
    cache = MagicMock()
    cache.get_sync.return_value = cache.get_async.return_value = real
    proxy = patches._ClientCacheProxy(cache)
    client = proxy.get_async("agent") if is_async else proxy.get_sync("agent")
    assert client.threads is real.threads
    assert client.runs.cancel is real.runs.cancel
    assert client.runs is client.runs
    for model in ("first", "second"):
        config["model"] = model
        result = client.runs.create(thread_id="thread", assistant_id="agent")
        if is_async:
            await result
        assert create.call_args.kwargs == {
            "thread_id": "thread",
            "assistant_id": "agent",
            "config": {"configurable": {"model": model, "model_provider": "provider"}},
        }


def test_merge_preserves_config_and_prioritizes_caller_without_mutation(monkeypatch):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "default", "model_provider": "default-provider"},
    )
    original = {"config": {"tags": ["tag"], "configurable": {"extra": 42}}}
    token = patches._caller_configurable.set(
        {"model": "caller", "model_provider": "caller-provider"}
    )
    try:
        merged = patches._merge_runs_config_kwargs(original)
    finally:
        patches._caller_configurable.reset(token)
    assert merged == {
        "config": {
            "tags": ["tag"],
            "configurable": {
                "extra": 42,
                "model": "caller",
                "model_provider": "caller-provider",
            },
        }
    }
    assert original == {"config": {"tags": ["tag"], "configurable": {"extra": 42}}}
    assert (
        patches._merge_runs_config_kwargs({})["config"]["configurable"]["model"]
        == "default"
    )


@pytest.mark.parametrize(
    ("server", "caller_model"),
    [
        # The server runs on a router that does not list the caller's model.
        ({"model": "claude-sonnet-5-5", "model_provider": "openrouter"}, "gpt-6.1-sol"),
        # The server runs on a provider that does not serve the caller's model.
        ({"model": "claude-sonnet-5-5", "model_provider": "anthropic"}, "gpt-6.1-sol"),
    ],
)
def test_merge_keeps_a_provider_less_caller_model_unpaired(
    monkeypatch, server, caller_model
):
    """A model picked WITHOUT a provider must not inherit the server's.

    ``model`` and ``model_provider`` travel as one unit: the server's
    config-default provider does not serve the caller's model, so pairing
    them either raises ``UnknownModelError`` or ships the model to a
    provider that does not serve it (issue #784). The sub-agent resolves
    the name from the registry instead, exactly like the main run does with
    the same provider-less name.
    """
    monkeypatch.setattr(patches, "_read_cfg_configurable", lambda: dict(server))
    token = patches._caller_configurable.set({"model": caller_model})
    try:
        merged = patches._merge_runs_config_kwargs({})
    finally:
        patches._caller_configurable.reset(token)
    configurable = merged["config"]["configurable"]
    assert (configurable.get("model"), configurable.get("model_provider")) == (
        caller_model,
        None,
    )


def test_merge_keeps_the_callers_own_provider_pair(monkeypatch):
    """A caller that brought its own provider keeps that pair."""
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server-model", "model_provider": "server-provider"},
    )
    token = patches._caller_configurable.set(
        {"model": "caller-model", "model_provider": "caller-provider"}
    )
    try:
        merged = patches._merge_runs_config_kwargs({})
    finally:
        patches._caller_configurable.reset(token)
    configurable = merged["config"]["configurable"]
    assert (configurable["model"], configurable["model_provider"]) == (
        "caller-model",
        "caller-provider",
    )


def test_merge_passes_the_server_pair_through_without_a_caller(monkeypatch):
    """No caller override: the server's (model, model_provider) pair stands."""
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server-model", "model_provider": "server-provider"},
    )
    merged = patches._merge_runs_config_kwargs({})
    configurable = merged["config"]["configurable"]
    assert (configurable["model"], configurable["model_provider"]) == (
        "server-model",
        "server-provider",
    )


@pytest.mark.parametrize("is_async", [False, True])
async def test_child_run_on_a_provider_less_model_sends_no_provider(
    monkeypatch, is_async
):
    """On the wire: the sub-agent's run carries the caller's model and no
    provider, so the server resolves it from the registry like the main run."""
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "claude-sonnet-5-5", "model_provider": "openrouter"},
    )
    create = AsyncMock() if is_async else MagicMock()
    real = SimpleNamespace(
        runs=SimpleNamespace(create=create, get=MagicMock(), cancel=MagicMock()),
        threads=object(),
    )
    cache = MagicMock()
    cache.get_sync.return_value = cache.get_async.return_value = real
    proxy = patches._ClientCacheProxy(cache)
    client = proxy.get_async("agent") if is_async else proxy.get_sync("agent")
    token = patches._caller_configurable.set({"model": "gpt-6.1-sol"})
    try:
        result = client.runs.create(thread_id="child", assistant_id="agent")
        if is_async:
            await result
    finally:
        patches._caller_configurable.reset(token)
    assert create.call_args.kwargs["config"]["configurable"] == {"model": "gpt-6.1-sol"}


@pytest.mark.parametrize("config", [None, "invalid", {"configurable": None}])
def test_merge_handles_missing_or_non_mapping_config(monkeypatch, config):
    monkeypatch.setattr(patches, "_read_cfg_configurable", lambda: {"model": "live"})
    assert patches._merge_runs_config_kwargs({"config": config}) == {
        "config": {"configurable": {"model": "live"}},
    }


def test_empty_live_config_leaves_kwargs_unchanged(monkeypatch):
    monkeypatch.setattr(patches, "_read_cfg_configurable", dict)
    kwargs = {"thread_id": "thread"}
    assert patches._merge_runs_config_kwargs(kwargs) is kwargs


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, {}),
        ({"tags": ["x"]}, {}),
        ({"configurable": {"model_provider": "provider"}}, {}),
        ({"configurable": {"model": "live"}}, {"model": "live"}),
        (
            {"configurable": {"model": "live", "model_provider": "provider"}},
            {"model": "live", "model_provider": "provider"},
        ),
    ],
)
def test_extract_caller_config(config, expected):
    assert patches._extract_caller_configurable(config) == expected


def test_child_runs_do_not_inherit_the_approval_keys():
    """A background sub-agent's run gets the caller's model only: neither the
    suppression flag nor a session grant travels to the dev server with it."""
    config = {
        "configurable": {
            "model": "m",
            "model_provider": "p",
            "hitl_approve_all": True,
            "hitl_suppressed": True,
        }
    }
    assert patches._extract_caller_configurable(config) == {
        "model": "m",
        "model_provider": "p",
    }


def test_a_child_run_created_from_a_granted_call_has_no_approval_key(monkeypatch):
    """On the wire: `runs.create` for a background sub-agent started by a call
    that carries the grant gets the caller's model and nothing else."""
    monkeypatch.setattr(patches, "_read_cfg_configurable", dict)
    create = MagicMock()
    real = SimpleNamespace(
        runs=SimpleNamespace(create=create, get=MagicMock(), cancel=MagicMock()),
        threads=object(),
    )
    cache = MagicMock()
    cache.get_sync.return_value = real
    client = patches._ClientCacheProxy(cache).get_sync("agent")
    caller = {
        "configurable": {
            "thread_id": "parent",
            "model": "m",
            "hitl_approve_all": True,
            "hitl_suppressed": False,
        }
    }
    token = patches._caller_configurable.set(
        patches._extract_caller_configurable(caller)
    )
    try:
        client.runs.create(thread_id="child", assistant_id="agent")
    finally:
        patches._caller_configurable.reset(token)
    assert create.call_args.kwargs["config"] == {"configurable": {"model": "m"}}
