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
