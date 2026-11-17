"""Independent review of model propagation and actual model resolution."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from EvoScientist.gateway.types import resolve_per_run_config
from EvoScientist.llm import models, patches
from EvoScientist.llm.registry import _MODEL_ENTRIES, MODELS, UnknownModelError
from EvoScientist.middleware.configurable_model import ConfigurableModelMiddleware
from EvoScientist.middleware.expert_async_subagent import (
    EvoAsyncSubAgentMiddleware,
    _caller_model_scope,
)


def _runtime(configurable):
    return SimpleNamespace(config={"configurable": configurable})


def _merge_from(configurable, kwargs=None):
    with _caller_model_scope(_runtime(configurable)):
        return patches._merge_runs_config_kwargs(kwargs or {})["config"]["configurable"]


@pytest.mark.parametrize(
    ("caller", "server", "expected"),
    [
        (
            {"model": "client", "model_provider": "client-provider"},
            {"model": "server", "model_provider": "server-provider"},
            {"model": "client", "model_provider": "client-provider"},
        ),
        (
            {},
            {"model": "server", "model_provider": "server-provider"},
            {"model": "server", "model_provider": "server-provider"},
        ),
        ({"model": "client"}, {}, {"model": "client", "model_provider": None}),
        (
            {"model_provider": "client-provider"},
            {"model": "server", "model_provider": "server-provider"},
            {"model": "server", "model_provider": "server-provider"},
        ),
        (
            {"model": "client", "model_provider": None},
            {"model": "server", "model_provider": "server-provider"},
            {"model": "client", "model_provider": None},
        ),
    ],
)
def test_runtime_caller_combinations(monkeypatch, caller, server, expected):
    monkeypatch.setattr(patches, "_read_cfg_configurable", lambda: server)
    assert _merge_from(caller) == expected


def test_explicit_request_provider_is_removed_for_providerless_runtime(monkeypatch):
    monkeypatch.setattr(patches, "_read_cfg_configurable", dict)
    kwargs = {"config": {"configurable": {"model_provider": "explicit", "x": 1}}}
    assert _merge_from({"model": "client"}, kwargs) == {
        "model": "client",
        "x": 1,
        "model_provider": None,
    }
    assert kwargs["config"]["configurable"]["model_provider"] == "explicit"


def test_provider_only_without_server_default_is_a_noop(monkeypatch):
    monkeypatch.setattr(patches, "_read_cfg_configurable", dict)
    kwargs = {"thread_id": "child"}
    with _caller_model_scope(_runtime({"model_provider": "openai"})):
        assert patches._merge_runs_config_kwargs(kwargs) is kwargs
    assert _invoke_middleware(monkeypatch, {"model_provider": "openai"}) == "compiled"


@pytest.fixture
def sdk_route(monkeypatch):
    """Keep real registry/route selection but replace the network SDK object."""
    routes = []

    def construct(*, model, model_provider, **kwargs):
        route = (model, model_provider)
        routes.append(route)
        return SimpleNamespace(route=route)

    monkeypatch.setattr(models, "init_chat_model", construct)
    for name in (
        "_patch_openrouter_structured_output",
        "_enable_openrouter_429_retry",
        "_patch_anthropic_strip_foreign_reasoning",
        "_patch_anthropic_structured_output",
        "_patch_anthropic_thinking_binding",
        "_fill_claude_structured_output",
        "_fit_claude_effort",
        "apply_known_context_window",
    ):
        monkeypatch.setattr(models, name, lambda *args, **kwargs: None)
    return routes


def _invoke_middleware(monkeypatch, configurable, base="compiled"):
    import langgraph.config

    monkeypatch.setattr(
        langgraph.config, "get_config", lambda: {"configurable": configurable}
    )
    request = MagicMock()
    request.model = base
    request.override.side_effect = lambda **kw: SimpleNamespace(model=kw["model"])
    return ConfigurableModelMiddleware().wrap_model_call(request, lambda req: req.model)


@pytest.mark.parametrize(
    ("model", "server_provider", "expected_route"),
    [
        ("claude-sonnet-5-5", "openrouter", ("claude-sonnet-5-5", "anthropic")),
        ("gpt-6.1-sol", "anthropic", ("openai/gpt-6.1-sol", "openrouter")),
        ("openai/gpt-4o", "openrouter", ("openai/gpt-4o", "anthropic")),
        ("openai", "openrouter", ("openai", "anthropic")),
    ],
)
def test_real_registry_route_matches_main_and_child(
    monkeypatch, sdk_route, model, server_provider, expected_route
):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": server_provider},
    )
    caller = {"model": model}
    child = _merge_from(caller)
    # The provider is present-and-None, not absent: a missing key lets the
    # child assistant's SAVED provider back in when LangGraph merges it.
    assert child == {**caller, "model_provider": None}
    assert _invoke_middleware(monkeypatch, caller).route == expected_route
    assert _invoke_middleware(monkeypatch, child).route == expected_route
    assert sdk_route == [expected_route, expected_route]


def test_router_unknown_really_raises_before_middleware_fallback(
    monkeypatch, sdk_route
):
    caller = {"model": "claude-sonnet-5-5", "model_provider": "openrouter"}
    with pytest.raises(UnknownModelError):
        _invoke_middleware(monkeypatch, caller)
    assert not sdk_route


def test_old_inherited_providers_produce_wrong_routes(sdk_route):
    with pytest.raises(UnknownModelError):
        models.get_chat_model("claude-sonnet-5-5", provider="openrouter")
    assert models.get_chat_model("gpt-6.1-sol", provider="anthropic").route == (
        "openai/gpt-6.1-sol",
        "anthropic",
    )
    assert models.get_chat_model("gpt-6.1-sol").route == (
        "openai/gpt-6.1-sol",
        "openrouter",
    )


def test_model_named_like_provider_and_duplicate_short_names(monkeypatch, sdk_route):
    provider_names = {provider for _, _, provider in _MODEL_ENTRIES}
    assert "openai" in provider_names
    assert "openai" not in MODELS
    assert len([entry for entry in _MODEL_ENTRIES if entry[0] == "gpt-6.1-sol"]) > 1
    assert models.get_chat_model("openai").route == ("openai", "anthropic")
    assert models.get_chat_model("gpt-6.1-sol").route == (
        "openai/gpt-6.1-sol",
        "openrouter",
    )
    monkeypatch.setitem(models.MODELS, "openai", ("collision-model", "anthropic"))
    assert models.get_chat_model("openai").route == ("collision-model", "anthropic")


def test_unknown_bare_name_is_not_an_unknown_model_error(monkeypatch, sdk_route):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    caller = {"model": "not-a-real-model-xyz"}
    child = _merge_from(caller)
    assert _invoke_middleware(monkeypatch, caller).route == (
        "not-a-real-model-xyz",
        "anthropic",
    )
    assert _invoke_middleware(monkeypatch, child).route == sdk_route[0]


def test_unknown_missing_key_is_swallowed_equally_on_main_and_child(monkeypatch):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )

    def missing_key(**kwargs):
        raise RuntimeError("simulated SDK initialization error")

    monkeypatch.setattr(models, "init_chat_model", missing_key)
    caller = {"model": "not-a-real-model-xyz"}
    child = _merge_from(caller)
    assert _invoke_middleware(monkeypatch, caller) == "compiled"
    assert _invoke_middleware(monkeypatch, child) == "compiled"


def test_unknown_bare_name_with_real_sdk_and_no_credentials(monkeypatch):
    from EvoScientist.middleware.error_normalization import (
        MissingApiKeyError,
        _check_api_key,
    )

    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    caller = {"model": "not-a-real-model-xyz"}
    direct = models.get_chat_model(caller["model"])
    main = _invoke_middleware(monkeypatch, caller)
    child = _invoke_middleware(monkeypatch, _merge_from(caller))
    for selected in (direct, main, child):
        assert type(selected).__name__ == "ChatAnthropic"
        assert selected.model == "not-a-real-model-xyz"
    messages = []
    for selected in (main, child):
        with pytest.raises(MissingApiKeyError) as excinfo:
            _check_api_key(SimpleNamespace(model=selected))
        messages.append(str(excinfo.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("operation", ["start", "update"])
async def test_providerless_real_tool_start_and_update(
    monkeypatch, workspace, is_async, operation
):
    from EvoScientist.middleware import expert_async_subagent as module

    spec = {
        "name": "writing-agent",
        "description": "writer",
        "graph_id": "writing_agent",
    }
    middleware = EvoAsyncSubAgentMiddleware(workspace=workspace, async_subagents=[spec])
    client = MagicMock()
    if is_async:
        client.threads.create = AsyncMock(return_value={"thread_id": "task-abc"})
        client.runs.create = AsyncMock(return_value={"run_id": "run-xyz"})
        monkeypatch.setattr(module._ClientCache, "get_async", lambda *args: client)
    else:
        client.threads.create.return_value = {"thread_id": "task-abc"}
        client.runs.create.return_value = {"run_id": "run-xyz"}
        monkeypatch.setattr(module._ClientCache, "get_sync", lambda *args: client)
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    tracked = {
        "task_id": "task-abc",
        "agent_name": "writing-agent",
        "thread_id": "task-abc",
        "run_id": "old-run",
        "status": "running",
        "created_at": "2026-05-07T00:00:00Z",
        "last_checked_at": "2026-05-07T00:00:00Z",
        "last_updated_at": "2026-05-07T00:00:00Z",
    }
    runtime = SimpleNamespace(
        tool_call_id="tc1",
        config={"configurable": {"model": "claude-sonnet-5-5"}},
        context=None,
        state={"async_tasks": {"task-abc": tracked}},
    )
    tool = next(t for t in middleware.tools if t.name == f"{operation}_async_task")
    args = (
        {"description": "write", "subagent_type": "writing-agent", "runtime": runtime}
        if operation == "start"
        else {"task_id": "task-abc", "message": "continue", "runtime": runtime}
    )
    result = await tool.coroutine(**args) if is_async else tool.func(**args)
    assert not isinstance(result, str), result
    call = client.runs.create.await_args if is_async else client.runs.create.call_args
    configurable = call.kwargs["config"]["configurable"]
    assert configurable["model"] == "claude-sonnet-5-5"
    # An explicit None, not an absent key: LangGraph merges the child
    # assistant's saved config into the run, and a missing key lets that
    # saved provider back in.
    assert configurable["model_provider"] is None


def test_nested_scopes_propagate_and_restore(monkeypatch):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    with _caller_model_scope(_runtime({"model": "claude-sonnet-5-5"})):
        child = patches._merge_runs_config_kwargs({})["config"]["configurable"]
        with _caller_model_scope(_runtime(child)):
            grandchild = patches._merge_runs_config_kwargs({})["config"]["configurable"]
        assert patches._caller_configurable.get() == {"model": "claude-sonnet-5-5"}
    assert child == grandchild == {"model": "claude-sonnet-5-5", "model_provider": None}
    assert patches._caller_configurable.get() is None


async def test_runtime_continued_thread_does_not_retain_prior_provider(
    monkeypatch, workspace, runtime_store
):
    from langgraph_runtime_inmem.database import connect
    from langgraph_runtime_inmem.ops import Runs

    from EvoScientist.langgraph_dev.assistants import (
        ensure_workspace_assistants,
        workspace_assistant_id,
    )

    await ensure_workspace_assistants(workspace)
    assistant_id = workspace_assistant_id("writing-agent", workspace)
    thread_id = str(uuid4())

    async def launch(configurable):
        async with connect() as conn:
            runs = await Runs.put(
                conn,
                assistant_id,
                {"config": {"configurable": configurable}},
                thread_id=thread_id,
                run_id=uuid4(),
                metadata={},
                prevent_insert_if_inflight=False,
                if_not_exists="create",
            )
            return (await anext(runs))["kwargs"]["config"]["configurable"]

    first = await launch({"model": "old-model", "model_provider": "openrouter"})
    assert first["model_provider"] == "openrouter"
    for record in (*runtime_store["threads"], *runtime_store["runs"]):
        record["status"] = "error"
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    child = _merge_from({"model": "claude-sonnet-5-5"})
    second = await launch(child)
    assert second["model"] == "claude-sonnet-5-5"
    assert second["model_provider"] is None


async def test_runtime_assistant_default_does_not_reintroduce_provider(
    monkeypatch, workspace, runtime_store, sdk_route
):
    from langgraph_runtime_inmem.database import connect
    from langgraph_runtime_inmem.ops import Assistants, Runs

    from EvoScientist.langgraph_dev.assistants import (
        ensure_workspace_assistants,
        workspace_assistant_id,
    )

    await ensure_workspace_assistants(workspace)
    assistant_id = workspace_assistant_id("writing-agent", workspace)
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    child = _merge_from({"model": "claude-sonnet-5-5"})
    async with connect() as conn:
        changed = await Assistants.patch(
            conn,
            UUID(assistant_id),
            config={
                "configurable": {
                    "workspace_dir": workspace.key,
                    "model_provider": "openrouter",
                }
            },
        )
        await anext(changed)
    # Workspace repair accepts extra configurable keys, so this is persisted.
    await ensure_workspace_assistants(workspace)
    async with connect() as conn:
        runs = await Runs.put(
            conn,
            assistant_id,
            {"config": {"configurable": child}},
            thread_id=str(uuid4()),
            run_id=uuid4(),
            metadata={},
            prevent_insert_if_inflight=False,
            if_not_exists="create",
        )
        actual = (await anext(runs))["kwargs"]["config"]["configurable"]
    assert actual["model"] == "claude-sonnet-5-5"
    main = _invoke_middleware(monkeypatch, {"model": "claude-sonnet-5-5"})
    assert main.route == ("claude-sonnet-5-5", "anthropic")
    selected = _invoke_middleware(monkeypatch, actual)
    assert selected.route == main.route


async def test_null_provider_clears_runtime_assistant_default(
    monkeypatch, workspace, runtime_store, sdk_route
):
    from langgraph_runtime_inmem.database import connect
    from langgraph_runtime_inmem.ops import Assistants, Runs

    from EvoScientist.langgraph_dev.assistants import (
        ensure_workspace_assistants,
        workspace_assistant_id,
    )

    await ensure_workspace_assistants(workspace)
    assistant_id = workspace_assistant_id("writing-agent", workspace)
    async with connect() as conn:
        changed = await Assistants.patch(
            conn,
            UUID(assistant_id),
            config={
                "configurable": {
                    "workspace_dir": workspace.key,
                    "model_provider": "openrouter",
                }
            },
        )
        await anext(changed)
        runs = await Runs.put(
            conn,
            assistant_id,
            {
                "config": {
                    "configurable": {
                        "model": "claude-sonnet-5-5",
                        "model_provider": None,
                    }
                }
            },
            thread_id=str(uuid4()),
            run_id=uuid4(),
            metadata={},
            prevent_insert_if_inflight=False,
            if_not_exists="create",
        )
        actual = (await anext(runs))["kwargs"]["config"]["configurable"]
    assert actual.get("model_provider") is None
    assert _invoke_middleware(monkeypatch, actual).route == (
        "claude-sonnet-5-5",
        "anthropic",
    )


@pytest.mark.parametrize("is_async", [False, True])
async def test_sdk_serializes_explicit_null_provider(is_async):
    from langgraph_sdk.client import RunsClient, SyncRunsClient

    post = AsyncMock(return_value={}) if is_async else MagicMock(return_value={})
    client = (RunsClient if is_async else SyncRunsClient)(SimpleNamespace(post=post))
    result = client.create(
        thread_id="child",
        assistant_id="writing-agent",
        config={"configurable": {"model": "claude-sonnet-5-5", "model_provider": None}},
    )
    if is_async:
        await result
    payload = json.loads(json.dumps(post.call_args.kwargs["json"]))
    assert "model_provider" in payload["config"]["configurable"]
    assert payload["config"]["configurable"]["model_provider"] is None


def test_concurrent_threads_do_not_mix_models(monkeypatch):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    barrier = threading.Barrier(2, timeout=10)

    def invoke(model):
        with _caller_model_scope(_runtime({"model": model})):
            barrier.wait()
            merged = patches._merge_runs_config_kwargs({})["config"]["configurable"]
            barrier.wait()
        return merged, patches._caller_configurable.get()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke, "claude-sonnet-5-5")
        second = pool.submit(invoke, "gpt-6.1-sol")
        assert first.result() == (
            {"model": "claude-sonnet-5-5", "model_provider": None},
            None,
        )
        assert second.result() == (
            {"model": "gpt-6.1-sol", "model_provider": None},
            None,
        )


async def test_concurrent_async_proxy_calls_do_not_mix_models(monkeypatch):
    monkeypatch.setattr(
        patches,
        "_read_cfg_configurable",
        lambda: {"model": "server", "model_provider": "openrouter"},
    )
    entered = 0
    ready = asyncio.Event()
    create = AsyncMock(side_effect=lambda **kwargs: kwargs["config"]["configurable"])
    real = SimpleNamespace(runs=SimpleNamespace(create=create), threads=object())
    client = patches._ClientProxy(real, is_async=True)

    async def invoke(model):
        nonlocal entered
        with _caller_model_scope(_runtime({"model": model})):
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            await asyncio.sleep(0)
            return await client.runs.create(thread_id=model)

    assert await asyncio.gather(invoke("claude-sonnet-5-5"), invoke("gpt-6.1-sol")) == [
        {"model": "claude-sonnet-5-5", "model_provider": None},
        {"model": "gpt-6.1-sol", "model_provider": None},
    ]


def test_gateway_extra_model_still_inherits_session_provider():
    config = resolve_per_run_config(
        "thread",
        {"model": "claude-sonnet-5-5"},
        per_run_overrides={"model": "server", "model_provider": "openrouter"},
    )
    assert config["configurable"]["model_provider"] == "openrouter"
    assert patches._extract_caller_configurable(config) == {
        "model": "claude-sonnet-5-5",
        "model_provider": "openrouter",
    }
