"""Explicit refusals must fail local turns or become task notices, including v3 assembled messages."""

from __future__ import annotations

import io
import json
from functools import partial
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRefusalMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import ModelRefusalError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph_api.serde import json_dumpb
from typer.testing import CliRunner

from EvoScientist.cli import commands
from EvoScientist.cli._app import app
from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import (
    _get_default_middleware,
    _inject_subagent_middleware,
)
from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.gateway.types import GraphTarget, RunRequest
from EvoScientist.llm.errors import ProviderStreamError
from EvoScientist.middleware import error_normalization as errors
from EvoScientist.middleware import model_fallback as fallback
from EvoScientist.stream.json_sink import stream_json


def _event(kind, **data):
    return {"type": kind, **data}


@pytest.fixture(autouse=True)
def _clean_fallback_chain():
    fallback.clear_fallbacks()
    yield
    fallback._reset_chain_initialization()


_STOP_DETAILS = {"category": "safety", "explanation": "Cannot continue"}


def _message(reason="refusal", *, content=None):
    return AIMessage(
        content=content
        if content is not None
        else [{"type": "text", "text": "Partial answer"}],
        response_metadata={
            "model_provider": "anthropic",
            "output_version": "v1",
            "stop_reason": reason,
            "stop_details": _STOP_DETAILS,
        },
    )


def _model(model="claude-sonnet-4-6", **kwargs):
    return ChatAnthropic(model=model, api_key="test", max_retries=0, **kwargs)


async def _call(middleware, handler, use_async, request=None):
    async def ahandler(request):
        return handler(request)

    invoke = ahandler if use_async else handler
    for m in reversed(middleware):
        wrap = m.awrap_model_call if use_async else m.wrap_model_call
        invoke = partial(wrap, handler=invoke)
    request = request if request is not None else SimpleNamespace(model=_model())
    return await invoke(request) if use_async else invoke(request)


@pytest.mark.parametrize("use_async", [False, True])
async def test_refusal_detection_passes_normal_response(use_async):
    message = _message("end_turn")
    response = ModelResponse(result=[message])
    middleware = [errors.ModelRefusalDetectionMiddleware()]
    result = await _call(middleware, lambda _: response, use_async)
    assert result is response
    assert result.result[0] is message
    assert message.content == [{"type": "text", "text": "Partial answer"}]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("case", ["empty", "partial"])
async def test_refusal_detection_raises_on_metadata_only_refusal(use_async, case):
    text = {"type": "text", "text": "Partial answer"}
    content = {"empty": [], "partial": [text]}[case]
    message = _message(content=content)
    response = ModelResponse(result=[message])
    middleware = [errors.ModelRefusalDetectionMiddleware()]
    with pytest.raises(ModelRefusalError, match=r"safety.*Cannot continue") as caught:
        await _call(middleware, lambda _: response, use_async)
    raw = caught.value.raw_message
    assert raw.response_metadata == message.response_metadata
    assert sum(b["type"] == "refusal" for b in raw.content_blocks) == 1
    assert message.content == content


@pytest.mark.parametrize("use_async", [False, True])
async def test_existing_refusal_error_keeps_its_type_and_single_block(use_async):
    block = {"type": "refusal", "refusal": "Already refused"}
    message = _message(content=[block])
    response = ModelResponse(result=[message])
    middleware = [
        errors.ErrorNormalizationMiddleware(pass_through_refusals=True),
        ModelRefusalMiddleware(),
    ]
    with pytest.raises(ModelRefusalError, match="Already refused") as caught:
        await _call(middleware, lambda _: response, use_async)
    assert caught.value.raw_message is message
    assert message.content == [block]


def _provider_message(reason, *, output_tokens=0):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [],
        "stop_reason": reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": output_tokens},
    }


def _sse(reason, *, text="Partial answer"):
    delta = {
        "stop_reason": reason,
        "stop_sequence": None,
        "stop_details": _STOP_DETAILS,
    }
    events = [
        _event("message_start", message=_provider_message(None)),
        _event(
            "content_block_start", index=0, content_block={"type": "text", "text": ""}
        ),
        _event(
            "content_block_delta", index=0, delta={"type": "text_delta", "text": text}
        ),
        _event("content_block_stop", index=0),
        _event("message_delta", delta=delta, usage={"output_tokens": 3}),
        _event("message_stop"),
    ]
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


@pytest.fixture
async def anthropic_model():
    clients = []

    def build(reason, *, text="Partial answer"):
        def respond(request):
            assert json.loads(request.content)["stream"] is True
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(reason, text=text),
            )

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        clients.append(client)
        model = _model(max_tokens=1024)
        model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
            api_key="test", max_retries=0, http_client=client
        )
        return model

    yield build
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize("reason", ["refusal", "end_turn"])
@pytest.mark.parametrize("with_fallback", [False, True])
async def test_local_gateway_fails_turn_on_streamed_refusal(
    anthropic_model, reason, with_fallback, workspace, monkeypatch
):
    model = anthropic_model(reason)
    fallback_model = anthropic_model("end_turn", text="Fallback answer")
    monkeypatch.setattr(
        "EvoScientist.llm.models.get_chat_model", lambda **_: fallback_model
    )
    if with_fallback:
        fallback.add_fallback("fallback", "anthropic")
    graph = _main_graph(workspace, model)
    request = RunRequest(
        message="hello", thread_id="refusal-test", target=GraphTarget(local_graph=graph)
    )
    output = io.StringIO()
    failed = reason == "refusal" and not with_fallback
    if failed:
        with pytest.raises(ProviderStreamError, match="Model refused") as caught:
            await stream_json(LocalGraphGateway(), request, out=output)
        metadata = caught.value.__cause__.raw_message.response_metadata
        assert metadata["output_version"] == "v1"
    else:
        answer = await stream_json(LocalGraphGateway(), request, out=output)
        expected = "Fallback answer" if reason == "refusal" else "Partial answer"
        assert answer == expected
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(e["type"] == "text" and "Partial answer" in e["content"] for e in events)
    discarded = any(e["type"] == "model_output_discarded" for e in events)
    assert discarded == (reason == "refusal" and with_fallback)
    assert events[-1]["type"] == ("error" if failed else "done")
    if failed:
        assert not any(e["type"] == "done" for e in events)


@pytest.mark.parametrize("reason", ["refusal", "end_turn"])
@pytest.mark.parametrize("output_format", ["text", "stream-json"])
def test_single_shot_cli_reports_refusal(
    monkeypatch, tmp_path, anthropic_model, reason, output_format, workspace
):
    model = anthropic_model(reason)
    graph = _main_graph(workspace, model)
    config = EvoScientistConfig(ui_backend="cli", enable_async_subagents=False)
    monkeypatch.setattr("EvoScientist.config.get_effective_config", lambda _: config)
    monkeypatch.setattr("EvoScientist.config.apply_config_to_env", lambda _: None)
    monkeypatch.setattr(commands, "_ensure_async_subagent_server", lambda *a, **k: None)
    monkeypatch.setattr(commands, "_load_agent", lambda **kwargs: graph)
    monkeypatch.setattr(commands, "_wait_for_memory_workers", lambda *a, **k: None)
    result = CliRunner().invoke(
        app,
        ["-p", "hello", "--workdir", str(tmp_path), "--output-format", output_format],
    )
    assert result.exit_code == (1 if reason == "refusal" else 0), result.output
    if output_format == "stream-json":
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert events[-1]["type"] == ("error" if reason == "refusal" else "done")


def _task_graph(workspace, child):
    class Parent(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    task = {
        "name": "task",
        "id": "task-1",
        "args": {"subagent_type": "researcher", "description": "hello"},
    }
    parent = Parent(
        responses=[
            AIMessage(content="", tool_calls=[task]),
            AIMessage(content="Continue without the researcher"),
        ]
    )
    subagent = {
        "name": "researcher",
        "description": "Research",
        "system_prompt": "Research",
        "model": child,
        "middleware": _refusal_stack(workspace, child, "sync-subagent"),
    }
    return create_deep_agent(model=parent, subagents=[subagent])


def test_sync_task_reports_before_output_refusal_as_a_notice(workspace):
    def respond(request):
        payload = _provider_message("refusal", output_tokens=1)
        payload["stop_details"] = _STOP_DETAILS
        return httpx2.Response(200, json=payload)

    with httpx2.Client(transport=httpx2.MockTransport(respond)) as client:
        child = _model(max_tokens=1024)
        child.__dict__["_client"] = anthropic.Anthropic(
            api_key="test", max_retries=0, http_client=client
        )
        graph = _task_graph(workspace, child)
        result = graph.invoke({"messages": [{"role": "user", "content": "hello"}]})

    messages = result["messages"]
    (task,) = [m for m in messages if isinstance(m, ToolMessage)]
    assert "The subagent stopped because a model refused the request" in task.content
    assert "Cannot continue" in task.content
    assert messages[-1].content == "Continue without the researcher"


async def test_sync_task_reports_streamed_refusal_as_a_notice(
    anthropic_model, workspace
):
    graph = _task_graph(workspace, anthropic_model("refusal"))
    request = RunRequest(
        message="hello", thread_id="task-refusal", target=GraphTarget(local_graph=graph)
    )
    output = io.StringIO()
    answer = await stream_json(LocalGraphGateway(), request, out=output)

    assert answer == "Continue without the researcher"
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert {"subagent": "researcher", "content": "Partial answer"}.items() <= next(
        e for e in events if e["type"] == "subagent_text"
    ).items()
    (task,) = [e for e in events if e["type"] == "tool_result"]
    assert "The subagent stopped because a model refused the request" in task["content"]
    assert "Cannot continue" in task["content"]
    assert events[-1]["type"] == "done"


def test_main_refusal_keeps_its_webui_message():
    refusal = ModelRefusalError.from_refusal(
        {"type": "refusal", "refusal": "Cannot continue"}, raw_message=_message()
    )
    normalized = errors._normalize(SimpleNamespace(model=_model()), refusal)
    assert normalized is not None
    payload = json.loads(json_dumpb(normalized))
    assert "Model refused" in payload["message"]
    assert "Cannot continue" in payload["message"]
    assert payload["provider"] == "anthropic"


def _main_graph(workspace, model):
    return create_deep_agent(
        model=model, middleware=_refusal_stack(workspace, model, "main"), subagents=[]
    )


def _refusal_stack(workspace, model, kind):
    cfg = EvoScientistConfig()
    if kind == "sync-subagent":
        subs = [{"name": "researcher"}]
        _inject_subagent_middleware(
            subs, workspace=workspace, cfg=cfg, chat_model=model
        )
        middleware = subs[0]["middleware"]
    else:
        middleware = _get_default_middleware(
            workspace=workspace,
            cfg=cfg,
            chat_model=model,
            for_async_subagent=kind == "async-subagent",
        )
    # Exercise the factory's error-handling order without unrelated tools or memory.
    keep = {
        "error_normalization",
        "model_fallback",
        "model_call_boundary",
        "model_refusal_detection",
    }
    return [m for m in middleware if m.name in keep]


@pytest.mark.parametrize("kind", ["main", "async-subagent"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "fallback_reasons",
    [[], ["end_turn"], ["refusal", "end_turn"], ["refusal"]],
    ids=["no-fallback", "fallback-answers", "second-fallback-answers", "all-refuse"],
)
async def test_refused_primary_uses_configured_fallbacks(
    monkeypatch, workspace, kind, use_async, fallback_reasons
):
    primary = _model()
    fallbacks = [
        _model("fallback", base_url="https://api.minimaxi.com/anthropic")
        for _ in fallback_reasons
    ]
    reasons = {id(primary): "refusal"}
    reasons.update(zip(map(id, fallbacks), fallback_reasons, strict=True))
    models = {f"fallback-{i}": model for i, model in enumerate(fallbacks)}
    monkeypatch.setattr(
        "EvoScientist.llm.models.get_chat_model", lambda model, provider: models[model]
    )
    fallback.set_fallback_chain([(name, "anthropic") for name in models])
    middleware = _refusal_stack(workspace, primary, kind)
    called = []

    def answer(request):
        called.append(request.model)
        reason = reasons[id(request.model)]
        content = "Fallback answer" if reason == "end_turn" else None
        return ModelResponse(result=[_message(reason, content=content)])

    request = ModelRequest(model=primary, messages=[], tools=[], state={})
    run = partial(_call, middleware, answer, use_async, request)

    if "end_turn" in fallback_reasons:
        response = await run()
        assert response.result[-1].response_metadata["stop_reason"] == "end_turn"
        assert response.result[-1].content == "Fallback answer"
    else:
        with pytest.raises(ProviderStreamError, match="Model refused") as caught:
            await run()
        assert caught.value.provider == ("minimax" if fallbacks else "anthropic")
    assert called == [primary, *fallbacks]
