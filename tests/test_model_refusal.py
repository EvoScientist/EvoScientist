"""Explicit refusals must fail local turns, including v3 assembled messages."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware.types import ModelResponse
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage

from EvoScientist.middleware.error_normalization import ErrorNormalizationMiddleware


def _message(reason="refusal", *, content=None):
    return AIMessage(
        content=content
        if content is not None
        else [{"type": "text", "text": "Partial answer"}],
        response_metadata={
            "model_provider": "anthropic",
            "output_version": "v1",
            "stop_reason": reason,
            "stop_details": {"category": "safety", "explanation": "Cannot continue"},
        },
    )


def _request():
    from langchain_anthropic import ChatAnthropic

    return SimpleNamespace(
        model=ChatAnthropic(model="claude-sonnet-4-6", api_key="test", max_retries=0)
    )


async def _call(middleware, response, use_async):
    async def handler(_request):
        return response

    if use_async:
        return await middleware.awrap_model_call(_request(), handler)
    return middleware.wrap_model_call(_request(), lambda _: response)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "text", "text": "Partial answer"}],
        [
            {
                "type": "refusal",
                "refusal": "Cannot continue",
                "extras": {"category": "safety"},
            }
        ],
    ],
)
async def test_v1_metadata_refusal_raises_with_details(use_async, content):
    message = _message(content=content)
    response = ModelResponse(result=[message])
    middleware = ErrorNormalizationMiddleware()

    with pytest.raises(ModelRefusalError, match=r"safety.*Cannot continue") as caught:
        await _call(middleware, response, use_async)
    assert caught.value.raw_message.response_metadata == message.response_metadata
    assert (
        sum(b["type"] == "refusal" for b in caught.value.raw_message.content_blocks)
        == 1
    )
    assert message.content == content
    if any(b["type"] == "refusal" for b in content):
        assert caught.value.raw_message is message


@pytest.mark.parametrize("use_async", [False, True])
async def test_normal_v1_response_is_untouched(use_async):
    message = _message("end_turn")
    response = ModelResponse(result=[message])
    middleware = ErrorNormalizationMiddleware()

    result = await _call(middleware, response, use_async)
    assert result is response
    assert result.result[0] is message
    assert message.content == [{"type": "text", "text": "Partial answer"}]


@pytest.mark.parametrize("use_async", [False, True])
async def test_existing_refusal_error_keeps_its_type_and_single_block(use_async):
    from deepagents.middleware.subagents import _refusal_notice_for
    from langchain.agents.middleware import ModelRefusalMiddleware

    block = {"type": "refusal", "refusal": "Already refused"}
    message = _message(content=[block])
    response = ModelResponse(result=[message])
    outer = ErrorNormalizationMiddleware()
    inner = ModelRefusalMiddleware()

    async def invoke():
        async def answer(_request):
            return response

        if use_async:

            async def handler(request):
                return await inner.awrap_model_call(request, answer)

            return await outer.awrap_model_call(_request(), handler)
        return outer.wrap_model_call(
            _request(), lambda req: inner.wrap_model_call(req, lambda _: response)
        )

    with pytest.raises(ModelRefusalError, match="Already refused") as caught:
        await invoke()
    assert caught.value.raw_message is message
    assert message.content == [block]
    assert "The subagent stopped because a model refused" in _refusal_notice_for(
        caught.value
    )


def _sse(reason, *, text="Partial answer"):
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": reason,
                "stop_sequence": None,
                "stop_details": {
                    "category": "safety",
                    "explanation": "Cannot continue",
                },
            },
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


@pytest.fixture
async def anthropic_model():
    import anthropic
    import httpx2
    from langchain_anthropic import ChatAnthropic

    clients = []

    def build(reason):
        def respond(request):
            assert json.loads(request.content)["stream"] is True
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(reason)
            )

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        clients.append(client)
        model = ChatAnthropic(
            model="claude-sonnet-4-6", api_key="test", max_tokens=1024, max_retries=0
        )
        model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
            api_key="test", max_retries=0, http_client=client
        )
        return model

    yield build
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize("reason", ["refusal", "end_turn"])
async def test_real_anthropic_stream_through_local_gateway(anthropic_model, reason):
    from EvoScientist.gateway.local import LocalGraphGateway
    from EvoScientist.gateway.types import GraphTarget, RunRequest
    from EvoScientist.stream.json_sink import stream_json

    graph = create_deep_agent(
        model=anthropic_model(reason),
        middleware=[ErrorNormalizationMiddleware()],
        subagents=[],
    )
    request = RunRequest(
        message="hello", thread_id="refusal-test", target=GraphTarget(local_graph=graph)
    )
    output = io.StringIO()
    if reason == "refusal":
        with pytest.raises(ModelRefusalError) as caught:
            await stream_json(LocalGraphGateway(), request, out=output)
        assert caught.value.raw_message.response_metadata["output_version"] == "v1"
    else:
        assert (
            await stream_json(LocalGraphGateway(), request, out=output)
            == "Partial answer"
        )
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(e["type"] == "text" and "Partial answer" in e["content"] for e in events)
    assert events[-1]["type"] == ("error" if reason == "refusal" else "done")
    if reason == "refusal":
        assert not any(e["type"] == "done" for e in events)


@pytest.mark.parametrize("reason", ["refusal", "end_turn"])
@pytest.mark.parametrize("output_format", ["text", "stream-json"])
def test_single_shot_cli_reports_refusal(
    monkeypatch, tmp_path, anthropic_model, reason, output_format
):
    from typer.testing import CliRunner

    from EvoScientist.cli import commands
    from EvoScientist.cli._app import app
    from EvoScientist.config import EvoScientistConfig

    graph = create_deep_agent(
        model=anthropic_model(reason),
        middleware=[ErrorNormalizationMiddleware()],
        subagents=[],
    )
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


def test_sync_task_reports_before_output_refusal_as_a_notice():
    import anthropic
    import httpx2
    from langchain_anthropic import ChatAnthropic
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import ToolMessage

    class Parent(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def respond(request):
        return httpx2.Response(
            200,
            json={
                "id": "msg_refused",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": "refusal",
                "stop_sequence": None,
                "stop_details": {
                    "category": "safety",
                    "explanation": "Cannot continue",
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(respond)) as client:
        child = ChatAnthropic(
            model="claude-sonnet-4-6", api_key="test", max_tokens=1024, max_retries=0
        )
        child.__dict__["_client"] = anthropic.Anthropic(
            api_key="test", max_retries=0, http_client=client
        )
        parent = Parent(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "id": "task-1",
                            "args": {
                                "subagent_type": "researcher",
                                "description": "hello",
                            },
                        }
                    ],
                ),
                AIMessage(content="Continue without the researcher"),
            ]
        )
        graph = create_deep_agent(
            model=parent,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Research",
                    "system_prompt": "Research",
                    "model": child,
                    "middleware": [ErrorNormalizationMiddleware()],
                }
            ],
        )
        result = graph.invoke({"messages": [{"role": "user", "content": "hello"}]})

    tasks = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tasks) == 1
    assert (
        "The subagent stopped because a model refused the request" in tasks[0].content
    )
    assert "Cannot continue" in tasks[0].content
    assert result["messages"][-1].content == "Continue without the researcher"
