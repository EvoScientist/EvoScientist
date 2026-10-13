"""The whole chain on the real main graph: a model that drops mid-stream, then answers."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config

from EvoScientist.backends import MODEL_RETRY_RUN_KEY
from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.middleware.model_retry import ModelRetryMiddleware
from EvoScientist.paths import Workspace
from EvoScientist.stream.events import stream_agent_events

pytestmark = pytest.mark.usefixtures("no_mcp")


@pytest.fixture(autouse=True)
def test_config(monkeypatch):
    """The stream runner reads the retry budget from the active config, not the user's."""
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._config",
        EvoScientistConfig(enable_async_subagents=False),
    )


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """Every retry layer built in a test waits on this list instead of the clock."""
    recorded: list[float] = []
    build = ModelRetryMiddleware.__init__

    async def _sleep(seconds):
        recorded.append(seconds)

    def _build(self, **kwargs):
        build(self, **{**kwargs, "sleep": _sleep})

    monkeypatch.setattr(ModelRetryMiddleware, "__init__", _build)
    return recorded


class _Flaky(GenericFakeChatModel):
    """First call streams half a sentence then drops the connection; second call answers."""

    calls: int = 0

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content="Half a sen"))
            if run_manager:
                await run_manager.on_llm_new_token("Half a sen", chunk=chunk)
            yield chunk
            raise ConnectionError("dropped")
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self


def _agent(tmp_path, model, **config):
    cfg = EvoScientistConfig(enable_async_subagents=False, **config)
    return create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=model,
        checkpointer=InMemorySaver(),
    )


async def test_dropped_stream_is_retried_and_the_answer_is_clean(tmp_path, waits):
    model = _Flaky(messages=iter([AIMessage("Complete.")]))
    agent = _agent(tmp_path, model, model_retry_wait_seconds=60)
    events = [event async for event in stream_agent_events(agent, "hi", "t1")]
    retry = next(e for e in events if e["type"] == "model_retry")
    assert retry["discarded"]["text_chars"] == len("Half a sen")
    assert retry["attempt"] == 2
    assert len(waits) == 1
    done = next(e for e in events if e["type"] == "done")
    assert done["response"] == "Complete."
    assert model.calls == 2
    state = await agent.aget_state({"configurable": {"thread_id": "t1"}})
    assert [m.text for m in state.values["messages"] if m.type == "ai"] == ["Complete."]


async def test_local_gateway_run_carries_the_retry_key(tmp_path, monkeypatch):
    """The in-process run writes configurable["model_retry"] from the live config."""
    from EvoScientist import backends

    monkeypatch.setattr(
        backends,
        "model_retry_for_run",
        lambda config=None: {"unattended": True, "wait_budget_seconds": 7},
    )
    model = _Flaky(messages=iter([AIMessage("ok")]))
    model.calls = 1  # skip the flaky first call
    agent = _agent(tmp_path, model)
    seen = {}
    original = agent.astream_events

    def _spy(*args, **kwargs):
        seen["config"] = kwargs.get("config") or (args[1] if len(args) > 1 else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(agent, "astream_events", _spy)
    async for _ in stream_agent_events(agent, "hi", "t2"):
        pass
    assert seen["config"]["configurable"][MODEL_RETRY_RUN_KEY] == {
        "unattended": True,
        "wait_budget_seconds": 7,
    }


class _MetadataStripped:
    """A v3 stream as the server transport delivers it: message events carry no metadata."""

    def __init__(self, stream):
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        async for event in self._stream:
            params = event.get("params") or {}
            data = params.get("data")
            if event.get("method") == "messages" and isinstance(data, tuple):
                event = {**event, "params": {**params, "data": data[0]}}
            yield event


async def test_server_transport_shape_still_discards_the_failed_attempt(
    tmp_path, monkeypatch
):
    """The boundary events alone tell the decoder what to drop: no run_id, no node name."""
    import inspect

    model = _Flaky(messages=iter([AIMessage("Complete.")]))
    agent = _agent(tmp_path, model, model_retry_wait_seconds=60)
    original = agent.astream_events

    async def _stripped(*args, **kwargs):
        stream = original(*args, **kwargs)
        if inspect.isawaitable(stream):
            stream = await stream
        return _MetadataStripped(stream)

    monkeypatch.setattr(agent, "astream_events", _stripped)
    events = [event async for event in stream_agent_events(agent, "hi", "t3")]
    retry = next(e for e in events if e["type"] == "model_retry")
    assert retry["discarded"]["text_chars"] == len("Half a sen")
    assert next(e for e in events if e["type"] == "done")["response"] == "Complete."
    assert not [e for e in events if e["type"] == "model_call"]


class _Whole(FakeMessagesListChatModel):
    """Never streams: each call returns one whole message."""

    def bind_tools(self, tools, **kwargs):
        return self


async def test_error_finished_whole_message_is_retried_and_discarded(tmp_path):
    model = _Whole(
        responses=[
            AIMessage("Half", response_metadata={"finish_reason": "error"}),
            AIMessage("Whole."),
        ]
    )
    agent = _agent(tmp_path, model, model_retry_wait_seconds=60)
    events = [event async for event in stream_agent_events(agent, "hi", "t4")]
    retry = next(e for e in events if e["type"] == "model_retry")
    assert retry["cause"] == "finish_reason"
    assert retry["discarded"]["text_chars"] == len("Half")
    assert next(e for e in events if e["type"] == "done")["response"] == "Whole."


# --- inside a sub-agent, and with a fallback chain ---------------------------


def _text(text):
    return ChatGenerationChunk(message=AIMessageChunk(content=text))


class _Delegating(GenericFakeChatModel):
    """The main agent hands the work to a sub-agent whose first answer drops mid-stream."""

    main_calls: int = 0
    sub_calls: int = 0

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        metadata = get_config().get("metadata") or {}
        if metadata.get("lc_agent_name") == "general-purpose":
            self.sub_calls += 1
            if self.sub_calls == 1:
                yield _text("Half a sen")
                raise ConnectionError("dropped")
            yield _text("Sub done.")
            return
        self.main_calls += 1
        if self.main_calls > 1:
            yield _text("All done.")
            return
        args = {"subagent_type": "general-purpose", "description": "say hi"}
        call = {"name": "task", "args": json.dumps(args), "id": "call_1", "index": 0}
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", tool_call_chunks=[call])
        )

    def bind_tools(self, tools, **kwargs):
        return self


async def test_a_sub_agents_dropped_stream_is_retried_inside_the_sub_agent(
    tmp_path, waits
):
    model = _Delegating(messages=iter([]))
    agent = _agent(tmp_path, model, model_retry_wait_seconds=60)
    events = [event async for event in stream_agent_events(agent, "hi", "t5")]
    (retry,) = [e for e in events if e["type"] == "model_retry"]
    assert retry["subagent"] == "general-purpose"
    assert retry["instance_id"]
    assert retry["discarded"]["text_chars"] == len("Half a sen")
    assert len(waits) == 1
    assert (model.main_calls, model.sub_calls) == (2, 2)
    assert next(e for e in events if e["type"] == "done")["response"] == "All done."
    state = await agent.aget_state({"configurable": {"thread_id": "t5"}})
    (result,) = [m for m in state.values["messages"] if m.type == "tool"]
    assert result.text == "Sub done."


class _Answering(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _Rejecting(GenericFakeChatModel):
    """A fallback entry that fails for good: the provider rejects its key."""

    calls: int = 0

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        import openai

        self.calls += 1
        request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
        raise openai.AuthenticationError(
            "Error code: 401",
            response=httpx.Response(401, request=request),
            body=None,
        )
        yield  # an async generator, as the real stream is

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def fallback(monkeypatch):
    """A one-entry fallback chain served by whatever model the test puts in the holder."""
    from EvoScientist.middleware import model_fallback

    holder: dict = {}
    monkeypatch.setattr(
        "EvoScientist.llm.models.get_chat_model",
        lambda model, provider=None, **kwargs: holder["model"],
    )
    model_fallback.set_fallback_chain([("fallback-model", "openai")])
    yield holder
    model_fallback._reset_chain_initialization()


async def test_a_fallback_takes_over_a_dropped_stream_without_a_wait(
    tmp_path, waits, fallback
):
    primary = _Flaky(messages=iter([AIMessage("never used")]))
    fallback["model"] = _Answering(messages=iter([AIMessage("Fallback.")]))
    agent = _agent(tmp_path, primary, model_retry_wait_seconds=60)
    events = [event async for event in stream_agent_events(agent, "hi", "t6")]
    assert not [e for e in events if e["type"] == "model_retry"]
    assert waits == []
    (dropped,) = [e for e in events if e["type"] == "model_output_discarded"]
    assert dropped["discarded"]["text_chars"] == len("Half a sen")
    done = next(e for e in events if e["type"] == "done")
    assert done["response"] == "Fallback."
    assert primary.calls == 1


async def test_a_broken_fallback_leaves_the_primary_its_retry(
    tmp_path, waits, fallback
):
    primary = _Flaky(messages=iter([AIMessage("Complete.")]))
    fallback["model"] = _Rejecting(messages=iter([]))
    agent = _agent(tmp_path, primary, model_retry_wait_seconds=60)
    events = [event async for event in stream_agent_events(agent, "hi", "t7")]
    (retry,) = [e for e in events if e["type"] == "model_retry"]
    assert "dropped" in retry["reason"]
    assert len(waits) == 1
    assert next(e for e in events if e["type"] == "done")["response"] == "Complete."
    assert (primary.calls, fallback["model"].calls) == (2, 1)
    state = await agent.aget_state({"configurable": {"thread_id": "t7"}})
    assert [m.text for m in state.values["messages"] if m.type == "ai"] == ["Complete."]
