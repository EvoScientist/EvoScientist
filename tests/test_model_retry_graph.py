"""The whole chain on the real main graph: a model that drops mid-stream, then answers."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.backends import MODEL_RETRY_RUN_KEY
from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.paths import Workspace
from EvoScientist.stream.events import stream_agent_events


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


async def test_dropped_stream_is_retried_and_the_answer_is_clean(tmp_path, monkeypatch):
    from EvoScientist.middleware import model_retry as mod

    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)
    model = _Flaky(messages=iter([AIMessage("Complete.")]))
    agent = _agent(tmp_path, model, model_retry_wait_seconds=60)
    events = [event async for event in stream_agent_events(agent, "hi", "t1")]
    retry = next(e for e in events if e["type"] == "model_retry")
    assert retry["discarded"]["text_chars"] == len("Half a sen")
    assert retry["attempt"] == 2
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

    from EvoScientist.middleware import model_retry as mod

    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)
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


async def test_error_finished_whole_message_is_retried_and_discarded(
    tmp_path, monkeypatch
):
    from EvoScientist.middleware import model_retry as mod

    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)
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
