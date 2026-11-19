"""Tests for MessageTimestampMiddleware (response_metadata["evo_created_at"])."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from EvoScientist.middleware.message_timestamp import (
    CREATED_AT_KEY,
    MessageTimestampMiddleware,
    stamp_new_messages,
)

_ISO_UTC_MS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_EARLIER = "2020-01-01T00:00:00.000Z"


def _key(message):
    return message.response_metadata.get(CREATED_AT_KEY)


# --- before_agent scan -------------------------------------------------------


def test_new_human_message_is_stamped_in_place_by_id():
    human = HumanMessage("hi", id="h1")

    update = stamp_new_messages([human])

    [stamped] = update["messages"]
    assert stamped.id == "h1"
    assert _ISO_UTC_MS.match(_key(stamped))
    assert stamp_new_messages([stamped]) is None


def test_scan_stops_at_the_first_stamped_message():
    old = HumanMessage("old", id="h0", response_metadata={CREATED_AT_KEY: _EARLIER})
    tool_result = ToolMessage("r", tool_call_id="c", id="t1")
    new = HumanMessage("new", id="h1")

    update = stamp_new_messages([HumanMessage("older", id="hx"), old, tool_result, new])

    assert [m.id for m in update["messages"]] == ["t1", "h1"]


def test_scan_leaves_history_from_before_the_change_unstamped():
    history = [
        HumanMessage("q", id="h0"),
        AIMessage("", id="a0", tool_calls=[{"name": "t", "args": {}, "id": "c0"}]),
        ToolMessage("r", tool_call_id="c0", id="t0"),
        AIMessage("answer", id="a1"),
    ]

    update = stamp_new_messages([*history, HumanMessage("next", id="h1")])

    assert [m.id for m in update["messages"]] == ["h1"]


def test_scan_skips_a_message_without_an_id():
    assert stamp_new_messages([HumanMessage("no id")]) is None


def test_scan_never_mutates_the_state_message():
    human = HumanMessage("hi", id="h1")

    stamp_new_messages([human])

    assert CREATED_AT_KEY not in human.response_metadata


# --- wrap_model_call ----------------------------------------------------------


def test_model_reply_is_stamped():
    reply = AIMessage("done", id="a1")
    response = ModelResponse(result=[reply])

    out = MessageTimestampMiddleware().wrap_model_call(None, lambda _: response)

    assert _ISO_UTC_MS.match(_key(out.result[0]))
    assert CREATED_AT_KEY not in reply.response_metadata


def test_existing_stamp_is_never_rewritten():
    reply = AIMessage("done", response_metadata={CREATED_AT_KEY: _EARLIER})

    out = MessageTimestampMiddleware().wrap_model_call(
        None, lambda _: ModelResponse(result=[reply])
    )

    assert _key(out.result[0]) == _EARLIER


def test_messages_added_since_the_last_model_call_are_stamped_with_the_reply():
    history = [
        HumanMessage("go", id="h1", response_metadata={CREATED_AT_KEY: _EARLIER}),
        AIMessage("", id="a1", response_metadata={CREATED_AT_KEY: _EARLIER}),
        ToolMessage("rejected", tool_call_id="c", id="t1"),
    ]
    request = SimpleNamespace(state={"messages": history})

    out = MessageTimestampMiddleware().wrap_model_call(
        request, lambda _: ModelResponse(result=[AIMessage("done")])
    )

    [pending] = out.command.update["messages"]
    assert pending.id == "t1"
    assert _key(pending) == _key(out.model_response.result[0])
    assert CREATED_AT_KEY not in history[2].response_metadata


def test_reply_alone_when_nothing_is_pending():
    request = SimpleNamespace(
        state={
            "messages": [
                HumanMessage(
                    "go", id="h1", response_metadata={CREATED_AT_KEY: _EARLIER}
                )
            ]
        }
    )

    out = MessageTimestampMiddleware().wrap_model_call(
        request, lambda _: ModelResponse(result=[AIMessage("done")])
    )

    assert isinstance(out, ModelResponse)


def test_model_call_error_propagates_unstamped():
    def failing(_):
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        MessageTimestampMiddleware().wrap_model_call(None, failing)


async def test_async_model_reply_is_stamped():
    async def handler(_):
        return ModelResponse(result=[AIMessage("done")])

    out = await MessageTimestampMiddleware().awrap_model_call(None, handler)

    assert _key(out.result[0])


# --- wrap_tool_call -----------------------------------------------------------


def test_tool_message_is_stamped():
    out = MessageTimestampMiddleware().wrap_tool_call(
        None, lambda _: ToolMessage("r", tool_call_id="c")
    )

    assert _ISO_UTC_MS.match(_key(out))


def test_tool_command_messages_are_stamped():
    command = Command(
        update={"messages": [ToolMessage("r", tool_call_id="c")], "todos": []}
    )

    out = MessageTimestampMiddleware().wrap_tool_call(None, lambda _: command)

    assert _key(out.update["messages"][0])
    assert out.update["todos"] == []


def test_tool_result_keeps_an_existing_stamp():
    result = ToolMessage(
        "r", tool_call_id="c", response_metadata={CREATED_AT_KEY: _EARLIER}
    )

    out = MessageTimestampMiddleware().wrap_tool_call(None, lambda _: result)

    assert _key(out) == _EARLIER


def test_tool_command_without_a_message_list_is_returned_as_is():
    command = Command(update={"todos": []})

    out = MessageTimestampMiddleware().wrap_tool_call(None, lambda _: command)

    assert out is command


def test_tool_command_message_dicts_are_stamped():
    command = Command(
        update={"messages": [{"role": "tool", "content": "r", "tool_call_id": "c"}]}
    )

    out = MessageTimestampMiddleware().wrap_tool_call(None, lambda _: command)

    [message] = out.update["messages"]
    assert isinstance(message, ToolMessage)
    assert _ISO_UTC_MS.match(_key(message))


def test_list_tool_result_is_stamped_item_by_item():
    result = [
        ToolMessage("r", tool_call_id="c1"),
        Command(update={"messages": [ToolMessage("s", tool_call_id="c2")]}),
    ]

    message, command = MessageTimestampMiddleware().wrap_tool_call(
        None, lambda _: result
    )

    assert _ISO_UTC_MS.match(_key(message))
    assert _key(command.update["messages"][0]) == _key(message)


def test_tool_command_messages_share_one_time():
    command = Command(
        update={
            "messages": [
                ToolMessage("a", tool_call_id="c1"),
                ToolMessage("b", tool_call_id="c2"),
            ]
        }
    )

    first, second = (
        MessageTimestampMiddleware().wrap_tool_call(None, lambda _: command)
    ).update["messages"]

    assert _key(first) == _key(second)


async def test_async_tool_message_is_stamped():
    async def handler(_):
        return ToolMessage("r", tool_call_id="c")

    out = await MessageTimestampMiddleware().awrap_tool_call(None, handler)

    assert _key(out)


# --- through a real agent -----------------------------------------------------


class _ScriptedToolModel(BaseChatModel):
    replies: list

    @property
    def _llm_type(self) -> str:
        return "scripted-tools"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"found {query}"


def _tool_turn():
    return [
        AIMessage(
            "", tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}]
        ),
        AIMessage("done"),
    ]


def _agent(replies, checkpointer, *, stamped=True):
    return create_agent(
        model=_ScriptedToolModel(replies=replies),
        tools=[lookup],
        middleware=[MessageTimestampMiddleware()] if stamped else [],
        checkpointer=checkpointer,
    )


def test_agent_run_stamps_every_message_in_order():
    config = {"configurable": {"thread_id": "t"}}
    before = datetime.now(UTC)

    result = _agent(_tool_turn(), InMemorySaver()).invoke(
        {"messages": [HumanMessage("go")]}, config
    )

    stamps = [_key(m) for m in result["messages"]]
    assert [type(m).__name__ for m in result["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert all(_ISO_UTC_MS.match(s) for s in stamps)
    assert stamps == sorted(stamps)
    assert datetime.fromisoformat(stamps[0]) >= before.replace(microsecond=0)


def test_input_is_stamped_when_the_run_starts():
    """``before_agent`` stamps the input before the model sees it, not with
    the reply."""
    seen = []

    class _Recording(_ScriptedToolModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.extend(_key(m) for m in messages if isinstance(m, HumanMessage))
            return super()._generate(messages, stop, run_manager, **kwargs)

    agent = create_agent(
        model=_Recording(replies=[AIMessage("done")]),
        middleware=[MessageTimestampMiddleware()],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}}
    )

    assert seen
    assert all(seen)


def test_second_run_keeps_earlier_stamps():
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    first = _agent(_tool_turn(), saver).invoke(
        {"messages": [HumanMessage("a")]}, config
    )
    earlier = {m.id: _key(m) for m in first["messages"]}

    second = _agent([AIMessage("again")], saver).invoke(
        {"messages": [HumanMessage("b")]}, config
    )

    assert {m.id: _key(m) for m in second["messages"][:4]} == earlier
    assert all(_key(m) for m in second["messages"][4:])


def test_thread_from_before_the_change_keeps_old_messages_unstamped():
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    _agent(_tool_turn(), saver, stamped=False).invoke(
        {"messages": [HumanMessage("old")]}, config
    )

    result = _agent([AIMessage("new reply")], saver).invoke(
        {"messages": [HumanMessage("new")]}, config
    )

    assert [_key(m) is not None for m in result["messages"]] == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_stamping_the_input_does_not_restream_it():
    agent = _agent([AIMessage("done")], InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    streamed = [
        chunk
        for chunk, _meta in agent.stream(
            {"messages": [HumanMessage("go")]}, config, stream_mode="messages"
        )
    ]

    assert not any(isinstance(m, HumanMessage) for m in streamed)


class _RejectToolCalls(AgentMiddleware):
    """Answers every tool call in ``after_model``, like a HITL rejection."""

    def after_model(self, state, runtime):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return None
        return {
            "messages": [
                ToolMessage("rejected", tool_call_id=c["id"], status="error")
                for c in last.tool_calls
            ]
        }


def test_message_added_after_a_model_call_is_stamped_by_the_next_one():
    agent = create_agent(
        model=_ScriptedToolModel(replies=_tool_turn()),
        tools=[lookup],
        middleware=[MessageTimestampMiddleware(), _RejectToolCalls()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}}
    )

    *_, rejected, reply = result["messages"]
    assert rejected.content == "rejected"
    assert _key(rejected) == _key(reply)
    assert all(_key(m) for m in result["messages"])


def test_rejection_beside_an_approved_call_is_stamped():
    """The rejection is written before the approved call's stamped result."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    calls = [
        {"name": "lookup", "args": {"query": "a"}, "id": "c1"},
        {"name": "lookup", "args": {"query": "b"}, "id": "c2"},
    ]
    agent = create_agent(
        model=_ScriptedToolModel(
            replies=[AIMessage("", tool_calls=calls), AIMessage("done")]
        ),
        tools=[lookup],
        middleware=[
            MessageTimestampMiddleware(),
            HumanInTheLoopMiddleware(interrupt_on={"lookup": True}),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    agent.invoke({"messages": [HumanMessage("go")]}, config)

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject"}, {"type": "approve"}]}),
        config,
    )

    tool_results = {
        m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_results["c1"].status == "error"
    assert tool_results["c2"].content == "found b"
    assert all(_key(m) for m in result["messages"])


def test_stamping_costs_one_checkpoint_per_run():
    def checkpoints(stamped):
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "t"}}
        _agent(_tool_turn(), saver, stamped=stamped).invoke(
            {"messages": [HumanMessage("go")]}, config
        )
        return len(list(saver.list(config)))

    assert checkpoints(stamped=True) == checkpoints(stamped=False) + 1


def _deep_agent(replies, workspace):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    return create_deep_agent(
        model=_ScriptedToolModel(
            replies=replies, profile={"max_input_tokens": 200_000}
        ),
        tools=[lookup],
        backend=FilesystemBackend(root_dir=str(workspace.root), virtual_mode=True),
        middleware=[MessageTimestampMiddleware()],
        checkpointer=InMemorySaver(),
    )


def test_deep_agent_run_stamps_every_message(workspace):
    """``create_deep_agent`` stores messages with its own reducer."""
    config = {"configurable": {"thread_id": "t"}}

    result = _deep_agent(_tool_turn(), workspace).invoke(
        {"messages": [HumanMessage("go")]}, config
    )

    assert [type(m).__name__ for m in result["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert all(_ISO_UTC_MS.match(_key(m)) for m in result["messages"])


async def test_deep_agent_async_run_keeps_earlier_stamps(workspace):
    agent = _deep_agent([*_tool_turn(), AIMessage("again")], workspace)
    config = {"configurable": {"thread_id": "t"}}
    first = await agent.ainvoke({"messages": [HumanMessage("a")]}, config)
    earlier = {m.id: _key(m) for m in first["messages"]}

    second = await agent.ainvoke({"messages": [HumanMessage("b")]}, config)

    assert {m.id: _key(m) for m in second["messages"][:4]} == earlier
    assert len(second["messages"]) == 6
    assert all(_key(m) for m in second["messages"])


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai_responses"])
def test_stamp_never_reaches_a_provider_request(provider):
    """``response_metadata`` is not part of any request the providers build."""
    messages = [
        HumanMessage("go", response_metadata={CREATED_AT_KEY: _EARLIER}),
        AIMessage(
            "",
            tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}],
            response_metadata={CREATED_AT_KEY: _EARLIER},
        ),
        ToolMessage(
            "found x", tool_call_id="c1", response_metadata={CREATED_AT_KEY: _EARLIER}
        ),
    ]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-sonnet-4-5", api_key="placeholder")
    else:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="gpt-4o",
            api_key="placeholder",
            use_responses_api=provider == "openai_responses",
        )

    payload = repr(model._get_request_payload(messages))

    assert "found x" in payload
    assert CREATED_AT_KEY not in payload
    assert _EARLIER not in payload


# --- wiring -------------------------------------------------------------------


@pytest.fixture
def no_configured_model(monkeypatch):
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_chat_model",
        lambda: FakeMessagesListChatModel(responses=[]),
    )


def _names(middleware):
    return [m.name for m in middleware]


def test_main_stack_stamps_inside_retry_and_outside_fallback(
    workspace, no_configured_model
):
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.EvoScientist import _get_default_middleware

    names = _names(
        _get_default_middleware(workspace=workspace, cfg=EvoScientistConfig())
    )

    assert names.index("model_retry") < names.index("message_timestamp")
    assert names.index("message_timestamp") < names.index("model_fallback")


def test_subagent_stack_stamps_inside_retry(workspace, no_configured_model):
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "research", "middleware": []}]
    _inject_subagent_middleware(subs, workspace=workspace, cfg=EvoScientistConfig())

    names = _names(subs[0]["middleware"])
    assert names.index("model_retry") < names.index("message_timestamp")
