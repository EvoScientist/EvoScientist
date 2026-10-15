"""Tests for EvoScientist.proactive.shadow."""

from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from EvoScientist.config.settings import EvoScientistConfig
from EvoScientist.middleware.proactive_mode import ProactiveModeMiddleware
from EvoScientist.proactive.shadow import (
    ProactiveToolLeakError,
    flatten_tool_history,
    install_tool_strip_spy,
    run_shadow_turn,
    shadow_cfg,
)


class _ToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def execute(command: str) -> str:
    """Run a shell command."""
    return "ran"


def _model(*replies):
    return _ToolModel(messages=iter(list(replies)))


def _shadow_graph(model, *, strip=True):
    """A shadow-shaped graph: one real tool, the strip middleware, the spy."""
    install_tool_strip_spy(model)
    return create_agent(
        model,
        tools=[execute],
        middleware=[ProactiveModeMiddleware()] if strip else [],
        checkpointer=InMemorySaver(),
    )


SOURCE = [
    HumanMessage(content="run the benchmark", id="m1"),
    AIMessage(content="I will report back", id="m2"),
]


def test_shadow_cfg_overrides_only_the_side_effect_flags():
    cfg = shadow_cfg(EvoScientistConfig(model_fallbacks="a,b"))
    assert cfg.enable_scheduler is False
    assert cfg.memory_workers_enabled is False
    assert cfg.enable_ask_user is False
    assert cfg.auto_mode is True
    assert cfg.auto_approve is True
    assert cfg.model_fallbacks == ""


def test_spy_refuses_tools_and_keeps_the_model_type():
    model = _model(AIMessage(content="x"))
    install_tool_strip_spy(model)
    assert isinstance(model, _ToolModel)
    assert model.bind_tools([]) is model
    with pytest.raises(ProactiveToolLeakError, match="execute"):
        model.bind_tools([execute])


def test_returns_the_final_text_with_tools_stripped():
    graph = _shadow_graph(_model(AIMessage(content="The benchmark finished.")))
    assert (
        asyncio.run(run_shadow_turn(graph, SOURCE, "trigger"))
        == "The benchmark finished."
    )


def test_a_tool_reaching_the_model_fails_closed():
    graph = _shadow_graph(_model(AIMessage(content="x")), strip=False)
    assert asyncio.run(run_shadow_turn(graph, SOURCE, "trigger")) is None


def test_a_final_tool_call_is_not_a_reply():
    # No strip and no spy: the model asks for a tool, which then runs.
    calls = AIMessage(content="", tool_calls=[{"name": "nope", "args": {}, "id": "c1"}])
    graph = create_agent(_model(calls), tools=[], checkpointer=InMemorySaver())
    assert asyncio.run(run_shadow_turn(graph, SOURCE, "trigger")) is None


@pytest.mark.parametrize(
    ("metadata", "clean"),
    [
        ({"finish_reason": "stop"}, True),
        ({"stop_reason": "end_turn"}, True),
        ({}, True),
        ({"finish_reason": "length"}, False),
        ({"stop_reason": "max_tokens"}, False),
        ({"finish_reason": "content_filter"}, False),
    ],
)
def test_unclean_finish_is_discarded(metadata, clean):
    reply = AIMessage(content="text", response_metadata=metadata)
    graph = _shadow_graph(_model(reply))
    result = asyncio.run(run_shadow_turn(graph, SOURCE, "trigger"))
    assert (result == "text") is clean


def test_a_failing_turn_raises_and_deletes_its_thread():
    class Broken(_ToolModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("provider down")

    graph = _shadow_graph(Broken(messages=iter([])))
    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(run_shadow_turn(graph, SOURCE, "trigger"))
    assert list(graph.checkpointer.list(None)) == []


def test_shadow_thread_is_deleted_afterwards():
    graph = _shadow_graph(_model(AIMessage(content="ok")))
    asyncio.run(run_shadow_turn(graph, SOURCE, "trigger"))
    assert list(graph.checkpointer.list(None)) == []


class _TickState(TypedDict, total=False):
    reply: str


def test_shadow_inside_a_graph_node_stays_out_of_the_parent_run():
    """Run inside a cron run's node, the shadow must not inherit that run's
    context: its tokens would stream on the cron run, and its checkpoints could
    land in the cron thread."""
    shadow = _shadow_graph(_model(AIMessage(content="secret shadow text")))

    async def node(state):
        return {"reply": await run_shadow_turn(shadow, SOURCE, "trigger")}

    builder = StateGraph(_TickState)
    builder.add_node("tick", node)
    builder.add_edge(START, "tick")
    builder.add_edge("tick", END)
    parent_saver = InMemorySaver()
    parent = builder.compile(checkpointer=parent_saver)

    async def run():
        streamed, final = [], None
        async for mode, chunk in parent.astream(
            {},
            {"configurable": {"thread_id": "cron"}},
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                streamed.append(chunk[0].content)
            else:
                final = chunk
        return streamed, final

    streamed, final = asyncio.run(run())

    assert final["reply"] == "secret shadow text"
    assert streamed == []
    namespaces = {
        c.config["configurable"].get("checkpoint_ns", "")
        for c in parent_saver.list(None)
    }
    assert namespaces == {""}


# ---- tool history -----------------------------------------------------------

TOOL_HISTORY = [
    HumanMessage(content="what is in the folder?", id="m1"),
    AIMessage(
        content="Let me look.",
        id="m2",
        tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "call-1"}],
    ),
    ToolMessage(
        content="results.csv\nplot.png", tool_call_id="call-1", name="ls", id="m3"
    ),
    AIMessage(content="There is a results table and a plot.", id="m4"),
]


def test_flatten_turns_tool_calls_and_results_into_text():
    flat = flatten_tool_history(TOOL_HISTORY)

    assert [m.id for m in flat] == ["m1", "m2", "m3", "m4"]
    assert not any(isinstance(m, ToolMessage) for m in flat)
    assert all(not getattr(m, "tool_calls", None) for m in flat)
    assert flat[1].content == 'Let me look.\n[called tool `ls` with {"path": "/"}]'
    assert flat[2].content == "[result of tool `ls`]\nresults.csv\nplot.png"
    assert flat[3] is TOOL_HISTORY[3]


def test_flatten_keeps_a_tool_call_without_text():
    call = AIMessage(
        content="", id="a", tool_calls=[{"name": "x", "args": {}, "id": "c"}]
    )
    assert flatten_tool_history([call])[0].content == "[called tool `x` with {}]"


def test_flatten_drops_the_tool_block_of_an_unparsable_call():
    call = AIMessage(
        content=[
            {"type": "text", "text": "Trying."},
            {"type": "tool_use", "id": "c", "name": "x", "input": {}},
        ],
        id="a",
        invalid_tool_calls=[
            {"name": "x", "args": "{bad", "id": "c", "error": "bad json"}
        ],
    )

    [flat] = flatten_tool_history([call])

    assert flat.content == 'Trying.\n[called tool `x` with "{bad"]'


def test_flatten_puts_a_placeholder_for_an_image_result():
    data = "A" * 200_000
    result = ToolMessage(
        content=[
            {"type": "text", "text": "plot.png"},
            {"type": "image", "base64": data, "mime_type": "image/png"},
        ],
        tool_call_id="c",
        name="read_file",
        id="t",
    )

    [flat] = flatten_tool_history([result])

    assert flat.content == "[result of tool `read_file`]\nplot.png\n[image]"


class _StrictNoToolsModel(_ToolModel):
    """Refuses tool blocks when no tools are bound, as Anthropic's API does."""

    def _generate(self, messages, *args, **kwargs):
        for message in messages:
            if isinstance(message, ToolMessage) or getattr(message, "tool_calls", None):
                raise ValueError("requests with tool_use blocks must define tools")
        return super()._generate(messages, *args, **kwargs)


def test_a_conversation_with_tool_calls_reaches_a_strict_provider_as_text():
    graph = _shadow_graph(
        _StrictNoToolsModel(messages=iter([AIMessage(content="Done.")]))
    )
    assert asyncio.run(run_shadow_turn(graph, TOOL_HISTORY, "trigger")) == "Done."


def test_the_real_shadow_stack_calls_the_model_without_tools(workspace, monkeypatch):
    """No middleware of the real agent binds tools on a shadow turn."""
    import EvoScientist.EvoScientist as es_mod
    from EvoScientist.paths import SessionDirs
    from EvoScientist.proactive.shadow import build_shadow_graph

    monkeypatch.setattr(es_mod, "_load_mcp_tools_cached", lambda **kwargs: {})
    binds = []

    class Recording(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            binds.append(len(tools))
            return self

    model = Recording(
        messages=iter(
            [AIMessage(content="Done.", response_metadata={"finish_reason": "stop"})]
        )
    )
    graph = build_shadow_graph(EvoScientistConfig(), model, SessionDirs(workspace))

    assert asyncio.run(run_shadow_turn(graph, TOOL_HISTORY, "trigger")) == "Done."
    assert binds == []
