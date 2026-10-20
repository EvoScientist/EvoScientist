"""Tests for EvoScientist.middleware.proactive_mode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.middleware.proactive_mode import (
    ProactiveModeMiddleware,
    _read_proactive_mode,
    create_proactive_mode_middleware,
)


def _request(tools=None):
    """A minimal ModelRequest stand-in with ``tools`` and an immutable ``override``."""
    tools = [object(), object()] if tools is None else tools
    request = SimpleNamespace(tools=tools)
    request.override = lambda **kwargs: SimpleNamespace(
        tools=kwargs.get("tools", request.tools)
    )
    return request


# ---- _read_proactive_mode ---------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"configurable": {"proactive_mode": True}}, True),
        ({}, False),
        ({"configurable": {"other_field": "x"}}, False),
        ({"configurable": {"proactive_mode": False}}, False),
        # Only the literal ``True`` arms the strip; a truthy string does not.
        ({"configurable": {"proactive_mode": "yes"}}, False),
        ({"configurable": "not-a-dict"}, False),
        ("not-a-dict", False),
    ],
)
def test_read_proactive_mode(config, expected):
    with patch("langgraph.config.get_config", return_value=config):
        assert _read_proactive_mode() is expected


@patch("langgraph.config.get_config", side_effect=RuntimeError("outside context"))
def test_read_returns_false_outside_runnable_context(mock_get_config):
    assert _read_proactive_mode() is False


# ---- middleware -------------------------------------------------------------


@patch(
    "langgraph.config.get_config",
    return_value={"configurable": {"proactive_mode": True}},
)
def test_strips_all_tools_when_flag_set(mock_get_config):
    assert ProactiveModeMiddleware()._apply(_request()).tools == []


@patch("langgraph.config.get_config", return_value={"configurable": {}})
def test_passthrough_returns_the_same_request_when_flag_absent(mock_get_config):
    original = _request()
    tools_before = original.tools
    result = ProactiveModeMiddleware()._apply(original)
    assert result is original
    assert result.tools is tools_before


@patch(
    "langgraph.config.get_config",
    return_value={"configurable": {"proactive_mode": True}},
)
def test_wrap_model_call_passes_stripped_request_to_handler(mock_get_config):
    seen = {}

    def handler(req):
        seen["tools"] = req.tools
        return "response"

    result = ProactiveModeMiddleware().wrap_model_call(_request(), handler)
    assert result == "response"
    assert seen["tools"] == []


@patch(
    "langgraph.config.get_config",
    return_value={"configurable": {"proactive_mode": True}},
)
def test_awrap_model_call_passes_stripped_request_to_handler(mock_get_config):
    seen = {}

    async def handler(req):
        seen["tools"] = req.tools
        return "response"

    result = asyncio.run(
        ProactiveModeMiddleware().awrap_model_call(_request(), handler)
    )
    assert result == "response"
    assert seen["tools"] == []


def test_factory_returns_middleware_instance():
    assert isinstance(create_proactive_mode_middleware(), ProactiveModeMiddleware)


# ---- composition in _get_default_middleware ---------------------------------


def _stack(workspace, **kwargs):
    from EvoScientist.config.settings import EvoScientistConfig
    from EvoScientist.EvoScientist import _get_default_middleware

    model = MagicMock(profile={"max_input_tokens": 200_000})
    model.thinking = None
    model.reasoning = None
    return _get_default_middleware(
        workspace=workspace,
        cfg=EvoScientistConfig(enable_ask_user=False),
        chat_model=model,
        **kwargs,
    )


def test_main_agent_stack_has_proactive_mode_first(workspace):
    """First in our list, so it wraps tool_selector, memory and the rest of it.
    The compiled order (deepagents' core middleware outside ours) is pinned by
    the compiled-graph test below."""
    stack = _stack(workspace)
    assert isinstance(stack[0], ProactiveModeMiddleware)
    assert sum(isinstance(m, ProactiveModeMiddleware) for m in stack) == 1


def test_async_subagent_stack_has_no_proactive_mode(workspace):
    stack = _stack(workspace, for_async_subagent=True)
    assert not any(isinstance(m, ProactiveModeMiddleware) for m in stack)


# ---- the compiled agent ------------------------------------------------------


def test_compiled_deep_agent_calls_the_model_without_tools_on_a_proactive_run(
    workspace, tmp_path
):
    """On a real deepagents graph (its core middleware outside ours) a run with
    ``proactive_mode`` reaches the model with no tools and with the no-tools
    memory instructions; a normal run keeps both."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from EvoScientist.middleware.memory import create_memory_middleware

    seen = []
    bound = {"tools": []}

    class Recorder(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            bound["tools"] = list(tools)
            return self

        def _generate(self, messages, *args, **kwargs):
            seen.append((len(bound["tools"]), str(messages[0].content)))
            return super()._generate(messages, *args, **kwargs)

    @tool
    def execute(command: str) -> str:
        """Run a shell command."""
        return "ran"

    graph = create_deep_agent(
        model=Recorder(messages=iter([AIMessage(content="ok")] * 4)),
        tools=[execute],
        middleware=[
            ProactiveModeMiddleware(),
            create_memory_middleware(
                str(tmp_path / "memories"), workspace_dir=workspace.root
            ),
        ],
        checkpointer=InMemorySaver(),
    )

    def run(thread_id, **configurable):
        bound["tools"] = []
        graph.invoke(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"thread_id": thread_id, **configurable}},
        )
        return seen[-1]

    tools, system = run("normal")
    assert tools > 0
    assert "Required memory preflight" in system

    tools, system = run("proactive", proactive_mode=True)
    assert tools == 0
    assert "Required memory preflight" not in system
    assert "No memory tools are available on this turn" in system
