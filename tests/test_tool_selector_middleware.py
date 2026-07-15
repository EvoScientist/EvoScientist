"""Tests for LLMToolSelectorMiddleware integration and the event-sink handoff."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.tools import BaseTool, StructuredTool

from EvoScientist.middleware.tool_selector import (
    _ConditionalToolSelectorMiddleware,
    create_tool_selector_middleware,
)
from EvoScientist.stream.emitter import StreamEventEmitter
from EvoScientist.stream.sink import SessionEventSink
from EvoScientist.stream.tool_selection import _ToolSelectionSuppressor


class _RecordingSink:
    """Records selection lifecycle calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.active = False

    def on_tool_selection_started(self, total_tools: int) -> None:
        self.active = True
        self.calls.append(("started", total_tools))

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        self.calls.append(("selection", list(selected), total_tools))

    def on_tool_selection_ended(self) -> None:
        self.active = False
        self.calls.append(("ended",))

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        pass

    @property
    def tool_selection_active(self) -> bool:
        return self.active


def _tool(name: str) -> BaseTool:
    def _func(value: str = "") -> str:
        return value

    return StructuredTool.from_function(
        func=_func,
        name=name,
        description=f"{name} test tool",
    )


def _request(tools: list[BaseTool | dict[str, Any]]) -> ModelRequest:
    return ModelRequest(model=MagicMock(), messages=[], tools=tools)


def _mock_model():
    """Create a MagicMock model compatible with disable_thinking()."""
    m = MagicMock(profile={"max_input_tokens": 200_000})
    m.thinking = None
    m.reasoning = None
    return m


# Helper: patches needed to call create_tool_selector_middleware without LLM
def _factory_patches():
    return (
        patch(
            "EvoScientist.middleware.tool_selector.disable_thinking",
            return_value=MagicMock(),
            create=True,
        ),
        patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=MagicMock()),
        patch(
            "langchain.agents.middleware.LLMToolSelectorMiddleware",
            return_value=MagicMock(),
        ),
    )


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_create_tool_selector_returns_single_middleware():
    p1, p2, p3 = _factory_patches()
    with p1, p2, p3:
        result = create_tool_selector_middleware()
        assert isinstance(result, list)
        assert len(result) == 1
        assert type(result[0]).__name__ == "_ConditionalToolSelectorMiddleware"


def test_create_tool_selector_always_include():
    p1, p2, p3 = _factory_patches()
    with p1, p2, p3 as mock_cls:
        result = create_tool_selector_middleware(threshold=0)
        request = _request(
            [
                _tool("think_tool"),
                _tool("search_observations"),
                _tool("read_memory"),
                _tool("unrelated_tool"),
            ]
        )
        result[0].wrap_model_call(request, MagicMock())
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["always_include"] == [
            "read_memory",
            "search_observations",
            "think_tool",
        ]


def test_custom_threshold():
    p1, p2, p3 = _factory_patches()
    with p1, p2, p3:
        result = create_tool_selector_middleware(threshold=5)
        assert result[0]._threshold == 5


# ---------------------------------------------------------------------------
# Conditional selector unit tests
# ---------------------------------------------------------------------------


def test_conditional_skips_below_threshold():
    """When tools <= threshold, selector is skipped and nothing is reported."""
    mock_selector = MagicMock()
    selector_factory = MagicMock(return_value=mock_selector)
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=10,
        events=sink,
    )

    request = MagicMock()
    request.tools = [MagicMock() for _ in range(5)]
    handler = MagicMock()

    cond.wrap_model_call(request, handler)
    handler.assert_called_once_with(request)
    selector_factory.assert_not_called()
    mock_selector.wrap_model_call.assert_not_called()
    assert sink.calls == []  # no selection ran → no events


def test_conditional_runs_above_threshold():
    """When tools > threshold, selector runs."""
    mock_selector = MagicMock()
    selector_factory = MagicMock(return_value=mock_selector)
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=10,
    )

    request = MagicMock()
    request.tools = [MagicMock() for _ in range(15)]
    handler = MagicMock()

    cond.wrap_model_call(request, handler)
    selector_factory.assert_called_once_with([])
    mock_selector.wrap_model_call.assert_called_once()
    handler.assert_not_called()


def test_selection_lifecycle_reported_to_sink():
    """started(total) → selection(selected, total) → ended, reported to the sink."""
    # The fake selector filters the request down to two named tools before
    # calling the downstream handler.
    filtered = _request([_tool("read_file"), _tool("think_tool")])

    def fake_selector_call(request, handler):
        return handler(filtered)

    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = fake_selector_call
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])
    cond.wrap_model_call(request, MagicMock())

    assert sink.calls == [
        ("started", 10),
        ("selection", ["read_file", "think_tool"], 10),
        ("ended",),
    ]


def test_selector_failure_reports_ended_without_selection():
    """A selector that raises before the handler surfaces no selection event."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = RuntimeError("no structured output")
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])
    handler = MagicMock()
    cond.wrap_model_call(request, handler)

    # Falls back to all tools; only started/ended reported, no selection.
    handler.assert_called_once_with(request)
    assert ("started", 10) in sink.calls
    assert not any(c[0] == "selection" for c in sink.calls)
    assert sink.calls[-1] == ("ended",)


def test_selector_failure_ends_before_sync_fallback_handler():
    """All-tools fallback must not run while selector suppression is active."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = RuntimeError("no structured output")
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])

    def handler(req):
        sink.calls.append(("handler", sink.tool_selection_active))
        return MagicMock()

    cond.wrap_model_call(request, handler)

    assert sink.calls == [
        ("started", 10),
        ("ended",),
        ("handler", False),
    ]


@pytest.mark.asyncio
async def test_selector_failure_ends_before_async_fallback_handler():
    """Async all-tools fallback must see selection already closed."""
    mock_selector = MagicMock()
    mock_selector.awrap_model_call.side_effect = RuntimeError("no structured output")
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])

    async def handler(req):
        sink.calls.append(("handler", sink.tool_selection_active))
        return MagicMock()

    await cond.awrap_model_call(request, handler)

    assert sink.calls == [
        ("started", 10),
        ("ended",),
        ("handler", False),
    ]


def test_selector_always_includes_available_memory_tools():
    """Adaptive selection must mark available memory tools as mandatory."""
    calls = []

    class FakeSelector:
        def __init__(self, always_include):
            self.always_include = always_include

        def wrap_model_call(self, request, handler):
            calls.append(self.always_include)
            return handler(request)

    def selector_factory(always_include):
        return FakeSelector(always_include)

    request = _request(
        [
            _tool("think_tool"),
            _tool("search_observations"),
            _tool("read_memory"),
            _tool("unrelated_tool"),
        ]
    )

    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=0,
        always_include=frozenset(
            {
                "think_tool",
                "task",
                "search_observations",
                "read_memory",
                "record_observation",
            }
        ),
    )
    handler = MagicMock()

    cond.wrap_model_call(request, handler)

    assert calls == [
        [
            "read_memory",
            "search_observations",
            "think_tool",
        ]
    ]


def test_selector_resolved_once_across_repeated_requests():
    """Agent tools are stable, so build the selector once and reuse it."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = lambda request, handler: handler(
        request
    )
    selector_factory = MagicMock(return_value=mock_selector)

    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=0,
        always_include=frozenset({"think_tool", "search_observations"}),
    )
    tools = [
        _tool("think_tool"),
        _tool("search_observations"),
        _tool("unrelated_tool"),
    ]

    for _ in range(3):
        cond.wrap_model_call(_request(tools), MagicMock())

    selector_factory.assert_called_once_with(["search_observations", "think_tool"])
    assert mock_selector.wrap_model_call.call_count == 3


# ---------------------------------------------------------------------------
# R1: consume-once + dedup render sequences (sink + suppressor)
# ---------------------------------------------------------------------------


def _drive_selection(sink, suppressor, selected, total):
    """Mimic one selection turn: sink records it, the suppressor observes the
    selector JSON block, then a flush surfaces (or not) the UI event."""
    sink.on_tool_selection_started(total)
    sink.on_tool_selection(selected, total)
    sink.on_tool_selection_ended()
    # Suppressor observes the selector's structured-output tool block.
    suppressor.observe_tool_block("ToolSelectionResponse")
    return suppressor.flush_selection()


def test_render_sequences_table():
    """select → render; same selection again → no repeat; new selection → render."""
    cases = [
        # (label, selected, total, expect_render)
        ("first selection renders", ["read_file", "think_tool"], 5, True),
        ("same selection again does not repeat", ["read_file", "think_tool"], 5, False),
        ("new selection renders", ["execute", "think_tool"], 5, True),
        ("kept-all selection does not render", ["a", "b", "c"], 3, False),
    ]
    sink = SessionEventSink()
    suppressor = _ToolSelectionSuppressor(StreamEventEmitter(), sink)

    for label, selected, total, expect_render in cases:
        events = _drive_selection(sink, suppressor, selected, total)
        rendered = [e for e in events if e.get("type") == "tool_selection"]
        if expect_render:
            assert rendered, f"{label}: expected a tool_selection event"
            assert rendered[0]["tools"] == selected, label
        else:
            assert not rendered, f"{label}: expected no tool_selection event"


def test_consume_is_once_only():
    """A pending selection renders once; a second flush yields nothing."""
    sink = SessionEventSink()
    suppressor = _ToolSelectionSuppressor(StreamEventEmitter(), sink)

    first = _drive_selection(sink, suppressor, ["read_file"], 3)
    assert any(e.get("type") == "tool_selection" for e in first)

    # No new selection recorded; the observation flag was consumed.
    suppressor.observe_tool_block("ToolSelectionResponse")
    second = suppressor.flush_selection()
    assert not any(e.get("type") == "tool_selection" for e in second)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    side_effect=lambda *a, **kw: [
        _ConditionalToolSelectorMiddleware(
            selector_factory=MagicMock(return_value=MagicMock()),
            threshold=20,
        )
    ],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_tool_selector(mock_config, mock_model, mock_ts):
    mock_model.return_value = _mock_model()
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    type_names = [type(m).__name__ for m in mw]
    assert "_ConditionalToolSelectorMiddleware" in type_names


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_subagent_no_tool_selector(mock_model):
    mock_model.return_value = _mock_model()

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "test-agent"}]
    _inject_subagent_middleware(subs)

    type_names = [type(m).__name__ for m in subs[0]["middleware"]]
    assert "_ConditionalToolSelectorMiddleware" not in type_names


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    side_effect=lambda *a, **kw: [
        _ConditionalToolSelectorMiddleware(
            selector_factory=MagicMock(return_value=MagicMock()),
            threshold=20,
        )
    ],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_tool_selector_ordering(mock_config, mock_model, mock_ts):
    """ToolSelector should come after ToolErrorHandler and before Memory."""
    mock_model.return_value = _mock_model()
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    type_names = [type(m).__name__ for m in mw]

    ts_idx = type_names.index("_ConditionalToolSelectorMiddleware")
    te_idx = type_names.index("ToolErrorHandlerMiddleware")
    mem_idx = type_names.index("EvoMemoryMiddleware")
    assert te_idx < ts_idx < mem_idx


# ---------------------------------------------------------------------------
# disable_thinking: DeepSeek helper copies (issue #348)
# ---------------------------------------------------------------------------


def _deepseek_model(monkeypatch, **kwargs):
    from EvoScientist.llm.deepseek import EvoChatDeepSeek

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    return EvoChatDeepSeek(model="deepseek-v4-pro", **kwargs)


def test_disable_thinking_deepseek_sets_request_field(monkeypatch):
    """DeepSeek thinking is a server-side default; the helper copy must
    disable it in the request body, or the selector's forced tool_choice
    is rejected ("Thinking mode does not support this tool_choice")."""
    from EvoScientist.middleware.utils import disable_thinking

    model = _deepseek_model(monkeypatch)
    safe = disable_thinking(model)

    assert safe is not model
    assert safe.extra_body == {"thinking": {"type": "disabled"}}
    assert model.extra_body is None  # original untouched
    assert type(safe) is type(model)


def test_disable_thinking_deepseek_preserves_extra_body(monkeypatch):
    from EvoScientist.middleware.utils import disable_thinking

    model = _deepseek_model(monkeypatch, extra_body={"custom": 1})
    safe = disable_thinking(model)

    assert safe.extra_body == {"custom": 1, "thinking": {"type": "disabled"}}
    assert model.extra_body == {"custom": 1}


@pytest.mark.parametrize("provider", ["deepseek", "custom-openai"])
async def test_deepseek_selector_uses_copy_settings(monkeypatch, provider):
    import json

    import httpx
    from langchain_core.messages import HumanMessage

    from EvoScientist.llm.models import get_chat_model

    if provider == "deepseek":
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    else:
        monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://api.deepseek.com")
    captured = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ToolSelectionResponse",
                                        "arguments": json.dumps({"tools": ["tool_1"]}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = get_chat_model(
            "deepseek-v4-flash",
            provider=provider,
            http_async_client=client,
        )
        selector = create_tool_selector_middleware(model=model, threshold=0)[0]
        request = ModelRequest(
            model=model,
            messages=[HumanMessage("pick a tool")],
            tools=[_tool(f"tool_{index}") for index in range(3)],
        )
        selected = []

        async def handler(req):
            selected.extend(tool.name for tool in req.tools)

        await selector.awrap_model_call(request, handler)

    assert "response_format" not in captured
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["tool_choice"]["function"]["name"] == "ToolSelectionResponse"
    assert selected == ["tool_1"]
