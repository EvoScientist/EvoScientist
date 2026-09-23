"""Tests for LLMToolSelectorMiddleware integration and the event-sink handoff."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    # ``disable_thinking`` / ``disable_streaming`` are patched at the destination
    # namespace with ``create=True`` because the factory imports them lazily.
    # ``disable_streaming`` returns a MagicMock whose ``.model_copy`` returns
    # itself so the tag/callback update in the factory is a safe no-op for tests
    # that don't care about the tag wiring.
    return (
        patch(
            "EvoScientist.middleware.tool_selector.disable_thinking",
            return_value=MagicMock(),
            create=True,
        ),
        patch(
            "EvoScientist.middleware.tool_selector.disable_streaming",
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
    p1, p2, p3, p4 = _factory_patches()
    with p1, p2, p3, p4:
        result = create_tool_selector_middleware()
        assert isinstance(result, list)
        assert len(result) == 1
        assert type(result[0]).__name__ == "_ConditionalToolSelectorMiddleware"


def test_create_tool_selector_always_include():
    p1, p2, p3, p4 = _factory_patches()
    with p1, p2, p3, p4 as mock_cls:
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
    p1, p2, p3, p4 = _factory_patches()
    with p1, p2, p3, p4:
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


def test_selector_failure_warns_once_per_middleware_instance(caplog):
    """Repeated degradation stays visible without warning on every request."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = RuntimeError("revoked credentials")
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
    )
    request = _request([_tool(f"t{i}") for i in range(10)])

    caplog.set_level("WARNING", logger="EvoScientist.middleware.tool_selector")
    cond.wrap_model_call(request, MagicMock())
    cond.wrap_model_call(request, MagicMock())

    warnings = [
        record
        for record in caplog.records
        if "tool_selector.fallback" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_selector_provider_failure_allows_downstream_model_fallback(caplog):
    """A failed fixed selector model must not block a healthy request fallback."""
    from EvoScientist.llm.errors import ProviderStreamError

    mock_selector = MagicMock()
    mock_selector.awrap_model_call = AsyncMock(
        side_effect=ProviderStreamError(
            provider="openrouter",
            class_qualname="openrouter.ProviderError",
            message="primary unavailable",
        )
    )
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
    )
    request = _request([_tool(f"t{i}") for i in range(10)])
    response = MagicMock()
    handler = AsyncMock(return_value=response)

    caplog.set_level("WARNING", logger="EvoScientist.middleware.tool_selector")
    result = await cond.awrap_model_call(request, handler)

    assert result is response
    handler.assert_awaited_once_with(request)
    assert any(
        "tool_selector.fallback" in record.getMessage() for record in caplog.records
    )


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
# disable_streaming — kills per-chunk selector emissions
# ---------------------------------------------------------------------------


def test_disable_streaming_sets_disable_streaming_field():
    """Helper must set ``disable_streaming=True`` (BaseChatModel's official
    hard-disable field checked by ``_streaming_disabled()``), not the
    model's own ``streaming`` field.
    """
    from EvoScientist.middleware.utils import disable_streaming

    model = MagicMock()
    copied = MagicMock()
    model.model_copy.return_value = copied

    result = disable_streaming(model)

    model.model_copy.assert_called_once_with(update={"disable_streaming": True})
    assert result is copied


def test_disable_streaming_defeats_upstream_streaming_dispatch():
    """End-to-end mechanism test: a model copy produced by
    ``disable_streaming`` causes langchain's own ``_streaming_disabled``
    to return True.

    ``_streaming_disabled`` is the single check consulted by
    ``_should_stream`` / ``_should_use_protocol_streaming`` before
    dispatching to ``_stream`` / ``_astream``. If our field-setting fails
    or a future langchain version changes the check key, this test fails
    before the selector floods anything in production — strictly better
    than a runtime canary.
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from EvoScientist.middleware.utils import disable_streaming

    class _FakeModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

    model = _FakeModel()
    assert model._streaming_disabled() is False

    disabled = disable_streaming(model)

    assert disabled._streaming_disabled() is True
    # Original caller instance untouched.
    assert model._streaming_disabled() is False


def test_create_tool_selector_wires_nostream_and_flood_detector_on_model():
    """Factory chains ``disable_thinking`` → ``disable_streaming`` →
    ``model_copy`` with the ``nostream`` tag and ``_FLOOD_DETECTOR``
    callback, then passes the resulting model to
    ``LLMToolSelectorMiddleware``.

    Model-field wiring (over subclassing or invoke-config injection):
    ``chat_models.py:746-750`` reads ``self.tags`` / ``self.callbacks``
    into every ``CallbackManager.configure``, so the tag reaches
    ``on_chat_model_start`` and langgraph's ``pregel/_messages.py:141``
    check skips the messages emission. Callbacks propagate the same way
    so ``_FLOOD_DETECTOR`` fires on every selector call regardless of
    whether the tag is honored downstream.
    """
    from EvoScientist.middleware.tool_selector import _FLOOD_DETECTOR

    thinking_out = MagicMock(name="disable_thinking_output")
    streaming_out = MagicMock(name="disable_streaming_output")
    # Simulate a base model with pre-existing tags + callbacks so we can
    # verify the factory APPENDS rather than replaces. If the factory used
    # replace semantics, "pre_existing_tag" would be missing from the update.
    streaming_out.tags = ["pre_existing_tag"]
    _pre_existing_cb = MagicMock(name="pre_existing_callback")
    streaming_out.callbacks = [_pre_existing_cb]
    tagged_out = MagicMock(name="tagged_output")
    streaming_out.model_copy.return_value = tagged_out

    # Patch at the SOURCE module (utils) not the destination (tool_selector)
    # because the factory does ``from .utils import ...`` lazily inside its
    # body — patching the tool_selector namespace would be shadowed by that
    # local import binding.
    with (
        patch(
            "EvoScientist.middleware.utils.disable_thinking",
            return_value=thinking_out,
        ) as mock_dt,
        patch(
            "EvoScientist.middleware.utils.disable_streaming",
            return_value=streaming_out,
        ) as mock_ds,
        patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=MagicMock()),
        patch(
            "langchain.agents.middleware.LLMToolSelectorMiddleware",
            return_value=MagicMock(),
        ) as mock_selector,
    ):
        result = create_tool_selector_middleware(threshold=0)
        # selector_factory is lazy — trigger it via wrap_model_call so the
        # LLMToolSelectorMiddleware constructor actually fires and we can
        # observe what model was passed.
        result[0].wrap_model_call(_request([_tool("t")]), MagicMock())

    from langgraph.constants import TAG_NOSTREAM

    mock_dt.assert_called_once()
    mock_ds.assert_called_once_with(thinking_out)
    # model_copy applied on the disable_streaming output with the nostream
    # tag + flood-detector callback APPENDED to whatever the base model
    # already carried. Tag string is pulled from langgraph's own constants
    # — the import above is a build-time canary against langgraph renaming
    # or removing it.
    streaming_out.model_copy.assert_called_once()
    update_kwarg = streaming_out.model_copy.call_args.kwargs["update"]
    assert TAG_NOSTREAM in update_kwarg["tags"]
    assert _FLOOD_DETECTOR in update_kwarg["callbacks"]
    # Append (not replace): pre-existing tags/callbacks survive.
    assert "pre_existing_tag" in update_kwarg["tags"]
    assert _pre_existing_cb in update_kwarg["callbacks"]
    # The tagged model is what reaches LLMToolSelectorMiddleware.
    assert mock_selector.call_args.kwargs["model"] is tagged_out


# ---------------------------------------------------------------------------
# _SelectorFloodDetector — self-reports the provider quirk
# ---------------------------------------------------------------------------


def test_flood_detector_warns_above_threshold(caplog):
    """Detector emits a WARNING with the count + names when tool_calls
    length hits THRESHOLD. Proves the workaround self-reports so we can
    tell if the provider quirk is still recurring in production."""
    import logging as _logging

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from EvoScientist.middleware.tool_selector import _SelectorFloodDetector

    detector = _SelectorFloodDetector()
    tool_calls = [
        {"name": "ToolSelectionResponse", "args": {}, "id": f"id_{i}"}
        for i in range(_SelectorFloodDetector.THRESHOLD)
    ]
    msg = AIMessage(content="", tool_calls=tool_calls)
    result = LLMResult(generations=[[ChatGeneration(message=msg)]])

    with caplog.at_level(
        _logging.WARNING, logger="EvoScientist.middleware.tool_selector"
    ):
        detector.on_llm_end(result)

    assert any("tool_selector.flood" in rec.message for rec in caplog.records)
    assert any("ToolSelectionResponse" in rec.message for rec in caplog.records)


def test_flood_detector_silent_below_threshold(caplog):
    """Normal selector output (single tool_call) does not emit a warning
    — no noise on the fast path."""
    import logging as _logging

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from EvoScientist.middleware.tool_selector import _SelectorFloodDetector

    detector = _SelectorFloodDetector()
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "ToolSelectionResponse", "args": {"tools": ["x"]}, "id": "id"}
        ],
    )
    result = LLMResult(generations=[[ChatGeneration(message=msg)]])

    with caplog.at_level(
        _logging.WARNING, logger="EvoScientist.middleware.tool_selector"
    ):
        detector.on_llm_end(result)

    assert not any("tool_selector.flood" in rec.message for rec in caplog.records)


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


# ---------------------------------------------------------------------------
# Forced tool_choice rejected (Claude Fable 5.1 / Opus 5.5): retry with auto
# ---------------------------------------------------------------------------


class _BadRequest(Exception):
    status_code = 400


_FORCED_REJECTED = _BadRequest(
    'tool_choice: type "tool" and "any" are not supported for this model.'
)


def _switching_middleware(primary_error, auto_selector):
    primary = MagicMock()
    primary.wrap_model_call.side_effect = primary_error
    primary.awrap_model_call = AsyncMock(side_effect=primary_error)
    auto_factory = MagicMock(return_value=auto_selector)
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=primary),
        threshold=5,
        auto_selector_factory=auto_factory,
    )
    return cond, primary, auto_factory


def test_forced_tool_choice_rejection_switches_to_auto_selector():
    auto_selector = MagicMock()
    auto_selector.wrap_model_call.side_effect = lambda req, h: h(req)
    cond, primary, auto_factory = _switching_middleware(_FORCED_REJECTED, auto_selector)
    request = _request([_tool(f"t{i}") for i in range(10)])
    handler = MagicMock(return_value="ok")

    assert cond.wrap_model_call(request, handler) == "ok"
    assert cond.wrap_model_call(request, handler) == "ok"

    primary.wrap_model_call.assert_called_once()
    auto_factory.assert_called_once()
    assert auto_selector.wrap_model_call.call_count == 2


@pytest.mark.asyncio
async def test_forced_tool_choice_rejection_switches_to_auto_selector_async():
    async def _select(req, h):
        return await h(req)

    auto_selector = MagicMock()
    auto_selector.awrap_model_call = AsyncMock(side_effect=_select)
    cond, primary, auto_factory = _switching_middleware(_FORCED_REJECTED, auto_selector)
    request = _request([_tool(f"t{i}") for i in range(10)])
    handler = AsyncMock(return_value="ok")

    assert await cond.awrap_model_call(request, handler) == "ok"

    primary.awrap_model_call.assert_awaited_once()
    auto_factory.assert_called_once()
    auto_selector.awrap_model_call.assert_awaited_once()


def test_forced_tool_choice_rejection_in_error_body_switches_to_auto():
    """OpenRouter keeps the provider message in ``body``, not ``str(exc)``."""

    class _ProviderError(_BadRequest):
        body = '{"error": {"message": "tool_choice: type \\"tool\\" not supported"}}'

    auto_selector = MagicMock()
    auto_selector.wrap_model_call.side_effect = lambda req, h: h(req)
    cond, _, auto_factory = _switching_middleware(
        _ProviderError("Provider returned error"), auto_selector
    )

    cond.wrap_model_call(_request([_tool(f"t{i}") for i in range(10)]), MagicMock())

    auto_factory.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        # langchain's own validation error, no HTTP status
        ValueError("Model selected invalid tools: ['tool_choice']"),
        # transient provider error that happens to mention tool_choice
        type("_RateLimited", (Exception,), {"status_code": 429})("tool_choice busy"),
    ],
)
def test_tool_choice_text_without_400_does_not_switch(error):
    cond, _, auto_factory = _switching_middleware(error, MagicMock())

    cond.wrap_model_call(_request([_tool(f"t{i}") for i in range(10)]), MagicMock())

    auto_factory.assert_not_called()


def test_concurrent_rejection_retries_with_installed_auto_selector():
    """A request that failed on the forced selector after another request
    already switched still retries with the auto selector."""
    auto_selector = MagicMock()
    cond, primary, auto_factory = _switching_middleware(_FORCED_REJECTED, auto_selector)
    cond._build_selector(_request([_tool(f"t{i}") for i in range(10)]))

    assert cond._switch_to_auto_tool_choice(_FORCED_REJECTED, primary) is True
    assert cond._switch_to_auto_tool_choice(_FORCED_REJECTED, primary) is True

    auto_factory.assert_called_once()
    assert cond._selector is auto_selector


def test_rejection_restores_auto_selector_if_overwritten():
    """A late primary-selector assignment must not pin the rejected selector."""
    auto_selector = MagicMock()
    cond, primary, _ = _switching_middleware(_FORCED_REJECTED, auto_selector)
    cond._build_selector(_request([_tool(f"t{i}") for i in range(10)]))
    cond._switch_to_auto_tool_choice(_FORCED_REJECTED, primary)
    cond._selector = primary  # simulate a racing first-time build

    assert cond._switch_to_auto_tool_choice(_FORCED_REJECTED, primary) is True
    assert cond._selector is auto_selector


def test_rejected_auto_selector_is_not_retried():
    """Once auto itself is rejected, each turn makes one auto call, not two."""
    auto_selector = MagicMock()
    auto_selector.wrap_model_call.side_effect = _FORCED_REJECTED
    cond, primary, _ = _switching_middleware(_FORCED_REJECTED, auto_selector)
    request = _request([_tool(f"t{i}") for i in range(10)])

    cond.wrap_model_call(request, MagicMock())
    handler = MagicMock()
    cond.wrap_model_call(request, handler)

    primary.wrap_model_call.assert_called_once()
    assert auto_selector.wrap_model_call.call_count == 2  # one per turn
    handler.assert_called_once_with(request)


def test_other_selector_errors_do_not_switch_to_auto():
    cond, _, auto_factory = _switching_middleware(
        RuntimeError("no structured output"), MagicMock()
    )
    request = _request([_tool(f"t{i}") for i in range(10)])
    handler = MagicMock()

    cond.wrap_model_call(request, handler)

    auto_factory.assert_not_called()
    handler.assert_called_once_with(request)


def test_auto_selector_failure_falls_back_to_all_tools():
    auto_selector = MagicMock()
    auto_selector.wrap_model_call.side_effect = RuntimeError("still malformed")
    cond, _, _ = _switching_middleware(_FORCED_REJECTED, auto_selector)
    request = _request([_tool(f"t{i}") for i in range(10)])
    handler = MagicMock()

    cond.wrap_model_call(request, handler)

    handler.assert_called_once_with(request)


def test_auto_selector_sends_auto_tool_choice_at_low_effort(monkeypatch):
    """Wire-level: the auto selector binds tool_choice=auto, lowers effort, and
    parses a stringified tool list."""
    import json

    import anthropic
    from packaging.version import Version

    from EvoScientist.llm.models import get_chat_model

    if Version(anthropic.__version__) >= Version("1"):
        import httpx2 as httpx
    else:
        import httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    [cond] = create_tool_selector_middleware(
        model=get_chat_model("claude-opus-5-5", provider="anthropic")
    )
    selector = cond._auto_selector_factory([])
    model = selector.model._model
    captured = {}

    def respond(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "ToolSelectionResponse",
                        "input": {"tools": '["ls", "grep"]'},
                    }
                ],
                "model": "claude-opus-5-5",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    model._client = anthropic.Anthropic(
        api_key="sk-test",
        timeout=None,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    schema = {
        "title": "ToolSelectionResponse",
        "type": "object",
        "properties": {"tools": {"type": "array", "items": {"type": "string"}}},
        "required": ["tools"],
    }

    result = selector.model.with_structured_output(schema).invoke("pick tools")

    assert captured["tool_choice"] == {"type": "auto"}
    assert captured["output_config"]["effort"] == "low"
    assert result == {"tools": ["ls", "grep"]}


def test_real_selector_switches_to_auto_end_to_end(monkeypatch):
    """Drives langchain's real LLMToolSelectorMiddleware through the switch, so a
    future langchain change to how it uses ``.model`` fails here first."""
    import json

    import anthropic
    from langchain_core.messages import HumanMessage
    from packaging.version import Version

    from EvoScientist.llm.models import get_chat_model

    if Version(anthropic.__version__) >= Version("1"):
        import httpx2 as httpx
    else:
        import httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("tool_choice", {}).get("type") != "auto":
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": 'tool_choice: type "tool" and "any" are not '
                        "supported for this model.",
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "ToolSelectionResponse",
                        "input": {"tools": ["t1", "t2"]},
                    }
                ],
                "model": "claude-opus-5-5",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    model = get_chat_model("claude-opus-5-5", provider="anthropic")
    # Every selector copy shares this cached client.
    model._client = anthropic.Anthropic(
        api_key="sk-test",
        timeout=None,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    [cond] = create_tool_selector_middleware(model=model, threshold=5)
    request = ModelRequest(
        model=MagicMock(),
        messages=[HumanMessage("find papers")],
        tools=[_tool(f"t{i}") for i in range(10)],
    )
    handler = MagicMock(return_value="response")

    assert cond.wrap_model_call(request, handler) == "response"

    assert [b.get("tool_choice", {}).get("type") for b in bodies] == ["tool", "auto"]
    assert bodies[1]["output_config"]["effort"] == "low"
    hint = "Always respond by calling the ToolSelectionResponse tool"
    assert hint not in json.dumps(bodies[0]["system"])
    assert hint in json.dumps(bodies[1]["system"])
    assert [t.name for t in handler.call_args.args[0].tools] == ["t1", "t2"]


def test_auto_selector_model_delegates_other_attributes():
    from EvoScientist.middleware.tool_selector import _AutoToolChoiceSelectorModel

    inner = MagicMock(profile={"max_input_tokens": 1})
    wrapped = _AutoToolChoiceSelectorModel(inner)

    assert wrapped.profile == {"max_input_tokens": 1}
    assert wrapped.with_config is inner.with_config


def test_auto_selector_model_survives_copy():
    """Forwarding must not hand copy protocols to the wrapped model."""
    import copy

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from EvoScientist.middleware.tool_selector import _AutoToolChoiceSelectorModel

    wrapped = _AutoToolChoiceSelectorModel(FakeListChatModel(responses=["x"]))

    for clone in (copy.copy(wrapped), copy.deepcopy(wrapped)):
        assert type(clone) is _AutoToolChoiceSelectorModel
        assert isinstance(clone._model, FakeListChatModel)
