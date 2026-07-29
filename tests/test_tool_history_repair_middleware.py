from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from EvoScientist.middleware.tool_history_repair import (
    ToolHistoryRepairMiddleware,
    repair_tool_history,
)


def _request(messages):
    return ModelRequest(
        messages=messages,
        model=MagicMock(),
        state={},
        runtime=MagicMock(),
        system_message=MagicMock(),
    )


def _tool_call(tool_call_id):
    return {"id": tool_call_id, "name": "execute", "args": {}}


def _invalid_tool_call(tool_call_id):
    return {
        "id": tool_call_id,
        "name": "execute",
        "args": "{not valid json",
        "error": "could not parse args",
    }


def test_synthesizes_results_for_interrupted_tool_calls():
    messages = [
        HumanMessage("run tools"),
        AIMessage(content="", tool_calls=[_tool_call("one"), _tool_call("two")]),
        HumanMessage("continue"),
    ]

    repaired = repair_tool_history(messages)

    assert [type(message) for message in repaired] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert [message.tool_call_id for message in repaired[2:4]] == ["one", "two"]
    assert all(message.status == "error" for message in repaired[2:4])


def test_drops_orphan_tool_results():
    messages = [
        HumanMessage("old request"),
        ToolMessage("late result", tool_call_id="orphan"),
        HumanMessage("continue"),
    ]

    assert repair_tool_history(messages) == [messages[0], messages[2]]


def test_preserves_complete_tool_exchanges():
    messages = [
        HumanMessage("run tool"),
        AIMessage(content="", tool_calls=[_tool_call("complete")]),
        ToolMessage("done", tool_call_id="complete"),
        HumanMessage("continue"),
    ]

    assert repair_tool_history(messages) == messages


def test_removes_unnamed_calls_before_serialization():
    from langchain_openai.chat_models.base import _convert_message_to_dict

    raw_valid = {
        "id": "raw-good",
        "function": {"name": "execute", "arguments": "{}"},
    }
    message = AIMessage(content="").model_copy(
        update={
            "tool_calls": [{"id": "bad", "name": "", "args": {}}],
            "invalid_tool_calls": [
                {**_invalid_tool_call("invalid"), "name": None},
            ],
            "additional_kwargs": {
                "tool_calls": [
                    {"id": "raw-bad", "function": {"arguments": "{}"}},
                    raw_valid,
                ]
            },
        }
    )
    messages = [
        message,
        ToolMessage("bad", tool_call_id="bad"),
        ToolMessage("raw-good", tool_call_id="raw-good"),
    ]

    repaired = repair_tool_history(messages)

    assert repaired[0].tool_calls == []
    assert repaired[0].invalid_tool_calls == []
    assert _convert_message_to_dict(repaired[0])["tool_calls"] == [raw_valid]
    assert [message.tool_call_id for message in repaired[1:]] == ["raw-good"]


def test_malformed_raw_entries_are_dropped_without_crashing():
    message = AIMessage(content="").model_copy(
        update={
            "additional_kwargs": {
                "tool_calls": [
                    {"id": ["a"], "function": {"name": "x", "arguments": "{}"}},
                    {"id": "c1", "function": {"name": ["evil"], "arguments": "{}"}},
                    {"id": "c2", "function": {"name": 7, "arguments": "{}"}},
                ]
            },
        }
    )

    repaired = repair_tool_history([message])

    assert "tool_calls" not in repaired[0].additional_kwargs
    assert not any(isinstance(m, ToolMessage) for m in repaired)


def test_non_str_raw_id_entry_is_dropped_with_its_result():
    message = AIMessage(content="").model_copy(
        update={
            "additional_kwargs": {
                "tool_calls": [
                    {"id": 123, "function": {"name": "f", "arguments": "{}"}}
                ]
            },
        }
    )

    repaired = repair_tool_history([message, ToolMessage("real", tool_call_id="123")])

    assert "tool_calls" not in repaired[0].additional_kwargs
    assert not any(isinstance(m, ToolMessage) for m in repaired)


def test_non_list_raw_tool_calls_value_is_dropped():
    for junk in ({"id": "bad", "function": {"name": "x"}}, "bad", 1):
        message = AIMessage(content="").model_copy(
            update={"additional_kwargs": {"extra": "kept", "tool_calls": junk}}
        )

        repaired = repair_tool_history([message])

        assert "tool_calls" not in repaired[0].additional_kwargs
        assert repaired[0].additional_kwargs["extra"] == "kept"


def test_removes_raw_tool_calls_key_when_all_entries_invalid():
    message = AIMessage(content="").model_copy(
        update={
            "additional_kwargs": {
                "extra": "kept",
                "tool_calls": [{"id": "x", "function": {"arguments": "{}"}}],
            },
        }
    )

    repaired = repair_tool_history([message, ToolMessage("x", tool_call_id="x")])

    assert "tool_calls" not in repaired[0].additional_kwargs
    assert repaired[0].additional_kwargs["extra"] == "kept"
    assert len(repaired) == 1


def test_mixed_named_and_unnamed_parsed_calls():
    message = AIMessage(content="").model_copy(
        update={
            "tool_calls": [
                {"id": "good", "name": "execute", "args": {}},
                {"id": "bad", "name": "", "args": {}},
            ],
        }
    )
    messages = [
        message,
        ToolMessage("ok", tool_call_id="good"),
        ToolMessage("junk", tool_call_id="bad"),
    ]

    repaired = repair_tool_history(messages)

    assert [call["id"] for call in repaired[0].tool_calls] == ["good"]
    assert [m.tool_call_id for m in repaired[1:]] == ["good"]


def test_synthesizes_result_for_unanswered_raw_call():
    message = AIMessage(content="").model_copy(
        update={
            "additional_kwargs": {
                "tool_calls": [
                    {"id": "raw-1", "function": {"name": "grep", "arguments": "{}"}}
                ]
            },
        }
    )

    repaired = repair_tool_history([message])

    assert repaired[-1].tool_call_id == "raw-1"
    assert repaired[-1].name == "grep"
    assert repaired[-1].status == "error"


def test_repair_is_idempotent():
    messages = [
        AIMessage(content="").model_copy(
            update={
                "tool_calls": [
                    _tool_call("kept"),
                    {"id": "bad", "name": "", "args": {}},
                ],
                "additional_kwargs": {
                    "tool_calls": [
                        {
                            "id": "raw-1",
                            "function": {"name": "grep", "arguments": "{}"},
                        },
                        {"id": "raw-2", "function": {"arguments": "{}"}},
                    ]
                },
            }
        ),
        ToolMessage("done", tool_call_id="kept"),
        ToolMessage("junk", tool_call_id="bad"),
    ]

    once = repair_tool_history(messages)

    assert repair_tool_history(once) == once


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_inject_subagent_includes_tool_history_repair(mock_model):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "test-agent"}]
    _inject_subagent_middleware(subs)

    assert any(
        isinstance(m, ToolHistoryRepairMiddleware) for m in subs[0]["middleware"]
    )


def test_wrap_model_call_repairs_request():
    request = _request(
        [
            ToolMessage("late result", tool_call_id="orphan"),
            HumanMessage("continue"),
        ]
    )
    handler = MagicMock(return_value="ok")

    assert ToolHistoryRepairMiddleware().wrap_model_call(request, handler) == "ok"
    assert handler.call_args.args[0].messages == [request.messages[1]]


async def test_awrap_model_call_repairs_request():
    request = _request(
        [
            AIMessage(content="", tool_calls=[_tool_call("interrupted")]),
            HumanMessage("continue"),
        ]
    )
    handler = AsyncMock(return_value="ok")

    assert (
        await ToolHistoryRepairMiddleware().awrap_model_call(request, handler) == "ok"
    )
    repaired = handler.call_args.args[0].messages
    assert isinstance(repaired[1], ToolMessage)
    assert repaired[1].tool_call_id == "interrupted"


def test_synthesizes_results_for_invalid_tool_calls():
    messages = [
        HumanMessage("run tools"),
        AIMessage(
            content="",
            tool_calls=[_tool_call("good")],
            invalid_tool_calls=[_invalid_tool_call("bad")],
        ),
        HumanMessage("continue"),
    ]

    repaired = repair_tool_history(messages)

    assert [type(message) for message in repaired] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert [message.tool_call_id for message in repaired[2:4]] == ["good", "bad"]
    assert all(message.status == "error" for message in repaired[2:4])


def test_preserves_tool_call_name_in_synthesized_result():
    messages = [
        HumanMessage("run tool"),
        AIMessage(content="", tool_calls=[_tool_call("one")]),
    ]

    repaired = repair_tool_history(messages)

    assert repaired[-1].name == "execute"


def test_warning_deduplicates_across_calls(caplog):
    messages = [
        HumanMessage("run tools"),
        AIMessage(content="", tool_calls=[_tool_call("one")]),
    ]
    warned: set[str] = set()

    with caplog.at_level("WARNING"):
        repair_tool_history(messages, warned=warned)
        first_warnings = len(caplog.records)
        repair_tool_history(messages, warned=warned)
        second_warnings = len(caplog.records)

    assert first_warnings == 1
    assert second_warnings == 1
    assert warned == {"one"}


def test_middleware_warns_once_per_thread(caplog):
    middleware = ToolHistoryRepairMiddleware()
    request = _request(
        [
            AIMessage(content="", tool_calls=[_tool_call("interrupted")]),
            HumanMessage("continue"),
        ]
    )
    handler = MagicMock(return_value="ok")

    with caplog.at_level("WARNING"):
        middleware.wrap_model_call(request, handler)
        middleware.wrap_model_call(request, handler)

    assert len(caplog.records) == 1
