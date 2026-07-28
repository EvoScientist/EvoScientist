from unittest.mock import AsyncMock, MagicMock

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
