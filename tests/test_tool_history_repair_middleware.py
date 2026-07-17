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
