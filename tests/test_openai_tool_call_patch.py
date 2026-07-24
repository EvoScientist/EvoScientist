from langchain_core.messages import AIMessage
from langchain_openai.chat_models import base as openai_base

from EvoScientist.llm import patches as _patches  # noqa: F401


def test_malformed_function_tool_calls_are_removed_from_history():
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {"id": "missing", "name": None, "args": "{", "error": "invalid"},
            {"id": "valid", "name": "search", "args": "{", "error": "invalid"},
        ],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "raw-missing",
                    "type": "function",
                    "function": {"arguments": "{}"},
                }
            ]
        },
    )

    result = openai_base._convert_message_to_dict(message)

    assert result["tool_calls"] == [
        {
            "type": "function",
            "id": "valid",
            "function": {"name": "search", "arguments": "{"},
        }
    ]


def test_removing_only_tool_call_leaves_valid_assistant_content():
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {"id": "missing", "name": None, "args": "{", "error": "invalid"}
        ],
    )

    result = openai_base._convert_message_to_dict(message)

    assert "tool_calls" not in result
    assert result["content"] == ""


def test_raw_tool_call_without_name_is_removed_from_history():
    message = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "raw-missing",
                    "type": "function",
                    "function": {"arguments": "{}"},
                }
            ]
        },
    )

    result = openai_base._convert_message_to_dict(message)

    assert "tool_calls" not in result
    assert result["content"] == ""
