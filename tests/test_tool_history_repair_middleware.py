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


# ---------------------------------------------------------------------------
# Regression tests for issue #345: blank tool_call_id from streaming providers
# (Kimi, Zhipu, etc.) was passed through to the next model call, where strict
# providers rejected it as `invalid tool_call_id` (HTTP 400, code 3).
# ---------------------------------------------------------------------------


def _blank_tool_call():
    return {"id": "", "name": "execute", "args": {"cmd": "ls"}}


def test_normalizes_blank_tool_call_id_in_paired_exchange():
    """AIMessage(blank id) + ToolMessage(blank id) → both get the same fresh id."""
    messages = [
        HumanMessage("run task"),
        AIMessage(content="thinking", tool_calls=[_blank_tool_call()]),
        ToolMessage(content="result", tool_call_id="", name="execute"),
        HumanMessage("continue"),
    ]

    repaired = repair_tool_history(messages)

    ai_msg = next(m for m in repaired if isinstance(m, AIMessage))
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1, "paired ToolMessage must be preserved"

    new_id = ai_msg.tool_calls[0]["id"]
    assert isinstance(new_id, str)
    assert new_id
    assert new_id != ""
    assert tool_msgs[0].tool_call_id == new_id, (
        "ToolMessage must adopt the same fresh id as its AIMessage call"
    )
    assert tool_msgs[0].content == "result"


def test_normalizes_none_tool_call_id_and_synthesizes_missing_result():
    """AIMessage(None id) with no matching ToolMessage gets a fresh id plus a
    synthesized interrupted-result ToolMessage."""
    messages = [
        HumanMessage("run task"),
        AIMessage(
            content="thinking",
            tool_calls=[{"id": None, "name": "execute", "args": {}}],
        ),
        HumanMessage("continue"),
    ]

    repaired = repair_tool_history(messages)

    assert [type(message) for message in repaired] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
    ]
    ai_msg = repaired[1]
    tool_msg = repaired[2]
    new_id = ai_msg.tool_calls[0]["id"]
    assert isinstance(new_id, str)
    assert new_id
    assert tool_msg.tool_call_id == new_id
    assert tool_msg.status == "error"


def test_normalizes_multiple_blank_ids_positionally():
    """Multiple blank-id calls in one AIMessage pair FIFO with subsequent
    blank-id ToolMessages."""
    messages = [
        HumanMessage("run"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "", "name": "first", "args": {}},
                {"id": "   ", "name": "second", "args": {}},
            ],
        ),
        ToolMessage(content="result-1", tool_call_id="", name="first"),
        ToolMessage(content="result-2", tool_call_id="   ", name="second"),
    ]

    repaired = repair_tool_history(messages)

    ai_msg = next(m for m in repaired if isinstance(m, AIMessage))
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2

    first_id = ai_msg.tool_calls[0]["id"]
    second_id = ai_msg.tool_calls[1]["id"]
    assert first_id
    assert second_id
    assert first_id != second_id
    assert tool_msgs[0].tool_call_id == first_id
    assert tool_msgs[1].tool_call_id == second_id
    assert [m.content for m in tool_msgs] == ["result-1", "result-2"]


def test_drops_orphan_blank_tool_message():
    """A blank-id ToolMessage with no preceding blank-id AIMessage call is
    still dropped (existing orphan behavior preserved)."""
    messages = [
        HumanMessage("old request"),
        ToolMessage("late result", tool_call_id=""),
        HumanMessage("continue"),
    ]

    repaired = repair_tool_history(messages)

    assert [type(message) for message in repaired] == [HumanMessage, HumanMessage]
    assert not any(isinstance(m, ToolMessage) for m in repaired)


def test_pending_slots_scoped_per_exchange_not_global_fifo():
    """Regression for CodeRabbit review on PR #399: an interrupted blank call
    in an earlier exchange must not leak its fresh id into a later exchange's
    blank ToolMessage via the FIFO queue.

    Sequence:
        AIMessage1(blank A) -- interrupted, no ToolMessage
        HumanMessage         -- boundary closes the queue
        AIMessage2(blank B)
        ToolMessage(blank)   -- must pair with B, not A

    Before the fix, the FIFO queue still held A's id at the top, so the
    ToolMessage was mis-tagged with A's id; the main loop then dropped the
    real B result as an orphan and synthesized a fake interrupted result
    for B.
    """
    messages = [
        HumanMessage("turn 1"),
        AIMessage(
            content="",
            tool_calls=[{"id": "", "name": "interrupted_call", "args": {}}],
        ),
        HumanMessage("turn 2"),
        AIMessage(
            content="",
            tool_calls=[{"id": "", "name": "real_call", "args": {}}],
        ),
        ToolMessage(content="real result", tool_call_id="", name="real_call"),
    ]

    repaired = repair_tool_history(messages)

    ai_msgs = [m for m in repaired if isinstance(m, AIMessage)]
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]

    # Two ToolMessages expected: one synthesized (interrupted_call, error)
    # and one preserved (real_call, with its real content).
    assert len(tool_msgs) == 2
    real_results = [m for m in tool_msgs if m.content == "real result"]
    synthesized = [m for m in tool_msgs if m.status == "error"]
    assert len(real_results) == 1, "the real tool result must be preserved"
    assert len(synthesized) == 1, "the interrupted call must be synthesized"

    # Pairing integrity: each AIMessage's id has exactly one matching ToolMessage.
    real_result = real_results[0]
    owning_ai = next(
        (
            ai
            for ai in ai_msgs
            if any(c["id"] == real_result.tool_call_id for c in ai.tool_calls)
        ),
        None,
    )
    assert owning_ai is not None, "real result must match the second AIMessage's call"
    assert any(c["name"] == "real_call" for c in owning_ai.tool_calls), (
        "the real result must be paired with real_call, not interrupted_call"
    )


def test_normalizes_blank_id_in_invalid_tool_calls_only():
    """Regression for CodeRabbit review on PR #399: a message with blank id
    in ``invalid_tool_calls`` only (no valid ``tool_calls``) must also be
    repaired.

    langchain-openai's serializer puts ``tool_calls + invalid_tool_calls`` on
    the wire (it does NOT skip invalid calls), so a blank id on an invalid
    call reaches the provider just as readily. Before the fix the pre-pass
    skipped this case entirely because ``any_changed`` was only set inside
    the valid-call loop.
    """
    from langchain_openai.chat_models.base import _convert_message_to_dict

    message = AIMessage(content="").model_copy(
        update={
            "invalid_tool_calls": [
                {"id": "", "name": "broken_call", "args": "{bad", "error": "parse"}
            ],
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "broken_call", "arguments": "{bad"},
                    }
                ]
            },
        }
    )
    messages = [HumanMessage("hi"), message, HumanMessage("again")]

    repaired = repair_tool_history(messages)

    ai_msg = next(m for m in repaired if isinstance(m, AIMessage))
    new_id = ai_msg.invalid_tool_calls[0]["id"]
    assert isinstance(new_id, str)
    assert new_id, "invalid_tool_call id must be non-blank after repair"

    # Wire payload must not contain any blank id.
    payload = _convert_message_to_dict(ai_msg)
    wire_ids = [tc.get("id") for tc in payload.get("tool_calls", [])]
    blanks = [i for i in wire_ids if not isinstance(i, str) or not i.strip()]
    assert not blanks, f"blank tool_call_id reached the wire: {blanks!r}"
    assert new_id in wire_ids, "fresh id must be the one on the wire"


def test_blank_id_repair_yields_provider_serializable_payload():
    """End-to-end: the repaired history, when serialized by langchain-openai,
    must not put any blank tool_call_id on the wire."""
    from langchain_openai.chat_models.base import _convert_message_to_dict

    messages = [
        HumanMessage(content="run task"),
        AIMessage(
            content="thinking",
            tool_calls=[{"id": "", "name": "read_file", "args": {"path": "x"}}],
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "x"}',
                        },
                    }
                ]
            },
        ),
        ToolMessage(content="result", tool_call_id="", name="read_file"),
        HumanMessage(content="continue"),
    ]

    repaired = repair_tool_history(messages)

    wire_ids: list[str] = []
    for message in repaired:
        payload = _convert_message_to_dict(message)
        if payload.get("role") == "assistant":
            for call in payload.get("tool_calls", []):
                wire_ids.append(call.get("id"))
        elif payload.get("role") == "tool":
            wire_ids.append(payload.get("tool_call_id"))

    blanks = [cid for cid in wire_ids if not isinstance(cid, str) or not cid.strip()]
    assert not blanks, f"blank tool_call_id reached the wire: {blanks!r}"
    # pairing preserved: each assistant id has a matching tool result
    assistant_ids = [
        call.get("id")
        for m in repaired
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    tool_ids = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
    assert sorted(assistant_ids) == sorted(tool_ids)


def test_blank_id_repair_does_not_spam_warnings(caplog):
    """Across repeated repair calls (as on every model call within one run),
    blank-id normalization must not re-emit the synthesized/dropped warning.

    Deterministic ids (``_repair_{msg_idx}_v{call_idx}``) make the same blank
    call get the same id on every repair pass, so the main loop's existing
    ``warned``-set dedup suppresses the warning from the second call on --
    matching the behavior of ``test_warning_deduplicates_across_calls`` for
    non-blank interrupted calls.
    """
    messages = [
        HumanMessage("run"),
        AIMessage(
            content="",
            tool_calls=[{"id": "", "name": "execute", "args": {}}],
        ),
        HumanMessage("continue"),
    ]
    warned: set[str] = set()

    with caplog.at_level("WARNING"):
        repair_tool_history(messages, warned=warned)
        first_warnings = len(caplog.records)
        repair_tool_history(messages, warned=warned)
        second_warnings = len(caplog.records)

    assert first_warnings == 1, (
        "first call should warn once for the synthesized interrupted result"
    )
    assert second_warnings == 1, "second call must not repeat the warning"
    assert warned == {"_repair_1_v0"}, "deterministic id should be stable across calls"


def test_invalid_blank_id_not_pushed_to_pending_slots():
    """Regression for din0s review on PR #399: an invalid_tool_call's fresh id
    must NOT be pushed to ``pending_slots``, because invalid calls are never
    executed -- pushing the slot would let an orphan blank ToolMessage from
    some other call mis-pair with the invalid call's id."""
    message = AIMessage(content="").model_copy(
        update={
            "invalid_tool_calls": [
                {"id": "", "name": "broken_call", "args": "{bad", "error": "parse"}
            ],
        }
    )
    orphan_content = "orphan from somewhere else"
    messages = [
        HumanMessage("go"),
        message,
        ToolMessage(content=orphan_content, tool_call_id="", name="other"),
        HumanMessage("stop"),
    ]

    repaired = repair_tool_history(messages)

    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    # The orphan's content must NOT survive -- it must not have claimed the
    # invalid call's fresh id and paired its content with broken_call.
    assert all(orphan_content not in str(m.content) for m in tool_msgs), (
        "orphan ToolMessage content leaked into a repaired tool result -- "
        "its blank id was mis-paired with the invalid call's fresh id"
    )
    # The invalid call's fresh id is still synthesized as interrupted by the
    # main loop (invalid calls with non-blank ids are added to pending), but
    # that's the existing main-loop behavior and uses the interrupted-result
    # string, never the orphan's content.
    invalid_id = "_repair_1_i0"
    synth = [m for m in tool_msgs if m.tool_call_id == invalid_id]
    assert len(synth) == 1, "the invalid call should get a synthesized result"
    assert synth[0].status == "error"
    assert synth[0].name == "broken_call"
