"""Tests for the ask_user middleware, stream events, state, and UI helpers."""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Middleware data types
# ---------------------------------------------------------------------------


class TestDataTypes:
    """Test Question, Choice, AskUserRequest construction."""

    def test_choice_construction(self):
        from EvoScientist.middleware.ask_user import Choice

        choice: Choice = {"value": "CIFAR-10"}
        assert choice["value"] == "CIFAR-10"

    def test_question_text_construction(self):
        from EvoScientist.middleware.ask_user import Question

        q: Question = {"question": "Which dataset?", "type": "text"}
        assert q["question"] == "Which dataset?"
        assert q["type"] == "text"

    def test_question_multiple_choice_construction(self):
        from EvoScientist.middleware.ask_user import Question

        q: Question = {
            "question": "Which dataset?",
            "type": "multiple_choice",
            "choices": [{"value": "CIFAR-10"}, {"value": "ImageNet"}],
        }
        assert q["type"] == "multiple_choice"
        assert len(q["choices"]) == 2

    def test_ask_user_request_construction(self):
        from EvoScientist.middleware.ask_user import AskUserRequest

        req: AskUserRequest = {
            "type": "ask_user",
            "questions": [{"question": "test?", "type": "text"}],
            "tool_call_id": "tc_123",
        }
        assert req["type"] == "ask_user"
        assert req["tool_call_id"] == "tc_123"

    def test_ask_user_answered_construction(self):
        from EvoScientist.middleware.ask_user import AskUserAnswered

        result: AskUserAnswered = {"type": "answered", "answers": ["CIFAR-10"]}
        assert result["type"] == "answered"

    def test_ask_user_cancelled_construction(self):
        from EvoScientist.middleware.ask_user import AskUserCancelled

        result: AskUserCancelled = {"type": "cancelled"}
        assert result["type"] == "cancelled"


# ---------------------------------------------------------------------------
# BeforeValidator: _coerce_questions_list
# ---------------------------------------------------------------------------


class TestCoerceQuestionsList:
    """Test _coerce_questions_list() — the BeforeValidator that handles
    LLMs serializing the questions param as a JSON string instead of a list."""

    def test_json_string_parsed_to_list(self):
        from EvoScientist.middleware.ask_user import _coerce_questions_list

        raw = '[{"question": "Which dataset?", "type": "text"}]'
        result = _coerce_questions_list(raw)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["question"] == "Which dataset?"

    def test_list_passthrough(self):
        from EvoScientist.middleware.ask_user import _coerce_questions_list

        data = [{"question": "Q?", "type": "text"}]
        result = _coerce_questions_list(data)
        assert result is data  # same object, no copy

    def test_non_list_json_string_passthrough(self):
        from EvoScientist.middleware.ask_user import _coerce_questions_list

        # JSON object string should NOT be parsed (not a list)
        raw = '{"question": "Q?", "type": "text"}'
        result = _coerce_questions_list(raw)
        assert result == raw  # returned as-is

    def test_invalid_json_string_passthrough(self):
        from EvoScientist.middleware.ask_user import _coerce_questions_list

        raw = "not json at all"
        result = _coerce_questions_list(raw)
        assert result == raw

    def test_empty_list_json_string(self):
        from EvoScientist.middleware.ask_user import _coerce_questions_list

        result = _coerce_questions_list("[]")
        assert result == []

    def test_non_string_non_list_passthrough(self):
        from EvoScientist.middleware.ask_user import _coerce_questions_list

        assert _coerce_questions_list(42) == 42
        assert _coerce_questions_list(None) is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateQuestions:
    """Test _validate_questions()."""

    def test_empty_list_raises(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        with pytest.raises(ValueError, match="at least one question"):
            _validate_questions([])

    def test_missing_question_text_raises(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        with pytest.raises(ValueError, match="non-empty 'question' text"):
            _validate_questions([{"question": "", "type": "text"}])

    def test_wrong_type_raises(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        with pytest.raises(ValueError, match="unsupported"):
            _validate_questions([{"question": "Q?", "type": "radio"}])

    def test_multiple_choice_no_choices_raises(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        with pytest.raises(ValueError, match="non-empty 'choices' list"):
            _validate_questions([{"question": "Q?", "type": "multiple_choice"}])

    def test_text_with_choices_raises(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        with pytest.raises(ValueError, match="must not define 'choices'"):
            _validate_questions(
                [
                    {
                        "question": "Q?",
                        "type": "text",
                        "choices": [{"value": "A"}],
                    }
                ]
            )

    def test_valid_text_question(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        # Should not raise
        _validate_questions([{"question": "What dataset?", "type": "text"}])

    def test_valid_multiple_choice_question(self):
        from EvoScientist.middleware.ask_user import _validate_questions

        _validate_questions(
            [
                {
                    "question": "Which?",
                    "type": "multiple_choice",
                    "choices": [{"value": "A"}, {"value": "B"}],
                }
            ]
        )


# ---------------------------------------------------------------------------
# _parse_answers
# ---------------------------------------------------------------------------


class TestParseAnswers:
    """Test _parse_answers()."""

    def test_answered_status(self):
        from EvoScientist.middleware.ask_user import _parse_answers

        questions = [{"question": "Q1?", "type": "text"}]
        result = _parse_answers(
            {"answers": ["my answer"], "status": "answered"},
            questions,
            "tc_1",
        )
        assert hasattr(result, "update")
        msgs = result.update["messages"]
        assert len(msgs) == 1
        assert "Q: Q1?" in msgs[0].content
        assert "A: my answer" in msgs[0].content

    def test_cancelled_status(self):
        from EvoScientist.middleware.ask_user import _parse_answers

        questions = [{"question": "Q1?", "type": "text"}]
        result = _parse_answers(
            {"status": "cancelled"},
            questions,
            "tc_1",
        )
        msgs = result.update["messages"]
        assert "(cancelled)" in msgs[0].content

    def test_malformed_payload_non_dict(self):
        from EvoScientist.middleware.ask_user import _parse_answers

        questions = [{"question": "Q1?", "type": "text"}]
        result = _parse_answers("not a dict", questions, "tc_1")
        msgs = result.update["messages"]
        assert "(error:" in msgs[0].content

    def test_missing_answers_key(self):
        from EvoScientist.middleware.ask_user import _parse_answers

        questions = [{"question": "Q1?", "type": "text"}]
        result = _parse_answers(
            {"status": "answered"},
            questions,
            "tc_1",
        )
        msgs = result.update["messages"]
        assert "(error:" in msgs[0].content

    def test_unknown_status(self):
        from EvoScientist.middleware.ask_user import _parse_answers

        questions = [{"question": "Q1?", "type": "text"}]
        result = _parse_answers(
            {"answers": ["x"], "status": "unknown_status"},
            questions,
            "tc_1",
        )
        msgs = result.update["messages"]
        assert "(error:" in msgs[0].content


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class TestAskUserMiddleware:
    """Test AskUserMiddleware initialization and tool creation."""

    def test_init_creates_tool(self):
        from EvoScientist.middleware.ask_user import AskUserMiddleware

        mw = AskUserMiddleware()
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "ask_user"

    def test_system_prompt_set(self):
        from EvoScientist.middleware.ask_user import (
            ASK_USER_SYSTEM_PROMPT,
            AskUserMiddleware,
        )

        mw = AskUserMiddleware()
        assert mw.system_prompt == ASK_USER_SYSTEM_PROMPT

    def test_custom_prompt(self):
        from EvoScientist.middleware.ask_user import AskUserMiddleware

        mw = AskUserMiddleware(system_prompt="custom prompt")
        assert mw.system_prompt == "custom prompt"

    def test_system_prompt_mentions_resource_estimation(self):
        from EvoScientist.middleware.ask_user import ASK_USER_SYSTEM_PROMPT

        assert "estimation" in ASK_USER_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_timeout(self):
        from EvoScientist.middleware.ask_user import ASK_USER_SYSTEM_PROMPT

        assert "timeout" in ASK_USER_SYSTEM_PROMPT.lower()

    def test_tool_description_mentions_resource(self):
        from EvoScientist.middleware.ask_user import ASK_USER_TOOL_DESCRIPTION

        assert "resource" in ASK_USER_TOOL_DESCRIPTION.lower()


# ---------------------------------------------------------------------------
# Stream event emitter
# ---------------------------------------------------------------------------


class TestStreamEmitter:
    """Test ask_user_interrupt event creation."""

    def test_ask_user_interrupt_event_structure(self):
        from EvoScientist.stream.emitter import StreamEventEmitter

        emitter = StreamEventEmitter()
        event = emitter.ask_user_interrupt(
            interrupt_id="default",
            questions=[{"question": "Q?", "type": "text"}],
            tool_call_id="tc_1",
        )
        assert event.type == "ask_user"
        assert event.data["type"] == "ask_user"
        assert event.data["interrupt_id"] == "default"
        assert event.data["tool_call_id"] == "tc_1"
        assert len(event.data["questions"]) == 1

    def test_ask_user_interrupt_default_tool_call_id(self):
        from EvoScientist.stream.emitter import StreamEventEmitter

        emitter = StreamEventEmitter()
        event = emitter.ask_user_interrupt("ns1", [])
        assert event.data["tool_call_id"] == ""


# ---------------------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------------------


class TestStreamState:
    """Test StreamState handling of ask_user events."""

    def test_pending_ask_user_starts_none(self):
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        assert state.pending_ask_user is None

    def test_handle_ask_user_sets_pending(self):
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        event = {
            "type": "ask_user",
            "interrupt_id": "default",
            "questions": [{"question": "Q?", "type": "text"}],
            "tool_call_id": "tc_1",
        }
        result = state.handle_event(event)
        assert result == "ask_user"
        assert state.pending_ask_user is not None
        assert state.pending_ask_user["tool_call_id"] == "tc_1"

    def test_ask_user_does_not_affect_pending_interrupt(self):
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        event = {
            "type": "ask_user",
            "interrupt_id": "default",
            "questions": [],
            "tool_call_id": "tc_1",
        }
        state.handle_event(event)
        assert state.pending_interrupt is None
        assert state.pending_ask_user is not None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    """Test enable_ask_user config field."""

    def test_default_is_true(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig()
        assert cfg.enable_ask_user is True

    def test_set_to_false(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig(enable_ask_user=False)
        assert cfg.enable_ask_user is False


# ---------------------------------------------------------------------------
# Rich CLI prompt (mocking input)
# ---------------------------------------------------------------------------


class TestRichCLIPrompt:
    """Test _resolve_ask_user_prompt with mocked prompt_toolkit.prompt()."""

    _PT_PROMPT = "prompt_toolkit.prompt"

    def test_text_question_returns_answered(self):
        from EvoScientist.stream.display import _resolve_ask_user_prompt

        data = {
            "questions": [{"question": "What dataset?", "type": "text"}],
            "tool_call_id": "tc_1",
        }
        with patch(self._PT_PROMPT, side_effect=["CIFAR-10"]):
            result = _resolve_ask_user_prompt(data)
        assert result["status"] == "answered"
        assert result["answers"] == ["CIFAR-10"]

    def test_keyboard_interrupt_returns_cancelled(self):
        from EvoScientist.stream.display import _resolve_ask_user_prompt

        data = {
            "questions": [{"question": "What?", "type": "text"}],
            "tool_call_id": "tc_1",
        }
        with patch(self._PT_PROMPT, side_effect=KeyboardInterrupt):
            result = _resolve_ask_user_prompt(data)
        assert result["status"] == "cancelled"

    def test_empty_questions_returns_empty(self):
        from EvoScientist.stream.display import _resolve_ask_user_prompt

        data = {"questions": [], "tool_call_id": "tc_1"}
        result = _resolve_ask_user_prompt(data)
        assert result["status"] == "answered"
        assert result["answers"] == []

    def test_multiple_choice_letter_mapping(self):
        from EvoScientist.stream.display import _resolve_ask_user_prompt

        data = {
            "questions": [
                {
                    "question": "Which?",
                    "type": "multiple_choice",
                    "choices": [{"value": "CIFAR-10"}, {"value": "ImageNet"}],
                }
            ],
            "tool_call_id": "tc_1",
        }
        with patch(self._PT_PROMPT, side_effect=["B"]):
            result = _resolve_ask_user_prompt(data)
        assert result["status"] == "answered"
        assert result["answers"] == ["ImageNet"]


# ---------------------------------------------------------------------------
# TUI widget (basic construction)
# ---------------------------------------------------------------------------


class TestAskUserWidget:
    """Test AskUserWidget basic construction."""

    def test_widget_instantiation(self):
        from EvoScientist.cli.widgets.ask_user_widget import AskUserWidget

        questions = [{"question": "Q?", "type": "text"}]
        w = AskUserWidget(questions)
        assert w._questions == questions
        assert w._answers == []

    def test_answered_message_class_exists(self):
        from EvoScientist.cli.widgets.ask_user_widget import AskUserWidget

        msg = AskUserWidget.Answered(["answer1"])
        assert msg.answers == ["answer1"]

    def test_cancelled_message_class_exists(self):
        from EvoScientist.cli.widgets.ask_user_widget import AskUserWidget

        msg = AskUserWidget.Cancelled()
        assert isinstance(msg, AskUserWidget.Cancelled)


# ---------------------------------------------------------------------------
# Middleware __init__ exports
# ---------------------------------------------------------------------------


class TestMiddlewareExports:
    """Test that ask_user types are exported from middleware package."""

    def test_ask_user_middleware_exported(self):
        from EvoScientist.middleware import AskUserMiddleware

        assert AskUserMiddleware is not None

    def test_ask_user_request_exported(self):
        from EvoScientist.middleware import AskUserRequest

        assert AskUserRequest is not None

    def test_question_exported(self):
        from EvoScientist.middleware import Question

        assert Question is not None

    def test_choice_exported(self):
        from EvoScientist.middleware import Choice

        assert Choice is not None

    def test_widget_result_exported(self):
        from EvoScientist.middleware import AskUserWidgetResult

        assert AskUserWidgetResult is not None
