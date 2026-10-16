"""Rich rendering of retry notices."""

from __future__ import annotations

import io

from rich.console import Console

from EvoScientist.stream.display import create_streaming_display, model_retry_text


def _render(**kwargs) -> str:
    out = io.StringIO()
    Console(file=out, width=120, color_system=None).print(
        create_streaming_display(response_text="done", **kwargs)
    )
    return out.getvalue()


def test_model_retry_text():
    event = {"attempt": 2, "wait_seconds": 8.0, "reason": "503: qwen is starting up"}
    assert (
        model_retry_text(event)
        == "Model request failed (503: qwen is starting up); retrying in 8 s (attempt 2)"
    )


def test_final_frame_lists_each_retry():
    text = _render(
        is_final=True,
        model_retries=[
            {"attempt": 2, "wait_seconds": 1.0, "reason": "503: a"},
            {"attempt": 3, "wait_seconds": 2.0, "reason": "502: b"},
        ],
    )
    assert "retrying in 1 s (attempt 2)" in text
    assert "retrying in 2 s (attempt 3)" in text


def test_state_display_args_carry_the_retries():
    from EvoScientist.stream.state import StreamState

    state = StreamState()
    state.handle_event(
        {
            "type": "model_retry",
            "attempt": 2,
            "wait_seconds": 1.0,
            "reason": "x",
            "budget_left_seconds": 1.0,
            "cause": "exception",
            "subagent": "",
            "instance_id": "",
            "discarded": {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []},
        }
    )
    assert state.get_display_args()["model_retries"] == state.model_retries


def test_retry_line_names_the_subagent_or_grader():
    from EvoScientist.stream.display import model_retry_text

    event = {"reason": "503: down", "wait_seconds": 8.0, "attempt": 2}
    assert model_retry_text({**event, "subagent": ""}) == (
        "Model request failed (503: down); retrying in 8 s (attempt 2)"
    )
    assert model_retry_text({**event, "subagent": "rubric_grader"}).startswith(
        "rubric_grader: Model request failed"
    )
