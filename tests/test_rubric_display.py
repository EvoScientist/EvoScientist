"""Rich rendering of acceptance-grading results."""

from __future__ import annotations

import io

from rich.console import Console

from EvoScientist.stream.display import create_streaming_display


def _render(**kwargs) -> str:
    out = io.StringIO()
    Console(file=out, width=120, color_system=None).print(
        create_streaming_display(response_text="done", **kwargs)
    )
    return out.getvalue()


def test_failed_pass_lists_each_failing_criterion():
    text = _render(
        is_final=True,
        rubric_evaluations=[
            {
                "result": "needs_revision",
                "explanation": "one gap",
                "criteria": [
                    {"name": "has table", "passed": True},
                    {"name": "has figure", "passed": False, "gap": "figures/ is empty"},
                ],
            }
        ],
    )
    assert "Acceptance criteria not yet satisfied: one gap" in text
    assert "- has figure: figures/ is empty" in text
    assert "has table" not in text


def test_satisfied_and_limit_lines():
    text = _render(
        is_final=True,
        rubric_evaluations=[
            {"result": "max_iterations_reached", "explanation": "", "criteria": []},
            {"result": "satisfied", "explanation": "all good", "criteria": []},
        ],
    )
    assert "Acceptance criteria not satisfied (pass limit reached)" in text
    assert "Acceptance criteria satisfied" in text


def test_model_written_markup_is_printed_literally():
    text = _render(
        is_final=True,
        rubric_evaluations=[
            {
                "result": "needs_revision",
                "explanation": "[red]boom[/red] [/unclosed",
                "criteria": [{"name": "[bold]x", "passed": False, "gap": "[/]"}],
            }
        ],
    )
    assert "[red]boom[/red] [/unclosed" in text
    assert "- [bold]x: [/]" in text


def test_unknown_result_still_prints_a_line():
    text = _render(
        is_final=True,
        rubric_evaluations=[{"result": "some_new_verdict", "explanation": "why"}],
    )
    assert "Acceptance criteria check ended: why" in text


def test_checking_shows_a_status_line_while_streaming():
    text = _render(rubric_checking=True)
    assert "Checking acceptance criteria" in text


def test_no_rubric_adds_nothing():
    assert "cceptance criteria" not in _render(is_final=True)


def test_final_frame_never_shows_the_checking_line():
    """A run stopped mid-grading must not leave a frozen spinner on screen."""
    assert "Checking acceptance criteria" not in _render(
        is_final=True, rubric_checking=True
    )
