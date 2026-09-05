"""Tests for offline replay and scoring of tool-selection events."""

import pytest

from EvoScientist.evals import (
    ToolSelectionExpectation,
    replay_tool_selections,
)


def _expectation(case_id: str, *tools: str) -> ToolSelectionExpectation:
    return ToolSelectionExpectation(case_id, frozenset(tools))


def test_scores_perfect_missing_extra_and_disjoint_selections():
    events = [
        {"type": "tool_selection", "tools": ["read_file", "search"]},
        {"type": "tool_selection", "tools": ["read_file"]},
        {"type": "tool_selection", "tools": ["read_file", "execute"]},
        {"type": "tool_selection", "tools": ["write_file"]},
    ]
    expectations = [
        _expectation("perfect", "read_file", "search"),
        _expectation("missing", "read_file", "search"),
        _expectation("extra", "read_file"),
        _expectation("disjoint", "search"),
    ]

    report = replay_tool_selections(events, expectations)

    assert report.case_count == 4
    perfect, missing, extra, disjoint = report.scores
    assert perfect.exact_match
    assert perfect.precision == perfect.recall == perfect.f1 == 1.0
    assert missing.false_negatives == frozenset({"search"})
    assert missing.precision == 1.0
    assert missing.recall == 0.5
    assert missing.f1 == pytest.approx(2 / 3)
    assert extra.false_positives == frozenset({"execute"})
    assert extra.precision == 0.5
    assert extra.recall == 1.0
    assert disjoint.precision == disjoint.recall == disjoint.f1 == 0.0


def test_reports_macro_and_micro_aggregates():
    report = replay_tool_selections(
        [
            {"type": "tool_selection", "tools": ["a", "b"]},
            {"type": "tool_selection", "tools": ["c"]},
        ],
        [_expectation("one", "a"), _expectation("two", "c", "d")],
    )

    assert report.macro_precision == 0.75
    assert report.macro_recall == 0.75
    assert report.macro_f1 == pytest.approx(2 / 3)
    assert report.exact_match_rate == 0.0
    assert report.micro_precision == pytest.approx(2 / 3)
    assert report.micro_recall == pytest.approx(2 / 3)
    assert report.micro_f1 == pytest.approx(2 / 3)


def test_empty_expectations_follow_explicit_zero_denominator_policy():
    report = replay_tool_selections(
        [
            {"type": "tool_selection", "tools": []},
            {"type": "tool_selection", "tools": ["unexpected"]},
            {"type": "tool_selection", "tools": []},
        ],
        [
            _expectation("empty"),
            _expectation("false-positive"),
            _expectation("false-negative", "missing"),
        ],
    )

    empty, false_positive, false_negative = report.scores
    assert empty.exact_match
    assert empty.precision == empty.recall == empty.f1 == 1.0
    assert not false_positive.exact_match
    assert false_positive.precision == 0.0
    assert false_positive.recall == 0.0
    assert false_positive.f1 == 0.0
    assert false_negative.precision == 0.0
    assert false_negative.recall == 0.0
    assert false_negative.f1 == 0.0


def test_replay_ignores_unknown_events_and_deduplicates_tool_names():
    report = replay_tool_selections(
        [
            {"type": "thinking", "content": "select tools"},
            {"type": "future_event", "payload": 1},
            {"type": "tool_selection", "tools": ["search", "search"]},
            {"type": "done", "response": "ok"},
        ],
        [ToolSelectionExpectation("deduplicated", frozenset({"search"}))],
    )

    assert report.scores[0].selected_tools == frozenset({"search"})
    assert report.scores[0].exact_match


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([{"type": "tool_selection"}], "index 0 is missing 'tools'"),
        (
            [{"type": "tool_selection", "tools": "search"}],
            "index 0 tools must be a list",
        ),
        (
            [{"type": "tool_selection", "tools": ("search",)}],
            "index 0 tools must be a list",
        ),
        (
            [{"type": "tool_selection", "tools": {"search": True}}],
            "index 0 tools must be a list",
        ),
        (
            [{"type": "tool_selection", "tools": ["search", 7]}],
            "index 0 tools\\[1\\] must be a non-empty",
        ),
        (["not-an-event"], "event at index 0 must be a mapping"),
    ],
)
def test_replay_rejects_malformed_events_with_event_position(events, message):
    with pytest.raises(ValueError, match=message):
        replay_tool_selections(events, [_expectation("case", "search")])


@pytest.mark.parametrize(
    ("events", "expectations", "message"),
    [
        (
            [],
            [_expectation("missing", "search")],
            "expected 1, found 0; missing selection for case 'missing'",
        ),
        (
            [
                {"type": "thinking"},
                {"type": "tool_selection", "tools": ["search"]},
            ],
            [],
            "expected 0, found 1; first extra selection is at event index 1",
        ),
    ],
)
def test_replay_rejects_selection_count_mismatches(events, expectations, message):
    with pytest.raises(ValueError, match=message):
        replay_tool_selections(events, expectations)


def test_expectations_validate_case_and_tool_names():
    with pytest.raises(ValueError, match="case_id must be a non-empty string"):
        ToolSelectionExpectation("", frozenset({"search"}))
    with pytest.raises(ValueError, match="expectation 'bad' expected_tools"):
        ToolSelectionExpectation("bad", frozenset({""}))


def test_replay_rejects_duplicate_case_ids():
    with pytest.raises(ValueError, match=r"duplicate expectation case_id.*'same'"):
        replay_tool_selections(
            [
                {"type": "tool_selection", "tools": []},
                {"type": "tool_selection", "tools": []},
            ],
            [_expectation("same"), _expectation("same")],
        )
