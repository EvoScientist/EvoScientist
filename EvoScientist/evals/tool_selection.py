"""Offline scoring for tool-selection events.

The runtime already exposes each selected tool set as a ``tool_selection``
event in its ``stream-json`` protocol. This module consumes that public event
shape directly so evaluation stays independent of providers and API keys.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def _tool_names(value: object, *, context: str) -> frozenset[str]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{context} must be an iterable of tool-name strings")

    names: set[str] = set()
    for index, name in enumerate(value):
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context}[{index}] must be a non-empty tool-name string")
        names.add(name)
    return frozenset(names)


def _event_tool_names(value: object, *, event_index: int) -> frozenset[str]:
    context = f"tool_selection event at index {event_index} tools"
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of tool-name strings")
    return _tool_names(value, context=context)


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


@dataclass(frozen=True, slots=True)
class ToolSelectionExpectation:
    """Expected tools for one recorded selection, identified by ``case_id``."""

    case_id: str
    expected_tools: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        object.__setattr__(
            self,
            "expected_tools",
            _tool_names(
                self.expected_tools,
                context=f"expectation {self.case_id!r} expected_tools",
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolSelectionScore:
    """Set-based score for one expected tool selection."""

    case_id: str
    selected_tools: frozenset[str]
    expected_tools: frozenset[str]
    true_positives: frozenset[str]
    false_positives: frozenset[str]
    false_negatives: frozenset[str]
    precision: float
    recall: float
    f1: float
    exact_match: bool


@dataclass(frozen=True, slots=True)
class ToolSelectionReport:
    """Per-case scores plus macro and micro aggregates."""

    scores: tuple[ToolSelectionScore, ...]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    exact_match_rate: float
    micro_precision: float
    micro_recall: float
    micro_f1: float

    @property
    def case_count(self) -> int:
        return len(self.scores)


def _score(
    selected: frozenset[str], expectation: ToolSelectionExpectation
) -> ToolSelectionScore:
    expected = expectation.expected_tools
    true_positives = selected & expected
    false_positives = selected - expected
    false_negatives = expected - selected

    # Only empty/empty is a perfect abstention. If exactly one side is empty,
    # the undefined metric is zero so macro aggregates do not reward a miss.
    both_empty = not selected and not expected
    precision = _ratio(len(true_positives), len(selected), empty=float(both_empty))
    recall = _ratio(len(true_positives), len(expected), empty=float(both_empty))
    f1 = _ratio(2 * precision * recall, precision + recall, empty=0.0)

    return ToolSelectionScore(
        case_id=expectation.case_id,
        selected_tools=selected,
        expected_tools=expected,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=selected == expected,
    )


def _report(scores: tuple[ToolSelectionScore, ...]) -> ToolSelectionReport:
    if not scores:
        return ToolSelectionReport(scores, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    case_count = len(scores)
    true_positives = sum(len(score.true_positives) for score in scores)
    false_positives = sum(len(score.false_positives) for score in scores)
    false_negatives = sum(len(score.false_negatives) for score in scores)
    all_empty = not true_positives and not false_positives and not false_negatives
    micro_precision = _ratio(
        true_positives,
        true_positives + false_positives,
        empty=float(all_empty),
    )
    micro_recall = _ratio(
        true_positives,
        true_positives + false_negatives,
        empty=float(all_empty),
    )
    micro_f1 = _ratio(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
        empty=0.0,
    )

    return ToolSelectionReport(
        scores=scores,
        macro_precision=sum(score.precision for score in scores) / case_count,
        macro_recall=sum(score.recall for score in scores) / case_count,
        macro_f1=sum(score.f1 for score in scores) / case_count,
        exact_match_rate=sum(score.exact_match for score in scores) / case_count,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
    )


def replay_tool_selections(
    events: Iterable[Mapping[str, Any]],
    expectations: Iterable[ToolSelectionExpectation],
) -> ToolSelectionReport:
    """Replay recorded ``tool_selection`` events against ordered expectations.

    Unrecognized event types are ignored for forward compatibility with the
    stream protocol. Relevant events are validated strictly and errors identify
    their zero-based event position. Duplicate tool names use set semantics.
    """

    expected = tuple(expectations)
    seen_case_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for expectation in expected:
        if expectation.case_id in seen_case_ids:
            duplicate_ids.add(expectation.case_id)
        seen_case_ids.add(expectation.case_id)
    if duplicate_ids:
        duplicate_list = ", ".join(sorted(repr(case_id) for case_id in duplicate_ids))
        raise ValueError(f"duplicate expectation case_id values: {duplicate_list}")

    selections: list[tuple[int, frozenset[str]]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"event at index {event_index} must be a mapping")
        if event.get("type") != "tool_selection":
            continue
        if "tools" not in event:
            raise ValueError(
                f"tool_selection event at index {event_index} is missing 'tools'"
            )
        selections.append(
            (
                event_index,
                _event_tool_names(event["tools"], event_index=event_index),
            )
        )

    if len(selections) < len(expected):
        missing = expected[len(selections)]
        raise ValueError(
            "tool_selection count mismatch: "
            f"expected {len(expected)}, found {len(selections)}; "
            f"missing selection for case {missing.case_id!r}"
        )
    if len(selections) > len(expected):
        extra_event_index = selections[len(expected)][0]
        raise ValueError(
            "tool_selection count mismatch: "
            f"expected {len(expected)}, found {len(selections)}; "
            f"first extra selection is at event index {extra_event_index}"
        )

    scores = tuple(
        _score(selected, expectation)
        for (_, selected), expectation in zip(selections, expected, strict=True)
    )
    return _report(scores)
