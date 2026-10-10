"""Tests for message_meta helpers."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, convert_to_messages

from EvoScientist.message_meta import (
    is_pinned_skill,
    message_source,
    message_text,
    parse_skill_slashes,
    pinned_skill_name,
    skill_label,
)


def _pinned(name: str = "alpha") -> HumanMessage:
    return HumanMessage(
        content=f'<skill name="{name}" path="/skills/{name}/SKILL.md">\nbody\n</skill>',
        additional_kwargs={
            "lc_source": "pinned_skill",
            "skill": {"name": name, "path": f"/skills/{name}/SKILL.md"},
        },
    )


class TestParseSkillSlashes:
    def test_single(self):
        assert parse_skill_slashes("/alpha write the intro") == ["alpha"]

    def test_several_leading(self):
        assert parse_skill_slashes("/alpha /beta-2 go") == ["alpha", "beta-2"]

    def test_duplicates_kept_once(self):
        assert parse_skill_slashes("/alpha /alpha go") == ["alpha"]

    def test_stops_at_first_plain_word(self):
        assert parse_skill_slashes("/alpha go /beta") == ["alpha"]

    def test_absolute_path_is_not_a_skill(self):
        assert parse_skill_slashes("/Users/me/data.csv summarise this") == []
        assert parse_skill_slashes("/tmp/a.txt explain") == []

    def test_uppercase_and_bad_hyphens_rejected(self):
        assert parse_skill_slashes("/Alpha go") == []
        assert parse_skill_slashes("/-alpha go") == []
        assert parse_skill_slashes("/alpha- go") == []
        assert parse_skill_slashes("/al--pha go") == []

    def test_name_longer_than_64_rejected(self):
        assert parse_skill_slashes("/" + "a" * 65 + " go") == []
        assert parse_skill_slashes("/" + "a" * 64 + " go") == ["a" * 64]

    def test_no_slash(self):
        assert parse_skill_slashes("alpha go") == []
        assert parse_skill_slashes("") == []

    def test_leading_whitespace_allowed(self):
        assert parse_skill_slashes("  /alpha go") == ["alpha"]


class TestMessageText:
    def test_string(self):
        assert message_text("hi") == "hi"

    def test_blocks(self):
        content = [
            {"type": "text", "text": "/alpha"},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "go"},
        ]
        assert message_text(content) == "/alpha go"

    def test_other(self):
        assert message_text(None) == ""


class TestPinnedSkill:
    def test_base_message(self):
        msg = _pinned()
        assert message_source(msg) == "pinned_skill"
        assert is_pinned_skill(msg)
        assert pinned_skill_name(msg) == "alpha"

    def test_dict(self):
        msg = {
            "type": "human",
            "content": "x",
            "additional_kwargs": {"lc_source": "pinned_skill", "skill": {"name": "b"}},
        }
        assert is_pinned_skill(msg)
        assert pinned_skill_name(msg) == "b"

    def test_round_trip_through_convert_to_messages(self):
        """Server threads come back as dicts and go through convert_to_messages."""
        dumped = _pinned().model_dump()
        (restored,) = convert_to_messages([dumped])
        assert is_pinned_skill(restored)
        assert pinned_skill_name(restored) == "alpha"

    def test_plain_and_summarization_messages(self):
        assert not is_pinned_skill(HumanMessage("hi"))
        summary = HumanMessage("s", additional_kwargs={"lc_source": "summarization"})
        assert message_source(summary) == "summarization"
        assert not is_pinned_skill(summary)
        assert pinned_skill_name(summary) is None

    def test_object_without_additional_kwargs(self):
        assert not is_pinned_skill(SimpleNamespace(type="human", content="hi"))

    def test_missing_name(self):
        msg = HumanMessage("x", additional_kwargs={"lc_source": "pinned_skill"})
        assert is_pinned_skill(msg)
        assert pinned_skill_name(msg) is None


def test_skill_label():
    assert skill_label("alpha") == "skill: alpha"


def test_this_run_messages_starts_at_the_latest_typed_message():
    from langchain_core.messages import AIMessage, HumanMessage

    from EvoScientist.message_meta import this_run_messages

    history = [
        HumanMessage("old task"),
        AIMessage("old answer"),
        HumanMessage("new task"),
        HumanMessage("skill body", additional_kwargs={"lc_source": "pinned_skill"}),
        AIMessage("draft"),
        HumanMessage("revise", additional_kwargs={"lc_source": "rubric_grader"}),
        AIMessage("revised"),
    ]
    assert [m.content for m in this_run_messages(history)] == [
        "new task",
        "skill body",
        "draft",
        "revise",
        "revised",
    ]


def test_this_run_messages_keeps_everything_without_a_typed_message():
    from langchain_core.messages import AIMessage

    from EvoScientist.message_meta import this_run_messages

    history = [AIMessage("only an answer")]
    assert this_run_messages(history) == history


def test_rubric_feedback_is_recognised_by_its_source_tag():
    from langchain_core.messages import HumanMessage

    from EvoScientist.message_meta import RUBRIC_FEEDBACK_LABEL, is_rubric_feedback

    feedback = HumanMessage("revise", additional_kwargs={"lc_source": "rubric_grader"})
    summary = HumanMessage("summary", additional_kwargs={"lc_source": "summarization"})

    assert is_rubric_feedback(feedback)
    assert not is_rubric_feedback(summary)
    assert not is_rubric_feedback(HumanMessage("hello"))
    assert RUBRIC_FEEDBACK_LABEL == "rubric: revision requested"


def test_harness_label_names_what_replaces_the_message():
    from langchain_core.messages import HumanMessage

    from EvoScientist.message_meta import harness_label

    pinned = HumanMessage(
        "body",
        additional_kwargs={"lc_source": "pinned_skill", "skill": {"name": "alpha"}},
    )
    unnamed = HumanMessage("body", additional_kwargs={"lc_source": "pinned_skill"})
    feedback = HumanMessage("revise", additional_kwargs={"lc_source": "rubric_grader"})
    summary = HumanMessage("summary", additional_kwargs={"lc_source": "summarization"})

    assert harness_label(pinned) == "skill: alpha"
    assert harness_label(unnamed) == "skill: unknown"
    assert harness_label(feedback) == "rubric: revision requested"
    assert harness_label(summary) is None
    assert harness_label(HumanMessage("hello")) is None
