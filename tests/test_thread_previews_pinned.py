"""Thread previews must show what the user typed, not a pinned skill body."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.gateway.server import _thread_preview
from EvoScientist.sessions import _extract_preview

_PINNED = HumanMessage(
    "<skill name='alpha'>SKILL BODY</skill>",
    additional_kwargs={"lc_source": "pinned_skill", "skill": {"name": "alpha"}},
)


def test_server_preview_uses_the_user_message():
    messages = [HumanMessage("/alpha draft it"), _PINNED, AIMessage("done")]
    assert _thread_preview(messages) == "/alpha draft it"


def test_local_preview_skips_a_leading_pinned_message():
    messages = [_PINNED, HumanMessage("/alpha draft it"), AIMessage("done")]
    assert _extract_preview(messages) == "/alpha draft it"


_FEEDBACK = HumanMessage(
    "A grader reviewed your work against the rubric and asked for revisions.",
    additional_kwargs={"lc_source": "rubric_grader"},
)


def test_server_preview_skips_grader_feedback():
    messages = [HumanMessage("write the report"), AIMessage("draft"), _FEEDBACK]
    assert _thread_preview(messages) == "write the report"


def test_local_preview_skips_grader_feedback():
    messages = [_FEEDBACK, HumanMessage("write the report"), AIMessage("done")]
    assert _extract_preview(messages) == "write the report"


def test_previews_still_fall_back_to_a_summary_message():
    """Only pinned skills and grader feedback are skipped; a compacted thread keeps its preview."""
    summary = HumanMessage(
        "Summary of the earlier conversation.",
        additional_kwargs={"lc_source": "summarization"},
    )
    messages = [summary, AIMessage("ok")]
    assert _thread_preview(messages) == "Summary of the earlier conversation."
    assert _extract_preview(messages) == "Summary of the earlier conversation."
