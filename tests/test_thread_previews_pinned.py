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
