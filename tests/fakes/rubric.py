"""Grader verdicts and scripted models for the rubric tests."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

FAIL = {
    "result": "needs_revision",
    "explanation": "not yet",
    "criteria": [{"name": "c1", "passed": False, "gap": "missing"}],
}

PASS = {
    "result": "satisfied",
    "explanation": "ok",
    "criteria": [{"name": "c1", "passed": True}],
}


class MainModel(GenericFakeChatModel):
    """Replies with the scripted messages and accepts any tools."""

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedGrader(BaseChatModel):
    """Answers each grading pass with the next scripted verdict (or raises it)."""

    verdicts: list
    calls: int = 0
    last_payload: str = ""

    @property
    def _llm_type(self) -> str:
        return "scripted-grader"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        item = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        self.last_payload = next(
            (str(m.content) for m in reversed(messages) if m.type == "human"), ""
        )
        if isinstance(item, Exception):
            raise item
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "GraderResponse",
                    "args": item,
                    "id": f"call-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])
