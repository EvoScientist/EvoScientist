"""Grader verdicts and scripted models for the rubric tests."""

from __future__ import annotations

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

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


class RaisingFileBackend(FilesystemBackend):
    """A workspace where reading one path raises out of the file tools.

    A symlink inside the workspace pointing outside it makes upstream
    ``_resolve_path`` raise ``ValueError``, and ``read`` catches only
    ``OSError`` / ``RuntimeError``, so the error escapes ``read_file`` (and
    ``ls`` for the same path). Reproduced here by name, so the test needs no
    symlink privileges and the same workspace can be read by every other path.
    """

    RAISES_FOR = "/link.md"

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        if str(file_path).endswith(self.RAISES_FOR):
            raise ValueError(f"Path:{file_path} outside root directory")
        return super().read(file_path, offset, limit)


class MainModel(GenericFakeChatModel):
    """Replies with the scripted messages and accepts any tools."""

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedGrader(BaseChatModel):
    """Answers each grading pass with the next scripted verdict (or raises it).

    ``preamble`` messages are replayed before the first verdict, so a script
    can call a tool (``read_file``, ``ls``) and read its result before grading.
    """

    verdicts: list
    preamble: list = Field(default_factory=list)
    calls: int = 0
    last_payload: str = ""
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-grader"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if self.preamble:
            return ChatResult(
                generations=[ChatGeneration(message=self.preamble.pop(0))]
            )
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
