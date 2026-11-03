"""Pin the skills a user names with leading ``/skill-name`` words."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepagents.middleware.skills import SkillsState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Overwrite

from ..message_meta import message_source, message_text, parse_skill_slashes


def skills_to_pin(state: Mapping[str, Any]) -> list[str]:
    """Return the skills named by the newest message when it is fresh user input.

    Only the last message counts, so a resumed (HITL) run never pins twice.
    """
    messages = state.get("messages") or []
    if not messages:
        return []
    last = messages[-1]
    if not isinstance(last, HumanMessage) or message_source(last) is not None:
        return []
    return parse_skill_slashes(message_text(last.content))


class SkillPinMiddleware(AgentMiddleware):
    """Hand ``/skill-name`` input to deepagents' ``pinned_skills``.

    deepagents matches the names against the loaded skills and appends each
    ``SKILL.md`` before the next model call; unknown names are skipped.
    Overwrites, so a pin left by a turn that never reached the model cannot
    leak into the next one.
    """

    state_schema = SkillsState

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"pinned_skills": Overwrite(skills_to_pin(state))}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return self.before_agent(state, runtime)
