"""Re-read the skills library at the start of every main-agent turn."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.skills import SkillsState
from langchain.agents.middleware.types import AgentMiddleware


class SkillsReloadMiddleware(AgentMiddleware):
    """Make deepagents rescan the skills library at the start of every main-agent turn.

    Runs before SkillsMiddleware.before_agent, so any turn sees skills changed since the
    last one, however that turn ended; an unchanged library renders the same prompt.
    """

    state_schema = SkillsState

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"skills_metadata": None}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"skills_metadata": None}
