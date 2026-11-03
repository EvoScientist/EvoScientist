"""Route ``/skill-name`` input that no slash command claims to the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..paths import Workspace


def skill_slash_names(text: str, workspace: Workspace | None) -> list[str]:
    """Return the skills to label when *text* should go to the agent as a pin.

    Empty for non-slash input, a registered command, or a first word that
    names no installed skill.
    """
    stripped = text.lstrip()
    if workspace is None or not stripped.startswith("/"):
        return []
    from ..tools.skills_manager import pinned_skill_names
    from .manager import manager as cmd_manager

    if cmd_manager.resolve(stripped) is not None:
        return []
    try:
        return pinned_skill_names(stripped, workspace)
    except OSError:
        # An unreadable skills folder falls back to the unknown-command path.
        return []
