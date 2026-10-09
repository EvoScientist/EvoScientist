from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..paths import Workspace


class CompletionKind(StrEnum):
    """Discriminator for the kind of completion result."""

    COMMANDS = "commands"
    SUBCOMMANDS = "subcommands"
    EMPTY = "empty"


_CATEGORY_ORDER = ["Session", "Skills", "MCP", "Channels", "Model", "General"]

SKILL_CATEGORY = "Use a skill"
_SKILL_DESCRIPTION_WIDTH = 80


def _skill_description(description: str) -> str:
    text = "(skill) " + " ".join(description.split())
    if len(text) <= _SKILL_DESCRIPTION_WIDTH:
        return text
    return text[: _SKILL_DESCRIPTION_WIDTH - 1] + "…"


@dataclass(frozen=True)
class CompletionCandidate:
    """A single completion suggestion with its replacement range."""

    text: str
    description: str
    replace_start: int
    replace_end: int
    category: str = ""


@dataclass(frozen=True)
class CompletionResult:
    """The result of parsing a slash command input for completions."""

    kind: CompletionKind
    candidates: list[CompletionCandidate]


def _skill_candidates(
    prefix: str, end: int, workspace: Workspace | None
) -> list[CompletionCandidate]:
    """Installed skills matching *prefix*, minus names a command already owns."""
    if workspace is None:
        return []
    from ..message_meta import parse_skill_slashes
    from ..tools.skills_manager import installed_skill_index
    from .manager import manager as cmd_manager

    try:
        index = installed_skill_index(workspace)
    except OSError:
        # An unreadable skills folder leaves the built-in commands only.
        return []
    candidates: list[CompletionCandidate] = []
    for name, description in sorted(index.items()):
        text = f"/{name}"
        if not text.startswith(prefix) or cmd_manager.get_command(text) is not None:
            continue
        # The pin parser only accepts lowercase, digit and hyphen names.
        if parse_skill_slashes(text) != [name]:
            continue
        candidates.append(
            CompletionCandidate(
                text=text,
                description=_skill_description(description),
                replace_start=0,
                replace_end=end,
                category=SKILL_CATEGORY,
            )
        )
    return candidates


def compute_completions(
    text: str, cursor_pos: int, *, workspace: Workspace | None = None
) -> CompletionResult:
    """Parse *text* up to *cursor_pos* and return completion candidates.

    ``workspace`` is the session's workspace, forwarded to commands whose
    completions list what is installed there (e.g. ``/expert``).

    This is the shared engine used by both the Rich CLI
    (``SlashCommandCompleter``) and the TUI (``on_text_area_changed``).
    Both thin adapters only need to translate the returned candidates
    into their respective render/apply primitives.
    """
    from .manager import manager as cmd_manager

    before = text[:cursor_pos]

    if not before.startswith("/"):
        return CompletionResult(CompletionKind.EMPTY, [])

    parts = before.split()
    if not parts:
        return CompletionResult(CompletionKind.EMPTY, [])

    cmd_name = parts[0].lower()
    has_trailing_space = before.endswith(" ")

    # --- Top-level command completion ---
    if len(parts) == 1:
        prefix = before.lower().rstrip()

        # Match commands by canonical name AND aliases
        by_cat: dict[str, list[tuple[str, str]]] = {}
        for cmd in cmd_manager.get_all_commands():
            all_names = [cmd.name.lower()] + [
                a.lower() if a.startswith("/") else f"/{a.lower()}" for a in cmd.alias
            ]
            if any(n.startswith(prefix) for n in all_names):
                by_cat.setdefault(cmd.category, []).append((cmd.name, cmd.description))

        # Whether the typed prefix resolves to an exact command/alias
        exact_cmd = cmd_manager.get_command(prefix)

        if exact_cmd and not has_trailing_space:
            return CompletionResult(CompletionKind.EMPTY, [])

        if exact_cmd and has_trailing_space:
            completions = exact_cmd.get_completions([""], workspace=workspace)
            if completions:
                insert_pos = len(before)
                return CompletionResult(
                    CompletionKind.SUBCOMMANDS,
                    [
                        CompletionCandidate(
                            text=name,
                            description=desc,
                            replace_start=insert_pos,
                            replace_end=insert_pos,
                        )
                        for name, desc in completions
                    ],
                )
            return CompletionResult(CompletionKind.EMPTY, [])

        skill_candidates = (
            []
            if has_trailing_space
            else _skill_candidates(prefix, len(before), workspace)
        )
        all_matches = [v for vs in by_cat.values() for v in vs]
        if not all_matches and not skill_candidates:
            return CompletionResult(CompletionKind.EMPTY, [])

        # Build candidates ordered by category
        candidates: list[CompletionCandidate] = []
        for cat in _CATEGORY_ORDER:
            for cmd_text, desc in by_cat.get(cat, []):
                candidates.append(
                    CompletionCandidate(
                        text=cmd_text,
                        description=desc,
                        replace_start=0,
                        replace_end=len(before),
                        category=cat,
                    )
                )
        for cat, items in by_cat.items():
            if cat not in _CATEGORY_ORDER:
                for cmd_text, desc in items:
                    candidates.append(
                        CompletionCandidate(
                            text=cmd_text,
                            description=desc,
                            replace_start=0,
                            replace_end=len(before),
                            category=cat,
                        )
                    )

        # Enter in the TUI applies the first candidate, so an exact skill name
        # moves its whole group to the front, keeping the group contiguous.
        if any(c.text == prefix for c in skill_candidates):
            skills = sorted(skill_candidates, key=lambda c: c.text != prefix)
            return CompletionResult(CompletionKind.COMMANDS, skills + candidates)
        return CompletionResult(CompletionKind.COMMANDS, candidates + skill_candidates)

    # --- Subcommand / argument completion (len(parts) >= 2) ---
    cmd = cmd_manager.get_command(cmd_name)
    if cmd is None:
        return CompletionResult(CompletionKind.EMPTY, [])

    # Delegate to Command.get_completions for all depths
    tokens = parts[1:]
    if has_trailing_space:
        tokens.append("")
    completions = cmd.get_completions(tokens, workspace=workspace)
    if not completions:
        return CompletionResult(CompletionKind.EMPTY, [])

    # Compute replacement range.
    if tokens[-1]:
        # User is typing a partial — replace it
        sub_start = before.rfind(tokens[-1])
        if sub_start < 0:
            sub_start = len(before)
        replace_end = len(before)
    elif len(tokens) >= 2 and tokens[-2]:
        # Trailing space after a token. Check if the previous token is
        # a known subcommand name — if so, the completion is for the
        # NEXT argument (insert at cursor). If not, the completions
        # refine the partial (replace it).
        prev = tokens[-2]
        is_known_sub = any(sc.name == prev for sc in cmd.subcommands)
        if not is_known_sub:
            sub_start = before.rfind(prev)
            if sub_start < 0:
                sub_start = len(before)
        else:
            sub_start = len(before)
        replace_end = len(before)
    else:
        sub_start = len(before)
        replace_end = len(before)

    return CompletionResult(
        CompletionKind.SUBCOMMANDS,
        [
            CompletionCandidate(
                text=name,
                description=desc,
                replace_start=sub_start,
                replace_end=replace_end,
            )
            for name, desc in completions
        ],
    )
