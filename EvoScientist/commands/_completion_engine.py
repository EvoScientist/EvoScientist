from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompletionKind(StrEnum):
    """Discriminator for the kind of completion result."""

    COMMANDS = "commands"
    SUBCOMMANDS = "subcommands"
    EMPTY = "empty"


@dataclass(frozen=True)
class CompletionCandidate:
    """A single completion suggestion with its replacement range."""

    text: str
    description: str
    replace_start: int
    replace_end: int


@dataclass(frozen=True)
class CompletionResult:
    """The result of parsing a slash command input for completions."""

    kind: CompletionKind
    candidates: list[CompletionCandidate]


def compute_completions(text: str, cursor_pos: int) -> CompletionResult:
    """Parse *text* up to *cursor_pos* and return completion candidates.

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
        commands = cmd_manager.list_commands()
        matches = [(c, d) for c, d in commands if c.startswith(prefix)]

        # Whether the typed prefix is itself a complete command name. Checked
        # via membership rather than ``len(matches) == 1`` so a command whose
        # name is a strict prefix of another (e.g. ``/model`` vs
        # ``/model-fallback``) still counts as an exact match — otherwise the
        # popup never hides on the shorter command and Enter completes instead
        # of submitting it.
        exact = any(c == prefix for c, _ in matches)

        # Exact match with no trailing space → hide
        if exact and not has_trailing_space:
            return CompletionResult(CompletionKind.EMPTY, [])

        # Exact match + trailing space + has subcommands → show subcommands
        if exact and has_trailing_space:
            if not cmd_manager.get_subcommands(prefix):
                return CompletionResult(CompletionKind.EMPTY, [])
            sub_items = cmd_manager.list_subcommands(cmd_name)
            if sub_items:
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
                        for name, desc in sub_items
                    ],
                )

        if matches:
            return CompletionResult(
                CompletionKind.COMMANDS,
                [
                    CompletionCandidate(
                        text=cmd,
                        description=desc,
                        replace_start=0,
                        replace_end=len(before),
                    )
                    for cmd, desc in matches
                ],
            )

        return CompletionResult(CompletionKind.EMPTY, [])

    # --- Subcommand completion (len(parts) >= 2) ---
    if len(parts) >= 3:
        return CompletionResult(CompletionKind.EMPTY, [])

    cmd = cmd_manager.get_command(cmd_name)
    if cmd is None or not cmd.subcommands:
        return CompletionResult(CompletionKind.EMPTY, [])

    sub_prefix = parts[1].lower()
    sub_matches = [
        (name, desc)
        for name, desc in cmd_manager.list_subcommands(cmd_name)
        if name.startswith(sub_prefix)
    ]

    if not sub_matches:
        return CompletionResult(CompletionKind.EMPTY, [])

    # Exact match (sub_prefix == name) → the subcommand is already
    # complete.  Hide regardless of trailing space — the user is done
    # with the subcommand and ready to type arguments.  Without this
    # guard, Tab on ``/mcp list`` re-inserts ``list`` and ``/mcp list ``
    # oscillates between adding and removing the trailing space.
    if len(sub_matches) == 1 and sub_matches[0][0] == sub_prefix:
        return CompletionResult(CompletionKind.EMPTY, [])

    sub_start = before.rfind(sub_prefix) if sub_prefix else len(before)
    if sub_start < 0:
        sub_start = len(parts[0]) + 1
    # When the user has typed a trailing space, exclude it from the
    # replace range — the apply step preserves it via
    # ``current[replace_end:]`` so we don't double up the space.
    replace_end = len(before) - 1 if has_trailing_space else len(before)

    return CompletionResult(
        CompletionKind.SUBCOMMANDS,
        [
            CompletionCandidate(
                text=name,
                description=desc,
                replace_start=sub_start,
                replace_end=replace_end,
            )
            for name, desc in sub_matches
        ],
    )
