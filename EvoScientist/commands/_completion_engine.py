from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompletionKind(StrEnum):
    """Discriminator for the kind of completion result."""

    COMMANDS = "commands"
    SUBCOMMANDS = "subcommands"
    EMPTY = "empty"


_CATEGORY_ORDER = ["Session", "Skills", "MCP", "Channels", "Model", "General"]


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
            completions = exact_cmd.get_completions([""])
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

        all_matches = [v for vs in by_cat.values() for v in vs]
        if not all_matches:
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

        return CompletionResult(CompletionKind.COMMANDS, candidates)

    # --- Subcommand / argument completion (len(parts) >= 2) ---
    cmd = cmd_manager.get_command(cmd_name)
    if cmd is None:
        return CompletionResult(CompletionKind.EMPTY, [])

    # Delegate to Command.get_completions for all depths
    tokens = parts[1:]
    if has_trailing_space:
        tokens.append("")
    completions = cmd.get_completions(tokens)
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
