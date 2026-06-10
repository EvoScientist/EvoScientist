from __future__ import annotations

import logging
import shlex
from typing import ClassVar

from .base import Command, CommandContext

_logger = logging.getLogger(__name__)


class CommandManager:
    """Manages slash command registration and execution."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Register a command and its aliases."""
        names = [command.name, *command.alias]
        for name in names:
            name = name.lower()
            if not name.startswith("/"):
                name = f"/{name}"
            self._commands[name] = command

    def get_command(self, name: str) -> Command | None:
        """Lookup a command by name."""
        return self._commands.get(name.lower())

    def resolve(self, command_str: str) -> tuple[Command, list[str]] | None:
        """Return ``(command, args)`` for the dispatch of ``command_str``.

        Uses the same parsing as :meth:`execute` so callers can inspect
        metadata (e.g. call :meth:`Command.needs_agent`) without
        re-implementing ``shlex`` quirks.
        """
        command_str = command_str.strip()
        if not command_str:
            return None
        try:
            parts = shlex.split(command_str)
        except ValueError:
            parts = command_str.split()
        if not parts:
            return None
        cmd = self.get_command(parts[0])
        if cmd is None:
            return None
        return cmd, parts[1:]

    def list_commands(self) -> list[tuple[str, str]]:
        """List all registered command names and descriptions."""
        seen = set()
        results = []
        for cmd in self._commands.values():
            if cmd not in seen:
                results.append((cmd.name, cmd.description))
                seen.add(cmd)
        return results

    def get_all_commands(self) -> list[Command]:
        """Return all registered command instances."""
        seen = set()
        results = []
        for cmd in self._commands.values():
            if cmd not in seen:
                results.append(cmd)
                seen.add(cmd)
        return results

    _CATEGORY_ORDER: ClassVar[list[str]] = [
        "Session",
        "Skills",
        "MCP",
        "Channels",
        "Model",
        "General",
    ]

    def get_completions_for_input(self, text: str) -> list[tuple[str, str, str]]:
        """Multi-stage completion resolver.

        Returns ``(completion_text, description, category)`` triples.
        Stage 0 returns top-level commands; stage 1+ delegates to
        :meth:`Command.get_completions`.
        """
        trailing_space = text.endswith(" ")
        text = text.strip()
        if not text.startswith("/"):
            return []

        parts = text.split()
        if trailing_space and parts:
            parts.append("")

        if len(parts) <= 1:
            prefix = text.lower()
            seen: set[int] = set()
            by_cat: dict[str, list[tuple[str, str]]] = {}
            for cmd in self._commands.values():
                if id(cmd) in seen:
                    continue
                # Match against canonical name and all registered aliases
                all_names = [cmd.name.lower()] + [
                    a.lower() if a.startswith("/") else f"/{a.lower()}"
                    for a in cmd.alias
                ]
                if any(n.startswith(prefix) for n in all_names):
                    seen.add(id(cmd))
                    by_cat.setdefault(cmd.category, []).append(
                        (cmd.name, cmd.description)
                    )
            # Exact match on a leaf command → hide popup
            exact_cmd = self.get_command(prefix)
            if exact_cmd and not exact_cmd.subcommands:
                all_matches = [v for vs in by_cat.values() for v in vs]
                if len(all_matches) == 1:
                    return []
            results: list[tuple[str, str, str]] = []
            for cat in self._CATEGORY_ORDER:
                for name, desc in by_cat.get(cat, []):
                    results.append((name, desc, cat))
            for cat, items in by_cat.items():
                if cat not in self._CATEGORY_ORDER:
                    for name, desc in items:
                        results.append((name, desc, cat))
            return results

        # Stage 1+: resolve command (works for aliases via get_command)
        cmd = self.get_command(parts[0])
        if cmd is None:
            return []
        return [(t, d, "") for t, d in cmd.get_completions(parts[1:])]

    async def execute(self, command_str: str, ctx: CommandContext) -> bool:
        """Parse and execute a command string.

        Returns True if a command was found and executed, False otherwise.
        """
        command_str = command_str.strip()
        if not command_str:
            return False

        try:
            parts = shlex.split(command_str)
        except ValueError:
            # Fallback for unbalanced quotes
            parts = command_str.split()

        if not parts:
            return False

        cmd_name = parts[0].lower()
        args = parts[1:]

        cmd = self.get_command(cmd_name)
        if not cmd:
            return False

        ctx.command_error = None
        try:
            await cmd.execute(ctx, args)
            await ctx.ui.flush()
            return True
        except Exception as e:
            _logger.exception(f"Error executing command {cmd_name}: {e}")
            ctx.command_error = str(e)
            ctx.ui.append_system(f"Error executing {cmd_name}: {e}", style="red")
            await ctx.ui.flush()
            return True


# Global manager instance
manager = CommandManager()
