from __future__ import annotations

import logging
import shlex
from typing import Dict, List, Tuple

from .base import Command, CommandContext

_logger = logging.getLogger(__name__)


class CommandManager:
    """Manages slash command registration and execution."""

    def __init__(self) -> None:
        self._commands: Dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Register a command and its aliases."""
        names = [command.name] + command.alias
        for name in names:
            name = name.lower()
            if not name.startswith("/"):
                name = f"/{name}"
            self._commands[name] = command

    def get_command(self, name: str) -> Command | None:
        """Lookup a command by name."""
        return self._commands.get(name.lower())

    def list_commands(self) -> List[Tuple[str, str]]:
        """List all registered command names and descriptions."""
        seen = set()
        results = []
        for cmd in self._commands.values():
            if cmd not in seen:
                results.append((cmd.name, cmd.description))
                seen.add(cmd)
        return results

    def get_all_commands(self) -> List[Command]:
        """Return all registered command instances."""
        seen = set()
        results = []
        for cmd in self._commands.values():
            if cmd not in seen:
                results.append(cmd)
                seen.add(cmd)
        return results

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

        try:
            await cmd.execute(ctx, args)
            await ctx.ui.flush()
            return True
        except Exception as e:
            _logger.exception(f"Error executing command {cmd_name}: {e}")
            ctx.ui.append_system(f"Error executing {cmd_name}: {e}", style="red")
            await ctx.ui.flush()
            return True


# Global manager instance
manager = CommandManager()
