"""Tests for the /exit command (ExitCommand → ctx.ui.force_quit)."""

from unittest.mock import MagicMock

from tests.conftest import run_async as _run


class TestExitCommand:
    def test_execute_calls_force_quit(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import ExitCommand

        ui = MagicMock()
        ctx = CommandContext(
            agent=None,
            thread_id="tid",
            ui=ui,
        )
        cmd = ExitCommand()
        _run(cmd.execute(ctx, []))
        ui.force_quit.assert_called_once()

    def test_aliases_registered(self):
        """/quit and /q resolve to the same ExitCommand as /exit."""
        from EvoScientist.commands.manager import manager

        assert manager.get_command("/exit").name == "/exit"
        assert manager.get_command("/quit").name == "/exit"
        assert manager.get_command("/q").name == "/exit"
