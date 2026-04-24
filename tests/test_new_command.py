"""Tests for the /new command."""

from unittest.mock import MagicMock

from tests.conftest import run_async as _run


class TestNewCommand:
    def test_execute_calls_start_new_session(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ctx = CommandContext(
            agent=None,
            thread_id="old-tid",
            ui=ui,
            workspace_dir="/old/ws",
        )
        _run(NewCommand().execute(ctx, []))
        ui.start_new_session.assert_called_once()

    def test_requires_agent_false(self):
        from EvoScientist.commands.implementation.session import NewCommand

        assert NewCommand().requires_agent is False

    def test_no_agent_access(self):
        """Command body must not touch ctx.agent (it's still loading)."""
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ctx = CommandContext(agent=None, thread_id="tid", ui=ui)
        # No AttributeError even though ctx.agent is None
        _run(NewCommand().execute(ctx, []))
