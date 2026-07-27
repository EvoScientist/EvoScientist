"""Tests for the /new command."""

from unittest.mock import AsyncMock, MagicMock


class TestNewCommand:
    async def test_execute_calls_start_new_session(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ui.start_new_session = AsyncMock()
        ctx = CommandContext(
            agent=None,
            thread_id="old-tid",
            ui=ui,
            workspace_dir="/old/ws",
        )
        await NewCommand().execute(ctx, [])
        ui.start_new_session.assert_awaited_once()

    def test_requires_agent_false(self):
        from EvoScientist.commands.implementation.session import NewCommand

        assert NewCommand().requires_agent is False

    async def test_no_agent_access(self):
        """Command body must not touch ctx.agent (it's still loading)."""
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ui.start_new_session = AsyncMock()
        ctx = CommandContext(agent=None, thread_id="tid", ui=ui)
        # No AttributeError even though ctx.agent is None
        await NewCommand().execute(ctx, [])

    async def test_clears_invited_experts_and_announces(self):
        """/new dismisses invited experts uniformly with channel-shutdown clear."""
        from EvoScientist.commands.base import ChannelRuntime, CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ui.start_new_session = AsyncMock()
        runtime = ChannelRuntime()
        runtime.active_teams = ["idea-brainstorm"]
        ctx = CommandContext(
            agent=None,
            thread_id="tid",
            ui=ui,
            channel_runtime=runtime,
        )
        await NewCommand().execute(ctx, [])
        assert runtime.active_teams == []
        messages = [call.args[0] for call in ui.append_system.call_args_list]
        assert any(
            "Dismissed experts on new session: idea-brainstorm" in msg
            for msg in messages
        )

    async def test_no_announcement_when_no_experts_invited(self):
        """No noise on ``/new`` when the invite list is already empty."""
        from EvoScientist.commands.base import ChannelRuntime, CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ui.start_new_session = AsyncMock()
        runtime = ChannelRuntime()
        ctx = CommandContext(
            agent=None,
            thread_id="tid",
            ui=ui,
            channel_runtime=runtime,
        )
        await NewCommand().execute(ctx, [])
        ui.append_system.assert_not_called()

    async def test_no_announcement_without_channel_runtime(self):
        """Runs cleanly when no ChannelRuntime is attached."""
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import NewCommand

        ui = MagicMock()
        ui.start_new_session = AsyncMock()
        ctx = CommandContext(agent=None, thread_id="tid", ui=ui, channel_runtime=None)
        await NewCommand().execute(ctx, [])
        ui.append_system.assert_not_called()
