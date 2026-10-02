"""Tests for the /current command."""

from unittest.mock import MagicMock


class TestCurrentCommand:
    async def test_prints_thread_workspace_and_memory(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.general import CurrentCommand

        ui = MagicMock()
        ctx = CommandContext(
            agent=None,
            thread_id="abc123",
            ui=ui,
            workspace_dir="/tmp/ws",
        )
        await CurrentCommand().execute(ctx, [])
        # Three append_system calls: Thread, Workspace, Memory dir.
        calls = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Thread: abc123" in s for s in calls)
        assert any("Workspace:" in s for s in calls)
        assert any("Memory dir:" in s for s in calls)

    async def test_skips_workspace_when_none(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.general import CurrentCommand

        ui = MagicMock()
        ctx = CommandContext(
            agent=None,
            thread_id="abc123",
            ui=ui,
            workspace_dir=None,
        )
        await CurrentCommand().execute(ctx, [])
        calls = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Thread: abc123" in s for s in calls)
        assert not any("Workspace:" in s for s in calls)
