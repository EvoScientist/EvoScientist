"""Tests for the /current command."""

from pathlib import Path
from unittest.mock import MagicMock

from EvoScientist.paths import SessionDirs
from tests.fakes import TEST_WORKSPACE


class TestCurrentCommand:
    async def test_prints_thread_workspace_and_memory(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.general import CurrentCommand

        ui = MagicMock()
        ctx = CommandContext(
            dirs=SessionDirs(TEST_WORKSPACE, Path("/tmp/ws/runs/r1")),
            agent=None,
            thread_id="abc123",
            ui=ui,
        )
        await CurrentCommand().execute(ctx, [])
        # Four append_system calls: Thread, Workspace, Run folder, Memory dir.
        calls = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Thread: abc123" in s for s in calls)
        assert any("Workspace:" in s for s in calls)
        assert any("Run folder:" in s for s in calls)
        assert any("Memory dir:" in s for s in calls)

    async def test_skips_run_folder_when_none(self):
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.general import CurrentCommand

        ui = MagicMock()
        ctx = CommandContext(
            dirs=SessionDirs(TEST_WORKSPACE),
            agent=None,
            thread_id="abc123",
            ui=ui,
        )
        await CurrentCommand().execute(ctx, [])
        calls = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Thread: abc123" in s for s in calls)
        assert any("Workspace:" in s for s in calls)
        assert not any("Run folder:" in s for s in calls)
