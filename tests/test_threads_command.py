"""Tests for the /threads command."""

from pathlib import PurePath
from unittest.mock import MagicMock

import pytest
from rich.table import Table

from EvoScientist.paths import SessionDirs
from tests.fakes import TEST_WORKSPACE, FakeGraphGateway, FakeThreadStore


def _ctx(**overrides):
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = overrides.pop("supports_interactive", True)
    store = overrides.pop("thread_store", FakeThreadStore())
    return CommandContext(
        dirs=SessionDirs(TEST_WORKSPACE),
        agent=None,
        thread_id=overrides.pop("thread_id", "tid-1"),
        ui=ui,
        graph_gateway=FakeGraphGateway(thread_store=store),
    ), ui


class TestThreadsCommand:
    async def test_empty_list_prints_message(self):
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx()
        await ThreadsCommand().execute(ctx, [])
        ui.append_system.assert_called_once()
        assert "No saved sessions" in ui.append_system.call_args.args[0]

    async def test_renders_table_with_current_marker(self):
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx(thread_id="current")
        threads = [
            {
                "thread_id": "current",
                "preview": "foo",
                "message_count": 5,
                "model": "claude",
                "updated_at": None,
            },
            {
                "thread_id": "other",
                "preview": "bar",
                "message_count": 2,
                "model": None,
                "updated_at": None,
            },
        ]
        store = FakeThreadStore(threads=threads)
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        await ThreadsCommand().execute(ctx, [])
        ui.mount_renderable.assert_called_once()
        table = ui.mount_renderable.call_args.args[0]
        assert isinstance(table, Table)
        # Footer hint (ported from the pre-migration inline /threads handler)
        footer = ui.append_system.call_args.args[0]
        assert "/resume" in footer
        assert "/delete" in footer
        assert "/new" in footer

    async def test_footer_hint_suppressed_in_channel_mode(self):
        """Channels don't get the footer — keeps outbound text short."""
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx(supports_interactive=False)
        threads = [
            {
                "thread_id": "t",
                "preview": "p",
                "message_count": 1,
                "model": "m",
                "updated_at": None,
            }
        ]
        store = FakeThreadStore(threads=threads)
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        await ThreadsCommand().execute(ctx, [])
        ui.append_system.assert_not_called()

    async def test_channel_mode_drops_model_column(self):
        """Non-interactive (channel) UIs get a narrower table."""
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx(supports_interactive=False)
        threads = [
            {
                "thread_id": "t",
                "preview": "p",
                "message_count": 1,
                "model": "m",
                "updated_at": None,
            }
        ]
        store = FakeThreadStore(threads=threads)
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        await ThreadsCommand().execute(ctx, [])
        # Channel mode: no Model column. 5 columns: ID, Preview, Msgs, Workspace,
        # Last Used.
        table = ui.mount_renderable.call_args.args[0]
        column_headers = [col.header for col in table.columns]
        assert "Model" not in column_headers

    @staticmethod
    def _workspace_cells(ui) -> list[str]:
        table = ui.mount_renderable.call_args.args[0]
        column = next(col for col in table.columns if col.header == "Workspace")
        return list(column.cells)

    @staticmethod
    def _workspace_threads(tmp_path) -> list[dict]:
        project = (tmp_path / "projA").as_posix()
        return [
            {"thread_id": "root", "workspace_dir": project},
            {
                "thread_id": "run",
                "workspace_dir": project,
                "run_dir": f"{project}/runs/20261001_215846",
            },
            {"thread_id": "old"},  # Stored before workspaces were recorded.
            {"thread_id": "bad", "workspace_dir": 123},  # Not a folder at all.
        ]

    async def test_workspace_column_shows_the_work_folder(self, tmp_path):
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx()
        store = FakeThreadStore(threads=self._workspace_threads(tmp_path))
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        await ThreadsCommand().execute(ctx, [])

        assert self._workspace_cells(ui) == [
            "projA",
            "projA/runs/20261001_215846",
            "",
            "",
        ]

    async def test_channel_mode_shows_only_the_folder_name(self, tmp_path):
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx(supports_interactive=False)
        store = FakeThreadStore(threads=self._workspace_threads(tmp_path))
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        await ThreadsCommand().execute(ctx, [])

        assert self._workspace_cells(ui) == ["projA", "20261001_215846", "", ""]

    @pytest.mark.parametrize("supports_interactive", [True, False])
    async def test_workspace_at_the_filesystem_root(
        self, tmp_path, supports_interactive
    ):
        """A root folder (``/`` in a container) has no name, so the label
        falls back to the root itself instead of ``.`` or an empty cell."""
        from EvoScientist.commands.implementation.session import ThreadsCommand
        from EvoScientist.paths import Workspace

        root = Workspace(tmp_path.anchor).root.as_posix()
        run = PurePath(root, "runs", "20261001_215846").as_posix()
        threads = [
            {"thread_id": "root", "workspace_dir": root},
            {"thread_id": "run", "workspace_dir": root, "run_dir": run},
        ]
        ctx, ui = _ctx(supports_interactive=supports_interactive)
        ctx.graph_gateway = FakeGraphGateway(
            thread_store=FakeThreadStore(threads=threads)
        )
        await ThreadsCommand().execute(ctx, [])

        expected_run = run if supports_interactive else "20261001_215846"
        assert self._workspace_cells(ui) == [root, expected_run]

    async def test_id_stays_whole_next_to_a_long_workspace(self, tmp_path):
        from rich.console import Console

        from EvoScientist.commands.implementation.session import ThreadsCommand

        project = (tmp_path / "cifar-resnet-ablations").as_posix()
        threads = [
            {
                "thread_id": "a1b2c3d4-5678",
                "preview": "Reproduce the baseline on CIFAR-10 with ResNet-18",
                "message_count": 140,
                "model": "claude-sonnet-4-6",
                "workspace_dir": project,
                "run_dir": f"{project}/runs/20261001_215846",
            }
        ]
        ctx, ui = _ctx(thread_id="a1b2c3d4-5678")
        ctx.graph_gateway = FakeGraphGateway(
            thread_store=FakeThreadStore(threads=threads)
        )
        await ThreadsCommand().execute(ctx, [])

        console = Console(width=100)
        with console.capture() as capture:
            console.print(ui.mount_renderable.call_args.args[0])
        assert "a1b2c3d4 *" in capture.get()
