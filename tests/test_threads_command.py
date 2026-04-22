"""Tests for the /threads command."""

from unittest.mock import AsyncMock, MagicMock, patch

from rich.table import Table

from tests.conftest import run_async as _run


def _ctx(**overrides):
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = overrides.pop("supports_interactive", True)
    return CommandContext(
        agent=None,
        thread_id=overrides.pop("thread_id", "tid-1"),
        ui=ui,
        workspace_dir=overrides.pop("workspace_dir", "/ws"),
    ), ui


class TestThreadsCommand:
    def test_empty_list_prints_message(self):
        from EvoScientist.commands.implementation.session import ThreadsCommand

        ctx, ui = _ctx()
        with patch(
            "EvoScientist.sessions.list_threads",
            new=AsyncMock(return_value=[]),
        ):
            _run(ThreadsCommand().execute(ctx, []))
        ui.append_system.assert_called_once()
        assert "No saved sessions" in ui.append_system.call_args.args[0]

    def test_renders_table_with_current_marker(self):
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
        with patch(
            "EvoScientist.sessions.list_threads",
            new=AsyncMock(return_value=threads),
        ):
            _run(ThreadsCommand().execute(ctx, []))
        ui.mount_renderable.assert_called_once()
        table = ui.mount_renderable.call_args.args[0]
        assert isinstance(table, Table)
        # Footer hint (ported from the pre-migration inline /threads handler)
        footer = ui.append_system.call_args.args[0]
        assert "/resume" in footer
        assert "/delete" in footer
        assert "/new" in footer

    def test_footer_hint_suppressed_in_channel_mode(self):
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
        with patch(
            "EvoScientist.sessions.list_threads",
            new=AsyncMock(return_value=threads),
        ):
            _run(ThreadsCommand().execute(ctx, []))
        ui.append_system.assert_not_called()

    def test_channel_mode_drops_model_column(self):
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
        with patch(
            "EvoScientist.sessions.list_threads",
            new=AsyncMock(return_value=threads),
        ):
            _run(ThreadsCommand().execute(ctx, []))
        # Channel mode: no Model column. 4 columns: ID, Preview, Msgs, Last Used.
        table = ui.mount_renderable.call_args.args[0]
        column_headers = [col.header for col in table.columns]
        assert "Model" not in column_headers
