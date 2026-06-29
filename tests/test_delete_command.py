"""Tests for the /delete command."""

from unittest.mock import AsyncMock, MagicMock

from tests.conftest import run_async as _run
from tests.fakes import FakeGraphGateway, FakeThreadStore


def _ctx(thread_id="current", thread_store=None):
    from EvoScientist.commands.base import CommandContext

    store = thread_store or FakeThreadStore()
    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(
        agent=None,
        thread_id=thread_id,
        ui=ui,
        graph_gateway=FakeGraphGateway(thread_store=store),
    ), ui


class TestDeleteCommand:
    def test_refuses_to_delete_current(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        thread_store = FakeThreadStore(resolved_thread_id="current", deleted=True)
        ctx, ui = _ctx(thread_id="current", thread_store=thread_store)
        _run(DeleteCommand().execute(ctx, ["current"]))
        assert ("delete_thread", "current") not in thread_store.calls
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Cannot delete the current session" in m for m in msgs)

    def test_happy_path_success(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx(
            thread_id="current",
            thread_store=FakeThreadStore(
                resolved_thread_id="other-thread",
                deleted=True,
            ),
        )
        _run(DeleteCommand().execute(ctx, ["other-thread"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Deleted session other-thread" in m for m in msgs)

    def test_not_found(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        _run(DeleteCommand().execute(ctx, ["missing"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("not found" in m for m in msgs)

    def test_ambiguous_prefix(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx(thread_store=FakeThreadStore(matches=["abc-one", "abc-two"]))
        _run(DeleteCommand().execute(ctx, ["abc"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Ambiguous" in m for m in msgs)

    def test_prefix_resolves_to_unique_match(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx(
            thread_store=FakeThreadStore(resolved_thread_id="abc-one", deleted=True)
        )
        _run(DeleteCommand().execute(ctx, ["abc"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Deleted session abc-one" in m for m in msgs)

    def test_no_arg_empty_sessions_prints_notice(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        _run(DeleteCommand().execute(ctx, []))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("No sessions to delete" in m for m in msgs)

    def test_no_arg_calls_picker_returns_none(self):
        """When no arg and picker returns None, nothing is deleted."""
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        ui.wait_for_thread_pick = AsyncMock(return_value=None)
        threads = [
            {
                "thread_id": "t1",
                "preview": "",
                "message_count": 1,
                "model": None,
                "updated_at": None,
            }
        ]
        store = FakeThreadStore(threads=threads)
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        _run(DeleteCommand().execute(ctx, []))
        ui.wait_for_thread_pick.assert_awaited_once()
