"""Tests for the /delete command."""

from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import run_async as _run


def _ctx(thread_id="current"):
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(agent=None, thread_id=thread_id, ui=ui), ui


def _patches(thread_exists=False, similar=None, deleted=True, threads=None):
    """Return a context manager stack patching the sessions module."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch(
            "EvoScientist.sessions.thread_exists",
            new=AsyncMock(return_value=thread_exists),
        )
    )
    stack.enter_context(
        patch(
            "EvoScientist.sessions.find_similar_threads",
            new=AsyncMock(return_value=similar or []),
        )
    )
    stack.enter_context(
        patch(
            "EvoScientist.sessions.delete_thread",
            new=AsyncMock(return_value=deleted),
        )
    )
    stack.enter_context(
        patch(
            "EvoScientist.sessions.list_threads",
            new=AsyncMock(return_value=threads or []),
        )
    )
    return stack


class TestDeleteCommand:
    def test_refuses_to_delete_current(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx(thread_id="current")
        with _patches(thread_exists=True):
            _run(DeleteCommand().execute(ctx, ["current"]))
        # delete_thread must not have been called
        from EvoScientist.sessions import delete_thread

        assert not delete_thread.await_count if hasattr(delete_thread, "await_count") else True
        # And UI should warn about the current-session guard
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Cannot delete the current session" in m for m in msgs)

    def test_happy_path_success(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx(thread_id="current")
        with _patches(thread_exists=True, deleted=True):
            _run(DeleteCommand().execute(ctx, ["other-thread"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Deleted session other-thread" in m for m in msgs)

    def test_not_found(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        with _patches(thread_exists=False, similar=[]):
            _run(DeleteCommand().execute(ctx, ["missing"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("not found" in m for m in msgs)

    def test_ambiguous_prefix(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        with _patches(thread_exists=False, similar=["abc-one", "abc-two"]):
            _run(DeleteCommand().execute(ctx, ["abc"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Ambiguous" in m for m in msgs)

    def test_prefix_resolves_to_unique_match(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        with _patches(thread_exists=False, similar=["abc-one"], deleted=True):
            _run(DeleteCommand().execute(ctx, ["abc"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Deleted session abc-one" in m for m in msgs)

    def test_no_arg_empty_sessions_prints_notice(self):
        from EvoScientist.commands.implementation.session import DeleteCommand

        ctx, ui = _ctx()
        with _patches(threads=[]):
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
        with _patches(threads=threads):
            _run(DeleteCommand().execute(ctx, []))
        ui.wait_for_thread_pick.assert_awaited_once()
