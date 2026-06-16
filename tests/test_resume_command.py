"""Tests for the /resume command."""

from unittest.mock import AsyncMock, MagicMock

from tests.conftest import run_async as _run
from tests.fakes import FakeGraphGateway, FakeThreadStore


def _ctx(thread_id="current", workspace_dir="/ws", thread_store=None):
    from EvoScientist.commands.base import CommandContext

    store = thread_store or FakeThreadStore()
    ui = MagicMock()
    ui.supports_interactive = True
    ui.wait_for_thread_pick = AsyncMock()
    ui.handle_session_resume = AsyncMock()
    return CommandContext(
        agent=None,
        thread_id=thread_id,
        ui=ui,
        workspace_dir=workspace_dir,
        graph_gateway=FakeGraphGateway(thread_store=store),
    ), ui


class TestResumeCommand:
    def test_with_arg_resolves_and_calls_ui(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx(
            thread_store=FakeThreadStore(
                resolved_thread_id="target-tid",
                metadata={"workspace_dir": "/restored"},
            )
        )
        _run(ResumeCommand().execute(ctx, ["target-tid"]))
        ui.handle_session_resume.assert_awaited_once_with("target-tid", "/restored")
        # ctx mutations
        assert ctx.thread_id == "target-tid"
        assert ctx.workspace_dir == "/restored"

    def test_no_arg_empty_threads_prints_message(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx()
        _run(ResumeCommand().execute(ctx, []))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("No sessions to resume" in m for m in msgs)
        ui.wait_for_thread_pick.assert_not_called()
        ui.handle_session_resume.assert_not_called()

    def test_no_arg_calls_picker(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx()
        ui.wait_for_thread_pick.return_value = "picked-tid"
        threads = [{"thread_id": "picked-tid", "preview": "p", "message_count": 1}]
        store = FakeThreadStore(
            threads=threads,
            resolved_thread_id="picked-tid",
        )
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        _run(ResumeCommand().execute(ctx, []))
        ui.wait_for_thread_pick.assert_awaited_once()
        ui.handle_session_resume.assert_awaited_once()

    def test_picker_cancel_returns(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx()
        ui.wait_for_thread_pick.return_value = None
        threads = [{"thread_id": "t1", "preview": "", "message_count": 0}]
        store = FakeThreadStore(threads=threads)
        ctx.graph_gateway = FakeGraphGateway(thread_store=store)
        _run(ResumeCommand().execute(ctx, []))
        ui.handle_session_resume.assert_not_called()

    def test_ambiguous_prefix(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx(thread_store=FakeThreadStore(matches=["abc-one", "abc-two"]))
        _run(ResumeCommand().execute(ctx, ["abc"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Ambiguous" in m for m in msgs)
        ui.handle_session_resume.assert_not_called()

    def test_not_found(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx()
        _run(ResumeCommand().execute(ctx, ["missing"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("not found" in m for m in msgs)
        ui.handle_session_resume.assert_not_called()

    def test_prefix_resolves_to_unique_match(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx(
            thread_store=FakeThreadStore(
                resolved_thread_id="abc-one",
                metadata={"workspace_dir": "/ws1"},
            )
        )
        _run(ResumeCommand().execute(ctx, ["abc"]))
        ui.handle_session_resume.assert_awaited_once_with("abc-one", "/ws1")
        assert ctx.thread_id == "abc-one"

    def test_empty_workspace_metadata_preserves_ctx_workspace(self):
        from EvoScientist.commands.implementation.session import ResumeCommand

        ctx, ui = _ctx(
            workspace_dir="/keep",
            thread_store=FakeThreadStore(resolved_thread_id="tid", metadata={}),
        )
        _run(ResumeCommand().execute(ctx, ["tid"]))
        # ResumeCommand only overwrites ctx.workspace_dir if metadata has one
        assert ctx.workspace_dir == "/keep"
        # Callback still fires with the metadata value (empty string)
        ui.handle_session_resume.assert_awaited_once_with("tid", "")
