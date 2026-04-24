"""Tests for ``dispatch_channel_slash_command`` in ``cli/channel.py``.

Regression coverage for issue #181 — slash commands arriving over a
messaging channel must route through ``cmd_manager`` instead of being
fed to the LLM as a plain prompt, on every UI surface (Rich CLI, TUI,
headless ``serve``).
"""

from unittest.mock import AsyncMock, MagicMock, patch

from EvoScientist.cli.channel import (
    ChannelMessage,
    dispatch_channel_slash_command,
)
from tests.conftest import run_async as _run


def _make_msg(
    content: str = "/evoskills core", msg_id: str = "msg-1"
) -> ChannelMessage:
    return ChannelMessage(
        msg_id=msg_id,
        content=content,
        sender="+44XXXXXX",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="+44XXXXXX",
        message_id="ts-1",
    )


def test_non_slash_returns_false():
    """Plain text messages must fall through to the agent."""
    msg = _make_msg(content="hello agent")
    append = MagicMock()
    handled = _run(
        dispatch_channel_slash_command(
            msg,
            agent=None,
            thread_id="t1",
            workspace_dir=None,
            checkpointer=None,
            append_system=append,
        )
    )
    assert handled is False
    append.assert_not_called()


def test_unresolved_slash_returns_false():
    """Unknown slash commands must fall through (matches TUI behavior)."""
    msg = _make_msg(content="/unknown-cmd")
    append = MagicMock()
    with patch(
        "EvoScientist.commands.manager.manager.resolve",
        return_value=None,
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=append,
            )
        )
    assert handled is False


def test_successful_slash_execution_sets_response_and_breadcrumb():
    """Known slash command: cmd_manager.execute ran, helper returns True,
    sends a confirmation to the channel user, and appends a local log line."""
    msg = _make_msg()
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = False
    append = MagicMock()
    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, ["core"]),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=AsyncMock(return_value=True),
        ) as mock_execute,
        patch("EvoScientist.cli.channel._set_channel_response") as mock_set_resp,
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent="fake-agent",
                thread_id="t1",
                workspace_dir="/tmp",
                checkpointer=None,
                append_system=append,
            )
        )
    assert handled is True
    mock_execute.assert_awaited_once()
    mock_set_resp.assert_called_once()
    assert mock_set_resp.call_args[0][0] == "msg-1"
    assert "Command executed" in mock_set_resp.call_args[0][1]
    breadcrumbs = [call.args[0] for call in append.call_args_list]
    assert any("Executed command from" in t for t in breadcrumbs)


def test_needs_agent_awaits_loader_and_passes_result():
    """Commands with needs_agent=True must await the loader and the
    resulting agent must flow through the CommandContext."""
    msg = _make_msg()
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = True
    append = MagicMock()
    await_called = MagicMock()

    async def _await_ready():
        await_called()
        return "ready-agent"

    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=AsyncMock(return_value=True),
        ) as mock_execute,
        patch("EvoScientist.cli.channel._set_channel_response"),
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=append,
                await_agent_ready=_await_ready,
            )
        )
    assert handled is True
    await_called.assert_called_once()
    ctx_arg = mock_execute.await_args.args[1]
    assert ctx_arg.agent == "ready-agent"


def test_await_agent_ready_failure_sets_error_response():
    msg = _make_msg()
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = True
    append = MagicMock()

    async def _await_ready():
        raise RuntimeError("agent blew up")

    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch("EvoScientist.cli.channel._set_channel_response") as mock_set_resp,
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=append,
                await_agent_ready=_await_ready,
            )
        )
    assert handled is True
    mock_set_resp.assert_called_once()
    resp_text = mock_set_resp.call_args[0][1]
    assert "Error" in resp_text
    assert "agent blew up" in resp_text


def test_cmd_manager_raises_returns_true_with_error():
    """If cmd_manager.execute raises past its own try/except, the helper
    must absorb it, return True, and report via _set_channel_response."""
    msg = _make_msg()
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = False
    append = MagicMock()
    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch("EvoScientist.cli.channel._set_channel_response") as mock_set_resp,
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=append,
            )
        )
    assert handled is True
    mock_set_resp.assert_called_once()
    resp_text = mock_set_resp.call_args[0][1]
    assert "Command error" in resp_text
    assert "boom" in resp_text


def test_on_cmd_completed_awaited_with_ctx_original_agent_and_cmd():
    """After a successful slash execute, the on_cmd_completed hook must
    be awaited with (ctx, original_agent, cmd) so Rich CLI can adopt an
    ``/model`` agent swap and refresh status for state-mutating commands."""
    msg = _make_msg()
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = False
    fake_cmd.name = "/model"
    append = MagicMock()
    captured: dict[str, object] = {}

    async def _fake_execute(command_str, ctx):
        # Simulate /model swapping ctx.agent to a new handle.
        ctx.agent = "swapped-agent"
        return True

    async def _on_completed(ctx, original_agent, cmd):
        captured["ctx_agent"] = ctx.agent
        captured["original_agent"] = original_agent
        captured["cmd_name"] = getattr(cmd, "name", None)

    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=_fake_execute,
        ),
        patch("EvoScientist.cli.channel._set_channel_response"),
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent="original-agent",
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=append,
                on_cmd_completed=_on_completed,
            )
        )
    assert handled is True
    assert captured["ctx_agent"] == "swapped-agent"
    assert captured["original_agent"] == "original-agent"
    assert captured["cmd_name"] == "/model"


def test_on_cmd_completed_receives_cmd_for_new_and_compact():
    """``/new`` / ``/compact`` invoked via channel must flow the cmd into
    the hook so the callback can still refresh status when the agent
    didn't swap — mirrors REPL ``interactive.py:1027-1030``."""
    for cmd_name in ("/new", "/compact"):
        fake_cmd = MagicMock()
        fake_cmd.needs_agent.return_value = False
        fake_cmd.name = cmd_name
        captured: dict[str, str | None] = {}

        async def _on_completed(ctx, original_agent, cmd, _captured=captured):
            _captured["cmd_name"] = getattr(cmd, "name", None)

        with (
            patch(
                "EvoScientist.commands.manager.manager.resolve",
                return_value=(fake_cmd, []),
            ),
            patch(
                "EvoScientist.commands.manager.manager.execute",
                new=AsyncMock(return_value=True),
            ),
            patch("EvoScientist.cli.channel._set_channel_response"),
        ):
            _run(
                dispatch_channel_slash_command(
                    _make_msg(content=cmd_name),
                    agent="same-agent",
                    thread_id="t1",
                    workspace_dir=None,
                    checkpointer=None,
                    append_system=MagicMock(),
                    on_cmd_completed=_on_completed,
                )
            )
        assert captured["cmd_name"] == cmd_name, cmd_name


def test_on_cmd_completed_skipped_on_fall_through_and_error():
    """The hook must NOT fire for unresolved slash, non-slash text, or
    when cmd_manager.execute raised."""
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = False
    completed = MagicMock()

    async def _noop(*args, **kwargs):
        completed(*args, **kwargs)

    # Non-slash
    with patch("EvoScientist.cli.channel._set_channel_response"):
        _run(
            dispatch_channel_slash_command(
                _make_msg(content="hi"),
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=MagicMock(),
                on_cmd_completed=_noop,
            )
        )
    # Unresolved slash
    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=None,
        ),
        patch("EvoScientist.cli.channel._set_channel_response"),
    ):
        _run(
            dispatch_channel_slash_command(
                _make_msg(content="/nope"),
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=MagicMock(),
                on_cmd_completed=_noop,
            )
        )
    # Execute raises
    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch("EvoScientist.cli.channel._set_channel_response"),
    ):
        _run(
            dispatch_channel_slash_command(
                _make_msg(),
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=MagicMock(),
                on_cmd_completed=_noop,
            )
        )

    completed.assert_not_called()


def test_on_cmd_completed_exception_is_absorbed():
    """A raising hook must NOT prevent the channel response from being set."""
    msg = _make_msg()
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = False

    async def _boom(ctx, original_agent, cmd):
        raise RuntimeError("hook blew up")

    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=AsyncMock(return_value=True),
        ),
        patch("EvoScientist.cli.channel._set_channel_response") as mock_set_resp,
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent="orig",
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=MagicMock(),
                on_cmd_completed=_boom,
            )
        )
    assert handled is True
    mock_set_resp.assert_called_once()
    assert "Command executed" in mock_set_resp.call_args[0][1]


def test_cmd_execute_returning_false_falls_through():
    """When cmd_manager.execute returns False (empty/unparseable input),
    the helper must return False so the caller falls through to the agent."""
    msg = _make_msg(content="/")
    fake_cmd = MagicMock()
    fake_cmd.needs_agent.return_value = False
    append = MagicMock()
    with (
        patch(
            "EvoScientist.commands.manager.manager.resolve",
            return_value=(fake_cmd, []),
        ),
        patch(
            "EvoScientist.commands.manager.manager.execute",
            new=AsyncMock(return_value=False),
        ),
        patch("EvoScientist.cli.channel._set_channel_response") as mock_set_resp,
    ):
        handled = _run(
            dispatch_channel_slash_command(
                msg,
                agent=None,
                thread_id="t1",
                workspace_dir=None,
                checkpointer=None,
                append_system=append,
            )
        )
    assert handled is False
    mock_set_resp.assert_not_called()
