"""The ``/rubric`` command: operations recorded in the session, never thread writes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import EvoScientist.commands.implementation  # noqa: F401  (registers the commands)
from EvoScientist.commands.base import ChannelRuntime, CommandContext
from EvoScientist.commands.manager import manager
from EvoScientist.paths import SessionDirs, Workspace
from tests.fakes import FakeGraphGateway

_LOADED_AGENT = object()


def _ctx(tmp_path, *, agent=_LOADED_AGENT, ui=None, gateway=None, runtime=None):
    ui = ui if ui is not None else MagicMock()
    ui.flush = AsyncMock()  # the manager awaits it after every command
    runtime = runtime if runtime is not None else ChannelRuntime()
    ctx = CommandContext(
        agent=agent,
        thread_id="t",
        ui=ui,
        dirs=SessionDirs(Workspace(tmp_path)),
        channel_runtime=runtime,
        graph_gateway=gateway if gateway is not None else FakeGraphGateway(),
    )
    return ctx, ui, runtime


def _lines(ui) -> list[str]:
    """Every line the command showed: system lines and plain renderables."""
    lines = [str(c.args[0]) for c in ui.append_system.call_args_list]
    for c in ui.mount_renderable.call_args_list:
        lines.append(getattr(c.args[0], "plain", str(c.args[0])))
    return lines


def _renderables(ui) -> list[str]:
    return [getattr(c.args[0], "plain", "") for c in ui.mount_renderable.call_args_list]


async def test_set_records_a_sticky_operation_with_the_raw_text(tmp_path):
    ctx, ui, runtime = _ctx(tmp_path)
    assert await manager.execute("/rubric set - a 'quoted'\n- b", ctx)
    assert runtime.rubric.changes_for("t") == [
        {"session": runtime.rubric.session_id, "seq": 1, "sticky": "- a 'quoted'\n- b"}
    ]
    assert any("next message" in line for line in _lines(ui))
    assert ctx.graph_gateway.updated_states == []


async def test_next_and_clear_record_their_operations(tmp_path):
    ctx, _ui, runtime = _ctx(tmp_path)
    assert await manager.execute("/rubric next - once", ctx)
    assert await manager.execute("/rubric clear", ctx)
    ops = runtime.rubric.changes_for("t")
    assert [op["seq"] for op in ops] == [1, 2]
    assert ops[0]["next"] == "- once"
    assert ops[1] == {
        "session": runtime.rubric.session_id,
        "seq": 2,
        "sticky": None,
        "next": None,
    }


async def test_at_file_is_read_relative_to_the_work_dir(tmp_path):
    (tmp_path / "r.md").write_text("- from file\n", encoding="utf-8")
    ctx, _ui, runtime = _ctx(tmp_path)
    assert await manager.execute("/rubric set @r.md", ctx)
    assert runtime.rubric.changes_for("t")[0]["sticky"] == "- from file"


async def test_unreadable_file_prints_an_error_and_records_nothing(tmp_path):
    ctx, ui, runtime = _ctx(tmp_path)
    assert await manager.execute("/rubric set @missing.md", ctx)
    assert runtime.rubric.changes_for("t") == []
    assert any("could not read" in line for line in _lines(ui))


async def test_set_without_text_prints_usage_and_records_nothing(tmp_path):
    ctx, ui, runtime = _ctx(tmp_path)
    assert await manager.execute("/rubric set", ctx)
    assert runtime.rubric.changes_for("t") == []
    assert any("Usage" in line for line in _lines(ui))


async def test_show_prints_thread_values_and_pending_operations(tmp_path):
    gateway = FakeGraphGateway(
        state_values={"_sticky_rubric": {"text": "- saved", "id": "s:1"}}
    )
    ctx, ui, runtime = _ctx(tmp_path, gateway=gateway)
    runtime.rubric.record("t", next="- pending")
    assert await manager.execute("/rubric show", ctx)
    text = "\n".join(_lines(ui))
    assert "- saved" in text
    assert "- pending" in text
    assert "not yet applied" in text


async def test_bare_rubric_is_show(tmp_path):
    gateway = FakeGraphGateway(state_values={})
    ctx, ui, _runtime = _ctx(tmp_path, gateway=gateway)
    assert await manager.execute("/rubric", ctx)
    assert any("No acceptance criteria" in line for line in _lines(ui))


async def test_show_without_a_loaded_agent_prints_only_pending(tmp_path):
    ctx, ui, runtime = _ctx(tmp_path, agent=None)
    runtime.rubric.record("t", sticky="- soon")
    assert await manager.execute("/rubric show", ctx)
    text = "\n".join(_lines(ui))
    assert "- soon" in text
    assert "not loaded" in text


async def test_refused_on_a_channel_ui(tmp_path):
    ui = MagicMock()
    ui.supports_interactive = False
    ctx, ui, runtime = _ctx(tmp_path, ui=ui)
    assert await manager.execute("/rubric set - a", ctx)
    assert runtime.rubric.changes_for("t") == []
    assert any("terminal session" in line for line in _lines(ui))


async def test_refused_when_the_session_executes_on_the_server(tmp_path):
    from EvoScientist.gateway.server import LangGraphServerGateway

    server = MagicMock(spec=LangGraphServerGateway)
    ctx, ui, runtime = _ctx(tmp_path, gateway=server)
    assert await manager.execute("/rubric set - a", ctx)
    assert runtime.rubric.changes_for("t") == []
    assert any("local gateway" in line for line in _lines(ui))

    composite = MagicMock()
    composite.execute_gateway = server
    ctx, ui, runtime = _ctx(tmp_path, gateway=composite)
    assert await manager.execute("/rubric next - a", ctx)
    assert runtime.rubric.changes_for("t") == []


async def test_raw_args_keeps_the_text_after_the_command_name(tmp_path):
    ctx, _ui, _runtime = _ctx(tmp_path)
    await manager.execute("/rubric set x  y\nz", ctx)
    assert ctx.raw_args == "set x  y\nz"
    await manager.execute("/current", ctx)
    assert ctx.raw_args == ""


def test_subcommands_are_declared_for_completion():
    names = [sc.name for sc in manager.get_subcommands("/rubric")]
    assert names == ["set", "next", "show", "clear"]
    assert manager.get_command("/rubric").needs_agent(["set", "- a"]) is False


async def test_show_drops_operations_the_thread_already_applied(tmp_path):
    # An earlier acknowledgement read failed, so a consumed next is still queued.
    ctx, ui, runtime = _ctx(tmp_path)
    runtime.rubric.record("t", next="- n")
    ctx.graph_gateway.state_values = {
        "_rubric_applied": {runtime.rubric.session_id: 1},
        "_next_rubric": None,
    }
    assert await manager.execute("/rubric show", ctx)
    text = "\n".join(_lines(ui))
    assert "No acceptance criteria" in text
    assert "Next task only" not in text
    assert "not yet applied" not in text
    assert runtime.rubric.changes_for("t") == []


async def test_checklist_and_error_text_reach_the_ui_as_plain_renderables(tmp_path):
    from rich.markup import render

    gateway = FakeGraphGateway(
        state_values={
            "_sticky_rubric": {"text": "- [x] no [/missing] tag", "id": "s:1"}
        }
    )
    ctx, ui, runtime = _ctx(tmp_path, gateway=gateway)
    assert await manager.execute("/rubric set - needs [/closing] here", ctx)
    assert runtime.rubric.changes_for("t")[0]["sticky"] == "- needs [/closing] here"
    assert await manager.execute("/rubric show", ctx)
    assert await manager.execute("/rubric set @[x]missing.md", ctx)
    # Text the user wrote (or a path they gave) is shown as it is, in both UIs.
    plain = _renderables(ui)
    assert "- needs [/closing] here" in plain
    assert any(
        line.startswith("/rubric set could not read") and "[x]missing.md" in line
        for line in plain
    )
    assert not any("\\[" in line for line in plain)
    # Lines built by the command itself stay markup-safe system lines.
    for line in (str(c.args[0]) for c in ui.append_system.call_args_list):
        render(line)


async def test_quoted_file_operand_is_unquoted(tmp_path):
    (tmp_path / "criteria space.md").write_text("- spaced\n", encoding="utf-8")
    ctx, _ui, runtime = _ctx(tmp_path)
    assert await manager.execute('/rubric set @"criteria space.md"', ctx)
    assert await manager.execute("/rubric next @'criteria space.md'", ctx)
    ops = runtime.rubric.changes_for("t")
    assert ops[0]["sticky"] == "- spaced"
    assert ops[1]["next"] == "- spaced"
