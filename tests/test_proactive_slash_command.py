"""Tests for the ``/proactive`` slash command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.commands.base import ChannelRuntime
from EvoScientist.commands.implementation.proactive import ProactiveCommand
from EvoScientist.commands.manager import manager


class _UI:
    def __init__(self):
        self.lines = []

    def append_system(self, text, style="dim"):
        self.lines.append((text, style))

    @property
    def text(self):
        return "\n".join(t for t, _ in self.lines)


def _ctx(controller=None, *, with_runtime=True):
    runtime = (
        ChannelRuntime(thread_id="t1", proactive=controller) if with_runtime else None
    )
    return SimpleNamespace(thread_id="t1", ui=_UI(), channel_runtime=runtime)


def _controller(*, armed=False):
    controller = MagicMock()
    controller.available = True
    controller.unavailable_reason = None
    controller.is_armed.return_value = armed
    return controller


async def _run(ctx, *args):
    await ProactiveCommand().execute(ctx, list(args))
    return ctx.ui


def test_command_is_registered():
    assert manager.get_command("/proactive") is not None


@pytest.mark.parametrize("with_runtime", [False, True])
async def test_outside_serve_it_explains_and_changes_nothing(with_runtime):
    ui = await _run(_ctx(None, with_runtime=with_runtime), "on")
    assert "run under `EvoSci serve`" in ui.text


async def test_unavailable_controller_reports_the_reason():
    controller = _controller()
    controller.available = False
    controller.unavailable_reason = "the dev server was unavailable when serve started"
    ui = await _run(_ctx(controller), "on")
    assert "not available: the dev server was unavailable" in ui.text
    controller.arm.assert_not_called()


async def test_on_arms_the_current_thread():
    controller = _controller()
    ui = await _run(_ctx(controller), "on")
    controller.arm.assert_called_once_with("t1")
    assert "on for this conversation" in ui.text
    assert "quiet hours" in ui.text


async def test_off_disarms_the_current_thread():
    controller = _controller(armed=True)
    ui = await _run(_ctx(controller), "OFF")
    controller.disarm.assert_called_once_with("t1")
    assert "off for this conversation" in ui.text


@pytest.mark.parametrize(("armed", "state"), [(True, "on"), (False, "off")])
async def test_status(armed, state):
    controller = _controller(armed=armed)
    ui = await _run(_ctx(controller))
    controller.arm.assert_not_called()
    controller.disarm.assert_not_called()
    assert f"Proactive follow-ups: {state} for this conversation." in ui.text


async def test_unknown_argument_prints_usage():
    controller = _controller()
    ui = await _run(_ctx(controller), "maybe")
    assert "Usage: /proactive [on|off]" in ui.text
    controller.arm.assert_not_called()


async def test_a_dev_server_error_is_reported():
    controller = _controller()
    controller.arm.side_effect = RuntimeError("connection refused")
    ui = await _run(_ctx(controller), "on")
    assert "Could not change proactive follow-ups: connection refused" in ui.text
