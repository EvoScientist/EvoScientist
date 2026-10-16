"""Serve wiring of the proactive controller (EvoScientist.cli.commands)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from EvoScientist.cli.commands import (
    ServeRuntimeState,
    _make_serve_handle_session_resume_cb,
    _make_serve_proactive_switch,
    _serve_proactive_call,
    _serve_proactive_controller,
)
from EvoScientist.commands.base import ChannelRuntime
from EvoScientist.config import EvoScientistConfig
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.gateway.composite import CompositeGraphGateway
from EvoScientist.paths import SessionDirs, Workspace
from EvoScientist.runtime import AsyncRuntime
from tests.fakes import FakeGraphGateway, FakeThreadStore


def _gateways(graph_gateway=None) -> RuntimeGateways:
    store = FakeThreadStore(generated_thread_id="unused")
    return RuntimeGateways(
        thread_store=store,
        graph_gateway=graph_gateway or FakeGraphGateway(thread_store=store),
    )


@pytest.mark.parametrize(
    ("backend", "reason"),
    [
        ("local", "gateway_backend_serve: langgraph_server"),
        ("langgraph_server", "dev server was unavailable"),
    ],
)
def test_controller_is_unavailable_without_the_server_gateway(
    workspace, backend, reason
):
    controller = _serve_proactive_controller(
        EvoScientistConfig(),
        _gateways(),
        gateway_backend=backend,
        workspace=workspace,
    )
    assert controller.available is False
    assert reason in controller.unavailable_reason


def test_controller_is_available_on_the_server_gateway(workspace):
    composite = MagicMock(spec=CompositeGraphGateway)
    with patch(
        "EvoScientist.langgraph_dev.sdk.get_langgraph_sync_client",
        return_value=MagicMock(),
    ) as build:
        controller = _serve_proactive_controller(
            EvoScientistConfig(),
            _gateways(composite),
            gateway_backend="langgraph_server",
            workspace=workspace,
        )
    assert controller.available is True
    assert controller.workspace == workspace
    # Serve's message loop makes these calls: never the SDK's 300 s read.
    assert build.call_args.kwargs["timeout"] == 10.0


def test_a_controller_failure_never_raises():
    def boom():
        raise RuntimeError("dev server down")

    _serve_proactive_call(boom)


def _state(dirs: SessionDirs) -> ServeRuntimeState:
    return ServeRuntimeState(
        agent=MagicMock(spec=CompiledStateGraph),
        thread_id="old-tid",
        dirs=dirs,
        config=EvoScientistConfig(),
        runtime_gateways=_gateways(),
        async_runtime=MagicMock(spec=AsyncRuntime),
    )


async def test_resume_in_the_same_workspace_mirrors_the_thread():
    dirs = SessionDirs(Workspace("/ws"))
    state = _state(dirs)
    proactive = MagicMock()
    runtime = ChannelRuntime(
        agent=state.agent, thread_id="old-tid", proactive=proactive
    )
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=state.config)

    await cb("new-tid", dirs)

    proactive.leave.assert_called_once_with("old-tid")
    proactive.on_turn.assert_called_once_with("new-tid")
    proactive.adopt.assert_called_once_with("new-tid")
    proactive.mirror.assert_called_once_with()
    proactive.stop.assert_not_called()
    proactive.move_to.assert_not_called()


async def test_resume_into_another_workspace_moves_the_cron_with_the_server():
    state = _state(SessionDirs(Workspace("/old-ws")))
    proactive = MagicMock()
    runtime = ChannelRuntime(
        agent=state.agent, thread_id="old-tid", proactive=proactive
    )
    order = []
    proactive.stop.side_effect = lambda: order.append("stop")
    proactive.move_to.side_effect = lambda ws: order.append(("move_to", ws))
    proactive.adopt.side_effect = lambda tid: order.append(("adopt", tid))
    proactive.mirror.side_effect = lambda: order.append("mirror")
    proactive.reconcile.side_effect = lambda: order.append("reconcile")
    proactive.leave.side_effect = lambda tid: order.append(("leave", tid))
    proactive.on_turn.side_effect = lambda tid: order.append(("on_turn", tid))

    async def move(*_args, **_kwargs):
        order.append("move server")
        return True

    cb = _make_serve_handle_session_resume_cb(state, runtime, config=state.config)
    with (
        patch("EvoScientist.cli.commands._load_agent", return_value=MagicMock()),
        patch(
            "EvoScientist.cli.commands._move_server_or_keep_session",
            new=AsyncMock(side_effect=move),
        ),
    ):
        await cb("new-tid", SessionDirs(Workspace("/new-ws")))

    assert order == [
        ("leave", "old-tid"),
        ("on_turn", "new-tid"),
        "stop",
        "move server",
        ("move_to", Workspace("/new-ws")),
        "reconcile",
        ("adopt", "new-tid"),
        "mirror",
    ]


async def test_a_restart_in_the_workspace_serve_moved_from_restores_nothing(tmp_path):
    from EvoScientist.proactive.serve import ProactiveServe
    from tests.test_proactive_serve import _Client, _source

    old_ws, new_ws = (Workspace(tmp_path / name) for name in ("ws1", "ws2"))
    client = _Client()
    _source(client, "old-tid")
    client.threads.update("old-tid", metadata=SessionDirs(old_ws).metadata())
    proactive = ProactiveServe(workspace=old_ws, client=client)
    proactive.arm("old-tid")
    state = _state(SessionDirs(old_ws))
    runtime = ChannelRuntime(
        agent=state.agent, thread_id="old-tid", proactive=proactive
    )
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=state.config)
    with (
        patch("EvoScientist.cli.commands._load_agent", return_value=MagicMock()),
        patch(
            "EvoScientist.cli.commands._move_server_or_keep_session",
            new=AsyncMock(return_value=True),
        ),
    ):
        await cb("new-tid", SessionDirs(new_ws))

    assert client.threads.get("old-tid")["metadata"]["proactive"] == "paused"
    restarted = ProactiveServe(workspace=old_ws, client=client)
    assert restarted.restore([], restore_origin=lambda tid, origin: None) is None


async def test_a_refused_server_move_keeps_the_current_thread_armed(tmp_path):
    from EvoScientist.langgraph_dev.manager import WorkspaceMismatchError
    from EvoScientist.proactive.serve import ProactiveServe
    from tests.test_proactive_serve import _Client, _source

    old_ws, new_ws = (Workspace(tmp_path / name) for name in ("ws1", "ws2"))
    client = _Client()
    _source(client, "old-tid")
    proactive = ProactiveServe(
        workspace=old_ws, client=client, can_deliver=lambda tid: True
    )
    proactive.arm("old-tid")
    state = _state(SessionDirs(old_ws))
    runtime = ChannelRuntime(
        agent=state.agent, thread_id="old-tid", proactive=proactive
    )
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=state.config)
    with (
        patch("EvoScientist.cli.commands._load_agent", return_value=MagicMock()),
        patch(
            "EvoScientist.cli.commands._move_server_or_keep_session",
            new=AsyncMock(side_effect=WorkspaceMismatchError("held elsewhere")),
        ),
        pytest.raises(WorkspaceMismatchError),
    ):
        await cb("new-tid", SessionDirs(new_ws))

    assert state.thread_id == "old-tid"
    assert proactive.is_armed("old-tid")
    assert client.threads.get("old-tid")["metadata"]["proactive"] == "on"
    assert len(client.crons.rows) == 1


async def test_a_refused_server_move_does_not_arm_a_thread_that_was_not_armed(
    tmp_path,
):
    from EvoScientist.langgraph_dev.manager import WorkspaceMismatchError
    from EvoScientist.proactive.serve import ProactiveServe
    from tests.test_proactive_serve import _Client, _source

    old_ws, new_ws = (Workspace(tmp_path / name) for name in ("ws1", "ws2"))
    client = _Client()
    _source(client, "old-tid")
    client.threads.update("old-tid", metadata={"proactive": "paused"})
    proactive = ProactiveServe(
        workspace=old_ws, client=client, can_deliver=lambda tid: True
    )
    proactive.armed.add("other-tid")
    state = _state(SessionDirs(old_ws))
    runtime = ChannelRuntime(
        agent=state.agent, thread_id="old-tid", proactive=proactive
    )
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=state.config)
    with (
        patch("EvoScientist.cli.commands._load_agent", return_value=MagicMock()),
        patch(
            "EvoScientist.cli.commands._move_server_or_keep_session",
            new=AsyncMock(side_effect=WorkspaceMismatchError("held elsewhere")),
        ),
        pytest.raises(WorkspaceMismatchError),
    ):
        await cb("new-tid", SessionDirs(new_ws))

    assert not proactive.is_armed("old-tid")
    assert client.threads.get("old-tid")["metadata"]["proactive"] == "paused"
    assert proactive.is_armed("other-tid")
    assert len(client.crons.rows) == 1


async def test_a_failed_server_move_turns_proactive_off_for_the_session():
    state = _state(SessionDirs(Workspace("/old-ws")))
    proactive = MagicMock()
    proactive.unavailable_reason = None
    runtime = ChannelRuntime(
        agent=state.agent, thread_id="old-tid", proactive=proactive
    )
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=state.config)
    with (
        patch("EvoScientist.cli.commands._load_agent", return_value=MagicMock()),
        patch(
            "EvoScientist.cli.commands._move_server_or_keep_session",
            new=AsyncMock(return_value=False),
        ),
    ):
        await cb("new-tid", SessionDirs(Workspace("/new-ws")))

    proactive.move_to.assert_not_called()
    assert "did not move" in proactive.unavailable_reason


def test_a_push_to_the_restored_thread_resumes_it_in_its_folders(workspace):
    import asyncio

    state = _state(SessionDirs(Workspace("/fresh-ws")))
    state.async_runtime.run_sync.side_effect = lambda make: asyncio.run(make())
    runtime = ChannelRuntime(agent=state.agent, thread_id="old-tid")
    switch = _make_serve_proactive_switch(state, runtime)
    apply = AsyncMock()
    with (
        patch(
            "EvoScientist.sessions.get_thread_metadata",
            new=AsyncMock(return_value={"workspace_dir": workspace.key}),
        ),
        patch("EvoScientist.cli.commands._apply_serve_resume_state", new=apply),
    ):
        switch("restored-tid")

    apply.assert_awaited_once_with(
        state, runtime, thread_id="restored-tid", dirs=SessionDirs(workspace)
    )


def test_serve_restores_then_reconciles_and_deletes_the_cron_when_it_stops(
    monkeypatch, tmp_path
):
    from EvoScientist.cli import commands
    from tests.test_cli_serve import _make_config, _run_serve_once

    calls = []
    controller = MagicMock()
    controller.reconcile.side_effect = lambda: calls.append("reconcile")
    controller.stop.side_effect = lambda: calls.append("stop")
    monkeypatch.setattr(
        commands, "_serve_proactive_controller", lambda *a, **k: controller
    )
    monkeypatch.setattr(
        commands, "_serve_restore_proactive", lambda *a: calls.append("restore")
    )

    _run_serve_once(monkeypatch, _make_config(), workdir=str(tmp_path))

    assert calls == ["restore", "reconcile", "stop"]
