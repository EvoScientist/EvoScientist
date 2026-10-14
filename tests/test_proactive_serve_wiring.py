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
        "stop",
        "move server",
        ("move_to", Workspace("/new-ws")),
        ("adopt", "new-tid"),
        "mirror",
    ]


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
