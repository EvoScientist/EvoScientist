"""Tests for the serve-mode ``on_cmd_completed`` hook factory.

Regression coverage for the follow-up to issue #181 — `/model` invoked
over a channel in ``EvoSci serve`` must swap the running agent for
subsequent messages, not silently keep the stale one the while-loop
captured at startup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from EvoScientist.cli.channel import (
    ChannelMessage,
    _register_channel_request,
)
from EvoScientist.cli.commands import (
    ServeRuntimeState,
    _make_serve_cmd_completed_hook,
    _make_serve_handle_session_resume_cb,
    _make_serve_start_new_session_cb,
    _serve_process_message,
)
from EvoScientist.commands.base import ChannelRuntime
from EvoScientist.config import EvoScientistConfig
from EvoScientist.gateway import RuntimeGateways, ThreadStore
from tests.conftest import run_async as _run
from tests.fakes import FakeGraphGateway, FakeThreadStore


def _agent(name: str = "agent") -> CompiledStateGraph:
    return MagicMock(name=name, spec=CompiledStateGraph)


def _config() -> EvoScientistConfig:
    return EvoScientistConfig()


def _thread_store(thread_id: str = "unused") -> ThreadStore:
    return FakeThreadStore(generated_thread_id=thread_id)


def _runtime_gateways(thread_store: ThreadStore | None = None) -> RuntimeGateways:
    store = thread_store or _thread_store()

    return RuntimeGateways(
        thread_store=store,
        graph_gateway=FakeGraphGateway(thread_store=store),
    )


def _runtime_state(
    *,
    agent: CompiledStateGraph | None = None,
    thread_id: str = "tid",
    workspace_dir: str | None = None,
    config: EvoScientistConfig | None = None,
    thread_store: ThreadStore | None = None,
    runtime_gateways: RuntimeGateways | None = None,
) -> ServeRuntimeState:
    store = thread_store or _thread_store()
    return ServeRuntimeState(
        agent=agent if agent is not None else _agent(),
        thread_id=thread_id,
        workspace_dir=workspace_dir,
        config=config,
        runtime_gateways=runtime_gateways or _runtime_gateways(store),
    )


def test_hook_updates_runtime_state_on_agent_swap():
    """``/model`` mutates ``ctx.agent`` to a new handle — the hook must
    push that handle into the shared runtime state so the outer poll loop sees
    it on the next message."""
    original_agent = _agent("original-agent")
    new_agent = _agent("new-agent")
    state = _runtime_state(agent=original_agent)
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = new_agent
    ctx.thread_id = state.thread_id
    cmd = MagicMock()
    cmd.name = "/model"

    _run(hook(ctx, original_agent, cmd))

    assert state.agent is new_agent


def test_hook_syncs_channel_runtime():
    """Other readers (the bus) look at ``ChannelRuntime.agent``; the
    hook keeps the runtime in sync with the runtime state update."""
    original_agent = _agent("original-agent")
    new_agent = _agent("new-agent")
    state = _runtime_state(agent=original_agent, thread_id="t")
    runtime = ChannelRuntime(agent=original_agent, thread_id="t")
    hook = _make_serve_cmd_completed_hook(state, runtime)

    ctx = MagicMock()
    ctx.agent = new_agent
    # Pin ctx.thread_id explicitly — a bare MagicMock would let the
    # hook's getattr fall through to a fresh MagicMock attribute and
    # silently mutate runtime.thread_id, hiding regressions.
    ctx.thread_id = "t"
    cmd = MagicMock()
    cmd.name = "/model"

    _run(hook(ctx, original_agent, cmd))

    assert runtime.agent is new_agent
    assert runtime.thread_id == "t"


def test_hook_noop_when_agent_unchanged():
    """Commands like ``/evoskills`` don't touch ``ctx.agent`` — the
    runtime state must stay put."""
    original_agent = _agent("original-agent")
    state = _runtime_state(agent=original_agent)
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = original_agent  # no swap
    ctx.thread_id = state.thread_id
    cmd = MagicMock()
    cmd.name = "/evoskills"

    _run(hook(ctx, original_agent, cmd))

    assert state.agent is original_agent


def test_hook_noop_when_ctx_agent_is_none():
    """Guard against commands that reset ``ctx.agent`` to ``None`` —
    we never want to write ``None`` into runtime state."""
    original_agent = _agent("original-agent")
    state = _runtime_state(agent=original_agent)
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = None
    ctx.thread_id = state.thread_id
    cmd = MagicMock()
    cmd.name = "/whatever"

    _run(hook(ctx, original_agent, cmd))

    assert state.agent is original_agent


def test_hook_updates_thread_id_on_resume():
    """``/resume`` mutates ``ctx.thread_id`` — the hook must push the
    new id into runtime state so the outer poll loop runs subsequent
    messages on the resumed thread."""
    agent = _agent("a")
    state = _runtime_state(agent=agent, thread_id="original-tid")
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = agent  # no agent swap
    ctx.thread_id = "new-tid"
    ctx.workspace_dir = None
    cmd = MagicMock()
    cmd.name = "/resume"

    _run(hook(ctx, agent, cmd))

    assert state.thread_id == "new-tid"


def test_hook_updates_workspace_dir_on_resume():
    """`/resume` can restore a different workspace; serve must reload for it."""
    cfg = _config()
    old_agent = _agent("old-agent")
    reloaded_agent = _agent("reloaded-agent")
    state = _runtime_state(
        agent=old_agent,
        thread_id="original-tid",
        workspace_dir="/old-ws",
        config=cfg,
    )
    hook = _make_serve_cmd_completed_hook(state, config=cfg)

    ctx = MagicMock()
    ctx.agent = old_agent
    ctx.thread_id = "new-tid"
    ctx.workspace_dir = "/restored-ws"
    cmd = MagicMock()
    cmd.name = "/resume"

    with (
        patch(
            "EvoScientist.cli.commands._sync_background_agent_server_workspace",
            new=AsyncMock(),
        ) as sync_server,
        patch(
            "EvoScientist.cli.commands._load_agent",
            return_value=reloaded_agent,
        ) as load_agent,
    ):
        _run(hook(ctx, old_agent, cmd))

    sync_server.assert_awaited_once_with(cfg, workspace_dir="/restored-ws")
    load_agent.assert_called_once_with(workspace_dir="/restored-ws", config=cfg)
    assert state.workspace_dir == "/restored-ws"
    assert state.agent is reloaded_agent


def test_hook_syncs_channel_runtime_thread_id():
    """The bus reads ``ChannelRuntime.thread_id``; hook must sync it
    alongside the runtime state update."""
    agent = _agent("a")
    state = _runtime_state(agent=agent, thread_id="original-tid")
    runtime = ChannelRuntime(agent=agent, thread_id="original-tid")
    hook = _make_serve_cmd_completed_hook(state, runtime)

    ctx = MagicMock()
    ctx.agent = agent
    ctx.thread_id = "new-tid"
    ctx.workspace_dir = None
    cmd = MagicMock()
    cmd.name = "/resume"

    _run(hook(ctx, agent, cmd))

    assert runtime.thread_id == "new-tid"


def test_hook_noop_when_thread_id_unchanged():
    """Most commands don't touch thread_id — runtime state stays put."""
    agent = _agent("a")
    state = _runtime_state(agent=agent, thread_id="same-tid")
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = agent
    ctx.thread_id = "same-tid"
    cmd = MagicMock()
    cmd.name = "/evoskills"

    _run(hook(ctx, agent, cmd))

    assert state.thread_id == "same-tid"


def test_hook_skips_resume_warning_when_thread_unchanged():
    """Bare ``/resume`` with no argument prints usage but leaves
    ``ctx.thread_id`` unchanged — the in-memory-state warning must NOT
    fire because no resume actually happened."""
    agent = _agent("a")
    state = _runtime_state(agent=agent, thread_id="original-tid")
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = agent
    ctx.thread_id = "original-tid"  # unchanged — bare /resume case
    ctx.workspace_dir = None
    cmd = MagicMock()
    cmd.name = "/resume"

    _run(hook(ctx, agent, cmd))

    ctx.ui.append_system.assert_not_called()
    ctx.ui.flush.assert_not_called()


def test_hook_emits_resume_warning_when_thread_changed():
    """``/resume <tid>`` that actually changes thread_id must surface
    the in-memory-state warning via ``ctx.ui``."""
    agent = _agent("a")
    state = _runtime_state(agent=agent, thread_id="original-tid")
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    # Mock out async flush so the test can synchronously run the hook.
    ctx.ui.flush = AsyncMock()
    ctx.agent = agent
    ctx.thread_id = "abc12345-resumed-tid"
    ctx.workspace_dir = None
    cmd = MagicMock()
    cmd.name = "/resume"

    _run(hook(ctx, agent, cmd))

    ctx.ui.append_system.assert_called_once()
    warn_text, warn_kwargs = (
        ctx.ui.append_system.call_args.args,
        ctx.ui.append_system.call_args.kwargs,
    )
    assert "in-memory state" in warn_text[0]
    assert "abc12345" in warn_text[0]
    assert warn_kwargs.get("style") == "yellow"
    ctx.ui.flush.assert_awaited_once()


def test_start_new_session_cb_rotates_thread_id():
    """``/new`` via channel calls this callback — must generate a new
    thread id, push into runtime state, and sync the channel runtime."""
    agent = _agent("a")
    state = _runtime_state(
        agent=agent,
        thread_id="old-tid",
        thread_store=_thread_store("freshly-generated-tid"),
    )
    runtime = ChannelRuntime(agent=agent, thread_id="old-tid")

    cb = _make_serve_start_new_session_cb(
        state,
        runtime,
    )
    _run(cb())

    assert state.thread_id == "freshly-generated-tid"
    assert runtime.thread_id == "freshly-generated-tid"


def test_start_new_session_cb_leaves_agent_alone():
    """``/new`` rotates thread only — agent handle must stay put
    (serve's agent is a single pre-loaded instance, not per-thread)."""
    agent = _agent("a")
    state = _runtime_state(
        agent=agent,
        thread_id="old-tid",
        thread_store=_thread_store("new-tid"),
    )

    cb = _make_serve_start_new_session_cb(state)
    _run(cb())

    assert state.agent is agent


def test_serve_resume_callback_syncs_reloads_and_adopts_workspace():
    cfg = _config()
    old_agent = _agent("old-agent")
    reloaded_agent = _agent("reloaded-agent")
    state = _runtime_state(
        agent=old_agent,
        thread_id="old-tid",
        workspace_dir="/old-ws",
        config=cfg,
    )
    runtime = ChannelRuntime(agent=old_agent, thread_id="old-tid")
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=cfg)
    call_order: list[str] = []

    def _load_agent(**_kwargs):
        call_order.append("load")
        return reloaded_agent

    async def _sync_server(*_args, **_kwargs):
        call_order.append("sync")

    with (
        patch(
            "EvoScientist.cli.commands._sync_background_agent_server_workspace",
            new=AsyncMock(side_effect=_sync_server),
        ) as sync_server,
        patch(
            "EvoScientist.cli.commands._load_agent",
            side_effect=_load_agent,
        ) as load_agent,
    ):
        _run(cb("new-tid", "/new-ws"))

    sync_server.assert_awaited_once_with(cfg, workspace_dir="/new-ws")
    load_agent.assert_called_once_with(workspace_dir="/new-ws", config=cfg)
    assert call_order == ["load", "sync"]
    assert state.thread_id == "new-tid"
    assert state.workspace_dir == "/new-ws"
    assert state.agent is reloaded_agent
    assert runtime.thread_id == "new-tid"
    assert runtime.agent is reloaded_agent


def test_hook_emits_resume_warning_after_resume_callback_adopts_thread():
    cfg = _config()
    old_agent = _agent("old-agent")
    reloaded_agent = _agent("reloaded-agent")
    state = _runtime_state(
        agent=old_agent,
        thread_id="old-tid",
        workspace_dir="/old-ws",
        config=cfg,
    )
    runtime = ChannelRuntime(agent=old_agent, thread_id="old-tid")
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=cfg)

    with (
        patch(
            "EvoScientist.cli.commands._sync_background_agent_server_workspace",
            new=AsyncMock(),
        ),
        patch(
            "EvoScientist.cli.commands._load_agent",
            return_value=reloaded_agent,
        ),
    ):
        _run(cb("abc12345-resumed-tid", "/new-ws"))

    hook = _make_serve_cmd_completed_hook(state, runtime, config=cfg)
    ctx = MagicMock()
    ctx.ui.flush = AsyncMock()
    ctx.agent = reloaded_agent
    ctx.thread_id = "abc12345-resumed-tid"
    ctx.workspace_dir = "/new-ws"
    cmd = MagicMock()
    cmd.name = "/resume"

    _run(hook(ctx, reloaded_agent, cmd))

    ctx.ui.append_system.assert_called_once()
    assert "in-memory state" in ctx.ui.append_system.call_args.args[0]
    ctx.ui.flush.assert_awaited_once()


def test_serve_resume_callback_preserves_state_when_sync_fails():
    cfg = _config()
    old_agent = _agent("old-agent")
    loaded_but_not_adopted = _agent("loaded-but-not-adopted")
    state = _runtime_state(
        agent=old_agent,
        thread_id="old-tid",
        workspace_dir="/old-ws",
        config=cfg,
    )
    runtime = ChannelRuntime(agent=old_agent, thread_id="old-tid")
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=cfg)

    with (
        patch(
            "EvoScientist.cli.commands._sync_background_agent_server_workspace",
            new=AsyncMock(side_effect=RuntimeError("workspace conflict")),
        ),
        patch(
            "EvoScientist.cli.commands._load_agent",
            return_value=loaded_but_not_adopted,
        ) as load_agent,
        patch("EvoScientist.cli.commands.set_active_workspace") as set_active,
        pytest.raises(RuntimeError, match="workspace conflict"),
    ):
        _run(cb("new-tid", "/new-ws"))

    load_agent.assert_called_once_with(workspace_dir="/new-ws", config=cfg)
    set_active.assert_called_once_with("/old-ws")
    assert state.agent is old_agent
    assert state.resume_warning_thread_id is None
    assert state.thread_id == "old-tid"
    assert state.workspace_dir == "/old-ws"
    assert state.config is cfg
    assert runtime.agent is old_agent
    assert runtime.thread_id == "old-tid"


def test_serve_resume_callback_load_failure_does_not_sync_or_adopt():
    cfg = _config()
    old_agent = _agent("old-agent")
    state = _runtime_state(
        agent=old_agent,
        thread_id="old-tid",
        workspace_dir="/old-ws",
        config=cfg,
    )
    runtime = ChannelRuntime(agent=old_agent, thread_id="old-tid")
    cb = _make_serve_handle_session_resume_cb(state, runtime, config=cfg)

    with (
        patch(
            "EvoScientist.cli.commands._load_agent",
            side_effect=RuntimeError("load failed"),
        ) as load_agent,
        patch("EvoScientist.cli.commands.set_active_workspace") as set_active,
        patch(
            "EvoScientist.cli.commands._sync_background_agent_server_workspace",
            new=AsyncMock(),
        ) as sync_server,
        pytest.raises(RuntimeError, match="load failed"),
    ):
        _run(cb("new-tid", "/new-ws"))

    load_agent.assert_called_once_with(workspace_dir="/new-ws", config=cfg)
    set_active.assert_called_once_with("/old-ws")
    sync_server.assert_not_awaited()
    assert state.resume_warning_thread_id is None
    assert state.agent is old_agent
    assert state.thread_id == "old-tid"
    assert state.workspace_dir == "/old-ws"
    assert state.config is cfg
    assert runtime.agent is old_agent
    assert runtime.thread_id == "old-tid"


def test_hook_handles_both_agent_and_thread_swap():
    """Edge case: a command that changes both (hypothetical). Both
    updates must land in runtime state."""
    old_agent = _agent("old-agent")
    new_agent = _agent("new-agent")
    state = _runtime_state(agent=old_agent, thread_id="old-tid")
    hook = _make_serve_cmd_completed_hook(state)

    ctx = MagicMock()
    ctx.agent = new_agent
    ctx.thread_id = "new-tid"
    cmd = MagicMock()

    _run(hook(ctx, old_agent, cmd))

    assert state.agent is new_agent
    assert state.thread_id == "new-tid"


def test_serve_process_message_reports_slash_dispatch_error_without_fallback():
    """Defensive: if ``dispatch_channel_slash_command`` ever leaks an
    exception past its own wrapper, ``_serve_process_message`` must set
    one error response and not fall through to ``run_streaming``.
    """
    msg = ChannelMessage(
        msg_id="msg-1",
        content="/evoskills core",
        sender="channel-user",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="channel-user",
        message_id="ts-1",
    )
    thread_store = _thread_store()
    state = _runtime_state(
        agent=_agent(),
        thread_id="tid",
        thread_store=thread_store,
        runtime_gateways=_runtime_gateways(thread_store),
    )

    with (
        patch(
            "EvoScientist.cli.commands.dispatch_channel_slash_command",
            new=AsyncMock(side_effect=RuntimeError("slash broke")),
        ),
        patch("EvoScientist.cli.commands._set_channel_response") as mock_set_resp,
        patch("EvoScientist.cli.tui_runtime.run_streaming") as mock_run_streaming,
    ):
        _register_channel_request(msg)
        _serve_process_message(
            msg,
            runtime_state=state,
            model="model",
            workspace_dir="/tmp",
            show_thinking=False,
        )

    mock_set_resp.assert_called_once_with("msg-1", "Command error: slash broke")
    mock_run_streaming.assert_not_called()


def test_serve_process_message_uses_runtime_workspace_from_state():
    """After `/resume`, serve should use the adopted workspace, not startup ws."""
    msg = ChannelMessage(
        msg_id="msg-2",
        content="hello",
        sender="channel-user",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="channel-user",
        message_id="ts-2",
    )
    thread_store = _thread_store()
    state = _runtime_state(
        agent=_agent(),
        thread_id="tid",
        workspace_dir="/restored-workspace",
        thread_store=thread_store,
        runtime_gateways=_runtime_gateways(thread_store),
    )
    captured: dict[str, str] = {}

    async def _fake_dispatch(*args, **kwargs):
        captured["slash_workspace"] = kwargs["workspace_dir"]
        return False

    def _fake_build_metadata(workspace_dir: str, _model: str | None):
        captured["meta_workspace"] = workspace_dir
        return {}

    with (
        patch(
            "EvoScientist.cli.commands.dispatch_channel_slash_command",
            new=AsyncMock(side_effect=_fake_dispatch),
        ),
        patch(
            "EvoScientist.cli.commands.build_metadata",
            side_effect=_fake_build_metadata,
        ),
        patch("EvoScientist.cli.tui_runtime.run_streaming", return_value="ok"),
    ):
        _register_channel_request(msg)
        _serve_process_message(
            msg,
            runtime_state=state,
            model="model",
            workspace_dir="/startup-workspace",
            show_thinking=False,
        )

    assert captured["slash_workspace"] == "/restored-workspace"
    assert captured["meta_workspace"] == "/restored-workspace"
