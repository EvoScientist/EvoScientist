"""Tests for the serve-mode ``on_cmd_completed`` hook factory.

Regression coverage for the follow-up to issue #181 — `/model` invoked
over a channel in ``EvoSci serve`` must swap the running agent for
subsequent messages, not silently keep the stale one the while-loop
captured at startup.
"""

from unittest.mock import MagicMock, patch

from EvoScientist.cli.commands import (
    _make_serve_cmd_completed_hook,
    _make_serve_start_new_session_cb,
)
from tests.conftest import run_async as _run


def test_hook_updates_holder_on_agent_swap():
    """``/model`` mutates ``ctx.agent`` to a new handle — the hook must
    push that handle into the shared holder so the outer poll loop sees
    it on the next message."""
    holder = {"agent": "original-agent"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "new-agent"
    cmd = MagicMock()
    cmd.name = "/model"

    _run(hook(ctx, "original-agent", cmd))

    assert holder["agent"] == "new-agent"


def test_hook_syncs_channel_module_global():
    """Other readers (the bus) look at ``cli.channel._cli_agent``; the
    hook keeps that global in sync with the holder update."""
    import EvoScientist.cli.channel as _ch_mod

    original_global = _ch_mod._cli_agent
    holder = {"agent": "original-agent"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "new-agent"
    cmd = MagicMock()
    cmd.name = "/model"

    try:
        _run(hook(ctx, "original-agent", cmd))
        assert _ch_mod._cli_agent == "new-agent"
    finally:
        _ch_mod._cli_agent = original_global


def test_hook_noop_when_agent_unchanged():
    """Commands like ``/evoskills`` don't touch ``ctx.agent`` — the
    holder must stay put."""
    holder = {"agent": "original-agent"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "original-agent"  # no swap
    cmd = MagicMock()
    cmd.name = "/evoskills"

    _run(hook(ctx, "original-agent", cmd))

    assert holder["agent"] == "original-agent"


def test_hook_noop_when_ctx_agent_is_none():
    """Guard against commands that reset ``ctx.agent`` to ``None`` —
    we never want to write ``None`` into the holder."""
    holder = {"agent": "original-agent"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = None
    cmd = MagicMock()
    cmd.name = "/whatever"

    _run(hook(ctx, "original-agent", cmd))

    assert holder["agent"] == "original-agent"


def test_hook_updates_thread_id_on_resume():
    """``/resume`` mutates ``ctx.thread_id`` — the hook must push the
    new id into the holder so the outer poll loop runs subsequent
    messages on the resumed thread."""
    holder = {"agent": "a", "thread_id": "original-tid"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "a"  # no agent swap
    ctx.thread_id = "new-tid"
    cmd = MagicMock()
    cmd.name = "/resume"

    _run(hook(ctx, "a", cmd))

    assert holder["thread_id"] == "new-tid"


def test_hook_syncs_channel_module_thread_id():
    """The bus reads ``cli.channel._cli_thread_id``; hook must sync it
    alongside the holder update."""
    import EvoScientist.cli.channel as _ch_mod

    original_tid_global = _ch_mod._cli_thread_id
    holder = {"agent": "a", "thread_id": "original-tid"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "a"
    ctx.thread_id = "new-tid"
    cmd = MagicMock()
    cmd.name = "/resume"

    try:
        _run(hook(ctx, "a", cmd))
        assert _ch_mod._cli_thread_id == "new-tid"
    finally:
        _ch_mod._cli_thread_id = original_tid_global


def test_hook_noop_when_thread_id_unchanged():
    """Most commands don't touch thread_id — holder stays put."""
    holder = {"agent": "a", "thread_id": "same-tid"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "a"
    ctx.thread_id = "same-tid"
    cmd = MagicMock()
    cmd.name = "/evoskills"

    _run(hook(ctx, "a", cmd))

    assert holder["thread_id"] == "same-tid"


def test_start_new_session_cb_rotates_thread_id():
    """``/new`` via channel calls this callback — must generate a new
    thread id, push into holder, and sync the channel-module global."""
    import EvoScientist.cli.channel as _ch_mod

    original_tid_global = _ch_mod._cli_thread_id
    holder = {"agent": "a", "thread_id": "old-tid"}

    with patch(
        "EvoScientist.sessions.generate_thread_id",
        return_value="freshly-generated-tid",
    ):
        cb = _make_serve_start_new_session_cb(holder)
        try:
            cb()
            assert holder["thread_id"] == "freshly-generated-tid"
            assert _ch_mod._cli_thread_id == "freshly-generated-tid"
        finally:
            _ch_mod._cli_thread_id = original_tid_global


def test_start_new_session_cb_leaves_agent_alone():
    """``/new`` rotates thread only — agent handle must stay put
    (serve's agent is a single pre-loaded instance, not per-thread)."""
    holder = {"agent": "a", "thread_id": "old-tid"}

    with patch(
        "EvoScientist.sessions.generate_thread_id",
        return_value="new-tid",
    ):
        cb = _make_serve_start_new_session_cb(holder)
        cb()

    assert holder["agent"] == "a"


def test_hook_handles_both_agent_and_thread_swap():
    """Edge case: a command that changes both (hypothetical). Both
    updates must land in the holder."""
    holder = {"agent": "old-agent", "thread_id": "old-tid"}
    hook = _make_serve_cmd_completed_hook(holder)

    ctx = MagicMock()
    ctx.agent = "new-agent"
    ctx.thread_id = "new-tid"
    cmd = MagicMock()

    _run(hook(ctx, "old-agent", cmd))

    assert holder["agent"] == "new-agent"
    assert holder["thread_id"] == "new-tid"
