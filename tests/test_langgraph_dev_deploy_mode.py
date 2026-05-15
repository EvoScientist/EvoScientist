"""Tests for ``start_langgraph_dev(deploy_mode=...)`` env var injection.

Verifies the mutually-exclusive env var routing:
- ``deploy_mode=True``  → sets ``EVOSCIENTIST_DEPLOY_MODE`` only
- ``deploy_mode=False`` → sets ``EVOSCIENTIST_DEPLOYED_NO_MCP`` only
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from EvoScientist.langgraph_dev import manager


class _PopenAbort(Exception):
    """Raised by the fake ``Popen`` to short-circuit ``start_langgraph_dev``
    after the env dict is constructed but before health-polling runs."""


def _patch_start_prereqs(monkeypatch, tmp_path: Path) -> dict:
    """Mock everything ``start_langgraph_dev`` does before ``subprocess.Popen``
    so we can run it end-to-end up to the point where the env dict is captured.
    Returns a ``captured`` dict that the test populates from the fake Popen."""
    captured: dict = {}

    monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/usr/bin/langgraph")

    fake_config = tmp_path / "langgraph.json"
    fake_config.write_text("{}")
    monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)

    # No conflicts, no stale process — straight to spawn.
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_: False)
    monkeypatch.setattr(manager, "_is_port_occupied", lambda _port: False)
    monkeypatch.setattr(manager, "_wait_for_port_bindable", lambda _port: True)
    monkeypatch.setattr(manager, "_kill_owned_stale_process", lambda _port: False)
    monkeypatch.setattr(
        manager, "_wait_for_port_release", lambda _port, timeout=10.0: True
    )

    # _PID_DIR / _LOG_FILE point under user dir — redirect to tmp.
    pid_dir = tmp_path / "pid_dir"
    monkeypatch.setattr(manager, "_PID_DIR", pid_dir)
    monkeypatch.setattr(manager, "_LOG_FILE", tmp_path / "langgraph_dev.log")

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        captured["cwd"] = kwargs.get("cwd")
        raise _PopenAbort("env captured")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return captured


def test_deploy_mode_true_sets_deploy_env_only(monkeypatch, tmp_path):
    captured = _patch_start_prereqs(monkeypatch, tmp_path)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16174,
            deploy_mode=True,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOY_MODE") == "true", (
        "deploy_mode=True must inject EVOSCIENTIST_DEPLOY_MODE=true"
    )
    assert "EVOSCIENTIST_DEPLOYED_NO_MCP" not in env, (
        "deploy_mode=True must NOT inject EVOSCIENTIST_DEPLOYED_NO_MCP"
    )


def test_deploy_mode_false_default_sets_no_mcp_only(monkeypatch, tmp_path):
    captured = _patch_start_prereqs(monkeypatch, tmp_path)

    with pytest.raises(_PopenAbort):
        # deploy_mode omitted → defaults to False
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16175,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOYED_NO_MCP") == "true", (
        "deploy_mode=False (default) must inject EVOSCIENTIST_DEPLOYED_NO_MCP=true"
    )
    assert "EVOSCIENTIST_DEPLOY_MODE" not in env, (
        "deploy_mode=False must NOT inject EVOSCIENTIST_DEPLOY_MODE"
    )


def test_deploy_mode_explicitly_false_sets_no_mcp_only(monkeypatch, tmp_path):
    """Same as default, but with deploy_mode=False stated explicitly."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16176,
            deploy_mode=False,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOYED_NO_MCP") == "true"
    assert "EVOSCIENTIST_DEPLOY_MODE" not in env


def test_both_env_vars_never_set_simultaneously(monkeypatch, tmp_path):
    """Regression: confirm the two env vars are mutually exclusive in
    every code path."""
    for deploy_mode in (True, False):
        captured = _patch_start_prereqs(monkeypatch, tmp_path)
        with pytest.raises(_PopenAbort):
            manager.start_langgraph_dev(
                workspace_dir=tmp_path,
                port=16177,
                deploy_mode=deploy_mode,
            )
        env = captured["env"]
        has_deploy = "EVOSCIENTIST_DEPLOY_MODE" in env
        has_no_mcp = "EVOSCIENTIST_DEPLOYED_NO_MCP" in env
        assert has_deploy ^ has_no_mcp, (
            f"deploy_mode={deploy_mode}: exactly ONE of "
            "EVOSCIENTIST_DEPLOY_MODE / EVOSCIENTIST_DEPLOYED_NO_MCP must be set "
            f"(deploy={has_deploy}, no_mcp={has_no_mcp})"
        )


def test_workspace_dir_env_var_set_regardless_of_mode(monkeypatch, tmp_path):
    """EVOSCIENTIST_WORKSPACE_DIR is independent of deploy_mode."""
    for deploy_mode in (True, False):
        captured = _patch_start_prereqs(monkeypatch, tmp_path)
        with pytest.raises(_PopenAbort):
            manager.start_langgraph_dev(
                workspace_dir=tmp_path,
                port=16178,
                deploy_mode=deploy_mode,
            )
        assert captured["env"].get("EVOSCIENTIST_WORKSPACE_DIR") == str(tmp_path)


# =============================================================================
# Module-load behavior — _ASYNC_SUBAGENTS_AVAILABLE reads env var on import
# =============================================================================


def test_async_subagents_available_init_from_env_true(monkeypatch):
    """When ``EVOSCIENTIST_DEPLOY_MODE=true`` is set in the env at module
    import time, ``_ASYNC_SUBAGENTS_AVAILABLE`` initializes to True so the
    deployed main agent's ``_maybe_swap_async_subagents`` swaps eagerly
    without waiting for ``start_langgraph_dev`` to flip the flag (which it
    can't — the deploy subprocess never calls that function on itself)."""
    monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "true")
    # Re-import the module to re-run the module-level initialization.
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr

    reloaded = importlib.reload(mgr)
    try:
        assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is True
        assert reloaded.is_async_subagents_available() is True
    finally:
        # Restore: reload again without the env var so subsequent tests
        # see the normal initialization.
        monkeypatch.delenv("EVOSCIENTIST_DEPLOY_MODE", raising=False)
        importlib.reload(mgr)


def test_async_subagents_available_init_false_without_env(monkeypatch):
    """When the env var is unset, ``_ASYNC_SUBAGENTS_AVAILABLE`` initializes
    to False — the pre-existing safety behavior (fall back to sync if
    langgraph dev isn't reachable)."""
    monkeypatch.delenv("EVOSCIENTIST_DEPLOY_MODE", raising=False)
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr

    reloaded = importlib.reload(mgr)
    assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is False
