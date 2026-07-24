"""Tests for ``start_langgraph_dev(deploy_mode=...)`` env var injection.

Verifies the single-env-var enum routing:
- ``deploy_mode=True``  → ``EVOSCIENTIST_DEPLOY_MODE=full``
- ``deploy_mode=False`` → ``EVOSCIENTIST_DEPLOY_MODE=stripped``
- (parent process / plain import) → ``EVOSCIENTIST_DEPLOY_MODE`` unset
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from EvoScientist.langgraph_dev import manager


class _PopenAbort(Exception):
    """Raised by the fake ``Popen`` to short-circuit ``start_langgraph_dev``
    after the env dict is constructed but before health-polling runs."""


def _patch_start_prereqs(monkeypatch, tmp_path: Path, runtime_paths) -> dict:
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

    # Redirect the log file — pid_dir already rooted under tmp via the fixture.
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, log_file=tmp_path / "langgraph_dev.log"),
    )

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        captured["cwd"] = kwargs.get("cwd")
        raise _PopenAbort("env captured")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return captured


def test_deploy_mode_true_sets_full(monkeypatch, tmp_path, runtime_paths):
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16174,
            deploy_mode=True,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOY_MODE") == "full", (
        "deploy_mode=True must inject EVOSCIENTIST_DEPLOY_MODE=full"
    )


def test_deploy_mode_false_default_sets_stripped(monkeypatch, tmp_path, runtime_paths):
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        # deploy_mode omitted → defaults to False
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16175,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOY_MODE") == "stripped", (
        "deploy_mode=False (default) must inject EVOSCIENTIST_DEPLOY_MODE=stripped"
    )


def test_deploy_mode_explicitly_false_sets_stripped(
    monkeypatch, tmp_path, runtime_paths
):
    """Same as default, but with deploy_mode=False stated explicitly."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16176,
            deploy_mode=False,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOY_MODE") == "stripped"


def test_deploy_mode_always_set_to_one_of_full_or_stripped(
    monkeypatch, tmp_path, runtime_paths
):
    """Regression: the subprocess always sees exactly one of the two enum
    values for ``EVOSCIENTIST_DEPLOY_MODE`` — never unset, never garbage."""
    for deploy_mode, expected in ((True, "full"), (False, "stripped")):
        captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
        with pytest.raises(_PopenAbort):
            manager.start_langgraph_dev(
                workspace_dir=tmp_path,
                port=16177,
                deploy_mode=deploy_mode,
            )
        env = captured["env"]
        assert env.get("EVOSCIENTIST_DEPLOY_MODE") == expected, (
            f"deploy_mode={deploy_mode}: expected EVOSCIENTIST_DEPLOY_MODE="
            f"{expected!r}, got {env.get('EVOSCIENTIST_DEPLOY_MODE')!r}"
        )


def test_inherited_stripped_overridden_when_deploy_mode_true(
    monkeypatch, tmp_path, runtime_paths
):
    """If the parent process exports ``EVOSCIENTIST_DEPLOY_MODE=stripped`` and
    we ask for deploy mode, the subprocess env must see the resolved value
    (``full``), not the stale inherited one."""
    monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "stripped")
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16180,
            deploy_mode=True,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOY_MODE") == "full", (
        "inherited stripped value must be overridden when deploy_mode=True"
    )


def test_inherited_full_overridden_when_deploy_mode_false(
    monkeypatch, tmp_path, runtime_paths
):
    """Symmetric: parent exports ``EVOSCIENTIST_DEPLOY_MODE=full``, CLI/serve
    calls start_langgraph_dev with default (deploy_mode=False), inherited
    value must be overridden to ``stripped``."""
    monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "full")
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16181,
        )

    env = captured["env"]
    assert env.get("EVOSCIENTIST_DEPLOY_MODE") == "stripped", (
        "inherited full value must be overridden when deploy_mode=False"
    )


def test_inherited_arbitrary_value_overridden(monkeypatch, tmp_path, runtime_paths):
    """Defense against an unexpected inherited value (e.g. legacy ``true``
    from before the enum rename, or any user-set garbage). The resolved
    deploy_mode always wins."""
    for inherited in ("true", "garbage", "FULL", ""):
        for deploy_mode, expected in ((True, "full"), (False, "stripped")):
            monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", inherited)
            captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

            with pytest.raises(_PopenAbort):
                manager.start_langgraph_dev(
                    workspace_dir=tmp_path,
                    port=16182,
                    deploy_mode=deploy_mode,
                )

            env = captured["env"]
            assert env.get("EVOSCIENTIST_DEPLOY_MODE") == expected, (
                f"inherited={inherited!r}, deploy_mode={deploy_mode}: "
                f"expected EVOSCIENTIST_DEPLOY_MODE={expected!r}, "
                f"got {env.get('EVOSCIENTIST_DEPLOY_MODE')!r}"
            )


def test_workspace_dir_env_var_set_regardless_of_mode(
    monkeypatch, tmp_path, runtime_paths
):
    """EVOSCIENTIST_WORKSPACE_DIR is independent of deploy_mode."""
    for deploy_mode in (True, False):
        captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
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


def test_async_subagents_available_init_from_env_full(monkeypatch):
    """When ``EVOSCIENTIST_DEPLOY_MODE=full`` is set in the env at module
    import time, ``_ASYNC_SUBAGENTS_AVAILABLE`` initializes to True so the
    deployed main agent's ``_maybe_swap_async_subagents`` swaps eagerly
    without waiting for ``start_langgraph_dev`` to flip the flag (which it
    can't — the deploy subprocess never calls that function on itself)."""
    monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "full")
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


def test_async_subagents_available_init_false_for_stripped(monkeypatch):
    """``stripped`` is the CLI/serve subprocess mode — async sub-agents stay
    disabled at module-load time (they get enabled later by ``ensure_langgraph_dev``
    in the parent process, NOT by the subprocess flipping its own flag)."""
    monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "stripped")
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr

    reloaded = importlib.reload(mgr)
    try:
        assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is False
    finally:
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


def test_tunnel_true_appends_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel=True`` must add ``--tunnel`` to the langgraph dev argv."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16190,
            tunnel=True,
        )

    assert "--tunnel" in captured["args"]


def test_tunnel_false_default_omits_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel`` defaults to False — no ``--tunnel`` in the argv."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16191,
        )

    assert "--tunnel" not in captured["args"]
