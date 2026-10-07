"""Tests for the headless standalone channel runner's dev-server startup."""

from __future__ import annotations

import os
from types import SimpleNamespace

import EvoScientist.channels.standalone as standalone


def _patch_manager(monkeypatch, *, gateway_backend, default_workdir):
    """Stub the dev-server spawn + workspace helpers; return (config, ensure_calls).

    ``ensure_calls`` captures each ``ensure_langgraph_dev`` invocation so a test
    can assert whether (and with what workspace) the dev server was ensured.
    """
    import EvoScientist.langgraph_dev.manager as manager_mod
    import EvoScientist.paths as paths_mod

    ensure_calls: list[dict] = []
    monkeypatch.setattr(
        manager_mod,
        "ensure_langgraph_dev",
        lambda cfg, *, workspace_dir, backend=None: ensure_calls.append(
            {"config": cfg, "workspace_dir": workspace_dir, "backend": backend}
        ),
    )
    # Avoid mutating the real process-global workspace / creating dirs.
    monkeypatch.setattr(paths_mod, "set_workspace_root", lambda path: None)
    monkeypatch.setattr(paths_mod, "ensure_dirs", lambda: None)
    config = SimpleNamespace(
        gateway_backend=gateway_backend,
        default_workdir=default_workdir,
    )
    return config, ensure_calls


def test_ensure_dev_server_spawns_on_server_backend(monkeypatch, tmp_path):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )

    standalone._ensure_standalone_dev_server(config, backend="langgraph_server")

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == os.path.abspath(str(tmp_path))
    # The resolved backend is forwarded so the manager spawns in full mode.
    assert ensure_calls[0]["backend"] == "langgraph_server"


def test_ensure_dev_server_noop_on_local_backend(monkeypatch, tmp_path):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="local", default_workdir=str(tmp_path)
    )

    standalone._ensure_standalone_dev_server(config, backend="local")

    assert ensure_calls == []


def test_ensure_dev_server_falls_back_to_cwd(monkeypatch):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=""
    )

    standalone._ensure_standalone_dev_server(config, backend="langgraph_server")

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == os.getcwd()


def test_run_standalone_ensures_dev_server_only_with_agent(monkeypatch):
    """``run_standalone`` resolves config and ensures the dev server iff use_agent."""
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(gateway_backend="local", default_workdir=""),
    )
    ensure_configs: list[object] = []
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda cfg, *, backend=None: ensure_configs.append(cfg),
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)
    assert len(ensure_configs) == 1

    ensure_configs.clear()
    standalone.run_standalone(channel=None, bus=None, use_agent=False)
    assert ensure_configs == []


def test_run_standalone_starts_ccproxy_before_dev_server(monkeypatch):
    """OAuth standalone must bootstrap ccproxy before spawning the dev server.

    The dev server builds its child env from os.environ.copy(), so the
    ANTHROPIC_*/OPENAI_* vars written by maybe_start_ccproxy have to exist
    before _ensure_standalone_dev_server runs (din0s, PR #571).
    """
    import EvoScientist.ccproxy_manager as ccproxy_mod
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(
            gateway_backend="langgraph_server",
            default_workdir="",
            anthropic_auth_mode="oauth",
            openai_auth_mode="api_key",
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        ccproxy_mod,
        "maybe_start_ccproxy",
        lambda cfg: calls.append("ccproxy") or None,
    )
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda cfg, *, backend=None: calls.append("dev_server"),
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert calls == ["ccproxy", "dev_server"]


def test_run_standalone_skips_ccproxy_without_agent(monkeypatch):
    """``use_agent=False`` never touches ccproxy (config block is skipped)."""
    import EvoScientist.ccproxy_manager as ccproxy_mod
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(gateway_backend="local", default_workdir=""),
    )
    ccproxy_calls: list[object] = []
    monkeypatch.setattr(
        ccproxy_mod,
        "maybe_start_ccproxy",
        lambda cfg: ccproxy_calls.append(cfg),
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=False)

    assert ccproxy_calls == []


def test_run_standalone_registers_ccproxy_shutdown(monkeypatch):
    """A ccproxy handle returned at startup is registered with atexit."""
    import atexit

    import EvoScientist.ccproxy_manager as ccproxy_mod
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(
            gateway_backend="local",
            default_workdir="",
            anthropic_auth_mode="oauth",
            openai_auth_mode="api_key",
        ),
    )
    fake_proc = object()
    monkeypatch.setattr(ccproxy_mod, "maybe_start_ccproxy", lambda cfg: fake_proc)
    registered: list[tuple] = []
    monkeypatch.setattr(
        atexit, "register", lambda fn, *args: registered.append((fn, args))
    )
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda cfg, *, backend=None: None,
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(registered) == 1
    fn, args = registered[0]
    assert fn is ccproxy_mod.stop_ccproxy
    assert args == (fake_proc,)
