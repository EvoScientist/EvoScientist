"""Tests for the environment ``_spawn_server`` gives the server.

The server always runs with ``EVOSCIENTIST_SERVER_PROCESS`` set; the parent
process and a plain ``import EvoScientist`` leave it unset.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from EvoScientist.langgraph_dev import boundary, manager
from tests.fakes.langgraph_dev import PopenAbort, patch_start_prereqs


@pytest.mark.parametrize("inherited", [None, "", "0"])
def test_start_sets_server_process_env(monkeypatch, tmp_path, runtime_paths, inherited):
    """Whatever the caller's shell exports, the server process is marked."""
    if inherited is None:
        monkeypatch.delenv("EVOSCIENTIST_SERVER_PROCESS", raising=False)
    else:
        monkeypatch.setenv("EVOSCIENTIST_SERVER_PROCESS", inherited)
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16174)

    assert captured["env"]["EVOSCIENTIST_SERVER_PROCESS"] == "1"


def test_start_runs_server_in_server_home(monkeypatch, tmp_path, runtime_paths):
    """The server serves every registered project, so it gets no workspace of
    its own. The user's ``EVOSCIENTIST_WORKSPACE_DIR`` (Docker sets it) passes
    as is."""
    from EvoScientist import paths

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", "/inherited")
    monkeypatch.chdir(tmp_path)
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178)

    env = captured["env"]
    assert env["EVOSCIENTIST_WORKSPACE_DIR"] == "/inherited"
    assert captured["cwd"] == str(paths.SERVER_HOME)
    if os.name == "posix":
        assert paths.SERVER_HOME.stat().st_mode & 0o777 == 0o700


def test_start_gives_server_no_dotenv_keys(monkeypatch, tmp_path, runtime_paths):
    """Every agent takes its keys from config.yaml and the environment: a
    project ``.env`` key reaches neither this process nor the server."""
    from EvoScientist.config import settings

    dotenv = tmp_path / ".env"
    dotenv.write_text("TAVILY_API_KEY=tvly-dotenv\nOPENAI_API_KEY=sk-dotenv\n")
    monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-shell")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings.get_effective_config()
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178)

    env = captured["env"]
    assert env["TAVILY_API_KEY"] == "tvly-shell"
    assert "OPENAI_API_KEY" not in env


@pytest.mark.parametrize(("lasting", "passed"), [(True, False), (False, True)])
def test_session_flag_reaches_only_server_it_owns(
    monkeypatch, tmp_path, runtime_paths, lasting, passed
):
    """A lasting server serves later sessions and every project: this
    session's --dangerous (or --debug) does not reach it; the auth mode does
    (test_auth_mode_flag_reaches_lasting_server). A server EvoSci deploy or
    the desktop app owns gets its starter's flags."""
    from EvoScientist.config import settings

    monkeypatch.setenv("EVOSCIENTIST_DANGEROUS_MODE", "true")
    settings.record_env_origin("EVOSCIENTIST_DANGEROUS_MODE", "cli")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16192, lasting=lasting)

    env = captured["env"]
    origins = json.loads(env[settings.ENV_ORIGINS_VAR])
    assert ("EVOSCIENTIST_DANGEROUS_MODE" in env) is passed
    assert ("EVOSCIENTIST_DANGEROUS_MODE" in origins) is passed


def test_auth_mode_flag_reaches_lasting_server(monkeypatch, tmp_path, runtime_paths):
    """One server has one set of credentials: the session that starts it
    picks them."""
    from EvoScientist.config import settings

    monkeypatch.setenv("EVOSCIENTIST_OPENAI_AUTH_MODE", "oauth")
    settings.record_env_origin("EVOSCIENTIST_OPENAI_AUTH_MODE", "cli")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16194)

    assert captured["env"]["EVOSCIENTIST_OPENAI_AUTH_MODE"] == "oauth"


@pytest.mark.parametrize(
    ("export", "flag", "reaches"),
    [("1", False, True), ("yes", False, True), (None, True, False)],
)
def test_dangerous_mode_reaching_lasting_server(
    monkeypatch, tmp_path, runtime_paths, export, flag, reaches
):
    """Through the session's own config resolution: the user's export
    reaches the server however it is spelled; a --dangerous flag does not."""
    from EvoScientist.config import settings

    monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: "")
    if export is None:
        monkeypatch.delenv("EVOSCIENTIST_DANGEROUS_MODE", raising=False)
    else:
        monkeypatch.setenv("EVOSCIENTIST_DANGEROUS_MODE", export)
    config = settings.get_effective_config({"dangerous_mode": True} if flag else None)
    assert config.dangerous_mode
    settings.apply_config_to_env(config)
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16193)

    assert ("EVOSCIENTIST_DANGEROUS_MODE" in captured["env"]) is reaches


def test_start_sets_the_browser_boundary(monkeypatch, tmp_path, runtime_paths):
    """Set on every start, so nothing inherited (a workspace ``.env``) widens
    it; cloudflared inherits the server's environment."""
    monkeypatch.setenv("CORS_CONFIG", '{"allow_origins": ["*"]}')
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("TUNNEL_HTTP_HOST_HEADER", "localhost")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(
            port=16178,
            allowed_origins="https://smith.langchain.com",
        )
    env = captured["env"]
    cors = json.loads(env["CORS_CONFIG"])
    assert cors["allow_origins"] == ["https://smith.langchain.com"]
    assert json.loads(env[boundary.BOUNDARY_ENV])["origins"] == cors["allow_origins"]
    assert "CORS_ALLOW_ORIGINS" not in env
    assert "TUNNEL_HTTP_HOST_HEADER" not in env


def test_start_passes_this_process_data_folder(monkeypatch, tmp_path, runtime_paths):
    """The server runs in the workspace and loads its ``.env``; a relative or
    later-changed variable must not give it another folder."""
    from EvoScientist import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", "rel-data")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178)
    assert captured["env"]["EVOSCIENTIST_DATA_DIR"] == str(tmp_path / "data")


def test_start_trusts_the_bind_host(monkeypatch, tmp_path, runtime_paths):
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178, host="myhost.local")
    names = json.loads(captured["env"][boundary.BOUNDARY_ENV])["names"]
    assert "myhost.local" in names


def test_ensure_passes_backend_allowed_origins(monkeypatch, tmp_path, runtime_paths):
    from EvoScientist.config import EvoScientistConfig
    from tests.fakes.langgraph_dev import Server

    server = Server(monkeypatch)
    config = EvoScientistConfig(
        langgraph_dev_port=16178, backend_allowed_origins="https://smith.langchain.com"
    )
    manager.ensure_server(config, backend="langgraph_server")
    assert server.spawns[0]["allowed_origins"] == "https://smith.langchain.com"


# =============================================================================
# Bind host — argv flag + env propagation
# =============================================================================


def test_host_defaults_to_loopback_in_argv(monkeypatch, tmp_path, runtime_paths):
    """``--host`` must always be emitted rather than left to the langgraph
    CLI's own default, so the bind stays pinned to _DEFAULT_HOST even if that
    default moves."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178)

    args = captured["args"]
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_explicit_wildcard_host_reaches_argv(monkeypatch, tmp_path, runtime_paths):
    """The opt-in to a public bind has to survive all the way into argv."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178, host="0.0.0.0")

    args = captured["args"]
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_env_carries_explicit_bind_host(monkeypatch, tmp_path, runtime_paths):
    """The subprocess resolves its self-dispatch URL from config, so the
    caller-resolved host must be injected — mirrors the port propagation."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178, host="192.168.1.5")

    assert captured["env"].get("EVOSCIENTIST_LANGGRAPH_DEV_HOST") == "192.168.1.5"


def test_env_host_replaces_inherited(monkeypatch, tmp_path, runtime_paths):
    """A stray export in the user's shell must not override the host this
    caller resolved — otherwise the bind and the dispatch URL desync."""
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_HOST", "10.0.0.9")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16178, host="0.0.0.0")

    assert captured["env"].get("EVOSCIENTIST_LANGGRAPH_DEV_HOST") == "0.0.0.0"


# =============================================================================
# Module-load behavior — _ASYNC_SUBAGENTS_AVAILABLE reads env var on import
# =============================================================================


def test_async_subagents_available_init_true_in_server(monkeypatch):
    """In the server (``paths.SERVER_PROCESS``, read from
    ``EVOSCIENTIST_SERVER_PROCESS`` at import), ``_ASYNC_SUBAGENTS_AVAILABLE``
    initializes to True so the
    deployed main agent's ``_maybe_swap_async_subagents`` swaps eagerly
    without waiting for ``ensure_server`` to flip the flag (which it
    can't — the server never calls that function on itself)."""
    # Re-import the module to re-run the module-level initialization.
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr
    from EvoScientist import paths

    monkeypatch.setattr(paths, "SERVER_PROCESS", True)
    reloaded = importlib.reload(mgr)
    try:
        assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is True
        assert reloaded.is_async_subagents_available() is True
    finally:
        # Restore: reload again as a CLI process so subsequent tests see
        # the normal initialization.
        monkeypatch.setattr(paths, "SERVER_PROCESS", False)
        importlib.reload(mgr)


def test_async_subagents_available_init_false_without_env(monkeypatch):
    """In a CLI process, ``_ASYNC_SUBAGENTS_AVAILABLE`` initializes to False —
    the pre-existing safety behavior (fall back to sync if langgraph dev
    isn't reachable)."""
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr
    from EvoScientist import paths

    monkeypatch.setattr(paths, "SERVER_PROCESS", False)

    reloaded = importlib.reload(mgr)
    assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is False


def test_tunnel_true_appends_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel=True`` must add ``--tunnel`` to the langgraph dev argv."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(PopenAbort):
        manager._spawn_server(
            port=16190,
            tunnel=True,
        )

    assert "--tunnel" in captured["args"]


def test_tunnel_false_default_omits_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel`` defaults to False — no ``--tunnel`` in the argv."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(PopenAbort):
        manager._spawn_server(
            port=16191,
        )

    assert "--tunnel" not in captured["args"]


def test_node_for_mcp_servers_is_installed_before_start(monkeypatch):
    from EvoScientist.mcp import client as mcp_client

    servers = {"fs": {"transport": "stdio", "command": "npx"}}
    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: servers)
    installed: list = []
    monkeypatch.setattr(mcp_client, "_ensure_node_for_stdio", installed.append)

    manager._install_node_for_mcp(manager._Waits(None, None))

    assert installed == [servers]


def test_failing_node_check_does_not_block_start(monkeypatch, caplog):
    from EvoScientist.mcp import client as mcp_client

    def _fail(_config):
        raise RuntimeError("no network")

    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: {"fs": None})
    monkeypatch.setattr(mcp_client, "_ensure_node_for_stdio", _fail)

    manager._install_node_for_mcp(manager._Waits(None, None))

    assert "Could not check Node.js" in caplog.text


def test_ensure_installs_node_only_for_start(monkeypatch, runtime_paths):
    from tests.fakes.langgraph_dev import Server

    server = Server(monkeypatch)
    installs: list = []
    monkeypatch.setattr(
        manager, "_install_node_for_mcp", lambda waits: installs.append(1)
    )
    config = manager.EvoScientistConfig(enable_async_subagents=True)

    manager.ensure_server(config)
    manager.ensure_server(config)

    assert installs == [1]
    assert len(server.spawns) == 1


@pytest.mark.parametrize(("lasting", "flag"), [(True, "1"), (False, "0")])
def test_start_passes_server_what_it_records(
    monkeypatch, tmp_path, runtime_paths, lasting, flag
):
    """The server records its starter's PID, whether it outlives it, and the
    configuration it was started for, which it cannot know itself."""
    from EvoScientist.langgraph_dev import server_record

    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16179, lasting=lasting, config_fingerprint="fp1")

    env = captured["env"]
    assert env[server_record.SPAWNER_PID_VAR] == str(os.getpid())
    assert env[server_record.LASTING_VAR] == flag
    assert env[server_record.FINGERPRINT_VAR] == "fp1"


def test_start_passes_no_inherited_fingerprint(monkeypatch, tmp_path, runtime_paths):
    from EvoScientist.langgraph_dev import server_record

    monkeypatch.setenv(server_record.FINGERPRINT_VAR, "stale")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16179)

    assert server_record.FINGERPRINT_VAR not in captured["env"]


@pytest.mark.parametrize("os_name", ["nt", "posix"])
def test_start_knows_server_leads_its_process_group_only_on_windows(
    monkeypatch, tmp_path, runtime_paths, os_name
):
    """Only a server spawned as a group leader is sent Ctrl+Break on stop:
    for any other PID Windows sends it to the whole console."""
    patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(manager, "_BREAK_GROUP_PIDS", set())
    monkeypatch.setattr(manager.os, "name", os_name)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_kw: SimpleNamespace(pid=4242)
    )

    manager._spawn_server(port=16179)

    assert (4242 in manager._BREAK_GROUP_PIDS) is (os_name == "nt")


def test_ensure_stops_unhealthy_server_without_clean_stop(monkeypatch, runtime_paths):
    """A server that never answered has no run to put back in the queue."""
    from tests.fakes.langgraph_dev import Server

    server = Server(monkeypatch)
    server.records = False
    stops: list = []
    monkeypatch.setattr(
        manager, "stop_langgraph_dev", lambda *a, **kw: stops.append((a, kw))
    )

    result = manager.ensure_server(
        manager.EvoScientistConfig(enable_async_subagents=True)
    )

    assert result.problem == "start_failed"
    assert stops == [((server.procs[0],), {"clean": False})]


def test_start_puts_the_server_in_the_callers_job(monkeypatch, tmp_path, runtime_paths):
    """The server is spawned with the job's flags and joins the job right
    after the spawn (``started`` stops the run here)."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(manager, "_PROCESS", None)
    proc = SimpleNamespace(pid=4242)

    def _popen(args, **kwargs):
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)

    class _Job:
        creationflags = 0x4

        def started(self, process):
            captured["started"] = process
            # Already recorded, so a caller's stop reaches it if this fails.
            captured["recorded"] = manager._PROCESS
            raise PopenAbort("joined")

    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16179, job=_Job())
    assert captured["kwargs"]["creationflags"] & 0x4
    assert captured["started"] is proc
    assert captured["recorded"] is proc


def test_start_without_a_job_keeps_the_spawn_flags(
    monkeypatch, tmp_path, runtime_paths
):
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    def _popen(args, **kwargs):
        captured["kwargs"] = kwargs
        raise PopenAbort("spawned")

    monkeypatch.setattr(subprocess, "Popen", _popen)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16179)
    if manager.os.name == "nt":
        assert captured["kwargs"]["creationflags"] == (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        assert captured["kwargs"]["start_new_session"] is True
        assert "creationflags" not in captured["kwargs"]


def test_start_passes_env_origins_to_the_server_only(
    monkeypatch, tmp_path, runtime_paths
):
    """The record of where config variables came from reaches the server's
    environment, never this process's (so no other child inherits it)."""
    from EvoScientist.config import settings

    monkeypatch.setattr(
        settings, "_ENV_ORIGINS", {"OPENAI_API_KEY": {"source": "file"}}
    )
    monkeypatch.delenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", raising=False)
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16190)

    origins = json.loads(captured["env"][settings.ENV_ORIGINS_VAR])
    assert origins["OPENAI_API_KEY"] == {"source": "file"}
    # Not in the file, not the default port: a --port flag.
    assert origins["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == {"source": "cli"}
    assert settings.ENV_ORIGINS_VAR not in os.environ


def test_env_origins_leave_the_users_own_export_alone(
    monkeypatch, tmp_path, runtime_paths
):
    from EvoScientist.config import settings

    monkeypatch.setattr(settings, "_ENV_ORIGINS", {})
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "16191")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager._spawn_server(port=16191)

    origins = json.loads(captured["env"][settings.ENV_ORIGINS_VAR])
    assert "EVOSCIENTIST_LANGGRAPH_DEV_PORT" not in origins
