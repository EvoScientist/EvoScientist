"""Shared fixtures for EvoScientist tests."""

import errno
import ipaddress
import os
import shutil
import socket
import sys
import tempfile
from asyncio.proactor_events import BaseProactorEventLoop
from pathlib import Path

import pytest

_NONEXISTENT_DOTENV = str(Path(__file__).with_name(".pytest-dotenv-does-not-exist"))
_test_home: Path | None = None


def pytest_configure(config):
    """Point the home folder at an empty one before EvoScientist is imported.

    ``EvoScientist.paths`` resolves the data and config folders from it at
    import, and the import itself migrates files there, so a fixture would
    act too late. ``XDG_CONFIG_HOME`` and ``EVOSCIENTIST_DATA_DIR`` would
    override it, so they go too.
    """
    global _test_home
    if "EvoScientist" in sys.modules:
        raise pytest.UsageError(
            "EvoScientist was imported before tests/conftest.py could isolate "
            "the home folder; import it inside fixtures or tests instead"
        )
    _test_home = Path(tempfile.mkdtemp(prefix="evosci-test-home-"))
    os.environ["HOME"] = str(_test_home)
    os.environ["USERPROFILE"] = str(_test_home)
    for name in ("XDG_CONFIG_HOME", "EVOSCIENTIST_DATA_DIR"):
        os.environ.pop(name, None)


def pytest_unconfigure(config):
    if _test_home is not None:
        shutil.rmtree(_test_home, ignore_errors=True)


_CLEARED_ENV_PREFIXES = ("EVOSCIENTIST_", "LANGCHAIN_", "LANGSMITH_")
_CLEARED_ENV_SUFFIXES = ("_API_KEY", "_BASE_URL")
_CLEARED_ENV_NAMES = frozenset(
    {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
    | {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
)


def _cleared_env_name(name: str) -> bool:
    from EvoScientist.config.settings import _ENV_MAPPINGS

    return (
        name in _CLEARED_ENV_NAMES
        or name in _ENV_MAPPINGS.values()
        or name.upper().startswith(_CLEARED_ENV_PREFIXES)
        or name.upper().endswith(_CLEARED_ENV_SUFFIXES)
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Start every test without the developer's provider, EvoScientist,
    tracing and proxy settings, and undo whatever it writes to the
    environment. Production code (``apply_config_to_env``, the ccproxy setup)
    assigns ``os.environ`` directly, which monkeypatch cannot undo.

    A hook, not a fixture: it restores after every fixture, monkeypatch
    included, has been torn down. Scoped fixtures must not set environment
    variables; the restore after each test removes them.
    """
    saved = dict(os.environ)
    for name in [name for name in os.environ if _cleared_env_name(name)]:
        del os.environ[name]
    # Child processes are out of the socket guard's reach: git, uv and pip
    # refuse the network at once instead of waiting out a timeout.
    os.environ.update(
        {"GIT_ALLOW_PROTOCOL": "file", "UV_OFFLINE": "1", "PIP_NO_INDEX": "1"}
    )
    try:
        return (yield)
    finally:
        for name in [name for name in os.environ if name not in saved]:
            del os.environ[name]
        for name, value in saved.items():
            if os.environ.get(name) != value:
                os.environ[name] = value


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Refuse every connection a test opens to another machine, and fail the
    test that tried; loopback stays open for the servers tests start.

    Nothing answers on the langgraph dev port either, as on CI: a developer's
    running server would otherwise serve the tests its threads and schedules.
    A refusal raises, or returns from ``connect_ex``, as a closed port would,
    so code that handles an offline network carries on.

    uvloop connects in C, past both patches; a test that installs it is not
    guarded.
    """
    from EvoScientist import update_check
    from EvoScientist.langgraph_dev.sdk import DEFAULT_LANGGRAPH_DEV_PORT

    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sock_connect = BaseProactorEventLoop.sock_connect

    def _refused(sock: socket.socket, address) -> bool:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return False
        host, port = str(address[0]), address[1]
        if not _is_loopback(host):
            attempts.append(f"{host}:{port}")
            return True
        return port == DEFAULT_LANGGRAPH_DEV_PORT

    def connect(sock, address):
        if _refused(sock, address):
            raise ConnectionRefusedError(errno.ECONNREFUSED, "refused by the tests")
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if _refused(sock, address):
            return errno.ECONNREFUSED
        return real_connect_ex(sock, address)

    # Windows' default event loop connects through ConnectEx, never calling
    # socket.connect, so async clients are checked here instead.
    async def sock_connect(loop, sock, address):
        if _refused(sock, address):
            raise ConnectionRefusedError(errno.ECONNREFUSED, "refused by the tests")
        return await real_sock_connect(loop, sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(BaseProactorEventLoop, "sock_connect", sock_connect)
    # The TUI and CLI check PyPI for a newer release at start.
    monkeypatch.setattr(update_check, "is_update_available", lambda: (False, None))
    yield
    assert not attempts, f"the test tried to reach the network: {attempts}"


@pytest.fixture
def sample_tool_call():
    """A minimal tool call dict."""
    return {"id": "tc_001", "name": "execute", "args": {"command": "ls -la"}}


@pytest.fixture
def sample_tool_result():
    """A minimal tool result dict."""
    return {
        "id": "tc_001",
        "name": "execute",
        "content": "[OK] file1.py file2.py",
        "success": True,
    }


@pytest.fixture
def sample_events():
    """A sequence of stream event dicts covering common types."""
    return [
        {"type": "thinking", "content": "Let me think..."},
        {"type": "text", "content": "Here is the answer."},
        {
            "type": "tool_call",
            "id": "tc_001",
            "name": "execute",
            "args": {"command": "ls"},
        },
        {
            "type": "tool_result",
            "id": "tc_001",
            "name": "execute",
            "content": "[OK] done",
            "success": True,
        },
        {
            "type": "subagent_start",
            "name": "research-agent",
            "description": "Find papers",
            "instance_id": "task:research",
            "tool_call_id": "tc_task_001",
        },
        {
            "type": "subagent_tool_call",
            "subagent": "research-agent",
            "instance_id": "task:research",
            "name": "tavily_search",
            "args": {"query": "test"},
            "id": "tc_sa_001",
        },
        {
            "type": "subagent_tool_result",
            "subagent": "research-agent",
            "instance_id": "task:research",
            "name": "tavily_search",
            "content": "Results...",
            "success": True,
            "id": "tc_sa_001",
        },
        {
            "type": "subagent_end",
            "name": "research-agent",
            "instance_id": "task:research",
        },
        {"type": "done", "response": "Here is the answer."},
    ]


@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a temporary workspace directory path."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture(autouse=True)
def _fresh_langgraph_dev_state(monkeypatch):
    """Start every test with no server handle and no drift recorded.

    ``langgraph_dev.manager`` keeps the server it started and what it found
    in module globals; a test that leaves one set (a fake ``Popen`` among
    them) changes what the next test's ``ensure_langgraph_dev`` does.
    """
    from EvoScientist.langgraph_dev import manager

    for name, value in {
        "_PROCESS": None,
        "_PROCESS_WORKSPACE": None,
        "_PROCESS_RUN_DIR": None,
        "SERVED_WORKSPACE": None,
        "CONFIG_DRIFT_SINCE_LAUNCH": False,
        "AGENT_SHELL_DRIFT": None,
        "_ASYNC_SUBAGENTS_AVAILABLE": False,
        "_LOG_OFFSET_AT_START": 0,
    }.items():
        monkeypatch.setattr(manager, name, value)


@pytest.fixture(autouse=True)
def _quickjs_snapshots_without_diffs(monkeypatch):
    """Store every QuickJS snapshot whole, as ``encode_snapshot`` itself does
    when a diff would not pay off. Diffing it against the previous turn's
    costs seconds per turn of a real graph, and no test reads the diff."""
    from langchain_quickjs import _snapshot, middleware

    monkeypatch.setattr(
        middleware,
        "encode_snapshot",
        lambda payload, prior: (_snapshot.SNAP, payload),
    )


@pytest.fixture(autouse=True)
def _no_health_server(monkeypatch):
    """Channel runners start no health server unless a test asks for a port.

    The default binds ``0.0.0.0:8080``: open to the LAN while tests run, and
    shared by every test that starts a runner at the same time.
    """
    from EvoScientist.channels.channel_manager import ChannelManager

    monkeypatch.setitem(ChannelManager.__init__.__kwdefaults__, "health_port", 0)


@pytest.fixture(autouse=True)
def _fresh_process_workspace():
    """``process_workspace()`` is read once per process; tests change env/cwd."""
    from EvoScientist.paths import process_session_dirs, process_workspace

    process_workspace.cache_clear()
    process_session_dirs.cache_clear()
    yield
    process_workspace.cache_clear()
    process_session_dirs.cache_clear()


@pytest.fixture
def no_mcp(monkeypatch):
    """For tests that build the real agent: it would load the user's MCP servers,
    which cannot start inside a running loop. Opt in with ``usefixtures``."""
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._load_mcp_tools_cached", lambda *a, **k: {}
    )


@pytest.fixture
def workspace(tmp_path):
    """A ``Workspace`` rooted at a fresh temporary folder."""
    from EvoScientist.paths import Workspace

    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root)


@pytest.fixture
def runtime_paths(tmp_path, monkeypatch):
    """Isolate ``langgraph_dev.manager.RUNTIME`` under a temp directory.

    Replaces the module-level ``RUNTIME`` with a fully temp-rooted bundle
    so every path (``pid_dir``, ``pid_file``, ``log_file``,
    ``workspace_sidecar``, ``lock_file``) is contained under ``tmp_path``.

    Tests that need a variant of a single field can still call
    ``dataclasses.replace(runtime_paths, log_file=…)`` etc. — the
    baseline is already isolated, so forgetting a field just keeps it
    under ``tmp_path``, never ``~/.config/evoscientist``.
    """
    from EvoScientist.langgraph_dev import manager

    runtime = manager.LanggraphRuntimePaths.for_directory(tmp_path / "runtime")
    monkeypatch.setattr(manager, "RUNTIME", runtime)
    return runtime


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch):
    """Keep the developer's real .env out of the test environment.

    ``get_effective_config`` merges the cwd .env into ``os.environ``, which
    would leak into every later test; a nonexistent path makes it a no-op.
    """
    monkeypatch.setattr(
        "EvoScientist.config.settings.find_dotenv",
        lambda *args, **kwargs: _NONEXISTENT_DOTENV,
    )


@pytest.fixture(autouse=True)
def _isolate_user_dirs(tmp_path_factory, monkeypatch):
    """Keep the developer's real data and config folders out of every test.

    Both are read once at import (``paths.DATA_DIR``, ``paths.CONFIG_DIR``), so
    setting ``EVOSCIENTIST_DATA_DIR`` or ``XDG_CONFIG_HOME`` in a test does not
    move them; point them, and the copies taken at import, at fresh folders.
    """
    from EvoScientist import paths, update_check
    from EvoScientist.cli import tui_interactive
    from EvoScientist.mcp import client as mcp_client

    root = tmp_path_factory.mktemp("user")
    data_dir = root / ".evoscientist"
    config_dir = root / "config" / "evoscientist"
    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", data_dir / "skills")
    monkeypatch.setattr(paths, "GLOBAL_MEMORIES_DIR", data_dir / "memories")
    monkeypatch.setattr(paths, "MEMORIES_DIR", data_dir / "memories")
    monkeypatch.setattr(paths, "SERVER_HOME", data_dir / "server")
    monkeypatch.setattr(tui_interactive, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(mcp_client, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(mcp_client, "USER_MCP_CONFIG", config_dir / "mcp.yaml")
    monkeypatch.setattr(update_check, "CACHE_DIR", config_dir)
    monkeypatch.setattr(update_check, "CACHE_FILE", config_dir / "latest_version.json")


@pytest.fixture(autouse=True)
def _no_agent_python_probe(monkeypatch):
    """Keep the agent-shell python decision away from the real machine.

    It is cached per process and, unstubbed, any test that builds a backend,
    launches a background job or starts a server would run a real ``python``
    probe against the developer's PATH (or a test's faked ``subprocess`` /
    ``shutil.which``) and read the real research environment. Pinned to "no
    python, no environment"; ``tests/test_setup_research_env.py`` restores
    the real functions for its own tests.
    """
    from EvoScientist.setup import research_env

    monkeypatch.setattr(research_env, "_agent_python", lambda: (None, None))


@pytest.fixture(autouse=True)
def _no_uv_probes(monkeypatch):
    """Keep uv lookups away from the real machine: no ``uv ... --help`` run
    (a test's faked ``subprocess`` would record it), and no uv recorded in
    the developer's data dir, only ``uv`` on PATH (a test's faked
    ``shutil.which``). ``tests/test_uv_tool.py`` tests the real functions."""
    import shutil

    from EvoScientist import uv_tool

    monkeypatch.setattr(uv_tool, "_knows_system_certs", lambda uv: True)
    monkeypatch.setattr(uv_tool, "find_uv", lambda: shutil.which("uv"))


@pytest.fixture(autouse=True)
def _no_agent_bash(monkeypatch):
    """Keep the agent-shell bash decision away from the real machine.

    It is cached per process and reads ``tools/git.json`` from the real data
    dir, so on a Windows developer machine every ``execute`` test would run in
    that Git Bash. Pinned to "no bash" (``cmd.exe`` on Windows, ``/bin/sh``
    elsewhere); ``tests/test_agent_shell.py`` opts into a real bash. The setup
    hint leaves out the Git Bash part, so hint tests read the same on every
    OS, and its once-per-process log is silenced
    (``tests/test_setup_research_env.py`` restores it).
    """
    from EvoScientist import agent_shell

    monkeypatch.setattr(agent_shell, "agent_bash", lambda: None)
    monkeypatch.setattr(agent_shell, "_bash_missing", lambda: False)
    monkeypatch.setattr(agent_shell, "log_setup_hint", lambda: None)


@pytest.fixture
def no_git(tmp_path, monkeypatch):
    """Run as if git were not installed.

    ``EvoScientist.git_cli.subprocess.run`` is the process-wide
    ``subprocess.run``, so the fake asserts the command is git: a test must not
    pass because some other subprocess call failed. ``DATA_DIR`` points at an
    empty dir, so on Windows the retry with a recorded PortableGit finds none
    instead of changing this process's ``PATH``.
    """
    from unittest.mock import patch

    from EvoScientist import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "no-git-data")

    def fake_run(argv, *args, **kwargs):
        assert argv[0] == "git", f"unexpected subprocess.run call: {argv!r}"
        raise FileNotFoundError(2, "No such file or directory", "git")

    with patch("EvoScientist.git_cli.subprocess.run", side_effect=fake_run):
        yield
