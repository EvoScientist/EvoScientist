"""langgraph dev lifecycle management for async sub-agent support.

Provides functions to start/stop/health-check a ``langgraph dev`` subprocess
that hosts the EvoScientist main agent and async sub-agents (e.g.
``writing-agent``). The CLI calls ``ensure_langgraph_dev(config, ...)`` at
startup so users can run ``EvoSci -p "..."`` without manually managing the
langgraph dev server.

Mirrors the lifecycle pattern used by ``ccproxy_manager.py``.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import httpx

from EvoScientist.config import EvoScientistConfig

logger = logging.getLogger(__name__)


# Reentrant lock guarding ``_PROCESS`` / ``_PROCESS_WORKSPACE`` /
# ``_ASYNC_SUBAGENTS_AVAILABLE`` mutations and the ``ensure_langgraph_dev``
# decision/start/stop flow. Reentrant because ``ensure_langgraph_dev`` can call
# ``stop_langgraph_dev`` from inside its own critical section during a
# workspace-driven restart, and both mutate the same module-level state.
_LOCK = threading.RLock()


# Default port (Kaprekar's constant — see config/settings.py for the rationale).
# Overridable per-call via ``start_langgraph_dev(port=...)`` /
# ``ensure_langgraph_dev`` (which reads ``config.langgraph_dev_port``) and the
# corresponding url= field on AsyncSubAgent specs.
_DEFAULT_PORT = 6174


def _base_url(port: int = _DEFAULT_PORT) -> str:
    return f"http://localhost:{port}"


_PID_DIR = Path.home() / ".config" / "evoscientist"
_PID_FILE = _PID_DIR / "langgraph_dev.pid"
_LOG_FILE = _PID_DIR / "langgraph_dev.log"

# Module-level handle to the langgraph dev subprocess we started, if any.
# Stays None when we reused an existing process (managed by the user).
_PROCESS: subprocess.Popen | None = None

# Workspace directory the running subprocess was launched with. Used by
# ``ensure_langgraph_dev`` to detect a workspace switch (e.g., on /resume of
# a thread from a different workspace) and trigger a restart so the deployed
# sub-agents' cwd / EVOSCIENTIST_WORKSPACE_DIR env match the new workspace.
_PROCESS_WORKSPACE: Path | None = None

# Whether async sub-agents are usable in this CLI process. Only True after
# ``ensure_langgraph_dev`` confirms the subprocess is healthy (or already
# running). Stays False on startup failure so ``_maybe_swap_async_subagents``
# can fall back to in-process sync delegation instead of routing tool calls
# at a dead URL.
_ASYNC_SUBAGENTS_AVAILABLE: bool = False


def is_async_subagents_available() -> bool:
    """Return True if the langgraph dev subprocess is up and reachable.

    Used by ``_maybe_swap_async_subagents`` to decide whether to swap dict
    sub-agents to ``AsyncSubAgent`` references. False means a graceful
    fallback to synchronous in-process delegation.
    """
    return _ASYNC_SUBAGENTS_AVAILABLE


# =============================================================================
# Availability & health
# =============================================================================


def _langgraph_exe() -> str | None:
    """Return the path to the langgraph CLI binary, or None if not found."""
    found = shutil.which("langgraph")
    if found:
        return found
    import sys as _sys

    candidate = os.path.join(os.path.dirname(_sys.executable), "langgraph")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def is_langgraph_dev_available() -> bool:
    """Check whether the ``langgraph`` CLI binary is available."""
    return _langgraph_exe() is not None


def is_langgraph_dev_running(
    base_url: str | None = None,
    *,
    port: int = _DEFAULT_PORT,
) -> bool:
    """Check whether a langgraph dev API is already serving at ``base_url``.

    ``base_url`` overrides ``port`` when given.
    """
    url = base_url or _base_url(port)
    try:
        return httpx.get(f"{url}/ok", timeout=1.0).status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def _is_port_occupied(port: int) -> bool:
    """Return True if anything is listening on ``port`` (TCP, IPv4)."""
    import socket as _socket

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        # connect_ex returns 0 on success (something accepted), nonzero otherwise
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _wait_for_port_release(port: int, timeout: float = 10.0) -> bool:
    """Poll until ``port`` is released or ``timeout`` elapses.

    Used after ``stop_langgraph_dev`` / ``_kill_owned_stale_process`` to
    bridge the kernel's TIME_WAIT delay before we try to bind again. Returns
    True if the port is free, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while _is_port_occupied(port) and time.monotonic() < deadline:
        time.sleep(0.5)
    return not _is_port_occupied(port)


def _can_bind_port(port: int) -> bool:
    """Return True if a fresh ``bind()`` to ``port`` succeeds right now.

    More reliable than ``_is_port_occupied`` when the previous listener has
    just exited: ``connect_ex`` can already report "free" while ``bind()``
    still fails because the kernel hasn't fully released the socket
    (TIME_WAIT for accepted connections, SO_REUSEADDR rules, etc.). This
    actually attempts the bind that langgraph dev would attempt, then
    closes immediately.
    """
    import socket as _socket

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _wait_for_port_bindable(port: int, timeout: float = 60.0) -> bool:
    """Poll until a real ``bind()`` to ``port`` can succeed, or timeout.

    Use this immediately before ``subprocess.Popen("langgraph dev")`` —
    matches the strictness of the bind langgraph dev itself will perform,
    so we don't pass the lighter ``_is_port_occupied`` gate only to fail
    on the actual bind a few seconds later.

    Default 60s timeout matches macOS's TCP TIME_WAIT duration — a port
    held by an exited listener is genuinely unbindable for up to that long
    on a tight CLI exit + restart cycle. Shorter timeouts give up too early.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _can_bind_port(port):
            return True
        time.sleep(0.5)
    return False


def _list_pids_on_port(port: int) -> list[int]:
    """Return list of PIDs bound to ``port``, or empty list on lookup failure.

    Read-only; never sends signals. Use this to *inspect* port state before
    deciding what (if anything) to clean up.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _kill_owned_stale_process(port: int) -> bool:
    """Kill ONLY a previously-owned langgraph dev process bound to ``port``.

    "Owned" means the PID written to ``_PID_FILE`` by an earlier
    ``start_langgraph_dev`` invocation in this user account. Returns True if
    a stale-but-owned process was cleaned up; returns False (without sending
    any signals) if the port is occupied by an unowned process — caller
    should treat that as a hard conflict and refuse to start.

    Why this matters: lsof may report any process bound to the port, including
    user-run dev servers that legitimately took 6174. SIGKILL'ing those is a
    data-loss event. Ownership verification keeps the cleanup safe.
    """
    if not _PID_FILE.exists():
        return False
    try:
        owned_pid = int(_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return False

    occupiers = _list_pids_on_port(port)
    if owned_pid not in occupiers:
        return False  # Port is held by a different process now.

    try:
        os.kill(owned_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        _PID_FILE.unlink()
    except OSError:
        pass
    return True


def _packaged_langgraph_config() -> Path:
    """Return path to the package-shipped ``langgraph.json``.

    Lives at ``EvoScientist/langgraph_dev/langgraph.json`` and is included
    in the wheel via ``pyproject.toml`` ``package-data`` so it's available
    regardless of how EvoScientist was installed (pip / editable / source).
    """
    import EvoScientist.langgraph_dev as _pkg

    return Path(_pkg.__file__).resolve().parent / "langgraph.json"


# =============================================================================
# Process management
# =============================================================================


def start_langgraph_dev(
    workspace_dir: Path | None = None,
    *,
    port: int = _DEFAULT_PORT,
    file_persistence: bool = True,
) -> subprocess.Popen:
    """Start langgraph dev as a background subprocess.

    Args:
        workspace_dir: Working directory for the subprocess (subprocess ``cwd``).
            Determines where deployed agents' filesystem operations land
            (``CustomSandboxBackend`` derives its workspace root from cwd via
            ``paths.WORKSPACE_ROOT``). Defaults to ``Path.cwd()``.
        port: TCP port to bind. Defaults to 6174 (Kaprekar's constant).
        file_persistence: When True (default), langgraph dev writes its full
            ``.langgraph_api/`` cache so async-task / Store / scheduler state
            survives subprocess restarts. Set False to suppress periodic
            flushes (workspace stays cleaner; state is in-memory only).

    Returns:
        The Popen handle for the langgraph dev process.

    Raises:
        FileNotFoundError: If the langgraph CLI or packaged ``langgraph.json``
            is missing.
        RuntimeError: If langgraph dev exits early or never becomes healthy.
    """
    global _PROCESS

    exe = _langgraph_exe()
    if exe is None:
        raise FileNotFoundError(
            "langgraph CLI not found. Reinstall EvoScientist (langgraph-cli is "
            "a hard dependency): pip install -e '.[dev]'"
        )

    config_file = _packaged_langgraph_config()
    if not config_file.exists():
        raise FileNotFoundError(
            f"Packaged langgraph.json not found at {config_file}. "
            "This indicates a broken EvoScientist installation — reinstall."
        )

    workspace_dir = workspace_dir or Path.cwd()

    # Defensive: handle a port that's occupied but not serving /ok.
    # Two cases:
    #   (a) Our own previous langgraph dev (PID matches _PID_FILE) — kill it.
    #   (b) Someone else's process — refuse with a clear error rather than
    #       SIGKILL'ing an unowned PID (could be the user's dev server).
    if not is_langgraph_dev_running(port=port) and _is_port_occupied(port):
        if _kill_owned_stale_process(port):
            logger.warning(
                "Cleaned up stale langgraph dev (pid from %s) on port %d",
                _PID_FILE,
                port,
            )
            # After SIGKILL the kernel may keep the port in TIME_WAIT for
            # several seconds before fully releasing it. Poll until the port
            # is genuinely free so the upcoming bind() doesn't race a
            # half-released socket and crash with "Port already in use".
            _wait_for_port_release(port)
        else:
            raise RuntimeError(
                f"Port {port} is occupied by a non-langgraph process. "
                f"Free the port (e.g., `lsof -ti:{port}` to find the owner) "
                f"or change ports with: "
                f"`EvoSci config set langgraph_dev_port <other-port>`"
            )

    # Final defense: poll until a real ``bind()`` to ``port`` succeeds before
    # spawning langgraph dev. ``_is_port_occupied`` (connect-based) can report
    # the port as "free" while langgraph dev's stricter bind still fails —
    # that mismatch is what makes back-to-back CLI exit + restart show
    # "Port already in use" even though our pre-checks passed. By probing
    # the same operation langgraph dev will do, we either wait it out or
    # fail clearly with an actionable message. 60s covers macOS TIME_WAIT.
    if not _wait_for_port_bindable(port):
        raise RuntimeError(
            f"Port {port} cannot be bound after waiting 60s (kernel TIME_WAIT "
            f"or another process holds it). Free the port with `lsof -ti:{port}`, "
            f"or change ports with: `EvoSci config set langgraph_dev_port <other-port>`"
        )

    _PID_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = open(_LOG_FILE, "ab")  # file handle handed to subprocess

    # Propagate workspace to the subprocess so deployed sub-agents resolve
    # paths.WORKSPACE_ROOT to the same dir as the CLI's main agent. cwd alone
    # is fragile (relative paths in MCP configs etc.); env var is explicit.
    #
    # Note: ``EVOSCIENTIST_WORKSPACE_DIR`` serves a dual role in this codebase.
    # config/settings.py:_ENV_MAPPINGS reads it as a user-facing override of
    # ``default_workdir`` (parent process). Here we WRITE it on the subprocess
    # env to propagate the resolved workspace into langgraph dev. Both
    # purposes mean "this is the user's workspace", so they don't conflict;
    # the explicit write below always wins for the subprocess regardless of
    # what the parent had inherited from its own environment.
    sub_env = os.environ.copy()
    sub_env["EVOSCIENTIST_WORKSPACE_DIR"] = str(workspace_dir)

    # By default, let langgraph dev write its full ``.langgraph_api/`` cache
    # so future use cases — cross-session async tasks, Store API persistence,
    # cron job state across CLI restarts — work without further changes. Users
    # who want a clean workspace can opt out via:
    #   EvoSci config set langgraph_dev_file_persistence false
    if not file_persistence:
        sub_env["LANGGRAPH_DISABLE_FILE_PERSISTENCE"] = "true"

    # Skip MCP loading inside the langgraph dev subprocess. The CLI's main
    # agent already loaded MCP servers in the foreground process; without
    # this guard, ``main_graph.py`` would import ``EvoScientist_agent`` and
    # trigger ``_get_default_agent`` → ``load_mcp_and_build_kwargs`` →
    # spawning a SECOND copy of every MCP server in the subprocess.
    # The deployed main agent is currently only reachable via HTTP (for
    # future Web UI / SDK clients), and none of those are in use, so the
    # duplicate MCP pool is pure waste. Async sub-agents don't load MCP at
    # all (their factory bypasses ``load_mcp_and_build_kwargs``), so they
    # are unaffected.
    sub_env["EVOSCIENTIST_DEPLOYED_NO_MCP"] = "true"

    proc = subprocess.Popen(
        [
            exe,
            "dev",
            "--config",
            str(config_file),
            "--port",
            str(port),
            "--n-jobs-per-worker",
            "10",
            "--no-browser",
        ],
        cwd=str(workspace_dir),
        stdout=log_handle,
        stderr=log_handle,
        env=sub_env,
        start_new_session=True,
    )
    _PID_FILE.write_text(str(proc.pid))
    global _PROCESS_WORKSPACE
    _PROCESS = proc
    _PROCESS_WORKSPACE = workspace_dir

    # langgraph dev cold-starts in ~10-15s normally; first-time npx-based MCP
    # servers can push this to 30-60s while npm fetches packages, so the budget
    # is generous. Subsequent runs are much faster thanks to npm cache.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = _LOG_FILE.read_text()[-2000:]
            except Exception:
                pass
            raise RuntimeError(
                f"langgraph dev exited immediately with code {proc.returncode}.\n"
                f"Log tail:\n{tail}"
            )
        if is_langgraph_dev_running(port=port):
            logger.info(
                "langgraph dev started on %s (pid=%d)", _base_url(port), proc.pid
            )
            return proc
        time.sleep(0.5)

    stop_langgraph_dev(proc)
    raise RuntimeError(
        f"langgraph dev did not become healthy within 60 seconds. Check {_LOG_FILE}"
    )


def stop_langgraph_dev(proc: subprocess.Popen | None = None) -> None:
    """Gracefully stop a langgraph dev process.

    Sends SIGTERM to the process group (langgraph dev spawns worker children),
    falling back to SIGKILL after 5 seconds. Safe to call with ``None``.

    Acquires ``_LOCK`` (reentrant) before mutating ``_PROCESS`` /
    ``_PROCESS_WORKSPACE`` so concurrent ``ensure_langgraph_dev`` callers
    (which also hold ``_LOCK``) don't observe partially-cleared state.
    """
    global _PROCESS, _PROCESS_WORKSPACE
    with _LOCK:
        proc = proc if proc is not None else _PROCESS
        if proc is None:
            return

        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, OSError):
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if proc is _PROCESS:
            _PROCESS = None
            _PROCESS_WORKSPACE = None
    if _PID_FILE.exists():
        try:
            _PID_FILE.unlink()
        except OSError:
            pass

    # Note: ``.langgraph_api/`` is intentionally NOT removed — it holds
    # langgraph dev's persisted async-task / scheduler / Store state that
    # may be useful across CLI restarts. Users who want a clean workspace
    # can ``rm -rf .langgraph_api/`` manually or set
    # ``langgraph_dev_file_persistence: false`` in config to suppress writes.


# =============================================================================
# High-level orchestration
# =============================================================================


def ensure_langgraph_dev(
    config: EvoScientistConfig,
    workspace_dir: Path | str | None = None,
) -> subprocess.Popen | None:
    """Conditionally start langgraph dev based on ``config.enable_async_subagents``.

    Behavior:
    - flag false: no-op, returns None
    - flag true + already running on the configured port: reuse, returns None
      (we don't own it; warns if the workspace can't be verified)
    - flag true + not running: start subprocess, register atexit cleanup, return Popen

    Args:
        config: Active EvoScientistConfig.
        workspace_dir: Workspace to inherit on the subprocess. Set to the CLI's
            resolved workspace so deployed async sub-agents see the same files
            as the main in-process agent. If None, the subprocess uses its
            own ``Path.cwd()`` (the CLI's launch directory).

    Errors during startup are logged but don't abort the CLI — the user can
    still chat with sync sub-agents; only async sub-agent calls will fail.
    """
    global _ASYNC_SUBAGENTS_AVAILABLE
    if not getattr(config, "enable_async_subagents", False):
        return None

    # Serialize concurrent callers (e.g., rapid /resume in succession, channel
    # threads, future parallel CLI sessions) so the check / start / stop flow
    # and the module-level state mutations don't interleave.
    with _LOCK:
        return _ensure_langgraph_dev_locked(config, workspace_dir)


def _ensure_langgraph_dev_locked(
    config: EvoScientistConfig,
    workspace_dir: Path | str | None,
) -> subprocess.Popen | None:
    """Locked critical section of ``ensure_langgraph_dev`` — must hold ``_LOCK``."""
    global _ASYNC_SUBAGENTS_AVAILABLE
    port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    file_persistence = bool(getattr(config, "langgraph_dev_file_persistence", True))

    ws_path = Path(workspace_dir) if workspace_dir is not None else None

    # If a subprocess we own is running with a *different* workspace than what
    # was just requested (typical trigger: user just /resumed a thread from a
    # different workspace), the deployed sub-agents' cwd / EVOSCIENTIST_WORKSPACE_DIR
    # are stale. Stop it so the start-fresh path below relaunches with the right
    # workspace. We only act when WE own the process — never kill an externally-
    # managed langgraph dev.
    if (
        ws_path is not None
        and _PROCESS is not None
        and _PROCESS.poll() is None
        and _PROCESS_WORKSPACE is not None
        and _PROCESS_WORKSPACE.resolve() != ws_path.resolve()
    ):
        logger.info(
            "Workspace changed (%s -> %s); restarting langgraph dev so deployed "
            "sub-agents pick up the new workspace.",
            _PROCESS_WORKSPACE,
            ws_path,
        )
        stop_langgraph_dev()
        # Crucial: stop_langgraph_dev unlinks the PID file. If we then fell
        # through with the port still in TIME_WAIT, the next defensive
        # ``_kill_owned_stale_process`` call inside start_langgraph_dev would
        # see no PID file, treat the lingering socket as a foreign process,
        # and abort with a hard "non-langgraph process" error — turning a
        # clean owned restart into a permanent async-disable. Wait inline for
        # the kernel to release the port before continuing.
        _wait_for_port_release(port)
        _ASYNC_SUBAGENTS_AVAILABLE = False  # cleared until restart succeeds

    if is_langgraph_dev_running(port=port):
        # If WE own the running process, workspace was already verified above
        # via _PROCESS_WORKSPACE comparison. If we DON'T own it (some other
        # langgraph dev started by the user / another CLI), we have no way to
        # confirm its workspace matches what was just requested — async
        # sub-agents could end up operating on a different project's files.
        # Warn loudly so the user notices.
        if _PROCESS is None and ws_path is not None:
            logger.warning(
                "Reusing externally-managed langgraph dev on %s — cannot verify "
                "its workspace matches the requested %s. Async sub-agents may "
                "operate on a different workspace's files.",
                _base_url(port),
                ws_path,
            )
        else:
            logger.info("langgraph dev already running on %s, reusing", _base_url(port))
        _ASYNC_SUBAGENTS_AVAILABLE = True
        return None

    try:
        proc = start_langgraph_dev(
            workspace_dir=ws_path,
            port=port,
            file_persistence=file_persistence,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        # Startup failed — keep async subagents disabled so the main agent
        # falls back to in-process sync delegation rather than routing tool
        # calls at a dead URL.
        _ASYNC_SUBAGENTS_AVAILABLE = False
        logger.warning(
            "Failed to start langgraph dev — async sub-agents disabled, "
            "falling back to in-process sync delegation. %s",
            exc,
        )
        return None

    _ASYNC_SUBAGENTS_AVAILABLE = True
    atexit.register(stop_langgraph_dev, proc)
    return proc
