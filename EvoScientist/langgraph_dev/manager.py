"""langgraph dev lifecycle management for background agent support.

Provides functions to start/stop/health-check a ``langgraph dev`` subprocess
that hosts the EvoScientist main agent, async sub-agents (e.g.
``writing-agent``), and EvoMemory background workers. The CLI calls
``ensure_langgraph_dev(config, ...)`` at startup so users can run
``EvoSci -p "..."`` without manually managing the langgraph dev server.

Mirrors the lifecycle pattern used by ``ccproxy_manager.py``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from EvoScientist.config import (
    EvoScientistConfig,
    MemoryControls,
    MemoryObservationTarget,
)

logger = logging.getLogger(__name__)


def needs_langgraph_dev(config: EvoScientistConfig) -> bool:
    """Return whether this config needs the background langgraph dev server."""
    if config.enable_async_subagents:
        return True
    memory_controls = MemoryControls.from_config(config)
    return memory_controls.worker_needed(
        MemoryObservationTarget.TURN_WORKER
    ) or memory_controls.worker_needed(MemoryObservationTarget.SUBAGENT_WORKER)


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

# Workspace fingerprint sidecar — JSON recording the workspace + pid of the
# running langgraph dev. Cross-process callers (e.g. TUI starting up while
# ``EvoSci deploy`` is already running) read this on the reuse path to refuse
# silently operating on a different workspace's files. Missing/corrupt sidecar
# degrades gracefully to a log warning for backward compatibility with
# langgraph devs started before this protocol existed.
_WORKSPACE_SIDECAR = _PID_DIR / "langgraph_dev.workspace.json"


class WorkspaceMismatchError(RuntimeError):
    """Raised when a caller would reuse a langgraph dev whose recorded
    workspace differs from the workspace the caller requested.

    Surfaced by ``ensure_langgraph_dev`` on the cross-process reuse path so
    callers (CLI / serve) can print a clear refuse-with-hint message instead
    of silently routing async sub-agent calls to a process pinned to a
    different workspace.
    """


def _write_workspace_sidecar(workspace_dir: Path, pid: int) -> None:
    """Record the workspace + pid of the langgraph dev we just started.

    Atomic write via temp-file + ``os.replace``: without this, a concurrent
    reader could observe a partially-written file, fail JSON parse, and
    silently downgrade to the "no sidecar" fallback path — which skips the
    workspace mismatch check entirely. ``os.replace`` is atomic on POSIX
    and on Windows; the temp file lives in the same directory so the rename
    stays within one filesystem.

    Best-effort: failures are logged and swallowed. A missing sidecar
    degrades gracefully to the pre-feature behavior (log-warning only) in
    ``ensure_langgraph_dev``.
    """
    try:
        _PID_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _WORKSPACE_SIDECAR.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"workspace": str(workspace_dir), "pid": pid}))
        os.replace(tmp, _WORKSPACE_SIDECAR)
    except OSError as exc:
        logger.warning(
            "Failed to write workspace sidecar %s: %s", _WORKSPACE_SIDECAR, exc
        )


def _read_workspace_sidecar() -> dict | None:
    """Read the workspace sidecar. Returns None if missing, corrupt, or
    structurally wrong (must be a dict whose ``workspace`` value is a
    non-empty string).

    Schema validation matters because the reuse branch in
    ``_ensure_langgraph_dev_locked`` runs ``Path(sidecar["workspace"]).resolve()``
    directly — without the value-type check, a payload like
    ``{"workspace": null}`` or ``{"workspace": []}`` would parse fine, pass
    a naive ``"workspace" in data`` check, then raise ``TypeError`` inside
    ``Path(...)`` and surface as an unhandled exception instead of the
    documented log-warning fallback.
    """
    if not _WORKSPACE_SIDECAR.exists():
        return None
    try:
        data = json.loads(_WORKSPACE_SIDECAR.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    workspace = data.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return None
    return data


def _unlink_workspace_sidecar() -> None:
    """Best-effort sidecar removal — called alongside every ``_PID_FILE.unlink()``
    so the workspace fingerprint never outlives the PID file it pairs with."""
    try:
        _WORKSPACE_SIDECAR.unlink()
    except OSError:
        pass


# Cross-process file lock for ``ensure_langgraph_dev``. Without this, two
# concurrent CLI shells racing on the cold-start window can SIGKILL each
# other's still-booting subprocesses (Shell B sees Shell A's port-bound but
# not-yet-/ok subprocess as a "stale process to clean up"). With the lock,
# Shell B blocks until Shell A's health-check finishes, then sees the
# healthy server and reuses it. ``threading.RLock`` is process-local and
# can't coordinate across CLI invocations.
_FILE_LOCK_PATH = _PID_DIR / "langgraph_dev.lock"
_FILE_LOCK_TIMEOUT = 120.0  # 60s cold-start health-check + buffer

# Module-level handle to the langgraph dev subprocess we started, if any.
# Stays None when we reused an existing process (managed by the user).
_PROCESS: subprocess.Popen | None = None

# Workspace directory the running subprocess was launched with. Used by
# ``ensure_langgraph_dev`` to detect a workspace switch (e.g., on /resume of
# a thread from a different workspace) and trigger a restart so the deployed
# sub-agents' cwd / EVOSCIENTIST_WORKSPACE_DIR env match the new workspace.
_PROCESS_WORKSPACE: Path | None = None

# Whether async sub-agents are usable in this process.
#
# - CLI / serve parent process: starts False; flipped True after
#   ``ensure_langgraph_dev`` confirms the subprocess is healthy. Stays False
#   on startup failure so ``_maybe_swap_async_subagents`` can fall back to
#   in-process sync delegation instead of routing tool calls at a dead URL.
# - langgraph dev subprocess spawned by ``EvoSci deploy``: starts True via
#   ``EVOSCIENTIST_DEPLOY_MODE=full`` env var. The deployed main agent IS the
#   langgraph dev server, so http://localhost:{port} is always reachable for
#   self-loop async sub-agent dispatch.
# - langgraph dev subprocess spawned by ``EvoSci`` / ``EvoSci serve``: env
#   var is ``stripped``, stays False — the deployed main agent in that
#   subprocess is dead code (only sub-agent graphs are invoked), so async
#   swap is unnecessary.
_ASYNC_SUBAGENTS_AVAILABLE: bool = (
    os.environ.get("EVOSCIENTIST_DEPLOY_MODE", "").lower() == "full"
)


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
    except (httpx.TransportError, OSError):
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

    Cross-platform via ``psutil.net_connections`` — works on POSIX and Windows
    without depending on ``lsof`` / ``netstat`` shell tools.
    """
    try:
        return list(
            {
                conn.pid
                for conn in psutil.net_connections(kind="inet")
                if conn.laddr and conn.laddr.port == port and conn.pid is not None
            }
        )
    except (psutil.AccessDenied, psutil.Error):
        return []


def _kill_owned_stale_process(port: int) -> bool:
    """Kill ONLY a previously-owned langgraph dev process bound to ``port``.

    "Owned" means the PID written to ``_PID_FILE`` by an earlier
    ``start_langgraph_dev`` invocation in this user account, AND the live
    process at that PID still has ``langgraph`` in its command line (defense
    against PID recycling). Returns True if a stale-but-owned process was
    cleaned up; returns False (without sending any signals) if the port is
    occupied by an unowned process or the PID has been recycled — caller
    should treat that as a hard conflict and refuse to start.

    Why this matters:
      1. ``net_connections`` may report any process bound to the port,
         including user-run dev servers that legitimately took 6174.
         SIGKILL'ing those is a data-loss event.
      2. Even with PID-file ownership, the OS may have recycled the PID
         to an unrelated process between sessions (e.g., after a SIGKILL'd
         CLI left the PID file behind). The cmdline check rules that out.
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

    # Defense-in-depth: PID could have been recycled to an unrelated process.
    # Verify the live process at that PID still looks like langgraph dev
    # before sending any signals.
    try:
        proc = psutil.Process(owned_pid)
        cmdline = proc.cmdline()
    except psutil.NoSuchProcess:
        # PID file points at a dead process — clean up the file but don't
        # try to kill anything.
        try:
            _PID_FILE.unlink()
        except OSError:
            pass
        _unlink_workspace_sidecar()
        return False
    except psutil.AccessDenied:
        return False

    # Loose substring match by design: PID-file ownership is the primary
    # guard; this check only hardens against PID recycling between sessions.
    # A foreign process happening to have "langgraph" in its argv (e.g., a
    # text editor with langgraph_dev.py open) would slip through, but the
    # ownership check above already excluded externally-owned PIDs, so the
    # window is the narrow case where our exact PID was reused. Keeping the
    # match loose avoids version skew with langgraph CLI invocation styles.
    if not any("langgraph" in arg for arg in cmdline):
        # PID was recycled by an unrelated process. Refuse to kill it, but
        # still clean up the PID file — our original langgraph dev with that
        # PID is definitely gone (PIDs are only recycled after the original
        # process exits), so the file's claim is stale. Mirrors the cleanup
        # in the NoSuchProcess branch above.
        logger.warning(
            "PID file %s claims pid %d for langgraph dev, but that pid now "
            "points at a different process (cmdline=%s). Refusing to kill, "
            "removing stale PID file.",
            _PID_FILE,
            owned_pid,
            cmdline,
        )
        try:
            _PID_FILE.unlink()
        except OSError:
            pass
        _unlink_workspace_sidecar()
        return False

    try:
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        _PID_FILE.unlink()
    except OSError:
        pass
    _unlink_workspace_sidecar()
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
    jobs_per_worker: int = 10,
    deploy_mode: bool = False,
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
    # Three cases:
    #   (a) Our own previous langgraph dev (PID matches _PID_FILE) — kill it.
    #   (b) Our own previous langgraph dev exited but the kernel still holds
    #       the socket in TIME_WAIT — no live PID for lsof to match, and the
    #       PID file may already be gone (stop_langgraph_dev unlinks it). The
    #       bind poll below correctly waits this out.
    #   (c) Foreign process legitimately holds the port — we must NOT kill it.
    #       The bind poll will keep failing and raise an actionable error.
    # We don't try to disambiguate (b) vs (c) here: ``_kill_owned_stale_process``
    # only verifies PID-file ownership, so absence of a match conflates "stale
    # TIME_WAIT" with "foreign process". Falling through to the bind poll
    # disambiguates by behavior — TIME_WAIT clears, foreign listeners don't.
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
            # No owned stale PID — could be foreign or kernel-only TIME_WAIT
            # from a previous subprocess. Defer to the bind poll below.
            logger.info(
                "Port %d occupied with no owned stale PID — waiting for "
                "kernel TIME_WAIT release (or bind-poll timeout if a "
                "foreign process holds it).",
                port,
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
    # Open the log file once and hand it to subprocess.Popen as stdout/stderr.
    # Popen duplicates the fd into the child via fork+exec, so closing our
    # parent-side handle in the finally below releases this process's fd
    # without affecting the child. Without the close, every restart leaks
    # one fd — a problem on heavy ``/resume`` cycling that could eventually
    # exhaust the process's open-file limit.
    log_handle = open(_LOG_FILE, "ab")  # closed in finally below

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

    # Subprocess mode flag — single env var with enum values:
    #
    #   - ``EVOSCIENTIST_DEPLOY_MODE=full`` (deploy_mode=True): set by
    #     ``EvoSci deploy``. Subprocess is the primary programmatic entry
    #     point; main agent loads MCP and ``_ASYNC_SUBAGENTS_AVAILABLE``
    #     flips to True at module load, enabling self-loop async dispatch.
    #
    #   - ``EVOSCIENTIST_DEPLOY_MODE=stripped`` (deploy_mode=False): set by
    #     ``EvoSci`` / ``EvoSci serve``. The CLI's main agent already loaded
    #     MCP in the foreground process; the subprocess skips MCP to avoid
    #     spawning a SECOND copy of every MCP server. The deployed main
    #     agent in this mode is dead code — only sub-agent graphs are
    #     invoked over HTTP — so the duplicate MCP pool would be pure waste.
    #
    #   - (unset): parent process or plain ``import EvoScientist``. Loads
    #     MCP normally; async sub-agents stay disabled (no langgraph dev
    #     server to self-loop into).
    #
    # Strip any inherited value first so a stray export in the user's shell
    # cannot override the mode resolved by this caller.
    sub_env.pop("EVOSCIENTIST_DEPLOY_MODE", None)
    sub_env["EVOSCIENTIST_DEPLOY_MODE"] = "full" if deploy_mode else "stripped"

    try:
        proc = subprocess.Popen(
            [
                exe,
                "dev",
                "--config",
                str(config_file),
                "--port",
                str(port),
                "--n-jobs-per-worker",
                str(jobs_per_worker),
                "--no-browser",
            ],
            cwd=str(workspace_dir),
            stdout=log_handle,
            stderr=log_handle,
            env=sub_env,
            start_new_session=True,
        )
    finally:
        # The child has its own copy of the fd; closing ours prevents an
        # accumulating leak across restarts. Run even if Popen raises.
        try:
            log_handle.close()
        except Exception:
            pass
    _PID_FILE.write_text(str(proc.pid))
    _write_workspace_sidecar(workspace_dir=workspace_dir, pid=proc.pid)
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
            # Subprocess died on its own — clear our module-level bookkeeping
            # (``_PROCESS``, ``_PROCESS_WORKSPACE``, ``_PID_FILE``) before
            # raising. Without this, ``_PROCESS`` would keep pointing at the
            # dead handle and ``_PID_FILE`` at a non-existent PID, leading
            # the next ``ensure_langgraph_dev`` to misjudge state. Pass
            # ``proc`` directly so ``stop_langgraph_dev`` works against the
            # one we just spawned even if the global state was overwritten.
            stop_langgraph_dev(proc)
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
            # No live process to stop, but stale PID/sidecar files may still
            # be on disk from a previous run that died unexpectedly — fall
            # through to the unconditional file cleanup below so subsequent
            # ensure_langgraph_dev calls don't read stale workspace info.
            pass
        else:
            if proc.poll() is None:
                # Cross-platform process-tree shutdown: walk children explicitly
                # because POSIX process groups (``os.killpg``) don't exist on
                # Windows. ``psutil.Process.children(recursive=True)`` works on
                # both — we mirror the previous SIGTERM-then-SIGKILL escalation.
                try:
                    parent = psutil.Process(proc.pid)
                    descendants = parent.children(recursive=True)
                    for child in descendants:
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    parent.terminate()
                    proc.wait(timeout=5)
                except psutil.NoSuchProcess:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        parent = psutil.Process(proc.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
                    # Reap the Popen handle so we don't leave a zombie until
                    # the CLI itself exits. Short timeout because parent.kill()
                    # above already issued SIGKILL to the process tree.
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass

            if proc is _PROCESS:
                _PROCESS = None
                _PROCESS_WORKSPACE = None
    if _PID_FILE.exists():
        try:
            _PID_FILE.unlink()
        except OSError:
            pass
    _unlink_workspace_sidecar()

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
    """Start or reuse langgraph dev for async/background agent work.

    Behavior:
    - already running on the configured port: reuse, returns None
      (we don't own it; warns if the workspace can't be verified)
    - not running: start subprocess, register atexit cleanup, return Popen

    Args:
        config: Active EvoScientistConfig.
        workspace_dir: Workspace to inherit on the subprocess. Set to the CLI's
            resolved workspace so deployed async sub-agents see the same files
            as the main in-process agent. If None, the subprocess uses its
            own ``Path.cwd()`` (the CLI's launch directory).

    Errors during startup are logged but don't abort the CLI — the user can
    still chat with sync sub-agents; only async sub-agent calls and EvoMemory
    background workers will fail.
    """
    global _ASYNC_SUBAGENTS_AVAILABLE

    if not needs_langgraph_dev(config):
        _ASYNC_SUBAGENTS_AVAILABLE = False
        return None

    # Two layers of locking:
    #   1. ``FileLock`` — cross-process coordination. Without it, two CLI
    #      shells (TUI + ``-p`` + ``serve``) racing on the cold-start window
    #      can SIGKILL each other's still-booting subprocesses via
    #      ``_kill_owned_stale_process`` (Shell A's PID is in the file and
    #      bound to the port, but ``/ok`` isn't responding yet, so Shell B
    #      thinks it's stale).
    #   2. ``_LOCK`` (in-process RLock) — serializes intra-process callers
    #      (rapid ``/resume`` in succession, channel threads). Reentrant so
    #      the workspace-restart path can call ``stop_langgraph_dev`` from
    #      inside the critical section.
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(_FILE_LOCK_PATH), timeout=_FILE_LOCK_TIMEOUT):
            with _LOCK:
                return _ensure_langgraph_dev_locked(config, workspace_dir)
    except FileLockTimeout:
        logger.warning(
            "Timed out waiting %.0fs for cross-process langgraph dev lock at %s. "
            "Another CLI shell may be stuck during cold-start. Falling back to "
            "sync sub-agent delegation for this session.",
            _FILE_LOCK_TIMEOUT,
            _FILE_LOCK_PATH,
        )
        _ASYNC_SUBAGENTS_AVAILABLE = False
        return None


def _ensure_langgraph_dev_locked(
    config: EvoScientistConfig,
    workspace_dir: Path | str | None,
) -> subprocess.Popen | None:
    """Locked critical section of ``ensure_langgraph_dev`` — must hold ``_LOCK``."""
    global _ASYNC_SUBAGENTS_AVAILABLE
    port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    file_persistence = bool(getattr(config, "langgraph_dev_file_persistence", True))
    jobs_per_worker = int(getattr(config, "langgraph_dev_jobs_per_worker", 10))

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
        # If WE own the running process AND it's still alive, workspace was
        # already verified above via _PROCESS_WORKSPACE comparison. Otherwise
        # — we never owned it (EvoSci deploy in another terminal, or a
        # langgraph dev the user spawned manually) OR our handle is stale (our
        # subprocess died and a different one rebound the port) — check the
        # workspace sidecar, the only cross-process source of truth for the
        # running instance's workspace. A stale non-None _PROCESS must NOT
        # short-circuit this check, or we'd silently reuse a wrong-workspace
        # server.
        owned_running = _PROCESS is not None and _PROCESS.poll() is None
        if not owned_running and ws_path is not None:
            sidecar = _read_workspace_sidecar()
            if sidecar is not None:
                recorded = Path(sidecar["workspace"]).resolve()
                if recorded != ws_path.resolve():
                    raise WorkspaceMismatchError(
                        f"An EvoSci langgraph dev is already running on "
                        f"{_base_url(port)} for workspace {recorded}, but the "
                        f"current process requested workspace {ws_path}. "
                        f"Stop the other EvoSci session (deploy / TUI / serve) "
                        f"or rerun with --workdir {recorded}."
                    )
                logger.info(
                    "Reusing externally-managed langgraph dev on %s; sidecar "
                    "confirms matching workspace %s.",
                    _base_url(port),
                    recorded,
                )
            else:
                # Pre-feature langgraph dev — no sidecar to verify against.
                # Fall back to the original log-warning behavior so users
                # running an older subprocess don't get bricked.
                logger.warning(
                    "Reusing externally-managed langgraph dev on %s — no "
                    "workspace sidecar, cannot verify it matches the requested "
                    "%s. Async sub-agents may operate on a different workspace's "
                    "files.",
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
            jobs_per_worker=jobs_per_worker,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        # Startup failed — keep async subagents disabled so the main agent
        # falls back to in-process sync delegation rather than routing tool
        # calls at a dead URL. EvoMemory workers will also skip until the
        # server is reachable.
        _ASYNC_SUBAGENTS_AVAILABLE = False
        logger.warning(
            "Failed to start langgraph dev — async sub-agents will fall back "
            "to in-process delegation, and EvoMemory background workers will "
            "not run. %s",
            exc,
        )
        return None

    _ASYNC_SUBAGENTS_AVAILABLE = True
    atexit.register(stop_langgraph_dev, proc)
    return proc
