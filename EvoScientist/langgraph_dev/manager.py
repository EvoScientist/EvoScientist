"""langgraph dev lifecycle management for background agent support.

Provides functions to start/stop/health-check a ``langgraph dev`` subprocess
that hosts the EvoScientist main agent, async sub-agents (e.g.
``writing-agent``), and EvoMemory background workers. The CLI calls
``ensure_langgraph_dev(config, ...)`` at startup so users can run
``EvoSci -p "..."`` without manually managing the langgraph dev server.

Mirrors the lifecycle pattern used by ``ccproxy_manager.py``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import psutil
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from EvoScientist import paths
from EvoScientist.config import (
    EvoScientistConfig,
    MemoryControls,
    MemoryObservationTarget,
)
from EvoScientist.config.settings import (
    _ENV_MAPPINGS,
    ENV_ORIGINS_VAR,
    _same_value,
    config_value_origin,
    env_origin,
    env_origins,
    is_client_only,
)
from EvoScientist.langgraph_dev import boundary, registration, server_home
from EvoScientist.langgraph_dev.registration import (
    RegistrationError,
    ServerUnusableError,
)

if TYPE_CHECKING:
    from EvoScientist.deploy.backend_job import BackendJob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanggraphRuntimePaths:
    """All on-disk paths the langgraph_dev manager writes to.

    Grouped into a single object (rather than a handful of free-floating
    module-level constants) so tests can substitute *one* object to
    redirect every file the manager touches, instead of patching each
    path constant separately. The previous design — five separate
    ``_PID_DIR`` / ``_PID_FILE`` / ``_LOG_FILE`` / ``_WORKSPACE_SIDECAR``
    / ``_FILE_LOCK_PATH`` names — invited inconsistent patches: a test
    might redirect ``_LOG_FILE`` but leave ``_PID_DIR`` pointing at
    ``~/.config/evoscientist``, so the function under test still wrote
    to the user's real home directory. With a single object there's
    one knob to turn; if you replace it, *every* path moves with it.

    Production code constructs ``RUNTIME`` below with the conventional
    ``~/.config/evoscientist/`` layout. Tests call
    :meth:`for_directory` to spin up an isolated instance.
    """

    pid_dir: Path
    pid_file: Path
    log_file: Path
    workspace_sidecar: Path
    lock_file: Path

    @classmethod
    def for_directory(cls, pid_dir: Path) -> LanggraphRuntimePaths:
        """Build a runtime-paths bundle rooted at ``pid_dir``.

        Used by tests (and any future embedded-deployment override) to
        spin up an isolated set of paths without spelling out every
        individual path by hand.
        """
        return cls(
            pid_dir=pid_dir,
            pid_file=pid_dir / "langgraph_dev.pid",
            log_file=pid_dir / "langgraph_dev.log",
            workspace_sidecar=pid_dir / "langgraph_dev.workspace.json",
            lock_file=pid_dir / "langgraph_dev.lock",
        )


# Module-level runtime paths. Defaulted to the conventional
# ``~/.config/evoscientist/`` layout; tests override ``RUNTIME`` with
# :meth:`LanggraphRuntimePaths.for_directory` to point at a temp dir
# without touching the user's real home directory.
DEFAULT_PID_DIR = Path.home() / ".config" / "evoscientist"
RUNTIME: LanggraphRuntimePaths = LanggraphRuntimePaths.for_directory(DEFAULT_PID_DIR)


def needs_langgraph_dev(
    config: EvoScientistConfig,
    *,
    backend: str | None = None,
) -> bool:
    """Return whether this config needs the background langgraph dev server.

    ``backend`` is the calling surface's resolved gateway backend; when
    ``None`` it falls back to the global ``config.gateway_backend`` (the
    pre-per-surface behavior). Passing it lets one surface force the dev server
    on its own backend without the global flag being set.
    """
    effective_backend = (
        backend
        if backend is not None
        else str(getattr(config, "gateway_backend", "local") or "local")
    )
    if effective_backend == "langgraph_server":
        # On the server backend, execution itself is routed to the dev server,
        # so it must exist even when no async sub-agent / scheduler / memory
        # worker independently asks for it.
        return True
    if config.enable_async_subagents:
        return True
    if config.enable_scheduler:
        return True
    if config.memory_skill_synthesis_enabled:
        return True
    memory_controls = MemoryControls.from_config(config)
    return memory_controls.worker_needed(
        MemoryObservationTarget.TURN_WORKER
    ) or memory_controls.worker_needed(MemoryObservationTarget.SUBAGENT_WORKER)


# Reentrant lock guarding ``_PROCESS`` / ``_ASYNC_SUBAGENTS_AVAILABLE``
# mutations and the ``ensure_langgraph_dev`` decision/start/stop flow.
# Reentrant because a start that fails inside ``ensure_langgraph_dev``'s
# critical section calls ``stop_langgraph_dev``, which takes it too.
_LOCK = threading.RLock()


# Set by ``ensure_langgraph_dev`` when it reuses a server whose
# recorded launch-time config fingerprint differs from the current effective
# config. The CLI reads it after startup to surface a "restart to apply"
# hint — the server itself is never restarted automatically.
CONFIG_DRIFT_SINCE_LAUNCH = False

# Set by ``ensure_langgraph_dev`` to the warning text when it reuses a server
# whose agents got another ``python`` or shell than this session resolves (see
# ``agent_shell.shell_drift_message``). The CLI prints it after startup.
AGENT_SHELL_DRIFT: str | None = None

# Set by ``ensure_langgraph_dev``: the session needs the server and does not
# have it (it did not start, or did not register the project). The TUI shows
# :data:`SERVER_UNAVAILABLE_MESSAGE` in the app, where the terminal is hidden.
SERVER_UNAVAILABLE = False
# Set by ``ensure_langgraph_dev``: how the server that registered the
# session's project can be reached from other machines, as it answered;
# ``None`` when no server registered it.
SERVER_EXPOSURE: registration.ServerExposure | None = None
# Shown when CONFIG_DRIFT_SINCE_LAUNCH is set, by the CLI and in the TUI.
CONFIG_DRIFT_MESSAGE = (
    "Configuration or version changed since the background agent server was "
    "launched — it still runs the old settings or version. Apply them with "
    "EvoSci server stop (which ends its running background work), then restart "
    "EvoSci."
)
SERVER_UNAVAILABLE_MESSAGE = (
    "The background agent server is unavailable for this project; async "
    "subagents and EvoMemory workers are off for this session."
)

# Default for sidecar fields that are left out of the record.
_NOT_RECORDED = object()


# Default port (Kaprekar's constant — see config/settings.py for the rationale).
# Overridable per-call via ``start_langgraph_dev(port=...)`` /
# ``ensure_langgraph_dev`` (which reads ``config.langgraph_dev_port``) and the
# corresponding url= field on AsyncSubAgent specs.
_DEFAULT_PORT = 6174

# Default bind interface — loopback, matching ``config.langgraph_dev_host``.
# SECURITY: this is the unauthenticated agent API; launchers print a PUBLIC
# BIND banner while it is exposed.
_DEFAULT_HOST = "127.0.0.1"

# Wildcard bind addresses: the server listens on every interface, but you
# cannot meaningfully *connect* to them (0.0.0.0 is routed to loopback on
# Linux and outright rejected on Windows), so clients target loopback instead.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})


def _probe_host(host: str = _DEFAULT_HOST) -> str:
    """Map a bind address to one a client can actually connect to.

    A wildcard bind includes loopback, so clients use ``127.0.0.1``; a
    specific interface is returned as-is — loopback would not reach it.
    """
    return "127.0.0.1" if host in _WILDCARD_HOSTS else host


def _format_hostport(host: str, port: int) -> str:
    """Render ``host:port`` for a URL, bracketing IPv6 literals per RFC 3986."""
    probe = _probe_host(host)
    return f"[{probe}]:{port}" if ":" in probe else f"{probe}:{port}"


def configured_address(config: object) -> tuple[int, str]:
    """The server's port and bind interface from *config*, with defaults."""
    port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    host = str(getattr(config, "langgraph_dev_host", "") or "").strip()
    return port, host or _DEFAULT_HOST


def _base_url(port: int = _DEFAULT_PORT, host: str = _DEFAULT_HOST) -> str:
    return f"http://{_format_hostport(host, port)}"


# Default rollover threshold for ``RUNTIME.log_file`` — once the log
# exceeds this size, the next ``start_langgraph_dev`` invocation rotates
# it to ``langgraph_dev.log.1`` (overwriting any existing backup) and
# starts fresh. Single-backup policy keeps the disk footprint bounded
# at roughly 2x the threshold even under heavy use (chatty MCP servers,
# repeated failure paths with stack traces). See #209.
_LOG_ROTATION_BYTES = 50 * 1024 * 1024  # 50 MB


def _rotate_log_if_needed(log_path: Path) -> None:
    """Rotate ``log_path`` to ``<log_path>.1`` when it exceeds the
    module's ``_LOG_ROTATION_BYTES`` threshold.

    Single-backup policy: at most one rotated copy is kept on disk. The
    active log is fresh (zero bytes) after rotation, so the next
    ``open(log_path, "ab")`` writes at offset 0.

    Best-effort: failures are logged and swallowed. A failed rotation
    must NOT block ``start_langgraph_dev`` — the worst case is the log
    keeps growing for one more session and the next ``start`` try
    rotates it.
    """
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size <= _LOG_ROTATION_BYTES:
            return
        backup = log_path.with_name(log_path.name + ".1")
        os.replace(log_path, backup)
    except OSError as exc:
        logger.warning(
            "Failed to rotate log %s: %s. Continuing with the existing log.",
            log_path,
            exc,
        )


# Ownership record — JSON recording the pid of the running langgraph dev, the
# project of the client that started it, and what it was launched with.
# Cross-process callers (e.g. a TUI starting while ``EvoSci deploy`` runs)
# read it on the reuse path to report configuration drift.


def _write_workspace_sidecar(
    workspace_dir: Path,
    pid: int,
    config_fingerprint: str | None = None,
    agent_python: str | object | None = _NOT_RECORDED,
    agent_bash: str | object | None = _NOT_RECORDED,
    host: str | None = None,
    lasting: bool | None = None,
    port: int | None = None,
) -> None:
    """Record the pid of the langgraph dev we just started, and the project
    of the client that started it.

    ``config_fingerprint`` (optional) captures the launch-time config subset
    the server consumed; a reuse compares it to detect drift.
    ``agent_python`` (optional) is the ``python`` the server's agents got,
    None when they have none; reuse compares it to warn about a session whose
    PATH resolves another one. ``agent_bash`` (optional) is the Git Bash
    their commands run in, None for cmd.exe or a POSIX shell; compared the
    same way. ``data_dir`` and ``config_dir`` are this process's data and
    config folders, which the server inherits (``paths.dir_env``), so a
    client attaching from another environment reads the server's own.
    ``host`` is the interface the server binds, so a client can tell
    whether its probe reaches this server. ``port`` is the port it listens
    on, for a client started apart from it (a WebUI run by hand) that would
    otherwise only know the configured one. ``lasting`` says
    whether it outlives the process that started it.

    Atomic write via temp-file + ``os.replace``: without this, a concurrent
    reader could observe a partially-written file, fail JSON parse, and
    silently downgrade to the "no sidecar" fallback path — which skips the
    drift checks. ``os.replace`` is atomic on POSIX and on Windows; the temp
    file lives in the same directory so the rename stays within one filesystem.

    Best-effort: failures are logged and swallowed. Without a sidecar,
    ``ensure_langgraph_dev`` reuses the server without the drift checks.
    """
    try:
        RUNTIME.pid_dir.mkdir(parents=True, exist_ok=True)
        tmp = RUNTIME.workspace_sidecar.with_suffix(".json.tmp")
        payload: dict = {
            "workspace": str(workspace_dir),
            "pid": pid,
            "data_dir": str(paths.DATA_DIR),
            "config_dir": str(paths.CONFIG_DIR),
        }
        if host is not None:
            payload["host"] = host
        if port is not None:
            payload["port"] = port
        if lasting is not None:
            payload["lasting"] = lasting
        if config_fingerprint is not None:
            payload["config_fingerprint"] = config_fingerprint
        if agent_python is not _NOT_RECORDED:
            from EvoScientist.setup.research_env import SIDECAR_KEY

            payload[SIDECAR_KEY] = agent_python
        if agent_bash is not _NOT_RECORDED:
            from EvoScientist.agent_shell import SIDECAR_KEY as BASH_KEY

            payload[BASH_KEY] = agent_bash
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, RUNTIME.workspace_sidecar)
    except OSError as exc:
        logger.warning(
            "Failed to write workspace sidecar %s: %s", RUNTIME.workspace_sidecar, exc
        )


def _read_workspace_sidecar() -> dict | None:
    """Read the workspace sidecar. Returns None if missing, corrupt, or
    structurally wrong (must be a dict whose ``workspace`` value is a
    non-empty string).

    Readers show ``workspace`` as a path, so a payload like
    ``{"workspace": null}`` counts as no record.
    """
    if not RUNTIME.workspace_sidecar.exists():
        return None
    try:
        data = json.loads(RUNTIME.workspace_sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    workspace = data.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return None
    return data


def is_lasting(sidecar: dict | None) -> bool:
    """Whether the recorded server outlives the process that started it,
    shared by every session; not one ``EvoSci deploy`` or the desktop app
    owns, nor one an older release recorded."""
    return sidecar is not None and sidecar.get("lasting") is True


def _unlink_workspace_sidecar() -> None:
    """Best-effort sidecar removal — called alongside every ``RUNTIME.pid_file.unlink()``
    so the workspace fingerprint never outlives the PID file it pairs with."""
    try:
        RUNTIME.workspace_sidecar.unlink()
    except OSError:
        pass


# Cross-process file lock for ``ensure_langgraph_dev``. Without this, two
# concurrent CLI shells racing on the cold-start window can SIGKILL each
# other's still-booting subprocesses (Shell B sees Shell A's port-bound but
# not-yet-/ok subprocess as a "stale process to clean up"). With the lock,
# Shell B blocks until Shell A's health-check finishes, then sees the
# healthy server and reuses it. ``threading.RLock`` is process-local and
# can't coordinate across CLI invocations.
# Lock path lives on ``RUNTIME.lock_file``; timeout stays module-level.
_FILE_LOCK_TIMEOUT = 120.0  # 60s cold-start health-check + buffer

# Module-level handle to the langgraph dev subprocess we started, if any.
# Stays None when we reused an existing process (managed by the user).
_PROCESS: subprocess.Popen | None = None

# Byte offset into ``RUNTIME.log_file`` captured the instant before the current
# subprocess was spawned. ``read_tunnel_url`` scans only bytes written after
# this point so a stale ``trycloudflare.com`` URL from a previous (appended,
# not-yet-rotated) session can never be misreported as the live tunnel.
_LOG_OFFSET_AT_START: int = 0

# Cloudflare quick-tunnel public URL, as printed by cloudflared into the
# langgraph dev log. Exclude the API endpoint logged when tunnel creation fails.
_TUNNEL_URL_RE = re.compile(
    r"https://(?!api\.)[A-Za-z0-9.-]+\.trycloudflare\.com", re.IGNORECASE
)

# Whether async sub-agents are usable in this process.
#
# - CLI / serve parent process: starts False; flipped True after
#   ``ensure_langgraph_dev`` confirms the subprocess is healthy. Stays False
#   on startup failure so ``_maybe_swap_async_subagents`` can fall back to
#   in-process sync delegation instead of routing tool calls at a dead URL.
# - langgraph dev subprocess: starts True (``paths.SERVER_PROCESS``, from the
#   variable ``start_langgraph_dev`` sets). The served main agent runs inside
#   the server, so http://localhost:{port} is always reachable for self-loop
#   async sub-agent dispatch.
_ASYNC_SUBAGENTS_AVAILABLE: bool = paths.SERVER_PROCESS


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
    import sys

    executable_dir = os.path.dirname(sys.executable)
    candidate_names = (
        ["langgraph.exe", "langgraph"] if os.name == "nt" else ["langgraph"]
    )
    for candidate_name in candidate_names:
        candidate = os.path.join(executable_dir, candidate_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("langgraph")
    if found:
        return found
    return None


def is_langgraph_dev_available() -> bool:
    """Check whether the ``langgraph`` CLI binary is available."""
    return _langgraph_exe() is not None


def is_langgraph_dev_running(
    base_url: str | None = None,
    *,
    port: int = _DEFAULT_PORT,
    host: str = _DEFAULT_HOST,
) -> bool:
    """Check whether a langgraph dev API is already serving at ``base_url``.

    ``base_url`` overrides ``port``/``host`` when given.
    """
    url = base_url or _base_url(port, host)
    try:
        # trust_env=False: this is a loopback probe of our own server. httpx
        # otherwise routes it through the environment/OS proxy — and on Windows
        # getproxies() reads the system (registry/IE) proxy even with no *_PROXY
        # env vars set, so a corporate proxy silently swallows the 127.0.0.1
        # request and the health check never sees the healthy server.
        return httpx.get(f"{url}/ok", timeout=1.0, trust_env=False).status_code == 200
    except (httpx.TransportError, OSError):
        return False


def _is_port_occupied(port: int, host: str = _DEFAULT_HOST) -> bool:
    """Return True if anything is listening on ``host:port`` (TCP)."""
    import socket as _socket

    probe = _probe_host(host)
    family = _socket.AF_INET6 if ":" in probe else _socket.AF_INET
    s = _socket.socket(family, _socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        # connect_ex returns 0 on success (something accepted), nonzero otherwise
        return s.connect_ex((probe, port)) == 0
    finally:
        s.close()


def _wait_for_port_release(
    port: int, timeout: float = 10.0, host: str = _DEFAULT_HOST
) -> bool:
    """Poll until ``port`` is released or ``timeout`` elapses.

    Used after ``stop_langgraph_dev`` / ``_kill_owned_stale_process`` to
    wait for the old server to exit and close its listener before we try to
    bind again. Returns True if the port is free, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while _is_port_occupied(port, host) and time.monotonic() < deadline:
        time.sleep(0.5)
    return not _is_port_occupied(port, host)


def _can_bind_port(port: int, host: str = _DEFAULT_HOST) -> bool:
    """Return True if a fresh ``bind()`` to ``host:port`` succeeds right now.

    More reliable than ``_is_port_occupied`` when the previous listener has
    just exited: ``connect_ex`` can already report "free" while ``bind()``
    still fails because the kernel hasn't fully released the socket. This
    actually attempts the bind that langgraph dev would attempt, then
    closes immediately.

    On POSIX it sets ``SO_REUSEADDR`` like uvicorn does, so connections the
    old server left in TIME_WAIT (60s on Linux) don't make a port look busy
    that the server could bind. Not on Windows, where the option also allows
    binding over a live listener.

    Binds the *literal* ``host`` — not ``_probe_host(host)`` — because this
    must replicate the server's own bind: a loopback probe can succeed while
    the real wildcard bind still fails on another interface's conflict.
    """
    import socket as _socket

    family = _socket.AF_INET6 if ":" in host else _socket.AF_INET
    s = _socket.socket(family, _socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _wait_for_port_bindable(
    port: int, timeout: float = 60.0, host: str = _DEFAULT_HOST
) -> bool:
    """Poll until a real ``bind()`` to ``port`` can succeed, or timeout.

    Use this immediately before ``subprocess.Popen("langgraph dev")`` —
    matches the strictness of the bind langgraph dev itself will perform,
    so we don't pass the lighter ``_is_port_occupied`` gate only to fail
    on the actual bind a few seconds later.

    The 60s default gives a previous server that is still exiting time to
    release the port. TIME_WAIT connections it leaves behind don't count:
    ``_can_bind_port`` binds with ``SO_REUSEADDR`` on POSIX, as the server
    does.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _can_bind_port(port, host):
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

    "Owned" means the PID written to ``RUNTIME.pid_file`` by an earlier
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
    if not RUNTIME.pid_file.exists():
        return False
    try:
        owned_pid = int(RUNTIME.pid_file.read_text(encoding="utf-8").strip())
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
            RUNTIME.pid_file.unlink()
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
            RUNTIME.pid_file,
            owned_pid,
            cmdline,
        )
        try:
            RUNTIME.pid_file.unlink()
        except OSError:
            pass
        _unlink_workspace_sidecar()
        return False

    try:
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        RUNTIME.pid_file.unlink()
    except OSError:
        pass
    _unlink_workspace_sidecar()
    return True


# Packaged sub-agent specs — consumed at graph build; module constant so
# tests can redirect it.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_SUBAGENTS_DIR = _PACKAGE_DIR / "subagents"

# Fields the server reads from config.yaml itself: no environment variable
# carries them, so a command-line override (--auto-approve, --auto-mode, the
# ask_user switch) never reaches it. The fingerprint takes them from the file
# too, so a server started with such a flag is not drift, while a change in
# the file still is.
_FINGERPRINT_FILE_FIELDS = frozenset({"auto_approve", "auto_mode", "enable_ask_user"})


def _server_config_fingerprint(config: EvoScientistConfig) -> str:
    """Hash of everything the langgraph dev subprocess consumes at launch.

    Deployed graphs read config once at import (``subagents/_factory.py``,
    ``EvoScientist.py``), so a running server keeps serving those values
    until restarted. Iterates the full ``EvoScientistConfig`` field list
    minus the fields that never reach the server
    (:func:`~EvoScientist.config.settings.is_client_only`) — a new config
    field counts toward drift by default — and folds in ``mcp.yaml`` plus the packaged
    ``subagents/*.yaml``, which are consumed at graph build too. The
    EvoScientist version is part of it, and for an editable install each
    source file's path and modification time, so a server left running
    across an upgrade, or across a source checkout's new commit, counts as
    drift as well. Secrets
    only feed a truncated one-way digest; nothing recoverable is stored.
    getattr with defaults: deploy/WebUI (and their tests) routinely hand
    this module duck-typed config objects missing dataclass fields.
    :data:`_FINGERPRINT_FILE_FIELDS`, and the fields a command-line flag of
    this session set that a lasting server does not get, come from
    ``config.yaml``, not from ``config``, so every caller hashes what the
    server reads.
    """
    from EvoScientist.config.settings import load_config

    file_config = load_config()
    parts = []
    for field in dataclass_fields(EvoScientistConfig):
        name = field.name
        if is_client_only(name):
            continue
        from_file = name in _FINGERPRINT_FILE_FIELDS or _withheld_flag(
            _ENV_MAPPINGS.get(name, "")
        )
        source = file_config if from_file else config
        parts.append((name, str(getattr(source, name, None))))
    try:
        parts.append(("version", importlib.metadata.version("EvoScientist")))
    except importlib.metadata.PackageNotFoundError:
        parts.append(("version", None))
    digest = hashlib.sha256(repr(parts).encode("utf-8"))
    try:
        from EvoScientist.config.settings import get_config_dir

        mcp_yaml = get_config_dir() / "mcp.yaml"
        if mcp_yaml.exists():
            digest.update(mcp_yaml.read_bytes())
    except OSError:
        pass
    try:
        for yaml_path in sorted(_SUBAGENTS_DIR.glob("*.yaml")):
            digest.update(yaml_path.name.encode("utf-8"))
            digest.update(yaml_path.read_bytes())
    except OSError:
        pass
    from EvoScientist.ccproxy_manager import _is_editable_install

    # A source checkout changes its code without a version bump (a pull, a
    # branch switch): a few milliseconds to notice it.
    try:
        if _is_editable_install():
            for path in sorted(_PACKAGE_DIR.rglob("*.py")):
                stamp = f"{path.relative_to(_PACKAGE_DIR)}:{path.stat().st_mtime_ns}"
                digest.update(stamp.encode("utf-8"))
    except OSError:
        pass
    return digest.hexdigest()[:16]


class ServerKeptError(RuntimeError):
    """``EvoSci service stop`` left the recorded server running: ``EvoSci
    deploy`` or the desktop app owns it, or another EvoSci process is starting
    or stopping it."""


def owned_by_its_starter() -> bool:
    """Whether the recorded server is a live one that ``EvoSci deploy`` or
    the desktop app owns, not the background service: ``EvoSci service
    stop`` leaves it running. A record of an older release does not say."""
    sidecar = _read_workspace_sidecar()
    return (
        sidecar is not None
        and sidecar.get("lasting") is False
        and _langgraph_cmdline(sidecar.get("pid")) is not None
    )


def stop_recorded_server(*, keep_owned: bool = False) -> int | None:
    """Explicitly stop the langgraph dev recorded in our PID file.

    Backs the user-facing ``EvoSci server stop`` command: a server that
    outlives its CLI needs a first-class way to stop it. Ownership = our
    PID file + a live process whose cmdline still contains ``langgraph``
    (same loose anti-PID-recycling match as ``_kill_owned_stale_process``,
    with PID-file ownership as the primary guard). Holds the cross-process
    file lock so a concurrent start can't have its fresh PID/sidecar records
    wiped by this stop's cleanup. Kills the whole process tree, then removes
    the PID file + sidecar. Returns the stopped pid, or ``None`` when nothing
    was stopped (stale/corrupt files, if any, are still cleaned up).

    With *keep_owned* (``EvoSci service stop``), a live server that ``EvoSci
    deploy`` or the desktop app owns is left running, and so is one whose lock
    another process holds too long: :class:`ServerKeptError`. A record of an
    older release, which does not say, is stopped.
    """
    try:
        with FileLock(str(RUNTIME.lock_file), timeout=_FILE_LOCK_TIMEOUT):
            if keep_owned and owned_by_its_starter():
                raise ServerKeptError(
                    "The running server is not the background service: an "
                    "EvoSci deploy or the desktop app owns it, so it was left "
                    "running (EvoSci server stop stops it anyway)."
                )
            return _stop_recorded_server_locked()
    except FileLockTimeout:
        if keep_owned:
            raise ServerKeptError(
                "Another EvoSci process is starting or stopping the server, so "
                "it was left running; try again in a moment."
            ) from None
        logger.warning(
            "Timed out waiting for the langgraph dev lock — another EvoSci "
            "process is mid lifecycle change; not stopping anything."
        )
        return None


def stop_inflight_owned_server() -> int | None:
    """Stop only the langgraph dev *this* process spawned, never one recorded
    on disk by another process.

    The launcher's mid-start teardown uses this instead of
    :func:`stop_recorded_server`. Before its own child is spawned, the shared
    on-disk PID file still names a *different* server — e.g. another
    session's backend on a different port — so killing the recorded
    server on a close-during-boot would take down an unrelated session. The
    in-memory ``_PROCESS`` is set only by this process's ``start_langgraph_dev``
    (after Popen), so it can only ever name our own in-flight child; a close
    before Popen finds it ``None`` and stops nothing. Returns the stopped pid,
    or ``None`` when we had not spawned anything yet.
    """
    with _LOCK:
        if _PROCESS is not None and _PROCESS.poll() is None:
            pid = _PROCESS.pid
            stop_langgraph_dev()
            return pid
    return None


def _stop_recorded_server_locked() -> int | None:
    with _LOCK:
        if _PROCESS is not None and _PROCESS.poll() is None:
            pid = _PROCESS.pid
            stop_langgraph_dev()
            return pid
    if not RUNTIME.pid_file.exists():
        return None
    try:
        owned_pid = int(RUNTIME.pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        stop_langgraph_dev()  # corrupt PID file — clean it up as promised
        return None
    except OSError:
        return None
    try:
        proc = psutil.Process(owned_pid)
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        stop_langgraph_dev()  # dead/inaccessible — clean the stale files
        return None
    if not any("langgraph" in arg for arg in cmdline):
        stop_langgraph_dev()  # pid recycled by a foreign process — files only
        return None
    try:
        children = proc.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            for child in children:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            proc.kill()
        # The parent exiting promptly doesn't prove its workers did — sweep
        # the pre-kill snapshot for survivors.
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    stop_langgraph_dev()
    return owned_pid


def _pid_serves_port(pid: object, port: int) -> bool:
    """Best-effort check that ``pid`` is a langgraph dev serving ``port``.

    Used to attribute an occupied port to the sidecar's recorded server
    before printing its details — avoids blaming a stale record. Relies on
    ``--port`` always being in ``start_langgraph_dev``'s argv, not on
    port→PID mapping (root-only on macOS via psutil).
    """
    cmdline = _langgraph_cmdline(pid)
    return cmdline is not None and str(port) in cmdline


def _langgraph_cmdline(pid: object) -> list[str] | None:
    """The command line of ``pid`` while it is a live langgraph dev, on any
    port; ``None`` otherwise."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        cmdline = psutil.Process(pid).cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return cmdline if any("langgraph" in arg for arg in cmdline) else None


def recorded_address() -> tuple[int, str] | None:
    """The port and host of the server in the PID file, the one ``EvoSci
    server stop`` ends, while it is a live langgraph dev, read from its
    command line (``langgraph_dev_port`` or ``langgraph_dev_host`` may have
    changed since it started); ``None`` without one."""
    try:
        pid = int(RUNTIME.pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    cmdline = _langgraph_cmdline(pid)
    if cmdline is None:
        return None
    started_with = dict(pairwise(cmdline))
    try:
        port = int(started_with["--port"])
    except (KeyError, ValueError):
        return None
    # Releases before the bind host was configurable passed no --host and
    # bound loopback.
    return port, started_with.get("--host", _DEFAULT_HOST)


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


def _from_dotenv(env_key: str) -> bool:
    """Whether this process's ``.env`` merge set *env_key*; the server gets
    none of those."""
    return (env_origin(env_key) or {}).get("source") == "dotenv"


def _set_by_flag(env_key: str) -> bool:
    """Whether a command-line flag of this session set *env_key*. A lasting
    server serves later sessions and every project, so it does not get it."""
    return (env_origin(env_key) or {}).get("source") == "cli"


# The flags a lasting server still takes from the session that starts it.
# The auth mode picks the credentials its requests use, of which one server
# has one set: the starter's choice stands, and a later session with another
# one sees the drift warning. (--auth-mode for Anthropic writes no variable:
# with OAuth, ccproxy's variables carry the choice to the server.)
_FLAGS_SERVER_TAKES = frozenset({"EVOSCIENTIST_OPENAI_AUTH_MODE"})


def _withheld_flag(env_key: str) -> bool:
    """Whether *env_key* is a flag of this session that a lasting server
    does not get."""
    return _set_by_flag(env_key) and env_key not in _FLAGS_SERVER_TAKES


def _passed_to_server(env_key: str, *, lasting: bool) -> bool:
    """Whether the server gets this process's *env_key*: never a ``.env``
    setting, and a withheld flag only when the server is this process's own."""
    return not _from_dotenv(env_key) and not (lasting and _withheld_flag(env_key))


def _server_env_origins(port: str, host: str, *, lasting: bool) -> dict:
    """This process's record of environment origins, plus the port and host
    :func:`start_langgraph_dev` always sets for the server.

    A variable passed on unchanged keeps what is recorded for it, or no record
    for the user's own export; a variable the server does not get
    (:func:`_passed_to_server`) is not recorded.
    """
    origins = {
        key: origin
        for key, origin in env_origins().items()
        if _passed_to_server(key, lasting=lasting)
    }
    for env_key, field, value in (
        ("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "langgraph_dev_port", port),
        ("EVOSCIENTIST_LANGGRAPH_DEV_HOST", "langgraph_dev_host", host),
    ):
        inherited = os.environ.get(env_key)
        if inherited is not None and _same_value(field, inherited, value):
            continue
        origins[env_key] = {"source": config_value_origin(field, value)}
    return origins


def _refuse_running_server() -> None:
    """Raise ``RuntimeError`` when another server uses this data folder, after
    ``server_home.LOCK_WAIT`` for one stopped just before that is exiting."""
    try:
        running = server_home.server_running(wait=server_home.LOCK_WAIT)
    except OSError as exc:
        raise RuntimeError(
            f"Could not use the server folder {paths.SERVER_HOME}: {exc}"
        ) from exc
    if running:
        raise server_home.ServerRunningError(server_home.running_server_message())


def start_langgraph_dev(
    workspace_dir: Path | None = None,
    *,
    port: int = _DEFAULT_PORT,
    host: str = _DEFAULT_HOST,
    file_persistence: bool = True,
    jobs_per_worker: int = 10,
    tunnel: bool = False,
    config_fingerprint: str | None = None,
    allowed_origins: str = "",
    job: BackendJob | None = None,
    lasting: bool = True,
) -> subprocess.Popen:
    """Start langgraph dev as a background subprocess.

    The server works in the server home (``paths.SERVER_HOME``), which holds
    its runtime store, and serves every registered project.

    Args:
        workspace_dir: The project of the client starting the server, passed
            as ``paths.LAUNCHER_WORKSPACE_ENV``: ``GET /api/workspace``
            reports it once the client registered it. Defaults to
            ``Path.cwd()``.
        port: TCP port to bind. Defaults to 6174 (Kaprekar's constant).
        host: Network interface to bind. Defaults to loopback. SECURITY:
            widening this exposes an unauthenticated API that serves every
            registered project and whose agent can run shell commands — only
            pass ``0.0.0.0`` on trusted networks.
        file_persistence: When True (default), langgraph dev writes its full
            ``.langgraph_api/`` cache in the server home so async-task / Store
            / scheduler state survives subprocess restarts. Set False to
            suppress periodic flushes (state is in-memory only).
        jobs_per_worker: Concurrent runs per worker (``--n-jobs-per-worker``).
        tunnel: When True, pass ``--tunnel`` so langgraph dev exposes the
            server over a public Cloudflare quick-tunnel. The random
            ``*.trycloudflare.com`` URL is written to the log; read it back
            with :func:`read_tunnel_url`. SECURITY: the tunnel has no auth,
            exposes every registered project and the agent can run shell —
            only enable for trusted use.
        job: A Windows job object the server joins before it runs (started
            suspended, then resumed), so it ends with the caller's process.
        lasting: Whether the server outlives its caller, shared by every
            session (recorded for ``EvoSci update``, which starts a lasting
            server again). False for ``EvoSci deploy``, which stops the
            server it started.

    Returns:
        The Popen handle for the langgraph dev process.

    Raises:
        FileNotFoundError: If the langgraph CLI or packaged ``langgraph.json``
            is missing.
        RuntimeError: If another server uses this data folder, or langgraph
            dev exits early or never becomes healthy.
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

    # Install Node for npx MCP servers here, where progress is visible and no
    # health deadline applies; the server itself never downloads it.
    try:
        from ..mcp.client import _ensure_node_for_stdio, load_mcp_config

        _ensure_node_for_stdio(load_mcp_config())
    except Exception:
        logger.warning("Could not check Node.js for MCP servers", exc_info=True)

    # Defensive: handle a port that's occupied but not serving /ok.
    # Three cases:
    #   (a) Our own previous langgraph dev (PID matches RUNTIME.pid_file) — kill it.
    #   (b) Our own previous langgraph dev is still exiting and its listener
    #       is still open, but the PID file may already be gone
    #       (stop_langgraph_dev unlinks it), so there's nothing to match it
    #       against. The bind poll below waits this out.
    #   (c) Foreign process legitimately holds the port — we must NOT kill it.
    #       The bind poll will keep failing and raise an actionable error.
    # We don't try to disambiguate (b) vs (c) here: ``_kill_owned_stale_process``
    # only verifies PID-file ownership, so absence of a match conflates "our
    # server, still exiting" with "foreign process". Falling through to the
    # bind poll disambiguates by behavior: an exiting server releases the
    # port, a foreign listener doesn't.
    if not is_langgraph_dev_running(port=port, host=host) and _is_port_occupied(
        port, host
    ):
        if _kill_owned_stale_process(port):
            logger.warning(
                "Cleaned up stale langgraph dev (pid from %s) on port %d",
                RUNTIME.pid_file,
                port,
            )
            # After SIGKILL the process can take a moment to exit and close
            # its listener. Poll until the port is free so the upcoming bind()
            # doesn't race it and crash with "Port already in use".
            _wait_for_port_release(port, host=host)
        else:
            # No owned stale PID — could be foreign or a previous server
            # that is still exiting. Defer to the bind poll below.
            logger.info(
                "Port %d occupied with no owned stale PID — waiting for it "
                "to be released (or bind-poll timeout if a foreign process "
                "holds it).",
                port,
            )

    # Final defense: poll until a real ``bind()`` to ``port`` succeeds before
    # spawning langgraph dev. ``_is_port_occupied`` (connect-based) can report
    # the port as "free" while langgraph dev's stricter bind still fails —
    # that mismatch is what makes back-to-back CLI exit + restart show
    # "Port already in use" even though our pre-checks passed. By probing
    # the same operation langgraph dev will do, we either wait it out or
    # fail clearly with an actionable message.
    if not _wait_for_port_bindable(port, host=host):
        raise RuntimeError(
            f"{host}:{port} cannot be bound after waiting 60s (another process "
            f"holds it). Free the port with `lsof -ti:{port}`, "
            f"or change ports with: `EvoSci config set langgraph_dev_port <other-port>`"
        )
    # One server per data folder: the server takes the lock itself, so a
    # refusal here says why rather than "exited immediately".
    _refuse_running_server()

    RUNTIME.pid_dir.mkdir(parents=True, exist_ok=True)
    # Rotate the log if it has grown past the threshold so this session's
    # output starts on a fresh file. Failure is non-fatal (see
    # ``_rotate_log_if_needed``). See #209.
    _rotate_log_if_needed(RUNTIME.log_file)
    # Open the log file once and hand it to subprocess.Popen as stdout/stderr.
    # Popen duplicates the fd into the child via fork+exec, so closing our
    # parent-side handle in the finally below releases this process's fd
    # without affecting the child. Without the close, every restart leaks
    # one fd — a problem on heavy ``/resume`` cycling that could eventually
    # exhaust the process's open-file limit.
    log_handle = open(RUNTIME.log_file, "ab")  # closed in finally below
    # Remember where this session's output begins so ``read_tunnel_url`` only
    # scans lines this subprocess writes — never a stale URL left in the
    # appended-to log by a previous tunnel session.
    global _LOG_OFFSET_AT_START
    try:
        _LOG_OFFSET_AT_START = RUNTIME.log_file.stat().st_size
    except OSError:
        _LOG_OFFSET_AT_START = 0

    # The server serves every registered project from the server home, so it
    # gets no workspace of its own; it is told the launcher's project, as an
    # absolute path since it works in another folder. A run names its run
    # folder itself. ``EVOSCIENTIST_WORKSPACE_DIR`` passes as inherited: it is
    # the user's ``default_workdir`` override (Docker sets it), which the
    # settings API reports. A project ``.env`` sets only this command's own
    # settings, each only where the environment had none, so the server gets
    # the environment without them. A lasting server, which serves later
    # sessions too, gets none of this session's flags (``--dangerous``,
    # ``--debug``) but the auth mode: it follows config.yaml and the user's
    # own environment. It gets this process's data and config folders so both use
    # the same files.
    sub_env = paths.dir_env(
        {
            key: value
            for key, value in os.environ.items()
            if _passed_to_server(key, lasting=lasting)
        }
    )
    sub_env.pop("EVOSCIENTIST_RUN_DIR", None)
    sub_env[paths.LAUNCHER_WORKSPACE_ENV] = os.path.abspath(workspace_dir)
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"

    # By default, let langgraph dev write its full ``.langgraph_api/`` cache
    # (in the server home) so future use cases — cross-session async tasks, Store API persistence,
    # cron job state across CLI restarts — work without further changes. Users
    # who want no cache on disk can opt out via:
    #   EvoSci config set langgraph_dev_file_persistence false
    if not file_persistence:
        sub_env["LANGGRAPH_DISABLE_FILE_PERSISTENCE"] = "true"

    # ``EVOSCIENTIST_SERVER_PROCESS`` marks the process as the server:
    # ``_ASYNC_SUBAGENTS_AVAILABLE`` starts True so async sub-agents dispatch
    # back to this server, main-graph middleware events go out over the
    # stream, and MCP setup never downloads Node here. The variable is unset
    # in the CLI process and on a plain ``import EvoScientist``.
    sub_env["EVOSCIENTIST_SERVER_PROCESS"] = "1"

    # Propagate the effective bind port into the subprocess's config resolution
    # via the standard ``EVOSCIENTIST_LANGGRAPH_DEV_PORT`` override (see
    # ``EvoScientist/config/settings.py``). Without this, ``EvoSci deploy
    # --port X`` binds to X but the deployed main agent still reads
    # ``cfg.langgraph_dev_port`` from disk and dispatches self-loop async
    # tasks (start_async_task → http://localhost:{cfg.port}) to whatever
    # the config file says — which desyncs from the bind port whenever
    # ``--port`` differs from the persisted ``langgraph_dev_port``, and
    # every async subagent launch fails with "All connection attempts failed".
    # The server reads no ``.env``, so nothing shadows the caller-resolved port.
    sub_env["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] = str(port)
    # Same reasoning for the bind interface: the deployed agent resolves its
    # self-dispatch URL from ``cfg.langgraph_dev_host``, so a host resolved by
    # this caller (``EvoSci deploy --host X``) must reach the subprocess too,
    # or async sub-agent launches would target whatever the config file says.
    sub_env["EVOSCIENTIST_LANGGRAPH_DEV_HOST"] = host
    # The browser boundary: which pages and host names the server accepts
    # (``boundary.server_env``), always set here so nothing inherited, such as
    # a workspace ``.env``, widens it. CORS_ALLOW_ORIGINS would only matter
    # without our CORS_CONFIG; TUNNEL_HTTP_HOST_HEADER would make cloudflared,
    # which inherits this environment, send tunnel requests with a local Host.
    sub_env.update(boundary.server_env(host, allowed_origins))
    sub_env.pop("CORS_ALLOW_ORIGINS", None)
    sub_env.pop("TUNNEL_HTTP_HOST_HEADER", None)
    # Where the server's config-mapped variables came from, for the settings
    # API's ``sources``: this process's record plus the two values set above.
    sub_env[ENV_ORIGINS_VAR] = json.dumps(
        _server_env_origins(str(port), host, lasting=lasting)
    )

    # POSIX: own session so the child can be group-signalled on cleanup, and
    # outlives the terminal. Windows: own process group, so Ctrl+C in the
    # terminal does not reach the server. A lasting server also gets a console
    # of its own, without a window, so closing the terminal does not end it;
    # its children (MCP servers, Git Bash commands) share that hidden console
    # and open no windows. A server a session owns keeps the terminal's.
    if os.name == "nt":
        # getattr keeps this import-safe off Windows (the flags are Windows-only).
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if lasting:
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _spawn_kwargs = {"creationflags": flags}
    else:
        _spawn_kwargs = {"start_new_session": True}
    if job is not None:
        _spawn_kwargs["creationflags"] = (
            _spawn_kwargs.get("creationflags", 0) | job.creationflags
        )

    try:
        server_folder = server_home.ensure_server_home()
    except OSError as exc:
        log_handle.close()
        raise RuntimeError(
            f"Could not create the server folder {paths.SERVER_HOME}: {exc}"
        ) from exc
    try:
        logger.info("Starting langgraph dev with CLI: %s", exe)
        proc = subprocess.Popen(
            [
                exe,
                "dev",
                "--config",
                str(config_file),
                "--host",
                host,
                "--port",
                str(port),
                "--n-jobs-per-worker",
                str(jobs_per_worker),
                "--no-browser",
                "--no-reload",
                *(["--tunnel"] if tunnel else []),
            ],
            cwd=str(server_folder),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            env=sub_env,
            **_spawn_kwargs,
        )
    finally:
        # The child has its own copy of the fd; closing ours prevents an
        # accumulating leak across restarts. Run even if Popen raises.
        try:
            log_handle.close()
        except Exception:
            pass
    # Recorded before anything else can fail or be interrupted, so every
    # later failure path, and a caller's stop_langgraph_dev(), reaches it.
    _PROCESS = proc
    if job is not None:
        job.started(proc)
    RUNTIME.pid_file.write_text(str(proc.pid), encoding="utf-8")
    from EvoScientist.agent_shell import sidecar_bash
    from EvoScientist.setup.research_env import agent_python

    _write_workspace_sidecar(
        workspace_dir=workspace_dir,
        pid=proc.pid,
        config_fingerprint=config_fingerprint,
        # The server inherits this process's environment and data dir, so it
        # resolves the same python and reads the same Git Bash record.
        agent_python=agent_python(),
        agent_bash=sidecar_bash(),
        host=host,
        lasting=lasting,
        port=port,
    )

    # langgraph dev cold-starts in ~10-15s normally, so the budget is
    # generous. MCP servers load later, when a project's graph is first built.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = RUNTIME.log_file.read_text(encoding="utf-8", errors="replace")[
                    -2000:
                ]
            except Exception:
                pass
            # Subprocess died on its own — clear our module-level bookkeeping
            # (``_PROCESS``, ``RUNTIME.pid_file``) before raising. Without this,
            # ``_PROCESS`` would keep pointing at the dead handle and
            # ``RUNTIME.pid_file`` at a non-existent PID, leading the next
            # ``ensure_langgraph_dev`` to misjudge state. Pass ``proc`` directly
            # so ``stop_langgraph_dev`` works against the one we just spawned even
            # if the global state was overwritten.
            stop_langgraph_dev(proc)
            raise RuntimeError(
                f"langgraph dev exited immediately with code {proc.returncode}.\n"
                f"Log tail:\n{tail}"
            )
        if is_langgraph_dev_running(port=port, host=host):
            logger.info(
                "langgraph dev started on %s (pid=%d)", _base_url(port, host), proc.pid
            )
            return proc
        time.sleep(0.5)

    stop_langgraph_dev(proc)
    raise RuntimeError(
        f"langgraph dev did not become healthy within 60 seconds. Check {RUNTIME.log_file}"
    )


def read_tunnel_url(timeout: float = 35.0, poll_interval: float = 0.5) -> str | None:
    """Poll the langgraph dev log for the Cloudflare quick-tunnel public URL.

    Started with ``tunnel=True``, langgraph dev shells out to cloudflared,
    which prints a random ``https://<words>.trycloudflare.com`` URL once the
    tunnel is established — typically a few seconds after the local server is
    already healthy. We scan only the bytes written since this subprocess
    started (``_LOG_OFFSET_AT_START``) so a stale URL from an earlier session
    in the same appended-to log is never returned.

    Args:
        timeout: Max seconds to wait for the URL to appear. cloudflared may
            also need to download its binary on first use, so the default is
            generous (langgraph_api itself waits up to 30s internally).
        poll_interval: Seconds between log re-reads.

    Returns:
        The public tunnel URL, or ``None`` if it never appeared in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(RUNTIME.log_file, "rb") as fh:
                fh.seek(_LOG_OFFSET_AT_START)
                chunk = fh.read().decode("utf-8", errors="replace")
        except OSError:
            chunk = ""
        match = _TUNNEL_URL_RE.search(chunk)
        if match:
            return match.group(0)
        time.sleep(poll_interval)
    return None


def stop_langgraph_dev(proc: subprocess.Popen | None = None) -> None:
    """Gracefully stop a langgraph dev process.

    Sends SIGTERM to the process group (langgraph dev spawns worker children),
    falling back to SIGKILL after 5 seconds. Safe to call with ``None``.

    Acquires ``_LOCK`` (reentrant) before mutating ``_PROCESS`` so
    concurrent ``ensure_langgraph_dev`` callers (which also hold ``_LOCK``)
    don't observe partially-cleared state.
    """
    global _PROCESS
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
    if RUNTIME.pid_file.exists():
        try:
            RUNTIME.pid_file.unlink()
        except OSError:
            pass
    _unlink_workspace_sidecar()

    # Note: ``.langgraph_api/`` is intentionally NOT removed — it holds
    # langgraph dev's persisted async-task / scheduler / Store state that
    # may be useful across CLI restarts. It lives in the server home; set
    # ``langgraph_dev_file_persistence: false`` in config to suppress writes.


# =============================================================================
# High-level orchestration
# =============================================================================


def ensure_langgraph_dev(
    config: EvoScientistConfig,
    workspace_dir: Path | str | None = None,
    *,
    backend: str | None = None,
) -> bool:
    """Start or reuse langgraph dev for async/background agent work, and
    register *workspace_dir* with it.

    A server already running on the configured port is reused; otherwise one
    is started, which outlives this process for every session to share until
    ``EvoSci server stop``. Either way the server
    then registers *workspace_dir*, after the locks are released (a
    registration restores the project's threads).

    Returns whether the session has the server it needs: True when it serves
    *workspace_dir* (or, without one, runs), or when no server is needed.
    Errors during startup and a failed registration
    (:class:`~EvoScientist.langgraph_dev.registration.RegistrationError`) are
    logged and return False: the user can still chat with sync sub-agents;
    only async sub-agent calls and EvoMemory background workers will fail.

    Raises a
    :class:`~EvoScientist.langgraph_dev.registration.ServerUnusableError`
    when the server cannot be used for the project at all. For a known
    project whose folder is missing it leaves this process's server state as
    it was, so a session that stays where it is keeps its server; a server
    from an older EvoScientist serves no project of this session.

    Args:
        config: Active EvoScientistConfig.
        workspace_dir: The project root to register. A server this call
            starts reports it at ``GET /api/workspace``. If None, nothing is
            registered.
        backend: The calling surface's resolved gateway backend. When ``None``,
            falls back to the global ``config.gateway_backend``. Determines
            whether the server is needed at all.
    """

    def settle(
        found: _Found,
        *,
        needed: bool = True,
        exposure: registration.ServerExposure | None = None,
    ) -> bool:
        global _ASYNC_SUBAGENTS_AVAILABLE, CONFIG_DRIFT_SINCE_LAUNCH
        global AGENT_SHELL_DRIFT, SERVER_UNAVAILABLE, SERVER_EXPOSURE
        _ASYNC_SUBAGENTS_AVAILABLE = found.available
        CONFIG_DRIFT_SINCE_LAUNCH, AGENT_SHELL_DRIFT = found.drift, found.shell_drift
        SERVER_UNAVAILABLE = needed and not found.available
        SERVER_EXPOSURE = exposure
        return not SERVER_UNAVAILABLE

    if not needs_langgraph_dev(config, backend=backend):
        return settle(_Found(), needed=False)

    # Two layers of locking:
    #   1. ``FileLock`` — cross-process coordination. Without it, two CLI
    #      shells (TUI + ``-p`` + ``serve``) racing on the cold-start window
    #      can SIGKILL each other's still-booting subprocesses via
    #      ``_kill_owned_stale_process`` (Shell A's PID is in the file and
    #      bound to the port, but ``/ok`` isn't responding yet, so Shell B
    #      thinks it's stale).
    #   2. ``_LOCK`` (in-process RLock) — serializes intra-process callers
    #      (rapid ``/resume`` in succession, channel threads).
    lock = FileLock(str(RUNTIME.lock_file), timeout=_FILE_LOCK_TIMEOUT)
    try:
        RUNTIME.pid_dir.mkdir(parents=True, exist_ok=True)
        lock.acquire()
    except FileLockTimeout:
        logger.warning(
            "Timed out waiting %.0fs for cross-process langgraph dev lock at %s. "
            "Another CLI shell may be stuck during cold-start. Falling back to "
            "sync sub-agent delegation for this session.",
            _FILE_LOCK_TIMEOUT,
            RUNTIME.lock_file,
        )
        return settle(_Found())
    except OSError as exc:
        # The runtime folder or its lock cannot be used: a failed start.
        logger.warning(
            "Cannot use the lock %s (%s); falling back to sync sub-agent "
            "delegation for this session.",
            RUNTIME.lock_file,
            exc,
        )
        return settle(_Found())
    try:
        with _LOCK:
            found = _ensure_langgraph_dev_locked(config, workspace_dir)
    except BaseException:
        settle(_Found())
        raise
    finally:
        lock.release()
    if not found.available or workspace_dir is None:
        return settle(found)

    from .sdk import langgraph_dev_url

    workspace = paths.Workspace(workspace_dir)
    try:
        exposure = registration.register_folder(
            workspace, url=langgraph_dev_url(config)
        )
    except registration.OlderServerError:
        # Unusable for any project, the session's current one included.
        settle(_Found())
        raise
    except ServerUnusableError:
        # The session stays in its project: its state stays as it was.
        raise
    except RegistrationError as exc:
        logger.warning(
            "%s Falling back to sync sub-agent delegation for this session.",
            exc,
        )
        return settle(replace(found, available=False))
    except BaseException:
        settle(_Found())
        raise
    return settle(found, exposure=exposure)


def serve_project(
    config: EvoScientistConfig, workspace_dir: Path, *, with_ccproxy: bool = True
) -> bool:
    """Start the background server, or reuse a running one, for
    *workspace_dir* (``EvoSci service start``, ``EvoSci server start``, the
    restart after an update). With Anthropic OAuth the server routes through
    ccproxy: *with_ccproxy* starts it first, lasting like the server
    (``RuntimeError`` or ``ValueError`` when it cannot start). Returns and
    raises like :func:`ensure_langgraph_dev`."""
    from ..ccproxy_manager import maybe_start_ccproxy, setup_ccproxy_env, uses_oauth

    if with_ccproxy:
        maybe_start_ccproxy(config, lasting=True)
    elif uses_oauth(config):
        setup_ccproxy_env(config.ccproxy_port)
    return ensure_langgraph_dev(
        config, workspace_dir=workspace_dir, backend="langgraph_server"
    )


@dataclass(frozen=True)
class _Found:
    """What the locked section found or started."""

    available: bool = False
    drift: bool = False
    shell_drift: str | None = None


def _ensure_langgraph_dev_locked(
    config: EvoScientistConfig,
    workspace_dir: Path | str | None,
) -> _Found:
    """Locked critical section of ``ensure_langgraph_dev`` — must hold ``_LOCK``."""
    config_fp = _server_config_fingerprint(config)
    port, host = configured_address(config)
    file_persistence = bool(getattr(config, "langgraph_dev_file_persistence", True))
    jobs_per_worker = int(getattr(config, "langgraph_dev_jobs_per_worker", 10))
    ws_path = Path(workspace_dir) if workspace_dir is not None else None

    if is_langgraph_dev_running(port=port, host=host):
        # The server serves every registered project, so any EvoScientist
        # server on the port will do; the registration that follows tells
        # one from an older release. A server this process did not start
        # (EvoSci deploy in another terminal, an earlier session) may have
        # been launched with other settings: its ownership record says.
        owned_running = _PROCESS is not None and _PROCESS.poll() is None
        sidecar = None if owned_running else _read_workspace_sidecar()
        drift, shell_drift = False, None
        if sidecar is not None:
            recorded_fp = sidecar.get("config_fingerprint")
            if isinstance(recorded_fp, str) and recorded_fp != config_fp:
                drift = True
                logger.warning(
                    "Configuration or version changed since the running "
                    "langgraph dev was launched — it still serves the old "
                    "settings or version until it is restarted "
                    "(EvoSci server stop)."
                )
            from EvoScientist.agent_shell import shell_drift_message

            shell_drift = shell_drift_message(sidecar)
            if shell_drift is not None:
                # INFO: the CLI prints it after startup and the TUI shows
                # it in the app; a WARNING here would print it twice.
                logger.info(shell_drift)
        logger.info(
            "langgraph dev already running on %s, reusing", _base_url(port, host)
        )
        return _Found(available=True, drift=drift, shell_drift=shell_drift)

    try:
        start_langgraph_dev(
            workspace_dir=ws_path,
            port=port,
            host=host,
            file_persistence=file_persistence,
            jobs_per_worker=jobs_per_worker,
            config_fingerprint=config_fp,
            allowed_origins=getattr(config, "backend_allowed_origins", ""),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        # Startup failed — keep async subagents disabled so the main agent
        # falls back to in-process sync delegation rather than routing tool
        # calls at a dead URL. EvoMemory workers will also skip until the
        # server is reachable.
        logger.warning(
            "Failed to start langgraph dev — async sub-agents will fall back "
            "to in-process delegation, and EvoMemory background workers will "
            "not run. %s",
            exc,
        )
        return _Found()

    # The server outlives this session, with its PID file and sidecar, for
    # every session to share until ``EvoSci server stop``.
    return _Found(available=True)
