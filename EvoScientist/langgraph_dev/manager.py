"""langgraph dev lifecycle management for background agent support.

Provides functions to start/stop/health-check a ``langgraph dev`` subprocess
that hosts the EvoScientist main agent, async sub-agents (e.g.
``writing-agent``), and EvoMemory background workers. The CLI calls
``ensure_server(config, ...)`` at startup so users can run
``EvoSci -p "..."`` without manually managing the langgraph dev server.

One server serves a data folder. It records itself in the server home
(:mod:`.server_record`), and every client finds it there
(:func:`read_record`).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
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
from EvoScientist.langgraph_dev import (
    boundary,
    registration,
    sdk,
    server_home,
    server_record,
)
from EvoScientist.langgraph_dev.registration import (
    OlderServerError,
    RegistrationError,
    ServerUnusableError,
)

if TYPE_CHECKING:
    from EvoScientist.deploy.backend_job import BackendJob

logger = logging.getLogger(__name__)


class LanggraphRuntimePaths:
    """The files of this data folder's server, in the server home
    (``paths.SERVER_HOME``): the server's record, the lock the server holds,
    the lock a start holds, the server's log and the WebUI's.

    Each is read from ``paths.SERVER_HOME`` when used, so the files follow it
    (tests point it at a temporary folder).
    """

    @property
    def record(self) -> Path:
        return server_record.record_path()

    @property
    def server_lock(self) -> Path:
        return server_home.lock_path()

    @property
    def lock_file(self) -> Path:
        return paths.SERVER_HOME / "langgraph_dev.lock"

    @property
    def log_file(self) -> Path:
        return paths.SERVER_HOME / "langgraph_dev.log"

    @property
    def ccproxy_log(self) -> Path:
        """The output of the ccproxy the server starts (Anthropic OAuth)."""
        return paths.SERVER_HOME / "ccproxy.log"

    @property
    def ccproxy_failure(self) -> Path:
        """Why the server could not start its ccproxy, written as it exits
        for its starter to tell the user."""
        return paths.SERVER_HOME / "ccproxy_failure.txt"

    @property
    def webui_log(self) -> Path:
        return paths.SERVER_HOME / "webui.log"


RUNTIME = LanggraphRuntimePaths()

# Where releases before the server home kept the record, the PID file, the
# start lock and the logs, one set for every data folder. A server such a
# release started is still found there (:func:`read_record`).
OLDER_RUNTIME_DIR = Path.home() / ".config" / "evoscientist"
_OLDER_PID_FILE = "langgraph_dev.pid"


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
    from ..ccproxy_manager import uses_oauth

    if uses_oauth(config):
        # Anthropic's OAuth goes through the ccproxy the server runs.
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
# mutations and the ``ensure_server`` decision/start/stop flow.
# Reentrant because a start that fails inside ``ensure_server``'s
# critical section calls ``stop_langgraph_dev``, which takes it too.
_LOCK = threading.RLock()


# Set by ``ensure_server`` when it reuses a server whose
# recorded launch-time config fingerprint differs from the current effective
# config. The CLI reads it after startup to surface a "restart to apply"
# hint — the server itself is never restarted automatically.
CONFIG_DRIFT_SINCE_LAUNCH = False

# Set by ``ensure_server`` to the warning text when it reuses a server
# whose agents got another ``python`` or shell than this session resolves (see
# ``agent_shell.shell_drift_message``). The CLI prints it after startup.
AGENT_SHELL_DRIFT: str | None = None

# Set by ``ensure_server`` to the notice when it reuses a server an older
# release started (:attr:`ServerResult.older_message`).
OLDER_SERVER: str | None = None

# Set by ``ensure_server`` to the notice when Anthropic's OAuth goes through
# the server's ccproxy and nothing answers on ``ccproxy_port``
# (:func:`_ccproxy_missing`).
CCPROXY_MISSING: str | None = None

# Set by ``ensure_server``: the session needs the server and does not
# have it (it did not start, or did not register the project). The TUI shows
# :data:`SERVER_UNAVAILABLE_MESSAGE` in the app, where the terminal is hidden,
# with :data:`SERVER_UNAVAILABLE_REASON` when there is one.
SERVER_UNAVAILABLE = False
SERVER_UNAVAILABLE_REASON: str | None = None
# Set by ``ensure_server``: how the server that registered the
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
OLDER_SERVER_MESSAGE = (
    "The background agent server was started by an older EvoScientist "
    "release; restart it to run this version: EvoSci server stop; it starts "
    "again with your next session."
)


# Default port (Kaprekar's constant — see config/settings.py for the rationale).
# Overridable per-call via ``ensure_server(port=...)`` (which by default reads
# ``config.langgraph_dev_port``) and the
# corresponding url= field on AsyncSubAgent specs.
_DEFAULT_PORT = 6174

# Default bind interface — loopback, matching ``config.langgraph_dev_host``.
# SECURITY: this is the unauthenticated agent API; launchers print a PUBLIC
# BIND banner while it is exposed.
_DEFAULT_HOST = server_record.DEFAULT_HOST


def _probe_host(host: str = _DEFAULT_HOST) -> str:
    """Map a bind address to one a client can actually connect to.

    A wildcard bind includes loopback, so clients use loopback: you cannot
    meaningfully *connect* to it (0.0.0.0 is routed to loopback on Linux
    and outright rejected on Windows). ``::`` maps to ``::1``, as an IPv6
    socket may not take IPv4 (it does not by default on Windows). A
    specific interface is returned as-is — loopback would not reach it.
    """
    if not boundary.is_wildcard_host(host):
        return host
    return "::1" if host.strip().strip("[]") == "::" else "127.0.0.1"


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
# exceeds this size, the next start (``_spawn_server``) rotates
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
    must NOT block a start — the worst case is the log
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


# The server's record (``server_record``): which server serves this data
# folder, at which address, and what it was launched with. Every reader goes
# through :func:`read_record`.


@dataclass(frozen=True)
class ServerRecord:
    """This data folder's running server, as its record names it.

    ``pid`` is the process its starter started (the root of the tree a stop
    ends); ``created`` its creation time, which a stop checks so it never
    signals a process that reused the PID (None in a record of an older
    release). ``phase`` is ``starting`` until the server serves, then
    ``serving``; a record without one (an older release) counts as serving.
    ``lasting`` is whether it outlives its starter, None when the record does
    not say (a server started by hand, an older release). ``fields`` is the
    record as written, for the drift checks. ``older`` is True for the
    record an older release kept in :data:`OLDER_RUNTIME_DIR`.
    """

    pid: int
    port: int
    host: str
    created: float | None = None
    phase: str = server_record.SERVING
    lasting: bool | None = None
    break_group: bool = False
    fields: dict = dataclass_field(default_factory=dict)
    older: bool = False

    @property
    def url(self) -> str:
        return _base_url(self.port, self.host)

    def same_server(self, other: ServerRecord | None) -> bool:
        """Whether *other* names the same server process."""
        return other is not None and _identity(other) == _identity(self)

    @property
    def data_dir(self) -> str:
        return str(self.fields.get("data_dir") or paths.DATA_DIR)

    @property
    def config_dir(self) -> str:
        return str(self.fields.get("config_dir") or paths.CONFIG_DIR)


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def read_record(*, older: bool = False, lock: str | None = None) -> ServerRecord | None:
    """The record of this data folder's running server, or None.

    A record counts when it parses, the server lock is held and its PID is
    still the process it recorded (:func:`_runs_as_recorded`): the server
    writes it right after it takes the lock, and a server that cannot write
    it exits. A record a stopped or killed server left is ignored once its
    lock is free, or, while a new server holds it, once its PID is gone or
    names another process. Without one in the
    server home, and with *older*, the record an older release kept in
    :data:`OLDER_RUNTIME_DIR` counts when it names this data folder and a
    live langgraph dev that serves the port it recorded
    (:func:`_older_record`). *lock* is what the caller found of the lock
    (:func:`server_home.lock_state`); checked here when not given. A lock
    that cannot be checked proves nothing: the record is not trusted.
    """
    data = server_record.load_json(RUNTIME.record)
    if data is not None:
        record = _current_record(data)
        if lock is None:
            lock = server_home.lock_state()
        if (
            record is not None
            and lock == server_home.HELD
            and _runs_as_recorded(record)
        ):
            return record
    return _older_record() if older else None


def _runs_as_recorded(record: ServerRecord) -> bool:
    """Whether the record's PID is alive and still the process it recorded:
    the same creation time, one that cannot be read, or this server home's
    langgraph dev (:func:`_serves_this_home`)."""
    if not psutil.pid_exists(record.pid):
        return False
    if record.created is None:
        return True
    now = _create_time(record.pid)
    if now is None or abs(now - record.created) <= _CREATE_TIME_TOLERANCE:
        return True
    return _serves_this_home(record.pid)


def _serves_this_home(pid: int) -> bool:
    """Whether *pid* is a langgraph dev working in this data folder's server
    home, as every server is started: a clock step moves the creation time
    psutil reports for it."""
    if _langgraph_cmdline(pid) is None:
        return False
    try:
        cwd = psutil.Process(pid).cwd()
        return paths.normalize_path(cwd) == paths.normalize_path(paths.SERVER_HOME)
    except (psutil.Error, OSError, ValueError):
        return False


def _current_record(data: dict) -> ServerRecord | None:
    pid = _positive_int(data.get("pid"))
    port = _positive_int(data.get("port"))
    if pid is None or port is None:
        return None
    created = data.get(server_record.CREATE_TIME_KEY)
    if isinstance(created, bool) or not isinstance(created, int | float):
        created = None
    phase = data.get(server_record.PHASE_KEY)
    lasting = data.get("lasting")
    return ServerRecord(
        pid=pid,
        port=port,
        host=str(data.get("host") or _DEFAULT_HOST),
        created=float(created) if created is not None else None,
        phase=phase if phase == server_record.STARTING else server_record.SERVING,
        lasting=lasting if isinstance(lasting, bool) else None,
        break_group=data.get(server_record.BREAK_GROUP_KEY) is True,
        fields=data,
    )


def _same_folder(recorded: object, folder: Path) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    try:
        return paths.normalize_path(recorded) == paths.normalize_path(folder)
    except (OSError, ValueError):
        return False


def _older_record() -> ServerRecord | None:
    """The record an older release kept for this data folder's server, while
    that server runs. Its PID file and record are deleted once the server
    they name is gone (no process has its PID, or another one does); a
    record of another data folder's server is left, and so is one whose
    process refuses access, which this read then ignores."""
    data = server_record.load_json(OLDER_RUNTIME_DIR / server_record.RECORD_FILE)
    if data is None or not _same_folder(data.get("data_dir"), paths.DATA_DIR):
        return None
    pid = _positive_int(data.get("pid"))
    try:
        cmdline = _process_cmdline(pid) if pid is not None else None
    except psutil.AccessDenied:
        return None
    if cmdline is None or not server_record.is_langgraph_dev(cmdline):
        _forget_older_record()
        return None
    bound_port, bound_host = server_record.address_of(cmdline)
    port = _positive_int(data.get("port")) or bound_port
    if port != bound_port:
        # The PID went to another langgraph dev.
        _forget_older_record()
        return None
    created = data.get("ctrl_break_create_time")
    lasting = data.get("lasting")
    return ServerRecord(
        pid=pid,
        port=port,
        # Releases before the bind host was configurable recorded none and
        # bound loopback.
        host=str(data.get("host") or bound_host),
        lasting=lasting if isinstance(lasting, bool) else None,
        break_group=isinstance(created, float) and _same_process(pid, created),
        fields=data,
        older=True,
    )


def _forget_older_record() -> None:
    """Delete the PID file and record an older release kept."""
    for name in (_OLDER_PID_FILE, server_record.RECORD_FILE):
        with contextlib.suppress(OSError):
            (OLDER_RUNTIME_DIR / name).unlink()


def log_file_of(record: ServerRecord | None) -> Path:
    """The log the recorded server writes: an older release's in
    :data:`OLDER_RUNTIME_DIR`, any other in the server home."""
    if record is not None and record.older:
        return OLDER_RUNTIME_DIR / "langgraph_dev.log"
    return RUNTIME.log_file


def is_lasting(record: ServerRecord | None) -> bool:
    """Whether the recorded server outlives the process that started it,
    shared by every session; not one ``EvoSci deploy`` or the desktop app
    owns, nor one whose record does not say."""
    return record is not None and record.lasting is True


# Module-level handle to the langgraph dev subprocess we started, if any.
# Stays None when we reused an existing process (managed by the user).
_PROCESS: subprocess.Popen | None = None

# Windows: servers this process spawned as the leader of their own process
# group, the only PIDs a Ctrl+Break may be sent to (for any other PID Windows
# sends it to every process on the console). The record says the same for a
# server another process started (``break_group``).
_BREAK_GROUP_PIDS: set[int] = set()
_CREATE_TIME_TOLERANCE = 1.0

# Windows: the access right a stop asks for to hold a recorded server's PID
# (:func:`_hold_pid`), and the error ``OpenProcess`` gives for a PID no process
# has. Query-limited access is enough, as nothing waits on the handle, and an
# elevated process grants it to a non-elevated caller, where it refuses
# ``PROCESS_QUERY_INFORMATION``.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_INVALID_PARAMETER = 87
_ERROR_ACCESS_DENIED = 5
_ERROR_PRIVILEGE_NOT_HELD = 1314

# How long a stop on Windows waits for the server to exit after Ctrl+Break
# before it ends the process tree hard, and how long the helper that sends it
# may take.
CLEAN_STOP_TIMEOUT = 10.0
_CLEAN_STOP_HELPER_TIMEOUT = 5.0
# How long a stop waits after terminate() before it kills, and how long the
# server's children get to finish their own shutdown after the server exited.
_TERMINATE_TIMEOUT = 5.0
_SURVIVOR_GRACE = 2.0

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
# cloudflared's quick-tunnel creation failures ("failed to request quick
# Tunnel: ...", "failed to read quick-tunnel response: ..."); it exits after one.
_TUNNEL_FAILED_RE = re.compile(
    r"failed to [a-z ]*quick[ -]tunnel[^\r\n]*", re.IGNORECASE
)

# Whether async sub-agents are usable in this process.
#
# - CLI / serve parent process: starts False; flipped True after
#   ``ensure_server`` confirms the subprocess is healthy. Stays False
#   on startup failure so ``_maybe_swap_async_subagents`` can fall back to
#   in-process sync delegation instead of routing tool calls at a dead URL.
# - langgraph dev subprocess: starts True (``paths.SERVER_PROCESS``, from the
#   variable ``_spawn_server`` sets). The served main agent runs inside
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
    timeout: float = 1.0,
) -> bool:
    """Check whether a langgraph dev API is already serving at ``base_url``,
    answering ``/ok`` within *timeout* seconds.

    ``base_url`` overrides ``port``/``host`` when given.
    """
    url = base_url or _base_url(port, host)
    try:
        # trust_env=False: this is a loopback probe of our own server. httpx
        # otherwise routes it through the environment/OS proxy — and on Windows
        # getproxies() reads the system (registry/IE) proxy even with no *_PROXY
        # env vars set, so a corporate proxy silently swallows the 127.0.0.1
        # request and the health check never sees the healthy server.
        return (
            httpx.get(f"{url}/ok", timeout=timeout, trust_env=False).status_code == 200
        )
    except (httpx.TransportError, OSError):
        return False


def _can_bind_port(port: int, host: str = _DEFAULT_HOST) -> bool:
    """Return True if a fresh ``bind()`` to ``host:port`` succeeds right now.

    More reliable than ``_is_port_occupied`` when the previous listener has
    just exited: ``connect_ex`` can already report "free" while ``bind()``
    still fails because the kernel hasn't fully released the socket.

    On POSIX it sets ``SO_REUSEADDR`` like uvicorn does, so connections the
    old server left in TIME_WAIT don't make a port look busy that the server
    could bind. Not on Windows, where the option also allows binding over a
    live listener.

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
        s.close()


# How long a start waits for a listener that was just stopped to free the port.
_BIND_WAIT = 10.0


def _wait_for_port_bindable(port: int, host: str = _DEFAULT_HOST) -> bool:
    """Poll until a real ``bind()`` to ``port`` can succeed, for up to
    :data:`_BIND_WAIT`; right before the spawn, so a server that has just
    exited does not make the new one fail its bind."""
    deadline = time.monotonic() + _BIND_WAIT
    while True:
        if _can_bind_port(port, host):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_INTERVAL)


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
_UNFINGERPRINTED_FIELDS = frozenset({"langgraph_dev_port"})


def _server_config_fingerprint(config: EvoScientistConfig) -> str:
    """Hash of everything the langgraph dev subprocess consumes at launch.

    Deployed graphs read config once at import (``subagents/_factory.py``,
    ``EvoScientist.py``), so a running server keeps serving those values
    until restarted. Iterates the full ``EvoScientistConfig`` field list
    minus the fields that never reach the server
    (:func:`~EvoScientist.config.settings.is_client_only`) and the port,
    which the record names and every caller compares on its own — a new config
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
        if is_client_only(name) or name in _UNFINGERPRINTED_FIELDS:
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
    """The stop left the recorded server running. ``EvoSci server stop``
    keeps a server that ``EvoSci deploy`` or the desktop app owns; any stop
    raises :class:`ServerStopFailedError` when it could not end the server
    for a reason other than it being gone."""


class ServerStopFailedError(ServerKeptError):
    """The stop could not end the recorded server, for a reason other than
    the server being gone: access refused (another user's or an elevated
    server), or on Windows out of resources. It stopped nothing; the record
    stays."""


def _stop_failed(pid: int, cause: str) -> ServerStopFailedError:
    """The error of a stop that left the server *pid* running, and why."""
    return ServerStopFailedError(f"Could not stop langgraph dev (pid {pid}): {cause}.")


def _refused_cause() -> str:
    """Why a stop that was refused access stopped nothing, and what to do."""
    if os.name == "nt":
        return (
            "it runs elevated or as another user; stop it from an "
            "Administrator terminal"
        )
    return "it belongs to another user (as after a sudo EvoSci); stop it as that user"


def owned_by_its_starter(record: ServerRecord | None) -> bool:
    """Whether the recorded server is one that ``EvoSci deploy`` or the
    desktop app owns, not the background server: ``EvoSci server stop``
    leaves it running. A record that does not say counts as not."""
    return record is not None and record.lasting is False


def _request_clean_stop(pid: int) -> bool:
    """Windows: send Ctrl+Break to the server ``pid``, which must lead its own
    process group. True when it was sent.

    A helper process sends it (``console_break``): only a process attached to
    the server's console can, and attaching would cut this one off its own
    terminal. ``CREATE_NO_WINDOW`` keeps a windowless caller (the desktop app)
    from flashing a console window. The helper needs only the standard
    library, so it runs by its file path under ``-I``: found in any install
    (a user-site one included), and nothing in the caller's working directory
    or on ``PYTHONPATH`` (a project's ``signal.py``) shadows the standard
    library.
    """
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                os.path.join(os.path.dirname(__file__), "console_break.py"),
                str(pid),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CLEAN_STOP_HELPER_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            f"Could not ask langgraph dev (pid {pid}) to shut down cleanly "
            f"({exc}); ending it hard, so a run it was running stays unfinished."
        )
        return False
    if result.returncode != 0:
        logger.warning(
            f"Could not ask langgraph dev (pid {pid}) to shut down cleanly: the "
            f"Ctrl+Break helper exited with {result.returncode} "
            f"({result.stderr.strip()}); ending it hard, so a run it was "
            "running stays unfinished."
        )
        return False
    return True


def _wait_exit(
    process: psutil.Process, popen: subprocess.Popen | None, timeout: float
) -> bool:
    """Wait up to ``timeout`` for ``process`` to exit; through ``popen`` when we
    spawned it, so the handle is reaped. True when it exited."""
    try:
        if popen is not None:
            popen.wait(timeout=timeout)
        else:
            process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
        return False
    return True


def _end_server_tree(
    pid: int,
    popen: subprocess.Popen | None = None,
    *,
    break_group: bool = False,
) -> None:
    """End the langgraph dev ``pid`` and every process it started.

    With ``break_group`` (Windows, ``pid`` known to lead its own process
    group) it first asks the server to shut down cleanly with Ctrl+Break and
    waits up to :data:`CLEAN_STOP_TIMEOUT`: only a clean shutdown puts a run
    the server was running back in the queue for the next server. Then, and
    everywhere else, terminate, wait :data:`_TERMINATE_TIMEOUT`, kill: SIGTERM
    then SIGKILL on POSIX,
    where SIGTERM already shuts the server down cleanly.

    The children are listed first: once the server has exited they can no
    longer be found from it. Lets ``psutil.AccessDenied`` on ``pid`` through.
    """
    try:
        server = psutil.Process(pid)
        children = server.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    if break_group and _request_clean_stop(pid):
        if _wait_exit(server, popen, CLEAN_STOP_TIMEOUT):
            _kill_survivors(children)
            return
        logger.warning(
            f"langgraph dev (pid {pid}) did not exit within "
            f"{CLEAN_STOP_TIMEOUT:g} s of Ctrl+Break; ending it hard, so a run "
            "it was running stays unfinished."
        )
    try:
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        server.terminate()
        if not _wait_exit(server, popen, _TERMINATE_TIMEOUT):
            for child in {*children, *server.children(recursive=True)}:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            server.kill()
            # Reap our handle; the kill above was SIGKILL / TerminateProcess.
            _wait_exit(server, popen, 2)
    except psutil.NoSuchProcess:
        pass
    _kill_survivors(children)


def _kill_survivors(children: list[psutil.Process]) -> None:
    """The server exiting does not prove its children did: they got the same
    signal, so give them :data:`_SURVIVOR_GRACE` to finish their own shutdown,
    then kill what is left. A wait that is refused (on Windows an elevated
    child denies the handle) counts every child as left."""
    try:
        _, alive = psutil.wait_procs(children, timeout=_SURVIVOR_GRACE)
    except psutil.Error:
        alive = children
    for child in alive:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _create_time(pid: int) -> float | None:
    """``pid``'s creation time, or None when it cannot be read."""
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _same_process(pid: int, created: float) -> bool:
    """Whether ``pid`` is still the process created at ``created``."""
    now = _create_time(pid)
    return now is not None and abs(now - created) <= _CREATE_TIME_TOLERANCE


def _kernel32():
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32, ctypes.get_last_error


@contextlib.contextmanager
def _hold_pid(pid: int):
    """Windows: hold a handle to ``pid`` for the block. While it is open,
    Windows does not give the PID to another process, so the checks of a
    stop and the Ctrl+Break and kills that follow reach the same process,
    even if it exits in between. When the handle is refused, nothing runs:
    ``psutil.NoSuchProcess`` when no process has ``pid``,
    ``psutil.AccessDenied`` for the errors psutil counts as permission
    errors, ``OSError`` with the system's message for any other. Off Windows
    it holds nothing."""
    if os.name != "nt":
        yield
        return
    kernel32, get_last_error = _kernel32()
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = get_last_error()
        if error == _ERROR_INVALID_PARAMETER:
            raise psutil.NoSuchProcess(pid)
        if error in (_ERROR_ACCESS_DENIED, _ERROR_PRIVILEGE_NOT_HELD):
            raise psutil.AccessDenied(pid)
        import ctypes

        raise ctypes.WinError(error)
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)


def read_record_and_lock() -> tuple[ServerRecord | None, str]:
    """The record (``read_record(older=True)``) and the server lock state it
    was read under (:func:`lock_problem` says what a lock without a record
    means)."""
    lock = server_home.lock_state()
    return read_record(older=True, lock=lock), lock


def reread_record(
    asked: ServerRecord | None,
) -> tuple[ServerRecord | None, str, bool]:
    """The record read again after a wait (a question to the user) about the
    server *asked* names, the lock state it was read under, and whether it
    is still that server or none: a server that appeared meanwhile, or took
    its place, is not."""
    current, lock = read_record_and_lock()
    same = current is None or (asked is not None and asked.same_server(current))
    return current, lock, same


def starting_without_record() -> bool:
    """With no record read: whether a server is starting, its start lock
    held by another process, or its record written since the read."""
    return start_in_progress() or read_record(older=True) is not None


def start_in_progress() -> bool:
    """Whether another process holds the start lock: it is starting a server
    that has no record yet."""
    if not paths.SERVER_HOME.is_dir():
        return False
    lock = FileLock(str(RUNTIME.lock_file))
    try:
        lock.acquire(timeout=0)
    except FileLockTimeout:
        return True
    except OSError:
        return False
    lock.release()
    return False


# What :func:`stop_server` takes when its caller has not read the record.
_UNREAD = object()


def stop_server(*, keep_owned: bool = False, record: object = _UNREAD) -> int | None:
    """Stop this data folder's recorded server (``EvoSci server stop``).

    Reads only the record (:func:`read_record`): a server without one is
    never touched. The PID is held from before the check that it is still
    the recorded process until the stop is done (:func:`_hold_pid`), and the
    whole process tree ends (:func:`_end_server_tree`: on Windows Ctrl+Break
    first when the record says the server leads its process group). Waits
    for it to exit, then removes its record, which the server's own removal
    at exit never reaches after a signal. Returns the stopped pid, or ``None`` when
    nothing was stopped (no record, or a server that is gone).

    *record* is what the caller read (``read_record(older=True)``); read here
    when not given. With *keep_owned* (``EvoSci server stop``), a server
    that ``EvoSci deploy`` or the desktop app owns is left running:
    :class:`ServerKeptError`. One the stop may not end (access refused) or
    could not open is left running too: :class:`ServerStopFailedError`.
    """
    if record is _UNREAD:
        record = read_record(older=True)
    if not isinstance(record, ServerRecord):
        return None
    if keep_owned and owned_by_its_starter(record):
        raise ServerKeptError(
            "The running server is not the background server: an "
            "EvoSci deploy or the desktop app owns it, so it was left "
            "running; it stops with its owner."
        )
    with _LOCK:
        own = _PROCESS if _PROCESS is not None and _PROCESS.pid == record.pid else None
    with contextlib.ExitStack() as hold:
        try:
            try:
                hold.enter_context(_hold_pid(record.pid))
                if not _still_recorded(record):
                    raise psutil.NoSuchProcess(record.pid)
            except psutil.NoSuchProcess:
                if record.older:
                    _forget_older_record()
                return None
            except OSError as exc:
                # Neither gone nor refused (the hold, or an OS error psutil
                # passes through): the server may be running, so it gets no
                # signal.
                raise _stop_failed(
                    record.pid, f"{str(exc).rstrip('.')}. It keeps running; try again"
                ) from exc
            try:
                _end_server_tree(record.pid, own, break_group=record.break_group)
            except psutil.NoSuchProcess:
                pass
        except psutil.AccessDenied as exc:
            # Opening, inspecting or signalling it refused: another user's
            # or an elevated server.
            raise _stop_failed(record.pid, _refused_cause()) from exc
    if own is not None:
        stop_langgraph_dev(own)
    if record.older:
        _forget_older_record()
    else:
        _forget_record(record.pid)
    return record.pid


def _forget_record(pid: int) -> None:
    """Delete the record that names *pid*, a server a stop ended, while no
    server holds the lock: the stop's signal ends the server before its own
    removal at exit runs. First end the ccproxy it names: a run that started
    while the server stopped started it again after the stop ended the
    server's children."""
    data = server_record.load_json(RUNTIME.record)
    if data is None or data.get("pid") != pid:
        return
    lock = FileLock(str(RUNTIME.server_lock))
    try:
        lock.acquire(timeout=server_home.PROBE_WAIT)
    except (FileLockTimeout, OSError):
        return
    try:
        data = server_record.load_json(RUNTIME.record)
        if data is not None and data.get("pid") == pid:
            from ..config import get_effective_config
            from . import server_ccproxy

            server_ccproxy.end_recorded(data, get_effective_config().ccproxy_port)
            with contextlib.suppress(OSError):
                RUNTIME.record.unlink()
    finally:
        lock.release()


def _still_recorded(record: ServerRecord) -> bool:
    """Whether the recorded process is still the one the record names: the
    same creation time or this server home's langgraph dev
    (:func:`_serves_this_home`, as :func:`read_record` takes it), or for a
    record without one (an older release's), a langgraph dev."""
    if record.created is not None:
        return _same_process(record.pid, record.created) or _serves_this_home(
            record.pid
        )
    return _langgraph_cmdline(record.pid) is not None


def _process_cmdline(pid: int) -> list[str] | None:
    """The command line of ``pid``, or ``None`` when no process has it.
    Lets ``psutil.AccessDenied`` through."""
    try:
        return psutil.Process(pid).cmdline()
    except psutil.NoSuchProcess:
        return None


def _langgraph_cmdline(pid: object) -> list[str] | None:
    """The command line of ``pid`` while it is a live langgraph dev, on any
    port; ``None`` otherwise."""
    pid = _positive_int(pid)
    if pid is None:
        return None
    try:
        cmdline = _process_cmdline(pid)
    except psutil.AccessDenied:
        return None
    if cmdline is None:
        return None
    return cmdline if server_record.is_langgraph_dev(cmdline) else None


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
# one sees the drift warning. With Anthropic's OAuth the server starts its
# ccproxy.
_FLAGS_SERVER_TAKES = frozenset(
    {"EVOSCIENTIST_ANTHROPIC_AUTH_MODE", "EVOSCIENTIST_OPENAI_AUTH_MODE"}
)


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
    :func:`_spawn_server` always sets for the server.

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


def _spawn_server(
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
    """Start langgraph dev as a background subprocess, without waiting for it.

    The server works in the server home (``paths.SERVER_HOME``), which holds
    its runtime store, and serves every registered project. It records
    itself there (``server_record``) with what this process passes it.
    Only :func:`ensure_server` calls this, holding the start lock, once it
    created the server home.

    Args:
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
        config_fingerprint: The configuration the server is started for,
            recorded for a later session's drift check.
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
    # gets no workspace of its own. A run names its run folder itself.
    # ``EVOSCIENTIST_WORKSPACE_DIR`` passes as inherited: it is the user's
    # ``default_workdir`` override (Docker sets it), which the settings API
    # reports. A project ``.env`` sets only this command's own
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
    # What the server records about itself but cannot know: which process
    # started it, whether it outlives that process, and the configuration it
    # was started for. It removes the variables once read.
    sub_env[server_record.SPAWNER_PID_VAR] = str(os.getpid())
    sub_env[server_record.LASTING_VAR] = "1" if lasting else "0"
    if config_fingerprint is not None:
        sub_env[server_record.FINGERPRINT_VAR] = config_fingerprint
    else:
        sub_env.pop(server_record.FINGERPRINT_VAR, None)

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
            cwd=str(paths.SERVER_HOME),
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
    if os.name == "nt":
        _BREAK_GROUP_PIDS.add(proc.pid)
    if job is not None:
        job.started(proc)
    logger.info("Started langgraph dev (pid=%d)", proc.pid)
    return proc


class TunnelFailedError(RuntimeError):
    """cloudflared logged that it could not create the quick tunnel; the
    message is its log line."""


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

    Raises:
        TunnelFailedError: cloudflared logged a quick-tunnel failure, so no
            URL will follow.
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
        failure = _TUNNEL_FAILED_RE.search(chunk)
        if failure:
            raise TunnelFailedError(failure.group(0).strip())
        time.sleep(poll_interval)
    return None


def stop_langgraph_dev(
    proc: subprocess.Popen | None = None, *, clean: bool = True
) -> None:
    """Stop a langgraph dev process this process started.

    Sends SIGTERM to the process tree (langgraph dev spawns worker children),
    falling back to SIGKILL after 5 seconds. On Windows a server this process
    spawned first gets Ctrl+Break and up to :data:`CLEAN_STOP_TIMEOUT` to
    exit before that (:func:`_end_server_tree`); ``clean=False``
    skips that for a server that never became ready, which has no run to put
    back in the queue and would only finish loading first. Safe to call with
    ``None``. Then its record is removed (:func:`_forget_record`).

    Acquires ``_LOCK`` (reentrant) before mutating ``_PROCESS`` so
    concurrent ``ensure_server`` callers (which also hold ``_LOCK``)
    don't observe partially-cleared state.

    ``.langgraph_api/`` is kept: it holds langgraph dev's persisted
    async-task / scheduler / Store state for the next server. It lives in the
    server home; set ``langgraph_dev_file_persistence: false`` in config to
    suppress writes.
    """
    global _PROCESS
    with _LOCK:
        proc = proc if proc is not None else _PROCESS
        if proc is None:
            return
        if proc.poll() is None:
            # Walk the tree explicitly: POSIX process groups
            # (``os.killpg``) don't exist on Windows.
            _end_server_tree(
                proc.pid,
                proc,
                break_group=clean and proc.pid in _BREAK_GROUP_PIDS,
            )
        _BREAK_GROUP_PIDS.discard(proc.pid)
        if proc is _PROCESS:
            _PROCESS = None
    _forget_record(proc.pid)


# =============================================================================
# High-level orchestration
# =============================================================================

# How long a start may take; see :func:`start_budget`.
_START_BUDGET = 120.0
# How long one probe of a server may take: a busy server answers late, and is
# not gone for it.
_PROBE_TIMEOUT = 5.0
# How many probes in a row a serving server may miss before it counts as not
# answering.
_SERVING_PROBES = 3
# How long a recorded server may hold its lock without listening: a stopping
# server drains its requests and gives its workers 5 s before it exits.
_DRAIN_WAIT = 20.0
# How long the server lock may be held without a record: a server writes it
# right after it takes the lock.
_SETTLE_WAIT = 2.0
_POLL_INTERVAL = 0.5
_LOG_TAIL_CHARS = 2000


def start_budget(config: EvoScientistConfig) -> float:
    """How long a server's start may take, and so each wait for one: for the
    start lock, for a started server's record, and for its first answer.
    With Anthropic OAuth the server starts its ccproxy first. Read from this
    client's configuration: a server another client started with OAuth
    (``--auth-mode oauth``) while this one's file says ``api_key`` can take
    longer than this client waits."""
    from .. import ccproxy_manager

    if ccproxy_manager.uses_oauth(config):
        # Its port may need a moment to come free first.
        return _START_BUDGET + _BIND_WAIT + ccproxy_manager.START_BUDGET_SECONDS
    return _START_BUDGET


@dataclass(frozen=True)
class ServerResult:
    """What :func:`ensure_server` found or started.

    ``available``: a server answers at ``url`` (``port``, ``host``).
    ``started``: this call started it, and ``proc`` is its handle.
    ``record``: the server's record, None for a server without one (started
    by hand outside the server home). ``drift`` and ``shell_drift``: a
    reused server was launched with another configuration or version, or
    gives its agents another ``python`` or shell. ``served``: the session has
    the server it needs, or needs none.

    Without a server, ``error`` says why and ``problem`` names it:
    ``foreign`` (another program holds the port), ``other_data_folder``
    (another data folder's server answers on it), ``older_server`` (a server
    too old to say whose it is answers on it), ``other_port`` (this data
    folder's server runs on another port than the configured one),
    ``unreachable`` (another server or program answers at the recorded
    server's address), ``running`` (this data folder's server runs, and the
    call does not reuse one), ``busy`` (another process held the start lock
    longer than :func:`start_budget`), ``stopping`` (the recorded server does
    not listen, past :data:`_DRAIN_WAIT`), ``unrecorded_lock`` (a server
    holds the lock without a record), ``lock_unreadable`` (the lock cannot
    be opened), ``unresponsive`` (the recorded server did not answer),
    ``start_failed``, ``exited`` (the server this call started exited, with
    ``exit_code``), ``ccproxy_failed`` (it could not start the ccproxy
    Anthropic's OAuth goes through, so it did not serve; ``error`` has its
    reason, ``exit_code`` is set when it exited),
    ``cancelled`` or ``unregistered`` (the server did not register the
    project).
    """

    available: bool = False
    served: bool = False
    url: str | None = None
    port: int | None = None
    host: str | None = None
    record: ServerRecord | None = None
    drift: bool = False
    shell_drift: str | None = None
    error: str | None = None
    problem: str | None = None
    exit_code: int | None = None
    proc: subprocess.Popen | None = None
    # A server an older release started, without a record here.
    older: bool = False
    # The lock in the way (the start lock or the server lock).
    lock_file: Path | None = None
    # The log of the server the result is about, when it has one.
    log_file: Path | None = None

    @property
    def started(self) -> bool:
        """This call started the server."""
        return self.proc is not None

    @property
    def from_older_release(self) -> bool:
        """An older release started the server: one without a record here,
        or one whose record that release kept."""
        return self.older or (self.record is not None and self.record.older)

    @property
    def older_message(self) -> str | None:
        """The notice to restart a server an older release started: with
        ``EvoSci server stop`` when that release's record lets it stop the
        server, by hand when it left none."""
        if self.record is not None and self.record.older:
            return OLDER_SERVER_MESSAGE
        return older_unrecorded_message() if self.older else None

    @property
    def reason(self) -> str | None:
        """``error`` on one line, for a notice: an exit names the log
        instead of quoting its end."""
        if not self.error:
            return None
        if self.problem == "ccproxy_failed":
            # Its last line says what to do.
            return " ".join(self.error.split())
        line = self.error.splitlines()[0]
        if self.problem == "exited":
            line = f"{line} See {RUNTIME.log_file}."
        return line


def ensure_server(
    config: EvoScientistConfig,
    workspace_dir: Path | str | None = None,
    *,
    backend: str | None = None,
    port: int | None = None,
    host: str | None = None,
    lasting: bool = True,
    tunnel: bool = False,
    job: Callable[[], BackendJob | None] | None = None,
    reuse: bool = True,
    any_port: bool = False,
    before_start: Callable[[], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    on_wait: Callable[[bool], None] | None = None,
) -> ServerResult:
    """Find this data folder's server, or start one, and register
    *workspace_dir* with it.

    What runs decides, read again at every look:

    - A record that is ``serving``: the address it records is asked whose
      server answers (``/ok`` and ``GET /api/instance``), and the session
      uses that address
      (:func:`~EvoScientist.langgraph_dev.sdk.configured_langgraph_dev_url`). This data folder's is reused,
      and never killed; one that misses :data:`_SERVING_PROBES` probes in a
      row is ``unresponsive``; any other answer is ``unreachable``. A record
      on another port than *port* is ``other_port`` (unless *any_port*).
    - A record that is ``starting``: the call waits up to
      :func:`start_budget` for the server to answer.
    - A ``serving`` record whose address refuses: its server has not bound
      yet, or stops and drains; the call waits up to :data:`_DRAIN_WAIT`
      (``stopping``). A record names its server until the process exits.
    - No record while the server lock is held: a server that has only just
      taken the lock writes it at once; after :data:`_SETTLE_WAIT` the lock
      is held without one (``unrecorded_lock``).
    - Neither: a server that answers on the port as this data folder's (one
      started by hand) is reused, any other is a conflict. Otherwise the call
      starts one under the start lock, once the port can be bound, which
      outlives this process for every session to share until ``EvoSci
      server stop`` (unless *lasting* is False), and waits for its record,
      then for its answer at the address it records, or for its exit.

    Then the server registers *workspace_dir*, after the start lock is
    released (a registration restores the project's threads).

    Returns a :class:`ServerResult`; its ``served`` is whether the session
    has the server it needs: one serving *workspace_dir* (or, without one,
    running), or no server when none is needed. A failure to find or start
    one, and a failed registration
    (:class:`~EvoScientist.langgraph_dev.registration.RegistrationError`),
    are logged and leave ``served`` False: the user can still chat with sync
    sub-agents; only async sub-agent calls and EvoMemory background workers
    will fail. Not so when the server could not start the ccproxy Anthropic's
    OAuth goes through: the session cannot reach Anthropic either, and a
    *workspace_dir* call raises. The module's notices (:data:`CONFIG_DRIFT_SINCE_LAUNCH`,
    :data:`SERVER_UNAVAILABLE` and the others) follow the result.

    Raises a
    :class:`~EvoScientist.langgraph_dev.registration.ServerUnusableError`
    when the server cannot be used for *workspace_dir* at all, or could not
    start its ccproxy (above). For a known
    project whose folder is missing it leaves this process's server state as
    it was, so a session that stays where it is keeps its server; a server
    from an older EvoScientist serves no project of this session.

    Args:
        config: Active EvoScientistConfig.
        workspace_dir: The project root to register. If None, nothing is
            registered.
        backend: The calling surface's resolved gateway backend. When ``None``,
            falls back to the global ``config.gateway_backend``. Determines
            whether the server is needed at all.
        port, host: The address to use, by default the configured one.
        lasting: Whether a server this call starts outlives this process.
        tunnel: Start the server with a public tunnel (:func:`_spawn_server`).
        job: Gives the Windows job object a started server joins; called
            after *before_start*, only for a start.
        reuse: False refuses a running server (``running``) instead of
            reusing it.
        any_port: Use this data folder's server on whatever port it records;
            otherwise only on *port*.
        before_start: Called under the start lock right before the start.
        cancelled: Checked during every wait; True ends the call
            (``cancelled``), stopping a server it started.
        on_wait: Called with True when the call waits for another process's
            start or stop, and with False once that wait is over.
    """

    def settle(
        result: ServerResult,
        *,
        needed: bool = True,
        exposure: registration.ServerExposure | None = None,
        failed: bool = False,
    ) -> ServerResult:
        global _ASYNC_SUBAGENTS_AVAILABLE, CONFIG_DRIFT_SINCE_LAUNCH
        global AGENT_SHELL_DRIFT, OLDER_SERVER, SERVER_UNAVAILABLE
        global SERVER_UNAVAILABLE_REASON, SERVER_EXPOSURE, CCPROXY_MISSING
        _ASYNC_SUBAGENTS_AVAILABLE = result.available
        # Not on the way out of an error or a cancel: asking the server to
        # start ccproxy can take its whole start budget.
        CCPROXY_MISSING = (
            _ccproxy_missing(config, result)
            if needed and not failed and not waits.cancelled()
            else None
        )
        if CCPROXY_MISSING is not None:
            logger.warning("%s", CCPROXY_MISSING)
        CONFIG_DRIFT_SINCE_LAUNCH, AGENT_SHELL_DRIFT = result.drift, result.shell_drift
        OLDER_SERVER = result.older_message
        SERVER_UNAVAILABLE = needed and not result.available
        SERVER_UNAVAILABLE_REASON = result.reason if SERVER_UNAVAILABLE else None
        SERVER_EXPOSURE = exposure
        sdk.set_served_url(result.url if result.available else None)
        return replace(result, served=not SERVER_UNAVAILABLE)

    if not needs_langgraph_dev(config, backend=backend):
        return settle(ServerResult(), needed=False)

    configured_port, configured_host = configured_address(config)
    port = configured_port if port is None else port
    host = (host or "").strip() or configured_host
    waits = _Waits(cancelled, on_wait, start_budget(config))
    try:
        with _LOCK:
            result = _find_or_start(
                config,
                port=port,
                host=host,
                lasting=lasting,
                tunnel=tunnel,
                job=job,
                reuse=reuse,
                any_port=any_port,
                before_start=before_start,
                waits=waits,
            )
    except BaseException:
        settle(ServerResult(), failed=True)
        raise
    finally:
        waits.report(False)
    if not result.available and result.problem == "ccproxy_failed":
        # Anthropic is out of reach in this process too: no fallback helps.
        logger.error("%s", result.error)
        if workspace_dir is not None:
            settle(result)
            raise ServerUnusableError(result.reason)
        return settle(result)
    if not result.available:
        logger.warning(
            "No background agent server: async sub-agents will fall back to "
            "in-process delegation, and EvoMemory background workers will not "
            "run. %s",
            result.error,
        )
        if result.problem == "older_server" and workspace_dir is not None:
            settle(result)
            raise registration.OlderServerError(result.error)
        return settle(result)
    if result.older_message is not None:
        # INFO: the CLI prints it after startup and the TUI shows it.
        logger.info(result.older_message)
    if workspace_dir is None:
        return settle(result)

    workspace = paths.Workspace(workspace_dir)
    try:
        exposure = registration.register_folder(workspace, url=result.url)
    except registration.OlderServerError as exc:
        # Unusable for any project, the session's current one included.
        settle(ServerResult(), failed=True)
        if result.older:
            # It holds the lock without a record: server stop cannot stop it.
            raise registration.OlderServerError(older_unrecorded_message()) from exc
        if result.record is None:
            # No record names it, nor does it hold the lock.
            raise registration.OlderServerError(
                "The running server is from an older EvoScientist and left no "
                f"record EvoSci can stop it by; stop the process that listens on "
                f"port {result.port}, then start again."
            ) from exc
        raise
    except ServerUnusableError:
        # The session stays in its project: its state stays as it was.
        raise
    except RegistrationError as exc:
        logger.warning(
            "%s Falling back to sync sub-agent delegation for this session.",
            exc,
        )
        return settle(
            replace(result, available=False, error=str(exc), problem="unregistered")
        )
    except BaseException:
        settle(ServerResult(), failed=True)
        raise
    return settle(result, exposure=exposure)


def _ccproxy_missing(config: EvoScientistConfig, result: ServerResult) -> str | None:
    """The notice for a session whose Anthropic model goes through the
    server's ccproxy (OAuth) when nothing answers on ``ccproxy_port`` and the
    server, asked to, does not start it; ``None`` otherwise, also when the
    server said why it could not start it."""
    from ..ccproxy_manager import is_ccproxy_running, uses_oauth
    from ..config.settings import providers_in_use

    if (
        not uses_oauth(config)
        or "anthropic" not in providers_in_use(config)
        or result.problem == "ccproxy_failed"
        or is_ccproxy_running(config.ccproxy_port)
    ):
        return None
    if not result.available:
        return (
            "Anthropic's OAuth goes through the background agent server's "
            "ccproxy, so Anthropic models fail without the server."
        )
    reason = registration.ensure_ccproxy(url=result.url)
    if reason is None and is_ccproxy_running(config.ccproxy_port):
        return None
    return (
        "The background agent server's ccproxy is not running on port "
        f"{config.ccproxy_port}, so Anthropic models fail (Anthropic's OAuth "
        "goes through it). "
        + (
            reason
            or _sentence(
                server_record.restart_hint(
                    result.record.lasting if result.record is not None else None
                )
            )
        )
    )


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


def serve_project(
    config: EvoScientistConfig,
    workspace_dir: Path | None,
) -> ServerResult:
    """Start the background server, or reuse a running one, and register
    *workspace_dir* with it (``EvoSci server start``); without one (the restart after an update) nothing is
    registered. With Anthropic OAuth this process goes through the server's
    ccproxy too. Returns and raises like :func:`ensure_server`."""
    from ..ccproxy_manager import use_server_ccproxy

    use_server_ccproxy(config)
    return ensure_server(
        config, workspace_dir=workspace_dir, backend="langgraph_server"
    )


class _Waits:
    """The pauses of one :func:`ensure_server` call: each checks for a
    cancel, and a wait for another process is reported to ``on_wait``."""

    def __init__(
        self,
        cancelled: Callable[[], bool] | None,
        on_wait: Callable[[bool], None] | None,
        budget: float,
    ) -> None:
        self._cancelled = cancelled
        self._on_wait = on_wait
        self._waiting = False
        # How long a start may take (:func:`start_budget`).
        self.budget = budget

    def cancelled(self) -> bool:
        return self._cancelled is not None and self._cancelled()

    def report(self, waiting: bool) -> None:
        if waiting != self._waiting and self._on_wait is not None:
            self._on_wait(waiting)
        self._waiting = waiting

    def pause(self, *, for_other: bool = False) -> None:
        self.report(for_other)
        time.sleep(_POLL_INTERVAL)

    def acquire(self, lock: FileLock) -> bool | None:
        """Take *lock* within :attr:`budget`: True once held, False on a
        cancel, None when another process kept it longer."""
        deadline = time.monotonic() + self.budget
        while True:
            if self.cancelled():
                return False
            try:
                lock.acquire(timeout=_POLL_INTERVAL)
            except FileLockTimeout:
                if time.monotonic() >= deadline:
                    return None
                self.report(True)
                continue
            self.report(False)
            return True


def _find_or_start(
    config: EvoScientistConfig,
    *,
    port: int,
    host: str,
    lasting: bool,
    tunnel: bool,
    job: Callable[[], BackendJob | None] | None,
    reuse: bool,
    any_port: bool,
    before_start: Callable[[], None] | None,
    waits: _Waits,
) -> ServerResult:
    """Locked part of :func:`ensure_server` — must hold ``_LOCK``: find the
    server; when there is none, start one."""
    fingerprint = _server_config_fingerprint(config)
    look = _Look(
        asked_url=_base_url(port, host),
        port=port,
        host=host,
        any_port=any_port,
        reuse=reuse,
        fingerprint=fingerprint,
    )
    while True:
        found = _find(look, waits)
        if found != "free":
            return found
        started = _spawn(
            config,
            look,
            lasting=lasting,
            tunnel=tunnel,
            job=job,
            before_start=before_start,
            waits=waits,
        )
        if started is not None:
            return started
        # Another process changed what runs: look again after a pause.
        waits.pause(for_other=True)


@dataclass(frozen=True)
class _Look:
    """What one :func:`ensure_server` call looks for."""

    asked_url: str
    port: int
    host: str
    any_port: bool
    reuse: bool
    fingerprint: str


@dataclass(frozen=True)
class _Seen:
    """One look: the record read after the probe, whether the server lock
    is held, and how the address asked (*url*) answered (:func:`ask`)."""

    record: ServerRecord | None
    lock: str
    url: str
    answer: str


class _Clock:
    """The deadlines and missed probes of one call, keyed by the state and
    the server they are about, so a record that names another server starts
    them afresh."""

    def __init__(self) -> None:
        self._deadlines: dict[tuple, float] = {}
        self._counts: dict[tuple, int] = {}

    def waited_out(self, key: tuple, seconds: float) -> bool:
        now = time.monotonic()
        return now >= self._deadlines.setdefault(key, now + seconds)

    def count(self, key: tuple) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def reset(self, key: tuple) -> None:
        self._counts.pop(key, None)
        self._deadlines.pop(key, None)


def _identity(record: ServerRecord) -> tuple:
    return record.pid, record.created


def _find(look: _Look, waits: _Waits) -> object:
    """Look until the state decides: a :class:`ServerResult`, or ``"free"``
    when no server runs and the port can be started on."""
    clock = _Clock()
    last: ServerRecord | None = None
    while True:
        if waits.cancelled():
            return ServerResult(problem="cancelled", error="Cancelled.")
        seen = _observe(look, last)
        last = seen.record
        if seen.url != _url_to_ask(seen.record, look):
            # The record changed since the probe: ask where it says.
            continue
        decided = _decide(seen, look, clock, waits.budget)
        if decided != "wait":
            return decided
        waits.pause(for_other=True)


def _url_to_ask(record: ServerRecord | None, look: _Look) -> str:
    """A recorded server is asked at its own address, the one sessions then
    use; without a record, the caller's."""
    return record.url if record is not None else look.asked_url


def _observe(look: _Look, last: ServerRecord | None) -> _Seen:
    """Probe first and read the record after, so a record that says a
    server serves was written by the server that answered (it marks itself
    serving before it answers). Where to ask follows the record of the look
    before (*last*)."""
    url = _url_to_ask(last, look)
    answer = ask(url)
    lock = server_home.lock_state()
    record = read_record(older=True, lock=lock)
    return _Seen(record=record, lock=lock, url=url, answer=answer)


def _decide(seen: _Seen, look: _Look, clock: _Clock, budget: float) -> object:
    """What one look means: a :class:`ServerResult`, ``"wait"`` to look
    again, or ``"free"`` when no server runs."""
    record, answer = seen.record, seen.answer
    if record is None:
        if seen.lock == server_home.UNKNOWN:
            # Nothing can be told of the lock: only a server that answers as
            # this data folder's is used, and none is started.
            if answer == "own":
                return _no_record(answer, look)
            return _lock_unreadable()
        if seen.lock == server_home.HELD:
            # A server that has only just taken its lock writes its record
            # at once; one that holds it longer without one keeps none: an
            # older release's, used when it answers as this data folder's.
            if not clock.waited_out(("settle",), _SETTLE_WAIT):
                return "wait"
            if answer != "own":
                return _unrecorded_lock()
            return replace(_no_record(answer, look), older=True)
        return _no_record(answer, look)
    clock.reset(("settle",))
    refusal = _refuse_recorded(
        record, port=look.port, reuse=look.reuse, any_port=look.any_port
    )
    if refusal is not None:
        return refusal
    key = _identity(record)
    if answer != "silent":
        # The probes a server missed count in a row.
        clock.reset(("silent", key))
    if answer in ("own", "other", "older", "foreign"):
        return _answered(answer, record, look.fingerprint)
    if record.phase == server_record.STARTING:
        if clock.waited_out(("starting", key), budget):
            return _unresponsive(record, _NOT_SERVING)
        return "wait"
    if answer == "silent":
        # A live server that does not answer in time.
        missed = clock.count(("silent", key))
        if missed >= _SERVING_PROBES:
            return _unresponsive(record, f"did not answer {missed} probes in a row")
        return "wait"
    # Nothing listens while it holds the lock and its record: not bound yet,
    # or stopping, draining its requests before it exits.
    if clock.waited_out(("drain", key), _DRAIN_WAIT):
        return _stopping(record)
    return "wait"


# How a record that still says ``starting`` failed: its server may run but
# not listen, or answer without having marked itself serving.
_NOT_SERVING = "did not answer as serving"


def _answered(answer: str, record: ServerRecord, fingerprint: str) -> ServerResult:
    """The result of an answer at the recorded server's address."""
    if answer == "own":
        return _reused(record, fingerprint)
    return _conflict(answer, record)


def _conflict(answer: str, record: ServerRecord) -> ServerResult:
    """Another server (``other``, ``older``) or program (``foreign``)
    answers at the recorded server's address."""
    if answer != "older":
        return _unreachable(record, answer)
    # An older release's record is one server stop stops by.
    error = (
        str(registration.OlderServerError())
        if record.older
        else (
            f"A server of an older EvoScientist release answers at "
            f"{record.url}, where this data folder's server is recorded; "
            f"stop the process that listens on port {record.port}, then "
            "start again."
        )
    )
    return ServerResult(
        record=record, port=record.port, problem="older_server", error=error
    )


def _no_record(answer: str | None, look: _Look) -> object:
    """No server recorded: the server that answers on the port, reused when
    it is this data folder's (one started by hand) and a conflict otherwise;
    ``"free"`` when nothing answers, for the start."""
    port, url = look.port, look.asked_url
    if answer == "foreign":
        return _foreign(port)
    if answer == "older":
        return ServerResult(
            problem="older_server",
            port=port,
            error=(
                f"The server on port {port} is from an older EvoScientist "
                "release that left no record EvoSci can stop it by; stop the "
                f"process that listens on port {port}, then start again."
            ),
        )
    if answer == "other":
        return ServerResult(
            problem="other_data_folder",
            port=port,
            error=(
                f"The EvoScientist server on port {port} uses another data "
                f"folder than {paths.DATA_DIR}. Stop it, or give this data "
                "folder another port: EvoSci config set langgraph_dev_port "
                "<port>."
            ),
        )
    if answer != "own":
        return "free"
    host = look.host
    if not look.reuse:
        return ServerResult(
            url=url,
            port=port,
            host=host,
            problem="running",
            error=(
                f"This data folder's background agent server already runs on "
                f"port {port}. Stop it first with EvoSci server stop."
            ),
        )
    logger.info("langgraph dev already running on %s, reusing", url)
    return ServerResult(available=True, url=url, port=port, host=host)


def _refuse_recorded(
    record: ServerRecord, *, port: int, reuse: bool, any_port: bool
) -> ServerResult | None:
    """Why the recorded server is not used without asking it, or None."""
    if not reuse:
        return ServerResult(
            record=record,
            url=record.url,
            port=record.port,
            host=record.host,
            problem="running",
            error=(
                f"This data folder's background agent server already runs "
                f"(pid {record.pid}, port {record.port}). Stop it first with "
                "EvoSci server stop."
            ),
        )
    if not any_port and record.port != port:
        return ServerResult(
            record=record,
            url=record.url,
            port=record.port,
            host=record.host,
            problem="other_port",
            error=(
                f"This data folder's background agent server runs on port "
                f"{record.port}, not on the configured port {port}. Stop it "
                "with EvoSci server stop, or use its port: EvoSci config set "
                f"langgraph_dev_port {record.port}."
            ),
        )
    return None


def _stop_lock_holder_by_hand() -> str:
    """How to find and stop the server that holds the lock without a record."""
    lock = RUNTIME.server_lock
    if os.name == "nt":
        return (
            f"find the process that holds {lock} (Resource Monitor, CPU tab, "
            "Associated Handles) and end it by hand"
        )
    return f"find its process with lsof {lock} (or fuser {lock}) and stop it by hand"


def older_unrecorded_message() -> str:
    """The notice for a reused server an older release started and left no
    record of: ``EvoSci server stop`` cannot stop it."""
    return (
        "The background agent server was started by an older EvoScientist "
        "release that left no record EvoSci can stop it by; restart it to run "
        f"this version: {_stop_lock_holder_by_hand()}; it starts again with "
        "your next session."
    )


def unrecorded_lock_message(then: str = "start again") -> str:
    """What to say of a server lock held without a record, and how to stop
    its holder by hand, *then* what to do."""
    lock = RUNTIME.server_lock
    find = _stop_lock_holder_by_hand()
    return (
        f"A server holds this data folder's lock ({lock}) without a record of "
        "itself, so EvoSci can neither use nor stop it: one an older release "
        f"started, or one that could not write its record. {find[0].upper()}"
        f"{find[1:]}, then {then}."
    )


def lock_unreadable_message() -> str:
    """What to say of a server lock that cannot be opened, and the fix."""
    lock = RUNTIME.server_lock
    if os.name == "nt":
        fix = f"give your account access to {lock} (its file properties)"
    else:
        fix = f"sudo chown $USER {lock}, or restore its permissions"
    return (
        f"This data folder's server lock ({lock}) cannot be opened (another "
        "user's, as after a sudo EvoSci), so EvoSci cannot tell whether a "
        f"server runs and starts none: {fix}."
    )


def lock_problem(lock: str) -> str | None:
    """What to say when there is no record because of the server lock in
    state *lock* (:func:`server_home.lock_state`): held without one, or
    unreadable; None when it is free."""
    if lock == server_home.HELD:
        return unrecorded_lock_message()
    if lock == server_home.UNKNOWN:
        return lock_unreadable_message()
    return None


def _lock_unreadable() -> ServerResult:
    return ServerResult(
        problem="lock_unreadable",
        error=lock_unreadable_message(),
        lock_file=RUNTIME.server_lock,
    )


def _unrecorded_lock() -> ServerResult:
    return ServerResult(
        problem="unrecorded_lock",
        error=unrecorded_lock_message(),
        lock_file=RUNTIME.server_lock,
    )


def _stopping(record: ServerRecord) -> ServerResult:
    """The recorded server refused at its own address past :data:`_DRAIN_WAIT`."""
    return ServerResult(
        record=record,
        problem="stopping",
        error=(
            f"This data folder's background agent server (pid {record.pid}) "
            f"is stopping, or does not answer at {record.url}. Try again in a "
            "moment, or run EvoSci server stop if it persists; see "
            f"{log_file_of(record)}."
        ),
        log_file=log_file_of(record),
    )


def _unreachable(record: ServerRecord, answer: str) -> ServerResult:
    """Another server or program answers at the recorded server's address."""
    who = "another EvoScientist server" if answer == "other" else "another program"
    return ServerResult(
        record=record,
        problem="unreachable",
        error=(
            f"This data folder's background agent server (pid {record.pid}) is "
            f"recorded at {record.url}, but {who} answers there. Stop it with "
            "EvoSci server stop."
        ),
        log_file=log_file_of(record),
    )


def _unresponsive(record: ServerRecord, what: str) -> ServerResult:
    return ServerResult(
        record=record,
        problem="unresponsive",
        error=(
            f"This data folder's background agent server (pid {record.pid}) at "
            f"{record.url} {what}; see {log_file_of(record)}. EvoSci server stop "
            "stops it."
        ),
        log_file=log_file_of(record),
    )


def _reused(record: ServerRecord, fingerprint: str) -> ServerResult:
    """The recorded server answered as this data folder's."""
    url, port, host = record.url, record.port, record.host
    recorded_fp = record.fields.get("config_fingerprint")
    drift = (
        not record.older and isinstance(recorded_fp, str) and recorded_fp != fingerprint
    )
    if drift:
        logger.warning(
            "Configuration or version changed since the running "
            "langgraph dev was launched — it still serves the old "
            "settings or version until it is restarted "
            "(EvoSci server stop)."
        )
    from EvoScientist.agent_shell import shell_drift_message

    shell_drift = shell_drift_message(record.fields)
    if shell_drift is not None:
        # INFO: the CLI prints it after startup and the TUI shows
        # it in the app; a WARNING here would print it twice.
        logger.info(shell_drift)
    logger.info("langgraph dev already running on %s, reusing", url)
    return ServerResult(
        available=True,
        url=url,
        port=port,
        host=host,
        record=record,
        drift=drift,
        shell_drift=shell_drift,
    )


def ask(url: str) -> str:
    """Whose server answers at *url*: ``own`` (this data folder's local
    secret), ``other``, ``older`` (too old to say), ``foreign``, ``silent``
    or ``refused``.

    ``foreign``: ``/ok`` answered with a status that is neither 200 nor a
    5xx. ``other``: ``GET /api/instance`` answered with an instance that is
    not this data folder's, or with none. ``older``: a 404 on
    ``/api/instance``. ``refused``: either request could not connect.
    Everything else is ``silent``, asked again: no answer within
    :data:`_PROBE_TIMEOUT`, a connection lost or broken after it was taken
    (a listener that does not speak HTTP among them), a 5xx on ``/ok``, any
    other error status on ``/api/instance``. A malformed *url* raises."""
    try:
        response = httpx.get(f"{url}/ok", timeout=_PROBE_TIMEOUT, trust_env=False)
    except (httpx.HTTPError, OSError) as exc:
        return "refused" if isinstance(exc, httpx.ConnectError) else "silent"
    if response.status_code >= 500:
        return "silent"
    if response.status_code != 200:
        return "foreign"
    try:
        return _whose_server(url)
    except RegistrationError as exc:
        return "refused" if isinstance(exc.__cause__, httpx.ConnectError) else "silent"


def _foreign(port: int) -> ServerResult:
    return ServerResult(
        problem="foreign",
        port=port,
        error=f"Port {port} is occupied by another process.",
    )


def _whose_server(url: str) -> str:
    """``own`` when the server at *url* holds this data folder's local
    secret, ``older`` for one too old to say, ``other`` otherwise.
    ``RegistrationError`` when it cannot be asked."""
    from EvoScientist.local_secret import instance_of, read_local_secret

    try:
        instance = registration.server_instance(url, timeout=_PROBE_TIMEOUT)
    except OlderServerError:
        return "older"
    secret = read_local_secret()
    if secret is not None and instance == instance_of(secret):
        return "own"
    return "other"


def _install_node_for_mcp(waits: _Waits) -> None:
    """Install Node for npx MCP servers before a start, where progress is
    visible; the server itself never downloads it. A cancel stops waiting
    for it (the download finishes in the background)."""

    def install() -> None:
        try:
            from ..mcp.client import _ensure_node_for_stdio, load_mcp_config

            _ensure_node_for_stdio(load_mcp_config())
        except Exception:
            logger.warning("Could not check Node.js for MCP servers", exc_info=True)

    worker = threading.Thread(target=install, name="node-install", daemon=True)
    worker.start()
    while worker.is_alive() and not waits.cancelled():
        worker.join(_POLL_INTERVAL)


def _spawn(
    config: EvoScientistConfig,
    look: _Look,
    *,
    lasting: bool,
    tunnel: bool,
    job: Callable[[], BackendJob | None] | None,
    before_start: Callable[[], None] | None,
    waits: _Waits,
) -> ServerResult | None:
    """Start a server under the start lock, once the port can be bound, and
    find it at its own address. Returns what was found (``started``, with
    the process, when available), a failure, or None when another process
    changed what runs while this one waited for the lock. The started
    process is stopped on any outcome but an available server, an exception
    included."""
    port, host = look.port, look.host
    lock = FileLock(str(RUNTIME.lock_file))
    try:
        server_home.ensure_server_home()
    except OSError as exc:
        return ServerResult(
            problem="start_failed",
            error=f"Cannot create the server home {paths.SERVER_HOME}: {exc}",
        )
    try:
        held = waits.acquire(lock)
    except OSError as exc:
        return ServerResult(
            problem="start_failed",
            error=f"Cannot use the start lock {RUNTIME.lock_file}: {exc}",
            lock_file=RUNTIME.lock_file,
        )
    if held is False:
        return ServerResult(problem="cancelled", error="Cancelled.")
    if held is None:
        return ServerResult(
            problem="busy",
            error=(
                f"Another EvoScientist process has been starting a server for "
                f"over {waits.budget:.0f} s; it holds {RUNTIME.lock_file}."
            ),
            lock_file=RUNTIME.lock_file,
        )
    try:
        lock_now = server_home.lock_state()
        if read_record(older=True, lock=lock_now) is not None:
            return None
        if lock_now == server_home.UNKNOWN:
            return _lock_unreadable()
        if lock_now == server_home.HELD:
            # Taken since the look: a new server could not take it.
            return _unrecorded_lock()
        if not _wait_for_port_bindable(port, host):
            return _foreign(port)
        _install_node_for_mcp(waits)
        if not waits.cancelled() and before_start is not None:
            before_start()
        if waits.cancelled():
            return ServerResult(problem="cancelled", error="Cancelled.")
        earlier = _PROCESS
        spawned_at = time.monotonic()
        try:
            proc = _spawn_server(
                port=port,
                host=host,
                file_persistence=bool(
                    getattr(config, "langgraph_dev_file_persistence", True)
                ),
                jobs_per_worker=int(
                    getattr(config, "langgraph_dev_jobs_per_worker", 10)
                ),
                tunnel=tunnel,
                config_fingerprint=look.fingerprint,
                allowed_origins=getattr(config, "backend_allowed_origins", ""),
                job=job() if job is not None else None,
                lasting=lasting,
            )
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            # A failure after the spawn (joining the job) leaves the handle.
            spawned = _PROCESS if _PROCESS is not earlier else None
            return _give_up(spawned, problem="start_failed", error=str(exc))
        try:
            record = _record_of(proc, spawned_at, waits)
        except BaseException:
            stop_langgraph_dev(proc, clean=False)
            raise
    finally:
        lock.release()
    if isinstance(record, ServerResult):
        return record
    try:
        return _answer_of(proc, record, spawned_at, waits)
    except BaseException:
        stop_langgraph_dev(proc, clean=False)
        raise


def _answer_of(
    proc: subprocess.Popen, record: ServerRecord, spawned_at: float, waits: _Waits
) -> ServerResult:
    """Wait for the server this call started to answer at its recorded
    address, within :func:`start_budget` of its start: started once it
    answers as this data folder's. Its exit, another server or program
    answering there, or the budget spent ends the wait, and the process is
    stopped. Only the process and the answer count; the record is read again
    once it answers, for what the server added as it marked itself serving,
    and kept as first read when that read fails."""
    url = record.url
    while True:
        if proc.poll() is not None:
            return _exited(proc, server_pid=record.pid)
        if waits.cancelled():
            return _give_up(proc, problem="cancelled", error="Cancelled.")
        answer = ask(url)
        if answer == "own":
            serving = read_record()
            if record.same_server(serving):
                record = serving
            logger.info("langgraph dev started on %s (pid=%d)", url, proc.pid)
            return ServerResult(
                available=True,
                url=url,
                port=record.port,
                host=record.host,
                record=record,
                proc=proc,
            )
        if answer in ("other", "older", "foreign"):
            stop_langgraph_dev(proc, clean=False)
            return _conflict(answer, record)
        if time.monotonic() >= spawned_at + waits.budget:
            return _not_ready(proc, spawned_at, server_pid=record.pid)
        waits.pause()


def _record_of(
    proc: subprocess.Popen, spawned_at: float, waits: _Waits
) -> ServerRecord | ServerResult:
    """The started server's record: one whose PID is the process started or
    one of its children (on Windows the server runs below a launcher), never
    a record another server left. A failure stops the process."""
    deadline = spawned_at + waits.budget
    while True:
        record = read_record()
        if record is not None and record.pid in _spawned_pids(proc):
            return record
        if proc.poll() is not None:
            return _exited(proc, server_pid=proc.pid)
        if waits.cancelled():
            return _give_up(proc, problem="cancelled", error="Cancelled.")
        if time.monotonic() >= deadline:
            return _not_ready(proc, spawned_at)
        waits.pause()


def _spawned_pids(proc: subprocess.Popen) -> set[int]:
    """The process started and its descendants."""
    pids = {proc.pid}
    try:
        pids |= {child.pid for child in psutil.Process(proc.pid).children(True)}
    except psutil.Error:
        pass
    return pids


def _give_up(
    proc: subprocess.Popen | None, *, problem: str, error: str
) -> ServerResult:
    """A failure; a server this call started is stopped first, and its log
    named."""
    if proc is None:
        return ServerResult(problem=problem, error=error)
    # Still starting: no run to put back in the queue.
    stop_langgraph_dev(proc, clean=False)
    return ServerResult(problem=problem, error=error, log_file=RUNTIME.log_file)


def _not_ready(
    proc: subprocess.Popen, spawned_at: float, *, server_pid: int | None = None
) -> ServerResult:
    if (reason := ccproxy_failure_of(server_pid)) is not None:
        stop_langgraph_dev(proc, clean=False)
        return _ccproxy_failed(reason, exit_code=None)
    elapsed = time.monotonic() - spawned_at
    return _give_up(
        proc,
        problem="start_failed",
        error=(
            f"langgraph dev did not become healthy within {elapsed:.0f} "
            f"seconds. Check {RUNTIME.log_file}"
        ),
    )


def ccproxy_failure_of(server_pid: int | None) -> str | None:
    """Why the server *server_pid* could not start its ccproxy, when it left
    the reason; a reason another server left is not its."""
    if server_pid is None:
        return None
    try:
        left = json.loads(RUNTIME.ccproxy_failure.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(left, dict) or left.get("pid") != server_pid:
        return None
    reason = left.get("reason")
    return reason.strip() if isinstance(reason, str) and reason.strip() else None


def _ccproxy_failed(reason: str, *, exit_code: int | None) -> ServerResult:
    return ServerResult(
        problem="ccproxy_failed",
        exit_code=exit_code,
        error=(
            "The server could not start ccproxy, which Anthropic's OAuth goes "
            f"through: {reason}"
        ),
        log_file=RUNTIME.log_file,
    )


def _exited(proc: subprocess.Popen, *, server_pid: int | None = None) -> ServerResult:
    """The server this call started exited: its exit code and the end of
    what it logged, or why it could not start its ccproxy (the server
    *server_pid*, once it recorded itself)."""
    stop_langgraph_dev(proc)
    if (reason := ccproxy_failure_of(server_pid)) is not None:
        return _ccproxy_failed(reason, exit_code=proc.returncode)
    return ServerResult(
        problem="exited",
        exit_code=proc.returncode,
        error=(
            (
                "langgraph dev was stopped (SIGTERM) before it answered, as by "
                "EvoSci server stop from another terminal.\n"
                if proc.returncode == -signal.SIGTERM
                else f"langgraph dev exited immediately with code {proc.returncode}.\n"
            )
            + f"Log tail:\n{_log_tail()}"
        ),
        log_file=RUNTIME.log_file,
    )


def _log_tail() -> str:
    """The end of what the server this process started last logged."""
    try:
        with open(RUNTIME.log_file, "rb") as fh:
            fh.seek(_LOG_OFFSET_AT_START)
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return text[-_LOG_TAIL_CHARS:]
