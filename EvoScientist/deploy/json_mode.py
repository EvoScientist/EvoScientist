"""``EvoSci deploy --json``: start or attach to the backend for a program.

stdout carries one JSON object per line and nothing else (Rich output goes to
stderr). Every line has ``"protocol": 1``, the number ``EvoSci setup --json``
uses, and an ``event``. Later versions may add fields and events, never rename
or remove them.

Events:

- ``starting``: ``workspace``, ``host``, ``port``.
- ``ready``: ``url``, ``owned``, ``workspace``, ``assistant_id``, ``data_dir``,
  ``config_dir``, ``log_file``, ``warnings`` (a list of strings), and with
  ``--tunnel`` also ``public_url`` (null when no URL was reported in time).
  ``owned`` is true when this command started the backend and stops it on
  exit, false when it attached to a running one for the same workspace and
  leaves it running. When attached, ``workspace``, ``data_dir`` and
  ``config_dir`` are the running backend's, from its ownership record (or
  this command's own when an older release wrote the record without them).
  ``data_dir`` and ``config_dir`` are paths to open files at, never to
  compare as strings: one folder can have more than one spelling (an 8.3
  short name on Windows, a symlink).
- ``error``: ``code``, ``message``, ``detail`` (an object, possibly empty).
  ``detail`` by code (see :data:`ERROR_CODES`):

  - ``invalid_port``: ``port``.
  - ``port_conflict``: ``reason``, always: ``foreign`` (another program holds
    the port), ``no_record`` (a langgraph dev without an ownership record, so
    its workspace is unknown) or ``record_mismatch`` (the ownership record
    names a process that does not serve this port).
  - ``workspace_conflict``: ``workspace``, the running backend's, and
    ``run_dir`` when it serves a run folder of this workspace.
  - ``ccproxy_failed``: empty.
  - ``backend_start_failed``: ``log_file`` when the start failed, or
    ``lock_file`` when another EvoScientist process held the start lock too
    long.
  - ``sign_in_required``: ``provider``, and ``problem`` when there is one.
  - ``backend_exited``: ``exit_code`` and ``log_file`` when the owned process
    ended; ``url`` when an owned backend stopped listening or an attached one
    stopped answering.
  - ``internal_error``: ``type``, the exception's class; or ``launcher_code``
    for a backend-decision error this mode has no code for.

  ``port_conflict`` and ``workspace_conflict`` also carry ``hint``, the line
  the human mode prints.
- ``stopped``: a clean exit. An owned backend has been stopped; an attached
  one is left running.

``backend_exited`` for an owned backend means its process ended, or nothing
listens on its port any more. An owned backend that keeps its port but stops
answering is left running: the app decides when to stop it. An attached
backend is reported as exited when ``/ok`` fails three times in a row, and is
never stopped.

Every run that reaches the command ends with exactly one ``stopped`` or
``error`` line, then exits (0 after ``stopped``, 1 after ``error``). Option
errors reported before the command runs print no event, and a killed process
prints nothing. When the reader closes stdout, the command stops as on a stop
request and writes nothing more.

How a program runs the command:

- Start ``EvoSci`` itself, not through a shell (in Node, no
  ``shell: true``): Node destroys a child's stdin when that child exits, so
  an intermediate shell would end the stdin of a command that still runs.
- Give it a stdin pipe. To stop, close stdin or write a
  ``{"command": "stop"}`` line, then wait for ``stopped`` or the exit before
  quitting. A terminal, ``/dev/null``, ``NUL`` or a file as stdin is ignored;
  the command then stops on Ctrl+C or SIGTERM.
- When the program exits without closing stdin, the command still stops: on
  macOS and Linux its stdin reaches end-of-file; on Windows an owned backend
  runs in a job object that ends it together with this command.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any

from ..setup.protocol import PROTOCOL, JsonEmitter, stdin_is_pipe

logger = logging.getLogger(__name__)

ASSISTANT_ID = "EvoScientist"

# The ``detail`` of each is in the module docstring. ``internal_error``:
# anything else, with the traceback on stderr.
ERROR_CODES = frozenset(
    {
        "invalid_port",
        "port_conflict",
        "workspace_conflict",
        "ccproxy_failed",
        "backend_start_failed",
        "sign_in_required",
        "backend_exited",
        "internal_error",
    }
)

_LAUNCHER_CODES = {
    "port_conflict": "port_conflict",
    "sidecar_port_mismatch": "port_conflict",
    "workspace_mismatch": "workspace_conflict",
}

# The health check runs this often; this many failures in a row mean the
# backend is gone (one slow answer is not).
HEALTH_INTERVAL = 2.0
HEALTH_FAILURES = 3
# How often the wait loop looks at the stop request and the owned process.
_TICK = 0.5
# A stdin line longer than this is not a command and is dropped.
_MAX_LINE = 65536
_STDIN_THREAD = "deploy-json-stdin"
# How long ``--tunnel`` waits for cloudflared's URL, as ``read_tunnel_url``.
_TUNNEL_WAIT = 35.0


class _Failure(Exception):
    """Ends the run with an ``error`` event."""

    def __init__(
        self, code: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown deploy error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail or {})

    def event(self) -> dict[str, Any]:
        return {
            "event": "error",
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


class _Events:
    """Writes the event lines. Once stdout is gone (the reader closed it),
    asks the run to stop and drops later lines."""

    def __init__(self, on_broken: threading.Event) -> None:
        self._emit = JsonEmitter()
        self._on_broken = on_broken
        self._broken = False

    def __call__(self, event: str, **fields: Any) -> None:
        if self._broken:
            return
        try:
            self._emit({"protocol": PROTOCOL, "event": event, **fields})
        except (OSError, ValueError):
            self._broken = True
            self._on_broken.set()
            # The failed line stays in sys.stdout's buffer; the flush at
            # interpreter shutdown would fail again and exit 120. Later
            # writes go nowhere instead.
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
                os.close(devnull)
            except (OSError, ValueError, AttributeError):
                pass


def _take_stdin() -> int:
    """Move the stdin pipe to a private descriptor and give stdin the null
    device; returns the private descriptor.

    On Windows a read pending on a pipe stalls every other process that holds
    the same pipe: a child that inherits it (the backend, ccproxy, and the MCP
    servers and shell commands the backend starts) hangs in its start-up. The
    copy is not inheritable (PEP 446), so children inherit the null device
    instead. The end of input still arrives: it depends on the writer only.
    """
    stdin_fd = sys.stdin.fileno()
    private = os.dup(stdin_fd)
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, stdin_fd)
    finally:
        os.close(null)
    if sys.platform == "win32" and stdin_fd == 0:
        # What ``Popen`` hands a child as its stdin.
        import ctypes
        import msvcrt

        ctypes.windll.kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(0))
    return private


def _watch_stdin(fd: int, stop: threading.Event) -> None:
    """Set ``stop`` at the end of ``fd`` or on a ``{"command": "stop"}`` line.

    Reads the descriptor itself: a thread blocked in ``sys.stdin`` holds its
    buffer lock, which interpreter shutdown may need.
    """
    pending = b""
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            pending += chunk
            *lines, pending = pending.split(b"\n")
            if any(_is_stop(line) for line in lines):
                break
            if len(pending) > _MAX_LINE:
                pending = b""  # not a command line; drop it
    except OSError as exc:
        logger.debug(f"stdin read failed: {exc}")
    finally:
        os.close(fd)
    stop.set()


def _is_stop(line: bytes) -> bool:
    try:
        message = json.loads(line)
    except ValueError:
        return False
    return isinstance(message, dict) and message.get("command") == "stop"


class _Run:
    def __init__(self, events: _Events, stop: threading.Event) -> None:
        self.events = events
        self.stop = stop
        self.host = ""
        self.port = 0
        self.starting = False  # an owned backend may exist from here on
        self.proc: Any = None
        self.ccproxy: Any = None
        self.job: Any = None

    def start(
        self,
        *,
        workdir: str | None,
        port: int | None,
        host: str | None,
        tunnel: bool,
        debug: bool,
    ) -> None:
        from .. import paths
        from ..agent_shell import server_setup_hint, setup_hint
        from ..langgraph_dev import manager as lgm
        from . import server
        from .launcher import BackendTarget

        config = server._load_config(debug, host, port)
        _check_sign_in(config)

        ws = str(paths.start_workspace_path(workdir, config.default_workdir))
        os.makedirs(ws, exist_ok=True)
        self.port = server._resolve_port(port, config)
        if not server._port_in_range(self.port):
            raise _Failure(
                "invalid_port",
                server._invalid_port_message(self.port),
                {"port": self.port},
            )
        self.host = server._resolve_host(host, config)
        self.events("starting", workspace=ws, host=self.host, port=self.port)

        target = BackendTarget(
            workspace_dir=ws, backend_host=self.host, backend_port=self.port
        )
        decision = self._decide_and_start(target, config, ws, tunnel)
        # A stop asked for while waiting for the lock, during the decision or
        # during the start: stop now, without the tunnel wait or ready.
        if decision is None or self.stop.is_set():
            return
        warnings = list(decision.warnings)
        public_url: str | None = None
        if decision.action == "reuse":
            record = decision.sidecar or {}
            # The running backend's bind, from its record; a record written
            # before it carried one gives no warning, since --host here says
            # nothing about that backend.
            bound = record.get("host")
            if isinstance(bound, str) and not lgm._is_loopback_host(bound):
                warnings.append(_public_bind_warning(bound))
            owned = False
            workspace = record.get("workspace", ws)
            data_dir = record.get("data_dir") or str(paths.DATA_DIR)
            config_dir = record.get("config_dir") or str(paths.CONFIG_DIR)
            hint = server_setup_hint(record)
            if tunnel:
                warnings.append(
                    "--tunnel applies only to a backend this command starts; "
                    "the running backend is used as it is."
                )
        else:
            owned = True
            workspace = ws
            # The human mode's banners, for the app to show.
            if not lgm._is_loopback_host(self.host):
                warnings.append(_public_bind_warning(self.host))
            if getattr(config, "dangerous_mode", False):
                from ..cli._constants import (
                    DANGEROUS_BANNER_LABEL,
                    DANGEROUS_BANNER_MESSAGE,
                )

                warnings.append(f"{DANGEROUS_BANNER_LABEL}: {DANGEROUS_BANNER_MESSAGE}")
            if tunnel:
                warnings.append(
                    "Public tunnel: no auth, share the URL only with people you trust."
                )
            data_dir = str(paths.DATA_DIR)
            config_dir = str(paths.CONFIG_DIR)
            if self.job is not None and self.job.warning:
                warnings.append(self.job.warning)
            hint = setup_hint()
            if tunnel:
                public_url = self._wait_for_tunnel_url()
                if self.stop.is_set():
                    return
        if hint is not None:
            warnings.append(hint)

        fields: dict[str, Any] = {
            "url": lgm._base_url(self.port, self.host),
            "owned": owned,
            "workspace": workspace,
            "assistant_id": ASSISTANT_ID,
            "data_dir": data_dir,
            "config_dir": config_dir,
            "log_file": str(lgm.RUNTIME.log_file),
            "warnings": warnings,
        }
        if tunnel:
            fields["public_url"] = public_url
        self.events("ready", **fields)

    def _decide_and_start(self, target: Any, config: Any, ws: str, tunnel: bool) -> Any:
        """The backend decision, and the start it calls for, under the lock
        the TUI's ``ensure_langgraph_dev()`` takes.

        Without it a second starter can see a backend that is bound but not
        answering yet and kill it as stale, or overwrite its ownership
        record. With it, a second ``deploy --json`` waits for the first
        backend to answer, then attaches. None when a stop came first.
        """
        from filelock import FileLock

        from ..langgraph_dev import manager as lgm
        from .launcher import LauncherError, _resolve_backend

        lgm.RUNTIME.pid_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lgm.RUNTIME.lock_file))
        if not self._acquire(lock, lgm._FILE_LOCK_TIMEOUT):
            return None
        try:
            try:
                decision = _resolve_backend(target, config, require_record=True)
            except LauncherError as exc:
                detail = dict(exc.fields)
                if exc.detail:
                    detail["hint"] = exc.detail
                code = _LAUNCHER_CODES.get(exc.code)
                if code is None:
                    detail["launcher_code"] = exc.code
                    code = "internal_error"
                raise _Failure(code, exc.message, detail) from exc
            if decision.action == "start" and not self.stop.is_set():
                self._start_backend(config, ws, tunnel)
            return decision
        finally:
            lock.release()

    def _acquire(self, lock: Any, limit: float) -> bool:
        """Take ``lock`` in short waits that check for a stop; False on a
        stop. Waiting longer than ``limit`` is ``backend_start_failed``: the
        holder is stuck in its own start."""
        from filelock import Timeout

        deadline = time.monotonic() + limit
        while not self.stop.is_set():
            try:
                lock.acquire(timeout=_TICK)
                return True
            except Timeout:
                if time.monotonic() >= deadline:
                    raise _Failure(
                        "backend_start_failed",
                        f"Another EvoScientist process has been starting a "
                        f"backend for over {limit:.0f} s.",
                        {"lock_file": str(lock.lock_file)},
                    ) from None
        return False

    def _start_backend(self, config: Any, ws: str, tunnel: bool) -> None:
        from pathlib import Path

        from ..langgraph_dev import manager as lgm
        from .backend_job import backend_job

        # First, so ccproxy joins it too: Node's job lets children break away,
        # so a ccproxy outside ours would outlive a killed app.
        self.job = backend_job()
        if config.anthropic_auth_mode == "oauth":
            from ..ccproxy_manager import maybe_start_ccproxy

            try:
                self.ccproxy = maybe_start_ccproxy(config)
            except RuntimeError as exc:
                raise _Failure(
                    "ccproxy_failed", f"ccproxy startup failed: {exc}"
                ) from exc
            # None: another session's ccproxy, reused and left alone.
            if self.ccproxy is not None and self.job is not None:
                self.job.adopt(self.ccproxy)
            # A stop during the ccproxy start: cleanup() stops only ccproxy.
            if self.stop.is_set():
                return

        self.starting = True
        try:
            self.proc = lgm.start_langgraph_dev(
                workspace_dir=Path(ws),
                port=self.port,
                host=self.host,
                file_persistence=bool(
                    getattr(config, "langgraph_dev_file_persistence", True)
                ),
                jobs_per_worker=int(
                    getattr(config, "langgraph_dev_jobs_per_worker", 10)
                ),
                tunnel=tunnel,
                config_fingerprint=lgm._server_config_fingerprint(config),
                allowed_origins=getattr(config, "backend_allowed_origins", ""),
                job=self.job,
            )
        except Exception as exc:
            raise _Failure(
                "backend_start_failed",
                f"langgraph dev startup failed: {exc}",
                {"log_file": str(lgm.RUNTIME.log_file)},
            ) from exc

    def _wait_for_tunnel_url(self) -> str | None:
        """The tunnel URL, in short waits that check for a stop between them;
        each read scans the log from this server's start."""
        from ..langgraph_dev import manager as lgm

        deadline = time.monotonic() + _TUNNEL_WAIT
        while not self.stop.is_set() and time.monotonic() < deadline:
            url = lgm.read_tunnel_url(timeout=_TICK)
            if url:
                return url
        return None

    def wait(self) -> None:
        """Block until a stop is asked for; raise ``backend_exited`` when the
        backend goes away first.

        An owned backend is gone when its process ends, or when nothing
        listens on its port any more (a server that stopped serving but did
        not exit). A slow ``/ok`` never counts for it: a stalled backend
        (swapping, a paused process, a blocked event loop) keeps its port,
        and killing it would lose the runs in progress; the app decides.
        An attached backend has no process handle here, so ``/ok`` is the
        check, and nothing is stopped.
        """
        from ..langgraph_dev import manager as lgm

        owned = self.proc is not None
        failures = 0
        next_check = time.monotonic() + HEALTH_INTERVAL
        while not self.stop.wait(_TICK):
            if self.proc is not None and self.proc.poll() is not None:
                raise _Failure(
                    "backend_exited",
                    f"The backend exited with code {self.proc.returncode}.",
                    {
                        "exit_code": self.proc.returncode,
                        "log_file": str(lgm.RUNTIME.log_file),
                    },
                )
            if time.monotonic() < next_check:
                continue
            next_check = time.monotonic() + HEALTH_INTERVAL
            if owned:
                healthy = lgm._is_port_occupied(self.port, self.host)
            else:
                healthy = lgm.is_langgraph_dev_running(port=self.port, host=self.host)
            if healthy:
                failures = 0
                continue
            failures += 1
            if failures >= HEALTH_FAILURES:
                what = "stopped listening" if owned else "stopped answering"
                raise _Failure(
                    "backend_exited",
                    f"The backend on port {self.port} {what}.",
                    {"url": lgm._base_url(self.port, self.host)},
                )

    def cleanup(self) -> None:
        """Stop what this command started: the backend, then ccproxy, then
        close the job (which would end a backend the stop missed). Each step
        runs even when the one before raised; ``run_json`` reports the error."""
        from ..langgraph_dev import manager as lgm

        try:
            if self.starting:
                # An interrupted or failed start returns no handle; the
                # manager records it right after spawning.
                proc = self.proc if self.proc is not None else lgm._PROCESS
                if proc is not None:
                    lgm.stop_langgraph_dev(proc)
        finally:
            try:
                if self.ccproxy is not None:
                    from ..ccproxy_manager import stop_ccproxy

                    stop_ccproxy(self.ccproxy)
            finally:
                if self.job is not None:
                    self.job.close()


def _public_bind_warning(host: str) -> str:
    return (
        f"Listening on {host}: no auth, and the agent can run shell. "
        "Trusted networks only."
    )


def _check_sign_in(config: Any) -> None:
    """``sign_in_required`` when OpenAI uses ChatGPT sign-in and nobody is
    signed in, as ``EvoSci deploy`` checks before anything starts."""
    if getattr(config, "openai_auth_mode", "api_key") != "oauth":
        return
    from ..llm import chatgpt_oauth

    state = chatgpt_oauth.status()
    if state.signed_in:
        return
    detail: dict[str, Any] = {"provider": "openai"}
    if state.problem:
        detail["problem"] = state.problem
    raise _Failure("sign_in_required", chatgpt_oauth.SIGN_IN_HINT, detail)


def _internal_error(exc: BaseException) -> dict[str, Any]:
    traceback.print_exception(exc, file=sys.stderr)
    return _Failure(
        "internal_error",
        str(exc) or type(exc).__name__,
        {"type": type(exc).__name__},
    ).event()


def run_json(
    *,
    workdir: str | None,
    port: int | None,
    host: str | None,
    tunnel: bool,
    debug: bool,
) -> int:
    """Run ``EvoSci deploy --json``; returns the exit code."""
    from ..stream.json_sink import redirect_console_to_stderr

    redirect_console_to_stderr()
    stop = threading.Event()
    events = _Events(stop)
    run = _Run(events, stop)

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        stop.set()

    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    # From the start, so an app that goes away while the backend starts
    # stops it as soon as the start returns.
    if stdin_is_pipe():
        threading.Thread(
            target=_watch_stdin,
            args=(_take_stdin(), stop),
            name=_STDIN_THREAD,
            daemon=True,
        ).start()
    terminal: dict[str, Any] = {"event": "stopped"}
    try:
        try:
            run.start(workdir=workdir, port=port, host=host, tunnel=tunnel, debug=debug)
            run.wait()
        except _Failure as failure:
            terminal = failure.event()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            terminal = _internal_error(exc)
        # A second Ctrl+C must not cut the stop short and leave the backend.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            run.cleanup()
        except Exception as exc:
            if terminal["event"] == "stopped":
                terminal = _internal_error(exc)
            else:
                traceback.print_exception(exc, file=sys.stderr)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    events(**terminal)
    return 0 if terminal["event"] == "stopped" else 1
