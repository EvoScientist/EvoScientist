"""Shell-agnostic WebUI launcher core.

Extracts the reusable start / health / stop logic out of
``deploy/webui.py:run_webui`` so it can be driven by *any* front-end:

- the existing terminal CLI (``run_webui`` is now a thin adapter over this), and
- an embedded desktop shell (imports :class:`WebUILauncher` in-process).

Design rules that keep it shell-agnostic:

- **No terminal coupling.** Never ``console.print`` / ``typer.Exit``. Fatal
  conditions raise :class:`LauncherError` carrying a machine-readable ``code``;
  non-fatal notes land in :attr:`LaunchResult.warnings`. The caller decides how
  to render them (Rich panel, GUI dialog, JSON).
- **The caller owns the main loop.** The core installs no signal handlers and
  registers no ``atexit`` hooks — it exposes :meth:`WebUILauncher.start`,
  :meth:`WebUILauncher.wait_ready`, :meth:`WebUILauncher.poll` and
  :meth:`WebUILauncher.stop`; the caller wires those into its own lifecycle.
- **The front-end is pluggable.** Backend (langgraph dev) handling is shared;
  only the front-end differs, behind :class:`WebUIRunner`. Today's npm-fetched
  front-end is :class:`NpxWebUIRunner`; the bundled Windows-desktop front-end
  (``node dist/server.js``) is :class:`BundledWebUIRunner`.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
import webbrowser
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Front-end npm package + spec. ``@latest`` → always the newest published UI.
_WEBUI_PACKAGE = "@evoscientist/webui@latest"
_DEFAULT_WEBUI_PORT = 4716
_DEFAULT_WEBUI_HOST = "127.0.0.1"


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LauncherConfig:
    """Resolved knobs for one launch. Everything is already resolved (no
    fallback to config/env happens inside the launcher core)."""

    workspace_dir: str
    backend_host: str
    backend_port: int
    webui_host: str
    webui_port: int
    deploy_mode: bool = True
    keepalive: bool = False
    # Honoured only by front-ends whose ``handles_browser_open`` is False
    # (e.g. the bundled runner). The npx runner opens the browser itself.
    open_browser: bool = False
    # Desktop shells set this: when the configured port is occupied by a
    # foreign process, fall back to the next free port instead of raising.
    # The CLI leaves it False so a terminal user gets the explicit conflict.
    auto_port: bool = False


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of a successful :meth:`WebUILauncher.start` / ``wait_ready``."""

    backend_url: str
    webui_url: str
    backend_started: bool  # True = we own backend teardown; False = reused
    warnings: list[str] = field(default_factory=list)


class LauncherError(Exception):
    """A fatal launch condition, tagged with a machine-readable ``code``.

    ``code`` is the stable contract every shell maps from: the CLI to a Rich
    panel, the desktop GUI to its error page. ``message`` is human-readable and
    actionable; ``detail`` is an optional secondary line.
    """

    def __init__(self, code: str, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


# --------------------------------------------------------------------------- #
# Front-end runners (pluggable)
# --------------------------------------------------------------------------- #
class WebUIRunner(Protocol):
    """A front-end process strategy. Backend handling is shared; this is the
    only part that differs between the CLI (npx) and the desktop (bundled)."""

    # True → the runner opens the system browser itself (so the launcher must
    # not). False → the launcher opens it when ``cfg.open_browser`` is set.
    handles_browser_open: bool

    def preflight(self, cfg: LauncherConfig) -> None:
        """Raise :class:`LauncherError` if this front-end cannot run."""

    def start(self, cfg: LauncherConfig, env: dict[str, str]) -> subprocess.Popen:
        """Spawn the front-end process. ``env`` is the shared scrubbed env."""

    def stop(self, proc: subprocess.Popen) -> None:
        """Terminate the front-end process tree (idempotent)."""


class NpxWebUIRunner:
    """Today's behaviour: ``npx --yes @evoscientist/webui@latest``.

    The published package's launcher prints progress and opens the browser
    itself, so ``handles_browser_open`` is True.
    """

    handles_browser_open = True

    def preflight(self, cfg: LauncherConfig) -> None:
        if shutil.which("npx") is None:
            raise LauncherError(
                "node_missing",
                "Node.js / npx was not found on PATH. The WebUI front-end "
                "ships as the npm package @evoscientist/webui and is launched "
                "with npx.",
                "Install Node.js 24 LTS (which includes npx), then re-run "
                "EvoSci — or switch UI modes with "
                "'EvoSci config set ui_backend tui'.",
            )

    def start(self, cfg: LauncherConfig, env: dict[str, str]) -> subprocess.Popen:
        npx = shutil.which("npx")
        if npx is None:  # narrowed for type-checkers; preflight already ran
            raise LauncherError("node_missing", "npx disappeared after preflight.")
        try:
            return subprocess.Popen(
                [npx, "--yes", _WEBUI_PACKAGE, "--port", str(cfg.webui_port)],
                env=env,
                **_popen_group_kwargs(),
            )
        except Exception as exc:  # pragma: no cover - OS-level failure
            raise LauncherError(
                "webui_start_failed", f"Failed to launch WebUI via npx: {exc}"
            ) from exc

    def stop(self, proc: subprocess.Popen) -> None:
        _stop_process_tree(proc)


class BundledWebUIRunner:
    """Phase-1 desktop front-end: run the bundled Next.js standalone server
    directly with a bundled Node binary — no npx, no npm, no network.

    ``@evoscientist/webui`` ships a prebuilt standalone build; its own ``bin``
    runs ``node dist/server.js``. We bypass the ``bin`` (which opens a browser
    and writes to the console) and spawn the server directly, so the launcher
    controls readiness, browser-open and console output. Hence
    ``handles_browser_open`` is False.

    Args:
        app_dir: Directory containing the unpacked front-end (expects
            ``<app_dir>/dist/server.js``).
        node_exe: Path to the bundled ``node`` binary.
        log_path: When set, the node process's stdout+stderr are appended here
            (the windowed desktop app has no console, so otherwise the front-end
            output is discarded). When None, output is left to inherit as before.
    """

    handles_browser_open = False

    def __init__(
        self, app_dir: Path, node_exe: Path, log_path: Path | None = None
    ) -> None:
        self.app_dir = Path(app_dir)
        self.node_exe = Path(node_exe)
        self.server_entry = self.app_dir / "dist" / "server.js"
        self.log_path = Path(log_path) if log_path else None
        self._log_fh = None

    def preflight(self, cfg: LauncherConfig) -> None:
        if not self.node_exe.exists():
            raise LauncherError(
                "node_missing",
                f"Bundled Node runtime not found at {self.node_exe}.",
                "This indicates a broken installation — reinstall EvoScientist.",
            )
        if not self.server_entry.exists():
            raise LauncherError(
                "node_missing",
                f"Bundled WebUI server not found at {self.server_entry}.",
                "This indicates a broken installation — reinstall EvoScientist.",
            )

    def start(self, cfg: LauncherConfig, env: dict[str, str]) -> subprocess.Popen:
        # The standalone server reads PORT / HOSTNAME / NODE_ENV; the base env
        # already carries PORT + HOSTNAME, we add production mode here.
        run_env = {**env, "NODE_ENV": "production"}
        kwargs = _popen_group_kwargs()
        if os.name == "nt":
            # Desktop shell: keep the node process off any console so no window
            # flashes. OR it into the process-group flag so tree-kill still
            # works. (CREATE_NO_WINDOW exists only on Windows.)
            kwargs["creationflags"] = (
                kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            )
        if self.log_path is not None:
            # Persist front-end diagnostics: without a console the node output
            # would be lost, leaving a WebUI failure with no trace to report.
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "ab")
            kwargs["stdout"] = self._log_fh
            kwargs["stderr"] = subprocess.STDOUT
        try:
            return subprocess.Popen(
                [str(self.node_exe), str(self.server_entry)],
                env=run_env,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - OS-level failure
            self._close_log()
            raise LauncherError(
                "webui_start_failed",
                f"Failed to launch bundled WebUI (node {self.server_entry}): {exc}",
            ) from exc

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            finally:
                self._log_fh = None

    def stop(self, proc: subprocess.Popen) -> None:
        _stop_process_tree(proc)
        self._close_log()


# --------------------------------------------------------------------------- #
# Launcher core
# --------------------------------------------------------------------------- #
@dataclass
class _BackendDecision:
    action: str  # "reuse" | "start"
    warnings: list[str] = field(default_factory=list)


# Backend-resolution errors an auto-port (GUI) shell recovers from by starting
# its own backend on a free port instead of raising. A busy port, a server for
# a different workspace, and a stripped CLI-mode server are all unusable for the
# desktop's own workspace, and there is no terminal to act on the message.
_AUTO_PORT_FALLBACK_CODES = frozenset(
    {"port_conflict", "workspace_mismatch", "stripped_backend", "sidecar_port_mismatch"}
)


class WebUILauncher:
    """Orchestrates the backend (langgraph dev) + a pluggable front-end.

    Lifecycle owned by the caller:

        launcher = WebUILauncher(config, cfg, NpxWebUIRunner())
        result = launcher.start()          # non-blocking
        result = launcher.wait_ready(60)   # poll both, or raise LauncherError
        ...                                # caller's own main loop / window
        launcher.stop()                    # idempotent teardown
    """

    def __init__(self, config: Any, cfg: LauncherConfig, runner: WebUIRunner) -> None:
        self._config = config
        self._cfg = cfg
        self._runner = runner
        self._backend_proc: subprocess.Popen | None = None
        self._backend_started = False
        # True from just before start_langgraph_dev() until its proc handle is
        # wired. start_langgraph_dev blocks up to 60s on health, so a stop()
        # during that window has no _backend_proc to kill yet — this flag lets
        # _teardown stop the in-flight server this process spawned (the
        # manager's in-memory record) so a close mid-boot can't orphan it.
        self._backend_start_initiated = False
        self._webui_proc: subprocess.Popen | None = None
        self._stopped = False
        # Guards _teardown: start() (boot thread) and stop() (GUI thread) can
        # run concurrently on the desktop.
        self._lock = threading.Lock()
        self._warnings: list[str] = []

    # -- properties ------------------------------------------------------- #
    @property
    def backend_url(self) -> str:
        from ..langgraph_dev.manager import _base_url

        return _base_url(self._cfg.backend_port, self._cfg.backend_host)

    @property
    def webui_url(self) -> str:
        from ..langgraph_dev.manager import _format_hostport

        return f"http://{_format_hostport(self._cfg.webui_host, self._cfg.webui_port)}"

    @property
    def workspace_dir(self) -> str:
        """The workspace this launcher's backend serves. Read by the desktop
        shell to seed the folder picker and skip a no-op switch, and by the
        controller when rebuilding a launcher for a new workspace."""
        return self._cfg.workspace_dir

    @property
    def backend_started(self) -> bool:
        """True when this launcher started (and thus owns teardown of) the
        backend; False when it reused an already-running one. Closing the
        desktop only kills the backend's runs when this is True."""
        return self._backend_started

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> LaunchResult:
        """Resolve/start the backend and start the front-end.

        Does not wait for the front-end to be *ready* (use :meth:`wait_ready`
        for that). It does block while the backend starts: on the start path
        ``start_langgraph_dev`` polls the backend's ``/ok`` until healthy (up to
        ~60s). The front-end process is only spawned, not awaited.

        Tears down whatever it already spawned if a later step raises, or if
        ``stop()`` fires concurrently — the desktop calls ``stop()`` from the GUI
        thread while this runs on the boot thread. Without this, a failed CLI
        launch (``atexit`` not yet registered) or a window closed mid-boot would
        orphan the backend holding the port.
        """
        from ..langgraph_dev.manager import _is_port_occupied

        self._runner.preflight(self._cfg)

        try:
            decision = self._resolve_backend_with_auto_port()
            self._warnings.extend(decision.warnings)
            if decision.action == "start":
                self._start_backend()
            self._raise_if_stopped()

            # Front-end port: auto-port shells move off an occupied port too;
            # otherwise it's a non-fatal warning (node will surface a hard bind
            # failure if it actually can't listen).
            if _is_port_occupied(self._cfg.webui_port, self._cfg.webui_host):
                if self._cfg.auto_port:
                    free = _find_free_port(self._cfg.webui_port, self._cfg.webui_host)
                    self._warnings.append(
                        f"Port {self._cfg.webui_port} was occupied; using {free} "
                        f"for the WebUI instead."
                    )
                    self._cfg = replace(self._cfg, webui_port=free)
                else:
                    self._warnings.append(
                        f"Port {self._cfg.webui_port} is already in use; the WebUI "
                        f"server may fail to start. Change it with "
                        f"'EvoSci config set webui_port <port>'."
                    )

            env = self._build_frontend_env()
            self._webui_proc = self._runner.start(self._cfg, env)
            self._raise_if_stopped()
        except BaseException:
            self._teardown()
            raise

        return self._result()

    def _raise_if_stopped(self) -> None:
        """Abort startup if ``stop()`` fired on another thread mid-boot."""
        if self._stopped:
            raise LauncherError("cancelled", "Launch cancelled by shutdown.")

    def wait_ready(self, timeout: float = 60.0) -> LaunchResult:
        """Block until both services answer, or raise ``LauncherError``.

        Opens the browser at the end iff ``cfg.open_browser`` and the runner
        does not open it itself.
        """
        from ..langgraph_dev.manager import is_langgraph_dev_running

        deadline = time.monotonic() + timeout
        # Backend health first — the front-end is useless without it.
        while not is_langgraph_dev_running(
            port=self._cfg.backend_port, host=self._cfg.backend_host
        ):
            if time.monotonic() >= deadline:
                raise LauncherError(
                    "not_ready",
                    f"Backend did not become ready at {self.backend_url} "
                    f"within {timeout:.0f}s.",
                )
            if self._webui_proc is not None and self._webui_proc.poll() is not None:
                raise LauncherError(
                    "webui_start_failed",
                    "WebUI process exited before the backend became ready.",
                )
            time.sleep(0.5)

        _poll_ready(
            self.webui_url,
            timeout=max(1.0, deadline - time.monotonic()),
            webui_proc=self._webui_proc,
        )

        if self._cfg.open_browser and not getattr(
            self._runner, "handles_browser_open", False
        ):
            try:
                webbrowser.open(self.webui_url)
            except Exception:  # pragma: no cover - best-effort
                pass

        return self._result()

    def poll(self) -> bool:
        """Return True while the front-end process is alive. A caller's main
        loop uses this to notice the user closing the front-end."""
        return self._webui_proc is not None and self._webui_proc.poll() is None

    def stop(self) -> None:
        """Idempotent teardown. Stops the front-end always; stops the backend
        only if we started it and keepalive is off. Safe before/during
        ``start()``: always marks the launcher stopped so an in-flight
        ``start()`` aborts and cleans up what it spawned."""
        self._stopped = True
        self._teardown()

    def _teardown(self) -> None:
        """Stop every process this launcher spawned, at most once each.

        Lock-guarded because ``start()`` (boot thread) and ``stop()`` (GUI
        thread) can enter concurrently; each handle is cleared as it is stopped
        so a second entry is a no-op. A reused backend has no handle here, so it
        is never touched.
        """
        with self._lock:
            if self._webui_proc is not None:
                proc, self._webui_proc = self._webui_proc, None
                self._runner.stop(proc)
            if self._backend_proc is not None and not self._cfg.keepalive:
                from ..langgraph_dev.manager import stop_langgraph_dev

                proc, self._backend_proc = self._backend_proc, None
                stop_langgraph_dev(proc)
            elif self._backend_start_initiated and not self._cfg.keepalive:
                # stop() fired while start_langgraph_dev was still running, so we
                # have no proc handle yet. Stop only the process THIS launcher
                # spawned (the manager's in-memory record), never the shared
                # on-disk PID file: before our child is spawned that file still
                # names a *different* server (e.g. a CLI backend on another port
                # we auto-ported around), and killing it would orphan an
                # unrelated session. Once our child is spawned it is tracked in
                # memory, so a close mid-health-wait still tears it down.
                from ..langgraph_dev.manager import stop_inflight_owned_server

                self._backend_start_initiated = False
                stopped = stop_inflight_owned_server()
                if stopped is not None:
                    logger.warning(
                        "Launcher torn down mid-backend-start; stopped the "
                        "in-flight langgraph dev (pid %s).",
                        stopped,
                    )

    # -- internals -------------------------------------------------------- #
    def _resolve_backend_with_auto_port(self) -> _BackendDecision:
        """Resolve the backend, retrying on a free port for auto-port shells.

        In auto-port (GUI) mode the desktop always wants its OWN backend for
        its OWN workspace, and there is no terminal to act on a conflict
        message. So a busy port (``port_conflict``), a reusable server pinned
        to a different workspace (``workspace_mismatch``), and a stripped
        CLI-mode server (``stripped_backend``) all fall back to starting a
        fresh backend on a free port rather than dead-ending. A reusable
        server for THIS workspace is still reused: ``_resolve_backend``
        returns ``reuse`` before it would raise. The CLI (auto_port=False)
        keeps the explicit conflict errors so a terminal user can act on them.
        """
        try:
            return _resolve_backend(self._cfg, self._config)
        except LauncherError as exc:
            if not self._cfg.auto_port or exc.code not in _AUTO_PORT_FALLBACK_CODES:
                raise
            free = _find_free_port(self._cfg.backend_port, self._cfg.backend_host)
            self._warnings.append(
                f"Port {self._cfg.backend_port} was unavailable ({exc.code}); "
                f"started a new backend on {free} instead."
            )
            self._cfg = replace(self._cfg, backend_port=free)
            # The new port is free, so this resolves to ``start``.
            return _resolve_backend(self._cfg, self._config)

    def _start_backend(self) -> None:
        from ..langgraph_dev.manager import (
            _server_config_fingerprint,
            start_langgraph_dev,
        )

        jobs_per_worker = int(
            getattr(self._config, "langgraph_dev_jobs_per_worker", 10)
        )
        file_persistence = bool(
            getattr(self._config, "langgraph_dev_file_persistence", True)
        )
        # Mark in-flight before the (up-to-60s blocking) call so a concurrent
        # stop() can tear down our in-flight process even though our proc handle
        # is not wired until this returns.
        self._backend_start_initiated = True
        try:
            self._backend_proc = start_langgraph_dev(
                workspace_dir=Path(self._cfg.workspace_dir),
                port=self._cfg.backend_port,
                host=self._cfg.backend_host,
                file_persistence=file_persistence,
                jobs_per_worker=jobs_per_worker,
                deploy_mode=self._cfg.deploy_mode,
                config_fingerprint=_server_config_fingerprint(self._config),
            )
        except Exception as exc:
            raise LauncherError(
                "backend_start_failed", f"langgraph dev startup failed: {exc}"
            ) from exc
        self._backend_started = True
        # Handle is wired now; the in-flight fallback is no longer needed.
        self._backend_start_initiated = False

    def _build_frontend_env(self) -> dict[str, str]:
        # The UI reaches the backend from the BROWSER; give it the backend port
        # for config prefill. Secrets are scrubbed — the browser UI never needs
        # LLM provider API keys. HOSTNAME is the front-end's only bind knob.
        return _scrubbed_env(
            {
                "EVOSCIENTIST_LANGGRAPH_DEV_PORT": str(self._cfg.backend_port),
                "PORT": str(self._cfg.webui_port),
                "HOSTNAME": self._cfg.webui_host,
            }
        )

    def _result(self) -> LaunchResult:
        return LaunchResult(
            backend_url=self.backend_url,
            webui_url=self.webui_url,
            backend_started=self._backend_started,
            warnings=list(self._warnings),
        )


def _resolve_backend(cfg: LauncherConfig, config: Any) -> _BackendDecision:
    """Decide whether to reuse an already-running backend or start a fresh one.

    Pure decision logic: raises :class:`LauncherError` for foreign occupants,
    workspace mismatch, stripped (CLI-mode) servers and sidecar port
    mismatches; returns ``reuse`` or ``start`` otherwise.
    Config-fingerprint drift is a warning, not an error.
    """
    from ..langgraph_dev.manager import (
        _is_port_occupied,
        _pid_serves_port,
        _read_workspace_sidecar,
        _server_config_fingerprint,
        is_langgraph_dev_running,
    )

    warnings: list[str] = []
    if not _is_port_occupied(cfg.backend_port, cfg.backend_host):
        return _BackendDecision(action="start")

    if not is_langgraph_dev_running(port=cfg.backend_port, host=cfg.backend_host):
        raise LauncherError(
            "port_conflict",
            f"Port {cfg.backend_port} is occupied by another process.",
            f"Free it (lsof -i :{cfg.backend_port}) or change it with "
            f"'EvoSci config set langgraph_dev_port <port>'.",
        )

    # An EvoSci server is already there — reuse only if it serves THIS
    # workspace, is full deploy-mode, and (soft) matches the current config.
    sidecar = _read_workspace_sidecar()
    ws = Path(cfg.workspace_dir).resolve()
    if sidecar is None:
        # No ownership record for the server on this port. The single global
        # sidecar is unlinked when any EvoSci server stops, so a surviving
        # sibling on another port can be left record-less. The CLI still reuses
        # it (backward-compat: pre-sidecar and externally-managed servers). But
        # an auto-port shell (the desktop) can't verify the workspace/deploy-mode
        # it needs and always wants its OWN backend, so it must not blindly reuse
        # an unverifiable server — fall back (via sidecar_port_mismatch, an
        # auto-port code) to starting a fresh backend on another port.
        if cfg.auto_port:
            raise LauncherError(
                "sidecar_port_mismatch",
                f"Port {cfg.backend_port} is serving a langgraph dev that EvoSci "
                f"has no ownership record for.",
                f"Free port {cfg.backend_port} (lsof -i :{cfg.backend_port}) or "
                f"change it with 'EvoSci config set langgraph_dev_port <port>'.",
            )
        return _BackendDecision(action="reuse", warnings=warnings)
    # A sidecar is present. It is a single global record with no port field, so a
    # fallback launch on another port can overwrite it. Confirm its PID actually
    # serves THIS port before trusting its workspace — otherwise a stale record
    # could reuse the wrong workspace, and teardown could stop the wrong server.
    # Auto-port shells fall back to a fresh backend; the CLI gets an explicit error.
    if not _pid_serves_port(sidecar.get("pid"), cfg.backend_port):
        raise LauncherError(
            "sidecar_port_mismatch",
            f"Port {cfg.backend_port} is serving a langgraph dev that EvoSci "
            f"has no matching ownership record for.",
            f"Free port {cfg.backend_port} (lsof -i :{cfg.backend_port}) or "
            f"change it with 'EvoSci config set langgraph_dev_port <port>'. "
            f"'EvoSci server stop' stops EvoSci's recorded server on another "
            f"port, not this one.",
        )
    if Path(sidecar["workspace"]).resolve() != ws:
        raise LauncherError(
            "workspace_mismatch",
            f"Port {cfg.backend_port} is already serving a langgraph dev "
            f"for a different workspace ({sidecar['workspace']}).",
            f"Stop that EvoSci session, or launch from that workspace "
            f"(--workdir {sidecar['workspace']}).",
        )
    if sidecar.get("deploy_mode") is False:
        raise LauncherError(
            "stripped_backend",
            f"Port {cfg.backend_port} is serving a stripped (CLI-mode) "
            f"langgraph dev — the WebUI needs the full deploy-mode server "
            f"(MCP + async sub-agents).",
            "Stop it with 'EvoSci server stop', then re-run EvoSci.",
        )
    recorded_fp = sidecar.get("config_fingerprint")
    if isinstance(recorded_fp, str) and recorded_fp != _server_config_fingerprint(
        config
    ):
        warnings.append(
            "Config changed since this server was launched — it still "
            "serves the old settings. Apply them with 'EvoSci server "
            "stop', then re-run EvoSci."
        )
    return _BackendDecision(action="reuse", warnings=warnings)


def _find_free_port(start_port: int, host: str, *, limit: int = 100) -> int:
    """Return the first free TCP port at or above ``start_port`` on ``host``.

    Scans upward (predictable ports near the default, so the shown backend URL
    stays close to the configured one). Raises ``port_conflict`` if the window
    up to ``limit`` ports is fully occupied.
    """
    from ..langgraph_dev.manager import _is_port_occupied

    for port in range(start_port, min(start_port + limit, 65536)):
        if not _is_port_occupied(port, host):
            return port
    raise LauncherError(
        "port_conflict",
        f"No free port found in {start_port}–{start_port + limit - 1}.",
        "Free a port in that range, or set an explicit port in config.",
    )


# --------------------------------------------------------------------------- #
# Process / env / readiness helpers (shared; imported back by webui.py)
# --------------------------------------------------------------------------- #
def _popen_group_kwargs() -> dict[str, Any]:
    """Popen kwargs that put the front-end in its own process group so the
    whole tree (npx/node → next server) tears down as a unit."""
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return kwargs


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a front-end process tree (idempotent)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        elif os.name == "nt":
            # taskkill /T terminates the whole child tree (node + next server).
            # CREATE_NO_WINDOW: the windowed EvoScientist.exe has no console, so
            # spawning the console app taskkill would otherwise flash a blank
            # terminal window on shutdown.
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:  # pragma: no cover - exotic platform
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _scrubbed_env(extra: dict[str, str]) -> dict[str, str]:
    """Inherit the parent environment minus secrets, then apply ``extra``.

    The WebUI is a browser client that only talks to the local langgraph
    server — it has no use for LLM provider API keys. Stripping
    credential-bearing variables keeps them out of the front-end package and
    its transitive npm dependencies. Names are matched loosely (``*_KEY`` /
    ``*API_KEY*`` / ``*TOKEN*`` / ``*SECRET*`` / ``*PASSWORD*``); node/npm
    essentials (PATH, HOME, NODE_*, npm_*, proxies, CA certs) carry none of
    these and pass through untouched.
    """
    secret_hints = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
    env = {
        k: v
        for k, v in os.environ.items()
        if not (
            k.upper().endswith("_KEY")
            or any(hint in k.upper() for hint in secret_hints)
        )
    }
    env.update(extra)
    return env


def _poll_ready(
    url: str, timeout: float, interval: float = 0.5, *, webui_proc=None
) -> None:
    """GET ``url`` until it answers (any HTTP status < 500) or raise.

    A served-but-erroring page (< 500) counts as ready — the Next server is up;
    per-route errors are the app's concern, not the launcher's.

    ``webui_proc`` (the front-end ``Popen``, when the caller has one) is checked
    each iteration: if it has exited, the poll fails fast with
    ``webui_start_failed`` instead of waiting out the whole timeout as
    ``not_ready``. The backend-health loop already does this before the front-end
    is up; node can still crash (bad bundle, port race) after the backend is up.
    """
    # No-proxy opener: this is a loopback probe of our own front-end. The
    # default opener honours the environment/OS proxy — and on Windows that
    # includes the system (registry/IE) proxy even with no *_PROXY env vars —
    # which routes the 127.0.0.1 request off-box so the poll never succeeds.
    from urllib.error import HTTPError

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_err: str | None = None
    while time.monotonic() < deadline:
        if webui_proc is not None and webui_proc.poll() is not None:
            raise LauncherError(
                "webui_start_failed",
                "WebUI process exited before it became ready.",
                last_err,
            )
        try:
            with opener.open(url, timeout=2) as resp:
                if getattr(resp, "status", 200) < 500:
                    return
                last_err = f"HTTP {resp.status}"
        except HTTPError as exc:
            # The default opener installs HTTPErrorProcessor, which RAISES for any
            # non-2xx response — so a served-but-erroring page (e.g. 404/401 for a
            # moved landing route or added auth) lands here, not the success path.
            # It still means the server is up and answering, so <500 is ready;
            # only a 5xx keeps us waiting.
            if exc.code < 500:
                return
            last_err = f"HTTP {exc.code}"
        except Exception as exc:  # connection refused while still starting
            last_err = str(exc)
        time.sleep(interval)
    raise LauncherError(
        "not_ready",
        f"WebUI did not become ready at {url} within {timeout:.0f}s.",
        last_err,
    )


# --------------------------------------------------------------------------- #
# Launcher-config resolution (shared by the CLI WebUI and the desktop shell)
# --------------------------------------------------------------------------- #
def build_launcher_config(
    config: Any,
    workspace_dir: str | None,
    *,
    auto_port: bool = False,
    keepalive: bool | None = None,
) -> LauncherConfig:
    """Resolve a :class:`LauncherConfig` from an ``EvoScientistConfig`` the
    same way ``run_webui`` does, so both entrypoints agree.

    ``auto_port`` is set by GUI shells (no terminal to act on a conflict); the
    CLI leaves it False so a busy port surfaces as an explicit error.

    ``keepalive`` defaults to the ``langgraph_dev_keepalive`` config value; pass
    an explicit bool to override it. The desktop passes ``False`` so its backend
    is always torn down (it owns the backend and has no terminal to reclaim a
    leftover from).
    """
    from ..langgraph_dev.manager import _DEFAULT_HOST, _DEFAULT_PORT

    if workspace_dir:
        ws = os.path.abspath(os.path.expanduser(workspace_dir))
    elif getattr(config, "default_workdir", ""):
        ws = os.path.abspath(os.path.expanduser(config.default_workdir))
    else:
        ws = os.getcwd()
    os.makedirs(ws, exist_ok=True)

    backend_port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    webui_port = int(getattr(config, "webui_port", _DEFAULT_WEBUI_PORT))
    backend_host = (
        str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or _DEFAULT_HOST)
    ).strip() or _DEFAULT_HOST
    webui_host = (
        str(getattr(config, "webui_host", _DEFAULT_WEBUI_HOST) or _DEFAULT_WEBUI_HOST)
    ).strip() or _DEFAULT_WEBUI_HOST
    return LauncherConfig(
        workspace_dir=ws,
        backend_host=backend_host,
        backend_port=backend_port,
        webui_host=webui_host,
        webui_port=webui_port,
        deploy_mode=True,
        keepalive=(
            bool(getattr(config, "langgraph_dev_keepalive", False))
            if keepalive is None
            else keepalive
        ),
        open_browser=False,
        auto_port=auto_port,
    )
