"""Shell-agnostic WebUI launcher core.

Extracts the reusable start / health / stop logic out of
``deploy/webui.py:run_webui`` so it can be driven by *any* front-end:

- the existing terminal CLI (``run_webui`` is now a thin adapter over this), and
- any other front-end that imports :class:`WebUILauncher` in-process.

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
  only the front-end differs, behind :class:`WebUIRunner`. The front-end runs
  as ``node dist/server.js``: :class:`BundledWebUIRunner` for a given copy,
  :class:`InstalledWebUIRunner` for the copy ``EvoSci setup`` installs.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

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
    keepalive: bool = False
    # Honoured by :meth:`WebUILauncher.wait_ready` for front-ends whose
    # ``handles_browser_open`` is False (every runner in this module).
    open_browser: bool = False


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of a successful :meth:`WebUILauncher.start` / ``wait_ready``."""

    backend_url: str
    webui_url: str
    backend_started: bool  # True = we own backend teardown; False = reused
    warnings: list[str] = field(default_factory=list)
    # True when wait_ready() opened a browser (False on a host without one).
    browser_opened: bool = False


class LauncherError(Exception):
    """A fatal launch condition, tagged with a machine-readable ``code``.

    ``code`` is the stable contract every front-end maps from: the CLI to a Rich
    panel, other front-ends to their own error display. ``message`` is
    human-readable and actionable; ``detail`` is an optional secondary line.
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
    only part that differs between front-ends."""

    # True → the runner opens the system browser itself (so the launcher must
    # not). False → the launcher opens it when ``cfg.open_browser`` is set.
    handles_browser_open: bool

    def preflight(self, cfg: LauncherConfig) -> None:
        """Raise :class:`LauncherError` if this front-end cannot run."""

    def start(self, cfg: LauncherConfig, env: dict[str, str]) -> subprocess.Popen:
        """Spawn the front-end process. ``env`` is the shared scrubbed env."""

    def stop(self, proc: subprocess.Popen) -> None:
        """Terminate the front-end process tree (idempotent)."""


class BundledWebUIRunner:
    """Locally installed front-end: run the Next.js standalone server directly
    with a given Node binary — no npx, no npm, no network.

    ``@evoscientist/webui`` ships a prebuilt standalone build; its own ``bin``
    runs ``node dist/server.js``. We bypass the ``bin`` (which opens a browser
    and writes to the console) and spawn the server directly, so the launcher
    controls readiness, browser-open and console output. Hence
    ``handles_browser_open`` is False.

    Args:
        app_dir: Directory containing the unpacked front-end (expects
            ``<app_dir>/dist/server.js``).
        node_exe: Path to the ``node`` binary.
        log_path: When set, the node process's stdout+stderr are appended here
            (a caller without a console would otherwise lose the front-end
            output). When None, output is left to inherit as before.
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
                f"Node runtime not found at {self.node_exe}.",
                "Run `EvoSci setup` to install it.",
            )
        if not self.server_entry.exists():
            raise LauncherError(
                "node_missing",
                f"WebUI server not found at {self.server_entry}.",
                "Run `EvoSci setup` to install it.",
            )

    def start(self, cfg: LauncherConfig, env: dict[str, str]) -> subprocess.Popen:
        from ..setup.node import is_private, node_child_env

        # The standalone server reads PORT / HOSTNAME / NODE_ENV; the base env
        # already carries PORT + HOSTNAME, we add production mode here. Our
        # private Node runs without the user's npm / Node options.
        run_env = node_child_env(
            {**env, "NODE_ENV": "production"}, private=is_private(self.node_exe)
        )
        kwargs = _popen_group_kwargs()
        if os.name == "nt":
            # Keep the node process off any console so no window flashes. OR it
            # into the process-group flag so tree-kill still works.
            # (CREATE_NO_WINDOW exists only on Windows.)
            kwargs["creationflags"] = (
                kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            )
        if self.log_path is not None:
            # Persist front-end diagnostics: without a console the node output
            # would be lost, leaving a WebUI failure with no trace to report.
            from ..langgraph_dev.manager import _rotate_log_if_needed

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_log_if_needed(self.log_path)
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
                f"Failed to launch the WebUI (node {self.server_entry}): {exc}",
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


class InstalledWebUIRunner:
    """The front-end ``EvoSci setup`` installs under ``tools/webui/``.

    :meth:`preflight` provides it on demand: a Node from ``ensure_node()``, then
    the WebUI from ``ensure_webui()``, which first switches to a copy the
    background check staged, and installs one when nothing is recorded (a pip
    user who never ran ``EvoSci setup``). It runs before the launcher starts
    the backend, so a failed install leaves nothing running. The copy is then
    marked in use for this process, older copies nobody runs are removed, and
    the launch itself goes through :class:`BundledWebUIRunner`.

    Args:
        progress: ``(fraction, message)`` callback for an on-demand install.
        log_path: Passed to :class:`BundledWebUIRunner`.
    """

    handles_browser_open = False

    def __init__(
        self,
        progress: Callable[[float, str], None] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._progress = progress
        self._log_path = log_path
        self._bundled: BundledWebUIRunner | None = None
        self._marker: Path | None = None
        self.node_exe: Path | None = None
        self.version: str | None = None

    def preflight(self, cfg: LauncherConfig) -> None:
        from ..setup.node import activate_runtime, ensure_node
        from ..setup.protocol import StageError
        from ..setup.webui import cleanup_old_versions, ensure_webui, mark_in_use

        try:
            node = ensure_node(progress=self._progress)
        except StageError as exc:
            raise LauncherError(
                "node_missing", exc.message, "Run `EvoSci setup` when online."
            ) from exc
        # A Node installed just now is not on PATH yet; the backend started
        # next, and the agent's shell inside it, need it there.
        activate_runtime()
        try:
            info = ensure_webui(node.path, progress=self._progress)
        except StageError as exc:
            raise LauncherError(
                exc.code, exc.message, "Run `EvoSci setup` when online."
            ) from exc
        self.node_exe, self.version = node.path, info.version
        if self._progress is not None:
            self._progress(1.0, f"WebUI {info.version}")
        if self._marker is None:
            self._marker = mark_in_use(info)
        cleanup_old_versions()
        self._bundled = BundledWebUIRunner(info.path, node.path, self._log_path)
        try:
            self._bundled.preflight(cfg)
        except BaseException:
            self.release()
            raise

    def start(self, cfg: LauncherConfig, env: dict[str, str]) -> subprocess.Popen:
        if self._bundled is None:  # narrowed for type-checkers
            raise LauncherError("node_missing", "preflight() did not run.")
        return self._bundled.start(cfg, env)

    def stop(self, proc: subprocess.Popen) -> None:
        if self._bundled is not None:
            self._bundled.stop(proc)
        self.release()

    def release(self) -> None:
        """Drop this process's in-use marker (idempotent). The launcher calls
        it on every teardown, also when the front-end never started."""
        from ..setup.webui import release_in_use

        release_in_use(self._marker)
        self._marker = None

    def start_update_check(self) -> threading.Thread | None:
        """Look for a newer WebUI in range in a daemon thread; a download is
        used from the next launch. Call once the UI is up."""
        from ..setup.webui import stage_update

        if self.node_exe is None:
            return None
        thread = threading.Thread(
            target=stage_update,
            args=(self.node_exe,),
            name="webui-update-check",
            daemon=True,
        )
        thread.start()
        return thread


# --------------------------------------------------------------------------- #
# Launcher core
# --------------------------------------------------------------------------- #
@dataclass
class _BackendDecision:
    action: str  # "reuse" | "start"
    warnings: list[str] = field(default_factory=list)


class WebUILauncher:
    """Orchestrates the backend (langgraph dev) + a pluggable front-end.

    Lifecycle owned by the caller:

        launcher = WebUILauncher(config, cfg, InstalledWebUIRunner())
        result = launcher.start()          # blocks while the backend starts (~60s max)
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
        # Guards _teardown: a caller may run start() on one thread and call
        # stop() from another.
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

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> LaunchResult:
        """Resolve/start the backend and start the front-end.

        Does not wait for the front-end to be *ready* (use :meth:`wait_ready`
        for that). It does block while the backend starts: on the start path
        ``start_langgraph_dev`` polls the backend's ``/ok`` until healthy (up to
        ~60s). The front-end process is only spawned, not awaited.

        Tears down whatever it already spawned if a later step raises, or if
        ``stop()`` fires concurrently from another thread. Without this, a
        failed CLI launch (``atexit`` not yet registered) or a caller that
        stops mid-boot would orphan the backend holding the port.
        """
        from ..langgraph_dev.manager import _is_port_occupied, _read_workspace_sidecar
        from ..setup.research_env import server_missing_python_hint

        self._runner.preflight(self._cfg)

        try:
            decision = _resolve_backend(self._cfg, self._config)
            self._warnings.extend(decision.warnings)
            # The agent is built inside the server, so its own hint lands in
            # the server log; the launching process shows it, for the server
            # it reuses or the one it starts.
            reused = _read_workspace_sidecar() if decision.action == "reuse" else None
            hint = server_missing_python_hint(reused)
            if hint is not None:
                self._warnings.append(hint)
            if decision.action == "start":
                self._start_backend()
            self._raise_if_stopped()

            # Front-end port: a non-fatal warning (node will surface a hard
            # bind failure if it actually can't listen).
            if _is_port_occupied(self._cfg.webui_port, self._cfg.webui_host):
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
            self._raise_if_stopped()
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

        try:
            _poll_ready(
                self.webui_url,
                timeout=max(1.0, deadline - time.monotonic()),
                webui_proc=self._webui_proc,
            )
        except LauncherError:
            self._raise_if_stopped()
            raise

        if self._cfg.open_browser and not getattr(
            self._runner, "handles_browser_open", False
        ):
            # Without a display (an SSH session), Python's webbrowser falls
            # back to console browsers on PATH (w3m, lynx, ...) and runs them
            # in the foreground; skip the open there unless BROWSER says
            # which one to use.
            opened = False
            headless = (
                sys.platform not in ("darwin", "win32")
                and not os.environ.get("BROWSER")
                and not os.environ.get("DISPLAY")
                and not os.environ.get("WAYLAND_DISPLAY")
            )
            if not headless:
                try:
                    opened = bool(webbrowser.open(self.webui_url))
                except Exception:  # pragma: no cover - best-effort
                    pass
            return dataclasses.replace(self._result(), browser_opened=opened)

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

        Lock-guarded because ``start()`` and ``stop()`` can enter concurrently
        from different threads; each handle is cleared as it is stopped
        so a second entry is a no-op. A reused backend has no handle here, so it
        is never touched.
        """
        with self._lock:
            if self._webui_proc is not None:
                proc, self._webui_proc = self._webui_proc, None
                self._runner.stop(proc)
            # A runner that marks its copy in use in preflight (the installed
            # one) releases it here too when a later step failed before the
            # front-end started.
            release = getattr(self._runner, "release", None)
            if release is not None:
                release()
            if self._backend_proc is not None and not self._cfg.keepalive:
                from ..langgraph_dev.manager import stop_langgraph_dev

                proc, self._backend_proc = self._backend_proc, None
                stop_langgraph_dev(proc)
            elif self._backend_start_initiated and not self._cfg.keepalive:
                # stop() fired while start_langgraph_dev was still running, so we
                # have no proc handle yet. Stop only the process THIS launcher
                # spawned (the manager's in-memory record), never the shared
                # on-disk PID file: before our child is spawned that file still
                # names a *different* server (e.g. another session's backend on
                # a different port), and killing it would take down an
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
                config_fingerprint=_server_config_fingerprint(self._config),
            )
        except Exception as exc:
            self._raise_if_stopped()
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
    workspace mismatch and sidecar port mismatches; returns ``reuse`` or
    ``start`` otherwise. Config-fingerprint drift (configuration or version)
    is a warning, not an error.
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
    # workspace, and (soft) matches the current config and version.
    sidecar = _read_workspace_sidecar()
    ws = Path(cfg.workspace_dir).resolve()
    if sidecar is None:
        # No ownership record for the server on this port. The single global
        # sidecar is unlinked when any EvoSci server stops, so a surviving
        # sibling on another port can be left record-less. Reuse it anyway
        # (backward-compat: pre-sidecar and externally-managed servers).
        return _BackendDecision(action="reuse", warnings=warnings)
    # A sidecar is present. It is a single global record with no port field, so a
    # launch on another port can overwrite it. Confirm its PID actually serves
    # THIS port before trusting its workspace — otherwise a stale record could
    # reuse the wrong workspace, and teardown could stop the wrong server.
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
    # The WebUI works in the workspace root; a server started for a
    # ``--mode=run`` session works in one of its run folders.
    if sidecar.get("run_dir"):
        raise LauncherError(
            "workspace_mismatch",
            f"Port {cfg.backend_port} is already serving a langgraph dev "
            f"for a run folder of this workspace ({sidecar['run_dir']}).",
            "Stop that EvoSci --mode=run session, then launch again."
            + (
                " If that session already ended, stop its kept-alive server "
                "with 'EvoSci server stop'."
                if cfg.keepalive
                else ""
            ),
        )
    recorded_fp = sidecar.get("config_fingerprint")
    if isinstance(recorded_fp, str) and recorded_fp != _server_config_fingerprint(
        config
    ):
        warnings.append(
            "Configuration or version changed since this server was "
            "launched — it still serves the old settings or version. Apply them with "
            "'EvoSci server stop', then re-run EvoSci."
        )
    from ..setup.research_env import python_drift_message

    drift = python_drift_message(sidecar)
    if drift is not None:
        warnings.append(drift)
    return _BackendDecision(action="reuse", warnings=warnings)


# --------------------------------------------------------------------------- #
# Process / env / readiness helpers (shared; imported back by webui.py)
# --------------------------------------------------------------------------- #
def _popen_group_kwargs() -> dict[str, Any]:
    """Popen kwargs that put the front-end in its own process group so the
    whole tree (node and anything it spawns) tears down as a unit."""
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
            # taskkill /T terminates the whole child tree (node and its children).
            # CREATE_NO_WINDOW: a caller without a console (a windowed app)
            # would otherwise flash a blank terminal window when spawning the
            # console app taskkill on shutdown.
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
# Launcher-config resolution (shared by every front-end)
# --------------------------------------------------------------------------- #
def build_launcher_config(
    config: Any,
    workspace_dir: str | None,
) -> LauncherConfig:
    """Resolve a :class:`LauncherConfig` from an ``EvoScientistConfig`` the
    same way ``run_webui`` does, so every entrypoint agrees. The launcher
    opens the browser once the WebUI answers."""
    from ..langgraph_dev.manager import _DEFAULT_HOST, _DEFAULT_PORT
    from ..paths import start_workspace_path

    ws = str(
        start_workspace_path(workspace_dir, getattr(config, "default_workdir", ""))
    )
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
        keepalive=bool(getattr(config, "langgraph_dev_keepalive", False)),
        open_browser=True,
    )
