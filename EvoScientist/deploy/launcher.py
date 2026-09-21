"""Shell-agnostic WebUI launcher core.

Extracts the reusable start / health / stop logic out of
``deploy/webui.py:run_webui`` so it can be driven by *any* front-end:

- the existing terminal CLI (``run_webui`` is now a thin adapter over this),
- an embedded desktop shell (imports :class:`WebUILauncher` in-process), and
- an out-of-process shell that spawns ``python -m EvoScientist.deploy.launcher``
  and parses the one-line JSON ready-signal from stdout.

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

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

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
    panel, a GUI to an error dialog, the JSON entrypoint to ``{"status":
    "error", "code": ...}``. ``message`` is human-readable and actionable;
    ``detail`` is an optional secondary line.
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
    """

    handles_browser_open = False

    def __init__(self, app_dir: Path, node_exe: Path) -> None:
        self.app_dir = Path(app_dir)
        self.node_exe = Path(node_exe)
        self.server_entry = self.app_dir / "dist" / "server.js"

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
        try:
            return subprocess.Popen(
                [str(self.node_exe), str(self.server_entry)],
                env=run_env,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - OS-level failure
            raise LauncherError(
                "webui_start_failed",
                f"Failed to launch bundled WebUI (node {self.server_entry}): {exc}",
            ) from exc

    def stop(self, proc: subprocess.Popen) -> None:
        _stop_process_tree(proc)


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
    {"port_conflict", "workspace_mismatch", "stripped_backend"}
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
        self._webui_proc: subprocess.Popen | None = None
        self._stopped = False
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
        """Resolve/start the backend and start the front-end. Non-blocking:
        does not wait for readiness (use :meth:`wait_ready`)."""
        from ..langgraph_dev.manager import _is_port_occupied

        self._runner.preflight(self._cfg)

        decision = self._resolve_backend_with_auto_port()
        self._warnings.extend(decision.warnings)
        if decision.action == "start":
            self._start_backend()

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

        return self._result()

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

        _poll_ready(self.webui_url, timeout=max(1.0, deadline - time.monotonic()))

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
        only if we started it and keepalive is off."""
        if self._stopped:
            return
        self._stopped = True
        if self._webui_proc is not None:
            self._runner.stop(self._webui_proc)
        if self._backend_started and not self._cfg.keepalive:
            from ..langgraph_dev.manager import stop_langgraph_dev

            stop_langgraph_dev(self._backend_proc)

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

    Pure decision logic (mirrors the guards in ``run_webui``): raises
    :class:`LauncherError` for foreign occupants, workspace mismatch and
    stripped (CLI-mode) servers; returns ``reuse`` or ``start`` otherwise.
    Config-fingerprint drift is a warning, not an error.
    """
    from ..langgraph_dev.manager import (
        _is_port_occupied,
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
    if sidecar is not None:
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
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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


def _poll_ready(url: str, timeout: float, interval: float = 0.5) -> None:
    """GET ``url`` until it answers (any HTTP status < 500) or raise.

    A served-but-erroring page (< 500) counts as ready — the Next server is up;
    per-route errors are the app's concern, not the launcher's.
    """
    # No-proxy opener: this is a loopback probe of our own front-end. The
    # default opener honours the environment/OS proxy — and on Windows that
    # includes the system (registry/IE) proxy even with no *_PROXY env vars —
    # which routes the 127.0.0.1 request off-box so the poll never succeeds.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=2) as resp:
                if getattr(resp, "status", 200) < 500:
                    return
                last_err = f"HTTP {resp.status}"
        except Exception as exc:  # connection refused while still starting
            last_err = str(exc)
        time.sleep(interval)
    raise LauncherError(
        "not_ready",
        f"WebUI did not become ready at {url} within {timeout:.0f}s.",
        last_err,
    )


# --------------------------------------------------------------------------- #
# Standalone JSON entrypoint (for an out-of-process shell, e.g. Electron)
# --------------------------------------------------------------------------- #
def build_launcher_config(
    config: Any, workspace_dir: str | None, *, auto_port: bool = False
) -> LauncherConfig:
    """Resolve a :class:`LauncherConfig` from an ``EvoScientistConfig`` the
    same way ``run_webui`` does, so both entrypoints agree.

    ``auto_port`` is set by GUI shells (no terminal to act on a conflict); the
    CLI leaves it False so a busy port surfaces as an explicit error.
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
        keepalive=bool(getattr(config, "langgraph_dev_keepalive", False)),
        open_browser=False,
        auto_port=auto_port,
    )


def _emit(obj: dict[str, Any]) -> None:
    """Write one JSON line to stdout and flush — the ready/error signal."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Standalone entrypoint: start the WebUI, emit a JSON ready-signal on
    stdout, then block (keeping the processes alive) until stdin closes or a
    termination signal arrives, then tear down.

    Success prints ``{"status": "ready", "backend_url", "webui_url"}``; failure
    prints ``{"status": "error", "code", "message", "detail"}`` to stdout and
    the error to stderr. stderr stays empty on success.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="EvoScientist.deploy.launcher")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    from ..config import apply_config_to_env, get_effective_config

    config = get_effective_config()
    apply_config_to_env(config)
    # Standalone entrypoint is an out-of-process GUI shell (e.g. Electron):
    # auto-port like the in-process desktop shell.
    cfg = build_launcher_config(config, args.workspace, auto_port=True)

    launcher = WebUILauncher(config, cfg, NpxWebUIRunner())
    try:
        launcher.start()
        result = launcher.wait_ready(timeout=args.ready_timeout)
    except LauncherError as exc:
        _emit(
            {
                "status": "error",
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            }
        )
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        launcher.stop()
        return 1

    _emit(
        {
            "status": "ready",
            "backend_url": result.backend_url,
            "webui_url": result.webui_url,
            "warnings": result.warnings,
        }
    )

    # Block until the shell closes our stdin, a signal arrives, or the
    # front-end dies on its own — then tear down what we started.
    shutdown = threading.Event()

    def _handle(signum: int, _frame: Any) -> None:
        shutdown.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    try:
        while not shutdown.is_set():
            if not launcher.poll():
                break
            if sys.stdin.closed:
                break
            shutdown.wait(timeout=0.5)
    finally:
        launcher.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
