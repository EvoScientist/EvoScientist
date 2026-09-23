"""pywebview glue for the desktop shell.

Keeps the GUI-specific code (creating the window, rendering status/error HTML,
wiring the close event) thin and separate from the boot logic in
:mod:`EvoScientist.desktop.controller`. ``webview`` is imported lazily so the
rest of the package imports without the optional ``desktop`` extra installed.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
from urllib.parse import parse_qs, urlparse

from .controller import DesktopController
from .setup import apply_setup, render_setup_html, setup_needed, validate_setup

logger = logging.getLogger("EvoScientist.desktop")

# UUID (any version) — the shape of a langgraph thread id. Used to reject a
# malformed ``threadId`` read from the WebUI URL before it reaches a backend
# query (a bad id would 404/400 the probe, which the switch would read as
# "can't tell" and wait forever).
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _first_path(selection) -> str | None:
    """Normalise a pywebview folder-dialog result to a single path or None.

    ``create_file_dialog`` returns a tuple/list of selected paths, ``None`` when
    the user cancels, and (on some backends) a bare string. Fold all three."""
    if not selection:
        return None
    if isinstance(selection, (list, tuple)):
        return selection[0] if selection else None
    return str(selection)


def _same_dir(a: str | None, b: str | None) -> bool:
    """True if two paths point at the same directory (case-insensitive on
    Windows), so a picker that reselects the current workspace is a no-op."""
    if not a or not b:
        return False
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _configure_logging() -> None:
    """Attach a file handler to the desktop logger (idempotent).

    Persists boot/shutdown/error diagnostics next to the backend log, so a
    failure on the windowed app leaves a trace a non-terminal user can report.
    """
    from . import app_paths

    if any(getattr(h, "_evosci_desktop", False) for h in logger.handlers):
        return
    try:
        path = app_paths.desktop_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:  # pragma: no cover - unwritable config dir; log to nowhere
        return
    handler._evosci_desktop = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_PAGE_STYLE = """
  :root { color-scheme: light dark; }
  body { margin: 0; font: 15px -apple-system, Segoe UI, system-ui, sans-serif;
         display: flex; min-height: 100vh; align-items: center;
         justify-content: center; background: #f6f7f9; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #16181c; color: #e6e6e6; } .card { background: #202329; }
  }
  .card { max-width: 560px; padding: 32px 36px; border-radius: 12px;
          background: #fff; box-shadow: 0 2px 24px rgba(0,0,0,.08); }
  h1 { font-size: 18px; margin: 0 0 12px; }
  p { margin: 6px 0; line-height: 1.5; }
  .detail { opacity: .75; font-size: 13px; }
  .code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 12px; opacity: .6; }
"""


def _status_html(message: str) -> str:
    return (
        f"<!doctype html><meta charset=utf-8><style>{_PAGE_STYLE}</style>"
        f"<div class=card><h1>EvoScientist</h1>"
        f"<p>{html.escape(message)}</p></div>"
    )


def _error_html(code: str, message: str, detail: str | None) -> str:
    from . import app_paths

    detail_block = f"<p class=detail>{html.escape(detail)}</p>" if detail else ""
    log_dir = html.escape(str(app_paths.desktop_log_path().parent))
    return (
        f"<!doctype html><meta charset=utf-8><style>{_PAGE_STYLE}</style>"
        f"<div class=card><h1>EvoScientist could not start</h1>"
        f"<p>{html.escape(message)}</p>{detail_block}"
        f"<p class=code>{html.escape(code)}</p>"
        f"<p class=code>Logs: {log_dir}</p></div>"
    )


class _WebviewWindow:
    """Adapts a pywebview window to the controller's ``DesktopWindow`` surface.

    Serializes navigations against pywebview's ``loaded`` event. ``load_html`` /
    ``load_url`` only block on the ``shown`` event, not on content readiness:
    each starts an async navigation and clears ``loaded``, which re-fires when
    that content finishes. Issuing the next swap before the previous navigation
    completes lets WebView2 drop the losing one — which is what silently
    dropped very-early error pages (a preflight failure whose error page raced
    the window's initial status page). Waiting for ``loaded`` before the first
    swap and after each one keeps navigations from overlapping.
    """

    # Bounded so a webview backend that never signals ``loaded`` can't hang the
    # boot thread; mirrors pywebview's own 20s ``shown``/``loaded`` waits.
    _SETTLE_TIMEOUT = 20.0

    def __init__(self, window) -> None:
        self._window = window
        self._initial_loaded = False

    def _await_initial_load(self) -> None:
        """Block until the window's initial (create-time) content has loaded, so
        the first swap does not race it. No-op after the first call."""
        if not self._initial_loaded:
            self._window.events.loaded.wait(self._SETTLE_TIMEOUT)
            self._initial_loaded = True

    def _load_html(self, markup: str) -> None:
        self._await_initial_load()
        self._window.load_html(markup)
        # load_html cleared ``loaded`` synchronously; wait for the new content
        # to settle before any subsequent swap can be issued.
        self._window.events.loaded.wait(self._SETTLE_TIMEOUT)

    def show_status(self, message: str) -> None:
        self._load_html(_status_html(message))

    def load_url(self, url: str) -> None:
        self._await_initial_load()
        self._window.load_url(url)

    def show_error(self, code: str, message: str, detail: str | None) -> None:
        self._load_html(_error_html(code, message, detail))

    def show_setup(self, markup: str) -> None:
        self._load_html(markup)

    def show_pending(self, message: str) -> None:
        """Inject/update a fixed banner over the LIVE WebUI (no navigation), so
        a pending workspace switch shows without hiding the current turn. Carries
        a Cancel button that calls the exposed ``cancel_workspace_switch`` bridge
        (see :func:`run_desktop`) to abort the wait and stay in the current
        workspace. Runs JS in the third-party WebUI page, so it is strictly
        best-effort — a failure must never block the switch. ``json.dumps`` makes
        the message a safe JS string literal."""
        self._eval_best_effort(
            "(function(){var id='__evosci_pending__';"
            "var d=document.getElementById(id);"
            "if(!d){d=document.createElement('div');d.id=id;"
            "d.style.cssText='position:fixed;top:0;left:0;right:0;"
            "z-index:2147483647;padding:8px 14px;display:flex;align-items:center;"
            "justify-content:center;gap:12px;"
            "font:13px -apple-system,Segoe UI,system-ui,sans-serif;"
            "background:#1a73e8;color:#fff;box-shadow:0 1px 6px rgba(0,0,0,.3);';"
            "var m=document.createElement('span');m.id='__evosci_pending_msg__';"
            "var b=document.createElement('button');b.textContent='Cancel';"
            "b.style.cssText='font:inherit;padding:2px 10px;border:1px solid #fff;"
            "border-radius:4px;background:transparent;color:#fff;cursor:pointer;';"
            "b.onclick=function(){this.disabled=true;this.textContent='Cancelling\\u2026';"
            "try{window.pywebview.api.cancel_workspace_switch();}catch(e){}};"
            "d.appendChild(m);d.appendChild(b);document.body.appendChild(d);}"
            f"document.getElementById('__evosci_pending_msg__').textContent="
            f"{json.dumps(message)};}})();"
        )

    def clear_pending(self) -> None:
        self._eval_best_effort(
            "(function(){var e=document.getElementById('__evosci_pending__');"
            "if(e)e.remove();})();"
        )

    def _eval_best_effort(self, script: str) -> None:
        try:
            self._window.evaluate_js(script)
        except Exception as exc:  # a banner must never block or crash the switch
            logger.warning("pending-banner JS failed: %s", exc)

    def current_thread_id(self) -> str | None:
        """The thread the user is viewing, parsed from the WebUI URL's
        ``threadId`` query param. Best-effort and validated as a UUID — returns
        None if it is absent, malformed, or unreadable, so the switch/close
        gating falls back to busy-only (never blocks on a bad id).

        Uses ``get_current_url`` (a cached attribute the webview updates on
        navigation, including SPA ``pushState``), NOT ``evaluate_js``: running JS
        synchronously from the ``closing`` event handler deadlocks the webview
        message loop (it waits on a JS-result semaphore that the closing pump
        can't release), which froze the app on close."""
        try:
            url = self._window.get_current_url()
        except Exception as exc:
            logger.warning("current-thread read failed: %s", exc)
            return None
        if not url:
            return None
        try:
            tid = parse_qs(urlparse(str(url)).query).get("threadId", [""])[0]
        except Exception:
            return None
        return tid.strip() if _UUID_RE.match(tid.strip()) else None


class _SetupApi:
    """pywebview ``js_api`` for the first-run setup form.

    Only ``submit`` is called from the page. It validates the answers, writes
    them to config via :func:`apply_setup`, and sets ``done`` so the boot
    thread (blocked in :func:`run_desktop`) can proceed. On invalid input it
    returns an error for the page to show inline, leaving ``done`` unset.
    """

    def __init__(self, config, done: threading.Event) -> None:
        self._config = config
        self._done = done

    def submit(self, payload: dict) -> dict:
        provider = str((payload or {}).get("provider", ""))
        model = str((payload or {}).get("model", ""))
        api_key = str((payload or {}).get("api_key", ""))
        workspace = str((payload or {}).get("workspace", ""))
        error = validate_setup(provider, model, api_key)
        if error:
            return {"ok": False, "error": error}
        try:
            apply_setup(provider, model, api_key, workspace, config=self._config)
        except Exception as exc:  # pragma: no cover - filesystem/permission failure
            return {"ok": False, "error": f"Could not save settings: {exc}"}
        self._done.set()
        return {"ok": True}


def run_desktop(workspace_dir: str | None = None) -> None:
    """Launch the desktop shell and block until the window closes.

    Builds a :class:`~EvoScientist.deploy.launcher.WebUILauncher` over the
    bundled Node/WebUI, shows a status page, boots services in pywebview's
    worker thread, then swaps to the WebUI URL. Tears down on window close.
    """
    import webview  # lazy: requires the optional ``desktop`` extra

    from ..config import apply_config_to_env, get_effective_config
    from ..deploy.launcher import (
        BundledWebUIRunner,
        WebUILauncher,
        build_launcher_config,
    )
    from . import app_paths

    _configure_logging()
    logger.info("desktop shell starting")

    config = get_effective_config()
    apply_config_to_env(config)
    logger.info(
        "config resolved: provider=%s model=%s workspace=%s",
        config.provider,
        config.model,
        config.default_workdir or workspace_dir or "(cwd)",
    )

    # First-run setup runs in the boot thread (below), so the launcher is built
    # only after config is final. ``done`` unblocks the boot thread when the
    # form is submitted; the close handler cancels it so closing the window
    # mid-setup exits cleanly instead of hanging.
    setup_done = threading.Event()
    cancelled = threading.Event()
    setup_api = _SetupApi(config, setup_done)

    window = webview.create_window(
        "EvoScientist",
        html=_status_html("Starting EvoScientist…"),
        width=1280,
        height=860,
        # pywebview defaults text_select=False, which injects
        # ``body { user-select: none }`` into every page — that makes the
        # error/status panels (and the WebUI) impossible to select or copy.
        # Enable selection so a user can copy an error message.
        text_select=True,
        js_api=setup_api,
    )
    win = _WebviewWindow(window)
    # The controller owns the live launcher (``switch_workspace`` replaces it),
    # so read the current backend/webui through ``controller.launcher`` — never a
    # cached launcher reference, which goes stale after a switch.
    state: dict = {"controller": None}
    switch_lock = threading.Lock()
    # Tripped by the banner's Cancel button (via the exposed bridge below) to
    # abort a pending switch's wait; cleared at the start of each switch.
    switch_cancel = threading.Event()

    def cancel_workspace_switch() -> None:
        """JS bridge, exposed as ``window.pywebview.api.cancel_workspace_switch``.
        The waiting banner's Cancel button trips this so the user can stop
        waiting for in-flight work and stay in the current workspace."""
        logger.info("workspace switch cancel requested from banner")
        switch_cancel.set()

    window.expose(cancel_workspace_switch)

    def _shutdown() -> None:
        cancelled.set()
        setup_done.set()  # release a boot thread blocked waiting on setup
        controller = state["controller"]
        if controller is not None and controller.launcher is not None:
            controller.launcher.stop()

    def _on_closing() -> bool:
        """pywebview ``closing`` handler (fires before close; returning False
        vetoes it). Confirm before tearing down a backend WE own that is still
        busy — closing then kills its in-flight runs and background sub-agents.
        Returns True (allow close) in every other case, including any probe
        error, so a flaky check never traps the user in an unclosable window.
        """
        controller = state["controller"]
        if controller is None or controller.launcher is None:
            return True
        launcher = controller.launcher
        try:
            from .shutdown import backend_has_active_runs, should_confirm_close

            active = backend_has_active_runs(
                launcher.backend_url, watched_thread_id=win.current_thread_id()
            )
            if not should_confirm_close(launcher.backend_started, active):
                return True
            ok = bool(
                window.create_confirmation_dialog(
                    "Quit EvoScientist?",
                    "Research tasks are still running. Quit and stop them?",
                )
            )
            logger.info(
                "close with active tasks: user chose %s", "quit" if ok else "stay"
            )
            return ok
        except Exception as exc:  # never trap the user in an unclosable window
            logger.warning("close-confirm check failed: %s", exc)
            return True

    def _switch_workspace() -> None:
        """Native "Workspace > Switch Workspace…" menu action.

        Picks a folder, then hands the switch to the controller on a background
        thread — the switch waits for in-flight runs to finish before restarting
        the backend, which must not block the GUI thread. The menu is built
        before the boot thread creates the controller, so clicks that arrive
        before services are up are ignored.
        """
        controller = state["controller"]
        if controller is None:
            logger.info("switch_workspace requested before services ready; ignoring")
            return
        current = controller.launcher.workspace_dir
        # Read the watched thread before the picker steals focus, so the switch
        # gates on the turn the user is on.
        watched = win.current_thread_id()
        try:
            import webview

            selection = window.create_file_dialog(
                webview.FileDialog.FOLDER, directory=current
            )
        except Exception as exc:  # a picker failure must not crash the menu
            logger.warning("workspace picker failed: %s", exc)
            return
        target = _first_path(selection)
        if target is None:
            logger.info("workspace switch cancelled at picker")
            return
        if _same_dir(target, current):
            logger.info("workspace switch to the current workspace; ignoring")
            return
        # Serialize switches: a second click while one is running is a no-op.
        if not switch_lock.acquire(blocking=False):
            logger.info("workspace switch already in progress; ignoring")
            return
        switch_cancel.clear()  # fresh wait; drop any leftover cancel from before

        def _run() -> None:
            try:
                controller.switch_workspace(target, watched_thread_id=watched)
            finally:
                switch_lock.release()

        threading.Thread(
            target=_run, name="evosci-workspace-switch", daemon=True
        ).start()

    # ``closing`` gates the close (confirm on active tasks); ``closed`` does the
    # actual idempotent teardown once the close is allowed to proceed.
    window.events.closing += _on_closing
    window.events.closed += _shutdown

    def make_launcher(ws: str | None) -> WebUILauncher:
        """Build a launcher pinned to workspace ``ws``. Used for the initial
        boot and, as the controller's ``launcher_factory``, for each workspace
        switch — so a switch reuses the exact same launch configuration, only
        the workspace changing. Reads the current ``config`` (which the setup
        step below may have re-resolved).

        Desktop shell: no terminal to act on a port conflict, so ``auto_port``
        falls back to a free port instead of dead-ending at the error panel.
        """
        cfg = build_launcher_config(config, ws, auto_port=True)
        runner = BundledWebUIRunner(
            app_dir=app_paths.webui_dir(),
            node_exe=app_paths.node_exe(),
            log_path=app_paths.webui_log_path(),
        )
        return WebUILauncher(config, cfg, runner)

    def boot() -> bool:
        nonlocal config
        if setup_needed(config):
            win.show_setup(
                render_setup_html(
                    provider=config.provider,
                    model=config.model,
                    workspace=config.default_workdir,
                )
            )
            setup_done.wait()
            if cancelled.is_set():
                return False
            # Re-resolve so the saved provider/model/key/workspace take effect.
            config = get_effective_config()
            apply_config_to_env(config)

        controller = DesktopController(
            make_launcher(workspace_dir),
            win,
            launcher_factory=make_launcher,
            # Abort a pending switch either on app shutdown or on a Cancel click.
            should_cancel=lambda: cancelled.is_set() or switch_cancel.is_set(),
        )
        state["controller"] = controller
        return controller.boot()

    from webview.menu import Menu, MenuAction

    app_menu = [
        Menu("Workspace", [MenuAction("Switch Workspace…", _switch_workspace)]),
    ]
    try:
        webview.start(boot, menu=app_menu)
    finally:
        _shutdown()
