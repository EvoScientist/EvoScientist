"""pywebview glue for the desktop shell.

Keeps the GUI-specific code (creating the window, rendering status/error HTML,
wiring the close event) thin and separate from the boot logic in
:mod:`EvoScientist.desktop.controller`. ``webview`` is imported lazily so the
rest of the package imports without the optional ``desktop`` extra installed.
"""

from __future__ import annotations

import html
import logging
import threading

from .controller import DesktopController
from .setup import apply_setup, render_setup_html, setup_needed, validate_setup

logger = logging.getLogger("EvoScientist.desktop")


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
    state: dict = {"launcher": None}

    def _shutdown() -> None:
        cancelled.set()
        setup_done.set()  # release a boot thread blocked waiting on setup
        launcher = state["launcher"]
        if launcher is not None:
            launcher.stop()

    def _on_closing() -> bool:
        """pywebview ``closing`` handler (fires before close; returning False
        vetoes it). Confirm before tearing down a backend WE own that is still
        busy — closing then kills its in-flight runs and background sub-agents.
        Returns True (allow close) in every other case, including any probe
        error, so a flaky check never traps the user in an unclosable window.
        """
        launcher = state["launcher"]
        if launcher is None:
            return True
        try:
            from .shutdown import backend_has_active_runs, should_confirm_close

            active = backend_has_active_runs(launcher.backend_url)
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

    # ``closing`` gates the close (confirm on active tasks); ``closed`` does the
    # actual idempotent teardown once the close is allowed to proceed.
    window.events.closing += _on_closing
    window.events.closed += _shutdown

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

        # Desktop shell: no terminal to act on a port conflict, so fall back to
        # a free port instead of dead-ending at the error panel.
        cfg = build_launcher_config(config, workspace_dir, auto_port=True)
        runner = BundledWebUIRunner(
            app_dir=app_paths.webui_dir(),
            node_exe=app_paths.node_exe(),
            log_path=app_paths.webui_log_path(),
        )
        launcher = WebUILauncher(config, cfg, runner)
        state["launcher"] = launcher
        return DesktopController(launcher, win).boot()

    try:
        webview.start(boot)
    finally:
        _shutdown()
