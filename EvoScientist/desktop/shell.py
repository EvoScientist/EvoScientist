"""pywebview glue for the desktop shell.

Keeps the GUI-specific code (creating the window, rendering status/error HTML,
wiring the close event) thin and separate from the boot logic in
:mod:`EvoScientist.desktop.controller`. ``webview`` is imported lazily so the
rest of the package imports without the optional ``desktop`` extra installed.
"""

from __future__ import annotations

import html

from .controller import DesktopController

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
    detail_block = f"<p class=detail>{html.escape(detail)}</p>" if detail else ""
    return (
        f"<!doctype html><meta charset=utf-8><style>{_PAGE_STYLE}</style>"
        f"<div class=card><h1>EvoScientist could not start</h1>"
        f"<p>{html.escape(message)}</p>{detail_block}"
        f"<p class=code>{html.escape(code)}</p></div>"
    )


class _WebviewWindow:
    """Adapts a pywebview window to the controller's ``DesktopWindow`` surface."""

    def __init__(self, window) -> None:
        self._window = window

    def show_status(self, message: str) -> None:
        self._window.load_html(_status_html(message))

    def load_url(self, url: str) -> None:
        self._window.load_url(url)

    def show_error(self, code: str, message: str, detail: str | None) -> None:
        self._window.load_html(_error_html(code, message, detail))


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

    config = get_effective_config()
    apply_config_to_env(config)
    cfg = build_launcher_config(config, workspace_dir)
    runner = BundledWebUIRunner(
        app_dir=app_paths.webui_dir(), node_exe=app_paths.node_exe()
    )
    launcher = WebUILauncher(config, cfg, runner)

    window = webview.create_window(
        "EvoScientist",
        html=_status_html("Starting EvoScientist…"),
        width=1280,
        height=860,
    )
    controller = DesktopController(launcher, _WebviewWindow(window))
    # Confirm-before-interrupt on close is a later reliability task; for now the
    # close handler tears down only the processes we started.
    window.events.closed += controller.shutdown

    try:
        webview.start(controller.boot)
    finally:
        controller.shutdown()
