"""Boot/shutdown orchestration for the desktop shell.

Deliberately GUI-free so it can be unit-tested without a real window: it drives
a :class:`DesktopWindow` (any object with ``show_status`` / ``load_url`` /
``show_error``) and a launcher (any object with ``start`` / ``wait_ready`` /
``stop``). The real pywebview glue lives in :mod:`EvoScientist.desktop.shell`.
"""

from __future__ import annotations

import logging
from typing import Protocol

from ..deploy.launcher import LauncherError

logger = logging.getLogger("EvoScientist.desktop")


class DesktopWindow(Protocol):
    """The window surface the controller needs. The real implementation wraps a
    pywebview window; tests pass a recording fake."""

    def show_status(self, message: str) -> None: ...

    def load_url(self, url: str) -> None: ...

    def show_error(self, code: str, message: str, detail: str | None) -> None: ...


class DesktopController:
    """Runs the startup sequence and owns teardown.

    ``boot`` never raises: every failure is turned into ``window.show_error`` so
    the window shows an actionable message instead of a blank page. ``shutdown``
    is idempotent (the launcher's ``stop`` is).
    """

    def __init__(self, launcher, window: DesktopWindow, ready_timeout: float = 90.0):
        self._launcher = launcher
        self._window = window
        self._ready_timeout = ready_timeout

    def boot(self) -> bool:
        """Start services and load the WebUI. Returns True on success."""
        try:
            self._window.show_status("Starting EvoScientist services…")
            self._launcher.start()
            self._window.show_status("Waiting for the WebUI to become ready…")
            result = self._launcher.wait_ready(self._ready_timeout)
            self._window.load_url(result.webui_url)
            logger.info("boot ok: webui=%s", result.webui_url)
            return True
        except LauncherError as exc:
            logger.error("boot failed [%s]: %s", exc.code, exc.message)
            self._window.show_error(exc.code, exc.message, exc.detail)
            return False
        except Exception as exc:  # never leave a blank window on an unexpected error
            logger.exception("boot failed unexpectedly")
            self._window.show_error("unexpected", str(exc), None)
            return False

    def switch_workspace(self) -> None:
        """Switch the active workspace (stub).

        Wired to the native "Workspace > Switch Workspace…" menu action so the
        desktop surface and the trigger seam exist now. The real behavior
        depends on the backend workspace contract (issue #488) and is not
        implemented yet: this is the single choke point the switch logic will
        land in, so the WebUI can later drive the same path.
        """
        logger.info("switch_workspace requested (not implemented yet)")

    def shutdown(self) -> None:
        """Tear down the services the launcher started (idempotent)."""
        self._launcher.stop()
