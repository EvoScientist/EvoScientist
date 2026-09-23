"""Boot/shutdown orchestration for the desktop shell.

Deliberately GUI-free so it can be unit-tested without a real window: it drives
a :class:`DesktopWindow` (any object with ``show_status`` / ``load_url`` /
``show_error``) and a launcher (any object with ``start`` / ``wait_ready`` /
``stop``). The real pywebview glue lives in :mod:`EvoScientist.desktop.shell`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
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

    def __init__(
        self,
        launcher,
        window: DesktopWindow,
        ready_timeout: float = 90.0,
        *,
        launcher_factory: Callable[[str], object] | None = None,
        busy_probe: Callable[[str], str] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = 1.0,
    ):
        self._launcher = launcher
        self._window = window
        self._ready_timeout = ready_timeout
        # ``switch_workspace`` needs to build a launcher for the new workspace;
        # the shell injects the same closure it uses for the initial boot. Left
        # None in tests that never switch (and the switch then no-ops).
        self._launcher_factory = launcher_factory
        self._busy_probe = busy_probe
        self._should_cancel = should_cancel
        self._sleep = sleep
        self._poll_interval = poll_interval

    @property
    def launcher(self):
        """The live launcher. Single source of truth: ``switch_workspace``
        replaces it, so the shell must read the current backend/webui through
        this rather than caching the launcher it saw at boot."""
        return self._launcher

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

    def switch_workspace(self, workspace_dir: str) -> None:
        """Switch the active workspace to ``workspace_dir`` (issue #484 Part 6).

        Single active workspace, controlled restart: wait for the app-owned
        backend to finish every active run (including background sub-agents),
        then tear it down and boot a fresh launcher pinned to the new workspace,
        reloading the WebUI. No confirm dialog — the wait, not a prompt, is how
        in-flight work is protected. Follows the desktop's existing restart path
        (stop launcher -> rebuild -> boot), so moving to #413 Phase 1 later only
        changes what this method calls.

        The single choke point the switch flows through: the native menu drives
        it today, and a future in-WebUI switcher drives the same method.
        """
        if self._launcher_factory is None:
            logger.info("switch_workspace requested but no launcher factory; ignoring")
            return

        from .shutdown import _probe_busy_state, wait_for_backend_idle

        backend_url = self._launcher.backend_url
        logger.info("workspace switch requested -> %s", workspace_dir)

        probe = self._busy_probe or _probe_busy_state
        if probe(backend_url) != "idle":
            self._window.show_status(
                "Waiting for running tasks to finish before switching workspace…"
            )
        idle = wait_for_backend_idle(
            backend_url,
            probe=probe,
            sleep=self._sleep,
            poll_interval=self._poll_interval,
            should_cancel=self._should_cancel,
        )
        if not idle:
            # Cancelled — the app is shutting down. Do not relaunch a backend.
            logger.info("workspace switch aborted before restart (cancelled)")
            return

        self._window.show_status("Switching workspace…")
        self._launcher.stop()
        self._launcher = self._launcher_factory(workspace_dir)
        self.boot()

    def shutdown(self) -> None:
        """Tear down the services the launcher started (idempotent)."""
        self._launcher.stop()
