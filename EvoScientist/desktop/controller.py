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

    # A non-blocking banner over the live WebUI (unlike ``show_status``, which
    # replaces the whole page). Used while a switch waits for the current turn
    # to finish, so the user can keep watching and answering it.
    def show_pending(self, message: str) -> None: ...

    def clear_pending(self) -> None: ...


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
        active_probe: Callable[[str], str] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        should_proceed_now: Callable[[], bool] | None = None,
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
        self._active_probe = active_probe
        self._should_cancel = should_cancel
        # The pending banner's "Stop tasks and switch now" button: ends the wait
        # and proceeds (the restart then kills the running tasks).
        self._should_proceed_now = should_proceed_now
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

    def switch_workspace(
        self, workspace_dir: str, watched_thread_id: str | None = None
    ) -> None:
        """Switch the active workspace to ``workspace_dir`` (issue #484 Part 6).

        Single active workspace, controlled restart: wait for the app-owned
        backend to finish every active turn, then tear it down and boot a fresh
        launcher pinned to the new workspace, reloading the WebUI. No confirm
        dialog: the wait, not a prompt, is how in-flight work is protected, and
        while it waits a non-blocking banner keeps the WebUI visible so the user
        can watch and answer the current turn. Follows the desktop's existing
        restart path (stop launcher -> rebuild -> boot), so moving to #413 Phase
        1 later only changes what this method calls.

        ``watched_thread_id`` is the thread the user has open in the WebUI (read
        from the URL by the shell). The wait blocks on any busy thread anywhere,
        but on an ``interrupted`` (HITL) thread only when it is this watched one
        — a stale interrupted turn the user is not looking at must not block the
        switch (interrupted turns are saved and resumable, and they accumulate).

        The single choke point the switch flows through: the native menu drives
        it today, and a future in-WebUI switcher drives the same method.
        """
        if self._launcher_factory is None:
            logger.info("switch_workspace requested but no launcher factory; ignoring")
            return

        from .shutdown import (
            _probe_active_state,
            running_bg_process_names,
            wait_for_backend_idle,
        )

        backend_url = self._launcher.backend_url
        logger.info(
            "workspace switch requested -> %s (watched thread %s)",
            workspace_dir,
            watched_thread_id or "none",
        )

        probe = self._active_probe or (
            lambda u: _probe_active_state(u, watched_thread_id=watched_thread_id)
        )
        waiting = probe(backend_url) != "idle"
        if waiting:
            # Keep the WebUI visible (non-blocking banner) so the user can keep
            # working the current turn; the switch completes once it finishes, or
            # the user stops the running tasks and switches now via the banner.
            names = running_bg_process_names(backend_url)
            if names:
                msg = (
                    "Switching workspace once running tasks finish: "
                    f"{', '.join(names)}. Or stop them and switch now."
                )
            else:
                msg = "Switching workspace once the current task finishes…"
            self._window.show_pending(msg)
        idle = wait_for_backend_idle(
            backend_url,
            probe=probe,
            sleep=self._sleep,
            poll_interval=self._poll_interval,
            should_cancel=self._should_cancel,
            should_proceed_now=self._should_proceed_now,
        )
        if not idle:
            # Cancelled — the app is shutting down. Do not relaunch a backend.
            if waiting:
                self._window.clear_pending()
            logger.info("workspace switch aborted before restart (cancelled)")
            return

        # ``show_status`` replaces the whole page, so the banner goes with it;
        # clearing first keeps the surface honest if the backend was idle from
        # the start (no banner shown) or the WebUI lingers a moment.
        if waiting:
            self._window.clear_pending()
        self._window.show_status("Switching workspace…")
        self._launcher.stop()
        self._launcher = self._launcher_factory(workspace_dir)
        self.boot()

    def shutdown(self) -> None:
        """Tear down the services the launcher started (idempotent)."""
        self._launcher.stop()
