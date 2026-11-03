"""A Windows job object that ends a backend together with ``EvoSci deploy --json``.

Node (the desktop app) puts its children in a job object that ends them when
Node exits or is killed. ``deploy --json`` then ends within seconds, often
before it has stopped its backend, and the backend, which runs outside that
job, would keep running with nobody to stop it. The backend joins a job that
only this process holds, with ``KILL_ON_JOB_CLOSE``: Windows closes the handle
when this process ends however it ends, and that ends the backend and
everything it started. A ccproxy this command starts joins the same job; one
reused from another session does not.

The handle is closed only after ``stop_langgraph_dev()`` and ``stop_ccproxy()``
have returned, so the job never ends a process this command still means to
run or stop itself.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from ..agent_shell import _CREATE_SUSPENDED, _job_api

logger = logging.getLogger(__name__)


class BackendJob:
    """The job for one backend: :attr:`creationflags` for ``Popen``, then
    :meth:`started` once the process exists, :meth:`close` after the stop.

    Never raises. When the job cannot be created or a process cannot join
    it, :attr:`warning` names the processes that run without it.
    """

    def __init__(self) -> None:
        self._handle: int | None = None
        self._error: int | None = None
        # The processes this command started outside the job, for the warning.
        self._missing: list[str] = []
        try:
            api = _job_api()
            job = api.create()
            if not job:
                self._error = api.last_error()
            elif not api.kill_on_close(job):
                self._error = api.last_error()
                api.close(job)
            else:
                self._handle = job
        except (OSError, AttributeError) as exc:
            # No kernel32 job API (ctypes could not load it).
            logger.debug(f"Job object API unavailable: {exc}")
            self._error = -1

    @property
    def creationflags(self) -> int:
        """Start the backend suspended, so it joins the job before it can
        start a worker that would escape it."""
        return _CREATE_SUSPENDED if self._handle is not None else 0

    def started(self, process: subprocess.Popen) -> None:
        """Put the suspended ``process`` in the job, then let it run. It is
        resumed whatever happens."""
        if self._handle is None:
            self._missing.append("the backend")
            return
        api = _job_api()
        try:
            if not api.assign(self._handle, int(process._handle)):
                self._failed("the backend", api.last_error())
        finally:
            api.resume(int(process._handle))

    def adopt(self, process: subprocess.Popen) -> None:
        """Put an already running ``process`` in the job (ccproxy). It needs
        no suspended start: what matters is that it ends with the command."""
        if self._handle is None:
            self._missing.append("ccproxy")
            return
        api = _job_api()
        if not api.assign(self._handle, int(process._handle)):
            self._failed("ccproxy", api.last_error())

    def _failed(self, what: str, error: int) -> None:
        self._missing.append(what)
        if self._error is None:
            self._error = error

    @property
    def warning(self) -> str | None:
        """Names every process this command started outside the job."""
        if not self._missing:
            return None
        what = " and ".join(self._missing)
        keeps = "keeps" if len(self._missing) == 1 else "keep"
        return (
            f"Could not put {what} in a Windows job object (error "
            f"{self._error}). If the app that started this command exits "
            f"without closing its stdin, or is killed, {what} {keeps} running."
        )

    def close(self) -> None:
        """Close the handle, which ends whatever is still in the job. Call it
        only after the backend was stopped. Safe to call more than once."""
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        _job_api().close(handle)


def backend_job() -> BackendJob | None:
    """A :class:`BackendJob` on Windows, None elsewhere: on macOS and Linux
    the app's exit closes our stdin, and the stdin watcher stops the backend."""
    if sys.platform != "win32":
        return None
    return BackendJob()
