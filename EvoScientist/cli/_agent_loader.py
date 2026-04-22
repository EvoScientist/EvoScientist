"""Background MCP/agent load lifecycle shared by CLI and TUI surfaces.

Holds no references to Rich, prompt_toolkit, or Textual — UI-specific
rendering and thread-hopping plug in via callbacks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

_logger = logging.getLogger(__name__)

ProgressEvent = str  # "start" | "success" | "error"
ProgressState = str  # "pending" | "ok" | "error"

ProgressCallback = Callable[[ProgressEvent, str, str], None]
SuccessCallback = Callable[[Any], None]
FailureCallback = Callable[[BaseException], None]


class MCPProgressTracker:
    """Per-server MCP load progress state.

    Reads and writes are GIL-atomic but iteration must go through
    :meth:`snapshot` — events can fire from a worker thread while the
    main thread renders.
    """

    __slots__ = ("progress",)

    def __init__(self) -> None:
        self.progress: dict[str, tuple[ProgressState, str]] = {}

    def prime(self) -> None:
        """Seed a ``pending`` entry for every configured server.

        Keeps the UI's "N / M" denominator stable from the first render.
        """
        try:
            from ..mcp import load_mcp_config

            cfg = load_mcp_config() or {}
            self.progress = dict.fromkeys(cfg, ("pending", ""))
        except Exception:
            self.progress = {}

    def record(
        self, event: ProgressEvent, server: str, detail: str
    ) -> ProgressState | None:
        """Apply an event and return the new state, or ``None`` if unknown."""
        if event == "start":
            self.progress.setdefault(server, ("pending", ""))
            return "pending"
        if event == "success":
            self.progress[server] = ("ok", detail)
            return "ok"
        if event == "error":
            self.progress[server] = ("error", detail)
            return "error"
        return None

    def snapshot(self) -> list[tuple[ProgressState, str]]:
        return list(self.progress.values())

    def totals(self) -> tuple[int, int]:
        """``(done, total)`` — done excludes ``pending``."""
        snap = self.snapshot()
        total = len(snap)
        done = sum(1 for state, _ in snap if state != "pending")
        return done, total


class BackgroundAgentLoader:
    """Owns the background ``_load_agent`` task and its generation token.

    Each :meth:`start` bumps an internal id; callbacks from a superseded
    load (the old worker thread keeps running after cancel, since
    ``asyncio.to_thread`` can't preempt arbitrary Python code) compare
    against it and drop silently.

    ``on_progress`` fires on the **worker thread**; UI callers hop
    threads inside it if needed.  ``on_success`` / ``on_failure`` fire
    on the event loop when the task completes.
    """

    def __init__(
        self,
        loader_fn: Callable[..., Any],
        *,
        on_progress: ProgressCallback | None = None,
        on_success: SuccessCallback | None = None,
        on_failure: FailureCallback | None = None,
    ) -> None:
        self._loader_fn = loader_fn
        self._on_progress = on_progress
        self._on_success = on_success
        self._on_failure = on_failure
        self.agent: Any = None
        self._task: asyncio.Task | None = None
        self._load_id: int = 0

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    @property
    def is_pending(self) -> bool:
        return self.agent is None and self._task is not None and not self._task.done()

    @property
    def needs_restart(self) -> bool:
        """True when no load is in flight and no agent is ready.

        Callers that want auto-retry behavior (e.g. TUI on the next
        user send after a failure) check this before :meth:`start`.
        """
        return self.agent is None and (self._task is None or self._task.done())

    def start(self, **loader_kwargs: Any) -> None:
        prev = self._task
        if prev is not None and not prev.done():
            prev.cancel()
        self._load_id += 1
        load_id = self._load_id
        self.agent = None

        def _gated_progress(event: str, server: str, detail: str) -> None:
            if load_id != self._load_id:
                return
            if self._on_progress is None:
                return
            try:
                self._on_progress(event, server, detail)
            except Exception:
                _logger.debug("MCP progress callback raised", exc_info=True)

        self._task = asyncio.create_task(
            asyncio.to_thread(
                self._loader_fn,
                on_mcp_progress=_gated_progress,
                **loader_kwargs,
            )
        )
        self._task.add_done_callback(lambda task, lid=load_id: self._on_done(task, lid))

    def adopt(self, agent: Any) -> None:
        """Install an externally-built agent and supersede any in-flight load.

        Used by ``/model`` (and any other caller that constructs a
        replacement agent directly): bumps the generation token so a
        late-arriving background load can't clobber ``self.agent`` via
        the done-callback, cancels the in-flight wrapper, and seats the
        new agent immediately.
        """
        prev = self._task
        if prev is not None and not prev.done():
            prev.cancel()
        self._load_id += 1
        self._task = None
        self.agent = agent

    async def await_ready(self) -> Any:
        """Return the loaded agent; re-raises on load failure.

        Idempotent.  State transitions (setting ``self.agent``, calling
        ``on_success`` / ``on_failure``) are handled exclusively by
        :meth:`_on_done`, which fires before this ``await`` resumes
        (asyncio guarantees done-callbacks run in registration order).
        """
        if self.agent is not None:
            return self.agent
        if self._task is None:
            raise RuntimeError(
                "BackgroundAgentLoader.await_ready called before start()"
            )
        await self._task
        return self.agent

    def _on_done(self, task: asyncio.Task, load_id: int) -> None:
        if load_id != self._load_id:
            return
        if task.cancelled():
            return
        try:
            self.agent = task.result()
        except Exception as exc:
            # Keep ``_task`` set so a later ``await_ready`` re-raises the
            # real exception instead of the "before start()" sentinel.
            self.agent = None
            if self._on_failure is not None:
                self._on_failure(exc)
            return
        if self._on_success is not None:
            self._on_success(self.agent)
