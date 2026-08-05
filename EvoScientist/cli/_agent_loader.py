"""Background MCP/agent load lifecycle shared by CLI and TUI surfaces.

Holds no references to Rich, prompt_toolkit, or Textual — UI-specific
rendering and thread-hopping plug in via callbacks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

_logger = logging.getLogger(__name__)

ProgressEvent = str  # "start" | "success" | "error"
ProgressState = str  # "pending" | "ok" | "error"

AgentT = TypeVar("AgentT")
ProgressCallback = Callable[[ProgressEvent, str, str], None]
SuccessCallback = Callable[[AgentT], None]
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


class BackgroundAgentLoader(Generic[AgentT]):
    """Owns the background ``_load_agent`` task and its generation token.

    Each :meth:`start` bumps an internal id; callbacks from a superseded
    load (the old worker thread keeps running after cancel, since
    ``asyncio.to_thread`` can't preempt arbitrary Python code) compare
    against it and drop silently.

    ``on_progress`` fires on the **worker thread**; UI callers hop
    threads inside it if needed.  ``on_success`` / ``on_failure`` fire
    on the event loop when the task completes.

    ``on_failure`` means the session has no agent.  A failed *in-place*
    rebuild is a different event — the previous agent keeps serving — and
    reports through ``on_rebuild_failed`` instead, so a UI can say "couldn't
    pick that up yet" rather than "the load died".

    ``build_token`` makes the seated agent self-invalidating: see
    :meth:`await_ready`.
    """

    def __init__(
        self,
        loader_fn: Callable[..., AgentT],
        *,
        on_progress: ProgressCallback | None = None,
        on_success: SuccessCallback | None = None,
        on_failure: FailureCallback | None = None,
        on_rebuild_started: Callable[[], None] | None = None,
        on_rebuild_failed: FailureCallback | None = None,
        build_token: Callable[[], Any] | None = None,
    ) -> None:
        self._loader_fn = loader_fn
        self._on_progress = on_progress
        self._on_success = on_success
        self._on_failure = on_failure
        self._on_rebuild_started = on_rebuild_started
        self._on_rebuild_failed = on_rebuild_failed
        self._build_token = build_token
        self.agent: AgentT | None = None
        self._task: asyncio.Task[AgentT] | None = None
        self._load_id: int = 0
        # True only while ``_reload`` is driving ``start`` + ``await_ready``.
        # ``_on_done`` fires the fatal ``on_failure`` before ``_reload``'s
        # ``except`` gets a chance to re-seat the previous agent, so without
        # this flag every in-place rebuild failure is reported as a dead
        # session.
        self._reloading: bool = False
        # Kwargs of the most recent ``start``, replayed verbatim by the
        # stale-token reload in ``await_ready``. Everything session-scoped
        # (checkpointer, event sink, workspace, config object) lives in
        # here, so replaying them rebuilds the agent without disturbing
        # the conversation — unlike ``/new``, which rotates the thread.
        self._last_kwargs: dict[str, Any] | None = None
        self._agent_token: Any = None

    @property
    def task(self) -> asyncio.Task[AgentT] | None:
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
        self._last_kwargs = dict(loader_kwargs)
        # Stamped BEFORE the build, not after: a mutation landing mid-build
        # may or may not be captured by it, so the pessimistic stamp costs
        # at most one extra rebuild and never leaves a stale agent seated.
        self._agent_token = self._read_build_token()

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

    def adopt(self, agent: AgentT) -> None:
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
        # The caller built this agent just now, so it already reflects the
        # current inputs; without re-stamping, the next ``await_ready``
        # would rebuild it for a token change it already contains.
        self._agent_token = self._read_build_token()

    def _read_build_token(self) -> Any:
        """Evaluate ``build_token``, treating any failure as "unknown".

        A token that can't be computed must not strand the session in a
        rebuild loop, so it degrades to ``None``. That is not a no-op once a
        real token has been read: ``None`` compares unequal to the hash
        already in ``_agent_token``, so the first raise forces one rebuild,
        after which ``start`` re-stamps ``None`` and it settles.

        Logged at warning because this is the single failure that silently
        turns mid-session expert pickup off entirely — at debug level nobody
        would ever learn why installs stopped taking effect.
        """
        if self._build_token is None:
            return None
        try:
            return self._build_token()
        except Exception:
            _logger.warning(
                "build_token callback raised; agent staleness checks are "
                "disabled until it succeeds again.",
                exc_info=True,
            )
            return None

    async def await_ready(self) -> AgentT:
        """Return the loaded agent; re-raises on load failure.

        Idempotent.  State transitions (setting ``self.agent``, calling
        ``on_success`` / ``on_failure``) are handled exclusively by
        :meth:`_on_done`, which fires before this ``await`` resumes
        (asyncio guarantees done-callbacks run in registration order).

        When a ``build_token`` was supplied and its value has moved since
        the seated agent was built, the agent is transparently rebuilt
        here from the stored ``start`` kwargs.  Every pre-turn agent
        access in both UIs funnels through this method, which is why the
        check lives here rather than in each turn loop: an agent input
        that changes mid-session (a ``skill_manager install`` firing from
        inside a tool call) takes effect on the next turn without the
        user running ``/new`` and losing the conversation.

        A failed rebuild keeps the previous agent seated — a broken
        install must degrade to "the new thing isn't there yet", never to
        a dead session — and reports through ``on_rebuild_failed`` rather
        than the fatal ``on_failure``.
        """
        if self.agent is not None:
            token = self._read_build_token()
            if token != self._agent_token and self._last_kwargs is not None:
                return await self._reload()
            return self.agent
        if self._task is None:
            raise RuntimeError(
                "BackgroundAgentLoader.await_ready called before start()"
            )
        await self._task
        if self.agent is None:
            raise RuntimeError("BackgroundAgentLoader completed without an agent")
        return self.agent

    async def _reload(self) -> AgentT:
        """Rebuild the seated agent in place, falling back to the old one."""
        assert self._last_kwargs is not None
        previous = self.agent
        stale_token = self._agent_token
        # Announced before the await, not after: the rebuild blocks the turn
        # for seconds and the UI is otherwise silent for all of it, which
        # reads as a hang rather than as work.
        if self._on_rebuild_started is not None:
            self._on_rebuild_started()
        self._reloading = True
        try:
            # ``start`` re-reads the token itself, so it is not passed in —
            # doing so would cost a second full skills-tree walk per rebuild
            # for a value only ever used in this log line.
            self.start(**self._last_kwargs)
            _logger.info(
                "Agent inputs changed (build token %r -> %r); rebuilding in place.",
                stale_token,
                self._agent_token,
            )
            return await self.await_ready()
        except Exception as exc:
            # Re-seat the old agent so the session keeps working. Its
            # token stays the failed one, so the rebuild is not retried
            # until the inputs change again — which is also the next
            # chance for the underlying problem to have been fixed.
            _logger.warning(
                "Agent rebuild failed; continuing with the previously loaded "
                "agent. New inputs will not be visible until the next rebuild.",
                exc_info=True,
            )
            self.agent = previous
            self._task = None
            if self._on_rebuild_failed is not None:
                self._on_rebuild_failed(exc)
            return previous  # type: ignore[return-value]
        finally:
            self._reloading = False

    def _on_done(self, task: asyncio.Task[AgentT], load_id: int) -> None:
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
            # During a reload the session is not dead — ``_reload``'s except
            # branch is about to re-seat the previous agent and report via
            # ``on_rebuild_failed``. Firing the fatal callback here would tell
            # the user the load died a moment before the next turn runs fine.
            if self._on_failure is not None and not self._reloading:
                self._on_failure(exc)
            return
        if self._on_success is not None:
            self._on_success(self.agent)
