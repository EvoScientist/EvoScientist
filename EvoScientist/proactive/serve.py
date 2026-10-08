"""Serve's side of proactive pushes: the cron's owner and the only writer.

Serve holds the channel origins, so it decides which threads the cron may
consider (the armed threads it can deliver to), keeps exactly one proactive
cron per workspace while any thread is armed, and applies the pushes the cron
decides. :class:`ProactiveServe` is that controller. Its methods are blocking
(they call the dev server through the sync SDK client) and serialised by one
lock, because serve's poll loop and its slash commands run on different
threads.

The cron is recreated, never kept, whenever serve (re)takes ownership of it:
after a dev-server restart with file persistence on, every tick of a kept
thread-bound cron fails (the stored payload comes back with an auth user that
does not serialise).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..paths import Workspace
from .commit import build_proactive_tag, proactive_message_id
from .cron import (
    PROACTIVE_RUN_KIND,
    PROACTIVE_SCHEDULE,
    create_proactive_cron,
    create_proactive_thread,
    cron_input,
    list_proactive_crons,
)

logger = logging.getLogger(__name__)

# How often serve reads the cron thread for new decisions.
POLL_INTERVAL_SECONDS = 20.0
# No tick for this long (2.5 schedule intervals) means the cron has stopped:
# serve recreates it. Shorter than two intervals would recreate a cron whose
# tick is merely slow.
STALE_TICK_SECONDS = 25 * 60
# The graph whose threads serve writes pushes into.
MAIN_GRAPH_ID = "EvoScientist"
# Node of the main graph a push is a reply of, for the fallback write.
_MODEL_NODE = "model"


@dataclass
class _Topology:
    """What the push write needs from the main graph's shape."""

    terminal_node: str | None
    after_model: frozenset[str]


def terminal_node(graph: dict[str, Any]) -> str | None:
    """The node whose only edge is an unconditional edge to ``__end__``, if unique.

    *graph* is ``assistants.get_graph`` output. An unconditional edge has no
    ``conditional`` key at all. The node set depends on config (HITL, the code
    interpreter), so the node is found by this rule, never by name.
    """
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges") or []:
        outgoing.setdefault(str(edge.get("source")), []).append(edge)
    found = [
        source
        for source, edges in outgoing.items()
        if source != "__start__"
        and len(edges) == 1
        and edges[0].get("target") == "__end__"
        and edges[0].get("conditional") is not True
    ]
    return found[0] if len(found) == 1 else None


def _successors(graph: dict[str, Any], node: str) -> frozenset[str]:
    return frozenset(
        str(edge.get("target"))
        for edge in graph.get("edges") or []
        if edge.get("source") == node
    )


def _checkpoint_id(state: Any) -> str | None:
    checkpoint = state.get("checkpoint") if isinstance(state, dict) else None
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id"):
        return str(checkpoint["checkpoint_id"])
    return None


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


@dataclass
class ProactiveServe:
    """Serve's proactive controller for the workspace it currently serves.

    ``unavailable_reason`` is set on the ``local`` backend and under #470's
    fallback; every method is then a no-op, and ``/proactive`` reports the
    reason.
    """

    workspace: Workspace
    client: Any = None
    timezone: str | None = None
    unavailable_reason: str | None = None
    # Threads armed in this serve session. Thread metadata holds the same flag
    # for the cron and for a restart.
    armed: set[str] = field(default_factory=set)
    # Delivery: (thread_id, text) -> delivered.
    publish: Callable[[str, str], bool] | None = None
    # Whether serve can deliver to a thread now (it has a channel origin).
    can_deliver: Callable[[str], bool] | None = None
    _cron_id: str | None = None
    _cron_thread_id: str | None = None
    _cron_created_at: float = 0.0
    _applied: set[str] = field(default_factory=set)
    _topology: _Topology | None = None
    _last_poll: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and self.client is not None

    # ---- candidates and the cron -----------------------------------------------

    def candidates(self) -> list[str]:
        """Armed threads serve can deliver to, in a stable order."""
        deliverable = self.can_deliver or (lambda _tid: False)
        return sorted(tid for tid in self.armed if deliverable(tid))

    def reconcile(self) -> None:
        """Recreate the workspace's proactive cron, or delete it when nothing is armed.

        Deletes every proactive cron of the workspace, unblocks the cron thread
        (a tick killed with SIGKILL stays ``running`` and makes every later
        fire fail with 409), and, while a thread is armed, creates one cron on
        the cron thread with the current candidates.
        """
        if not self.available:
            return
        with self._lock:
            for cron in list_proactive_crons(self.client, self.workspace):
                self.client.crons.delete(cron["cron_id"])
            self._cron_id = None
            if not self.armed:
                return
            thread_id = self._ensure_cron_thread()
            self._cancel_running(thread_id)
            cron = create_proactive_cron(
                self.client,
                self.workspace,
                thread_id=thread_id,
                candidates=self.candidates(),
                timezone=self.timezone,
            )
            self._cron_id = str(cron["cron_id"])
            self._cron_created_at = time.time()
            logger.info(
                "[proactive] cron %s on thread %s (%s, %d candidate(s))",
                self._cron_id,
                thread_id,
                PROACTIVE_SCHEDULE,
                len(self.candidates()),
            )

    def mirror(self) -> None:
        """Send the current candidates to the cron's input (the whole input).

        Idempotent. Creates the cron when a thread is armed and none exists.
        """
        if not self.available:
            return
        with self._lock:
            if self._cron_id is None:
                if self.armed:
                    self.reconcile()
                return
            self.client.crons.update(self._cron_id, input=cron_input(self.candidates()))

    def stop(self) -> None:
        """Delete the workspace's proactive crons (serve stops or moves away)."""
        if not self.available:
            return
        with self._lock:
            for cron in list_proactive_crons(self.client, self.workspace):
                self.client.crons.delete(cron["cron_id"])
            self._cron_id = None

    def move_to(self, workspace: Workspace) -> None:
        """Follow serve into another workspace: its cron belongs to the new one."""
        with self._lock:
            self.workspace = workspace
            self._cron_thread_id = None
            self._applied.clear()

    def _ensure_cron_thread(self) -> str:
        """The cron thread: the one already used in this workspace, else a new one.

        Reusing it keeps the record of what each thread was last evaluated on
        across serve restarts.
        """
        from langgraph_sdk.errors import NotFoundError

        if self._cron_thread_id is not None:
            try:
                self.client.threads.get(self._cron_thread_id)
                return self._cron_thread_id
            except NotFoundError:
                self._cron_thread_id = None
        metadata = {"run_kind": PROACTIVE_RUN_KIND, "workspace_dir": self.workspace.key}
        for thread in self.client.threads.search(metadata=metadata, limit=1) or []:
            self._cron_thread_id = str(thread["thread_id"])
            return self._cron_thread_id
        self._cron_thread_id = create_proactive_thread(self.client, self.workspace)
        return self._cron_thread_id

    def _cancel_running(self, thread_id: str) -> None:
        for run in self.client.runs.list(thread_id, status="running") or []:
            logger.warning(
                "[proactive] cancelling run %s left running on the cron thread",
                run.get("run_id"),
            )
            self.client.runs.cancel(thread_id, run["run_id"], action="interrupt")

    # ---- the poll ----------------------------------------------------------

    def poll(
        self,
        *,
        current_thread_id: str,
        switch_to: Callable[[str], None] | None = None,
        now: float | None = None,
    ) -> None:
        """Check the cron's health and apply new decisions; throttled.

        Called from serve's idle branch. *switch_to* makes a thread serve's
        current thread, for a push applied to a thread that is not current.
        """
        if not self.available or not self.armed:
            return
        now = time.time() if now is None else now
        if now - self._last_poll < POLL_INTERVAL_SECONDS:
            return
        self._last_poll = now
        with self._lock:
            try:
                decisions = self._check_cron(now)
            except Exception:
                logger.warning("[proactive] poll failed", exc_info=True)
                return
            for decision in decisions:
                try:
                    self._apply(
                        decision,
                        current_thread_id=current_thread_id,
                        switch_to=switch_to,
                    )
                except Exception:
                    logger.warning(
                        "[proactive] applying push %s failed",
                        decision.get("proactive_id"),
                        exc_info=True,
                    )

    def _check_cron(self, now: float) -> list[dict[str, Any]]:
        """Recreate a missing, failed or stalled cron; return the latest decisions."""
        crons = list_proactive_crons(self.client, self.workspace)
        if not crons or self._cron_thread_id is None:
            logger.warning("[proactive] cron missing; recreating it")
            self.reconcile()
            return []
        thread = self.client.threads.get(self._cron_thread_id)
        if thread.get("status") == "error":
            logger.warning("[proactive] last tick failed; recreating the cron")
            self.reconcile()
            return []
        state = self.client.threads.get_state(self._cron_thread_id)
        last_tick = _parse_time(state.get("created_at")) or 0.0
        if now - max(last_tick, self._cron_created_at) > STALE_TICK_SECONDS:
            logger.warning(
                "[proactive] no tick for %d minutes; recreating the cron",
                STALE_TICK_SECONDS // 60,
            )
            self.reconcile()
            return []
        values = state.get("values") or {}
        return [d for d in values.get("decisions") or [] if isinstance(d, dict)]

    # ---- applying a push ---------------------------------------------------

    def _apply(
        self,
        decision: dict[str, Any],
        *,
        current_thread_id: str,
        switch_to: Callable[[str], None] | None,
    ) -> None:
        from ..gateway.server import _state_interrupts

        proactive_id = str(decision.get("proactive_id") or "")
        thread_id = str(decision.get("thread_id") or "")
        text = str(decision.get("text") or "")
        if not proactive_id or not thread_id or not text:
            return
        if proactive_id in self._applied:
            return
        # Decided once, applied at most once, whatever the outcome below.
        self._applied.add(proactive_id)

        if thread_id not in self.armed:
            return self._skip(proactive_id, thread_id, "disarmed")
        if not (self.can_deliver or (lambda _tid: False))(thread_id):
            return self._skip(proactive_id, thread_id, "no_origin")
        thread = self.client.threads.get(thread_id)
        state = self.client.threads.get_state(thread_id)
        if thread.get("status") in ("busy", "interrupted") or _state_interrupts(state):
            return self._skip(proactive_id, thread_id, "in_flight")
        if _checkpoint_id(state) != decision.get("head"):
            return self._skip(proactive_id, thread_id, "stale")

        self._append(thread_id, proactive_id, text)
        if thread_id != current_thread_id and switch_to is not None:
            switch_to(thread_id)
        publish = self.publish or (lambda _tid, _text: False)
        if publish(thread_id, text):
            logger.info(
                "[proactive] push %s delivered to thread %s (%d chars)",
                proactive_id,
                thread_id,
                len(text),
            )
        else:
            logger.warning(
                "[proactive] push %s written to thread %s but not delivered",
                proactive_id,
                thread_id,
            )

    @staticmethod
    def _skip(proactive_id: str, thread_id: str, reason: str) -> None:
        logger.info(
            "[proactive] push %s for thread %s not applied: %s",
            proactive_id,
            thread_id,
            reason,
        )

    def _append(self, thread_id: str, proactive_id: str, text: str) -> None:
        """Append the push in one write as the graph's terminal node.

        Writing it as ``model`` would leave ``next`` on the node after it, and
        the server would report the thread as interrupted. When the terminal
        node is not unique, fall back to ``model`` plus a clear of exactly
        that post-append state.
        """
        message = {
            "type": "ai",
            "content": text,
            "id": proactive_message_id(proactive_id),
            "additional_kwargs": build_proactive_tag(proactive_id),
        }
        topology = self._main_topology()
        if topology.terminal_node is not None:
            self.client.threads.update_state(
                thread_id, {"messages": [message]}, as_node=topology.terminal_node
            )
            return
        logger.warning(
            "[proactive] main graph has no unique terminal node; appending as "
            "%r and clearing the step after it",
            _MODEL_NODE,
        )
        written = self.client.threads.update_state(
            thread_id, {"messages": [message]}, as_node=_MODEL_NODE
        )
        clear_post_append_next(
            self.client,
            thread_id,
            written_checkpoint_id=_checkpoint_id(written),
            after_model=topology.after_model,
        )

    def _main_topology(self) -> _Topology:
        if self._topology is None:
            graph = self.client.assistants.get_graph(MAIN_GRAPH_ID)
            self._topology = _Topology(
                terminal_node=terminal_node(graph),
                after_model=_successors(graph, _MODEL_NODE),
            )
        return self._topology


def clear_post_append_next(
    client: Any,
    thread_id: str,
    *,
    written_checkpoint_id: str | None,
    after_model: frozenset[str],
) -> bool:
    """Clear ``next`` only if the thread is exactly as the append left it.

    That is: the head is still the checkpoint the append wrote, ``next`` is
    the node(s) after ``model``, and nothing is waiting on an interrupt.
    Anything else (a user turn, a real interrupt) is not ours to touch.
    Stricter than ``_repair_stuck_thread_state``, which clears any non-empty
    ``next`` without an interrupt.
    """
    from ..gateway.server import _state_interrupts

    state = client.threads.get_state(thread_id)
    if (
        written_checkpoint_id is None
        or _checkpoint_id(state) != written_checkpoint_id
        or frozenset(state.get("next") or []) != after_model
        or not after_model
        or _state_interrupts(state)
    ):
        logger.warning(
            "[proactive] thread %s changed after the push was appended; "
            "leaving its state as it is",
            thread_id,
        )
        return False
    client.threads.update_state(thread_id, None, as_node="__end__")
    return True
