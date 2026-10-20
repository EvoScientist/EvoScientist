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

import json
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
# After a failed poll serve waits longer, doubling up to this cap.
MAX_POLL_BACKOFF_SECONDS = 10 * 60
# The runtime keeps every run of a thread a cron is bound to (one per tick) and
# never removes them; the cron moves to a fresh thread at least this often.
CRON_THREAD_MAX_AGE_SECONDS = 24 * 60 * 60
# A decision older than this is not applied. Each tick replaces the last one's
# decisions, so a live cron's are at most one interval (10 min) plus a poll
# old; an older one was left in the cron thread while serve was stopped.
DECISION_MAX_AGE_SECONDS = 15 * 60
# ``unavailable_reason`` while the dev server stays behind after a workspace
# move; the next move that succeeds clears it.
MOVE_FAILED_REASON = "the background agent server did not move to this workspace"
# The graph whose threads serve writes pushes into.
MAIN_GRAPH_ID = "EvoScientist"
# Node of the main graph a push is a reply of, for the fallback write.
_MODEL_NODE = "model"

# The channel-origin ``metadata`` keys a reply needs to land in place: Slack
# and Discord threads, Feishu's chat type, QQ's passive-reply handle, personal
# WeChat's context token. Nothing else is stored: the run metadata also
# reaches thread metadata and ``/threads``, and channels put personal data
# (email subjects, nicknames) in the same dict. The chat id is the one
# personal value kept (a Signal number, an email address): a reply needs it.
ORIGIN_METADATA_KEYS = (
    "thread_ts",
    "thread_id",
    "chat_type",
    "msg_type",
    "event_id",
    "context_token",
)


def origin_run_metadata(origin: Any) -> dict[str, str]:
    """Run metadata keys that persist a channel origin across restarts.

    ``origin`` is a ``cli.channel._ChannelOrigin``. The keys avoid the five the
    API strips from request metadata (``thread_id``, ``assistant_id``,
    ``run_id``, ``cron_id``, ``counters_since_delta_snapshot``).
    """
    meta = {
        "origin_channel_type": str(origin.channel_type),
        "origin_chat_id": str(origin.chat_id),
    }
    kept = {
        key: value
        for key, value in (origin.metadata or {}).items()
        if key in ORIGIN_METADATA_KEYS and value is not None
    }
    # Always sent: metadata merges, so a missing key would keep an older reply
    # handle (a Slack thread the chat has since left).
    meta["origin_metadata"] = ""
    if kept:
        try:
            meta["origin_metadata"] = json.dumps(kept, sort_keys=True)
        except (TypeError, ValueError):
            logger.warning(
                "[proactive] channel origin metadata does not encode; a push "
                "after a restart may land outside the chat's thread"
            )
    return meta


def origin_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """The channel origin stored by :func:`origin_run_metadata`, or None."""
    channel_type = row.get("origin_channel_type")
    chat_id = row.get("origin_chat_id")
    if not channel_type or not chat_id:
        return None
    metadata = None
    raw = row.get("origin_metadata")
    if raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                metadata = {
                    k: v for k, v in decoded.items() if k in ORIGIN_METADATA_KEYS
                }
        except ValueError:
            metadata = None
    return {
        "channel_type": str(channel_type),
        "chat_id": str(chat_id),
        # The sender is not stored: delivery needs only the chat.
        "sender": "",
        "metadata": metadata or None,
    }


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


@dataclass(frozen=True)
class LeaveSnapshot:
    """What serve puts back when the move it left *thread_id* for is refused."""

    thread_id: str
    armed: bool
    restored_thread_id: str | None
    # ``origin_of(restored_thread_id)``: ``on_turn`` forgets it.
    restored_origin: Any = None


@dataclass
class ProactiveServe:
    """Serve's proactive controller for the workspace it currently serves.

    ``unavailable_reason`` is set on the ``local`` backend and under #470's
    fallback; ``arm`` and ``disarm`` then raise with it (``/proactive``
    reports it), and every other method is a no-op.
    """

    workspace: Workspace
    client: Any = None
    timezone: str | None = None
    unavailable_reason: str | None = None
    # Armed threads of the served workspace, as far as this serve session
    # knows: set by ``/proactive``, the restore and :meth:`adopt`. Thread
    # metadata holds the flag the cron and a restart read.
    armed: set[str] = field(default_factory=set)
    # Delivery: (thread_id, text) -> delivered.
    publish: Callable[[str, str], bool] | None = None
    # Whether serve can deliver to a thread now (it has a channel origin).
    can_deliver: Callable[[str], bool] | None = None
    # Drop a thread's channel origin (serve stops delivering to it).
    forget_origin: Callable[[str], None] | None = None
    # A thread's channel origin, or None (a ``cli.channel._ChannelOrigin``).
    origin_of: Callable[[str], Any] | None = None
    # Record a thread's channel origin again (a value ``origin_of`` returned).
    remember_origin: Callable[[str, Any], None] | None = None
    # The armed thread restored at startup, until serve's thread changes or
    # its fresh thread has its first turn.
    restored_thread_id: str | None = None
    _cron_id: str | None = None
    _cron_thread_id: str | None = None
    _cron_created_at: float = 0.0
    # The candidate list the cron's input holds, as last sent.
    _mirrored: list[str] | None = None
    _applied: set[str] = field(default_factory=set)
    _topology: _Topology | None = None
    _next_poll: float = 0.0
    _failed_polls: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and self.client is not None

    # ---- arming (``/proactive``) -------------------------------------------------

    def arm(self, thread_id: str) -> None:
        """Arm *thread_id*: store the flag on the thread and give the cron the thread.

        The flag goes to thread metadata at once (a slash command runs no agent
        turn, so it writes no checkpoint); the runtime merges thread metadata
        into every later run, so the flag reaches the checkpoints too. The chat
        goes with it, so a restart before the next turn still finds where to
        deliver. A first arm whose cron cannot be created is undone.
        """
        self._require_available()
        origin = self.origin_of(thread_id) if self.origin_of is not None else None
        metadata = {"proactive": "on"}
        if origin is not None:
            metadata.update(origin_run_metadata(origin))
        with self._lock:
            was_armed = thread_id in self.armed
            self.client.threads.update(thread_id, metadata=metadata)
            self.armed.add(thread_id)
            try:
                self.mirror()
            except Exception:
                if not was_armed:
                    self.armed.discard(thread_id)
                    self.client.threads.update(thread_id, metadata={"proactive": "off"})
                raise

    def disarm(self, thread_id: str) -> None:
        """Disarm *thread_id*; delete the cron when it was the last armed thread.

        Writes ``"off"``: a metadata merge cannot delete the key.
        """
        self._require_available()
        with self._lock:
            self.client.threads.update(thread_id, metadata={"proactive": "off"})
            self.armed.discard(thread_id)
            if thread_id == self.restored_thread_id:
                self.restored_thread_id = None
            if self.armed:
                self.mirror()
            else:
                self.reconcile()

    def leave(self, thread_id: str) -> None:
        """Serve moved off *thread_id*: pause its follow-ups until it comes back.

        ``paused`` keeps a restart from restoring a conversation the user left;
        :meth:`adopt` turns it on again on ``/resume``.
        """
        if not self.available:
            return
        with self._lock:
            if thread_id in self.armed:
                self._pause(thread_id)

    def snapshot_leave(self, thread_id: str) -> LeaveSnapshot:
        """What :meth:`undo_leave` needs, taken before serve leaves *thread_id*."""
        with self._lock:
            restored = self.restored_thread_id
            origin = None
            if restored is not None and self.origin_of is not None:
                origin = self.origin_of(restored)
            return LeaveSnapshot(thread_id, thread_id in self.armed, restored, origin)

    def undo_leave(self, snapshot: LeaveSnapshot) -> None:
        """Serve stays where it was: arm again what :meth:`leave` and :meth:`on_turn` paused.

        Mirrors nothing: the caller mirrors.
        """
        if not self.available:
            return
        with self._lock:
            try:
                self._put_back_restored(snapshot)
            finally:
                if snapshot.armed:
                    self.adopt(snapshot.thread_id)

    def _put_back_restored(self, snapshot: LeaveSnapshot) -> None:
        """Arm the restored thread again and reopen the restore window."""
        restored = snapshot.restored_thread_id
        if restored is None or restored == snapshot.thread_id:
            return
        # Armed again only from its stored flag: a thread deleted meanwhile
        # stays out.
        self.armed.discard(restored)
        self.adopt(restored)
        if restored not in self.armed:
            return
        if snapshot.restored_origin is not None and self.remember_origin:
            self.remember_origin(restored, snapshot.restored_origin)
        self.restored_thread_id = restored

    def _pause(self, thread_id: str) -> None:
        self.client.threads.update(thread_id, metadata={"proactive": "paused"})
        self.armed.discard(thread_id)
        if self.armed:
            self.mirror()
        else:
            self.reconcile()

    def is_armed(self, thread_id: str) -> bool:
        return thread_id in self.armed

    def conversation_thread(self, thread_id: str) -> str:
        """The thread ``/proactive`` acts on when serve's current thread is *thread_id*.

        After a restart the chat's conversation is the restored thread until
        the user talks in serve's fresh thread, so the command acts on it
        while the fresh thread is not armed itself.
        """
        with self._lock:
            restored = self.restored_thread_id
            if restored is not None and thread_id not in self.armed:
                return restored
            return thread_id

    def _require_available(self) -> None:
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "proactive pushes are off")

    # ---- candidates and the cron -----------------------------------------------

    def candidates(self) -> list[str]:
        """Armed threads serve can deliver to, in a stable order."""
        deliverable = self.can_deliver or (lambda _tid: False)
        return sorted(tid for tid in self.armed if deliverable(tid))

    def reconcile(self) -> None:
        """Recreate the workspace's proactive cron, or delete it when nothing is armed.

        Deletes every proactive cron of the workspace and, while a thread is
        armed, creates one on a fresh cron thread with the current candidates
        (:meth:`_rotate_cron_thread`).
        """
        if not self.available:
            return
        with self._lock:
            for cron in list_proactive_crons(self.client, self.workspace):
                self.client.crons.delete(cron["cron_id"])
            self._cron_id = None
            self._mirrored = None
            if not self.armed:
                return
            self._restamp_armed()
            thread_id = self._rotate_cron_thread()
            candidates = self.candidates()
            cron = create_proactive_cron(
                self.client,
                self.workspace,
                thread_id=thread_id,
                candidates=candidates,
                timezone=self.timezone,
            )
            self._cron_id = str(cron["cron_id"])
            self._mirrored = candidates
            self._cron_created_at = time.time()
            logger.info(
                "[proactive] cron %s on thread %s (%s, %d candidate(s))",
                self._cron_id,
                thread_id,
                PROACTIVE_SCHEDULE,
                len(candidates),
            )

    def mirror(self) -> None:
        """Send the current candidates to the cron's input (the whole input).

        Idempotent. Creates the cron when a thread is armed and none exists.
        """
        from langgraph_sdk.errors import NotFoundError

        if not self.available:
            return
        with self._lock:
            if self._cron_id is None:
                if self.armed:
                    self.reconcile()
                return
            candidates = self.candidates()
            try:
                self.client.crons.update(self._cron_id, input=cron_input(candidates))
            except NotFoundError:
                # The dev server lost the cron (a restart without persistence).
                self.reconcile()
                return
            self._mirrored = candidates

    def stop(self) -> None:
        """Delete the workspace's proactive crons (serve stops or moves away)."""
        if not self.available:
            return
        with self._lock:
            for cron in list_proactive_crons(self.client, self.workspace):
                self.client.crons.delete(cron["cron_id"])
            self._cron_id = None
            self._mirrored = None

    def move_to(self, workspace: Workspace) -> None:
        """Follow serve into another workspace: its cron belongs to the new one.

        The armed threads stay with the old workspace: their flag is in their
        metadata, and :meth:`adopt` arms a thread again when serve resumes it.
        Called after the dev server moved, so a controller left unavailable by
        an earlier failed move is available again.
        """
        with self._lock:
            if self.unavailable_reason == MOVE_FAILED_REASON:
                self.unavailable_reason = None
            self.workspace = workspace
            self._cron_thread_id = None
            self._mirrored = None
            self._applied.clear()
            self.armed.clear()
            self.restored_thread_id = None

    def adopt(self, thread_id: str) -> None:
        """Take *thread_id*'s armed flag from its metadata (serve resumed it).

        A thread serve paused when it left is turned on again. Mirrors nothing:
        the caller mirrors once the resume is done.
        """
        if not self.available:
            return
        from langgraph_sdk.errors import NotFoundError

        try:
            thread = self.client.threads.get(thread_id)
        except NotFoundError:
            return
        flag = (thread.get("metadata") or {}).get("proactive")
        with self._lock:
            if flag == "paused":
                self.client.threads.update(thread_id, metadata={"proactive": "on"})
            if flag in ("on", "paused"):
                self.armed.add(thread_id)
            elif flag == "off":
                self.armed.discard(thread_id)

    def _restamp_armed(self) -> None:
        """Write the flag back where a dev-server restart without persistence lost it.

        The tick reads the flag from thread metadata only, and no later run
        stamps it again.
        """
        from langgraph_sdk.errors import NotFoundError

        for thread_id in sorted(self.armed):
            try:
                metadata = self.client.threads.get(thread_id).get("metadata") or {}
            except NotFoundError:
                continue
            if "proactive" not in metadata:
                self.client.threads.update(thread_id, metadata={"proactive": "on"})

    def _rotate_cron_thread(self) -> str:
        """A new cron thread carrying the newest old one's record; the old ones go.

        The runtime keeps every run of a thread a cron is bound to and never
        removes them, so each reconcile moves the cron to a fresh thread.
        ``evaluated`` and ``failures`` are copied over, so the move costs no
        extra shadow turn and no reset of a backoff. An old thread is unblocked
        first (a tick killed with SIGKILL stays ``running``) and then deleted;
        one that cannot be deleted is tried again at the next reconcile.
        """
        metadata = {"run_kind": PROACTIVE_RUN_KIND, "workspace_dir": self.workspace.key}
        old = [
            str(t["thread_id"])
            for t in self.client.threads.search(
                metadata=metadata, sort_by="updated_at", sort_order="desc", limit=100
            )
            or []
        ]
        new = create_proactive_thread(self.client, self.workspace)
        carried = False
        for thread_id in old:
            try:
                self._cancel_running(thread_id)
                if not carried:
                    carried = self._carry_record(thread_id, new)
                self.client.threads.delete(thread_id)
            except Exception:
                logger.warning(
                    "[proactive] could not retire cron thread %s",
                    thread_id,
                    exc_info=True,
                )
        self._cron_thread_id = new
        return new

    def _carry_record(self, source: str, target: str) -> bool:
        """Copy *source*'s ``evaluated`` and ``failures`` to *target*."""
        values = self.client.threads.get_state(source).get("values") or {}
        record = {k: values[k] for k in ("evaluated", "failures") if values.get(k)}
        if not record:
            return False
        self.client.threads.update_state(target, record, as_node="tick")
        return True

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
        current thread, for a push applied to a thread that is not current
        (the thread restored after a restart). It runs after the lock is
        released: switching mirrors the cron input, which takes the lock from
        another thread.
        """
        if not self.available or not self.armed:
            return
        now = time.time() if now is None else now
        if now < self._next_poll:
            return
        self._next_poll = now + POLL_INTERVAL_SECONDS
        switch: str | None = None
        with self._lock:
            try:
                # A thread's channel origin can appear after the last mirror
                # (a channel ``/resume`` re-binds it once the command is done).
                if self._cron_id is not None and self.candidates() != self._mirrored:
                    self.mirror()
                decisions = self._check_cron(now)
            except Exception:
                # A dev server that keeps refusing (one from before the upgrade,
                # with no ``proactive`` graph) would log a traceback every poll.
                self._failed_polls += 1
                logger.warning(
                    "[proactive] poll failed (%d in a row)",
                    self._failed_polls,
                    exc_info=self._failed_polls == 1,
                )
                self._next_poll = now + min(
                    POLL_INTERVAL_SECONDS * 2 ** min(self._failed_polls, 6),
                    MAX_POLL_BACKOFF_SECONDS,
                )
                return
            self._failed_polls = 0
            for decision in decisions:
                try:
                    applied_to = self._apply(decision)
                except Exception:
                    logger.warning(
                        "[proactive] applying push %s failed",
                        decision.get("proactive_id"),
                        exc_info=True,
                    )
                    continue
                if applied_to is not None and applied_to != current_thread_id:
                    switch = applied_to
            if switch is not None and switch == self.restored_thread_id:
                self.restored_thread_id = None
            if (
                self._cron_id is not None
                and now - self._cron_created_at > CRON_THREAD_MAX_AGE_SECONDS
            ):
                try:
                    self.reconcile()
                except Exception:
                    logger.warning("[proactive] cron rotation failed", exc_info=True)
        if switch is not None and switch_to is not None:
            switch_to(switch)

    # ---- restart survival --------------------------------------------------

    def restore(
        self,
        rows: list[dict[str, Any]],
        *,
        restore_origin: Callable[[str, dict[str, Any]], None],
    ) -> str | None:
        """Re-arm the newest armed thread of this workspace after a restart.

        The server's thread metadata holds the flag and the chat (``/proactive
        on`` writes both; serve pauses a thread it leaves), so the newest armed
        thread there comes first. *rows* come from
        ``sessions.latest_origin_checkpoints``, newest first: per thread, the
        stored folders and channel origin of its newest checkpoint that has an
        origin, and its newest stored flag; they stand in when a restart without
        file persistence lost the metadata, and the flag is written back then.
        A flag still on the server wins over a stored one. The restored thread
        stays a candidate until serve's thread changes or its fresh thread has
        its first turn.
        """
        from langgraph_sdk.errors import NotFoundError

        from ..paths import SessionDirs

        if not self.available:
            return None
        with self._lock:
            armed = self.client.threads.search(
                metadata={"proactive": "on", "workspace_dir": self.workspace.key},
                sort_by="updated_at",
                sort_order="desc",
                limit=1,
            )
            on_server = [
                {**(t.get("metadata") or {}), "thread_id": t["thread_id"]}
                for t in armed or []
            ]
            for row in [*on_server, *rows]:
                if row.get("proactive") != "on":
                    continue
                dirs = SessionDirs.from_stored(
                    row.get("workspace_dir"), row.get("run_dir")
                )
                if dirs is None or dirs.workspace.key != self.workspace.key:
                    continue
                origin = origin_from_row(row)
                if origin is None:
                    continue
                thread_id = str(row["thread_id"])
                try:
                    thread = self.client.threads.get(thread_id)
                except NotFoundError:
                    continue
                flag = (thread.get("metadata") or {}).get("proactive")
                if flag not in (None, "on"):
                    # ``off``, or ``paused``: serve had left the thread.
                    continue
                if flag != "on":
                    # The tick reads the flag from thread metadata only, which
                    # a dev-server restart without file persistence rebuilds
                    # without it; no later run restamps it.
                    self.client.threads.update(thread_id, metadata={"proactive": "on"})
                restore_origin(thread_id, origin)
                self.armed.add(thread_id)
                self.restored_thread_id = thread_id
                logger.info("[proactive] restored armed thread %s", thread_id)
                return thread_id
        return None

    def on_turn(self, thread_id: str) -> None:
        """A channel turn ran on *thread_id*, or serve moved to it.

        After a restart the restored thread is a candidate only until the user
        talks in, or moves to, another thread: from then on that is the
        conversation, and the restored one is paused. Moving to the restored
        thread makes it the current one.
        """
        if not self.available:
            return
        with self._lock:
            restored = self.restored_thread_id
            if restored is None:
                return
            self.restored_thread_id = None
            if restored == thread_id:
                return
            if self.forget_origin is not None:
                self.forget_origin(restored)
            self._pause(restored)

    def _check_cron(self, now: float) -> list[dict[str, Any]]:
        """Recreate a missing, failed or stalled cron; return the latest decisions."""
        crons = list_proactive_crons(self.client, self.workspace)
        if not crons or self._cron_thread_id is None:
            logger.warning("[proactive] cron missing; recreating it")
            self.reconcile()
            return []
        thread = self.client.threads.get(self._cron_thread_id)
        # The status stays ``error`` until the next tick ends; only a failure
        # after the last recreate is a new one.
        failed_at = _parse_time(thread.get("updated_at")) or 0.0
        if thread.get("status") == "error" and failed_at > self._cron_created_at:
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

    def _apply(self, decision: dict[str, Any]) -> str | None:
        """Apply one decision; return the thread written to, or None if skipped."""
        from ..gateway.server import _state_interrupts

        proactive_id = str(decision.get("proactive_id") or "")
        thread_id = str(decision.get("thread_id") or "")
        text = str(decision.get("text") or "")
        if not proactive_id or not thread_id or not text:
            return None
        if proactive_id in self._applied:
            return None
        # Decided once, applied at most once, whatever the outcome below.
        self._applied.add(proactive_id)

        decided_at = _parse_time(decision.get("decided_at"))
        if decided_at is None or time.time() - decided_at > DECISION_MAX_AGE_SECONDS:
            return self._skip(proactive_id, thread_id, "too_old")
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
        return thread_id

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
