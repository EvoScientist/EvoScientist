"""Commit decision and commit execution for a proactive push.

``decide_commit`` turns the shadow's reply into a decision (skip / stale /
would_commit) and, for a push, builds the exact tagged ``AIMessage`` + checkpoint
metadata the commit appends. ``commit_with_conflict_retry`` executes a
``would_commit`` decision through an injected ``commit_fn``, deferring and
retrying when the source thread has a run in flight. ``make_gateway_commit_fn``
is the gateway-backed ``commit_fn``: append the tagged message as the ``model``
node, then clear the pending ``next`` with ``as_node=END``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph_sdk.errors import ConflictError

from ..cli._constants import build_metadata

logger = logging.getLogger(__name__)

# Sentinel the shadow model emits to signal "do not push".
NO_PUSH_SENTINEL = "NO_PUSH"

# Longest reply that still counts as one channel message (a requested table or
# summary fits; Telegram's hard limit is 4096). Anything longer is a runaway
# generation, not a push, and is discarded rather than delivered.
MAX_PUSH_CHARS = 4000

# A model with no tools bound sometimes emits its tool-call envelope as plain
# text (observed: ``<tool_call>execute...</tool_call>``). That is an attempted
# action, never a message to send.
_TOOL_CALL_TEXT_MARKERS = ("<tool_call>", "</tool_call>")


def proactive_message_id(proactive_id: str) -> str:
    """Stable message id for a committed push (addressable for later removal)."""
    return f"proactive-{proactive_id}"


# The committed assistant message is tagged under this ``additional_kwargs``
# namespace so it can be told apart from normal replies:
# ``{"evoscientist": {"is_proactive_push": True, "proactive_id": "<id>"}}``.
PROACTIVE_TAG_NAMESPACE = "evoscientist"


def build_proactive_tag(proactive_id: str) -> dict[str, Any]:
    """``additional_kwargs`` payload that marks a message as a proactive push."""
    return {
        PROACTIVE_TAG_NAMESPACE: {
            "is_proactive_push": True,
            "proactive_id": proactive_id,
        }
    }


def read_proactive_tag(additional_kwargs: Any) -> dict[str, Any] | None:
    """Return the proactive-push tag inside ``additional_kwargs``, or ``None``.

    ``None`` when the kwargs are not a mapping, carry no ``evoscientist``
    namespace, or the namespace is not flagged ``is_proactive_push``.
    """
    if not isinstance(additional_kwargs, dict):
        return None
    tag = additional_kwargs.get(PROACTIVE_TAG_NAMESPACE)
    if not isinstance(tag, dict) or not tag.get("is_proactive_push"):
        return None
    return tag


@dataclass(frozen=True)
class CommitDecision:
    """Outcome of a commit decision.

    ``action`` is one of ``"skip"`` (NO_PUSH/empty), ``"stale"`` (source thread
    changed during shadow generation), or ``"would_commit"``. The ``message`` /
    ``metadata`` / ``as_node`` fields are populated only for ``"would_commit"``
    and describe exactly what a real commit WOULD append.
    """

    action: str
    reason: str
    message: AIMessage | None = None
    metadata: dict[str, Any] | None = None
    as_node: str | None = None


def decide_commit(
    reply: str | None,
    *,
    source_thread_id: str,
    workspace_dir: str | None,
    model: str | None,
    pre_head: str | None,
    current_head: str | None,
    proactive_id: str,
    max_reply_chars: int = MAX_PUSH_CHARS,
) -> CommitDecision:
    """Decide what to do with the shadow's reply; never writes.

    - ``reply`` None/empty/``NO_PUSH`` → skip; longer than ``max_reply_chars`` →
      skip with reason ``too_long``; a textual tool-call envelope → skip with
      reason ``tool_call_text``.
    - source ``pre_head != current_head`` (a user turn landed during the shadow) →
      stale-discard.
    - otherwise → build the tagged ``AIMessage`` + ``build_metadata`` + ``as_node
      ="model"`` the commit uses, log it, and return it.
    """
    if reply is None:
        return _skip(source_thread_id, "no_reply", reply)
    stripped = reply.strip()
    if not stripped:
        return _skip(source_thread_id, "empty", reply)
    if stripped == NO_PUSH_SENTINEL:
        return _skip(source_thread_id, "no_push", reply)
    if len(stripped) > max_reply_chars:
        logger.warning(
            "[proactive decision] skip: thread %s reply is %d chars (limit %d); "
            "not a concise message, discarding",
            source_thread_id,
            len(stripped),
            max_reply_chars,
        )
        return CommitDecision("skip", "too_long")
    if any(marker in stripped for marker in _TOOL_CALL_TEXT_MARKERS):
        logger.warning(
            "[proactive decision] skip: thread %s reply contains a tool-call "
            "envelope as text; discarding (preview=%r)",
            source_thread_id,
            stripped[:200],
        )
        return CommitDecision("skip", "tool_call_text")

    if pre_head != current_head:
        logger.info(
            "[proactive decision] skip: thread %s changed during shadow (%s -> %s)",
            source_thread_id,
            pre_head,
            current_head,
        )
        return CommitDecision("stale", "stale_source")

    message = AIMessage(
        content=reply,
        id=proactive_message_id(proactive_id),
        additional_kwargs=build_proactive_tag(proactive_id),
    )
    metadata = build_metadata(workspace_dir, model)
    logger.info(
        "[proactive decision] push for thread %s: append AIMessage "
        "(proactive_id=%s, as_node='model', %d chars) then clear "
        "(values=None, as_node=END); metadata=%s; preview=%r",
        source_thread_id,
        proactive_id,
        len(reply),
        metadata,
        reply[:200],
    )
    return CommitDecision(
        "would_commit",
        "ok",
        message=message,
        metadata=metadata,
        as_node="model",
    )


def _skip(source_thread_id: str, reason: str, reply: str | None) -> CommitDecision:
    # INFO, not DEBUG: a skipped check is the common outcome and the only trace
    # of what the shadow said, so it must be visible in ordinary server logs.
    logger.info(
        "[proactive decision] skip: thread %s reason=%s reply=%r",
        source_thread_id,
        reason,
        (reply or "")[:160],
    )
    return CommitDecision("skip", reason)


@dataclass(frozen=True)
class CommitOutcome:
    """Result of a proactive commit attempt.

    ``status`` is one of ``"committed"`` (the tagged AIMessage was appended),
    ``"stale"`` (a user turn landed on the source thread, so the shadow reply no
    longer reflects the conversation → discarded), or ``"conflict_exhausted"``
    (a user run stayed in flight across every retry → dropped; the next check
    re-evaluates from a fresh snapshot). ``attempts`` counts commit tries made.
    """

    status: str
    reason: str
    attempts: int


async def commit_with_conflict_retry(
    decision: CommitDecision,
    *,
    source_thread_id: str,
    commit_fn: Callable[
        [AIMessage, dict[str, Any] | None, str | None], Awaitable[None]
    ],
    read_current_head: Callable[[], str | Awaitable[str | None] | None],
    pre_head: str | None,
    max_retries: int = 3,
    backoff: Callable[[int], Awaitable[None]] | None = None,
) -> CommitOutcome:
    """Commit a decided proactive push, deferring-and-retrying on ConflictError.

    Gateway-agnostic: ``commit_fn`` performs the real append-tagged-AIMessage-as-
    ``model`` + clear. The server-gateway path (``gateway.update_state_values`` →
    langgraph_sdk) raises :class:`ConflictError` (409) when a user run is in flight;
    a ``commit_fn`` that never conflicts simply takes the clean path.

    On every attempt the source head is re-read first (``read_current_head`` may
    return an awaitable, e.g. a server ``get_state`` fetch): if it moved past
    ``pre_head`` a user turn (or another proactive tick) has landed, so the shadow
    reply is stale → discard (never overwrite a fresh turn with an older draft). On :class:`ConflictError` a user run is
    still executing: defer (``backoff``) and retry — the next iteration's head check
    discards the push once that run commits. Runs out of retries → drop.

    ``decision`` MUST have ``action == "would_commit"`` (its message/metadata/as_node
    are what gets committed).
    """
    if decision.action != "would_commit":
        raise ValueError(
            f"commit_with_conflict_retry expects a would_commit decision, "
            f"got {decision.action!r}"
        )
    assert decision.message is not None  # would_commit always carries a message

    for attempt in range(max_retries + 1):
        if await resolve_head(read_current_head) != pre_head:
            logger.info(
                "[proactive commit] skip: thread %s changed before commit "
                "(head moved from %s) — discarding stale push",
                source_thread_id,
                pre_head,
            )
            return CommitOutcome("stale", "stale_source", attempt)
        try:
            await commit_fn(decision.message, decision.metadata, decision.as_node)
        except ConflictError:
            logger.warning(
                "[proactive commit] ConflictError on thread %s (attempt %d/%d): "
                "a user run is in flight; deferring",
                source_thread_id,
                attempt + 1,
                max_retries + 1,
            )
            if backoff is not None:
                await backoff(attempt)
            continue
        logger.info(
            "[proactive commit] committed proactive push to thread %s (attempt %d)",
            source_thread_id,
            attempt + 1,
        )
        return CommitOutcome("committed", "ok", attempt + 1)

    logger.warning(
        "[proactive commit] gave up on thread %s: user run still in flight after "
        "%d attempts — dropping push (next check will re-evaluate)",
        source_thread_id,
        max_retries + 1,
    )
    return CommitOutcome("conflict_exhausted", "retries_exhausted", max_retries + 1)


async def resolve_head(
    read_current_head: Callable[[], str | Awaitable[str | None] | None],
) -> str | None:
    """Call ``read_current_head`` and await it if it returned an awaitable."""
    head = read_current_head()
    if inspect.isawaitable(head):
        head = await head
    return head


class _StateGateway(Protocol):
    """Minimal gateway surface the real commit uses: the widened
    ``update_state_values`` (server path, langgraph_sdk-backed → raises
    ``ConflictError`` on an in-flight run)."""

    async def update_state_values(
        self,
        target: Any,
        thread_id: str,
        values: dict[str, Any] | None,
        *,
        as_node: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


def make_gateway_commit_fn(
    gateway: _StateGateway,
    target: Any,
    thread_id: str,
) -> Callable[[AIMessage, dict[str, Any] | None, str | None], Awaitable[None]]:
    """Build the real server-path ``commit_fn`` for :func:`commit_with_conflict_retry`.

    Commits a decided proactive push INTO the source thread via the widened gateway
    ``update_state_values``: append the tagged ``AIMessage`` as the ``model`` node
    (carrying ``metadata`` so the checkpoint stays visible), then clear the pending
    ``next`` (``values=None`` with ``as_node=END`` — the only valid clear shape).
    Either call raises :class:`ConflictError` (from the SDK) when a user run is in
    flight; that propagates to ``commit_with_conflict_retry``'s defer-and-retry loop.

    ``gateway``/``target`` are injected, so this is unit-testable against a fake
    gateway.
    """

    async def _commit(
        message: AIMessage,
        metadata: dict[str, Any] | None,
        as_node: str | None,
    ) -> None:
        # The append is the conflict-prone commit: a ConflictError here propagates
        # so commit_with_conflict_retry can defer-and-retry (nothing committed yet).
        await gateway.update_state_values(
            target,
            thread_id,
            {"messages": [message]},
            as_node=as_node,
            metadata=metadata,
        )
        # The clear is best-effort cleanup (reset pending ``next``). It runs AFTER a
        # successful append, so its ConflictError must NOT propagate — a retry would
        # re-append and double-commit the push. A user run racing in right after the
        # append leaves a benign non-empty ``next`` the user's own turn resolves.
        try:
            await gateway.update_state_values(target, thread_id, None, as_node=END)
        except ConflictError:
            logger.warning(
                "[proactive commit] clear-next conflicted after a committed append "
                "on thread %s; leaving pending next (benign, user run resolves it)",
                thread_id,
            )

    return _commit
