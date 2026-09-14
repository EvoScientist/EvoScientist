"""Commit decision + message construction for a proactive push.

``decide_commit`` turns the shadow's reply into a decision (skip / stale /
would_commit) and, for a push, builds the exact tagged ``AIMessage`` + checkpoint
metadata the commit appends. This module is pure — it never writes. Under Shape
B the *apply* (append the tagged message as the ``model`` node, then clear the
pending ``next`` with ``as_node=END``) is performed in-process by serve, the
single serialized writer, so there is no out-of-band commit executor and no
409/conflict-retry machinery here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

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
    and describe exactly what serve appends when it applies the push.
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
      ="model"`` serve appends, log it, and return it.
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
