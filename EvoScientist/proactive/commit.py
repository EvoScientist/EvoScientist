"""Turn a shadow reply into a push decision, and the push message itself.

Pure: nothing here writes. A decided push is the tagged ``AIMessage`` that
serve appends to the source thread. Logs carry reasons and sizes, never the
reply text: a reply is a message to the user, and a server log is not where it
should be read.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# The shadow model returns this to decline.
NO_PUSH_SENTINEL = "NO_PUSH"
# Stripped from a line before it is compared with the sentinel: models wrap
# or punctuate it (``**NO_PUSH**``, ``NO_PUSH.``).
_SENTINEL_WRAPPING = "`*_\"'.!:- "

# Longest reply that still reads as one channel message (Telegram's limit is
# 4096). Anything longer is a runaway generation, not a push.
MAX_PUSH_CHARS = 4000

# A model with no tools bound sometimes writes its tool-call envelope as text
# (seen: ``<tool_call>execute...</tool_call>``; Claude writes
# ``<function_calls>``), or copies the form the flattened history uses for past
# calls. That is an attempted action, not a message.
_TOOL_CALL_TEXT_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<function_calls>",
    "[called tool `",
)

# A push is tagged under this ``additional_kwargs`` namespace:
# ``{"evoscientist": {"is_proactive_push": True, "proactive_id": "<id>"}}``.
PROACTIVE_TAG_NAMESPACE = "evoscientist"


def proactive_message_id(proactive_id: str) -> str:
    """Message id of a push, so it can be found and removed later."""
    return f"proactive-{proactive_id}"


def build_proactive_tag(proactive_id: str) -> dict[str, Any]:
    """``additional_kwargs`` that mark a message as a proactive push."""
    return {
        PROACTIVE_TAG_NAMESPACE: {
            "is_proactive_push": True,
            "proactive_id": proactive_id,
        }
    }


def read_proactive_tag(additional_kwargs: Any) -> dict[str, Any] | None:
    """The proactive tag in ``additional_kwargs``, or ``None`` when it is not a push."""
    if not isinstance(additional_kwargs, dict):
        return None
    tag = additional_kwargs.get(PROACTIVE_TAG_NAMESPACE)
    if not isinstance(tag, dict) or not tag.get("is_proactive_push"):
        return None
    return tag


def last_non_proactive_message_id(messages: Iterable[Any]) -> str | None:
    """Id of the newest message that is not a proactive push.

    This is the "has anything new arrived" fingerprint: a push does not change
    it, while a user turn or a notifier turn does. Messages without an id do
    not count.
    """
    for message in reversed(list(messages)):
        if read_proactive_tag(getattr(message, "additional_kwargs", None)):
            continue
        message_id = getattr(message, "id", None)
        if message_id:
            return str(message_id)
    return None


@dataclass(frozen=True)
class CommitDecision:
    """``action`` is ``"skip"``, ``"stale"`` or ``"push"``; only a push has a message."""

    action: str
    reason: str
    message: AIMessage | None = None


def _is_no_push(text: str) -> bool:
    """Whether the first or last non-empty line is the sentinel, wrapping aside.

    Models add markdown, punctuation or a line of justification around it.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    bare = {
        line.strip().strip(_SENTINEL_WRAPPING).upper() for line in (lines[0], lines[-1])
    }
    return NO_PUSH_SENTINEL in bare


def decide_commit(
    reply: str | None,
    *,
    source_thread_id: str,
    pre_head: str | None,
    current_head: str | None,
    proactive_id: str,
    max_reply_chars: int = MAX_PUSH_CHARS,
) -> CommitDecision:
    """Decide what to do with the shadow's reply.

    - no reply, an empty one, or ``NO_PUSH`` as its first or last line: skip;
    - longer than ``max_reply_chars`` (counted on the text that would be
      sent): skip as ``too_long``;
    - a tool-call envelope written as text: skip as ``tool_call_text``;
    - the source thread's head moved while the shadow ran: ``stale``;
    - otherwise a push of the stripped reply, tagged with ``proactive_id``.
    """
    text = (reply or "").strip()
    if reply is None:
        reason = "no_reply"
    elif not text:
        reason = "empty"
    elif _is_no_push(text):
        reason = "no_push"
    elif len(text) > max_reply_chars:
        reason = "too_long"
    elif any(marker in text for marker in _TOOL_CALL_TEXT_MARKERS):
        reason = "tool_call_text"
    else:
        reason = ""
    if reason:
        level = (
            logging.WARNING
            if reason in ("too_long", "tool_call_text")
            else logging.INFO
        )
        logger.log(
            level,
            "[proactive decision] skip thread %s: %s (%d chars)",
            source_thread_id,
            reason,
            len(text),
        )
        return CommitDecision("skip", reason)

    if pre_head != current_head:
        logger.info(
            "[proactive decision] stale thread %s: head moved during the shadow turn",
            source_thread_id,
        )
        return CommitDecision("stale", "stale_source")

    logger.info(
        "[proactive decision] push for thread %s (proactive_id=%s, %d chars)",
        source_thread_id,
        proactive_id,
        len(text),
    )
    return CommitDecision(
        "push",
        "ok",
        message=AIMessage(
            content=text,
            id=proactive_message_id(proactive_id),
            additional_kwargs=build_proactive_tag(proactive_id),
        ),
    )
