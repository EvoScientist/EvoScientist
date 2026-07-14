"""Transport-agnostic HITL and ask_user interaction engine.

The module defines the channel-side protocol shared by serve mode and the
CLI/TUI bridge: prompt formatting, reply grammar, stop handling, approval
policy, pending-reply routing, and the async engine coroutines for approval
and ask_user flows. Drivers provide transport-specific IO through
:class:`InteractionIO`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .capabilities import ChannelCapabilities

# ── timeout constants ──────────────────────────────────────────────────
# Per-flow defaults.  HITL approval is short (a yes/no gate); ask_user is
# longer because the human may need thinking time.
HITL_APPROVAL_TIMEOUT = 120.0  # seconds to wait for a HITL approval reply
ASK_USER_TIMEOUT = 300.0  # seconds to wait for an ask_user reply

# ── stop-command grammar ──────────────────────────────────────────
# Checked before reply parsing in *both* flows so a `/stop` mid-prompt
# always cancels instead of being captured as a literal answer.
_STOP_COMMANDS = frozenset(("/stop", "/cancel"))

# ── feedback strings ─────────────────────────────────────────
# Visible confirmations so a click/reply registers on channels without a
# message-recall API (e.g. QQ C2C).
APPROVED_FEEDBACK = "✅ Approved"
APPROVED_AUTO_FEEDBACK = "✅ Approved (auto-approving future actions)"
REJECTED_FEEDBACK = "❌ Rejected"
UNRECOGNIZED_FEEDBACK = "Unrecognized reply. Action rejected."
APPROVAL_TIMEOUT_FEEDBACK = "⏰ Approval timed out. Action rejected."
ASK_USER_TIMEOUT_FEEDBACK = "⏰ Response timed out."
OTHER_PROMPT = "Please type your answer:"

# ── stop / cancel helpers ──────────────────────────────────────────────


def is_stop_command(content: str | None) -> bool:
    """Whether incoming content is a stop/cancel slash command."""
    return (content or "").strip().lower() in _STOP_COMMANDS


def is_cancel_reply(content: str | None) -> bool:
    """Whether a reply is the literal ``cancel`` sentinel (case-insensitive)."""
    return (content or "").strip().lower() == "cancel"


# ── approval reply grammar ─────────────────────────────────────────────


def parse_approval_reply(text: str) -> str | None:
    """Parse a channel user's reply as an approval decision.

    Returns "approve", "reject", "auto", or None if not recognized.
    """
    t = text.strip().lower()
    if t in ("1", "y", "yes", "approve", "ok"):
        return "approve"
    if t in ("2", "n", "no", "reject"):
        return "reject"
    if t in ("3", "a", "auto", "approve all"):
        return "auto"
    return None


def approve_decisions(action_requests: list) -> list[dict]:
    """Build the ``decisions`` payload that approves every action request.

    Length matches ``action_requests`` (with a floor of 1, matching the
    consumer's historical ``len(...) or 1`` so an empty request list still
    yields a single approve — the shape ``Command(resume=...)`` expects).
    """
    n = len(action_requests) or 1
    return [{"type": "approve"} for _ in range(n)]


# ── approval prompt formatting ─────────────────────────────────────────


def format_approval_prompt(
    action_requests: list[dict], *, with_buttons: bool = False
) -> str:
    """Format an approval prompt as a text message for channel users.

    When *with_buttons* is True, the trailing "Reply: 1=Approve..."
    instruction is dropped — the buttons replace the textual cue.
    """
    lines = ["⚠️ Approval Required\n"]
    for i, req in enumerate(action_requests, 1):
        name = req.get("name", "")
        args = req.get("args", {})
        if isinstance(args, dict):
            command = args.get("command", args.get("path", ""))
        else:
            command = ""
        if command:
            lines.append(f"  {i}. {name}: {command}")
        else:
            lines.append(f"  {i}. {name}")
    if not with_buttons:
        lines.append("")
        lines.append("Reply: 1=Approve, 2=Reject, 3=Approve all")
        lines.append("(Auto-reject in 2 min if no reply)")
    return "\n".join(lines)


def approval_prompt_metadata(base_metadata: dict | None, *, with_buttons: bool) -> dict:
    """Outbound metadata for the HITL approval prompt.

    When *with_buttons* is True, attaches Approve/Reject/Auto buttons whose
    values match ``parse_approval_reply`` so a click flows through the same
    path as a typed ``"1"``/``"2"``/``"3"`` reply.
    """
    metadata = dict(base_metadata or {})
    if with_buttons:
        metadata["buttons"] = [
            {"text": "Approve", "value": "1", "type": "primary"},
            {"text": "Reject", "value": "2", "type": "danger"},
            {"text": "Approve all", "value": "3"},
        ]
    return metadata


# ── ask_user question formatting & answer grammar ──────────────────────


def _choice_value(choice: object, fallback: str = "") -> str:
    """Normalize one ask_user choice to its display/answer string.

    Choices arrive from model-produced tool args; the schema says dicts with
    a ``value`` key, but nothing enforces that at runtime, so plain strings
    (or anything else) must not crash the prompt.
    """
    if isinstance(choice, dict):
        return str(choice.get("value", fallback or choice))
    return str(choice)


def format_question_prompt(question: dict, index: int, total: int) -> str:
    """Format one ask_user *question* as a channel message.

    *index* is 0-based; *total* is the number of questions in the batch.
    """
    q_text = question.get("question", "")
    q_type = question.get("type", "text")
    required = question.get("required", True)

    if total == 1:
        header = "❓ Quick check-in from EvoScientist\n"
    else:
        header = f"❓ Question {index + 1}/{total}\n"

    lines: list[str] = [header, f"{index + 1}. {q_text}"]
    if not required:
        lines[-1] += " (optional)"

    if q_type == "multiple_choice":
        choices = question.get("choices", [])
        for j, choice in enumerate(choices):
            label = _choice_value(choice)
            letter = chr(ord("A") + j)
            lines.append(f"   {letter}. {label}")
        other_letter = chr(ord("A") + len(choices))
        lines.append(f"   {other_letter}. Other")
        letters = "/".join(chr(ord("A") + k) for k in range(len(choices) + 1))
        lines.append(f"\nReply with a letter ({letters}), or 'cancel'.")
    else:
        skip_hint = " Leave empty to skip." if not required else ""
        lines.append(f"\nReply with your answer, or 'cancel'.{skip_hint}")
    return "\n".join(lines)


def parse_choice_answer(raw: str, choices: list) -> tuple[str, str | None]:
    """Classify a multiple-choice reply.

    Returns ``(kind, value)``:

    * ``("other", None)`` — the "Other" letter was chosen; the caller must
      run the free-form sub-flow (send :data:`OTHER_PROMPT`, wait again).
    * ``("answer", value)`` — a resolved answer string (the chosen
      choice's ``value``, or the raw text when it isn't a valid letter).
    """
    other_letter = chr(ord("A") + len(choices))
    if len(raw) == 1 and raw.upper() == other_letter:
        return ("other", None)
    if len(raw) == 1 and raw.upper().isalpha():
        idx = ord(raw.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return ("answer", _choice_value(choices[idx], raw))
        return ("answer", raw)
    return ("answer", raw)


# ── approval policy ────────────────────────────────────────────────────


def config_auto_approve(action_requests: list[dict]) -> bool:
    """Whether config rules alone clear every action request.

    Returns True if no manual approval is needed via config: the global
    ``auto_approve`` flag, non-execute tools, or a ``shell_allow_list``
    match on every shell command. Fail-closed on config load errors.
    """
    if not action_requests:
        return True

    try:
        from ..config.settings import HITL_SHELL_TOOLS, load_config

        cfg = load_config()
    except Exception:
        return False  # fail-closed

    if cfg.auto_approve:
        return True

    shell_allow_list = (
        [s.strip() for s in cfg.shell_allow_list.split(",") if s.strip()]
        if cfg.shell_allow_list
        else []
    )

    for req in action_requests:
        name = req.get("name", "")
        if name not in HITL_SHELL_TOOLS:
            continue
        args = req.get("args", {})
        command = args.get("command", "") if isinstance(args, dict) else ""
        cmd = command.strip()
        if not any(cmd.startswith(prefix) for prefix in shell_allow_list):
            return False
    return True


class ApprovalPolicy:
    """Auto-approve policy backed by config rules and session grants.

    One instance is owned per process. The consumer keeps one on its event
    loop; the CLI bridge keeps one on the bus loop.
    """

    def __init__(self) -> None:
        self._granted_sessions: set[str] = set()

    def is_session_granted(self, session_key: str) -> bool:
        """Whether the user previously chose "Approve all" for this session."""
        return session_key in self._granted_sessions

    def grant_session(self, session_key: str) -> None:
        """Record an "Approve all" grant for this session."""
        self._granted_sessions.add(session_key)

    def clear_sessions(self) -> None:
        """Forget all session grants (test hygiene / session reset)."""
        self._granted_sessions.clear()

    def auto_decision(
        self, session_key: str, action_requests: list[dict]
    ) -> list[dict] | None:
        """Return an approve-all ``decisions`` list if this can auto-resolve.

        Auto-resolves when the session was granted "Approve all" or when
        config rules clear every request; otherwise returns ``None`` and
        the caller must prompt the user.
        """
        if self.is_session_granted(session_key) or config_auto_approve(action_requests):
            return approve_decisions(action_requests)
        return None


# ── transport adapter + reply registry ─────────────────────────────────


class InteractionIO(Protocol):
    """One conversation partner on one channel chat.

    A transport adapter: the engine coroutines below drive a human
    interaction entirely through this interface, so the same protocol
    logic runs over the consumer's async loop and over the CLI bus loop.

    Attributes
    ----------
    capabilities:
        The channel's :class:`ChannelCapabilities` — the engine reads
        ``inline_buttons`` to decide whether to attach approval buttons.
    base_metadata:
        The default outbound metadata for this chat (echoed back on each
        send unless the engine supplies richer metadata, e.g. buttons).
    """

    capabilities: ChannelCapabilities
    base_metadata: dict | None

    async def send(self, content: str, *, metadata: dict | None = None) -> bool:
        """Send *content* to the user; return True on success."""
        ...

    async def wait_reply(self, *, timeout: float) -> str | None:
        """Wait for the user's next reply; return None on timeout."""
        ...


@dataclass(frozen=True)
class PendingReply:
    """A pending prompt reply plus optional transport-specific context."""

    content: str
    context: object | None = None


class PendingReplyRegistry:
    """Route "the next message from this chat" into a waiting coroutine.

    One instance per process (the consumer owns one on its loop; the CLI
    bridge owns one on the bus loop).  ``register`` / ``wait`` are used by
    an :class:`InteractionIO` adapter to block for a reply; the inbound
    interception point calls ``try_resolve`` to hand a message to that
    waiter instead of enqueuing it as a fresh turn.

    Asyncio-based: register/resolve must happen on the same event loop.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[PendingReply]] = {}

    def register(self, session_key: str) -> asyncio.Future[PendingReply]:
        """Create and store a future awaiting the next reply for *session_key*."""
        loop = asyncio.get_running_loop()
        # A stale waiter for the same chat should never linger; cancel it
        # so its coroutine unwinds instead of hanging until timeout.
        stale = self._pending.get(session_key)
        if stale is not None and not stale.done():
            stale.cancel()
        fut: asyncio.Future[PendingReply] = loop.create_future()
        self._pending[session_key] = fut
        return fut

    def try_resolve(
        self,
        session_key: str,
        content: str,
        *,
        context: object | None = None,
    ) -> bool:
        """Deliver *content* to a pending waiter.  Returns True if consumed."""
        fut = self._pending.get(session_key)
        if fut is not None and not fut.done():
            fut.set_result(PendingReply(content=content, context=context))
            return True
        return False

    def discard(self, session_key: str) -> None:
        """Drop any pending waiter for *session_key* (idempotent)."""
        self._pending.pop(session_key, None)

    async def wait(self, session_key: str, timeout: float) -> str | None:
        """Register, await a reply for *timeout* seconds, then clean up.

        Returns the reply text, or ``None`` on timeout / cancellation.
        """
        reply = await self.wait_event(session_key, timeout)
        return reply.content if reply is not None else None

    async def wait_event(self, session_key: str, timeout: float) -> PendingReply | None:
        """Register, await a reply event, then clean up.

        Returns the full reply envelope, or ``None`` on timeout / registry
        cancellation. Cancellation of the task awaiting this method propagates.
        """
        fut = self.register(session_key)
        try:
            done, _pending = await asyncio.wait({fut}, timeout=timeout)
            if not done:
                fut.cancel()
                return None
            try:
                return fut.result()
            except asyncio.CancelledError:
                return None
        finally:
            # Identity-safe: only drop *our* slot, never a newer waiter that
            # re-registered on the same chat while we were unwinding.
            if self._pending.get(session_key) is fut:
                self._pending.pop(session_key, None)

    def clear(self) -> None:
        """Cancel and forget every pending waiter (shutdown / test hygiene)."""
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    def __contains__(self, session_key: str) -> bool:
        return session_key in self._pending


# ── engine coroutines ──────────────────────────────────────────────────


async def resolve_ask_user(
    questions: list[dict], io: InteractionIO, *, timeout: float = ASK_USER_TIMEOUT
) -> dict:
    """Drive an ask_user interrupt to a resume payload.

    Sends each question in turn, collects answers, and handles the choice
    grammar (letters + the "Other" free-form sub-flow).  ``/stop`` and
    ``cancel`` are checked *before* parsing every reply.

    Returns a dict suitable for ``Command(resume=...)``:
    ``{"answers": [...], "status": "answered"}`` or ``{"status": "cancelled"}``.
    """
    if not questions:
        return {"answers": [], "status": "answered"}

    total = len(questions)
    answers: list[str] = []

    for i, q in enumerate(questions):
        if not await io.send(format_question_prompt(q, i, total)):
            return {"status": "cancelled"}

        reply = await io.wait_reply(timeout=timeout)
        if reply is None:
            await io.send(ASK_USER_TIMEOUT_FEEDBACK)
            return {"status": "cancelled"}

        raw = reply.strip()
        required = q.get("required", True) is not False
        if raw == "":
            if required:
                return {"status": "cancelled"}
            answers.append("")
            continue

        if is_stop_command(raw) or is_cancel_reply(raw):
            return {"status": "cancelled"}

        if q.get("type", "text") == "multiple_choice":
            choices = q.get("choices", [])
            kind, value = parse_choice_answer(raw, choices)
            if kind == "other":
                if not await io.send(OTHER_PROMPT):
                    return {"status": "cancelled"}
                other = await io.wait_reply(timeout=timeout)
                if other is None:
                    await io.send(ASK_USER_TIMEOUT_FEEDBACK)
                    return {"status": "cancelled"}
                other_raw = other.strip()
                if other_raw == "":
                    if required:
                        return {"status": "cancelled"}
                    answers.append("")
                    continue
                if is_stop_command(other_raw) or is_cancel_reply(other_raw):
                    return {"status": "cancelled"}
                answers.append(other_raw)
            else:
                answers.append(value)
        else:
            answers.append(raw)

    return {"answers": answers, "status": "answered"}


@dataclass
class ApprovalOutcome:
    """Result of :func:`resolve_approval`.

    ``decisions`` is the approve-all payload on approve/auto, or ``None``
    when the action was declined (reject / timeout / stop / unrecognized).

    ``unrecognized_reply`` carries the raw reply text when parsing failed.
    The engine centralizes *parsing* but does not decide the transport
    policy for unparseable text — the drivers do: the consumer rejects the
    pending action and refeeds the text as a new agent turn (a channel user
    who ignores the prompt and types a fresh instruction must not lose it);
    the CLI bridge sends :data:`UNRECOGNIZED_FEEDBACK` and declines.
    """

    decisions: list[dict] | None = None
    unrecognized_reply: str | None = None


async def resolve_approval(
    action_requests: list,
    io: InteractionIO,
    policy: ApprovalPolicy,
    session_key: str,
    *,
    timeout: float = HITL_APPROVAL_TIMEOUT,
) -> ApprovalOutcome:
    """Drive a HITL approval interrupt to an :class:`ApprovalOutcome`.

    Auto-resolves via *policy* (session grant or config rule) without
    prompting.  Otherwise sends the approval prompt (with capability-driven
    buttons), waits for a reply, and parses it.  ``/stop`` cancels silently
    (it already got its own ack from the transport's stop fast-path).  An
    unrecognized reply declines *without feedback* and hands the raw text
    back to the driver via ``unrecognized_reply`` (see
    :class:`ApprovalOutcome` for the per-driver policy).
    """
    auto = policy.auto_decision(session_key, action_requests)
    if auto is not None:
        return ApprovalOutcome(decisions=auto)

    has_buttons = bool(io.capabilities.inline_buttons)
    prompt = format_approval_prompt(action_requests, with_buttons=has_buttons)
    metadata = approval_prompt_metadata(io.base_metadata, with_buttons=has_buttons)
    if not await io.send(prompt, metadata=metadata):
        return ApprovalOutcome()

    reply = await io.wait_reply(timeout=timeout)
    if reply is None:
        await io.send(APPROVAL_TIMEOUT_FEEDBACK)
        return ApprovalOutcome()

    if is_stop_command(reply):
        return ApprovalOutcome()

    decision = parse_approval_reply(reply)
    if decision == "auto":
        policy.grant_session(session_key)
        await io.send(APPROVED_AUTO_FEEDBACK)
        return ApprovalOutcome(decisions=approve_decisions(action_requests))
    if decision == "approve":
        await io.send(APPROVED_FEEDBACK)
        return ApprovalOutcome(decisions=approve_decisions(action_requests))
    if decision == "reject":
        await io.send(REJECTED_FEEDBACK)
        return ApprovalOutcome()

    # Unrecognized — decline and report the raw text; the driver chooses
    # the feedback / refeed policy.
    return ApprovalOutcome(unrecognized_reply=reply)
