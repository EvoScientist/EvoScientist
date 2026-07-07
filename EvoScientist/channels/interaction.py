"""Transport-agnostic HITL / ask_user interaction engine.

This module is the single home for the channel-side interaction protocol
shared by the two drivers that talk to a human over a chat channel:

* :mod:`EvoScientist.channels.consumer` — serve / standalone mode
  (natively async, runs on the consumer event loop).
* :mod:`EvoScientist.cli.channel` — CLI / TUI mode (a synchronous bridge
  that drives the engine on the bus loop).

Historically each driver carried its own copy of prompt formatting, the
reply grammar (choice letters, the "Other" free-form sub-flow,
approve/reject/approve-all, stop commands), the auto-approve policy and
the bilingual feedback strings — ~250 lines that had to be hand-synced
and had already drifted (``/stop`` handling existed only on the CLI
side).  Everything pure lives here now; both drivers point at it.

Sizing rule: this module is sized for the two current interaction flows
(approval + ask_user). New interrupt types should *extend the engine here*
rather than growing a third driver copy.
"""

from __future__ import annotations

# ── timeout constants ──────────────────────────────────────────────────
# Per-flow defaults.  HITL approval is short (a yes/no gate); ask_user is
# longer because the human may need thinking time.
HITL_APPROVAL_TIMEOUT = 120.0  # seconds to wait for a HITL approval reply
ASK_USER_TIMEOUT = 300.0  # seconds to wait for an ask_user reply

# ── stop-command grammar (G3) ──────────────────────────────────────────
# Checked before reply parsing in *both* flows so a `/stop` mid-prompt
# always cancels instead of being captured as a literal answer.
_STOP_COMMANDS = frozenset(("/stop", "/cancel"))

# ── bilingual feedback strings ─────────────────────────────────────────
# Visible confirmations so a click/reply registers on channels without a
# message-recall API (e.g. QQ C2C).
APPROVED_FEEDBACK = "✅ 已批准"
APPROVED_AUTO_FEEDBACK = "✅ 已批准（后续自动通过）"
REJECTED_FEEDBACK = "❌ 已拒绝"
UNRECOGNIZED_FEEDBACK = "Unrecognized reply. Action rejected."
APPROVAL_TIMEOUT_FEEDBACK = "⏰ Approval timed out. Action rejected."
ASK_USER_TIMEOUT_FEEDBACK = "⏰ Response timed out."
OTHER_PROMPT = "Please type your answer:"

# Feedback keyed by parsed approval decision.  ``None`` means "send
# nothing" (timeout / stop are silent — a late reply must not claim the
# user approved when they walked away, and /stop already got its own ack).
_DECISION_FEEDBACK: dict[str, str] = {
    "approve": APPROVED_FEEDBACK,
    "auto": APPROVED_AUTO_FEEDBACK,
    "reject": REJECTED_FEEDBACK,
}


def decision_feedback(decision: str | None) -> str:
    """Feedback string for a parsed approval *decision*.

    ``approve``/``auto``/``reject`` map to their bilingual strings; an
    unrecognized reply (``None``) maps to the English reject notice.
    """
    if decision is None:
        return UNRECOGNIZED_FEEDBACK
    return _DECISION_FEEDBACK.get(decision, UNRECOGNIZED_FEEDBACK)


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
            label = choice.get("value", str(choice))
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


def parse_choice_answer(raw: str, choices: list[dict]) -> tuple[str, str | None]:
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
            return ("answer", choices[idx].get("value", raw))
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
    """Auto-approve policy: config rules + a session auto-approve registry.

    One instance is owned per process (the consumer holds one on its loop;
    the CLI bridge holds one on the bus loop). Cross-process sharing is a
    non-goal — unifying the *implementation* removes the manual-sync
    burden without merging the two processes' registries.
    """

    def __init__(self) -> None:
        self._granted_sessions: set[str] = set()

    @staticmethod
    def session_key(channel: str, chat_id: str) -> str:
        """Derive the ``"channel:chat_id"`` session key."""
        return f"{channel}:{chat_id}"

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
