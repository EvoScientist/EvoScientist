"""Background channel management — bus mode with ChannelManager.

Architecture:
  Bus thread: runs ChannelManager + all channels + inbound consumer.
  Main CLI thread: runs agent invocations (to avoid event-loop conflicts).

The inbound consumer does NOT call the agent directly.  Instead it
enqueues a ``ChannelMessage`` on a thread-safe ``queue.Queue`` and waits
for the main thread to set a response via ``_set_channel_response()``.
"""

import asyncio
import logging
import queue
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from rich.panel import Panel
from rich.text import Text

from ..commands.base import ChannelRuntime
from ..stream.console import console

_channel_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue bridge: bus thread  ⇄  main CLI thread
# ---------------------------------------------------------------------------


@dataclass
class ChannelMessage:
    """A message from a channel, enqueued for the main CLI thread."""

    msg_id: str
    content: str
    sender: str
    channel_type: str
    metadata: dict | None = None
    # Filled by the bus consumer so the main thread can send callbacks
    channel_ref: Any = None  # Channel instance (for thinking / todo / file)
    bus_ref: Any = None  # MessageBus (for publishing outbound)
    chat_id: str = ""
    message_id: str | None = None


# Thread-safe queue: bus → main
_message_queue: queue.Queue[ChannelMessage] = queue.Queue()

# Pending responses:
# main → bus (msg_id → {"future": Future[str], "loop": loop, "response": str|None})
_pending_responses: dict[str, dict] = {}
_response_lock = threading.Lock()

_RESPONSE_TIMEOUT = 600.0
_LATE_RESPONSE_TIMEOUT = 86400.0
_LATE_RESPONSE_NOTICE = "Still working on it. I'll send the result when it's ready."
_channel_request_lock = threading.Lock()
_channel_requests: dict[str, dict[str, str]] = {}
_session_requests: dict[str, list[str]] = {}
_cancelled_channel_messages: set[str] = set()


def _enqueue_channel_message(msg: ChannelMessage) -> asyncio.Future[str]:
    """Enqueue a channel message for the main thread and return a wait future."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    with _response_lock:
        _pending_responses[msg.msg_id] = {
            "future": future,
            "loop": loop,
            "response": None,
        }
    _register_channel_request(msg)
    _message_queue.put(msg)
    return future


def _set_channel_response(msg_id: str, response: str) -> None:
    """Set the response for a channel message and unblock the bus consumer."""
    with _response_lock:
        slot = _pending_responses.get(msg_id)
        if slot:
            slot["response"] = response
            future = slot["future"]
            loop = slot["loop"]
        else:
            return

    def _resolve_future() -> None:
        if not future.done():
            future.set_result(response)

    loop.call_soon_threadsafe(_resolve_future)


def _pop_channel_response(msg_id: str, *, cancel_pending: bool = False) -> str | None:
    """Retrieve and remove the response for a channel message."""
    with _response_lock:
        slot = _pending_responses.pop(msg_id, None)
    if not slot:
        return None

    future = slot["future"]
    if cancel_pending and not future.done():
        future.cancel()
    return slot["response"]


def _channel_session_key(channel_type: str, chat_id: str) -> str:
    return f"{channel_type}:{chat_id}"


def _channel_message_session_key(msg: ChannelMessage) -> str:
    return _channel_session_key(msg.channel_type, msg.chat_id)


def _channel_message_cancel_scope(msg: ChannelMessage) -> str:
    return f"channel:{msg.channel_type}:{msg.chat_id}:{msg.msg_id}"


def _register_channel_request(msg: ChannelMessage) -> None:
    """Track a queued channel request so `/stop` can find it later."""
    session_key = _channel_message_session_key(msg)
    with _channel_request_lock:
        _channel_requests[msg.msg_id] = {
            "session_key": session_key,
            "cancel_scope": _channel_message_cancel_scope(msg),
            "state": "queued",
        }
        _session_requests.setdefault(session_key, []).append(msg.msg_id)


def _claim_channel_request(msg: ChannelMessage) -> bool:
    """Mark a queued request active. Returns False if it was cancelled first."""
    with _channel_request_lock:
        slot = _channel_requests.get(msg.msg_id)
        if slot is None or msg.msg_id in _cancelled_channel_messages:
            return False
        slot["state"] = "active"
        return True


def _claim_or_complete_channel_request(msg: ChannelMessage) -> bool:
    """Claim a request, or clean it up if `/stop` cancelled it while queued."""
    if _claim_channel_request(msg):
        return True
    _complete_channel_request(msg.msg_id)
    return False


def _channel_request_state(msg_id: str) -> str | None:
    with _channel_request_lock:
        slot = _channel_requests.get(msg_id)
        return slot.get("state") if slot is not None else None


def _complete_channel_request(
    msg_id: str,
    *,
    discard_cancel_scope: bool = True,
) -> None:
    """Forget a request once its waiter is resolved or cancelled."""
    with _channel_request_lock:
        slot = _channel_requests.pop(msg_id, None)
        _cancelled_channel_messages.discard(msg_id)
        if slot is not None:
            request_ids = _session_requests.get(slot["session_key"])
            if request_ids:
                try:
                    request_ids.remove(msg_id)
                except ValueError:
                    pass
                if not request_ids:
                    _session_requests.pop(slot["session_key"], None)

    if slot is not None and discard_cancel_scope:
        from ..stream.display import discard_stream_cancel

        discard_stream_cancel(slot["cancel_scope"])


def _cancel_channel_session(channel_type: str, chat_id: str) -> tuple[int, int]:
    """Cancel queued and active work for one channel chat session."""
    session_key = _channel_session_key(channel_type, chat_id)
    with _channel_request_lock:
        request_ids: list[str] = []
        cancelled_ids: list[str] = []
        active_scopes: list[str] = []
        with _response_lock:
            for msg_id in tuple(_session_requests.get(session_key, ())):
                request_slot = _channel_requests.get(msg_id)
                if request_slot is None:
                    continue
                response_slot = _pending_responses.get(msg_id)
                response_resolved = False
                if response_slot is not None:
                    future = response_slot["future"]
                    # Once a response is already resolved, leave the slot alone
                    # so the bus waiter can still publish it instead of falling
                    # back to "No response".
                    response_resolved = (
                        response_slot.get("response") is not None or future.done()
                    )
                    if not response_resolved:
                        request_ids.append(msg_id)

                should_cancel = False
                if response_slot is None:
                    should_cancel = request_slot.get("state") == "active"
                else:
                    should_cancel = not response_resolved

                if should_cancel:
                    cancelled_ids.append(msg_id)
                if request_slot.get("state") == "active" and should_cancel:
                    active_scopes.append(request_slot["cancel_scope"])
        _cancelled_channel_messages.update(cancelled_ids)

    for msg_id in request_ids:
        _pop_channel_response(msg_id, cancel_pending=True)

    if active_scopes:
        from ..stream.display import request_stream_cancel

        for cancel_scope in active_scopes:
            request_stream_cancel(cancel_scope)

    return len(request_ids), len(active_scopes)


# ---------------------------------------------------------------------------
# Slash command dispatch for channel messages
# ---------------------------------------------------------------------------
# Shared by all three UI surfaces that accept inbound channel messages:
# Rich CLI (``cli/interactive.py::_process_channel_message``), Textual
# TUI (``cli/tui_interactive.py``'s channel handler), and headless
# serve (``cli/commands.py::_serve_process_message``).  They all route
# ``/foo`` text through ``cmd_manager`` instead of feeding it to the
# LLM as a plain prompt.


async def dispatch_channel_slash_command(
    msg: ChannelMessage,
    *,
    agent: Any,
    thread_id: str,
    workspace_dir: str | None,
    checkpointer: Any,
    append_system: Callable[[str, str], None],
    start_new_session_cb: Callable[[], None] | None = None,
    handle_session_resume_cb: Callable[..., Awaitable[None]] | None = None,
    await_agent_ready: Callable[[], Awaitable[Any]] | None = None,
    on_cmd_completed: Callable[..., Awaitable[None]] | None = None,
    channel_runtime: ChannelRuntime | None = None,
) -> bool:
    """Dispatch a slash command from a channel message.

    Returns True if the helper handled the message (successfully or with
    an error) — the caller must then return without streaming anything
    to the agent.  Returns False for non-slash content or unresolved
    slash commands, so the caller can fall through to the agent
    streaming path (matches TUI behavior).

    Parameters
    ----------
    msg:
        The inbound ``ChannelMessage`` to inspect.
    agent:
        Default agent handle for the ``CommandContext``.  Commands that
        do not need the agent use this value directly.
    thread_id, workspace_dir, checkpointer:
        Populate ``CommandContext``.
    append_system:
        ``(text, style)`` callback for local CLI/TUI log output.  Used
        by ``ChannelCommandUI`` to surface system breadcrumbs and by
        this helper to print the "Executed command from ..." line.
    start_new_session_cb, handle_session_resume_cb:
        Optional lifecycle callbacks forwarded to ``ChannelCommandUI``.
        Headless serve passes ``None`` — ``/new`` and ``/resume`` degrade
        gracefully via the default ``ChannelCommandUI`` messages.
    await_agent_ready:
        Optional async resolver that blocks until the background agent
        load finishes.  Called only when ``cmd.needs_agent(args)`` is
        True.  Headless serve passes ``None`` because the agent is
        loaded up-front before the bus starts.
    on_cmd_completed:
        Optional ``async (ctx, original_agent, cmd) -> None`` callback
        fired only after ``cmd_manager.execute`` returns True.  The
        ``original_agent`` argument is the agent handle command execution
        started against: ``agent_for_ctx`` after any ``await_agent_ready``
        resolution, or the dispatcher's input agent when no resolver is
        supplied.  Callers can compare ``ctx.agent`` with
        ``original_agent`` to detect command-driven swaps.  Used by Rich
        CLI to (a) adopt an agent swap (``/model``) back into the
        running session and (b) refresh the status snapshot for
        commands that mutate session-level state (``/new``,
        ``/compact``) — mirrors the REPL dispatch at
        ``cli/interactive.py:1002-1030``.  Headless serve passes
        ``None`` since it cannot hot-swap its polling-loop agent.
    """
    if not msg.content.strip().startswith("/"):
        return False

    try:
        return await _dispatch_channel_slash_impl(
            msg,
            agent=agent,
            thread_id=thread_id,
            workspace_dir=workspace_dir,
            checkpointer=checkpointer,
            append_system=append_system,
            start_new_session_cb=start_new_session_cb,
            handle_session_resume_cb=handle_session_resume_cb,
            await_agent_ready=await_agent_ready,
            on_cmd_completed=on_cmd_completed,
            channel_runtime=channel_runtime,
        )
    except Exception as exc:
        # Last-ditch safety: any uncaught exception from inside the
        # dispatch pipeline (lazy import failure, ChannelCommandUI
        # construction, terminal I/O from ``append_system``, bus
        # publish races, ...) must not take down the caller's polling
        # loop — a crashed serve / dead channel queue task is worse
        # than one failed command.
        _channel_logger.exception(
            "Unexpected slash dispatch failure for %s (msg=%s)",
            msg.channel_type,
            msg.msg_id,
        )
        try:
            _set_channel_response(msg.msg_id, f"Command error: {exc}")
        except Exception:  # pragma: no cover — defensive
            pass
        # Return True so the caller treats the message as handled and
        # does not fall through to the agent streaming path.
        return True


async def _dispatch_channel_slash_impl(
    msg: ChannelMessage,
    *,
    agent: Any,
    thread_id: str,
    workspace_dir: str | None,
    checkpointer: Any,
    append_system: Callable[[str, str], None],
    start_new_session_cb: Callable[[], None] | None,
    handle_session_resume_cb: Callable[..., Awaitable[None]] | None,
    await_agent_ready: Callable[[], Awaitable[Any]] | None,
    on_cmd_completed: Callable[..., Awaitable[None]] | None,
    channel_runtime: ChannelRuntime | None,
) -> bool:
    """Inner body of ``dispatch_channel_slash_command``.

    Split from the public wrapper so the wrapper can guard with a
    top-level try/except without visually obscuring the main flow.
    """
    # Lazy imports: avoid coupling the channel module to ``commands`` at
    # import time (tui_interactive.py does the same).
    from ..commands.base import CommandContext
    from ..commands.channel_ui import ChannelCommandUI
    from ..commands.manager import manager as cmd_manager

    parsed = cmd_manager.resolve(msg.content)
    if parsed is None:
        # Unknown slash command — let the agent handle it (matches TUI).
        return False
    cmd, cmd_args = parsed

    agent_for_ctx = agent
    if cmd.needs_agent(cmd_args) and await_agent_ready is not None:
        try:
            agent_for_ctx = await await_agent_ready()
        except Exception as exc:
            _set_channel_response(msg.msg_id, f"Command error: {exc}")
            return True

    ui = ChannelCommandUI(
        msg,
        append_system_callback=append_system,
        start_new_session_callback=start_new_session_cb,
        handle_session_resume_callback=handle_session_resume_cb,
    )
    ctx = CommandContext(
        agent=agent_for_ctx,
        thread_id=thread_id,
        ui=ui,
        workspace_dir=workspace_dir,
        checkpointer=checkpointer,
        channel_runtime=channel_runtime,
    )

    try:
        cmd_executed = await cmd_manager.execute(msg.content, ctx)
    except Exception as exc:
        _channel_logger.debug(f"Channel command error: {exc}", exc_info=True)
        _set_channel_response(msg.msg_id, f"Command error: {exc}")
        return True  # must return — do NOT fall through to the agent

    if cmd_executed:
        if on_cmd_completed is not None:
            try:
                # Command output already flushed by ``cmd_manager.execute``
                # via ``ctx.ui.flush()`` — the hook does internal state
                # sync (agent adoption, status snapshot refresh) only,
                # so swallowing its errors keeps the user-visible reply
                # intact even if the sync path is broken.
                await on_cmd_completed(ctx, agent_for_ctx, cmd)
            except Exception as exc:
                _channel_logger.debug(
                    f"Channel command post-exec callback error: {exc}",
                    exc_info=True,
                )
        append_system(
            f"[{msg.channel_type}: Executed command from {msg.sender}]",
            "dim",
        )
        _set_channel_response(msg.msg_id, f"Command executed: {msg.content}")
        return True

    # ``cmd_manager.execute`` returned False (empty / unparseable input).
    # Fall through to the agent streaming path.
    return False


# ---------------------------------------------------------------------------
# HITL approval intercept: bus thread ⇄ main CLI thread
# ---------------------------------------------------------------------------
# When the main thread needs HITL approval from a channel user, it registers
# a pending HITL wait for (channel, chat_id).  The bus consumer checks this
# BEFORE normal enqueue, so the next reply from that user is intercepted.

_pending_hitl: dict[str, dict] = {}  # "channel:chat_id" -> {event, reply}
_hitl_lock = threading.Lock()
_hitl_auto_approve: set[str] = set()  # "channel:chat_id" keys with auto-approve

_HITL_APPROVAL_TIMEOUT = 120.0  # seconds to wait for HITL approval reply
_ASK_USER_TIMEOUT = (
    300.0  # seconds to wait for ask_user reply (longer for thinking time)
)
_STOP_COMMANDS = frozenset(("/stop", "/cancel"))


def _is_stop_command(content: str | None) -> bool:
    """Whether incoming content is a stop/cancel slash command."""
    return (content or "").strip().lower() in _STOP_COMMANDS


def _register_hitl_wait(channel_type: str, chat_id: str) -> threading.Event:
    """Register a pending HITL wait.  Returns a threading.Event to block on."""
    key = f"{channel_type}:{chat_id}"
    event = threading.Event()
    with _hitl_lock:
        _pending_hitl[key] = {"event": event, "reply": None}
    return event


def _pop_hitl_reply(channel_type: str, chat_id: str) -> str | None:
    """Pop and return the HITL reply (or None if not set)."""
    key = f"{channel_type}:{chat_id}"
    with _hitl_lock:
        slot = _pending_hitl.pop(key, None)
    return slot["reply"] if slot else None


def _try_set_hitl_reply(channel_type: str, chat_id: str, content: str) -> bool:
    """Try to intercept a message as a HITL reply.  Returns True if consumed."""
    key = f"{channel_type}:{chat_id}"
    with _hitl_lock:
        slot = _pending_hitl.get(key)
        if slot:
            slot["reply"] = content
            slot["event"].set()
            return True
    return False


def channel_ask_user_prompt(
    ask_user_data: dict,
    msg: "ChannelMessage | None" = None,
) -> dict:
    """Format ask_user questions and collect answers from a channel user.

    If *msg* is provided, sends questions via the bus and waits for a reply.
    Otherwise falls back to returning a cancelled result.

    Returns:
        ``{"answers": [...], "status": "answered"}`` or
        ``{"status": "cancelled"}``.
    """
    from ..channels.bus.events import OutboundMessage

    questions = ask_user_data.get("questions", [])
    if not questions:
        return {"answers": [], "status": "answered"}

    if msg is None or not msg.bus_ref:
        return {"status": "cancelled"}

    bus_loop = _bus_loop
    if not bus_loop:
        return {"status": "cancelled"}

    def _send(content: str) -> bool:
        try:
            asyncio.run_coroutine_threadsafe(
                msg.bus_ref.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel_type,
                        chat_id=msg.chat_id,
                        content=content,
                        metadata=msg.metadata,
                    )
                ),
                bus_loop,
            ).result(timeout=15)
            return True
        except Exception as exc:
            _channel_logger.debug("ask_user send failed: %s", exc)
            return False

    # Ask one question at a time (consistent with Rich CLI / TUI)
    total = len(questions)
    answers: list[str] = []

    for i, q in enumerate(questions):
        q_text = q.get("question", "")
        q_type = q.get("type", "text")
        required = q.get("required", True)

        # Format single question
        if total == 1:
            header = "\u2753 Quick check-in from EvoScientist\n"
        else:
            header = f"\u2753 Question {i + 1}/{total}\n"

        lines = [header, f"{i + 1}. {q_text}"]
        if not required:
            lines[-1] += " (optional)"

        if q_type == "multiple_choice":
            choices = q.get("choices", [])
            for j, choice in enumerate(choices):
                label = choice.get("value", str(choice))
                letter = chr(ord("A") + j)
                lines.append(f"   {letter}. {label}")
            other_letter = chr(ord("A") + len(choices))
            lines.append(f"   {other_letter}. Other")
            lines.append(
                f"\nReply with a letter ({'/'.join(chr(ord('A') + k) for k in range(len(choices) + 1))}), or 'cancel'."
            )
        else:
            skip_hint = " Leave empty to skip." if not required else ""
            lines.append(f"\nReply with your answer, or 'cancel'.{skip_hint}")

        if not _send("\n".join(lines)):
            return {"status": "cancelled"}

        # Wait for reply
        hitl_event = _register_hitl_wait(msg.channel_type, msg.chat_id)
        replied = hitl_event.wait(timeout=_ASK_USER_TIMEOUT)
        reply_text = _pop_hitl_reply(msg.channel_type, msg.chat_id)

        if not replied or not reply_text:
            _send("\u23f0 Response timed out.")
            return {"status": "cancelled"}

        raw = reply_text.strip()
        if _is_stop_command(raw):
            return {"status": "cancelled"}
        if raw.lower() == "cancel":
            return {"status": "cancelled"}

        # Parse answer
        if q_type == "multiple_choice":
            choices = q.get("choices", [])
            other_letter = chr(ord("A") + len(choices))
            if len(raw) == 1 and raw.upper() == other_letter:
                # Other selected — ask for free-form input
                if not _send("Please type your answer:"):
                    return {"status": "cancelled"}
                hitl_event = _register_hitl_wait(msg.channel_type, msg.chat_id)
                replied = hitl_event.wait(timeout=_ASK_USER_TIMEOUT)
                other_text = _pop_hitl_reply(msg.channel_type, msg.chat_id)
                if not replied or not other_text:
                    _send("\u23f0 Response timed out.")
                    return {"status": "cancelled"}
                if _is_stop_command(other_text):
                    return {"status": "cancelled"}
                if other_text.strip().lower() == "cancel":
                    return {"status": "cancelled"}
                answers.append(other_text.strip())
            elif len(raw) == 1 and raw.upper().isalpha():
                idx = ord(raw.upper()) - ord("A")
                if 0 <= idx < len(choices):
                    answers.append(choices[idx].get("value", raw))
                else:
                    answers.append(raw)
            else:
                answers.append(raw)
        else:
            answers.append(raw)

    return {"answers": answers, "status": "answered"}


def channel_hitl_prompt(
    action_requests: list,
    msg: "ChannelMessage",
) -> list[dict] | None:
    """Send HITL approval prompt to channel user and wait for reply.

    Blocking function — uses threading.Event.wait().  Safe to call from a
    background thread (CLI channel processing or asyncio.to_thread in TUI).

    Returns approval decisions list on approve/auto, or None on reject/timeout.
    """
    from ..channels.bus.events import OutboundMessage
    from ..channels.consumer import (
        _format_approval_prompt,
        _parse_approval_reply,
    )

    # Check session auto-approve (set by a previous "3" reply)
    session_key = f"{msg.channel_type}:{msg.chat_id}"
    if session_key in _hitl_auto_approve:
        return [{"type": "approve"} for _ in action_requests]

    bus_loop = _bus_loop
    if not (bus_loop and msg.bus_ref):
        _channel_logger.debug("HITL: no bus_loop or bus_ref, rejecting")
        return None

    def _send(content: str) -> bool:
        """Send a message to the channel user.  Returns True on success."""
        try:
            asyncio.run_coroutine_threadsafe(
                msg.bus_ref.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel_type,
                        chat_id=msg.chat_id,
                        content=content,
                        metadata=msg.metadata,
                    )
                ),
                bus_loop,
            ).result(timeout=15)
            return True
        except Exception as exc:
            _channel_logger.debug("HITL send failed: %s", exc)
            return False

    # 1. Send approval prompt
    prompt_text = _format_approval_prompt(action_requests)
    if not _send(prompt_text):
        return None

    # 2. Wait for channel user's reply
    hitl_event = _register_hitl_wait(msg.channel_type, msg.chat_id)
    replied = hitl_event.wait(timeout=_HITL_APPROVAL_TIMEOUT)
    reply_text = _pop_hitl_reply(msg.channel_type, msg.chat_id)

    if not replied or not reply_text:
        _send("\u23f0 Approval timed out. Action rejected.")
        return None

    if _is_stop_command(reply_text):
        # `/stop` already got its own immediate ack from the bus fast-path.
        # Treat it as a pure cancel signal here so we don't send a second,
        # contradictory "Unrecognized reply" message.
        return None

    # 3. Parse decision
    decision = _parse_approval_reply(reply_text)
    if decision == "auto":
        _hitl_auto_approve.add(session_key)
        return [{"type": "approve"} for _ in action_requests]
    if decision == "approve":
        return [{"type": "approve"} for _ in action_requests]

    feedback = (
        "Action rejected."
        if decision == "reject"
        else "Unrecognized reply. Action rejected."
    )
    _send(feedback)
    return None


# ---------------------------------------------------------------------------
# Module-level channel state (bus mode)
# ---------------------------------------------------------------------------

_manager: Any | None = None  # ChannelManager
_bus_loop: asyncio.AbstractEventLoop | None = None
_bus_thread: threading.Thread | None = None


def _channels_is_running(channel_type: str | None = None) -> bool:
    """Check whether channels are running."""
    if _manager is None:
        return False
    if channel_type:
        ch = _manager.get_channel(channel_type)
        return ch is not None and ch._running
    return _manager.is_running and bool(_manager.running_channels())


def _channels_running_list() -> list[str]:
    """Return names of running channels."""
    return _manager.running_channels() if _manager else []


def _channels_stop(
    channel_type: str | None = None,
    *,
    runtime: ChannelRuntime | None = None,
) -> None:
    """Stop channel(s) and clean up module-level state.

    ``runtime`` is the ``ChannelRuntime`` whose binding should be
    cleared once the channels are gone — the caller owns it (commands
    keep a reference via ``ctx.channel_runtime``).
    """
    global _manager, _bus_loop, _bus_thread

    if channel_type is None:
        # Stop everything
        if _bus_loop and _manager:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _manager.stop_all(),
                    _bus_loop,
                )
                future.result(timeout=10)
            except Exception as e:
                _channel_logger.debug(f"Error stopping channels: {e}")
        if _manager:
            _manager.bus.stop()
        if _bus_thread:
            _bus_thread.join(timeout=5)
        _manager = None
        _bus_loop = None
        _bus_thread = None
        if runtime is not None:
            runtime.clear()
        return

    # Stop a specific channel
    if _manager and _bus_loop:
        try:
            future = asyncio.run_coroutine_threadsafe(
                _manager.remove_channel(channel_type),
                _bus_loop,
            )
            future.result(timeout=5)
        except Exception as e:
            _channel_logger.debug(f"Error removing channel {channel_type}: {e}")

    if _manager and not _manager.running_channels() and runtime is not None:
        runtime.clear()


def _start_channels_bus_mode(
    config,
    agent,
    thread_id: str,
    *,
    send_thinking: bool | None = None,
) -> None:
    """Start all channels in bus mode with MessageBus + ChannelManager.

    Creates a single event loop in a daemon thread running the bus,
    ChannelManager, and the inbound consumer.
    """
    global _manager, _bus_loop, _bus_thread

    from ..channels.channel_manager import ChannelManager

    mgr = ChannelManager.from_config(config)

    effective_send_thinking = (
        getattr(config, "channel_send_thinking", True)
        if send_thinking is None
        else send_thinking
    )
    for channel in mgr._channels.values():
        channel.send_thinking = bool(effective_send_thinking)

    _manager = mgr

    def _bus_thread_entry():
        global _bus_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _bus_loop = loop

        async def _run():
            consumer = asyncio.create_task(_bus_inbound_consumer(mgr.bus, mgr))
            try:
                await mgr.start_all()
            finally:
                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass

        try:
            loop.run_until_complete(_run())
        except Exception as e:
            _channel_logger.error(
                "Bus thread terminated with error: %s", e, exc_info=True
            )
        finally:
            _channel_logger.debug("Bus thread event loop closed")
            loop.close()

    thread = threading.Thread(target=_bus_thread_entry, daemon=True)
    _bus_thread = thread
    thread.start()

    # Wait briefly for the loop to start
    for _ in range(20):
        if _bus_loop is not None:
            break
        time.sleep(0.1)


def _add_channel_to_running_bus(
    channel_type: str,
    config,
    *,
    send_thinking: bool | None = None,
) -> None:
    """Dynamically add a single channel to the already-running bus.

    Raises:
        RuntimeError: If the bus loop or manager is not initialised.
        ValueError: If the channel type is unknown or already registered.
    """
    if not _manager or not _bus_loop:
        raise RuntimeError("Bus not initialised")

    effective_send_thinking = (
        getattr(config, "channel_send_thinking", True)
        if send_thinking is None
        else send_thinking
    )

    async def _do_add():
        channel = await _manager.add_channel(channel_type, config)
        channel.send_thinking = bool(effective_send_thinking)

    future = asyncio.run_coroutine_threadsafe(_do_add(), _bus_loop)
    future.result(timeout=10)


async def _bus_inbound_consumer(bus, manager) -> None:
    """Consume inbound messages from bus and bridge to the main CLI thread.

    Task-based: each inbound message is handled in its own asyncio task
    so the consumer loop stays responsive for HITL approval replies.
    """
    _tasks: set[asyncio.Task] = set()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # /stop should preempt HITL interception so cancel works while
            # waiting for approvals/questions.  If a HITL wait is pending,
            # still release it so the blocking prompt can unwind immediately.
            if _is_stop_command(msg.content):
                if _try_set_hitl_reply(msg.channel, msg.chat_id, msg.content):
                    _channel_logger.info(
                        f"[bus] stop request released HITL wait for "
                        f"{msg.channel}:{msg.chat_id}"
                    )
                _task = asyncio.create_task(_handle_bus_message(bus, manager, msg))
                _tasks.add(_task)
                _task.add_done_callback(_tasks.discard)
                continue

            # Check if this message is a HITL approval reply
            if _try_set_hitl_reply(msg.channel, msg.chat_id, msg.content):
                _channel_logger.info(
                    f"[bus] HITL reply from {msg.channel}:{msg.sender_id}: "
                    f"{msg.content[:60]}"
                )
                continue

            # Regular message — handle in a separate task
            _task = asyncio.create_task(_handle_bus_message(bus, manager, msg))
            _tasks.add(_task)
            _task.add_done_callback(_tasks.discard)
    finally:
        for task in list(_tasks):
            task.cancel()
        if _tasks:
            await asyncio.gather(*_tasks, return_exceptions=True)


async def _handle_bus_message(bus, manager, msg) -> None:
    """Handle a single inbound bus message (runs as an independent task)."""
    from ..channels.bus.events import OutboundMessage

    _channel_logger.info(
        f"[bus] Received from {msg.channel}:{msg.sender_id}: {msg.content[:60]}..."
    )
    manager.record_message(msg.channel, "received")

    # Fast-path: /stop intercept. Handle on the bus task itself so we
    # don't deadlock behind the main-thread stream we're trying to
    # interrupt. No typing indicator, no queue entry.
    if _is_stop_command(msg.content):
        cancelled_count, active_count = _cancel_channel_session(
            msg.channel, msg.chat_id
        )
        try:
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Stopped.",
                    reply_to=msg.message_id or None,
                    metadata=msg.metadata,
                )
            )
            manager.record_message(msg.channel, "sent")
        except Exception as e:
            _channel_logger.error(f"[bus] /stop ack send error: {e}")
        else:
            if cancelled_count or active_count:
                _channel_logger.info(
                    "[bus] /stop cancelled %d request(s) (%d active) for %s:%s",
                    cancelled_count,
                    active_count,
                    msg.channel,
                    msg.chat_id,
                )
        return

    channel = manager.get_channel(msg.channel)
    typing_active = False
    if channel:
        await channel.start_typing(msg.chat_id)
        typing_active = True

    # Enqueue for main CLI thread to process with its own event loop
    cm = ChannelMessage(
        msg_id=str(uuid.uuid4()),
        content=msg.content,
        sender=msg.sender_id,
        channel_type=msg.channel,
        metadata=msg.metadata,
        channel_ref=channel,
        bus_ref=bus,
        chat_id=msg.chat_id,
        message_id=msg.message_id,
    )
    response_waiter = _enqueue_channel_message(cm)

    try:
        # Two-stage wait: first stage with timeout, then extended wait for late reply
        try:
            await asyncio.wait_for(
                asyncio.shield(response_waiter),
                timeout=_RESPONSE_TIMEOUT,
            )
            replied = True
        except TimeoutError:
            replied = False

        if not replied:
            _channel_logger.warning(
                f"[bus] Response timeout ({_RESPONSE_TIMEOUT}s) for {cm.msg_id}; "
                "keeping late-reply delivery active"
            )
            try:
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=_LATE_RESPONSE_NOTICE,
                        reply_to=msg.message_id or None,
                        metadata=msg.metadata,
                    )
                )
                manager.record_message(msg.channel, "sent")
            except Exception as e:
                _channel_logger.error(f"[bus] Late notice send error: {e}")
            if channel and typing_active:
                await channel.stop_typing(msg.chat_id)
                typing_active = False

            # Keep waiting for the actual response
            try:
                await asyncio.wait_for(
                    asyncio.shield(response_waiter),
                    timeout=_LATE_RESPONSE_TIMEOUT,
                )
                replied = True
            except TimeoutError:
                replied = False

            if not replied:
                _channel_logger.warning(
                    f"[bus] Late response timeout ({_LATE_RESPONSE_TIMEOUT}s) "
                    f"for {cm.msg_id}"
                )
                _pop_channel_response(cm.msg_id, cancel_pending=True)
                if _channel_request_state(cm.msg_id) != "active":
                    _complete_channel_request(cm.msg_id)
                return

        response = _pop_channel_response(cm.msg_id) or "No response"
        await bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=response,
                reply_to=msg.message_id or None,
                metadata=msg.metadata,
            )
        )
        manager.record_message(msg.channel, "sent")
    except asyncio.CancelledError:
        _pop_channel_response(cm.msg_id, cancel_pending=True)
        if _channel_request_state(cm.msg_id) != "active":
            _complete_channel_request(cm.msg_id)
        raise
    except Exception as e:
        _channel_logger.error(f"[bus] Outbound error: {e}")
    finally:
        if channel and typing_active:
            await channel.stop_typing(msg.chat_id)


def _print_channel_panel(channels: list[tuple[str, bool, str]]) -> None:
    """Print a summary panel for active channels.

    Args:
        channels: List of (name, ok, detail) tuples.
    """
    lines: list[Text] = []
    all_ok = True
    for name, ok, detail in channels:
        line = Text()
        if ok:
            line.append("\u25cf ", style="green")
            line.append(name, style="bold")
        else:
            line.append("\u2717 ", style="yellow")
            line.append(name, style="bold yellow")
            all_ok = False
        if detail:
            line.append(f"  {detail}", style="dim")
        lines.append(line)

    body = Text("\n").join(lines)
    border = "green" if all_ok else "yellow"
    console.print(
        Panel(body, title="[bold]Channels[/bold]", border_style=border, expand=False)
    )
    console.print()


def _auto_start_channel(
    agent: Any,
    thread_id: str,
    config,
    *,
    send_thinking: bool | None = None,
    runtime: ChannelRuntime | None = None,
) -> None:
    """Start channels automatically from config (bus mode).

    Args:
        agent: Compiled agent graph.
        thread_id: Current thread ID.
        config: EvoScientistConfig with channel settings.
        runtime: Caller-owned ``ChannelRuntime`` to bind so commands
            running over the channels can swap the agent later.  ``None``
            is accepted for callers that don't yet pass one.
    """
    if not config.channel_enabled:
        return

    if runtime is not None:
        runtime.bind(agent, thread_id)

    _start_channels_bus_mode(
        config,
        agent,
        thread_id,
        send_thinking=send_thinking,
    )
    types = [t.strip() for t in config.channel_enabled.split(",") if t.strip()]
    results = [(ct, True, "connected (bus)") for ct in types]
    _print_channel_panel(results)
