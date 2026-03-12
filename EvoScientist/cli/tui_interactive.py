"""Full-screen Textual interactive TUI for EvoScientist.

Widget-based rendering: each message/tool/sub-agent is an independent widget
mounted into a VerticalScroll container.  No timer-based Group rebuilds.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import random
import shlex
from typing import Any, Callable

from rich.console import Group
from rich.table import Table
from rich.text import Text

import EvoScientist.cli.channel as _ch_mod
from .channel import (
    ChannelMessage,
    _channels_is_running,
    _channels_running_list,
    _channels_stop,
    _auto_start_channel,
    _message_queue,
    _set_channel_response,
)
from ..sessions import (
    _format_relative_time,
    delete_thread,
    find_similar_threads,
    generate_thread_id,
    get_checkpointer,
    get_thread_messages,
    get_thread_metadata,
    list_threads,
    thread_exists,
)
from ..config.settings import get_config_dir
from ..stream.events import stream_agent_events
from ..stream.state import StreamState, _INTERNAL_TOOLS
from .history_suggester import HistorySuggester

from ._constants import LOGO_LINES, LOGO_GRADIENT, WELCOME_SLOGANS, build_metadata

_channel_logger = logging.getLogger(__name__)

_TUI_SLASH_COMMANDS = [
    ("/current", "Show current session info"),
    ("/threads", "List recent sessions"),
    ("/resume", "Resume a previous session"),
    ("/delete", "Delete a saved session"),
    ("/new", "Start a new session"),
    ("/clear", "Clear chat history"),
    ("/skills", "List installed skills"),
    ("/install-skill", "Add a skill from path or GitHub"),
    ("/uninstall-skill", "Remove an installed skill"),
    ("/mcp", "Manage MCP servers"),
    ("/channel", "Configure messaging channels"),
    ("/help", "Show available commands"),
    ("/exit", "Quit EvoScientist"),
]


def _shorten_path(path: str) -> str:
    """Shorten absolute path to a cwd-relative form (consistent with Rich CLI)."""
    if not path:
        return path
    from .agent import _shorten_path as _sp
    return _sp(path)



def _build_welcome_banner(
    *,
    thread_id: str,
    workspace_dir: str | None,
    mode: str | None,
    model: str | None,
    provider: str | None,
    ui_backend: str | None = None,
    channels: list[tuple[str, bool, str]] | None = None,
) -> Any:
    """Build CLI-matching welcome banner with logo, info line, and channels.

    Args:
        channels: List of (name, ok, detail) tuples for the channels panel.
    """
    banner = Text()
    for line, color in zip(LOGO_LINES, LOGO_GRADIENT):
        banner.append(f"{line}\n", style=f"bold {color}")

    # Info line — matches CLI print_banner format
    info = Text()
    parts: list[tuple[str, str]] = []
    if model:
        parts.append(("Model: ", model))
    if provider:
        parts.append(("Provider: ", provider))
    if mode:
        parts.append(("Mode: ", mode))
    if ui_backend:
        parts.append(("UI: ", ui_backend))
    if parts:
        info.append("  ", style="dim")
        for i, (label, value) in enumerate(parts):
            if i > 0:
                info.append("  ", style="dim")
            info.append(label, style="dim")
            info.append(value, style="magenta")
    # Directory line
    import os
    effective_dir = workspace_dir or os.getcwd()
    home = os.path.expanduser("~")
    dir_display = effective_dir.replace(home, "~", 1) if effective_dir.startswith(home) else effective_dir
    info.append("\n  ", style="dim")
    info.append("Directory: ", style="dim")
    info.append(dir_display, style="magenta")
    info.append("\n  Type ", style="#ffe082")
    info.append("/", style="#ffe082 bold")
    info.append(" for commands", style="#ffe082")
    banner.append_text(info)

    slogan = Text(f"\n  {random.choice(WELCOME_SLOGANS)}", style="dim italic")

    # Channels panel
    if channels:
        from rich.panel import Panel

        lines: list[Text] = []
        all_ok = True
        for name, ok, detail in channels:
            line = Text()
            if ok:
                line.append("\u25cf ", style="green")
                line.append(name, style="bold")
            else:
                line.append("\u25cb ", style="dim")
                line.append(name, style="bold dim")
                all_ok = False
            if detail:
                line.append(f"  {detail}", style="dim")
            lines.append(line)
        body = Text("\n").join(lines)
        border = "green" if all_ok else "dim"
        panel = Panel(body, title="[bold]Channels[/bold]", border_style=border, expand=False)
        return Group(banner, Text(""), panel, slogan)

    # No channels — append slogan directly to banner
    banner.append_text(slogan)
    return banner


def _is_final_response(state: StreamState) -> bool:
    """Check if all tools are done and no sub-agents are active."""
    n_visible = 0
    n_done = 0
    for i, tc in enumerate(state.tool_calls):
        if tc.get("name") in _INTERNAL_TOOLS:
            continue
        n_visible += 1
        if i < len(state.tool_results):
            n_done += 1
    has_pending = n_visible > n_done
    any_active_sa = any(sa.is_active for sa in state.subagents)
    return not has_pending and not any_active_sa and not state.is_processing


def run_textual_interactive(
    *,
    show_thinking: bool,
    channel_send_thinking: bool = True,
    workspace_dir: str | None,
    workspace_fixed: bool,
    mode: str | None,
    model: str | None,
    provider: str | None,
    run_name: str | None,
    thread_id: str | None,
    load_agent: Callable[..., Any],
    create_session_workspace: Callable[[str | None], str],
) -> None:
    """Run full-screen Textual interactive chat loop."""
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container, Horizontal, VerticalScroll
        from textual.events import MouseUp
        from textual.widgets import Input, Static

        from .clipboard import copy_selection_to_clipboard
        from .widgets import (
            LoadingWidget,
            ThinkingWidget,
            SummarizationWidget,
            AssistantMessage,
            ToolCallWidget,
            SubAgentWidget,
            TodoWidget,
            UserMessage,
            SystemMessage,
            UsageWidget,
        )
    except Exception as e:  # pragma: no cover - runtime fallback path
        raise RuntimeError(
            "Textual TUI backend requires 'textual'. Run: pip install textual"
        ) from e

    class EvoTextualInteractiveApp(App[None]):  # type: ignore[type-arg]
        """Deep-Agents-style full-screen TUI with independent widget rendering."""

        CSS = """
        Screen {
            layout: vertical;
            background: #16161a;
            color: #d1d5db;
        }
        #chat {
            height: 1fr;
            padding: 1 2;
            background: #16161a;
        }
        #welcome {
            height: auto;
            margin-bottom: 1;
        }
        #input-shell {
            height: auto;
            padding: 0 2 1 2;
            background: #16161a;
        }
        #input-row {
            height: 3;
            border: solid #0284c7;
            background: #1e1f26;
            padding: 0 1;
        }
        #input-cursor {
            width: 2;
            content-align: center middle;
            color: #0284c7;
            text-style: bold;
        }
        #prompt {
            width: 1fr;
            border: none;
            background: transparent;
            color: #e5e7eb;
        }
        #prompt:focus {
            border: none;
        }
        #queued-message {
            display: none;
            height: auto;
            background: #1e1f26;
            padding: 0 2;
            color: #9ca3af;
        }
        #completions {
            display: none;
            height: auto;
            max-height: 15;
            background: #1e1f26;
            padding: 0 1;
            border-bottom: solid #0284c7;
        }
        #status {
            height: 1;
            background: #171a20;
            color: #f59e0b;
            padding: 0 1;
        }
        """
        BINDINGS = [
            Binding("ctrl+c", "request_quit", "Quit", show=False),
            Binding("up", "edit_queued", show=False, priority=True),
            Binding("down", "down_delegate", show=False, priority=True),
            Binding("escape", "cancel_queued", show=False, priority=True),
        ]

        def __init__(
            self,
            *,
            agent: Any,
            thread_id_value: str,
            workspace: str | None,
            checkpointer: Any,
            channel_send_thinking_value: bool = True,
            resumed: bool = False,
            resume_warning: str = "",
        ) -> None:
            super().__init__()
            self._agent = agent
            self._conversation_tid = thread_id_value
            self._workspace_dir = workspace
            self._checkpointer = checkpointer
            self._channel_send_thinking = channel_send_thinking_value
            self._resumed = resumed
            self._resume_warning = resume_warning
            self._channel_timer: Any = None
            self._started_channel_types: list[str] = []
            self._busy = False
            self._run_task: Any = None  # asyncio.Task for current _run_turn
            self._queued_messages: list[str] = []  # queued messages to send after current turn
            self._comp_items: list[tuple[str, str]] = []
            self._comp_index: int = -1
            self._hitl_auto_approve: bool = False
            self._approval_future: asyncio.Future | None = None
            self._ask_user_future: asyncio.Future | None = None
            self._picker_future: asyncio.Future | None = None
            self._history_suggester = HistorySuggester(get_config_dir() / "history")

        # ── Layout ─────────────────────────────────────────────

        def compose(self) -> ComposeResult:
            with VerticalScroll(id="chat"):
                yield Static("", id="welcome")
                # Widgets are mounted directly here by _stream_with_widgets,
                # _append_system, _mount_renderable, etc.

            with Container(id="input-shell"):
                yield Static("", id="queued-message")
                yield Static("", id="completions")
                with Horizontal(id="input-row"):
                    yield Static(">", id="input-cursor")
                    yield Input(
                        placeholder="Type message (/ for commands)",
                        id="prompt",
                        suggester=self._history_suggester,
                    )

            yield Static("", id="status")

        def on_mount(self) -> None:
            self._render_welcome()
            self._render_status()
            self.query_one("#prompt", Input).focus()
            # Show resume status
            if self._resume_warning:
                self._append_system(self._resume_warning, style="yellow")
            elif self._resumed:
                self._append_system(
                    f"Resumed session: {self._conversation_tid}", style="green",
                )
                self.call_later(
                    lambda: asyncio.ensure_future(self._render_history(self._conversation_tid))
                )
            # Auto-start channels
            self._start_channels()

        # ── Channel integration ────────────────────────────────

        def _start_channels(self) -> None:
            """Auto-start channels if enabled in config."""
            try:
                from ..config import load_config

                cfg = load_config()
                if cfg and cfg.channel_enabled and not _channels_is_running():
                    _auto_start_channel(
                        self._agent,
                        self._conversation_tid,
                        cfg,
                        send_thinking=self._channel_send_thinking,
                    )
                    types = [
                        t.strip()
                        for t in cfg.channel_enabled.split(",")
                        if t.strip()
                    ]
                    self._started_channel_types = types
                    self._render_welcome()
            except Exception as e:
                _channel_logger.debug(f"Channel auto-start failed: {e}")
            self._channel_timer = self.set_interval(0.1, self._poll_channel_queue)

        def _poll_channel_queue(self) -> None:
            """Poll the channel message queue (called every 100ms)."""
            try:
                msg = _message_queue.get_nowait()
            except queue.Empty:
                return
            if self._busy:
                _message_queue.put(msg)
                return
            self.call_later(lambda m=msg: asyncio.ensure_future(self._process_channel_message(m)))

        # ── Widget helpers ─────────────────────────────────────

        def _append_system(self, text: str, *, style: str = "dim") -> None:
            """Mount a SystemMessage widget into #chat."""
            container = self.query_one("#chat", VerticalScroll)
            container.mount(SystemMessage(text, msg_style=style))
            container.scroll_end(animate=False)

        def _mount_renderable(self, renderable: Any) -> None:
            """Mount a Rich renderable (e.g. Table) as a Static widget."""
            container = self.query_one("#chat", VerticalScroll)
            container.mount(Static(renderable))
            container.scroll_end(animate=False)

        async def _wait_for_approval(self, approval_widget) -> Any:
            """Wait for user to interact with an ApprovalWidget.

            Returns the ``ApprovalWidget.Decided`` message, or ``None`` on
            timeout / cancellation.
            """
            self._approval_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._approval_future, timeout=300)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return None
            finally:
                self._approval_future = None

        def on_approval_widget_decided(self, event) -> None:  # type: ignore[override]
            """Handle ApprovalWidget.Decided message."""
            if self._approval_future and not self._approval_future.done():
                self._approval_future.set_result(event)

        async def _wait_for_ask_user(self, ask_w) -> dict:
            """Wait for the interactive ask_user widget to resolve via Future.

            Returns ``{"answers": [...], "status": "answered"}``
            or ``{"status": "cancelled"}``.
            """
            loop = asyncio.get_running_loop()
            self._ask_user_future = loop.create_future()
            ask_w.set_future(self._ask_user_future)

            try:
                result = await asyncio.wait_for(self._ask_user_future, timeout=300)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                ask_w.action_cancel()
                return {"status": "cancelled"}
            finally:
                self._ask_user_future = None

            if not isinstance(result, dict):
                return {"status": "cancelled"}

            result_type = result.get("type", "")
            if result_type == "answered":
                return {"answers": result.get("answers", []), "status": "answered"}
            return {"status": "cancelled"}

        async def _wait_for_thread_pick(self, picker_widget) -> str | None:
            """Wait for user to pick a thread from ThreadPickerWidget.

            Returns the selected thread_id, or ``None`` on cancel/timeout.
            """
            self._picker_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._picker_future, timeout=120)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return None
            finally:
                self._picker_future = None
                try:
                    picker_widget.remove()
                except Exception:
                    pass

        def on_thread_picker_widget_picked(self, event) -> None:  # type: ignore[override]
            """Handle ThreadPickerWidget.Picked message."""
            if self._picker_future and not self._picker_future.done():
                self._picker_future.set_result(event.thread_id)

        def on_thread_picker_widget_cancelled(self, event) -> None:  # type: ignore[override]
            """Handle ThreadPickerWidget.Cancelled message."""
            if self._picker_future and not self._picker_future.done():
                self._picker_future.set_result(None)

        # ── Streaming core ─────────────────────────────────────

        async def _stream_with_widgets(
            self,
            user_text: str,
            *,
            on_thinking_cb: Callable[[str], None] | None = None,
            on_todo_cb: Callable[[list[dict]], None] | None = None,
            on_media_cb: Callable[[str], None] | None = None,
            skip_user_message: bool = False,
            channel_hitl_fn: Callable[[list], list[dict] | None] | None = None,
            channel_ask_user_fn: Callable[[dict], dict] | None = None,
        ) -> str:
            """Stream agent events and mount widgets.  Returns response text.

            Shared by ``_run_turn`` (interactive) and
            ``_process_channel_message`` (channel).

            Args:
                skip_user_message: If True, don't mount UserMessage (caller
                    already mounted it — e.g. channel messages with labels).
                channel_hitl_fn: Optional channel-based HITL approval function.
                    When provided (channel messages), this is called instead
                    of mounting the ApprovalWidget.
                channel_ask_user_fn: Optional channel-based ask_user function.
                    When provided (channel messages), this is called instead
                    of mounting the AskUserWidget.
            """
            container = self.query_one("#chat", VerticalScroll)

            # 1. Mount user message + loading spinner
            if not skip_user_message:
                await container.mount(UserMessage(user_text))
            loading = LoadingWidget()
            await container.mount(loading)
            container.scroll_end(animate=False)

            # 2. Event-driven widget rendering
            state = StreamState()
            loading_removed = False
            thinking_w: ThinkingWidget | None = None
            summarization_w: SummarizationWidget | None = None
            assistant_w: AssistantMessage | None = None
            todo_w: TodoWidget | None = None
            tool_widgets: dict[str, ToolCallWidget] = {}
            subagent_widgets: dict[str, SubAgentWidget] = {}

            # Transient indicator widgets (auto-removed on state transitions)
            narration_w: Static | None = None   # dim italic intermediate text
            processing_w: Static | None = None  # "Analyzing results..."

            # Tool collapsing (matches CLI MAX_VISIBLE_TOOLS)
            _MAX_VISIBLE_TOOLS = 4
            completed_tool_order: list[str] = []  # tool_ids in completion order
            collapse_summary_w: Static | None = None
            has_used_tools = False

            _thinking_sent = False
            _todo_sent = False
            _media_sent: set[str] = set()
            _MIN_THINKING_LEN = 200
            _scroll_pending = False

            def _schedule_scroll() -> None:
                """Throttle scroll_end to at most once per 200ms.

                Uses call_after_refresh so the scroll happens after Textual
                finishes its layout pass — otherwise scroll_end may see
                stale widget heights and not scroll far enough.
                """
                nonlocal _scroll_pending
                if not _scroll_pending:
                    _scroll_pending = True
                    self.set_timer(0.2, _do_scroll)

            def _do_scroll() -> None:
                nonlocal _scroll_pending
                _scroll_pending = False
                self.call_after_refresh(
                    lambda: container.scroll_end(animate=False),
                )

            metadata = build_metadata(self._workspace_dir, model)
            response = ""

            async def _remove_w(w: Static | None) -> None:
                """Safely remove a transient indicator widget."""
                if w is not None:
                    try:
                        await w.remove()
                    except Exception:
                        pass

            async def _collapse_completed_tools() -> None:
                """Hide older completed tool widgets; show summary line."""
                nonlocal collapse_summary_w
                completed = [
                    (tid, tool_widgets[tid])
                    for tid in completed_tool_order
                    if tid in tool_widgets
                ]
                n = len(completed)
                if n <= _MAX_VISIBLE_TOOLS:
                    if collapse_summary_w is not None:
                        collapse_summary_w.display = False
                    return

                to_hide = n - _MAX_VISIBLE_TOOLS
                ok_count = 0
                fail_count = 0
                for i, (_, tw) in enumerate(completed):
                    if i < to_hide:
                        tw.display = False
                        if tw._status == "success":
                            ok_count += 1
                        else:
                            fail_count += 1
                    else:
                        tw.display = True

                summary = Text()
                summary.append(f"\u2713 {ok_count} completed", style="dim green")
                if fail_count > 0:
                    summary.append(f" | {fail_count} failed", style="dim red")

                if collapse_summary_w is None:
                    collapse_summary_w = Static(summary)
                    # Position before first visible tool widget
                    first_visible = None
                    for _, tw in completed[to_hide:]:
                        if tw.display:
                            first_visible = tw
                            break
                    if first_visible:
                        await container.mount(collapse_summary_w, before=first_visible)
                    else:
                        await container.mount(collapse_summary_w)
                else:
                    collapse_summary_w.update(summary)
                    collapse_summary_w.display = True

            def _find_or_rename_sa_widget(
                resolved_name: str, description: str = "",
            ) -> SubAgentWidget | None:
                """Look up a sub-agent widget, renaming 'sub-agent' entry if needed."""
                if resolved_name in subagent_widgets:
                    w = subagent_widgets[resolved_name]
                    if description and not w._description:
                        w.update_name(w._sa_name, description)
                    return w
                # Rename "sub-agent" → real name (mirrors state._get_or_create_subagent)
                if resolved_name != "sub-agent" and "sub-agent" in subagent_widgets:
                    w = subagent_widgets.pop("sub-agent")
                    w.update_name(resolved_name, description)
                    subagent_widgets[resolved_name] = w
                    return w
                return None

            _MAX_HITL_ROUNDS = 50
            _stream_input: Any = user_text  # str or Command for HITL resume

            for _hitl_round in range(_MAX_HITL_ROUNDS):
                state.pending_interrupt = None
                state.pending_ask_user = None
                _hitl_resuming = False
                # Reset per-round widgets so resumed streams get fresh ones
                if _hitl_round > 0:
                    thinking_w = None
                    summarization_w = None
                try:
                    async for event in stream_agent_events(
                        self._agent,
                        _stream_input,
                        self._conversation_tid,
                        metadata=metadata,
                    ):
                        event_type = state.handle_event(event)

                        # -- Channel callbacks (thinking, todo, media) --
                        if (
                            on_thinking_cb
                            and not _thinking_sent
                            and state.thinking_text
                            and event_type != "thinking"
                            and len(state.thinking_text) >= _MIN_THINKING_LEN
                        ):
                            on_thinking_cb(state.thinking_text.rstrip())
                            _thinking_sent = True

                        if (
                            on_todo_cb
                            and not _todo_sent
                            and event_type == "tool_call"
                            and event.get("name") == "write_todos"
                            and state.todo_items
                        ):
                            if (
                                    on_thinking_cb
                                    and not _thinking_sent
                                    and state.thinking_text
                                    and len(state.thinking_text) >= _MIN_THINKING_LEN
                            ):
                                    on_thinking_cb(state.thinking_text.rstrip())
                                    _thinking_sent = True
                            on_todo_cb(state.todo_items)
                            _todo_sent = True

                        if (
                            on_media_cb
                            and event_type == "tool_result"
                            and event.get("success")
                        ):
                            tool_name = event.get("name", "")
                            if tool_name in ("write_file", "read_file"):
                                    _forward_media_to_channel(
                                        state, tool_name, _media_sent, on_media_cb,
                                    )

                        # -- Remove loading spinner on first content event --
                        if not loading_removed and event_type in (
                            "thinking", "text", "tool_call", "summarization",
                        ):
                            await loading.cleanup()
                            loading_removed = True

                        # -- Widget dispatch --
                        if event_type == "thinking":
                            if thinking_w is None:
                                    thinking_w = ThinkingWidget(show_thinking=show_thinking)
                                    await container.mount(thinking_w)
                            thinking_w.append_text(event.get("content", ""))

                        elif event_type == "summarization":
                            content = event.get("content", "")
                            if content:
                                if summarization_w is None:
                                    summarization_w = SummarizationWidget()
                                    await container.mount(summarization_w)
                                summarization_w.set_content(content)

                        elif event_type == "text":
                            if thinking_w is not None and thinking_w._is_active:
                                    thinking_w.finalize()
                            # Clear processing indicator
                            await _remove_w(processing_w)
                            processing_w = None

                            if has_used_tools and not _is_final_response(state):
                                    # Tools still running — show intermediate narration
                                    await _remove_w(narration_w)
                                    narration_w = None
                                    last_line = state.latest_text.strip().split("\n")[-1].strip()
                                    if last_line:
                                        if len(last_line) > 60:
                                            last_line = last_line[:57] + "\u2026"
                                        narration_w = Static(
                                            Text(f"    {last_line}", style="dim italic"),
                                        )
                                        await container.mount(narration_w)
                            else:
                                    # Stream final response incrementally (both
                                    # text-only replies and post-tool responses).
                                    await _remove_w(narration_w)
                                    narration_w = None
                                    if assistant_w is None:
                                        assistant_w = AssistantMessage(state.response_text)
                                        await container.mount(assistant_w)
                                    else:
                                        await assistant_w.append_content(
                                            event.get("content", ""),
                                        )

                        elif event_type == "tool_call":
                            tool_name = event.get("name", "unknown")
                            tool_id = event.get("id", "")
                            tool_args = event.get("args", {})
                            # Finalize thinking if still active
                            if thinking_w is not None and thinking_w._is_active:
                                    thinking_w.finalize()
                            # Clear transient indicators
                            await _remove_w(narration_w)
                            narration_w = None
                            await _remove_w(processing_w)
                            processing_w = None
                            # Remove early AssistantMessage (text arrived before tools)
                            if assistant_w is not None:
                                    try:
                                        await assistant_w.remove()
                                    except Exception:
                                        pass
                                    assistant_w = None
                            # Skip internal tools and task (handled by SubAgentWidget)
                            if tool_name not in _INTERNAL_TOOLS and tool_name != "task":
                                    has_used_tools = True
                                    if tool_id and tool_id in tool_widgets:
                                        # Re-emitted with updated args — update in place
                                        existing = tool_widgets[tool_id]
                                        existing._tool_name = tool_name
                                        existing._tool_args = tool_args
                                        try:
                                            existing._render_header()
                                        except Exception:
                                            pass
                                    else:
                                        w = ToolCallWidget(tool_name, tool_args, tool_id)
                                        await container.mount(w)
                                        if tool_id:
                                            tool_widgets[tool_id] = w
                            # Update todo widget on write_todos
                            if tool_name == "write_todos" and state.todo_items:
                                    if todo_w is None:
                                        todo_w = TodoWidget(state.todo_items)
                                        await container.mount(todo_w)
                                    else:
                                        todo_w.update_items(state.todo_items)

                        elif event_type == "tool_result":
                            result_name = event.get("name", "unknown")
                            result_content = event.get("content", "")
                            result_success = event.get("success", True)
                            # Match via state's deduplicated tool_calls (uses tool_id)
                            matched = False
                            matched_tid = ""
                            result_idx = len(state.tool_results) - 1
                            if 0 <= result_idx < len(state.tool_calls):
                                    tc = state.tool_calls[result_idx]
                                    tid = tc.get("id", "")
                                    if tid and tid in tool_widgets:
                                        tw = tool_widgets[tid]
                                        if tw._status == "running":
                                            if result_success:
                                                    tw.set_success(result_content)
                                            else:
                                                    tw.set_error(result_content)
                                            matched = True
                                            matched_tid = tid
                            # Fallback: match first running widget with same name
                            if not matched:
                                    for fid, tw in tool_widgets.items():
                                        if tw.tool_name == result_name and tw._status == "running":
                                            if result_success:
                                                    tw.set_success(result_content)
                                            else:
                                                    tw.set_error(result_content)
                                            matched = True
                                            matched_tid = fid
                                            break
                            # Track completion order for collapsing
                            if matched_tid and matched_tid not in completed_tool_order:
                                    completed_tool_order.append(matched_tid)
                                    await _collapse_completed_tools()
                            # Update todo from results
                            if result_name in ("write_todos", "read_todos") and state.todo_items:
                                    if todo_w is None:
                                        todo_w = TodoWidget(state.todo_items)
                                        await container.mount(todo_w)
                                    else:
                                        todo_w.update_items(state.todo_items)
                            # Show "Analyzing results..." if all tools done, no text yet
                            if (
                                    _is_final_response(state)
                                    and not state.response_text
                                    and processing_w is None
                            ):
                                    processing_w = Static(
                                        Text("\u25cf Analyzing results...", style="cyan"),
                                    )
                                    await container.mount(processing_w)

                        elif event_type == "subagent_start":
                            sa_name = event.get("name", "sub-agent")
                            sa_desc = event.get("description", "")
                            existing = _find_or_rename_sa_widget(sa_name, sa_desc)
                            if existing is None:
                                    sa_w = SubAgentWidget(sa_name, sa_desc)
                                    await container.mount(sa_w)
                                    subagent_widgets[sa_name] = sa_w

                        elif event_type == "subagent_tool_call":
                            sa_name = event.get("subagent", "sub-agent")
                            sa_name = state._resolve_subagent_name(sa_name)
                            sa_w = _find_or_rename_sa_widget(sa_name)
                            if sa_w is None:
                                    sa_w = SubAgentWidget(sa_name)
                                    await container.mount(sa_w)
                                    subagent_widgets[sa_name] = sa_w
                            await sa_w.add_tool_call(
                                    event.get("name", "unknown"),
                                    event.get("args", {}),
                                    event.get("id", ""),
                            )

                        elif event_type == "subagent_tool_result":
                            sa_name = event.get("subagent", "sub-agent")
                            sa_name = state._resolve_subagent_name(sa_name)
                            sa_w = _find_or_rename_sa_widget(sa_name)
                            if sa_w is not None:
                                    sa_w.complete_tool(
                                        event.get("name", "unknown"),
                                        event.get("content", ""),
                                        event.get("success", True),
                                        event.get("id", ""),
                                    )

                        elif event_type == "subagent_end":
                            sa_name = event.get("name", "sub-agent")
                            sa_name = state._resolve_subagent_name(sa_name)
                            sa_w = _find_or_rename_sa_widget(sa_name)
                            if sa_w is not None:
                                    sa_w.finalize()

                        elif event_type == "ask_user":
                            questions = event.get("questions", [])
                            if questions:
                                # Channel messages: use channel-based text prompt
                                if channel_ask_user_fn is not None:
                                    self._append_system(
                                        "Waiting for channel user input...",
                                        style="dim italic",
                                    )
                                    _ask_fn = channel_ask_user_fn
                                    result = await asyncio.to_thread(
                                        lambda: _ask_fn(event),
                                    )
                                else:
                                    # Interactive TUI: display widget, collect via arrow keys
                                    from .widgets.ask_user_widget import AskUserWidget
                                    _prompt = self.query_one("#prompt", Input)
                                    _prompt.disabled = True
                                    ask_w = AskUserWidget(questions)
                                    await container.mount(ask_w)
                                    _schedule_scroll()
                                    self.call_after_refresh(ask_w.focus_active)
                                    result = await self._wait_for_ask_user(ask_w)
                                    try:
                                        await ask_w.remove()
                                    except Exception:
                                        pass
                                    _prompt.disabled = False
                                from langgraph.types import Command  # type: ignore[import-untyped]
                                _stream_input = Command(resume=result)
                                _hitl_resuming = True
                                break  # re-enter outer HITL loop

                        elif event_type == "interrupt":
                            action_reqs = event.get("action_requests", [])
                            n = len(action_reqs) or 1

                            # HITL: check session auto-approve first
                            if self._hitl_auto_approve:
                                    from langgraph.types import Command  # type: ignore[import-untyped]
                                    _stream_input = Command(resume={"decisions": [{"type": "approve"} for _ in range(n)]})
                                    _hitl_resuming = True
                                    break  # re-enter outer HITL loop

                            # Channel messages: use channel-based text approval
                            if channel_hitl_fn is not None:
                                    self._append_system(
                                        "Waiting for channel user approval...",
                                        style="dim italic",
                                    )
                                    decisions = await asyncio.to_thread(
                                        channel_hitl_fn, action_reqs,
                                    )
                                    if decisions is not None:
                                        from langgraph.types import Command  # type: ignore[import-untyped]
                                        _stream_input = Command(resume={"decisions": decisions})
                                        _hitl_resuming = True
                                        break  # re-enter outer HITL loop
                                    else:
                                        state.pending_interrupt = None
                                        for tw in tool_widgets.values():
                                            if tw._status == "running":
                                                tw.set_rejected()
                                        self._append_system(
                                            "Tool execution rejected by channel user.",
                                            style="yellow",
                                        )
                                    continue

                            # Interactive TUI: mount approval widget
                            # Disable main prompt so it can't steal focus
                            _prompt = self.query_one("#prompt", Input)
                            _prompt.disabled = True
                            from .widgets.approval_widget import ApprovalWidget
                            approval_w = ApprovalWidget(action_reqs)
                            await container.mount(approval_w)
                            _schedule_scroll()
                            decided_event = await self._wait_for_approval(approval_w)
                            await approval_w.remove()
                            _prompt.disabled = False
                            if decided_event and decided_event.decisions is not None:
                                    if decided_event.auto_approve_session:
                                        self._hitl_auto_approve = True
                                    from langgraph.types import Command  # type: ignore[import-untyped]
                                    _stream_input = Command(resume={"decisions": decided_event.decisions})
                                    _hitl_resuming = True
                                    break  # re-enter outer HITL loop with resume
                            else:
                                    state.pending_interrupt = None
                                    for tw in tool_widgets.values():
                                        if tw._status == "running":
                                            tw.set_rejected()
                                    self._append_system(
                                        "Tool execution rejected.", style="yellow",
                                    )

                        elif event_type == "done":
                            # Clean up transient indicators
                            await _remove_w(narration_w)
                            narration_w = None
                            await _remove_w(processing_w)
                            processing_w = None
                            # Mount final response
                            if assistant_w is None and state.response_text:
                                    # Strip trailing standalone "..."
                                    clean = state.response_text.strip()
                                    while clean.endswith("\n...") or clean.rstrip() == "...":
                                        clean = clean.rstrip().removesuffix("...").rstrip()
                                    assistant_w = AssistantMessage(clean or state.response_text)
                                    await container.mount(assistant_w)
                                    # Markdown rendering is async and needs multiple
                                    # layout cycles to compute final height.  Schedule
                                    # repeated deferred scrolls so long content stays
                                    # visible even when Markdown takes time to lay out.
                                    for delay in (0.15, 0.4, 0.8, 1.5):
                                        self.set_timer(
                                            delay,
                                            lambda: self.call_after_refresh(
                                                    lambda: container.scroll_end(animate=False),
                                            ),
                                        )
                            # Mount token usage stats
                            if state.total_input_tokens or state.total_output_tokens:
                                    await container.mount(
                                        UsageWidget(state.total_input_tokens, state.total_output_tokens)
                                    )

                        elif event_type == "error":
                            error_msg = event.get("message", "Unknown error")
                            self._append_system(f"Error: {error_msg}", style="red")

                        # Scroll after Textual processes the layout update
                        _schedule_scroll()

                    response = (state.response_text or "").strip()

                except asyncio.CancelledError:
                    # Ctrl+C cancellation
                    pass
                except Exception as exc:
                    error_msg = str(exc)
                    if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                        self._append_system(
                            "Error: API key not configured.", style="red",
                        )
                        self._append_system(
                            "Run EvoSci onboard to set up your API key.", style="dim",
                        )
                    else:
                        self._append_system(f"Error: {exc}", style="red")
                    response = f"Error: {exc}"
                finally:
                    # Clean up loading widget if it wasn't removed yet
                    if not loading_removed:
                        try:
                            await loading.cleanup()
                        except Exception:
                            pass
                    # Clean up transient indicators
                    for w in (narration_w, processing_w):
                        await _remove_w(w)
                    # Mark any still-running tool widgets as interrupted
                    # (skip if HITL approved — tools will continue next round)
                    if not _hitl_resuming:
                        for tw in tool_widgets.values():
                            if tw._status == "running":
                                try:
                                    tw.set_interrupted()
                                except Exception:
                                    pass
                    # Finalize any still-active sub-agents
                    for sa_w in subagent_widgets.values():
                        if sa_w._is_active:
                            try:
                                    sa_w.finalize()
                            except Exception:
                                    pass
                    # Finalize thinking widget
                    if thinking_w is not None and thinking_w._is_active:
                        try:
                            thinking_w.finalize()
                        except Exception:
                            pass
                    # Finalize assistant message stream
                    if assistant_w is not None:
                        await assistant_w.stop_stream()
                    # Flush remaining thinking callback
                    if (
                        on_thinking_cb
                        and not _thinking_sent
                        and state.thinking_text
                        and len(state.thinking_text) >= _MIN_THINKING_LEN
                    ):
                        on_thinking_cb(state.thinking_text.rstrip())
                    # Final scrolls to ensure last content is visible.
                    # Markdown layout is async — schedule multiple deferred
                    # scrolls so long content eventually scrolls into view.
                    self.call_after_refresh(
                        lambda: container.scroll_end(animate=False),
                    )
                    for delay in (0.3, 0.8):
                        self.set_timer(
                            delay,
                            lambda: self.call_after_refresh(
                                    lambda: container.scroll_end(animate=False),
                            ),
                        )

                # HITL / ask_user: if interrupt was handled, loop back to resume stream
                if state.pending_interrupt is None and state.pending_ask_user is None:
                    break  # normal completion or rejection — exit HITL loop
                # Otherwise _stream_input was set to Command(resume=...)
                # by the interrupt handler above; loop continues.

            return response

        async def _run_turn(self, user_text: str) -> None:
            """Handle a user turn: stream agent response with widgets."""
            self._busy = True
            self._render_status()
            cancelled = False

            try:
                await self._stream_with_widgets(user_text)
            except asyncio.CancelledError:
                cancelled = True
                self._append_system("Interrupted.", style="yellow")
            finally:
                self._busy = False
                self._run_task = None
                self._render_status()
                self.query_one("#prompt", Input).focus()

            # Process next queued message (FIFO) — skip if interrupted
            if not cancelled and self._queued_messages:
                next_msg = self._queued_messages.pop(0)
                self._render_queue_indicator()
                self._run_task = asyncio.ensure_future(self._run_turn(next_msg))

        async def _process_channel_message(self, msg: ChannelMessage) -> None:
            """Process a channel message: stream agent response and reply.

            Display order (matches Rich CLI):
              > message content
              [channel: Received from sender]
              (streaming response)
              [channel: Replied to sender]
            """
            self._busy = True
            self._render_status()

            prompt_widget = self.query_one("#prompt", Input)
            prompt_widget.disabled = True

            # Mount user message first, then "Received" label
            container = self.query_one("#chat", VerticalScroll)
            await container.mount(UserMessage(msg.content))
            self._append_system(
                f"[{msg.channel_type}: Received from {msg.sender}]",
                style="dim",
            )
            container.scroll_end(animate=False)

            # Build channel callbacks (fire-and-forget to avoid blocking UI)
            def _send_to_channel(coro, label: str) -> None:
                loop = _ch_mod._bus_loop
                if not loop:
                    return
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                future.add_done_callback(
                    lambda f: (
                        _channel_logger.debug(f"{label} send failed: {f.exception()}")
                        if f.exception() else None
                    )
                )

            def _send_thinking(thinking: str) -> None:
                ch = msg.channel_ref
                if ch and ch.send_thinking:
                    _send_to_channel(
                        ch.send_thinking_message(
                            sender=msg.chat_id, thinking=thinking, metadata=msg.metadata,
                        ),
                        "Thinking",
                    )

            def _send_todo(items: list[dict]) -> None:
                from ..channels.consumer import _format_todo_list

                if msg.channel_ref:
                    _send_to_channel(
                        msg.channel_ref.send_todo_message(
                            sender=msg.chat_id,
                            content=_format_todo_list(items),
                            metadata=msg.metadata,
                        ),
                        "Todo",
                    )

            def _send_media(file_path: str) -> None:
                if msg.channel_ref:
                    _send_to_channel(
                        msg.channel_ref.send_media(
                            recipient=msg.chat_id,
                            file_path=file_path,
                            metadata=msg.metadata,
                        ),
                        "Media",
                    )

            def _channel_hitl_prompt(action_requests: list) -> list[dict] | None:
                """Send HITL approval prompt to channel user and wait for reply.

                This runs in a thread (called via asyncio.to_thread) so it can
                block without freezing the Textual event loop.
                """
                return _ch_mod.channel_hitl_prompt(action_requests, msg)

            def _channel_ask_user(ask_user_data: dict) -> dict:
                """Send ask_user questions to channel user and wait for reply.

                This runs in a thread (called via asyncio.to_thread) so it can
                block without freezing the Textual event loop.
                """
                return _ch_mod.channel_ask_user_prompt(ask_user_data, msg)

            response = ""
            try:
                response = await self._stream_with_widgets(
                    msg.content,
                    on_thinking_cb=_send_thinking if self._channel_send_thinking else None,
                    on_todo_cb=_send_todo,
                    on_media_cb=_send_media,
                    skip_user_message=True,
                    channel_hitl_fn=_channel_hitl_prompt,
                    channel_ask_user_fn=_channel_ask_user,
                )
            except Exception as exc:
                response = f"Error: {exc}"
                self._append_system(f"Error: {exc}", style="red")
            finally:
                self._busy = False
                self._render_status()
                prompt_widget.disabled = False
                prompt_widget.focus()

            _set_channel_response(msg.msg_id, response)
            self._append_system(
                f"[{msg.channel_type}: Replied to {msg.sender}]",
                style="dim",
            )

        # ── Clipboard (copy on mouse select) ─────────────────

        def on_mouse_up(self, event: MouseUp) -> None:
            """Copy mouse-selected text to clipboard on release."""
            copy_selection_to_clipboard(self)

        # ── Input handling ─────────────────────────────────────

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            prompt = self.query_one("#prompt", Input)
            prompt.value = ""

            if not text:
                return

            if self._busy:
                # Queue the message to send after current turn finishes
                self._queued_messages.append(text)
                self._render_queue_indicator()
                return

            if text.startswith("/"):
                self._hide_completions()
                # Launch as independent task to free the message pump.
                # Commands like /resume mount interactive widgets that need
                # the pump to process key events and message bubbling.
                asyncio.ensure_future(self._handle_command(text))
                return

            self._history_suggester.append_entry(text)
            self._run_task = asyncio.ensure_future(self._run_turn(text))

        def on_input_changed(self, event: Input.Changed) -> None:
            text = event.value
            comp_widget = self.query_one("#completions", Static)
            if text.startswith("/"):
                prefix = text.lower()
                matches = [
                    (cmd, desc)
                    for cmd, desc in _TUI_SLASH_COMMANDS
                    if cmd.startswith(prefix)
                ]
                if len(matches) == 1 and matches[0][0] == prefix:
                    self._hide_completions()
                    return
                if matches:
                    self._comp_items = matches
                    self._comp_index = -1
                    self._render_completions()
                    comp_widget.display = True
                    return
            self._hide_completions()

        def _render_queue_indicator(self) -> None:
            """Render the queued messages indicator above the input."""
            queued_w = self.query_one("#queued-message", Static)
            if not self._queued_messages:
                queued_w.display = False
                return
            parts: list[tuple[str, str]] = []
            for msg in self._queued_messages:
                preview = msg if len(msg) <= 60 else msg[:57] + "\u2026"
                parts.append(("\u276f ", "bold"))
                parts.append((preview, ""))
                parts.append(("\n", ""))
            parts.append(("  [press up to edit last \u00b7 esc to cancel last]", "dim italic"))
            queued_w.update(Text.assemble(*parts))
            queued_w.display = True

        def action_cancel_queued(self) -> None:
            """Cancel the last queued message on Esc."""
            # Cancel ask_user if active (widget handles Escape internally,
            # but this is a safety fallback)
            if self._ask_user_future and not self._ask_user_future.done():
                try:
                    from .widgets.ask_user_widget import AskUserWidget
                    ask_w = self.query_one(AskUserWidget)
                    ask_w.action_cancel()
                except Exception:
                    # Force-resolve the future
                    self._ask_user_future.set_result({"type": "cancelled"})
                return
            # Delegate to ApprovalWidget or ThreadPickerWidget if focused
            focused = self.focused
            if focused is not None:
                from .widgets.approval_widget import ApprovalWidget
                from .widgets.thread_selector import ThreadPickerWidget
                if isinstance(focused, ApprovalWidget):
                    focused.action_select_reject()
                    return
                if isinstance(focused, ThreadPickerWidget):
                    focused.action_cancel()
                    return
            if self._queued_messages:
                self._queued_messages.pop()
                self._render_queue_indicator()

        def action_edit_queued(self) -> None:
            """Pop the last queued message back into input for editing."""
            # Skip if an ApprovalWidget, AskUserWidget, or ThreadPickerWidget has focus
            focused = self.focused
            if focused is not None:
                from .widgets.approval_widget import ApprovalWidget
                from .widgets.ask_user_widget import AskUserWidget
                from .widgets.thread_selector import ThreadPickerWidget
                if isinstance(focused, ApprovalWidget):
                    focused.action_move_up()
                    return
                if isinstance(focused, AskUserWidget):
                    focused.action_move_up()
                    return
                if isinstance(focused, ThreadPickerWidget):
                    focused.action_move_up()
                    return
            if self._queued_messages:
                last = self._queued_messages.pop()
                prompt = self.query_one("#prompt", Input)
                prompt.value = last
                prompt.cursor_position = len(prompt.value)
                prompt.focus()
                self._render_queue_indicator()

        def action_down_delegate(self) -> None:
            """Delegate down key to focused ApprovalWidget, AskUserWidget, or ThreadPickerWidget."""
            focused = self.focused
            if focused is not None:
                from .widgets.approval_widget import ApprovalWidget
                from .widgets.ask_user_widget import AskUserWidget
                from .widgets.thread_selector import ThreadPickerWidget
                if isinstance(focused, ApprovalWidget):
                    focused.action_move_down()
                    return
                if isinstance(focused, AskUserWidget):
                    focused.action_move_down()
                    return
                if isinstance(focused, ThreadPickerWidget):
                    focused.action_move_down()
                    return

        def on_key(self, event: Any) -> None:
            comp_widget = self.query_one("#completions", Static)
            if not (comp_widget.display and self._comp_items):
                return

            if event.key in ("tab", "down"):
                event.prevent_default()
                event.stop()
                self._comp_index = (self._comp_index + 1) % len(self._comp_items)
                self._apply_selected_completion()
            elif event.key == "up":
                event.prevent_default()
                event.stop()
                self._comp_index = (self._comp_index - 1) % len(self._comp_items)
                self._apply_selected_completion()
            elif event.key == "enter" and self._comp_index >= 0:
                event.prevent_default()
                event.stop()
                self._hide_completions()

        def _apply_selected_completion(self) -> None:
            selected_cmd = self._comp_items[self._comp_index][0]
            prompt = self.query_one("#prompt", Input)
            prompt.value = selected_cmd + " "
            prompt.cursor_position = len(prompt.value)
            self._render_completions()

        def _hide_completions(self) -> None:
            self._comp_items = []
            self._comp_index = -1
            comp_widget = self.query_one("#completions", Static)
            comp_widget.display = False

        def _render_completions(self) -> None:
            comp_text = Text()
            for i, (cmd, desc) in enumerate(self._comp_items):
                if i == self._comp_index:
                    comp_text.append("\u25b8 ", style="bold")
                    comp_text.append(f"{cmd:<22}", style="bold")
                    comp_text.append(desc, style="bold")
                else:
                    comp_text.append("  ", style="#888888")
                    comp_text.append(f"{cmd:<22}", style="#888888")
                    comp_text.append(desc, style="#888888")
                if i < len(self._comp_items) - 1:
                    comp_text.append("\n")
            self.query_one("#completions", Static).update(comp_text)

        # ── Slash commands ─────────────────────────────────────

        async def _handle_command(self, command: str) -> None:
            cmd, _, arg = command.strip().partition(" ")
            cmd = cmd.lower()
            arg = arg.strip()

            if cmd in ("/exit", "/quit", "/q"):
                self.action_request_quit()
                return

            if cmd == "/help":
                help_text = Text("Available commands:\n", style="bold")
                for hcmd, hdesc in _TUI_SLASH_COMMANDS:
                    help_text.append(f"  {hcmd:<22}", style="cyan")
                    help_text.append(f"{hdesc}\n", style="dim")
                self._mount_renderable(help_text)
                return

            if cmd == "/current":
                from .. import paths

                self._append_system(f"Thread: {self._conversation_tid}", style="dim")
                if self._workspace_dir:
                    self._append_system(
                        f"Workspace: {_shorten_path(self._workspace_dir)}",
                        style="dim",
                    )
                memory_path = _shorten_path(str(paths.MEMORY_DIR))
                if memory_path:
                    self._append_system(f"Memory dir: {memory_path}", style="dim")
                self._append_system("UI: tui", style="dim")
                return

            if cmd == "/new":
                # Clear all widgets except #welcome
                container = self.query_one("#chat", VerticalScroll)
                welcome = self.query_one("#welcome", Static)
                for child in list(container.children):
                    if child is not welcome:
                        await child.remove()

                if not workspace_fixed:
                    self._workspace_dir = create_session_workspace(run_name)
                self._conversation_tid = generate_thread_id()
                self._agent = load_agent(
                    workspace_dir=self._workspace_dir,
                    checkpointer=self._checkpointer,
                )
                if _channels_is_running():
                    _ch_mod._cli_agent = self._agent
                    _ch_mod._cli_thread_id = self._conversation_tid
                self._render_welcome()
                self._render_status()
                self._append_system(f"New session: {self._conversation_tid}", style="green")
                return

            if cmd == "/clear":
                container = self.query_one("#chat", VerticalScroll)
                welcome = self.query_one("#welcome", Static)
                for child in list(container.children):
                    if child is not welcome:
                        await child.remove()
                return

            if cmd == "/threads":
                await self._cmd_threads()
                return

            if cmd == "/resume":
                await self._cmd_resume(arg)
                return

            if cmd == "/delete":
                await self._cmd_delete(arg)
                return

            if cmd == "/skills":
                self._cmd_skills()
                return

            if cmd == "/install-skill":
                self._cmd_install_skill(arg)
                return

            if cmd == "/uninstall-skill":
                self._cmd_uninstall_skill(arg)
                return

            if cmd == "/mcp":
                self._cmd_mcp(arg)
                return

            if cmd == "/channel":
                self._cmd_channel(arg)
                return

            self._append_system(f"Unknown command: {command}", style="yellow")

        async def _resolve_thread_id(self, prefix: str) -> str | None:
            if await thread_exists(prefix):
                return prefix

            similar = await find_similar_threads(prefix)
            if len(similar) == 1:
                return similar[0]

            if len(similar) > 1:
                self._append_system(
                    f"Ambiguous thread ID '{prefix}'. Use a longer prefix.",
                    style="yellow",
                )
                for thread in similar:
                    self._append_system(f"  - {thread}", style="dim")
                return None

            self._append_system(f"Thread '{prefix}' not found.", style="red")
            return None

        async def _cmd_threads(self) -> None:
            threads = await list_threads(
                limit=0,
                include_message_count=True,
                include_preview=True,
            )
            if not threads:
                self._append_system("No saved sessions.", style="yellow")
                return

            table = Table(title="Sessions", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="bold")
            table.add_column("Preview", style="dim", max_width=50, no_wrap=True)
            table.add_column("Messages", justify="right")
            table.add_column("Model", style="dim")
            table.add_column("Last Used", style="dim")
            for thread in threads:
                thread_id_value = thread["thread_id"]
                marker = " *" if thread_id_value == self._conversation_tid else ""
                table.add_row(
                    f"{thread_id_value}{marker}",
                    thread.get("preview", "") or "",
                    str(thread.get("message_count", 0)),
                    thread.get("model", "") or "",
                    _format_relative_time(thread.get("updated_at")),
                )
            self._mount_renderable(table)

        async def _render_history(self, thread_id_value: str) -> None:
            """Render conversation history from a saved thread."""
            messages = await get_thread_messages(thread_id_value)
            if not messages:
                return

            container = self.query_one("#chat", VerticalScroll)
            await container.mount(SystemMessage("── Conversation history ──", msg_style="dim"))
            for message in messages:
                msg_type = getattr(message, "type", None)
                content = getattr(message, "content", "") or ""
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content = " ".join(parts) if parts else ""
                content = content.strip()
                if len(content) > 220:
                    content = content[:220] + "..."

                if msg_type == "human":
                    await container.mount(UserMessage(content))
                elif msg_type == "ai":
                    tool_calls = getattr(message, "tool_calls", None) or []
                    if content:
                        await container.mount(
                            Static(Text(content, style="dim"))
                        )
                    if tool_calls:
                        names = [tc.get("name", "?") for tc in tool_calls]
                        await container.mount(
                            Static(Text(f"  \u25b6 {', '.join(names)}", style="dim italic"))
                        )
            await container.mount(SystemMessage("── End of history ──", msg_style="dim"))
            container.scroll_end(animate=False)

        async def _cmd_resume(self, arg: str) -> None:
            if not arg:
                # Show inline thread picker
                threads = await list_threads(
                    limit=0,
                    include_message_count=True,
                    include_preview=True,
                )
                if not threads:
                    self._append_system("No sessions to resume.", style="yellow")
                    return

                from .widgets.thread_selector import ThreadPickerWidget

                container = self.query_one("#chat", VerticalScroll)
                picker = ThreadPickerWidget(
                    threads,
                    current_thread=self._conversation_tid,
                    title=">>> Select session to resume <<<",
                )
                await container.mount(picker)
                container.scroll_end(animate=False)
                picker.focus()

                selected = await self._wait_for_thread_pick(picker)
                if selected is None:
                    return
                arg = selected

            resolved = await self._resolve_thread_id(arg)
            if not resolved:
                return

            metadata = await get_thread_metadata(resolved)
            restored_workspace = (metadata or {}).get("workspace_dir", "")
            if restored_workspace:
                self._workspace_dir = restored_workspace

            self._conversation_tid = resolved
            self._agent = load_agent(
                workspace_dir=self._workspace_dir,
                checkpointer=self._checkpointer,
            )
            if _channels_is_running():
                _ch_mod._cli_agent = self._agent
                _ch_mod._cli_thread_id = self._conversation_tid
            self._render_welcome()
            self._render_status()
            self._append_system(f"Resumed session: {resolved}", style="green")
            await self._render_history(resolved)

        async def _cmd_delete(self, arg: str) -> None:
            if not arg:
                # Show inline thread picker for deletion
                threads = await list_threads(
                    limit=0,
                    include_message_count=True,
                    include_preview=True,
                )
                if not threads:
                    self._append_system("No sessions to delete.", style="yellow")
                    return

                from .widgets.thread_selector import ThreadPickerWidget

                container = self.query_one("#chat", VerticalScroll)
                picker = ThreadPickerWidget(
                    threads,
                    current_thread=self._conversation_tid,
                    title=">>> Select session to delete <<<",
                )
                await container.mount(picker)
                container.scroll_end(animate=False)
                picker.focus()

                selected = await self._wait_for_thread_pick(picker)
                if selected is None:
                    return
                arg = selected

            resolved = await self._resolve_thread_id(arg)
            if not resolved:
                return

            if resolved == self._conversation_tid:
                self._append_system(
                    "Cannot delete the current session.",
                    style="yellow",
                )
                return

            deleted = await delete_thread(resolved)
            if deleted:
                self._append_system(f"Deleted session {resolved}.", style="green")
            else:
                self._append_system(f"Session {resolved} not found.", style="red")

        def _cmd_skills(self) -> None:
            from ..tools.skills_manager import list_skills
            from ..paths import USER_SKILLS_DIR

            skills = list_skills(include_system=True)
            if not skills:
                self._append_system("No skills available.", style="dim")
                self._append_system("Install with: /install-skill <path-or-url>", style="dim")
                self._append_system(f"Skills directory: {_shorten_path(str(USER_SKILLS_DIR))}", style="dim")
                return

            user_skills = [s for s in skills if s.source == "user"]
            system_skills = [s for s in skills if s.source == "system"]

            if user_skills:
                table = Table(title=f"User Skills ({len(user_skills)})", show_header=True)
                table.add_column("Name", style="green")
                table.add_column("Description", style="dim")
                for s in user_skills:
                    table.add_row(s.name, s.description)
                self._mount_renderable(table)

            if system_skills:
                table = Table(title=f"Built-in Skills ({len(system_skills)})", show_header=True)
                table.add_column("Name", style="cyan")
                table.add_column("Description", style="dim")
                for s in system_skills:
                    table.add_row(s.name, s.description)
                self._mount_renderable(table)

            self._append_system(
                f"User skills folder: {_shorten_path(str(USER_SKILLS_DIR))}",
                style="dim",
            )

        def _cmd_install_skill(self, source: str) -> None:
            from ..tools.skills_manager import install_skill

            if not source:
                self._append_system("Usage: /install-skill <path-or-url>", style="yellow")
                self._append_system("Examples:", style="dim")
                self._append_system("  /install-skill ./my-skill", style="dim")
                self._append_system(
                    "  /install-skill https://github.com/user/repo/tree/main/skill-name",
                    style="dim",
                )
                self._append_system("  /install-skill user/repo@skill-name", style="dim")
                return

            self._append_system(f"Installing skill from: {source}", style="dim")
            result = install_skill(source)
            if result["success"]:
                self._append_system(f"Installed: {result['name']}", style="green")
                self._append_system(
                    f"Description: {result.get('description', '(none)')}",
                    style="dim",
                )
                self._append_system(f"Path: {_shorten_path(result['path'])}", style="dim")
                self._append_system("Reload with /new to apply.", style="dim")
            else:
                self._append_system(f"Failed: {result['error']}", style="red")

        def _cmd_uninstall_skill(self, name: str) -> None:
            from ..tools.skills_manager import uninstall_skill

            if not name:
                self._append_system("Usage: /uninstall-skill <skill-name>", style="yellow")
                self._append_system("Use /skills to see installed skills.", style="dim")
                return

            result = uninstall_skill(name)
            if result["success"]:
                self._append_system(f"Uninstalled: {name}", style="green")
                self._append_system("Reload with /new to apply.", style="dim")
            else:
                self._append_system(f"Failed: {result['error']}", style="red")

        def _cmd_mcp(self, args: str) -> None:
            args = args.strip()
            if not args or args == "list":
                self._mcp_list()
                return

            parts = args.split(maxsplit=1)
            subcmd = parts[0].lower()
            subargs = parts[1] if len(parts) > 1 else ""

            if subcmd == "config":
                self._mcp_config(subargs.strip())
            elif subcmd == "add":
                self._mcp_add(subargs)
            elif subcmd == "edit":
                self._mcp_edit(subargs)
            elif subcmd == "remove":
                self._mcp_remove(subargs.strip())
            else:
                self._append_system("MCP commands:", style="bold")
                self._append_system("  /mcp              List configured servers", style="dim")
                self._append_system("  /mcp list         List configured servers", style="dim")
                self._append_system("  /mcp config       Show detailed server config", style="dim")
                self._append_system("  /mcp add ...      Add a server", style="dim")
                self._append_system("  /mcp edit ...     Edit an existing server", style="dim")
                self._append_system("  /mcp remove ...   Remove a server", style="dim")

        def _mcp_list(self) -> None:
            from ..mcp import load_mcp_config
            from ..mcp.client import USER_MCP_CONFIG

            config = load_mcp_config()
            if not config:
                self._append_system("No MCP servers configured.", style="dim")
                self._append_system("Add one with: /mcp add <name> <command-or-url> [args...]", style="dim")
                return

            table = Table(title="MCP Servers", show_header=True)
            table.add_column("Server", style="cyan")
            table.add_column("Transport", style="green")
            table.add_column("Tools", style="yellow")
            table.add_column("Expose To", style="magenta")

            for name, server in config.items():
                transport = server.get("transport", "?")
                tools = server.get("tools")
                tools_str = ", ".join(tools) if tools else "(all)"
                expose_to = server.get("expose_to", ["main"])
                if isinstance(expose_to, str):
                    expose_to = [expose_to]
                expose_str = ", ".join(expose_to)
                table.add_row(name, transport, tools_str, expose_str)

            self._mount_renderable(table)
            self._append_system(f"Config file: {USER_MCP_CONFIG}", style="dim")

        def _mcp_config(self, name: str) -> None:
            from ..mcp import load_mcp_config
            from ..mcp.client import USER_MCP_CONFIG

            config = load_mcp_config()
            if not config:
                self._append_system("No MCP servers configured.", style="dim")
                return

            if name and name not in config:
                self._append_system(f"Server not found: {name}", style="red")
                return

            servers = {name: config[name]} if name else config
            for srv_name, srv in servers.items():
                table = Table(title=f"MCP Server: {srv_name}", show_header=True, title_style="bold cyan")
                table.add_column("Setting", style="cyan")
                table.add_column("Value")
                table.add_row("transport", str(srv.get("transport", "(not set)")))
                if srv.get("command"):
                    table.add_row("command", str(srv["command"]))
                if srv.get("args"):
                    table.add_row("args", " ".join(str(a) for a in srv["args"]))
                if srv.get("url"):
                    table.add_row("url", str(srv["url"]))
                if srv.get("headers"):
                    for k, v in srv["headers"].items():
                        table.add_row(f"header: {k}", str(v))
                if srv.get("env"):
                    for k, v in srv["env"].items():
                        table.add_row(f"env: {k}", str(v))
                tools = srv.get("tools")
                table.add_row("tools", ", ".join(tools) if tools else "(all)")
                expose_to = srv.get("expose_to", ["main"])
                if isinstance(expose_to, str):
                    expose_to = [expose_to]
                table.add_row("expose_to", ", ".join(expose_to))
                self._mount_renderable(table)

            self._append_system(f"Config file: {USER_MCP_CONFIG}", style="dim")

        def _mcp_add(self, args_str: str) -> None:
            from ..mcp import parse_mcp_add_args, add_mcp_server

            if not args_str.strip():
                self._append_system("Usage: /mcp add <name> <command-or-url> [args...]", style="yellow")
                return

            try:
                tokens = shlex.split(args_str)
                kwargs = parse_mcp_add_args(tokens)
                entry = add_mcp_server(**kwargs)
                self._append_system(
                    f"Added MCP server: {kwargs['name']} ({entry['transport']})",
                    style="green",
                )
                self._append_system("Reload with /new to apply.", style="dim")
            except ValueError as exc:
                self._append_system(f"{exc}", style="red")

        def _mcp_edit(self, args_str: str) -> None:
            from ..mcp import parse_mcp_edit_args, edit_mcp_server

            if not args_str.strip():
                self._append_system(
                    "Usage: /mcp edit <name> --<field> <value> ...",
                    style="yellow",
                )
                return

            try:
                tokens = shlex.split(args_str)
                name, fields = parse_mcp_edit_args(tokens)
                if not fields:
                    self._append_system(
                        "No fields to edit. Use --transport, --command, --url, --tools, --expose-to, etc.",
                        style="red",
                    )
                    return
                edit_mcp_server(name, **fields)
                self._append_system(f"Updated MCP server: {name}", style="green")
                for k, v in fields.items():
                    self._append_system(f"  {k}: {v}", style="dim")
                self._append_system("Reload with /new to apply.", style="dim")
            except (KeyError, ValueError) as exc:
                self._append_system(f"{exc}", style="red")

        def _mcp_remove(self, name: str) -> None:
            from ..mcp import remove_mcp_server

            if not name:
                self._append_system("Usage: /mcp remove <name>", style="yellow")
                return

            if remove_mcp_server(name):
                self._append_system(f"Removed MCP server: {name}", style="green")
                self._append_system("Reload with /new to apply.", style="dim")
            else:
                self._append_system(f"Server not found: {name}", style="red")

        def _cmd_channel(self, args: str) -> None:
            """Handle /channel command — start, stop, or show status."""
            from ..config import load_config
            from .channel import (
                _add_channel_to_running_bus,
                _start_channels_bus_mode,
                _channels_running_list,
            )

            args = args.strip().lower() if args else ""

            if args == "status" or (not args and _channels_is_running()):
                running = _channels_running_list()
                if running and _ch_mod._manager:
                    detailed = _ch_mod._manager.get_detailed_status()
                    table = Table(
                        title="Channel Status", show_header=True, expand=False,
                    )
                    table.add_column("Channel", style="cyan")
                    table.add_column("Status")
                    table.add_column("Uptime", style="dim")
                    table.add_column("Rx", justify="right")
                    table.add_column("Tx", justify="right")
                    for ch_name in running:
                        info = detailed.get(ch_name, {})
                        secs = info.get("uptime_seconds", 0)
                        mins, s = divmod(int(secs), 60)
                        hours, mins = divmod(mins, 60)
                        uptime = (
                            f"{hours}h{mins:02d}m" if hours else f"{mins}m{s:02d}s"
                        )
                        rx = str(info.get("received", 0))
                        tx = str(info.get("sent", 0))
                        table.add_row(
                            ch_name, "[green]running[/green]", uptime, rx, tx,
                        )
                    self._mount_renderable(table)
                else:
                    self._append_system("No channel running", style="dim")
                return

            if args.startswith("stop"):
                stop_type = args[len("stop"):].strip() or None
                if not _channels_is_running():
                    self._append_system("No channel running", style="dim")
                    return
                if stop_type:
                    if not _channels_is_running(stop_type):
                        self._append_system(
                            f"{stop_type} is not running", style="dim",
                        )
                        return
                    _channels_stop(stop_type)
                    if stop_type in self._started_channel_types:
                        self._started_channel_types.remove(stop_type)
                    self._append_system(f"{stop_type} stopped", style="dim")
                else:
                    running = _channels_running_list()
                    _channels_stop()
                    self._started_channel_types.clear()
                    self._append_system(
                        f"{', '.join(running)} stopped", style="dim",
                    )
                self._render_welcome()
                return

            # Start channel(s)
            app_config = load_config()
            channel_type = args if args else (
                app_config.channel_enabled if app_config else ""
            )
            if not channel_type:
                self._append_system("No channel configured.", style="yellow")
                self._append_system(
                    "Run EvoSci onboard or specify: /channel telegram",
                    style="dim",
                )
                return

            requested = [
                t.strip() for t in channel_type.split(",") if t.strip()
            ]

            if _channels_is_running():
                running = _channels_running_list()
                results: list[tuple[str, bool, str]] = []
                for ct in requested:
                    if ct in running:
                        results.append((ct, True, "already running"))
                    else:
                        try:
                            _add_channel_to_running_bus(
                                ct,
                                app_config,
                                send_thinking=self._channel_send_thinking,
                            )
                            results.append((ct, True, "connected (bus)"))
                        except Exception as e:
                            results.append((ct, False, str(e)))
            else:
                _ch_mod._cli_agent = self._agent
                _ch_mod._cli_thread_id = self._conversation_tid
                original = app_config.channel_enabled
                app_config.channel_enabled = channel_type
                try:
                    _start_channels_bus_mode(
                        app_config,
                        self._agent,
                        self._conversation_tid,
                        send_thinking=self._channel_send_thinking,
                    )
                    results = [
                        (ct, True, "connected (bus)") for ct in requested
                    ]
                except Exception as e:
                    results = [(ct, False, str(e)) for ct in requested]
                finally:
                    app_config.channel_enabled = original

            for ct, ok, _ in results:
                if ok and ct not in self._started_channel_types:
                    self._started_channel_types.append(ct)

            self._render_channel_results(results)
            self._render_welcome()

        def _render_channel_results(
            self, results: list[tuple[str, bool, str]],
        ) -> None:
            for name, ok, detail in results:
                if ok:
                    self._append_system(
                        f"\u25cf {name}  {detail}", style="green",
                    )
                else:
                    self._append_system(
                        f"\u2717 {name}  {detail}", style="yellow",
                    )

        # ── Quit handling ──────────────────────────────────────

        def action_request_quit(self) -> None:
            if self._busy:
                # Clear all queued messages on interrupt
                if self._queued_messages:
                    self._queued_messages.clear()
                    self._render_queue_indicator()
                if self._run_task is not None and not self._run_task.done():
                    self._run_task.cancel()
                else:
                    # Edge case: busy but no task — force reset
                    self._busy = False
                    self.query_one("#prompt", Input).focus()
                    self._render_status()
                    self._append_system("Interrupted.", style="yellow")
                return
            # Clean up channels
            if self._channel_timer is not None:
                self._channel_timer.stop()
                self._channel_timer = None
            self._started_channel_types.clear()
            if _channels_is_running():
                try:
                    _channels_stop()
                except Exception:
                    pass
            self.exit()

        # ── Banner & status ────────────────────────────────────

        def _render_welcome(self) -> None:
            channels_info: list[tuple[str, bool, str]] | None = None
            try:
                running = _channels_running_list()
                started = self._started_channel_types
                if running or started:
                    all_types = list(dict.fromkeys(running + started))
                    channels_info = [
                        (ct, True, "connected (bus)") for ct in all_types
                    ]
                else:
                    from ..config import load_config

                    cfg = load_config()
                    if cfg and cfg.channel_enabled:
                        types = [
                            t.strip()
                            for t in cfg.channel_enabled.split(",")
                            if t.strip()
                        ]
                        if types:
                            channels_info = [
                                (ct, False, "configured") for ct in types
                            ]
            except Exception:
                pass

            welcome = self.query_one("#welcome", Static)
            welcome.update(
                _build_welcome_banner(
                    thread_id=self._conversation_tid,
                    workspace_dir=self._workspace_dir,
                    mode=mode,
                    model=model,
                    provider=provider,
                    ui_backend="tui",
                    channels=channels_info,
                )
            )

        def _render_status(self) -> None:
            status = self.query_one("#status", Static)
            if self._busy:
                left = "vibe researching..."
                left_style = "bold #f59e0b"
            else:
                left = "/help for commands"
                left_style = "#f59e0b"

            status.update(
                Text.assemble(
                    (left, left_style),
                    ("  ", ""),
                    ("EvoScientist", "dim"),
                )
            )

    # ── Media forwarding helper (module-level) ──────────────

    _MEDIA_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg",
        ".pdf", ".mp3", ".wav", ".mp4",
    }

    def _forward_media_to_channel(
        state: StreamState,
        tool_name: str,
        media_sent: set[str],
        send_fn: Any,
    ) -> None:
        """Check tool calls for media files and forward to channel."""
        import os
        from ..paths import resolve_virtual_path

        arg_key = "path" if tool_name == "write_file" else "file_path"
        for tc in reversed(state.tool_calls):
            if tc.get("name") == tool_name:
                p = tc.get("args", {}).get(arg_key, "")
                if not p:
                    p = tc.get("args", {}).get("path", "")
                if p and p not in media_sent:
                    ext = os.path.splitext(p)[1].lower()
                    if ext in _MEDIA_EXTENSIONS:
                        real_path = str(resolve_virtual_path(p))
                        if not os.path.isfile(real_path) and os.path.isfile(p):
                            real_path = p
                        if os.path.isfile(real_path):
                            media_sent.add(p)
                            send_fn(real_path)
                break

    # ── Entry point ─────────────────────────────────────────

    async def _amain() -> None:
        async with get_checkpointer() as checkpointer:
            effective_workspace = workspace_dir
            effective_thread_id = thread_id
            resumed = False
            resume_warning = ""
            if thread_id:
                if await thread_exists(thread_id):
                    resolved = thread_id
                else:
                    similar = await find_similar_threads(thread_id)
                    resolved = similar[0] if len(similar) == 1 else None
                if resolved:
                    meta = await get_thread_metadata(resolved)
                    ws = (meta or {}).get("workspace_dir", "")
                    if ws:
                        effective_workspace = ws
                    effective_thread_id = resolved
                    resumed = True
                else:
                    resume_warning = (
                        f"Thread '{thread_id}' not found. Starting new session."
                    )
            if not effective_thread_id:
                effective_thread_id = generate_thread_id()

            initial_agent = load_agent(
                workspace_dir=effective_workspace,
                checkpointer=checkpointer,
            )
            app = EvoTextualInteractiveApp(
                agent=initial_agent,
                thread_id_value=effective_thread_id,
                workspace=effective_workspace,
                checkpointer=checkpointer,
                channel_send_thinking_value=channel_send_thinking,
                resumed=resumed,
                resume_warning=resume_warning,
            )
            await app.run_async()

    import nest_asyncio  # type: ignore[import-untyped]

    nest_asyncio.apply()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(_amain())
