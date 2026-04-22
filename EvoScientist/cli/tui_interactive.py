"""Full-screen Textual interactive TUI for EvoScientist.

Widget-based rendering: each message/tool/sub-agent is an independent widget
mounted into a VerticalScroll container.  No timer-based Group rebuilds.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import random
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar

from rich.console import Group
from rich.text import Text

import EvoScientist.cli.channel as _ch_mod
from EvoScientist.cli.widgets.thread_selector import ThreadPickerWidget

from ..commands import CommandContext
from ..commands import manager as cmd_manager
from ..paths import DATA_DIR
from ..sessions import (
    generate_thread_id,
    get_checkpointer,
    get_thread_messages,
    get_thread_metadata,
    resolve_thread_id_prefix,
    thread_exists,
)
from ..stream.events import stream_agent_events
from ..stream.state import _INTERNAL_TOOLS, StreamState
from ._agent_loader import BackgroundAgentLoader, MCPProgressTracker
from ._constants import LOGO_GRADIENT, LOGO_LINES, WELCOME_SLOGANS, build_metadata
from .channel import (
    ChannelMessage,
    _auto_start_channel,
    _channels_is_running,
    _channels_running_list,
    _channels_stop,
    _message_queue,
    _set_channel_response,
)
from .file_mentions import complete_file_mention, resolve_file_mentions
from .history_suggester import HistorySuggester
from .status_bar import (
    STATUS_BAR_BG,
    STATUS_DIM,
    STATUS_HINT_BUSY,
    STATUS_HINT_IDLE,
    apply_assistant_text_to_snapshot,
    apply_user_text_to_snapshot,
    build_session_status_snapshot,
    build_status_text,
    make_empty_status_snapshot,
    make_usage_status_snapshot,
)

_channel_logger = logging.getLogger(__name__)


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
    for line, color in zip(LOGO_LINES, LOGO_GRADIENT, strict=False):
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
    dir_display = (
        effective_dir.replace(home, "~", 1)
        if effective_dir.startswith(home)
        else effective_dir
    )
    info.append("\n  ", style="dim")
    info.append("Directory: ", style="dim")
    info.append(dir_display, style="magenta")
    _nl_key = "Option+Enter" if sys.platform == "darwin" else "Ctrl+J"
    info.append("\n  Enter ", style="#ffe082")
    info.append("send", style="#ffe082 bold")
    info.append(f" \u2022 {_nl_key} ", style="#ffe082")
    info.append("newline", style="#ffe082 bold")
    info.append(" \u2022 Type ", style="#ffe082")
    info.append("/", style="#ffe082 bold")
    info.append(" for commands", style="#ffe082")
    info.append(" \u2022 ", style="#ffe082")
    info.append("@ files", style="#ffe082 bold")
    info.append(" \u2022 Ctrl+C ", style="#ffe082")
    info.append("interrupt", style="#ffe082 bold")
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
        panel = Panel(
            body, title="[bold]Channels[/bold]", border_style=border, expand=False
        )
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


_SUMMARY_CONTINUATION_EVENTS = {
    "summarization_start",
    "summarization",
    "usage_stats",
}


def _should_finalize_active_summarization(event_type: str) -> bool:
    """Return whether an active summary panel should stop for this event."""
    return bool(event_type) and event_type not in _SUMMARY_CONTINUATION_EVENTS


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
        from textual.widgets import Static

        from .clipboard import copy_selection_to_clipboard, get_clipboard_text
        from .widgets import (
            AssistantMessage,
            CompactingWidget,
            LoadingWidget,
            MCPLoaderWidget,
            SubAgentWidget,
            SummarizationWidget,
            SystemMessage,
            ThinkingWidget,
            TodoWidget,
            ToolCallWidget,
            UsageWidget,
            UserMessage,
        )
        from .widgets.chat_input import ChatTextArea
    except Exception as e:  # pragma: no cover - runtime fallback path
        raise RuntimeError(
            "Textual TUI backend requires 'textual'. Run: pip install textual"
        ) from e

    class EvoTextualInteractiveApp(App[None]):  # type: ignore[type-arg]
        """Deep-Agents-style full-screen TUI with independent widget rendering."""

        @property
        def supports_interactive(self) -> bool:
            return True

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
            padding: 0 2 0 2;
            background: #16161a;
        }
        #input-row {
            height: auto;
            min-height: 3;
            max-height: 10;
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
            min-height: 1;
            max-height: 8;
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
            min-height: 1;
            background: #171a20;
            color: #cbd5e1;
            padding: 0 1;
        }
        """
        BINDINGS: ClassVar[list[Binding]] = [
            Binding("ctrl+c", "request_quit", "Quit", show=False, priority=True),
            Binding("ctrl+v", "paste_clipboard", "Paste", show=False),
            Binding("tab", "tab_complete", show=False, priority=True),
            Binding("up", "edit_queued", show=False, priority=True),
            Binding("down", "down_delegate", show=False, priority=True),
            Binding("escape", "cancel_queued", show=False, priority=True),
        ]

        def __init__(
            self,
            *,
            thread_id_value: str,
            workspace: str | None,
            checkpointer: Any,
            channel_send_thinking_value: bool = True,
            resumed: bool = False,
            resume_warning: str = "",
        ) -> None:
            super().__init__()
            self._progress_tracker = MCPProgressTracker()
            self._agent_loader = BackgroundAgentLoader(
                load_agent,
                on_progress=self._on_mcp_progress,
                on_success=self._on_agent_load_success,
                on_failure=self._on_agent_load_failure,
            )
            self._mcp_loader_widget: Any = None
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
            self._queued_messages: list[
                str
            ] = []  # queued messages to send after current turn
            self._comp_items: list[tuple[str, str]] = []
            self._comp_index: int = -1
            self._hitl_auto_approve: bool = False
            self._approval_future: asyncio.Future | None = None
            self._ask_user_future: asyncio.Future | None = None
            self._picker_future: asyncio.Future | None = None
            self._browser_future: asyncio.Future | None = None
            self._mcp_browser_future: asyncio.Future | None = None
            self._model_picker_future: asyncio.Future | None = None
            self._history_suggester = HistorySuggester(DATA_DIR / "history")
            self._history_index: int = -1  # -1 = not browsing history
            self._history_saved_input: str = ""  # saved current input before browsing
            self._background_tasks: set[asyncio.Task] = set()
            self._quit_pending: bool = False
            self._current_model: str | None = model
            self._current_provider: str | None = provider
            self._status_started_at = datetime.now()
            self._status_base_snapshot = make_empty_status_snapshot(self._current_model)
            self._status_snapshot = self._status_base_snapshot
            self._status_streaming_text = ""
            self._status_last_input_tokens: int | None = None
            self._compacting_widget: CompactingWidget | None = None

        # ── Background agent / MCP loading ───────────────────

        def _on_mcp_progress(self, event: str, server: str, detail: str) -> None:
            """Bridge worker-thread progress events to the Textual loop."""
            if event not in {"start", "success", "error"}:
                return
            try:
                self.call_from_thread(self._apply_mcp_progress, event, server, detail)
            except Exception:
                pass

        def _apply_mcp_progress(self, event: str, server: str, detail: str) -> None:
            """Update tracker + widget on the Textual thread."""
            state = self._progress_tracker.record(event, server, detail)
            if state is None:
                return
            widget = self._mcp_loader_widget
            if widget is None or widget.dismissed:
                self._mcp_loader_widget = None
                return
            widget.update_server(server, state, detail)

        def _on_agent_load_success(self, agent: Any) -> None:
            if _channels_is_running():
                _ch_mod._cli_agent = agent
                _ch_mod._cli_thread_id = self._conversation_tid
            self._finish_loader_widget()
            self._render_status()

        def _on_agent_load_failure(self, exc: BaseException) -> None:
            self._append_system(f"Agent failed to load: {exc}", style="red")
            self._finish_loader_widget()

        def _start_background_agent_load(self, workspace: str | None) -> None:
            self._progress_tracker.prime()
            self._mount_mcp_loader_widget()
            self._agent_loader.start(
                workspace_dir=workspace,
                checkpointer=self._checkpointer,
            )

        def _mount_mcp_loader_widget(self) -> None:
            if not self._progress_tracker.progress:
                return
            if self._mcp_loader_widget is not None:
                try:
                    self._mcp_loader_widget.remove()
                except Exception:
                    pass
                self._mcp_loader_widget = None
            widget = MCPLoaderWidget(list(self._progress_tracker.progress.keys()))
            try:
                shell = self.query_one("#input-shell", Container)
                children = list(shell.children)
                if children:
                    shell.mount(widget, before=children[0])
                else:
                    shell.mount(widget)
            except Exception:
                # Compose hasn't happened yet — the mount will be retried
                # from ``on_mount`` once the DOM is ready.
                return
            self._mcp_loader_widget = widget

        def _finish_loader_widget(self) -> None:
            """Call ``mark_finished`` and clear the ref if self-dismissed."""
            widget = self._mcp_loader_widget
            if widget is None:
                return
            try:
                widget.mark_finished()
            except Exception:
                pass
            if widget.dismissed:
                self._mcp_loader_widget = None

        async def _await_agent_ready(self) -> Any:
            """Await the agent load, auto-retrying on cold-start or failure."""
            if self._agent_loader.needs_restart:
                self._start_background_agent_load(self._workspace_dir)
            return await self._agent_loader.await_ready()

        # ── CommandUI implementation ─────────────────────────

        def append_system(self, text: str, style: str = "dim") -> None:
            self._append_system(text, style)

        def mount_renderable(self, renderable: Any) -> None:
            self._mount_renderable(renderable)

        async def start_compacting_indicator(self) -> None:
            await self._start_compacting_indicator()

        async def stop_compacting_indicator(self) -> None:
            await self._stop_compacting_indicator()

        async def wait_for_thread_pick(
            self, threads: list[dict], current_thread: str, title: str
        ) -> str | None:
            from .widgets.thread_selector import ThreadPickerWidget

            container = self.query_one("#chat", VerticalScroll)
            picker = ThreadPickerWidget(
                threads,
                current_thread=current_thread,
                title=title,
            )
            await container.mount(picker)
            self._schedule_scroll_to_bottom(container, delays=())
            picker.focus()

            return await self._wait_for_thread_pick(picker)

        async def wait_for_skill_browse(
            self, index: list[dict], installed_names: set[str], pre_filter_tag: str
        ) -> list[str] | None:
            from .widgets.skill_browser import SkillBrowserWidget

            container = self.query_one("#chat", VerticalScroll)
            browser = SkillBrowserWidget(
                index,
                installed_names,
                pre_filter_tag=pre_filter_tag,
            )
            await container.mount(browser)
            self._schedule_scroll_to_bottom(container, delays=())
            browser.focus()

            return await self._wait_for_skill_browse(browser)

        async def wait_for_mcp_browse(
            self, servers: list, installed_names: set[str], pre_filter_tag: str
        ) -> list | None:
            from .widgets.mcp_browser import MCPBrowserWidget

            container = self.query_one("#chat", VerticalScroll)
            browser = MCPBrowserWidget(
                servers,
                installed_names,
                pre_filter_tag=pre_filter_tag,
            )
            await container.mount(browser)
            self._schedule_scroll_to_bottom(container, delays=())
            browser.focus()

            return await self._wait_for_mcp_browse(browser)

        async def wait_for_model_pick(
            self,
            entries: list[tuple[str, str, str]],
            current_model: str | None,
            current_provider: str | None,
        ) -> tuple[str, str] | None:
            from .widgets.model_picker import ModelPickerWidget

            container = self.query_one("#chat", VerticalScroll)
            picker = ModelPickerWidget(
                entries,
                current_model=current_model,
                current_provider=current_provider,
            )
            await container.mount(picker)
            self._schedule_scroll_to_bottom(container, delays=())
            picker.focus()

            return await self._wait_for_model_pick(picker)

        def clear_chat(self) -> None:
            container = self.query_one("#chat", VerticalScroll)
            welcome = self.query_one("#welcome", Static)
            for child in list(container.children):
                if child is not welcome:
                    child.remove()

        def request_quit(self) -> None:
            self.action_request_quit()

        def start_new_session(self) -> None:
            # Clear all widgets except #welcome
            self.clear_chat()

            if not workspace_fixed:
                self._workspace_dir = create_session_workspace(run_name)
            self._conversation_tid = generate_thread_id()
            # Background reload: next user message awaits it.
            self._start_background_agent_load(self._workspace_dir)
            self._status_started_at = datetime.now()
            self._status_base_snapshot = make_empty_status_snapshot(self._current_model)
            self._status_snapshot = self._status_base_snapshot
            self._status_streaming_text = ""
            self._status_last_input_tokens = None
            self._render_welcome()
            self._render_status()
            refresh_task = asyncio.create_task(self._refresh_status_snapshot())
            self._background_tasks.add(refresh_task)
            refresh_task.add_done_callback(self._background_tasks.discard)
            self.append_system(f"New session: {self._conversation_tid}", style="green")

        async def handle_session_resume(
            self, thread_id: str, workspace_dir: str | None = None
        ) -> None:
            if workspace_dir:
                self._workspace_dir = workspace_dir

            self._conversation_tid = thread_id
            # Background reload: history renders immediately; next turn awaits.
            self._start_background_agent_load(self._workspace_dir)
            self._status_started_at = datetime.now()
            self._status_base_snapshot = make_empty_status_snapshot(self._current_model)
            self._status_snapshot = self._status_base_snapshot
            self._status_streaming_text = ""
            self._status_last_input_tokens = None
            self._render_welcome()
            await self._refresh_status_snapshot()
            self._render_status()
            self.append_system(f"Resumed session: {thread_id}", style="green")
            await self._render_history(thread_id)

        async def flush(self) -> None:
            """No-op for TUI, messages are already delivered incrementally."""
            pass

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
                    yield ChatTextArea(
                        placeholder="Type message (/ for commands)",
                        id="prompt",
                    )

            yield Static("", id="status")

        def on_mount(self) -> None:
            self._render_welcome()
            self._render_status()
            self.set_interval(1.0, self._render_status)
            # Kick off agent construction in the background so the TUI
            # appears instantly; MCP progress shows up in the status bar.
            if self._agent_loader.agent is None and self._agent_loader.task is None:
                self._start_background_agent_load(self._workspace_dir)
            refresh_task = asyncio.create_task(self._refresh_status_snapshot())
            self._background_tasks.add(refresh_task)
            refresh_task.add_done_callback(self._background_tasks.discard)
            prompt = self.query_one("#prompt", ChatTextArea)
            prompt.before_submit = self._handle_completion_enter
            prompt.focus()
            # Show resume status
            if self._resume_warning:
                self._append_system(self._resume_warning, style="yellow")
            elif self._resumed:
                self._append_system(
                    f"Resumed session: {self._conversation_tid}",
                    style="green",
                )
                self.call_later(
                    lambda: asyncio.ensure_future(
                        self._render_history(self._conversation_tid)
                    )
                )
            # Startup notifications
            self.notify(
                "EvoScientist is your research buddy.\n"
                "Tell it about your taste before cooking some meal!",
                severity="warning",
                timeout=10,
            )
            self.run_worker(
                self._check_for_updates, exclusive=True, group="update-check"
            )

            # Auto-start channels — needs the agent, so defer to after load
            async def _deferred_start_channels():
                try:
                    await self._await_agent_ready()
                except Exception:
                    _channel_logger.debug(
                        "Skipping channel auto-start because agent load failed",
                        exc_info=True,
                    )
                    return
                self._start_channels()

            ch_task = asyncio.create_task(_deferred_start_channels())
            self._background_tasks.add(ch_task)
            ch_task.add_done_callback(self._background_tasks.discard)

        # ── Update check ──────────────────────────────────────

        async def _check_for_updates(self) -> None:
            """Check PyPI for a newer EvoScientist version and notify."""
            try:
                from ..update_check import _installed_version, is_update_available

                available, latest = await asyncio.to_thread(is_update_available)
                if available:
                    current = _installed_version()
                    self.notify(
                        f"Update available: v{latest} (current: v{current}).\n"
                        "Run: uv tool upgrade EvoScientist",
                        severity="information",
                        timeout=15,
                    )
            except Exception:
                _channel_logger.debug("Background update check failed", exc_info=True)

        # ── Channel integration ────────────────────────────────

        def _start_channels(self) -> None:
            """Auto-start channels if enabled in config."""
            try:
                from ..config import load_config

                cfg = load_config()
                if cfg and cfg.channel_enabled and not _channels_is_running():
                    _auto_start_channel(
                        self._agent_loader.agent,
                        self._conversation_tid,
                        cfg,
                        send_thinking=self._channel_send_thinking,
                    )
                    types = [
                        t.strip() for t in cfg.channel_enabled.split(",") if t.strip()
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
            self.call_later(
                lambda m=msg: asyncio.ensure_future(self._process_channel_message(m))
            )

        # ── Widget helpers ─────────────────────────────────────

        def _schedule_scroll_to_bottom(
            self,
            container: VerticalScroll,
            *,
            delays: tuple[float, ...] = (0.3, 0.8),
            immediate: bool = True,
        ) -> None:
            """Schedule deferred scrolls so the viewport lands at the bottom.

            Markdown- and list-heavy widgets lay out across multiple refresh
            cycles, so a single ``scroll_end()`` may fire against a stale
            ``virtual_size`` and leave the viewport mid-content. Re-schedule
            ``scroll_end`` at each delay to follow subsequent reflows.
            """
            if immediate:
                self.call_after_refresh(
                    lambda: container.scroll_end(animate=False),
                )
            for delay in delays:
                self.set_timer(
                    delay,
                    lambda: self.call_after_refresh(
                        lambda: container.scroll_end(animate=False),
                    ),
                )

        def _append_system(self, text: str, style: str = "dim") -> None:
            """Mount a SystemMessage widget into #chat."""
            container = self.query_one("#chat", VerticalScroll)
            container.mount(SystemMessage(text, msg_style=style))
            container.scroll_end(animate=False)

        def _mount_renderable(self, renderable: Any) -> None:
            """Mount a Rich renderable (e.g. Table) as a Static widget."""
            container = self.query_one("#chat", VerticalScroll)
            try:
                from .commands import CompactSummaryRenderable
                from .widgets.compact_summary_widget import CompactSummaryWidget
            except Exception:
                CompactSummaryRenderable = None  # type: ignore[assignment]

            if CompactSummaryRenderable is not None and isinstance(
                renderable, CompactSummaryRenderable
            ):
                container.mount(CompactSummaryWidget(renderable.summary_text))
            else:
                container.mount(Static(renderable))
            container.scroll_end(animate=False)

        async def _start_compacting_indicator(self) -> None:
            """Show a transient timer widget while /compact is running."""
            await self._stop_compacting_indicator()
            container = self.query_one("#chat", VerticalScroll)
            widget = CompactingWidget()
            self._compacting_widget = widget
            await container.mount(widget)
            container.scroll_end(animate=False)

        async def _stop_compacting_indicator(self) -> None:
            """Remove the transient /compact progress widget, if present."""
            widget = self._compacting_widget
            self._compacting_widget = None
            if widget is not None:
                try:
                    await widget.cleanup()
                except Exception:
                    try:
                        await widget.remove()
                    except Exception:
                        pass

        async def _wait_for_approval(self, approval_widget) -> Any:
            """Wait for user to interact with an ApprovalWidget.

            Returns the ``ApprovalWidget.Decided`` message, or ``None`` on
            timeout / cancellation.
            """
            self._approval_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._approval_future, timeout=300)
            except (TimeoutError, asyncio.CancelledError):
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
            except (TimeoutError, asyncio.CancelledError):
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

        async def _wait_for_thread_pick(
            self, picker_widget: ThreadPickerWidget
        ) -> str | None:
            """Wait for user to pick a thread from ThreadPickerWidget.

            Returns the selected thread_id, or ``None`` on cancel/timeout.
            """
            self._picker_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._picker_future, timeout=120)
            except (TimeoutError, asyncio.CancelledError):
                return None
            finally:
                self._picker_future = None
                try:
                    picker_widget.remove()
                except Exception:
                    pass
                self.query_one("#prompt", ChatTextArea).focus()

        def on_thread_picker_widget_picked(self, event) -> None:  # type: ignore[override]
            """Handle ThreadPickerWidget.Picked message."""
            if self._picker_future and not self._picker_future.done():
                self._picker_future.set_result(event.thread_id)

        def on_thread_picker_widget_cancelled(self, event) -> None:  # type: ignore[override]
            """Handle ThreadPickerWidget.Cancelled message."""
            if self._picker_future and not self._picker_future.done():
                self._picker_future.set_result(None)

        async def _wait_for_skill_browse(self, browser_widget) -> list[str] | None:
            """Wait for user to complete skill browsing.

            Returns list of install sources, or None on cancel/timeout.
            """
            self._browser_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._browser_future, timeout=300)
            except (TimeoutError, asyncio.CancelledError):
                return None
            finally:
                self._browser_future = None
                try:
                    browser_widget.remove()
                except Exception:
                    pass
                self.query_one("#prompt", ChatTextArea).focus()

        def on_skill_browser_widget_confirmed(self, event) -> None:  # type: ignore[override]
            """Handle SkillBrowserWidget.Confirmed message."""
            if self._browser_future and not self._browser_future.done():
                self._browser_future.set_result(event.install_sources)

        def on_skill_browser_widget_cancelled(self, event) -> None:  # type: ignore[override]
            """Handle SkillBrowserWidget.Cancelled message."""
            if self._browser_future and not self._browser_future.done():
                self._browser_future.set_result(None)

        # ── MCP browser ───────────────────────────────────────

        async def _wait_for_mcp_browse(self, browser_widget) -> list | None:
            """Wait for user to complete MCP server browsing."""
            self._mcp_browser_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._mcp_browser_future, timeout=300)
            except (TimeoutError, asyncio.CancelledError):
                return None
            finally:
                self._mcp_browser_future = None
                try:
                    browser_widget.remove()
                except Exception:
                    pass
                self.query_one("#prompt", ChatTextArea).focus()

        def on_mcpbrowser_widget_confirmed(self, event) -> None:  # type: ignore[override]
            """Handle MCPBrowserWidget.Confirmed message."""
            if self._mcp_browser_future and not self._mcp_browser_future.done():
                self._mcp_browser_future.set_result(event.entries)

        def on_mcpbrowser_widget_cancelled(self, event) -> None:  # type: ignore[override]
            """Handle MCPBrowserWidget.Cancelled message."""
            if self._mcp_browser_future and not self._mcp_browser_future.done():
                self._mcp_browser_future.set_result(None)

        async def _wait_for_model_pick(self, picker_widget) -> tuple[str, str] | None:
            """Wait for user to pick a model from ModelPickerWidget.

            Returns ``(name, provider)`` or ``None`` on cancel/timeout.
            """
            self._model_picker_future = asyncio.get_event_loop().create_future()
            try:
                return await asyncio.wait_for(self._model_picker_future, timeout=120)
            except (TimeoutError, asyncio.CancelledError):
                return None
            finally:
                self._model_picker_future = None
                try:
                    picker_widget.remove()
                except Exception:
                    _channel_logger.debug("model picker cleanup failed", exc_info=True)
                self.query_one("#prompt", ChatTextArea).focus()

        def on_model_picker_widget_picked(self, event) -> None:  # type: ignore[override]
            """Handle ModelPickerWidget.Picked message."""
            if self._model_picker_future and not self._model_picker_future.done():
                self._model_picker_future.set_result((event.name, event.provider))

        def on_model_picker_widget_cancelled(self, event) -> None:  # type: ignore[override]
            """Handle ModelPickerWidget.Cancelled message."""
            if self._model_picker_future and not self._model_picker_future.done():
                self._model_picker_future.set_result(None)

        # ── Streaming core ─────────────────────────────────────

        async def _stream_with_widgets(
            self,
            user_text: str,
            *,
            display_text: str | None = None,
            on_thinking_cb: Callable[[str], None] | None = None,
            on_todo_cb: Callable[[list[dict]], None] | None = None,
            on_media_cb: Callable[[str], None] | None = None,
            skip_user_message: bool = False,
            file_warnings: list[str] | None = None,
            channel_hitl_fn: Callable[[list], list[dict] | None] | None = None,
            channel_ask_user_fn: Callable[[dict], dict] | None = None,
        ) -> str:
            """Stream agent events and mount widgets.  Returns response text.

            Shared by ``_run_turn`` (interactive) and
            ``_process_channel_message`` (channel).

            Args:
                display_text: Text to show in UserMessage widget. When
                    ``None`` (default), falls back to *user_text*.  This
                    allows callers to show the original user input while
                    sending the resolved (e.g. @file-expanded) text to
                    the agent.
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
                await container.mount(UserMessage(display_text or user_text))
            # Mount file warnings after user message so they appear in the
            # correct position (between user input and model response).
            for w in file_warnings or []:
                self._append_system(f"⚠ {w}", style="yellow")
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
            narration_w: Static | None = None  # dim italic intermediate text
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

            metadata = build_metadata(self._workspace_dir, self._current_model)
            response = ""

            async def _remove_w(w: Static | None) -> None:
                """Safely remove a transient indicator widget."""
                if w is not None:
                    try:
                        await w.remove()
                    except Exception:
                        pass

            def _finalize_active_summarization() -> None:
                """Stop the active summary timer once the stream moves on."""
                if summarization_w is not None and summarization_w._is_active:
                    summarization_w.finalize()

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
                resolved_name: str,
                description: str = "",
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
                        self._agent_loader.agent,
                        _stream_input,
                        self._conversation_tid,
                        metadata=metadata,
                    ):
                        event_type = state.handle_event(event)

                        if event_type == "usage_stats":
                            self._set_status_usage_baseline(state.last_input_tokens)

                        if _should_finalize_active_summarization(event_type):
                            _finalize_active_summarization()

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
                                    state,
                                    tool_name,
                                    _media_sent,
                                    on_media_cb,
                                )

                        # -- Remove loading spinner on first content event --
                        if not loading_removed and event_type in (
                            "thinking",
                            "text",
                            "tool_call",
                            "summarization_start",
                            "summarization",
                        ):
                            await loading.cleanup()
                            loading_removed = True

                        # -- Widget dispatch --
                        if event_type == "thinking":
                            if thinking_w is None:
                                thinking_w = ThinkingWidget(show_thinking=show_thinking)
                                await container.mount(thinking_w)
                            thinking_w.append_text(event.get("content", ""))

                        elif event_type == "summarization_start":
                            if (
                                summarization_w is not None
                                and not summarization_w._is_active
                            ):
                                summarization_w = None
                            if summarization_w is None:
                                summarization_w = SummarizationWidget()
                                await container.mount(summarization_w)

                        elif event_type == "summarization":
                            content = event.get("content", "")
                            if (
                                summarization_w is not None
                                and not summarization_w._is_active
                            ):
                                summarization_w = None
                            if summarization_w is None:
                                summarization_w = SummarizationWidget()
                                await container.mount(summarization_w)
                            if content:
                                summarization_w.append_text(content)

                        elif event_type == "tool_selection":
                            tools = event.get("tools", [])
                            if tools:
                                from .widgets.tool_selection_widget import (
                                    ToolSelectionWidget,
                                )

                                await container.mount(ToolSelectionWidget(tools))
                                _schedule_scroll()

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
                                last_line = (
                                    state.latest_text.strip().split("\n")[-1].strip()
                                )
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
                                self._set_status_streaming_text(state.response_text)

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
                            # Update todo widget on write_todos.
                            # Insert before tool call widget so Task List
                            # panel appears above the tool call.
                            if tool_name == "write_todos" and state.todo_items:
                                if todo_w is None:
                                    todo_w = TodoWidget(state.todo_items)
                                    if tool_id and tool_id in tool_widgets:
                                        await container.mount(
                                            todo_w,
                                            before=tool_widgets[tool_id],
                                        )
                                    else:
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
                                    if (
                                        tw.tool_name == result_name
                                        and tw._status == "running"
                                    ):
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
                            if (
                                result_name in ("write_todos", "read_todos")
                                and state.todo_items
                            ):
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
                                        lambda f=_ask_fn, e=event: f(e),
                                    )
                                else:
                                    # Interactive TUI: display widget, collect via arrow keys
                                    from .widgets.ask_user_widget import AskUserWidget

                                    _prompt = self.query_one("#prompt", ChatTextArea)
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
                                from langgraph.types import (
                                    Command,  # type: ignore[import-untyped]
                                )

                                _stream_input = Command(resume=result)
                                _hitl_resuming = True
                                break  # re-enter outer HITL loop

                        elif event_type == "interrupt":
                            action_reqs = event.get("action_requests", [])
                            n = len(action_reqs) or 1

                            # HITL: check session auto-approve first
                            if self._hitl_auto_approve:
                                from langgraph.types import (
                                    Command,  # type: ignore[import-untyped]
                                )

                                _stream_input = Command(
                                    resume={
                                        "decisions": [
                                            {"type": "approve"} for _ in range(n)
                                        ]
                                    }
                                )
                                _hitl_resuming = True
                                break  # re-enter outer HITL loop

                            # Channel messages: use channel-based text approval
                            if channel_hitl_fn is not None:
                                self._append_system(
                                    "Waiting for channel user approval...",
                                    style="dim italic",
                                )
                                decisions = await asyncio.to_thread(
                                    channel_hitl_fn,
                                    action_reqs,
                                )
                                if decisions is not None:
                                    from langgraph.types import (
                                        Command,  # type: ignore[import-untyped]
                                    )

                                    _stream_input = Command(
                                        resume={"decisions": decisions}
                                    )
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
                            _prompt = self.query_one("#prompt", ChatTextArea)
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
                                from langgraph.types import (
                                    Command,  # type: ignore[import-untyped]
                                )

                                _stream_input = Command(
                                    resume={"decisions": decided_event.decisions}
                                )
                                _hitl_resuming = True
                                break  # re-enter outer HITL loop with resume
                            else:
                                state.pending_interrupt = None
                                for tw in tool_widgets.values():
                                    if tw._status == "running":
                                        tw.set_rejected()
                                self._append_system(
                                    "Tool execution rejected.",
                                    style="yellow",
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
                                while (
                                    clean.endswith("\n...") or clean.rstrip() == "..."
                                ):
                                    clean = clean.rstrip().removesuffix("...").rstrip()
                                assistant_w = AssistantMessage(
                                    clean or state.response_text
                                )
                                await container.mount(assistant_w)
                                self._schedule_scroll_to_bottom(
                                    container,
                                    delays=(0.15, 0.4, 0.8, 1.5),
                                    immediate=False,
                                )
                            # Mount token usage stats
                            if state.total_input_tokens or state.total_output_tokens:
                                await container.mount(
                                    UsageWidget(
                                        state.total_input_tokens,
                                        state.total_output_tokens,
                                    )
                                )

                        elif event_type == "error":
                            error_msg = event.get("message", "Unknown error")
                            self._append_system(f"Error: {error_msg}", style="red")

                        # Scroll after Textual processes the layout update
                        _schedule_scroll()

                    response = (state.response_text or "").strip()

                except asyncio.CancelledError:
                    # Ctrl+C cancellation — re-raise so _run_turn can handle it
                    raise
                except Exception as exc:
                    error_msg = str(exc)
                    if (
                        "authentication" in error_msg.lower()
                        or "api_key" in error_msg.lower()
                    ):
                        self._append_system(
                            "Error: API key not configured.",
                            style="red",
                        )
                        self._append_system(
                            "Run EvoSci onboard to set up your API key.",
                            style="dim",
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
                    self._schedule_scroll_to_bottom(container)

                # HITL / ask_user: if interrupt was handled, loop back to resume stream
                if state.pending_interrupt is None and state.pending_ask_user is None:
                    break  # normal completion or rejection — exit HITL loop
                # Otherwise _stream_input was set to Command(resume=...)
                # by the interrupt handler above; loop continues.

            return response

        async def _run_turn(self, user_text: str) -> None:
            """Handle a user turn: stream agent response with widgets."""
            cancelled = False
            try:
                self._busy = True
                self._render_status()

                # Resolve @file mentions — inject file contents before sending to agent.
                # Use self._workspace_dir (current session) not the startup-captured
                # workspace_dir closure, which becomes stale after /new or /resume.
                _, message_to_send, file_warnings = await asyncio.to_thread(
                    resolve_file_mentions, user_text, self._workspace_dir
                )
                await self._refresh_status_snapshot(message_to_send)

                # Block the turn on MCP tools finishing, if still in flight.
                # ``_on_agent_load_failure`` is the sole reporter for load
                # errors; callers just return so the send is dropped.
                try:
                    await self._await_agent_ready()
                except Exception:
                    return

                await self._stream_with_widgets(
                    message_to_send,
                    display_text=user_text,
                    file_warnings=file_warnings,
                )
            except asyncio.CancelledError:
                cancelled = True
                self._append_system("\nInterrupted by user", style="dim italic #ffe082")
            finally:
                self._busy = False
                self._run_task = None
                await self._refresh_status_snapshot(reset_streaming_text=True)
                self._render_status()
                self.query_one("#prompt", ChatTextArea).focus()

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
            prompt_widget = None
            try:
                self._busy = True
                await self._refresh_status_snapshot(msg.content)
                self._render_status()

                prompt_widget = self.query_one("#prompt", ChatTextArea)
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
                            _channel_logger.debug(
                                f"{label} send failed: {f.exception()}"
                            )
                            if f.exception()
                            else None
                        )
                    )

                def _send_thinking(thinking: str) -> None:
                    ch = msg.channel_ref
                    if ch and ch.send_thinking:
                        _send_to_channel(
                            ch.send_thinking_message(
                                sender=msg.chat_id,
                                thinking=thinking,
                                metadata=msg.metadata,
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

                from ..commands.channel_ui import ChannelCommandUI

                # Handle slash commands from channel
                if msg.content.strip().startswith("/"):
                    # Only wait for the agent if the command actually
                    # needs it — otherwise ``/mcp add`` & friends would
                    # hang behind a failing MCP load they're meant to fix.
                    cmd = cmd_manager.resolve(msg.content)
                    agent = None
                    if cmd is not None and cmd.requires_agent:
                        try:
                            agent = await self._await_agent_ready()
                        except Exception as exc:
                            _set_channel_response(msg.msg_id, f"Error: {exc}")
                            return
                    ctx = CommandContext(
                        agent=agent,
                        thread_id=self._conversation_tid,
                        ui=ChannelCommandUI(
                            msg,
                            append_system_callback=self._append_system,
                            start_new_session_callback=self.start_new_session,
                            handle_session_resume_callback=self.handle_session_resume,
                        ),
                        workspace_dir=self._workspace_dir,
                        checkpointer=self._checkpointer,
                    )
                    try:
                        cmd_executed = await cmd_manager.execute(msg.content, ctx)
                    except Exception as _cmd_exc:
                        # Command raised — report the error and do NOT fall through
                        # to _stream_with_widgets (which would treat the slash
                        # command text as a plain user message to the agent).
                        _channel_logger.debug(
                            f"Channel command error: {_cmd_exc}", exc_info=True
                        )
                        _set_channel_response(msg.msg_id, f"Command error: {_cmd_exc}")
                        return  # outer finally handles _busy / widget cleanup

                    if cmd_executed:
                        self._append_system(
                            f"[{msg.channel_type}: Executed command from {msg.sender}]",
                            style="dim",
                        )
                        _set_channel_response(
                            msg.msg_id, f"Command executed: {msg.content}"
                        )
                        return  # outer finally handles _busy / widget cleanup

                # Non-slash message — streams through the agent, so wait
                # for readiness now.
                try:
                    await self._await_agent_ready()
                except Exception as exc:
                    _set_channel_response(msg.msg_id, f"Error: {exc}")
                    return

                response = ""
                try:
                    response = await self._stream_with_widgets(
                        msg.content,
                        on_thinking_cb=_send_thinking
                        if self._channel_send_thinking
                        else None,
                        on_todo_cb=_send_todo,
                        on_media_cb=_send_media,
                        skip_user_message=True,
                        channel_hitl_fn=_channel_hitl_prompt,
                        channel_ask_user_fn=_channel_ask_user,
                    )
                except Exception as exc:
                    response = f"Error: {exc}"
                    self._append_system(f"Error: {exc}", style="red")

                _set_channel_response(msg.msg_id, response)
                self._append_system(
                    f"[{msg.channel_type}: Replied to {msg.sender}]",
                    style="dim",
                )

            finally:
                self._busy = False
                await self._refresh_status_snapshot(reset_streaming_text=True)
                self._render_status()
                if prompt_widget is not None:
                    prompt_widget.disabled = False
                    prompt_widget.focus()

        # ── Clipboard (copy on mouse select) ─────────────────

        def on_mouse_up(self, event: MouseUp) -> None:
            """Copy mouse-selected text to clipboard on release."""
            copy_selection_to_clipboard(self)

        # ── Input handling ─────────────────────────────────────

        async def on_chat_text_area_submitted(
            self, event: ChatTextArea.Submitted
        ) -> None:
            text = event.value.strip()
            prompt = self.query_one("#prompt", ChatTextArea)
            prompt.value = ""
            self._quit_pending = False
            self._history_index = -1
            self._history_saved_input = ""

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
                _task = asyncio.create_task(self._handle_command(text))
                self._background_tasks.add(_task)
                _task.add_done_callback(self._background_tasks.discard)
                return

            self._history_suggester.append_entry(text)
            self._run_task = asyncio.ensure_future(self._run_turn(text))

        def on_text_area_changed(self, event: ChatTextArea.Changed) -> None:
            text = event.text_area.text
            comp_widget = self.query_one("#completions", Static)

            # @file mention completion
            if "@" in text:
                candidates = complete_file_mention(text, workspace_dir)
                if candidates:
                    self._comp_items = candidates
                    self._comp_index = -1
                    self._render_completions()
                    comp_widget.display = True
                    return

            if text.startswith("/"):
                prefix = text.lower()
                matches = [
                    (cmd, desc)
                    for cmd, desc in cmd_manager.list_commands()
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
            parts.append(
                ("  [press up to edit last \u00b7 esc to cancel last]", "dim italic")
            )
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
            # Delegate to focused interactive widget
            focused = self.focused
            if focused is not None:
                from .widgets.approval_widget import ApprovalWidget
                from .widgets.mcp_browser import MCPBrowserWidget
                from .widgets.model_picker import ModelPickerWidget
                from .widgets.skill_browser import SkillBrowserWidget
                from .widgets.thread_selector import ThreadPickerWidget

                if isinstance(focused, ApprovalWidget):
                    focused.action_select_reject()
                    return
                if isinstance(focused, ThreadPickerWidget):
                    focused.action_cancel()
                    return
                if isinstance(focused, SkillBrowserWidget):
                    focused.action_cancel()
                    return
                if isinstance(focused, MCPBrowserWidget):
                    focused.action_cancel()
                    return
                if isinstance(focused, ModelPickerWidget):
                    focused.action_cancel()
                    if (
                        self._model_picker_future
                        and not self._model_picker_future.done()
                    ):
                        self._model_picker_future.set_result(None)
                    return
            if self._queued_messages:
                self._queued_messages.pop()
                self._render_queue_indicator()

        def action_edit_queued(self) -> None:
            """Pop the last queued message back into input for editing."""
            # Handle completion list selection (up key)
            comp_widget = self.query_one("#completions", Static)
            if comp_widget.display and self._comp_items:
                self._comp_index = (self._comp_index - 1) % len(self._comp_items)
                self._render_completions()
                return

            # Skip if an interactive picker widget has focus
            focused = self.focused
            if focused is not None:
                from .widgets.approval_widget import ApprovalWidget
                from .widgets.ask_user_widget import AskUserWidget
                from .widgets.mcp_browser import MCPBrowserWidget
                from .widgets.model_picker import ModelPickerWidget
                from .widgets.skill_browser import SkillBrowserWidget
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
                if isinstance(focused, SkillBrowserWidget):
                    focused.action_move_up()
                    return
                if isinstance(focused, MCPBrowserWidget):
                    focused.action_move_up()
                    return
                if isinstance(focused, ModelPickerWidget):
                    focused.action_move_up()
                    return
            if self._queued_messages:
                last = self._queued_messages.pop()
                prompt = self.query_one("#prompt", ChatTextArea)
                prompt.value = last
                prompt.focus()
                self._render_queue_indicator()
                return

            # History browsing (up key)
            entries = self._history_suggester._entries
            if not entries:
                return
            prompt = self.query_one("#prompt", ChatTextArea)
            if self._history_index == -1:
                # Save current input before entering history
                self._history_saved_input = prompt.value
            if self._history_index + 1 < len(entries):
                self._history_index += 1
                prompt.value = entries[self._history_index]
                prompt.focus()

        def action_down_delegate(self) -> None:
            """Delegate down key to focused interactive widget."""
            # Handle completion list selection (down key)
            comp_widget = self.query_one("#completions", Static)
            if comp_widget.display and self._comp_items:
                self._comp_index = (self._comp_index + 1) % len(self._comp_items)
                self._render_completions()
                return

            focused = self.focused
            if focused is not None:
                from .widgets.approval_widget import ApprovalWidget
                from .widgets.ask_user_widget import AskUserWidget
                from .widgets.mcp_browser import MCPBrowserWidget
                from .widgets.model_picker import ModelPickerWidget
                from .widgets.skill_browser import SkillBrowserWidget
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
                if isinstance(focused, SkillBrowserWidget):
                    focused.action_move_down()
                    return
                if isinstance(focused, MCPBrowserWidget):
                    focused.action_move_down()
                    return
                if isinstance(focused, ModelPickerWidget):
                    focused.action_move_down()
                    return

            # History browsing (down key)
            if self._history_index >= 0:
                prompt = self.query_one("#prompt", ChatTextArea)
                self._history_index -= 1
                if self._history_index == -1:
                    # Back to saved input
                    prompt.value = self._history_saved_input
                else:
                    prompt.value = self._history_suggester._entries[self._history_index]
                prompt.focus()

        def action_paste_clipboard(self) -> None:
            """Paste text from system clipboard into the input field."""
            text = get_clipboard_text()
            if not text:
                self.notify(
                    "Clipboard is empty or unavailable",
                    severity="warning",
                    timeout=2,
                )
                return

            prompt = self.query_one("#prompt", ChatTextArea)
            prompt.insert(text)
            prompt.focus()

        def action_tab_complete(self) -> None:
            """Handle TAB: cycle completions when visible, otherwise no-op.

            Registered as a priority binding so it intercepts before Textual's
            default focus-next behaviour, which would steal focus from the input
            and lose the cursor.
            """
            comp_widget = self.query_one("#completions", Static)
            if not (comp_widget.display and self._comp_items):
                # No completions active — keep focus on the prompt.
                self.query_one("#prompt", ChatTextArea).focus()
                return
            self._comp_index = (self._comp_index + 1) % len(self._comp_items)
            self._apply_selected_completion()

        def _handle_completion_enter(self) -> bool:
            """Called by ChatTextArea before submitting on Enter.

            If a completion is active and an item is selected, apply it
            and suppress the submit.  If the list is visible but nothing
            is selected (index == -1), select the first item instead of
            submitting the raw prefix.

            Returns:
                True to suppress submit, False to allow it.
            """
            comp_widget = self.query_one("#completions", Static)
            if not (comp_widget.display and self._comp_items):
                return False

            # If no item highlighted yet, select the first one
            if self._comp_index < 0:
                self._comp_index = 0

            self._apply_selected_completion()
            self._hide_completions()
            return True

        def _apply_selected_completion(self) -> None:
            """Apply the currently selected completion to the input field.

            For ``@file`` completions the last ``@token`` is replaced in-place;
            for slash-command completions the entire input is replaced.
            """
            if self._comp_index < 0 or self._comp_index >= len(self._comp_items):
                return
            selected = self._comp_items[self._comp_index][0]
            prompt = self.query_one("#prompt", ChatTextArea)

            if selected.startswith("@"):
                import re as _re

                current = prompt.value
                m = _re.search(r"@[^\s]*$", current)
                if m:
                    new_val = current[: m.start()] + selected + " "
                else:
                    new_val = current + selected + " "
                prompt.value = new_val
            else:
                prompt.value = selected + " "

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
                    comp_text.append(f"{cmd:<30}", style="bold")
                    comp_text.append(desc, style="bold")
                else:
                    comp_text.append("  ", style="#888888")
                    comp_text.append(f"{cmd:<30}", style="#888888")
                    comp_text.append(desc, style="#888888")
                if i < len(self._comp_items) - 1:
                    comp_text.append("\n")
            self.query_one("#completions", Static).update(comp_text)

        # ── Slash commands ─────────────────────────────────────

        async def _handle_command(self, command: str) -> None:
            # Echo the command so the user sees what they ran
            self._append_system(command.strip(), style="cyan")

            # Block new user input while the command runs (important for slow
            # commands like /compact that call an LLM internally).
            prompt_widget = self.query_one("#prompt", ChatTextArea)
            self._busy = True
            prompt_widget.disabled = True
            self._render_status()

            try:
                # Only gate on agent readiness for commands that need it —
                # recovery commands like ``/mcp add`` must run even when
                # ``_await_agent_ready`` would hang on a broken MCP load.
                cmd = cmd_manager.resolve(command)
                agent = None
                if cmd is not None and cmd.requires_agent:
                    try:
                        agent = await self._await_agent_ready()
                    except Exception:
                        # ``_on_agent_load_failure`` already surfaced the error.
                        return
                ctx = CommandContext(
                    agent=agent,
                    thread_id=self._conversation_tid,
                    ui=self,
                    workspace_dir=self._workspace_dir,
                    checkpointer=self._checkpointer,
                    input_tokens_hint=self._status_last_input_tokens,
                )

                if await cmd_manager.execute(command, ctx):
                    # Sync agent back if command replaced it (e.g. /model)
                    if ctx.agent is not self._agent_loader.agent:
                        self._agent_loader.agent = ctx.agent
                        if _channels_is_running():
                            _ch_mod._cli_agent = ctx.agent
                            _ch_mod._cli_thread_id = self._conversation_tid
                    # Do NOT invalidate the usage baseline after /compact.
                    # build_session_status_snapshot() only counts raw checkpoint
                    # messages (~46 tokens) and misses system prompt + tool
                    # definitions (~50K overhead). The stale pre-compact count
                    # is far more accurate; the next LLM call will correct it.
                    await self._refresh_status_snapshot(
                        reset_streaming_text=True,
                    )
                    return

                self._append_system(f"Unknown command: {command}", style="yellow")
                self._render_status()
            finally:
                self._busy = False
                prompt_widget.disabled = False
                prompt_widget.focus()

        async def _render_history(self, thread_id_value: str) -> None:
            """Render conversation history from a saved thread.

            Restores human messages and AI responses (with Markdown and
            thinking panels). Tool calls and other intermediate steps are
            skipped — they are difficult to faithfully reproduce from
            checkpoint data.
            """
            messages = await get_thread_messages(thread_id_value)
            if not messages:
                return

            HISTORY_WINDOW = 50
            container = self.query_one("#chat", VerticalScroll)

            # Only human and ai messages; skip tool/system/other
            display = [
                m for m in messages if getattr(m, "type", None) in ("human", "ai")
            ]

            if len(display) > HISTORY_WINDOW:
                skipped = len(display) - HISTORY_WINDOW
                display = display[-HISTORY_WINDOW:]
                await container.mount(
                    SystemMessage(
                        f"── ... {skipped} earlier messages ──", msg_style="dim"
                    )
                )
            else:
                await container.mount(
                    SystemMessage("── Conversation history ──", msg_style="dim")
                )

            for message in display:
                msg_type = getattr(message, "type", None)
                content = getattr(message, "content", "") or ""

                if msg_type == "human":
                    if isinstance(content, list):
                        parts = [
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        ]
                        content = " ".join(parts) if parts else ""
                    content = content.strip()
                    if content:
                        await container.mount(UserMessage(content))

                elif msg_type == "ai":
                    # Extract thinking and text blocks from content list
                    thinking_text = ""
                    text_content = ""
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "thinking":
                                thinking_text += block.get("thinking", "")
                            elif block.get("type") == "text":
                                text_content += block.get("text", "")
                    else:
                        text_content = content or ""
                    text_content = text_content.strip()

                    # Render thinking as collapsed panel (click to expand)
                    if thinking_text.strip() and show_thinking:
                        w = ThinkingWidget(show_thinking=True)
                        await container.mount(w)
                        w.append_text(thinking_text)
                        w.finalize()

                    # Render AI response with full Markdown
                    if text_content:
                        await container.mount(AssistantMessage(text_content))

            await container.mount(
                SystemMessage("── End of history ──", msg_style="dim")
            )
            # History can hold dozens of Markdown-heavy AssistantMessages
            # whose async layout keeps growing virtual_size for several
            # seconds; schedule enough retries to catch the final reflow.
            self._schedule_scroll_to_bottom(
                container, delays=(0.1, 0.3, 0.6, 1.0, 1.8, 3.0)
            )

        # ── Quit handling ──────────────────────────────────────

        def _arm_quit_pending(self, shortcut: str) -> None:
            """Set the pending-quit flag and show a matching hint."""
            self._quit_pending = True
            quit_timeout = 3  # seconds
            self.notify(f"Press {shortcut} again to quit", timeout=quit_timeout)
            self.set_timer(
                quit_timeout,
                lambda: setattr(self, "_quit_pending", False),
            )

        def force_quit(self) -> None:
            """Exit immediately without double-press confirmation (used by /exit command)."""
            self._do_exit()

        def _do_exit(self) -> None:
            """Clean up channels and exit."""
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

        def action_request_quit(self) -> None:
            if self._busy:
                self._quit_pending = False
                # Clear all queued messages on interrupt
                if self._queued_messages:
                    self._queued_messages.clear()
                    self._render_queue_indicator()
                if self._run_task is not None and not self._run_task.done():
                    self._run_task.cancel()
                else:
                    # Edge case: busy but no task — force reset
                    self._busy = False
                    self.query_one("#prompt", ChatTextArea).focus()
                    self._render_status()
                    self._append_system(
                        "\nInterrupted by user", style="dim italic #ffe082"
                    )
                return
            # Double Ctrl+C to quit
            if self._quit_pending:
                self._do_exit()
            else:
                self._arm_quit_pending("Ctrl+C")

        # ── Banner & status ────────────────────────────────────

        async def _refresh_status_snapshot(
            self,
            pending_user_text: str | None = None,
            *,
            reset_streaming_text: bool = True,
        ) -> None:
            """Recompute persistent status metrics for the active thread."""
            pending = (pending_user_text or "").strip()
            if pending:
                if self._status_last_input_tokens is not None:
                    self._status_base_snapshot = apply_user_text_to_snapshot(
                        make_usage_status_snapshot(
                            self._status_last_input_tokens,
                            model_name=self._current_model,
                        ),
                        pending,
                    )
                else:
                    self._status_base_snapshot = await build_session_status_snapshot(
                        self._conversation_tid,
                        model_name=self._current_model,
                        pending_user_text=pending,
                    )
            elif self._status_last_input_tokens is not None:
                self._status_base_snapshot = make_usage_status_snapshot(
                    self._status_last_input_tokens,
                    model_name=self._current_model,
                )
            else:
                self._status_base_snapshot = await build_session_status_snapshot(
                    self._conversation_tid,
                    model_name=self._current_model,
                )
            if reset_streaming_text:
                self._status_streaming_text = ""
            self._rebuild_status_snapshot()

        def _set_status_usage_baseline(self, input_tokens: int) -> None:
            """Promote the latest real prompt usage into the status-bar base."""
            if input_tokens <= 0:
                return
            self._status_last_input_tokens = input_tokens
            self._status_base_snapshot = make_usage_status_snapshot(
                input_tokens,
                model_name=self._current_model,
            )
            self._rebuild_status_snapshot()

        def update_status_after_compact(self, tokens_after: int) -> None:
            """Update the status bar immediately after a successful /compact.

            Called by CompactCommand so the bar reflects the reduced context
            without waiting for the next LLM call.
            """
            if tokens_after <= 0:
                return
            self._status_last_input_tokens = tokens_after
            self._status_base_snapshot = make_usage_status_snapshot(
                tokens_after,
                model_name=self._current_model,
            )
            self._rebuild_status_snapshot()

        def update_status_after_model_change(
            self, new_model: str, new_provider: str | None = None
        ) -> None:
            """Update the status bar and welcome banner after /model switches the LLM."""
            self._current_model = new_model
            if new_provider is not None:
                self._current_provider = new_provider
            self._status_base_snapshot = make_empty_status_snapshot(new_model)
            self._rebuild_status_snapshot()
            self._render_welcome()

        def _set_status_streaming_text(self, text: str | None) -> None:
            """Update in-flight assistant text shown in the context bar."""
            new_text = text or ""
            if new_text == self._status_streaming_text:
                return
            self._status_streaming_text = new_text
            self._rebuild_status_snapshot()

        def _rebuild_status_snapshot(self) -> None:
            """Compose the displayed snapshot from base state + live overlay."""
            self._status_snapshot = apply_assistant_text_to_snapshot(
                self._status_base_snapshot,
                self._status_streaming_text,
            )
            self._render_status()

        def _render_welcome(self) -> None:
            channels_info: list[tuple[str, bool, str]] | None = None
            try:
                running = _channels_running_list()
                started = self._started_channel_types
                if running or started:
                    all_types = list(dict.fromkeys(running + started))
                    channels_info = [(ct, True, "connected (bus)") for ct in all_types]
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
                            channels_info = [(ct, False, "configured") for ct in types]
            except Exception:
                pass

            welcome = self.query_one("#welcome", Static)
            welcome.update(
                _build_welcome_banner(
                    thread_id=self._conversation_tid,
                    workspace_dir=self._workspace_dir,
                    mode=mode,
                    model=self._current_model,
                    provider=self._current_provider,
                    ui_backend="tui",
                    channels=channels_info,
                )
            )

        def _render_status(self) -> None:
            status = self.query_one("#status", Static)
            width = (
                getattr(status.size, "width", 0)
                or getattr(status.content_region, "width", 0)
                or getattr(self.screen.size, "width", 0)
                or 80
            )
            # MCP load progress lives in the dedicated MCPLoaderWidget
            # above the input bar — no need to duplicate it here.
            if self._busy:
                hint_label = "vibe researching..."
                hint_style = f"on {STATUS_BAR_BG} {STATUS_HINT_BUSY} bold"
            else:
                hint_label = "/help for commands"
                hint_style = f"on {STATUS_BAR_BG} {STATUS_HINT_IDLE}"

            hint = Text.assemble(
                (hint_label, hint_style),
                (" │ ", f"on {STATUS_BAR_BG} {STATUS_DIM}"),
            )
            remaining_width = max(1, width - len(hint.plain))
            metrics = build_status_text(
                self._status_snapshot,
                self._status_started_at,
                remaining_width,
            )
            line = Text(no_wrap=True, overflow="crop")
            line.append_text(hint)
            line.append_text(metrics)
            status.update(line)

    # ── Media forwarding helper (module-level) ──────────────

    _MEDIA_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".svg",
        ".pdf",
        ".mp3",
        ".wav",
        ".mp4",
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
            effective_thread_id: str | None = None
            resumed = False
            resume_warning = ""
            if thread_id:
                resolved, matches = await resolve_thread_id_prefix(thread_id)
                if resolved:
                    meta = await get_thread_metadata(resolved)
                    ws = (meta or {}).get("workspace_dir", "")
                    if ws:
                        effective_workspace = ws
                    effective_thread_id = resolved
                    resumed = True
                elif matches:
                    resume_warning = (
                        f"Thread prefix '{thread_id}' is ambiguous "
                        f"({', '.join(matches)}). Starting new session."
                    )
                else:
                    resume_warning = (
                        f"Thread '{thread_id}' not found. Starting new session."
                    )
            if not effective_thread_id:
                effective_thread_id = generate_thread_id()

            # The TUI opens instantly and starts MCP loading in the
            # background; ``on_mount`` in the app kicks off the real
            # ``load_agent`` call and awaits it before the first turn.
            app = EvoTextualInteractiveApp(
                thread_id_value=effective_thread_id,
                workspace=effective_workspace,
                checkpointer=checkpointer,
                channel_send_thinking_value=channel_send_thinking,
                resumed=resumed,
                resume_warning=resume_warning,
            )
            try:
                await app.run_async()
            finally:
                from .resume_hint import print_resume_hint

                # Best-effort resume hint — guarded so failures here (e.g.
                # DB teardown race during abnormal shutdown) cannot shadow
                # the original run_async traceback.
                exit_tid = getattr(app, "_conversation_tid", None)
                hint_tid: str | None = None
                if exit_tid:
                    try:
                        if await thread_exists(exit_tid):
                            hint_tid = exit_tid
                    except Exception:
                        _channel_logger.debug(
                            "resume-hint thread_exists lookup failed",
                            exc_info=True,
                        )
                try:
                    print_resume_hint(hint_tid)
                except Exception:
                    _channel_logger.debug("print_resume_hint failed", exc_info=True)

    import nest_asyncio  # type: ignore[import-untyped]

    nest_asyncio.apply()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(_amain())
