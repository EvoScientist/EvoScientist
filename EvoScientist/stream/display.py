"""Rich display functions for streaming CLI output.

Contains all rendering logic: tool call lines, sub-agent sections,
todo panels, streaming display layout, and final results display.
Also provides the shared console and formatter globals.
"""

import asyncio
import os
import sys
from typing import Any, Callable

from rich.console import Console, Group  # type: ignore[import-untyped]
from rich.live import Live  # type: ignore[import-untyped]
from rich.markdown import Markdown  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.spinner import Spinner  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]

from ..paths import resolve_virtual_path
from .formatter import ToolResultFormatter
from .state import StreamState, SubAgentState, _build_todo_stats, _parse_todo_items, _INTERNAL_TOOLS
from .utils import DisplayLimits, ToolStatus, format_tool_compact, is_success
from .events import stream_agent_events

# ---------------------------------------------------------------------------
# Shared globals
# ---------------------------------------------------------------------------

# Media file extensions that should trigger on_file_write callback
_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".pdf"}

console = Console(
    legacy_windows=(sys.platform == 'win32'),
    no_color=os.getenv('NO_COLOR') is not None,
)

formatter = ToolResultFormatter()


# ---------------------------------------------------------------------------
# Todo formatting
# ---------------------------------------------------------------------------

def _format_single_todo(item: dict) -> Text:
    """Format a single todo item with status symbol."""
    status = str(item.get("status", "todo")).lower()
    content_text = str(item.get("content", item.get("task", item.get("title", ""))))

    if status in ("done", "completed", "complete"):
        symbol = "\u2713"
        label = "done  "
        style = "green dim"
    elif status in ("active", "in_progress", "in-progress", "working"):
        symbol = "\u25cf"
        label = "active"
        style = "yellow"
    else:
        symbol = "\u25cb"
        label = "todo  "
        style = "dim"

    line = Text()
    line.append(f"    {symbol} ", style=style)
    line.append(label, style=style)
    line.append(" ", style="dim")
    # Truncate long content
    if len(content_text) > 60:
        content_text = content_text[:57] + "\u2026"
    line.append(content_text, style=style)
    return line


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------

def format_tool_result_compact(_name: str, content: str, max_lines: int = 5) -> list:
    """Format tool result as tree output.

    Special handling for write_todos: shows formatted checklist with status symbols.
    """
    elements = []

    if not content.strip():
        elements.append(Text("  \u2514 (empty)", style="dim"))
        return elements

    # Special handling for write_todos
    if _name == "write_todos":
        items = _parse_todo_items(content)
        if items:
            stats = _build_todo_stats(items)
            stats_line = Text()
            stats_line.append("  \u2514 ", style="dim")
            stats_line.append(stats, style="dim")
            elements.append(stats_line)
            elements.append(Text("", style="dim"))  # blank line

            max_preview = 4
            for item in items[:max_preview]:
                elements.append(_format_single_todo(item))

            remaining = len(items) - max_preview
            if remaining > 0:
                elements.append(Text(f"    ... {remaining} more", style="dim italic"))

            return elements

    lines = content.strip().split("\n")
    total_lines = len(lines)

    display_lines = lines[:max_lines]
    for i, line in enumerate(display_lines):
        prefix = "\u2514" if i == 0 else " "
        if len(line) > 80:
            line = line[:77] + "\u2026"
        style = "dim" if is_success(content) else "red dim"
        elements.append(Text(f"  {prefix} {line}", style=style))

    remaining = total_lines - max_lines
    if remaining > 0:
        elements.append(Text(f"    ... +{remaining} lines", style="dim italic"))

    return elements


# ---------------------------------------------------------------------------
# Tool call line rendering
# ---------------------------------------------------------------------------

def _render_tool_call_line(tc: dict, tr: dict | None) -> Text:
    """Render a single tool call line with status indicator."""
    is_task = tc.get('name', '').lower() == 'task'

    if tr is not None:
        content = tr.get('content', '')
        if is_success(content):
            style = "bold green"
            indicator = "\u2713" if is_task else ToolStatus.SUCCESS.value
        else:
            style = "bold red"
            indicator = "\u2717" if is_task else ToolStatus.ERROR.value
    else:
        style = "bold yellow" if not is_task else "bold cyan"
        indicator = "\u25b6" if is_task else ToolStatus.RUNNING.value

    # Try to get display name from args first
    tool_compact = format_tool_compact(tc['name'], tc.get('args'))

    # If args were empty and we have a result, try to infer memory operations from result
    tool_name = tc.get('name', '').lower()
    if tool_name in ('write_file', 'edit_file') and tr is not None:
        result_content = tr.get('content', '')
        if '/MEMORY.md' in result_content or 'MEMORY.md' in result_content:
            tool_compact = "Updating memory"
    elif tool_name == 'read_file' and tr is not None:
        result_content = tr.get('content', '')
        # read_file result doesn't contain path, check if args is empty and result looks like memory
        args = tc.get('args') or {}
        if not args.get('path') and '# EvoScientist Memory' in result_content:
            tool_compact = "Reading memory"

    tool_text = Text()
    tool_text.append(f"{indicator} ", style=style)
    tool_text.append(tool_compact, style=style)
    return tool_text


# ---------------------------------------------------------------------------
# Sub-agent section rendering
# ---------------------------------------------------------------------------

def _render_subagent_section(sa: 'SubAgentState', compact: bool = False) -> list:
    """Render a sub-agent's activity as a bordered section.

    Args:
        sa: Sub-agent state to render
        compact: If True, render minimal 1-line summary (completed sub-agents)

    Header uses "Cooking with {name}" style matching task tool format.
    Active sub-agents show bordered tool list; completed ones collapse to 1 line.
    """
    elements = []
    BORDER = "dim cyan" if sa.is_active else "dim"

    # Filter out tool calls with empty names
    valid_calls = [tc for tc in sa.tool_calls if tc.get("name")]

    # Split into completed and pending
    completed = []
    pending = []
    for tc in valid_calls:
        tr = sa.get_result_for(tc)
        if tr is not None:
            completed.append((tc, tr))
        else:
            pending.append(tc)

    succeeded = sum(1 for _, tr in completed if tr.get("success", True))
    _ = len(completed) - succeeded  # failed count, unused for now

    # Build display name
    display_name = f"Cooking with {sa.name}"
    if sa.description:
        desc = sa.description.split("\n")[0].strip()
        desc = desc[:50] + "\u2026" if len(desc) > 50 else desc
        display_name += f" \u2014 {desc}"

    # --- Compact mode: 1-line summary for completed sub-agents ---
    if compact:
        line = Text()
        if not sa.is_active:
            line.append("\u2713 ", style="green")
            line.append(display_name, style="green dim")
            total = len(valid_calls)
            line.append(f" ({total} tools)", style="dim")
        else:
            line.append("\u25b6 ", style="cyan")
            line.append(display_name, style="bold cyan")
        elements.append(line)
        return elements

    # --- Full mode: bordered section for Live streaming ---
    MAX_SA_VISIBLE = 3       # max completed tools shown
    MAX_SA_RUNNING = 2       # max running tools shown

    # Header
    header = Text()
    header.append("\u250c ", style=BORDER)
    if sa.is_active:
        header.append(f"\u25b6 {display_name}", style="bold cyan")
    else:
        header.append(f"\u2713 {display_name}", style="bold green")
    elements.append(header)

    # Completed tools — collapse older ones into a summary
    slots = max(0, MAX_SA_VISIBLE - len(pending))
    hidden = completed[:-slots] if slots and len(completed) > slots else (completed if not slots else [])
    visible = completed[-slots:] if slots else []

    if hidden:
        ok = sum(1 for _, tr in hidden if tr.get("success", True))
        fail = len(hidden) - ok
        summary = Text("\u2502 ", style=BORDER)
        summary.append(f"\u2713 {ok} completed", style="dim green")
        if fail > 0:
            summary.append(f" | {fail} failed", style="dim red")
        elements.append(summary)

    for tc, tr in visible:
        tc_line = Text("\u2502 ", style=BORDER)
        tc_name = format_tool_compact(tc["name"], tc.get("args"))
        if tr.get("success", True):
            tc_line.append(f"\u2713 {tc_name}", style="green")
        else:
            tc_line.append(f"\u2717 {tc_name}", style="red")
            content = tr.get("content", "")
            first_line = content.strip().split("\n")[0][:70]
            if first_line:
                err_line = Text("\u2502   ", style=BORDER)
                err_line.append(f"\u2514 {first_line}", style="red dim")
                elements.append(tc_line)
                elements.append(err_line)
                continue
        elements.append(tc_line)

    # Pending/running tools — limit visible
    hidden_running = len(pending) - MAX_SA_RUNNING
    if hidden_running > 0:
        run_summary = Text("\u2502 ", style=BORDER)
        run_summary.append(f"\u25cf {hidden_running} more running...", style="dim yellow")
        elements.append(run_summary)
        pending = pending[-MAX_SA_RUNNING:]

    for tc in pending:
        tc_line = Text("\u2502 ", style=BORDER)
        tc_name = format_tool_compact(tc["name"], tc.get("args"))
        tc_line.append(f"\u25cf {tc_name}", style="bold yellow")
        elements.append(tc_line)
        spinner_line = Text("\u2502   ", style=BORDER)
        spinner_line.append("\u21bb running...", style="yellow dim")
        elements.append(spinner_line)

    # Footer
    if not sa.is_active:
        total = len(valid_calls)
        footer = Text(f"\u2514 done ({total} tools)", style="dim green")
        elements.append(footer)
    elif valid_calls:
        footer = Text("\u2514 running...", style="dim cyan")
        elements.append(footer)

    return elements


# ---------------------------------------------------------------------------
# Todo panel
# ---------------------------------------------------------------------------

def _render_todo_panel(todo_items: list[dict]) -> Panel:
    """Render a bordered Task List panel from todo items.

    Matches the style: cyan border, status icons per item.
    """
    lines = Text()
    for i, item in enumerate(todo_items):
        if i > 0:
            lines.append("\n")
        status = str(item.get("status", "todo")).lower()
        content_text = str(item.get("content", item.get("task", item.get("title", ""))))

        if status in ("done", "completed", "complete"):
            symbol = "\u2713"  # checkmark
            style = "green dim"
        elif status in ("active", "in_progress", "in-progress", "working"):
            symbol = "\u23f3"  # hourglass
            style = "yellow"
        else:
            symbol = "\u25a1"  # empty square
            style = "dim"

        lines.append(f"{symbol} ", style=style)
        lines.append(content_text, style=style)

    return Panel(
        lines,
        title="Task List",
        title_align="center",
        border_style="cyan",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Streaming display layout
# ---------------------------------------------------------------------------

def create_streaming_display(
    thinking_text: str = "",
    response_text: str = "",
    latest_text: str = "",
    tool_calls: list | None = None,
    tool_results: list | None = None,
    is_thinking: bool = False,
    is_responding: bool = False,
    is_waiting: bool = False,
    is_processing: bool = False,
    show_thinking: bool = True,
    subagents: list | None = None,
    todo_items: list | None = None,
    is_final: bool = False,
    final_show_thinking: bool = False,
    final_thinking_max_length: int = DisplayLimits.THINKING_FINAL,
) -> Any:
    """Create Rich display layout for streaming output.

    Returns:
        Rich Group for Live display
    """
    elements = []
    tool_calls = tool_calls or []
    tool_results = tool_results or []
    subagents = subagents or []

    # Initial waiting state
    if is_waiting and not thinking_text and not response_text and not tool_calls:
        elements.append(Spinner("dots", text=" Thinking...", style="cyan"))
        return Group(*elements)

    # Thinking panel
    _show_thinking = (final_show_thinking if is_final else show_thinking)
    if _show_thinking and thinking_text:
        thinking_title = "Thinking"
        display_thinking = thinking_text.rstrip()
        if is_final:
            # Final frame: middle-elision truncation
            if len(display_thinking) > final_thinking_max_length:
                half = final_thinking_max_length // 2
                display_thinking = display_thinking[:half] + "\n\n... (truncated) ...\n\n" + display_thinking[-half:]
        else:
            if is_thinking:
                thinking_title += " ..."
            if len(display_thinking) > DisplayLimits.THINKING_STREAM:
                display_thinking = "..." + display_thinking[-DisplayLimits.THINKING_STREAM:]
        elements.append(Panel(
            Text(display_thinking, style="dim"),
            title=thinking_title,
            border_style="blue",
            padding=(0, 1),
        ))

    # Tool calls and results paired display
    # Collapse older completed tools to prevent overflow in Live mode
    # Task tool calls are ALWAYS visible (they represent sub-agent delegations)
    MAX_VISIBLE_TOOLS = 4
    MAX_VISIBLE_RUNNING = 3

    if tool_calls:
        # Split into categories
        completed_regular = []   # completed non-task tools
        task_tools = []          # task tools (always visible)
        running_regular = []     # running non-task tools

        for i, tc in enumerate(tool_calls):
            has_result = i < len(tool_results)
            tr = tool_results[i] if has_result else None
            is_task = tc.get('name') == 'task'

            # Skip internal middleware tools
            if tc.get('name') in _INTERNAL_TOOLS:
                continue

            if is_task:
                # Skip task calls with empty args (still streaming)
                if tc.get('args'):
                    task_tools.append((tc, tr))
            elif has_result:
                completed_regular.append((tc, tr))
            else:
                running_regular.append((tc, None))

        if is_final:
            # Final frame: show ALL tools expanded, no spinners, no collapsing
            shown_sa_names: set[str] = set()

            for tc, tr in completed_regular:
                elements.append(_render_tool_call_line(tc, tr))
                content = tr.get('content', '') if tr else ''
                if tr and not is_success(content):
                    result_elements = format_tool_result_compact(
                        tr['name'], content, max_lines=10,
                    )
                    elements.extend(result_elements)

            # Task tools with compact sub-agent summaries
            for tc, tr in task_tools:
                elements.append(_render_tool_call_line(tc, tr))
                sa_name = tc.get('args', {}).get('subagent_type', '')
                task_desc = tc.get('args', {}).get('description', '')
                matched_sa = None
                for sa in subagents:
                    if sa.name == sa_name or (task_desc and task_desc in (sa.description or '')):
                        matched_sa = sa
                        break
                if matched_sa:
                    shown_sa_names.add(matched_sa.name)
                    elements.extend(_render_subagent_section(matched_sa, compact=True))

            # Render any sub-agents not already shown via task tool calls
            for sa in subagents:
                if sa.name not in shown_sa_names and (sa.tool_calls or sa.is_active):
                    elements.extend(_render_subagent_section(sa, compact=True))

        else:
            # Streaming mode: collapse older tools, show spinners
            # --- Completed regular tools (collapsible) ---
            slots = max(0, MAX_VISIBLE_TOOLS - len(running_regular))
            hidden = completed_regular[:-slots] if slots and len(completed_regular) > slots else (completed_regular if not slots else [])
            visible = completed_regular[-slots:] if slots else []

            if hidden:
                ok = sum(1 for _, tr in hidden if is_success(tr.get('content', '')))
                fail = len(hidden) - ok
                summary = Text()
                summary.append(f"\u2713 {ok} completed", style="dim green")
                if fail > 0:
                    summary.append(f" | {fail} failed", style="dim red")
                elements.append(summary)

            for tc, tr in visible:
                elements.append(_render_tool_call_line(tc, tr))
                content = tr.get('content', '') if tr else ''
                if tr and not is_success(content):
                    result_elements = format_tool_result_compact(
                        tr['name'], content, max_lines=5,
                    )
                    elements.extend(result_elements)

            # --- Running regular tools (limit visible) ---
            hidden_running = len(running_regular) - MAX_VISIBLE_RUNNING
            if hidden_running > 0:
                summary = Text()
                summary.append(f"\u25cf {hidden_running} more running...", style="dim yellow")
                elements.append(summary)
                running_regular = running_regular[-MAX_VISIBLE_RUNNING:]

            for tc, tr in running_regular:
                elements.append(_render_tool_call_line(tc, tr))
                elements.append(Spinner("dots", text=" Running...", style="yellow"))

            # Task tool calls are rendered as part of sub-agent sections below

    # Response text handling — exclude internal tools (e.g. ExtractedMemory)
    # from the "done" calculation so they don't block final Markdown rendering.
    _n_visible = 0
    _n_visible_done = 0
    for i, tc in enumerate(tool_calls):
        if tc.get('name') in _INTERNAL_TOOLS:
            continue
        _n_visible += 1
        if i < len(tool_results):
            _n_visible_done += 1
    has_pending_tools = _n_visible > _n_visible_done
    any_active_subagent = any(sa.is_active for sa in subagents)
    has_used_tools = _n_visible > 0
    all_done = not has_pending_tools and not any_active_subagent and not is_processing

    if is_final:
        # Final frame: render todo panel + response (tools/subagents handled above).
        # Skip narration, spinners — but KEEP response so it persists on screen
        # when Live exits (transient=False).
        todo_items = todo_items or []
        if todo_items:
            elements.append(Text(""))  # blank separator
            elements.append(_render_todo_panel(todo_items))

        # Include response in final frame so it stays visible after Live exits
        if response_text:
            clean_response = response_text.strip()
            while clean_response.endswith("\n...") or clean_response.rstrip() == "...":
                clean_response = clean_response.rstrip().removesuffix("...").rstrip()
            if clean_response:
                elements.append(Text(""))  # blank separator
                elements.append(Markdown(clean_response))
    else:
        # Intermediate narration (tools still running) -- dim italic above Task List
        if latest_text and has_used_tools and not all_done:
            preview = latest_text.strip()
            if preview:
                last_line = preview.split("\n")[-1].strip()
                if last_line:
                    if len(last_line) > 60:
                        last_line = last_line[:57] + "\u2026"
                    elements.append(Text(f"    {last_line}", style="dim italic"))

        # Task List panel (persistent, updates on write_todos / read_todos)
        todo_items = todo_items or []
        if todo_items:
            elements.append(Text(""))  # blank separator
            elements.append(_render_todo_panel(todo_items))

        # Sub-agent activity sections
        # Active: full bordered view; Completed: compact 1-line summary
        for sa in subagents:
            if sa.tool_calls or sa.is_active:
                elements.extend(_render_subagent_section(sa, compact=not sa.is_active))

        # Processing state after tool execution
        if is_processing and not is_thinking and not is_responding and not response_text:
            # Check if any sub-agent is active
            any_active = any(sa.is_active for sa in subagents)
            if not any_active:
                elements.append(Spinner("dots", text=" Analyzing results...", style="cyan"))

        # Stream response in real-time as tokens arrive (all tools done)
        if response_text and all_done:
            elements.append(Text(""))  # blank separator
            elements.append(Markdown(response_text))

    if not elements:
        return Group(Spinner("dots", text=" Processing...", style="cyan"))
    return Group(*elements)


# ---------------------------------------------------------------------------
# Final results display
# ---------------------------------------------------------------------------

def display_final_results(
    state: StreamState,
    thinking_max_length: int = DisplayLimits.THINKING_FINAL,
    show_thinking: bool = True,
    show_tools: bool = True,
) -> None:
    """Display final results after streaming completes."""
    if show_thinking and state.thinking_text:
        display_thinking = state.thinking_text.rstrip()
        if len(display_thinking) > thinking_max_length:
            half = thinking_max_length // 2
            display_thinking = display_thinking[:half] + "\n\n... (truncated) ...\n\n" + display_thinking[-half:]
        console.print(Panel(
            Text(display_thinking, style="dim"),
            title="Thinking",
            border_style="blue",
        ))

    if show_tools and state.tool_calls:
        shown_sa_names: set[str] = set()

        for i, tc in enumerate(state.tool_calls):
            has_result = i < len(state.tool_results)
            tr = state.tool_results[i] if has_result else None
            content = tr.get('content', '') if tr is not None else ''
            tool_name = tc.get('name', '')
            is_task = tool_name.lower() == 'task'

            # Skip internal middleware tools
            if tool_name in _INTERNAL_TOOLS:
                continue

            # Task tools: show delegation line + compact sub-agent summary
            if is_task:
                console.print(_render_tool_call_line(tc, tr))
                sa_name = tc.get('args', {}).get('subagent_type', '')
                task_desc = tc.get('args', {}).get('description', '')
                matched_sa = None
                for sa in state.subagents:
                    if sa.name == sa_name or (task_desc and task_desc in (sa.description or '')):
                        matched_sa = sa
                        break
                if matched_sa:
                    shown_sa_names.add(matched_sa.name)
                    for elem in _render_subagent_section(matched_sa, compact=True):
                        console.print(elem)
                continue

            # Regular tools: show tool call line + result
            console.print(_render_tool_call_line(tc, tr))
            if has_result and tr is not None:
                result_elements = format_tool_result_compact(
                    tr['name'],
                    content,
                    max_lines=10,
                )
                for elem in result_elements:
                    console.print(elem)

        # Render any sub-agents not already shown via task tool calls
        for sa in state.subagents:
            if sa.name not in shown_sa_names and (sa.tool_calls or sa.is_active):
                for elem in _render_subagent_section(sa, compact=True):
                    console.print(elem)

        console.print()

    # Task List panel in final output
    if state.todo_items:
        console.print(_render_todo_panel(state.todo_items))
        console.print()

    if state.response_text:
        # Strip trailing standalone "..." lines
        clean_response = state.response_text.strip()
        while clean_response.endswith("\n...") or clean_response.rstrip() == "...":
            clean_response = clean_response.rstrip().removesuffix("...").rstrip()
        console.print()
        console.print(Markdown(clean_response or state.response_text))


# ---------------------------------------------------------------------------
# Async-to-sync bridge
# ---------------------------------------------------------------------------

def _create_event_loop() -> asyncio.AbstractEventLoop:
    """Create and set the event loop for asyncio.

    Returns:
        The created event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get the event loop for asyncio.

    If no event loop is set, a new one is created.

    Returns:
        The current event loop.
    """
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = _create_event_loop()
    return loop

def _run_streaming(
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool,
    interactive: bool,
    on_thinking: Callable[[str], None] | None = None,
    on_todo: Callable[[list[dict]], None] | None = None,
    on_file_write: Callable[[str], None] | None = None,
    metadata: dict | None = None,
) -> str:
    """Run async streaming and render with Rich Live display.

    Bridges the async stream_agent_events() into synchronous Rich Live rendering.

    Args:
        agent: Compiled agent graph
        message: User message
        thread_id: Thread ID
        show_thinking: Whether to show thinking panel
        interactive: If True, use simplified final display (no panel)
        on_thinking: Optional sync callback receiving full thinking text.
            Called once when thinking phase ends (transitions to tool/text)
            and accumulated thinking >= 200 chars.
        on_todo: Optional sync callback receiving todo items list.
            Called once when write_todos tool_call is detected.
        on_file_write: Optional sync callback receiving the real filesystem path
            when the agent writes a media file (image/pdf) via write_file.
        metadata: Optional metadata dict forwarded to ``stream_agent_events``
            for LangGraph checkpoint persistence.

    Returns:
        The final response text.
    """
    import nest_asyncio
    nest_asyncio.apply()

    state = StreamState()
    _thinking_sent = False
    _todo_sent = False
    _media_sent: set[str] = set()  # track sent media paths to avoid duplicates
    _MIN_THINKING_LEN = 200

    async def _consume() -> None:
        nonlocal _thinking_sent, _todo_sent
        async for event in stream_agent_events(agent, message, thread_id, metadata=metadata):
            event_type = state.handle_event(event)

            # Send thinking to channel when transitioning away from thinking
            if (on_thinking and not _thinking_sent
                    and state.thinking_text
                    and event_type != "thinking"
                    and len(state.thinking_text) >= _MIN_THINKING_LEN):
                on_thinking(state.thinking_text.rstrip())
                _thinking_sent = True

            # Send todo list to channel on first write_todos tool_call
            if (on_todo and not _todo_sent
                    and event_type == "tool_call"
                    and event.get("name") == "write_todos"
                    and state.todo_items):
                # Flush thinking before todo if not sent yet
                if (on_thinking and not _thinking_sent
                        and state.thinking_text
                        and len(state.thinking_text) >= _MIN_THINKING_LEN):
                    on_thinking(state.thinking_text.rstrip())
                    _thinking_sent = True
                on_todo(state.todo_items)
                _todo_sent = True

            # Send media file to channel when write_file succeeds
            if (on_file_write
                    and event_type == "tool_result"
                    and event.get("name") == "write_file"
                    and event.get("success")):
                wf_path = ""
                for tc in reversed(state.tool_calls):
                    if tc.get("name") == "write_file":
                        p = tc.get("args", {}).get("path", "")
                        if p and p not in _media_sent:
                            wf_path = p
                            break
                if wf_path:
                    ext = os.path.splitext(wf_path)[1].lower()
                    if ext in _MEDIA_EXTENSIONS:
                        real_path = str(resolve_virtual_path(wf_path))
                        if os.path.isfile(real_path):
                            _media_sent.add(wf_path)
                            on_file_write(real_path)

            # Send media file to channel when read_file returns an image
            if (on_file_write
                    and event_type == "tool_result"
                    and event.get("name") == "read_file"
                    and event.get("success")):
                rf_path = ""
                for tc in reversed(state.tool_calls):
                    if tc.get("name") == "read_file":
                        p = tc.get("args", {}).get("file_path", "") or tc.get("args", {}).get("path", "")
                        if p and p not in _media_sent:
                            rf_path = p
                            break
                if rf_path:
                    ext = os.path.splitext(rf_path)[1].lower()
                    if ext in _MEDIA_EXTENSIONS:
                        real_path = rf_path
                        if not os.path.isfile(real_path):
                            real_path = str(resolve_virtual_path(rf_path))
                        if os.path.isfile(real_path):
                            _media_sent.add(rf_path)
                            on_file_write(real_path)

            live.update(create_streaming_display(
                **state.get_display_args(),
                show_thinking=show_thinking,
            ))

    with Live(console=console, auto_refresh=False, transient=False, vertical_overflow="visible") as live:
        live.update(create_streaming_display(is_waiting=True))
        try:
            loop = _get_event_loop()
        except RuntimeError:
            # No current event loop
            loop = _create_event_loop()

        async def _run_with_refresh() -> None:
            async def _periodic_refresh() -> None:
                try:
                    while True:
                        await asyncio.sleep(0.1)
                        live.refresh()
                except asyncio.CancelledError:
                    pass

            refresh_task = asyncio.ensure_future(_periodic_refresh())
            try:
                await _consume()
            finally:
                refresh_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
                # Render clean final frame before Live exits (no spinners, expanded tools)
                if interactive:
                    final_display = create_streaming_display(
                        **state.get_display_args(),
                        show_thinking=show_thinking,
                        is_final=True,
                        final_show_thinking=False,
                    )
                else:
                    final_display = create_streaming_display(
                        **state.get_display_args(),
                        show_thinking=show_thinking,
                        is_final=True,
                        final_show_thinking=True,
                        final_thinking_max_length=DisplayLimits.THINKING_FINAL,
                    )
                live.update(final_display)
                live.refresh()

        loop.run_until_complete(_run_with_refresh())

    # Flush any remaining thinking that wasn't sent during streaming
    if on_thinking and not _thinking_sent and state.thinking_text:
        if len(state.thinking_text) >= _MIN_THINKING_LEN:
            on_thinking(state.thinking_text.rstrip())

    # Everything (tools, thinking, todos, response) is already on screen
    # from Live's final frame (transient=False). No need to re-print.

    return (state.response_text or "").strip()


# ---------------------------------------------------------------------------
# Thread-safe static streaming (for background channels)
# ---------------------------------------------------------------------------

async def _astream_to_console(
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool = True,
) -> str:
    """Stream agent events to console using static prints (thread-safe, no Live).

    Used by the background iMessage channel to show streaming output in the CLI
    without conflicting with prompt_toolkit's terminal handling in the main thread.

    Rich console.print() is thread-safe (internal lock), unlike Live which is not.

    Args:
        agent: Compiled agent graph
        message: User message
        thread_id: Thread ID for conversation persistence
        show_thinking: Whether to display thinking panel

    Returns:
        The final response text.
    """
    state = StreamState()

    async for event in stream_agent_events(agent, message, thread_id):
        etype = state.handle_event(event)

        # Only show subagent starts as real-time progress.
        # Full results rendered by display_final_results() after streaming.
        if etype == "subagent_start":
            name = event.get("name", "sub-agent")
            # Skip generic "sub-agent" — real name arrives later;
            # static prints can't be overwritten like Live display.
            if name and name != "sub-agent":
                desc = event.get("description", "")
                line = Text()
                line.append("\u25b6 ", style="cyan bold")
                line.append(f"Cooking with {name}", style="cyan bold")
                if desc:
                    short = desc[:50] + "\u2026" if len(desc) > 50 else desc
                    line.append(f" \u2014 {short}", style="dim")
                console.print(line)

    # Final output (streaming layout: tools → Task List → subagents → response)

    # Thinking
    if show_thinking and state.thinking_text:
        dt = state.thinking_text.rstrip()
        if len(dt) > 500:
            dt = dt[:250] + "\n\u2026truncated\u2026\n" + dt[-250:]
        console.print(Panel(Text(dt, style="dim"), title="Thinking", border_style="blue"))

    # 1) Regular (non-task) tools — above Task List
    for i, tc in enumerate(state.tool_calls):
        if tc.get("name", "").lower() == "task":
            continue
        tr = state.tool_results[i] if i < len(state.tool_results) else None
        console.print(_render_tool_call_line(tc, tr))
        if tr and not is_success(tr.get("content", "")):
            for elem in format_tool_result_compact(tr["name"], tr.get("content", "")):
                console.print(elem)

    # 2) Task List panel — middle
    if state.todo_items:
        console.print(_render_todo_panel(state.todo_items))
        console.print()

    # 3) Subagent sections (compact) — below Task List
    for sa in state.subagents:
        if sa.tool_calls or not sa.is_active:
            for elem in _render_subagent_section(sa, compact=True):
                console.print(elem)

    # 4) Response
    if state.response_text:
        clean = state.response_text.strip()
        while clean.endswith("\n...") or clean.rstrip() == "...":
            clean = clean.rstrip().removesuffix("...").rstrip()
        console.print()
        console.print(Markdown(clean or state.response_text))
        console.print()

    return (state.response_text or "").strip()
