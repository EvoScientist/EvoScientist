"""Sub-agent widget — bordered area with nested tool calls."""

from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static

from .tool_call_widget import ToolCallWidget

_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

# Keep in sync with display.py MAX_SA_VISIBLE / MAX_SA_RUNNING
_MAX_VISIBLE_COMPLETED = 3
_MAX_VISIBLE_RUNNING = 2


class SubAgentWidget(Vertical):
    """Displays a sub-agent's activity with a bordered frame.

    Active state::

        ┌ ▶ Cooking with research-agent — Search literature ─┐
        │   ✓ 8 completed                                     │
        │   ● tavily_search  query="LLM attention"           │
        │     ✓ 3 results                                     │
        └─────────────────────────────────────────────────────┘

    Completed state::

        ✓ Cooking with research-agent  (3 tools)
    """

    DEFAULT_CSS = """
    SubAgentWidget {
        height: auto;
        margin: 0 0;
    }
    SubAgentWidget .sa-header {
        height: auto;
        color: #22d3ee;
    }
    SubAgentWidget .sa-tools {
        height: auto;
        padding: 0 0 0 2;
    }
    SubAgentWidget .sa-collapse-summary {
        height: auto;
        padding: 0 0 0 2;
        display: none;
    }
    SubAgentWidget .sa-collapse-summary.--visible {
        display: block;
    }
    SubAgentWidget .sa-footer {
        height: auto;
        color: #22d3ee;
    }
    SubAgentWidget.--completed .sa-header {
        color: #4ade80;
    }
    SubAgentWidget.--completed .sa-footer {
        color: #4ade80;
    }
    """

    def __init__(self, name: str, description: str = "") -> None:
        super().__init__()
        self._sa_name = name
        self._description = description
        self._is_active = True
        self._frame = 0
        self._tool_count = 0
        self._timer_handle = None
        self._tool_widgets: dict[str, ToolCallWidget] = {}
        # Ordered lists to track completed / running tools for collapsing
        self._completed_ids: list[str] = []
        self._running_ids: list[str] = []

    @property
    def sa_name(self) -> str:
        return self._sa_name

    def update_name(self, name: str, description: str = "") -> None:
        """Update the sub-agent display name after resolution."""
        self._sa_name = name
        if description:
            self._description = description
        try:
            self._render_header()
        except Exception:
            pass  # Widget may not be mounted yet

    def compose(self):
        yield Static("", classes="sa-header")
        yield Static("", classes="sa-collapse-summary")
        yield Vertical(classes="sa-tools")
        yield Static("", classes="sa-footer")

    def on_mount(self) -> None:
        self._timer_handle = self.set_interval(0.1, self._tick)
        self._render_header()
        self._render_footer()

    def _tick(self) -> None:
        if self._is_active:
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self._render_header()

    def _display_name(self) -> str:
        name = f"Cooking with {self._sa_name}"
        if self._description:
            desc = self._description.split("\n")[0].strip()
            if len(desc) > 50:
                desc = desc[:47] + "\u2026"
            name += f" \u2014 {desc}"
        return name

    def _render_header(self) -> None:
        header = self.query_one(".sa-header", Static)
        line = Text()
        if self._is_active:
            char = _SPINNER_FRAMES[self._frame]
            line.append(
                f"\u250c \u25b6 {self._display_name()} {char}", style="bold cyan"
            )
        else:
            line.append(f"\u2713 {self._display_name()}", style="bold green")
            line.append(f"  ({self._tool_count} tools)", style="dim")
        header.update(line)

    def _render_footer(self) -> None:
        footer = self.query_one(".sa-footer", Static)
        if self._is_active:
            footer.update(Text("\u2514 running...", style="dim cyan"))
        else:
            footer.update(Text(""))

    async def add_tool_call(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        tool_id: str = "",
    ) -> ToolCallWidget:
        """Mount a new ToolCallWidget inside this sub-agent.

        If a widget with the same *tool_id* already exists (re-emitted with
        updated args during incremental streaming), update it in place instead
        of creating a duplicate.
        """
        if tool_id and tool_id in self._tool_widgets:
            # Re-emitted with updated args — update in place
            existing = self._tool_widgets[tool_id]
            existing._tool_name = tool_name
            existing._tool_args = tool_args or {}
            try:
                existing._render_header()
            except Exception:
                pass  # Widget may not be mounted yet
            return existing

        self._tool_count += 1
        w = ToolCallWidget(tool_name, tool_args, tool_id)
        tools_container = self.query_one(".sa-tools", Vertical)
        await tools_container.mount(w)
        key = tool_id or f"_anon_{self._tool_count}"
        self._tool_widgets[key] = w
        self._running_ids.append(key)
        self._update_visibility()
        return w

    def complete_tool(
        self,
        tool_name: str,
        content: str,
        success: bool = True,
        tool_id: str = "",
    ) -> None:
        """Update the matching ToolCallWidget with its result."""
        widget = None
        matched_key = ""
        if tool_id and tool_id in self._tool_widgets:
            widget = self._tool_widgets[tool_id]
            matched_key = tool_id
        else:
            # Match by name — find first running tool with this name
            for key, w in self._tool_widgets.items():
                if w.tool_name == tool_name and w._status == "running":
                    widget = w
                    matched_key = key
                    break
            if widget is None:
                # Fallback: find any running tool
                tools = self.query_one(".sa-tools", Vertical)
                for child in tools.children:
                    if isinstance(child, ToolCallWidget) and child._status == "running":
                        if child.tool_name == tool_name:
                            widget = child
                            # Find the key for this widget
                            for key, w in self._tool_widgets.items():
                                if w is widget:
                                    matched_key = key
                                    break
                            break

        if widget is not None:
            if success:
                widget.set_success(content)
            else:
                widget.set_error(content)
            # Move from running to completed (dedup guards against repeat
            # deliveries of the same tool result inflating the collapse summary).
            if matched_key and matched_key in self._running_ids:
                self._running_ids.remove(matched_key)
            if matched_key and matched_key not in self._completed_ids:
                self._completed_ids.append(matched_key)
            self._update_visibility()

    def _update_visibility(self) -> None:
        """Hide older completed tools, keep recent ones visible.

        Mirrors the Rich display.py collapsing logic:
        - At most ``_MAX_VISIBLE_COMPLETED`` completed tools shown
          (fewer if running tools take up slots).
        - At most ``_MAX_VISIBLE_RUNNING`` running tools shown.
        """
        # Determine how many completed slots are available
        running_visible = self._running_ids[-_MAX_VISIBLE_RUNNING:]
        completed_slots = max(0, _MAX_VISIBLE_COMPLETED - len(running_visible))
        completed_visible = (
            self._completed_ids[-completed_slots:] if completed_slots else []
        )
        completed_hidden = (
            self._completed_ids[:-completed_slots]
            if completed_slots and len(self._completed_ids) > completed_slots
            else (self._completed_ids if not completed_slots else [])
        )

        # Running tools to hide
        running_hidden = (
            self._running_ids[:-_MAX_VISIBLE_RUNNING]
            if len(self._running_ids) > _MAX_VISIBLE_RUNNING
            else []
        )

        # Apply visibility
        visible_keys = set(completed_visible) | set(running_visible)
        hidden_keys = set(completed_hidden) | set(running_hidden)

        for key in visible_keys:
            w = self._tool_widgets.get(key)
            if w is not None:
                w.display = True

        for key in hidden_keys:
            w = self._tool_widgets.get(key)
            if w is not None:
                w.display = False

        # Update collapse summary
        total_hidden = len(completed_hidden)
        hidden_running_count = len(running_hidden)

        summary_w = self.query_one(".sa-collapse-summary", Static)
        if total_hidden > 0 or hidden_running_count > 0:
            line = Text()
            if total_hidden > 0:
                # Count successes/failures among hidden completed
                ok = 0
                fail = 0
                for key in completed_hidden:
                    w = self._tool_widgets.get(key)
                    if w is not None:
                        if w._status == "error":
                            fail += 1
                        else:
                            ok += 1
                line.append(f"\u2713 {ok} completed", style="dim green")
                if fail > 0:
                    line.append(f" | {fail} failed", style="dim red")
            if hidden_running_count > 0:
                if total_hidden > 0:
                    line.append(" | ", style="dim")
                line.append(
                    f"\u25cf {hidden_running_count} more running...", style="dim yellow"
                )
            summary_w.update(line)
            summary_w.add_class("--visible")
        else:
            summary_w.remove_class("--visible")

    def finalize(self) -> None:
        """Mark sub-agent as completed and stop all nested timers."""
        self._is_active = False
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None
        # Mark any nested ToolCallWidgets still running as interrupted
        for tw in self._tool_widgets.values():
            if tw._status == "running":
                try:
                    tw.set_interrupted()
                except Exception:
                    pass
        self.add_class("--completed")
        self._render_header()
        self._render_footer()
