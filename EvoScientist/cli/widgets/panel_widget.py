"""Panel widget — in-eval ``task()`` fan-out live view.

Groups all sub-agent dispatches from a single ``code_interpreter`` eval into
one bordered container, one row per dispatch. Each row shows the expert /
subagent type, the label, a running elapsed timer, and a status dot that
flips to ``ok``/``err`` on completion.

Sourced from the ``custom`` stream events emitted by ``langchain_quickjs``
(see ``.stream.emitter.panel_dispatch_start`` etc.). Keyed by ``eval_id``
so parallel dispatches from the same eval appear stacked; distinct evals
get distinct panels.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

from ..status_bar import SPINNER_FRAMES

_ROW_LABEL_MAX_CHARS = 48


class _DispatchRow(Widget):
    """One row inside a PanelWidget — a single ``task()`` dispatch.

    Subclasses ``Widget`` directly and overrides ``render()`` to build the
    row's ``Text`` on demand. Earlier revisions subclassed ``Static`` (visual
    stayed ``None`` past the first paint) and ``Vertical`` with an inner
    ``Static`` (container-layout race); rendering from ``render()`` is the
    standard pattern for single-line widgets and dodges both issues by
    letting Textual manage the visual lifecycle itself.
    """

    DEFAULT_CSS = """
    _DispatchRow {
        height: 1;
        width: 100%;
    }
    """

    def __init__(self, subagent_type: str, label: str) -> None:
        super().__init__()
        self._subagent_type = subagent_type
        self._label = label
        self._started_at = time.monotonic()
        self._status: str = "running"  # "running" | "ok" | "err"
        self._duration_ms: int | None = None
        self._error: str = ""
        self._frame = 0

    def render(self) -> Text:
        line = Text()
        if self._status == "running":
            line.append(f"  {SPINNER_FRAMES[self._frame]} ", style="cyan")
        elif self._status == "ok":
            line.append("  \u2713 ", style="green")
        else:
            line.append("  \u2717 ", style="red")
        line.append(f"{self._subagent_type} ", style="bold")
        if self._label:
            trimmed = self._label
            if len(trimmed) > _ROW_LABEL_MAX_CHARS:
                trimmed = trimmed[: _ROW_LABEL_MAX_CHARS - 1] + "\u2026"
            line.append(f"\u2014 {trimmed} ", style="dim")
        line.append(self._elapsed_display(), style="dim")
        if self._status == "err" and self._error:
            err = self._error.split("\n", 1)[0]
            if len(err) > 60:
                err = err[:59] + "\u2026"
            line.append(f"  {err}", style="red")
        return line

    def tick(self) -> None:
        if self._status == "running":
            self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
        self.refresh()

    def complete(self, duration_ms: int) -> None:
        self._status = "ok"
        self._duration_ms = duration_ms
        self.refresh()

    def fail(self, duration_ms: int, error: str) -> None:
        self._status = "err"
        self._duration_ms = duration_ms
        self._error = error
        self.refresh()

    def _elapsed_display(self) -> str:
        if self._duration_ms is not None:
            secs = self._duration_ms / 1000.0
        else:
            secs = time.monotonic() - self._started_at
        return f"{secs:5.1f}s"


class PanelWidget(Vertical):
    """Container for one eval's fan-out — bordered box, one row per dispatch."""

    DEFAULT_CSS = """
    PanelWidget {
        height: auto;
        margin: 0 0;
    }
    PanelWidget .panel-header {
        height: auto;
        color: #22d3ee;
    }
    PanelWidget .panel-rows {
        height: auto;
        padding: 0 0 0 2;
    }
    PanelWidget .panel-footer {
        height: auto;
        color: #22d3ee;
    }
    PanelWidget.--completed .panel-header {
        color: #4ade80;
    }
    PanelWidget.--completed .panel-footer {
        color: #4ade80;
    }
    """

    def __init__(self, eval_id: str) -> None:
        super().__init__()
        self._eval_id = eval_id
        self._rows: dict[str, _DispatchRow] = {}
        self._timer_handle = None
        self._is_active = True

    @property
    def eval_id(self) -> str:
        return self._eval_id

    @property
    def dispatch_count(self) -> int:
        return len(self._rows)

    def compose(self):
        yield Static("", classes="panel-header")
        yield Vertical(classes="panel-rows")
        yield Static("", classes="panel-footer")

    def on_mount(self) -> None:
        self._timer_handle = self.set_interval(0.1, self._tick)
        self._render_header()
        self._render_footer()

    def _tick(self) -> None:
        for row in self._rows.values():
            row.tick()
        self._render_header()

    async def start_dispatch(
        self, dispatch_id: str, subagent_type: str, label: str
    ) -> None:
        if dispatch_id in self._rows:
            return
        # Re-arm if the panel already finalized: a Promise.allSettled retry
        # of a failed subset dispatches under the same eval_id, so a new
        # start after _maybe_finalize() has stopped the timer must undo
        # the three effects of finalize (latch, class, timer) or the new
        # row's spinner/elapsed stay frozen and future completions never
        # refresh the header.
        if not self._is_active:
            self._is_active = True
            self.remove_class("--completed")
            self._timer_handle = self.set_interval(0.1, self._tick)
            self._render_footer()
        row = _DispatchRow(subagent_type, label)
        rows_container = self.query_one(".panel-rows", Vertical)
        await rows_container.mount(row)
        self._rows[dispatch_id] = row
        self._render_header()

    def complete_dispatch(self, dispatch_id: str, duration_ms: int) -> None:
        row = self._rows.get(dispatch_id)
        if row is not None:
            row.complete(duration_ms)
        self._maybe_finalize()

    def fail_dispatch(self, dispatch_id: str, duration_ms: int, error: str) -> None:
        row = self._rows.get(dispatch_id)
        if row is not None:
            row.fail(duration_ms, error)
        self._maybe_finalize()

    def finalize_running(self, reason: str = "interrupted") -> None:
        """Fail all still-running rows with their measured elapsed time.

        Called from the TUI turn-cleanup ``finally`` so the 100ms interval
        timer stops when a turn is cancelled mid-dispatch — without this,
        no reference remains to stop the timer once the outer scope exits.
        """
        if not self._is_active:
            return
        now = time.monotonic()
        for dispatch_id, row in list(self._rows.items()):
            if row._status == "running":
                elapsed_ms = int((now - row._started_at) * 1000)
                self.fail_dispatch(dispatch_id, elapsed_ms, reason)
        # Cover the zero-running-rows case: cancel-during-mount can leave
        # the timer armed with either no rows registered (first dispatch)
        # or every registered row already terminal (allSettled retry).
        # The loop skips both, so call _maybe_finalize unconditionally —
        # it is a no-op once _is_active has flipped.
        self._maybe_finalize()

    def _maybe_finalize(self) -> None:
        if not self._is_active:
            return
        if all(row._status != "running" for row in self._rows.values()):
            self._is_active = False
            if self._timer_handle is not None:
                self._timer_handle.stop()
                self._timer_handle = None
            self.add_class("--completed")
            self._render_header()
            self._render_footer()

    def _summary_counts(self) -> tuple[int, int, int]:
        running = ok = err = 0
        for row in self._rows.values():
            if row._status == "running":
                running += 1
            elif row._status == "ok":
                ok += 1
            else:
                err += 1
        return running, ok, err

    def _render_header(self) -> None:
        header = self.query_one(".panel-header", Static)
        running, ok, err = self._summary_counts()
        line = Text()
        if self._is_active:
            line.append("\u250c \u25b6 Expert panel ", style="bold cyan")
            line.append(
                f"({running} running, {ok} done, {err} failed)", style="dim cyan"
            )
        else:
            line.append("\u2713 Expert panel ", style="bold green")
            line.append(f"({ok} done, {err} failed)", style="dim green")
        header.update(line)

    def _render_footer(self) -> None:
        footer = self.query_one(".panel-footer", Static)
        if self._is_active:
            footer.update(Text("\u2514 running...", style="dim cyan"))
        else:
            footer.update(Text(""))
