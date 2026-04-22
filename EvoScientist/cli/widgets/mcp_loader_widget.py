"""Live-updating widget that shows per-server MCP load progress.

Mounted above the chat input while MCP tools are being fetched in the
background.  Re-renders on a 100 ms tick so the spinner animates and the
per-server states transition smoothly from pending → ok/error.

When the load finishes:
- All-success runs auto-dismiss after a short grace period so the chat
  area isn't permanently crowded.
- Failures stick around longer so the user has time to read the error
  detail, then auto-dismiss — otherwise the widget pins itself above
  the input forever.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Static

from ..status_bar import SPINNER_FRAMES

_DIM = "#7c8594"
_STRONG = "#e5e7eb"
_GOOD = "#5fcf8b"
_WARN = "#d7b45a"
_BAD = "#d86f6f"

# How long to wait after an all-success load before auto-dismissing.
_AUTO_DISMISS_SECONDS = 2.5
# Longer grace on failure so the user has time to read error detail.
_AUTO_DISMISS_ON_ERROR_SECONDS = 12.0


class MCPLoaderWidget(Static):
    """Shows a header line + one line per MCP server with its live status."""

    DEFAULT_CSS = """
    MCPLoaderWidget {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    TICK_SECONDS = 0.1

    def __init__(self, servers: list[str]) -> None:
        # server_name -> (state, detail); state ∈ {"pending","ok","error"}.
        self._progress: dict[str, tuple[str, str]] = dict.fromkeys(
            servers, ("pending", "")
        )
        self._frame = 0
        self._tick_handle = None
        self._finished = False
        self._dismissed = False
        self._auto_dismiss_at: float | None = None
        # Seed with real content so Textual can measure us before the
        # first tick; ``self.update()`` during ``__init__`` is unsafe
        # (widget isn't attached yet), but we can pass the renderable
        # straight into ``Static.__init__``.
        super().__init__(self._build_renderable())

    def on_mount(self) -> None:
        self._tick_handle = self.set_interval(self.TICK_SECONDS, self._tick)

    def on_unmount(self) -> None:
        if self._tick_handle is not None:
            self._tick_handle.stop()
            self._tick_handle = None

    # ── Public API ───────────────────────────────────────────────────

    @property
    def dismissed(self) -> bool:
        """Whether the widget has already removed itself from the DOM."""
        return self._dismissed

    def update_server(self, name: str, state: str, detail: str = "") -> None:
        """Record a progress event for one server and re-render."""
        if self._dismissed or state not in ("pending", "ok", "error"):
            return
        # First-time-seen servers (e.g., ones missing from the initial
        # prime set because the config file changed mid-load) just get
        # appended — order stays stable for already-known entries.
        self._progress[name] = (state, detail)
        self._refresh_content()

    def mark_finished(self) -> None:
        """Call once the background load task resolves (success or error).

        If nothing ever progressed past ``pending``, the load was served
        from cache (no events emitted) — drop the widget immediately
        instead of flashing a misleading "0/N loaded" header.

        Otherwise schedule an auto-dismiss: short on full success so the
        chat area isn't cluttered, longer on failure so the user has
        time to read the error detail before it goes away.
        """
        if self._finished:
            return
        self._finished = True
        progressed = any(state != "pending" for state, _ in self._progress.values())
        if not progressed:
            self._dismiss()
            return
        has_errors = any(state == "error" for state, _ in self._progress.values())
        delay = _AUTO_DISMISS_ON_ERROR_SECONDS if has_errors else _AUTO_DISMISS_SECONDS
        self._auto_dismiss_at = time.monotonic() + delay
        self._refresh_content()

    # ── Internal ─────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
        if (
            self._auto_dismiss_at is not None
            and time.monotonic() >= self._auto_dismiss_at
        ):
            self._auto_dismiss_at = None
            self._dismiss()
            return
        if not self._finished:
            self._refresh_content()

    def _dismiss(self) -> None:
        """Stop the tick timer and detach from the DOM.

        Sets :attr:`dismissed` so the app can clear its widget reference
        and late progress events become no-ops.
        """
        if self._dismissed:
            return
        self._dismissed = True
        if self._tick_handle is not None:
            self._tick_handle.stop()
            self._tick_handle = None
        # Fire-and-forget remove — nothing awaits us.
        self.remove()

    def _build_renderable(self) -> Text:
        spinner = SPINNER_FRAMES[self._frame]
        pending = sum(1 for state, _ in self._progress.values() if state == "pending")
        total = len(self._progress)
        done = total - pending

        header = Text()
        if self._finished:
            errors = sum(1 for state, _ in self._progress.values() if state == "error")
            if errors:
                header.append("✗ MCP ", style=f"{_BAD} bold")
                header.append(
                    f"{done - errors}/{total} loaded, {errors} failed",
                    style=_STRONG,
                )
            else:
                header.append("✓ MCP ", style=f"{_GOOD} bold")
                header.append(f"{done}/{total} servers loaded", style=_STRONG)
        else:
            header.append(f"{spinner} ", style=f"{_WARN} bold")
            header.append("Loading MCP tools ", style=_STRONG)
            header.append(f"{done}/{total}", style=_DIM)

        lines: list[Text] = [header]
        for name, (state, detail) in self._progress.items():
            line = Text("  ")
            if state == "pending":
                line.append(f"{spinner} ", style=_WARN)
                line.append(name, style=_DIM)
            elif state == "ok":
                line.append("✓ ", style=_GOOD)
                line.append(name, style=_STRONG)
                if detail:
                    line.append(f"  {detail} tools", style=_DIM)
            else:  # error
                line.append("✗ ", style=_BAD)
                line.append(name, style=_STRONG)
                if detail:
                    summary = detail if len(detail) <= 80 else detail[:77] + "…"
                    line.append(f"  {summary}", style=_BAD)
            lines.append(line)

        return Text("\n").join(lines)

    def _refresh_content(self) -> None:
        # NB: don't name this ``_render`` — that shadows Textual's internal
        # ``Widget._render`` which must return a ``Visual``.  Silently
        # breaking that contract triggers ``'NoneType' object has no
        # attribute 'get_height'`` during layout.
        self.update(self._build_renderable())
