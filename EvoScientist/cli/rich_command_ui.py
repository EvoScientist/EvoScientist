"""CommandUI Protocol adapter for the Rich CLI surface.

Methods not exercised by the currently-migrated commands raise
``NotImplementedError`` rather than silently returning ``None``, so
future callers fail loudly instead of pretending the command ran.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.table import Table

from ..commands.base import CommandUI


class RichCLICommandUI(CommandUI):
    """CommandUI implementation that prints to a Rich ``Console``.

    Commands that affect CLI-closure state (session lifecycle, exit flag,
    status-bar snapshot) go through optional callbacks wired by the REPL.
    This mirrors ``ChannelCommandUI``'s injection pattern and keeps
    ``interactive.py``'s ``state`` dict as the single source of truth.
    """

    def __init__(
        self,
        console: Console,
        *,
        on_request_quit: Callable[[], None] | None = None,
        on_force_quit: Callable[[], None] | None = None,
        on_clear_chat: Callable[[], None] | None = None,
        on_status_after_compact: Callable[[int], None] | None = None,
    ) -> None:
        self.console = console
        self._on_request_quit = on_request_quit
        self._on_force_quit = on_force_quit
        self._on_clear_chat = on_clear_chat
        self._on_status_after_compact = on_status_after_compact
        # Bound ``console.status(...)`` context manager used by
        # /compact's start/stop indicator pair.
        self._compact_status_ctx: Any = None

    # ── Core I/O ─────────────────────────────────────────────

    @property
    def supports_interactive(self) -> bool:
        # Rich CLI has no picker widget, but wait_for_* fall back to
        # printing a table and returning None (see wait_for_model_pick).
        return True

    def append_system(self, text: str, style: str = "dim") -> None:
        self.console.print(text, style=style)

    def mount_renderable(self, renderable: Any) -> None:
        self.console.print(renderable)

    async def flush(self) -> None:
        # Rich console flushes synchronously; nothing to await.
        return

    # ── /model interactive picker fallback ──────────────────

    async def wait_for_model_pick(
        self,
        entries: list[tuple[str, str, str]],
        current_model: str | None,
        current_provider: str | None,
    ) -> tuple[str, str] | None:
        """Print the model table and return ``None``; user re-runs with
        ``/model <name>`` since the CLI has no interactive picker."""
        table = Table(
            title="Available Models",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Name", style="bold")
        table.add_column("Provider", style="dim")
        for name, _mid, prov in entries:
            marker = " *" if name == current_model and prov == current_provider else ""
            table.add_row(f"{name}{marker}", prov)
        self.console.print(table)
        self.console.print(
            "[dim]Usage: /model <name> [provider] [--save]  — "
            "provider is optional, auto-detected from model name[/dim]"
        )
        return None

    def update_status_after_model_change(
        self, new_model: str, new_provider: str | None = None
    ) -> None:
        """No-op; the CLI REPL refreshes status itself after detecting an
        ``ctx.agent`` change post-``cmd_manager.execute``."""
        return

    # ── Phase A: thread picker fallback, quit, clear, compact ──

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        """Render a thread table and return ``None``; user re-runs with
        ``/resume <id>`` or ``/delete <id>``. Phase B will upgrade this to
        use ``questionary.select`` (see ``_cmd_resume``)."""
        from ..sessions import _format_relative_time

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("ID", style="bold")
        table.add_column("Preview", style="dim", max_width=50, no_wrap=True)
        table.add_column("Msgs", justify="right")
        table.add_column("Model", style="dim")
        table.add_column("Last Used", style="dim")
        for t in threads:
            tid = t["thread_id"]
            marker = " *" if tid == current_thread else ""
            table.add_row(
                f"{tid}{marker}",
                t.get("preview", "") or "",
                str(t.get("message_count", 0)),
                t.get("model", "") or "",
                _format_relative_time(t.get("updated_at")),
            )
        self.console.print(table)
        self.console.print(
            "[dim]Usage: /resume <id> or /delete <id> "
            "(prefix match supported)[/dim]"
        )
        return None

    def clear_chat(self) -> None:
        if self._on_clear_chat is not None:
            self._on_clear_chat()
        else:
            self.console.clear()

    def request_quit(self) -> None:
        if self._on_request_quit is not None:
            self._on_request_quit()

    def force_quit(self) -> None:
        if self._on_force_quit is not None:
            self._on_force_quit()

    # /compact indicator pair — duck-typed by ``CompactCommand`` via
    # ``getattr``, not declared on the ``CommandUI`` Protocol.
    def start_compacting_indicator(self) -> None:
        status = self.console.status("[cyan]Compacting conversation...[/cyan]")
        status.__enter__()
        self._compact_status_ctx = status

    def stop_compacting_indicator(self) -> None:
        if self._compact_status_ctx is not None:
            self._compact_status_ctx.__exit__(None, None, None)
            self._compact_status_ctx = None

    def update_status_after_compact(self, input_tokens: int) -> None:
        if self._on_status_after_compact is not None:
            self._on_status_after_compact(input_tokens)

    # ── Not yet migrated (Phase B/C) ────────────────────────

    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None:
        raise NotImplementedError(
            "RichCLICommandUI.wait_for_skill_browse — implement when "
            "migrating /evoskills"
        )

    async def wait_for_mcp_browse(
        self, servers: list, installed_names: set[str], pre_filter_tag: str
    ) -> list | None:
        raise NotImplementedError(
            "RichCLICommandUI.wait_for_mcp_browse — implement when "
            "migrating /install-mcp"
        )

    def start_new_session(self) -> None:
        raise NotImplementedError(
            "RichCLICommandUI.start_new_session — implement when migrating /new"
        )

    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None:
        raise NotImplementedError(
            "RichCLICommandUI.handle_session_resume — implement when migrating /resume"
        )
