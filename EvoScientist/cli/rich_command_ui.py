"""CommandUI Protocol adapter for the Rich CLI surface.

Methods not exercised by the currently-migrated commands raise
``NotImplementedError`` rather than silently returning ``None``, so
future callers fail loudly instead of pretending the command ran.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from ..commands.base import CommandUI


class RichCLICommandUI(CommandUI):
    """CommandUI implementation that prints to a Rich ``Console``."""

    def __init__(self, console: Console) -> None:
        self.console = console

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

    # ── Not yet migrated ────────────────────────────────────

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        raise NotImplementedError(
            "RichCLICommandUI.wait_for_thread_pick — implement when "
            "migrating /threads / /resume"
        )

    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None:
        raise NotImplementedError(
            "RichCLICommandUI.wait_for_skill_browse — implement when "
            "migrating /skills / /evoskills"
        )

    async def wait_for_mcp_browse(
        self, servers: list, installed_names: set[str], pre_filter_tag: str
    ) -> list | None:
        raise NotImplementedError(
            "RichCLICommandUI.wait_for_mcp_browse — implement when migrating /mcp"
        )

    def clear_chat(self) -> None:
        raise NotImplementedError(
            "RichCLICommandUI.clear_chat — implement when migrating /clear"
        )

    def request_quit(self) -> None:
        raise NotImplementedError(
            "RichCLICommandUI.request_quit — implement when migrating /exit"
        )

    def force_quit(self) -> None:
        raise NotImplementedError(
            "RichCLICommandUI.force_quit — implement when migrating /exit"
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
