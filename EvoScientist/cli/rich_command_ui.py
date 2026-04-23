"""CommandUI Protocol adapter for the Rich CLI surface.

Methods not exercised by the currently-migrated commands raise
``NotImplementedError`` rather than silently returning ``None``, so
future callers fail loudly instead of pretending the command ran.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
        on_start_new_session: Callable[[], None] | None = None,
        on_handle_session_resume: (
            Callable[[str, str | None], Awaitable[None]] | None
        ) = None,
    ) -> None:
        self.console = console
        self._on_request_quit = on_request_quit
        self._on_force_quit = on_force_quit
        self._on_clear_chat = on_clear_chat
        self._on_status_after_compact = on_status_after_compact
        self._on_start_new_session = on_start_new_session
        self._on_handle_session_resume = on_handle_session_resume
        # Bound ``console.status(...)`` context manager used by
        # /compact's start/stop indicator pair.
        self._compact_status_ctx: Any = None

    # ── Core I/O ─────────────────────────────────────────────

    @property
    def supports_interactive(self) -> bool:
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

    # ── Phase B: workspace-grouped questionary picker ──

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        """Interactive workspace-grouped thread picker using ``questionary``.

        Ported from the pre-migration ``_cmd_resume`` implementation.
        Returns the selected ``thread_id`` string, or ``None`` on cancel.
        Callers (``ResumeCommand``/``DeleteCommand``) pre-check for
        empty thread lists before invoking this method.
        """
        import questionary  # type: ignore[import-untyped]
        from prompt_toolkit.layout.dimension import (  # type: ignore[import-untyped]
            Dimension,
        )
        from questionary.prompts.common import (  # type: ignore[import-untyped]
            InquirerControl,
        )

        from ..sessions import _format_relative_time
        from .widgets.thread_selector import PICKER_STYLE, _build_items

        choices: list[Any] = []
        for item in _build_items(threads):
            if item["type"] == "header":
                choices.append(questionary.Separator(f"── \U0001f4c2 {item['label']}"))
            elif item["type"] == "subheader":
                choices.append(questionary.Separator(f"   {item['label']}"))
            else:
                t = item["thread"]
                tid = t["thread_id"]
                preview = t.get("preview", "") or ""
                msgs = t.get("message_count", 0)
                model = t.get("model", "") or ""
                when = _format_relative_time(t.get("updated_at"))
                indent = "    " if item.get("indented") else "  "
                marker = " *" if tid == current_thread else ""
                parts = [f"{indent}{tid}{marker}"]
                if preview:
                    parts.append(preview[:40] + "…" if len(preview) > 40 else preview)
                parts.append(f"({msgs} msgs)")
                if model:
                    parts.append(model)
                if when:
                    parts.append(when)
                label = "  ".join(parts)
                choices.append(questionary.Choice(title=label, value=tid))

        prompt = questionary.select(title, choices=choices, style=PICKER_STYLE)
        # Limit visible list to 10 rows with scrolling.
        for window in prompt.application.layout.find_all_windows():
            if isinstance(window.content, InquirerControl):
                window.height = Dimension(max=10)
                break
        return prompt.ask()

    # ── Phase A/B callback-backed methods ─────────────────

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

    def start_new_session(self) -> None:
        if self._on_start_new_session is not None:
            self._on_start_new_session()

    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None:
        if self._on_handle_session_resume is not None:
            await self._on_handle_session_resume(thread_id, workspace_dir)

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

    # ── Phase C: skill / MCP browse pickers ──────────────

    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None:
        """Delegate to the extracted questionary picker on a worker
        thread — questionary blocks the event loop so the call must
        not happen on the main asyncio thread."""
        import asyncio

        from .skills_cmd import _pick_skills_interactive

        return await asyncio.to_thread(
            _pick_skills_interactive, index, installed_names, pre_filter_tag
        )

    async def wait_for_mcp_browse(
        self, servers: list, installed_names: set[str], pre_filter_tag: str
    ) -> list | None:
        """Delegate to the MCP browse picker on a worker thread."""
        import asyncio

        from .mcp_install_cmd import _browse_and_select

        return await asyncio.to_thread(
            _browse_and_select, servers, installed_names, pre_filter_tag
        )
