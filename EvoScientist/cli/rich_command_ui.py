"""CommandUI Protocol adapter for the Rich CLI surface.

Implements all CommandUI methods using Rich console output and questionary
interactive pickers for the plain CLI (non-TUI) mode.

Lifecycle commands (/new, /exit, /resume) set signal flags that the CLI
dispatch loop reads after ``cmd_manager.execute()`` returns, keeping state
management centralised in the main loop.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import questionary
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style as PtStyle
from questionary import Choice
from rich.console import Console
from rich.table import Table

from ..commands.base import CommandUI
from ..sessions import _format_relative_time

_PICKER_STYLE = PtStyle.from_dict(
    {
        "questionmark": "#888888",
        "question": "",
        "pointer": "bold",
        "highlighted": "bold",
        "text": "#888888",
        "answer": "bold",
    }
)

_INSTALLED_INDICATOR = ("fg:#4caf50", "\u2713 ")


def _checkbox_ask(choices: list, message: str, **kwargs):
    """Wrap ``questionary.checkbox`` so disabled items render with a checkmark.

    Args:
        choices: List of questionary Choice objects.
        message: Prompt message shown above the checkbox list.
        **kwargs: Forwarded to ``questionary.checkbox``.

    Returns:
        List of selected values, or ``None`` if the user cancels.
    """
    from questionary.prompts.common import InquirerControl

    original = InquirerControl._get_choice_tokens

    def _patched(self):
        tokens = original(self)
        return [
            _INSTALLED_INDICATOR
            if cls == "class:disabled" and text == "- "
            else (cls, text)
            for cls, text in tokens
        ]

    InquirerControl._get_choice_tokens = _patched
    try:
        return questionary.checkbox(
            message,
            choices=choices,
            style=_PICKER_STYLE,
            qmark="\u276f",
            **kwargs,
        ).ask()
    finally:
        InquirerControl._get_choice_tokens = original


class RichCLICommandUI(CommandUI):
    """CommandUI implementation that prints to a Rich ``Console``.

    Lifecycle commands set signal flags (``_quit_requested``,
    ``_new_session_requested``, ``_resume_request``, ``_compact_tokens``)
    instead of mutating session state directly.  The CLI dispatch loop
    reads and resets these after each ``cmd_manager.execute()`` call.

    Args:
        console: Rich Console instance used for all output.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._quit_requested: bool = False
        self._new_session_requested: bool = False
        self._resume_request: tuple[str, str | None] | None = None
        self._compact_tokens: int | None = None
        self._status_ctx: Any = None

    @property
    def supports_interactive(self) -> bool:
        return True

    def append_system(self, text: str, style: str = "dim") -> None:
        self.console.print(text, style=style)

    def mount_renderable(self, renderable: Any) -> None:
        self.console.print(renderable)

    async def flush(self) -> None:
        return

    async def wait_for_model_pick(
        self,
        entries: list[tuple[str, str, str]],
        current_model: str | None,
        current_provider: str | None,
    ) -> tuple[str, str] | None:
        """Prints the model table and returns ``None``.

        The CLI has no interactive picker; the user re-runs with
        ``/model <name>`` after consulting the printed table.

        Args:
            entries: ``(name, model_id, provider)`` tuples.
            current_model: Currently active model name, or ``None``.
            current_provider: Currently active provider, or ``None``.

        Returns:
            Always ``None``.
        """
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
            "[dim]Usage: /model <name> [provider] [--save]  \u2014 "
            "provider is optional, auto-detected from model name[/dim]"
        )
        return None

    def update_status_after_model_change(
        self, new_model: str, new_provider: str | None = None
    ) -> None:
        """No-op.  The CLI REPL refreshes status itself via post-execute sync."""
        return

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        """Shows a questionary select picker for session threads.

        Args:
            threads: Thread metadata dicts from ``list_threads()``.
            current_thread: ID of the active thread (marked in display).
            title: Prompt title shown above the picker.

        Returns:
            Selected thread ID, or ``None`` if the user cancels.
        """
        from .widgets.thread_selector import _build_items

        choices: list = []
        items = _build_items(threads)
        for item in items:
            if item["type"] == "header":
                choices.append(
                    questionary.Separator(f"\u2500\u2500 \U0001f4c2 {item['label']}")
                )
            elif item["type"] == "subheader":
                choices.append(questionary.Separator(f"   {item['label']}"))
            else:
                t = item["thread"]
                tid = t["thread_id"]
                preview = t.get("preview", "") or ""
                msgs = t.get("message_count", 0)
                t_model = t.get("model", "") or ""
                when = _format_relative_time(t.get("updated_at"))
                indent = "    " if item.get("indented") else "  "
                parts = [f"{indent}{tid}"]
                if preview:
                    parts.append(
                        preview[:40] + "\u2026" if len(preview) > 40 else preview
                    )
                parts.append(f"({msgs} msgs)")
                if t_model:
                    parts.append(t_model)
                if when:
                    parts.append(when)
                label = "  ".join(parts)
                choices.append(questionary.Choice(title=label, value=tid))

        if not choices:
            self.console.print("[yellow]No sessions available.[/yellow]")
            return None

        from questionary.prompts.common import InquirerControl

        prompt = questionary.select(
            title,
            choices=choices,
            style=_PICKER_STYLE,
        )
        for window in prompt.application.layout.find_all_windows():
            if isinstance(window.content, InquirerControl):
                window.height = Dimension(max=10)
                break
        return prompt.ask()

    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None:
        """Shows a questionary tag-filter + checkbox picker for EvoSkills.

        Args:
            index: Skill metadata dicts from the remote index.
            installed_names: Names of already-installed skills.
            pre_filter_tag: If non-empty, skip the tag picker and filter
                directly by this tag.

        Returns:
            List of ``install_source`` strings, or ``None`` on cancel.
        """
        tag_counter: Counter[str] = Counter()
        for s in index:
            for t in s.get("tags", []):
                tag_counter[t.lower()] += 1

        if pre_filter_tag:
            pre_lower = pre_filter_tag.lower()
            filtered = [
                s for s in index if pre_lower in [t.lower() for t in s.get("tags", [])]
            ]
            if not filtered:
                self.console.print(
                    f"[yellow]No skills found with tag: {pre_filter_tag}[/yellow]"
                )
                if tag_counter:
                    sorted_tags = sorted(
                        tag_counter.items(), key=lambda x: (-x[1], x[0])
                    )
                    tags_str = ", ".join(
                        f"{tag} ({count})" for tag, count in sorted_tags
                    )
                    self.console.print(f"[dim]Available tags: {tags_str}[/dim]")
                return None
        else:
            sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
            tag_choices = [Choice(title=f"All skills ({len(index)})", value="__all__")]
            for tag, count in sorted_tags:
                tag_choices.append(Choice(title=f"{tag} ({count})", value=tag))

            selected_tag = questionary.select(
                "Filter by tag:",
                choices=tag_choices,
                style=_PICKER_STYLE,
                qmark="\u276f",
            ).ask()

            if selected_tag is None:
                return None

            if selected_tag == "__all__":
                filtered = index
            else:
                filtered = [
                    s
                    for s in index
                    if selected_tag in [t.lower() for t in s.get("tags", [])]
                ]

        if all(s["name"] in installed_names for s in filtered):
            self.console.print(
                "[green]All skills in this category are already installed.[/green]"
            )
            return None

        choices: list = []
        for s in filtered:
            if s["name"] in installed_names:
                choices.append(
                    Choice(
                        title=[
                            ("", f"{s['name']} \u2014 {s['description'][:80]}"),
                            ("class:instruction", "  (installed)"),
                        ],
                        value=s["install_source"],
                        disabled=True,
                    )
                )
            else:
                choices.append(
                    Choice(
                        title=f"{s['name']} \u2014 {s['description'][:80]}",
                        value=s["install_source"],
                    )
                )

        selected = _checkbox_ask(choices, "Select skills to install:")
        if selected is None:
            return None
        return selected or None

    async def wait_for_mcp_browse(
        self, servers: list, installed_names: set[str], pre_filter_tag: str
    ) -> list | None:
        """Shows a questionary tag-filter + checkbox picker for MCP servers.

        Args:
            servers: ``MCPServerEntry`` objects from the marketplace index.
            installed_names: Names of already-configured servers.
            pre_filter_tag: If non-empty, skip the tag picker and filter
                directly by this tag.

        Returns:
            List of selected ``MCPServerEntry`` objects, or ``None`` on cancel.
        """
        tag_counter: Counter[str] = Counter()
        for entry in servers:
            for t in entry.tags:
                tag_counter[t.lower()] += 1

        if pre_filter_tag:
            pre_lower = pre_filter_tag.lower()
            filtered = [e for e in servers if pre_lower in [t.lower() for t in e.tags]]
            if not filtered:
                self.console.print(
                    f"[yellow]No servers found with tag: {pre_filter_tag}[/yellow]"
                )
                if tag_counter:
                    sorted_tags = sorted(
                        tag_counter.items(), key=lambda x: (-x[1], x[0])
                    )
                    tags_str = ", ".join(
                        f"{tag} ({count})" for tag, count in sorted_tags
                    )
                    self.console.print(f"[dim]Available tags: {tags_str}[/dim]")
                return None
        else:
            sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
            tag_choices = [
                Choice(title=f"All servers ({len(servers)})", value="__all__")
            ]
            for tag, count in sorted_tags:
                tag_choices.append(Choice(title=f"{tag} ({count})", value=tag))

            selected_tag = questionary.select(
                "Filter by tag:",
                choices=tag_choices,
                style=_PICKER_STYLE,
                qmark="\u276f",
            ).ask()

            if selected_tag is None:
                return None

            if selected_tag == "__all__":
                filtered = servers
            else:
                filtered = [
                    e for e in servers if selected_tag in [t.lower() for t in e.tags]
                ]

        if all(e.name in installed_names for e in filtered):
            self.console.print(
                "[green]All servers in this category are already configured.[/green]"
            )
            return None

        choices: list = []
        for entry in filtered:
            if entry.name in installed_names:
                choices.append(
                    Choice(
                        title=[
                            ("", f"{entry.name} \u2014 {entry.description[:80]}"),
                            ("class:instruction", "  (configured)"),
                        ],
                        value=entry,
                        disabled=True,
                    )
                )
            else:
                choices.append(
                    Choice(
                        title=f"{entry.name} \u2014 {entry.description[:80]}",
                        value=entry,
                    )
                )

        selected = _checkbox_ask(choices, "Select MCP servers to install:")
        if selected is None:
            return None
        return selected or None

    def clear_chat(self) -> None:
        """Clears the terminal screen."""
        self.console.clear()

    def request_quit(self) -> None:
        """Signals the CLI loop to exit after the current command."""
        self._quit_requested = True

    def force_quit(self) -> None:
        """Signals the CLI loop to exit immediately."""
        self._quit_requested = True

    def start_new_session(self) -> None:
        """Signals the CLI loop to create a fresh session."""
        self._new_session_requested = True

    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None:
        """Signals the CLI loop to resume the given session.

        Args:
            thread_id: Thread ID to resume.
            workspace_dir: Workspace directory restored from metadata.
        """
        self._resume_request = (thread_id, workspace_dir)

    def start_compacting_indicator(self) -> None:
        """Starts a Rich status spinner for the compacting operation."""
        self._status_ctx = self.console.status(
            "[cyan]Compacting conversation...[/cyan]"
        )
        self._status_ctx.__enter__()

    def stop_compacting_indicator(self) -> None:
        """Stops the compacting spinner."""
        if self._status_ctx:
            self._status_ctx.__exit__(None, None, None)
            self._status_ctx = None

    def update_status_after_compact(self, tokens_after: int) -> None:
        """Signals the reduced token count to the CLI loop.

        Args:
            tokens_after: Token count after compaction.
        """
        self._compact_tokens = tokens_after

    def reset_signals(self) -> None:
        """Clears all lifecycle signal flags.

        Called by the CLI dispatch loop after processing each command.
        """
        self._quit_requested = False
        self._new_session_requested = False
        self._resume_request = None
        self._compact_tokens = None
