"""CLI /install-mcp — questionary-based interactive browser.

Uses the same logic as the TUI command (``commands.implementation.mcp_install``)
but provides a questionary-based UI for the plain CLI interactive mode.
"""

from __future__ import annotations

from collections import Counter

import questionary
from questionary import Choice

from ..mcp.registry import (
    MCPServerEntry,
    fetch_marketplace_index,
    find_server_by_name,
    get_all_tags,
    get_installed_names,
    install_mcp_server,
    install_mcp_servers,
)
from ..stream.console import console
from .widgets.thread_selector import PICKER_STYLE

_INSTALLED_INDICATOR = ("fg:#4caf50", "\u2713 ")


def _checkbox_ask(choices, message: str, **kwargs):
    """questionary.checkbox that renders disabled items with checkmark."""
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
            style=PICKER_STYLE,
            qmark="\u276f",
            **kwargs,
        ).ask()
    finally:
        InquirerControl._get_choice_tokens = original


def _browse_and_select(
    servers: list[MCPServerEntry],
    installed_names: set[str],
    pre_filter_tag: str = "",
) -> list[MCPServerEntry] | None:
    """Questionary-based tag picker + checkbox selection.

    Returns selected entries, empty list if none selected, or None on cancel.
    """
    tag_counter: Counter[str] = Counter()
    for entry in servers:
        for t in entry.tags:
            tag_counter[t.lower()] += 1

    if pre_filter_tag:
        pre_filter_tag = pre_filter_tag.lower()
        filtered = [e for e in servers if pre_filter_tag in [t.lower() for t in e.tags]]
        if not filtered:
            console.print(
                f"[yellow]No servers found with tag: {pre_filter_tag}[/yellow]"
            )
            if tag_counter:
                sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
                tags_str = ", ".join(f"{tag} ({count})" for tag, count in sorted_tags)
                console.print(f"[dim]Available tags: {tags_str}[/dim]")
            return None
    else:
        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        tag_choices = [Choice(title=f"All servers ({len(servers)})", value="__all__")]
        for tag, count in sorted_tags:
            tag_choices.append(Choice(title=f"{tag} ({count})", value=tag))

        selected_tag = questionary.select(
            "Filter by tag:",
            choices=tag_choices,
            style=PICKER_STYLE,
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
        console.print(
            "[green]All servers in this category are already configured.[/green]"
        )
        return None

    choices = []
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
    return selected


def _cmd_install_mcp(args: str = "") -> None:
    """Entry point for ``/install-mcp`` in CLI mode."""
    args = args.strip()

    console.print("[dim]Fetching MCP server index...[/dim]")
    try:
        servers = fetch_marketplace_index()
    except Exception as e:
        console.print(f"[red]Failed to fetch server index: {e}[/red]")
        console.print()
        return

    if not servers:
        console.print("[yellow]No MCP servers found.[/yellow]")
        console.print()
        return

    # Direct name match
    if args:
        match = find_server_by_name(args, servers)
        if match:
            installed = get_installed_names()
            if match.name in installed:
                console.print(f"[yellow]{match.name} is already configured.[/yellow]")
                console.print()
                return
            if install_mcp_server(match):
                console.print(f"[green]Configured:[/green] [cyan]{match.name}[/cyan]")
                console.print("[dim]Reload with /new to apply.[/dim]")
            else:
                console.print(f"[red]Failed to configure {match.name}.[/red]")
            console.print()
            return

        # Tag match — fall through to browser
        if args.lower() not in get_all_tags(servers):
            console.print(f"[red]No server or tag found matching: {args}[/red]")
            close = [s.name for s in servers if args.lower() in s.name.lower()]
            if close:
                console.print(f"[dim]Did you mean: {', '.join(close)}?[/dim]")
            console.print()
            return

    # Interactive browse
    installed_names = get_installed_names()
    selected = _browse_and_select(servers, installed_names, pre_filter_tag=args)

    if not selected:
        if selected is None:
            console.print("[dim]Cancelled.[/dim]")
        else:
            console.print("[dim]No servers selected.[/dim]")
        console.print()
        return

    def _print(text: str, style: str = "") -> None:
        console.print(f"[{style}]{text}[/{style}]" if style else text)

    count = install_mcp_servers(selected, print_fn=_print)
    if count:
        console.print(f"\n[green]{count} server(s) configured.[/green]")
        console.print("[dim]Reload with /new to apply.[/dim]")
    console.print()
