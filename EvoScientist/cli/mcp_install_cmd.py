"""Slash command for MCP server browsing and installation: /install-mcp."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import questionary
from prompt_toolkit.styles import Style as PtStyle
from questionary import Choice

from ..stream.display import console
from .agent import _shorten_path


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

# Installed-item indicator style for disabled checkbox choices.
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
            style=_PICKER_STYLE,
            qmark="\u276f",
            **kwargs,
        ).ask()
    finally:
        InquirerControl._get_choice_tokens = original


def _is_file_path(arg: str) -> bool:
    """Heuristic: does *arg* look like a file path rather than a server name?"""
    return (
        arg.endswith((".yaml", ".yml"))
        or "/" in arg
        or "\\" in arg
        or arg.startswith(".")
        or arg.startswith("~")
    )


def _cmd_install_mcp(args: str = "") -> None:
    """Entry point for ``/install-mcp`` and ``/mcp install``.

    Dispatch:
    - No args → interactive browser (built-in + marketplace)
    - File path → import from YAML file
    - Otherwise → try name match, then tag match, then error
    """
    args = args.strip()

    if not args:
        _install_mcp_interactive()
        return

    if _is_file_path(args):
        _install_mcp_from_file(args)
        return

    # Try name match, then tag match
    _install_mcp_by_name_or_tag(args)


# =============================================================================
# Interactive browser
# =============================================================================


def _install_mcp_interactive(pre_filter_tag: str = "") -> None:
    """Browse and install MCP servers from the merged registry."""
    from ..mcp.registry import (
        MCPServerEntry,
        get_merged_registry,
        install_mcp_server,
    )
    from ..mcp.client import _load_user_config

    console.print("[dim]Fetching MCP server index...[/dim]")
    try:
        registry = get_merged_registry()
    except Exception as e:
        console.print(f"[red]Failed to fetch server index: {e}[/red]")
        console.print(
            "[dim]Add servers manually with: /mcp add <name> <command-or-url>[/dim]"
        )
        console.print()
        return

    if not registry:
        console.print("[yellow]No MCP servers found in registry.[/yellow]")
        console.print()
        return

    existing_config = _load_user_config()
    installed_names = set(existing_config.keys())

    # Build tag list
    tag_counter: Counter[str] = Counter()
    for entry in registry:
        for t in entry.tags:
            tag_counter[t.lower()] += 1

    # Pre-filter by tag?
    if pre_filter_tag:
        pre_filter_tag = pre_filter_tag.lower()
        filtered = [
            e for e in registry if pre_filter_tag in [t.lower() for t in e.tags]
        ]
        if not filtered:
            console.print(
                f"[yellow]No servers found with tag: {pre_filter_tag}[/yellow]"
            )
            if tag_counter:
                sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
                tags_str = ", ".join(f"{tag} ({count})" for tag, count in sorted_tags)
                console.print(f"[dim]Available tags: {tags_str}[/dim]")
            console.print()
            return
    else:
        # Interactive tag picker
        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        tag_choices = [Choice(title=f"All servers ({len(registry)})", value="__all__")]
        for tag, count in sorted_tags:
            tag_choices.append(Choice(title=f"{tag} ({count})", value=tag))

        selected_tag = questionary.select(
            "Filter by tag:",
            choices=tag_choices,
            style=_PICKER_STYLE,
            qmark="\u276f",
        ).ask()

        if selected_tag is None:
            console.print()
            return

        if selected_tag == "__all__":
            filtered = registry
        else:
            filtered = [
                e
                for e in registry
                if selected_tag in [t.lower() for t in e.tags]
            ]

    # All already installed?
    if all(e.name in installed_names for e in filtered):
        console.print(
            "[green]All servers in this category are already configured.[/green]"
        )
        console.print()
        return

    # Server selection checkbox
    _select_and_install(filtered, installed_names)


def _select_and_install(
    servers: list,
    installed_names: set[str],
) -> None:
    """Show checkbox picker and install selected servers."""
    from ..mcp.registry import install_mcp_server

    choices = []
    for entry in servers:
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
        console.print()
        return

    if not selected:
        console.print("[dim]No servers selected.[/dim]")
        console.print()
        return

    # Check npx availability for stdio servers that need it
    npx_needed = any(e.command == "npx" for e in selected)
    if npx_needed:
        import shutil

        if not shutil.which("npx"):
            console.print(
                "[yellow]\u26a0 npx not found — servers requiring npx will be configured but may fail to start.[/yellow]"
            )
            console.print(
                "[dim]Install Node.js to use npx-based MCP servers.[/dim]"
            )

    # Install selected servers
    installed_count = 0
    for entry in selected:
        if install_mcp_server(entry):
            console.print(f"[green]Configured:[/green] [cyan]{entry.name}[/cyan]")
            installed_count += 1
        else:
            console.print(f"[red]Failed:[/red] {entry.name}")

    if installed_count:
        console.print(f"\n[green]{installed_count} server(s) configured.[/green]")
        console.print("[dim]Reload with /new to apply.[/dim]")
    console.print()


# =============================================================================
# File import
# =============================================================================


def _install_mcp_from_file(path: str) -> None:
    """Import MCP servers from a YAML file."""
    from ..mcp.registry import load_servers_from_yaml
    from ..mcp.client import _load_user_config

    resolved = os.path.expanduser(path)

    try:
        servers = load_servers_from_yaml(resolved)
    except FileNotFoundError:
        console.print(f"[red]File not found:[/red] {path}")
        console.print()
        return
    except (ValueError, Exception) as e:
        console.print(f"[red]Failed to parse {path}:[/red] {e}")
        console.print()
        return

    if not servers:
        console.print(f"[yellow]No servers found in {path}.[/yellow]")
        console.print()
        return

    console.print(f"[dim]Found {len(servers)} server(s) in {path}[/dim]")

    existing_config = _load_user_config()
    installed_names = set(existing_config.keys())

    _select_and_install(servers, installed_names)


# =============================================================================
# Name / tag lookup
# =============================================================================


def _install_mcp_by_name_or_tag(arg: str) -> None:
    """Try to install by exact name, else fall back to tag filter."""
    from ..mcp.registry import get_merged_registry, install_mcp_server
    from ..mcp.client import _load_user_config

    console.print("[dim]Fetching MCP server index...[/dim]")
    try:
        registry = get_merged_registry()
    except Exception as e:
        console.print(f"[red]Failed to fetch server index: {e}[/red]")
        console.print()
        return

    # Exact name match
    arg_lower = arg.lower()
    match = next((e for e in registry if e.name.lower() == arg_lower), None)
    if match:
        existing_config = _load_user_config()
        if match.name in existing_config:
            console.print(
                f"[yellow]{match.name} is already configured.[/yellow]"
            )
            console.print()
            return

        if install_mcp_server(match):
            console.print(
                f"[green]Configured:[/green] [cyan]{match.name}[/cyan]"
            )
            console.print("[dim]Reload with /new to apply.[/dim]")
        else:
            console.print(f"[red]Failed to configure {match.name}.[/red]")
        console.print()
        return

    # Tag match — launch interactive browser pre-filtered
    all_tags = {t.lower() for e in registry for t in e.tags}
    if arg_lower in all_tags:
        _install_mcp_interactive(pre_filter_tag=arg)
        return

    # No match — show suggestions
    console.print(f"[red]No server or tag found matching: {arg}[/red]")
    names = [e.name for e in registry]
    # Fuzzy suggestions
    close = [n for n in names if arg_lower in n.lower()]
    if close:
        console.print(f"[dim]Did you mean: {', '.join(close)}?[/dim]")
    else:
        console.print(f"[dim]Available servers: {', '.join(names[:10])}{'...' if len(names) > 10 else ''}[/dim]")
    if all_tags:
        console.print(f"[dim]Available tags: {', '.join(sorted(all_tags))}[/dim]")
    console.print()
