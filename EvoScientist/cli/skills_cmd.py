"""Slash commands for skill management: /skills, /install-skill, /uninstall-skill, /evoskills."""

from pathlib import Path

from ..stream.console import console
from .agent import _shorten_path


def _cmd_list_skills() -> None:
    """List all available skills (workspace, global, and built-in)."""
    from ..paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR
    from ..tools.skills_manager import list_skills

    skills = list_skills(include_system=True)

    if not skills:
        console.print("[dim]No skills available.[/dim]")
        console.print("[dim]Install with:[/dim] /install-skill <path-or-url>")
        console.print(
            f"[dim]Global skills:[/dim] [cyan]{_shorten_path(str(GLOBAL_SKILLS_DIR))}[/cyan]"
        )
        console.print()
        return

    workspace_skills = [s for s in skills if s.source == "workspace"]
    global_skills = [s for s in skills if s.source == "global"]
    builtin_skills = [s for s in skills if s.source == "builtin"]

    sections = [
        ("Workspace Skills", workspace_skills, "green"),
        ("Global Skills", global_skills, "cyan"),
        ("Built-in Skills", builtin_skills, "blue"),
    ]

    printed = False
    for title, group, color in sections:
        if not group:
            continue
        if printed:
            console.print()
        console.print(f"[bold]{title}[/bold] ({len(group)}):")
        for skill in group:
            tags_str = f" [dim]({', '.join(skill.tags)})[/dim]" if skill.tags else ""
            console.print(
                f"  [{color}]{skill.name}[/{color}] - {skill.description}{tags_str}"
            )
        printed = True

    console.print(
        f"\n[dim]Global skills:[/dim] [cyan]{_shorten_path(str(GLOBAL_SKILLS_DIR))}[/cyan]"
    )
    console.print(
        f"[dim]Workspace skills:[/dim] [green]{_shorten_path(str(USER_SKILLS_DIR))}[/green]"
    )
    console.print()


def _pick_skills_interactive(
    index: list[dict],
    installed_names: set[str],
    pre_filter_tag: str,
) -> list[str] | None:
    """Interactive questionary picker for EvoSkills browse.

    Two-phase picker:
    1. tag filter — ``questionary.select`` (skipped if ``pre_filter_tag``)
    2. multi-select — ``questionary.checkbox`` with installed items disabled

    Returns:
        list of ``install_source`` strings selected by the user,
        ``None`` if the user cancelled at either phase, or
        ``[]`` if nothing was selectable / all-installed in the filter.
    """
    from collections import Counter

    import questionary
    from questionary import Choice

    from .widgets.thread_selector import PICKER_STYLE

    # Installed-item indicator style for disabled checkbox choices.
    _INSTALLED_INDICATOR = ("fg:#4caf50", "✓ ")

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
                qmark="❯",
                **kwargs,
            ).ask()
        finally:
            InquirerControl._get_choice_tokens = original

    pre_filter_tag = (pre_filter_tag or "").strip().lower()

    # Phase 1: tag filter (skip if pre-filtered via args)
    if pre_filter_tag:
        filtered = [
            s for s in index if pre_filter_tag in [t.lower() for t in s.get("tags", [])]
        ]
        if not filtered:
            console.print(
                f"[yellow]No skills found with tag: {pre_filter_tag}[/yellow]"
            )
            tag_counter: Counter[str] = Counter()
            for s in index:
                for t in s.get("tags", []):
                    tag_counter[t.lower()] += 1
            if tag_counter:
                sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
                tags_str = ", ".join(f"{tag} ({count})" for tag, count in sorted_tags)
                console.print(f"[dim]Available tags: {tags_str}[/dim]")
            return []
    else:
        tag_counter = Counter()
        for s in index:
            for t in s.get("tags", []):
                tag_counter[t.lower()] += 1

        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        tag_choices = [Choice(title=f"All skills ({len(index)})", value="__all__")]
        for tag, count in sorted_tags:
            tag_choices.append(Choice(title=f"{tag} ({count})", value=tag))

        selected_tag = questionary.select(
            "Filter by tag:",
            choices=tag_choices,
            style=PICKER_STYLE,
            qmark="❯",
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

    # Phase 2: skill selection checkbox
    if all(s["name"] in installed_names for s in filtered):
        console.print(
            "[green]All skills in this category are already installed.[/green]"
        )
        return []

    choices = []
    for s in filtered:
        if s["name"] in installed_names:
            choices.append(
                Choice(
                    title=[
                        ("", f"{s['name']} — {s['description'][:80]}"),
                        ("class:instruction", "  (installed)"),
                    ],
                    value=s["install_source"],
                    disabled=True,
                )
            )
        else:
            choices.append(
                Choice(
                    title=f"{s['name']} — {s['description'][:80]}",
                    value=s["install_source"],
                )
            )

    selected = _checkbox_ask(choices, "Select skills to install:")

    if selected is None:
        return None
    return list(selected)


def _cmd_install_skills(args: str = "") -> None:
    """Browse and install skills from the EvoSkills repository.

    Args:
        args: Optional tag name to pre-filter (e.g. "core").
    """
    from ..paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR
    from ..tools.skills_manager import fetch_remote_skill_index, install_skill

    console.print("[dim]Fetching skill index...[/dim]")
    try:
        index = fetch_remote_skill_index()
    except Exception as e:
        console.print(f"[red]Failed to fetch skill index: {e}[/red]")
        console.print(
            "[dim]Try installing directly: /install-skill EvoScientist/EvoSkills@skills[/dim]"
        )
        console.print()
        return

    if not index:
        console.print("[yellow]No skills found in the repository.[/yellow]")
        console.print()
        return

    installed_names: set[str] = set()
    for skills_dir in (Path(GLOBAL_SKILLS_DIR), Path(USER_SKILLS_DIR)):
        if skills_dir.exists():
            installed_names.update(e.name for e in skills_dir.iterdir() if e.is_dir())

    selected = _pick_skills_interactive(index, installed_names, args)

    if selected is None:
        console.print()
        return

    if not selected:
        console.print("[dim]No skills selected.[/dim]")
        console.print()
        return

    # Step 4: Install selected skills (default: global)
    installed_count = 0
    for source in selected:
        result = install_skill(source, global_install=True)
        if result.get("batch"):
            for item in result.get("installed", []):
                console.print(f"[green]Installed:[/green] {item['name']}")
                installed_count += 1
            for item in result.get("failed", []):
                console.print(f"[red]Failed:[/red] {item['name']} — {item['error']}")
        elif result.get("success"):
            console.print(f"[green]Installed:[/green] {result['name']}")
            installed_count += 1
        else:
            console.print(f"[red]Failed:[/red] {result.get('error', 'unknown')}")

    if installed_count:
        console.print(f"\n[green]{installed_count} skill(s) installed.[/green]")
        console.print("[dim]Reload with /new to apply.[/dim]")
    console.print()
