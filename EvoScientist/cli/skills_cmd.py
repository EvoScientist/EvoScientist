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


def _cmd_install_skill(args: str) -> None:
    """Install a skill from local path or GitHub URL.

    By default, installs to the global skills directory (~/.evoscientist/skills/).
    Append --local to install to the current workspace instead.

    Usage: /install-skill <path-or-url> [--local]
    """
    from ..paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR
    from ..tools.skills_manager import install_skill

    # Parse --local flag out of the args string
    local = "--local" in args.split()
    source = args.replace("--local", "").strip()

    if not source:
        console.print("[red]Usage:[/red] /install-skill <path-or-url> [--local]")
        console.print("[dim]Examples:[/dim]")
        console.print("  /install-skill ./my-skill")
        console.print(
            "  /install-skill https://github.com/user/repo/tree/main/skill-name"
        )
        console.print("  /install-skill user/repo@skill-name")
        console.print(
            "  /install-skill ./my-skill --local  [dim](workspace only)[/dim]"
        )
        console.print()
        return

    dest_label = (
        f"[cyan]{_shorten_path(str(USER_SKILLS_DIR))}[/cyan] [dim](workspace)[/dim]"
        if local
        else f"[cyan]{_shorten_path(str(GLOBAL_SKILLS_DIR))}[/cyan] [dim](global)[/dim]"
    )
    console.print(f"[dim]Installing skill from:[/dim] {source}")
    console.print(f"[dim]Destination:[/dim] {dest_label}")

    result = install_skill(source, global_install=not local)

    if result.get("batch"):
        # Batch install — multiple skills
        for item in result.get("installed", []):
            console.print(f"[green]Installed:[/green] {item['name']}")
            console.print(
                f"  [dim]Description:[/dim] {item.get('description', '(none)')}"
            )
            console.print(
                f"  [dim]Path:[/dim] [cyan]{_shorten_path(item['path'])}[/cyan]"
            )
        for item in result.get("failed", []):
            console.print(f"[red]Failed:[/red] {item['name']} — {item['error']}")
        installed_count = len(result.get("installed", []))
        if installed_count:
            console.print(f"\n[green]{installed_count} skill(s) installed.[/green]")
            console.print("[dim]Reload with /new to apply.[/dim]")
    elif result["success"]:
        console.print(f"[green]Installed:[/green] {result['name']}")
        console.print(f"[dim]Description:[/dim] {result.get('description', '(none)')}")
        console.print(f"[dim]Path:[/dim] [cyan]{_shorten_path(result['path'])}[/cyan]")
        console.print()
        console.print("[dim]Reload with /new to apply.[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")
    console.print()


def _cmd_uninstall_skill(name: str) -> None:
    """Uninstall a user-installed skill."""
    from ..tools.skills_manager import uninstall_skill

    if not name:
        console.print("[red]Usage:[/red] /uninstall-skill <skill-name>")
        console.print("[dim]Use /skills to see installed skills.[/dim]")
        console.print()
        return

    result = uninstall_skill(name)

    if result["success"]:
        console.print(f"[green]Uninstalled:[/green] {name}")
        console.print("[dim]Reload with /new to apply.[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")
    console.print()


def _cmd_install_skills(args: str = "") -> None:
    """Browse and install skills from the EvoSkills repository.

    Args:
        args: Optional tag name to pre-filter (e.g. "core").
    """
    from collections import Counter

    import questionary
    from questionary import Choice

    from ..paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR
    from ..tools.skills_manager import fetch_remote_skill_index, install_skill
    from .interactive import _PICKER_STYLE

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
                style=_PICKER_STYLE,
                qmark="❯",
                **kwargs,
            ).ask()
        finally:
            InquirerControl._get_choice_tokens = original

    # Step 1: Fetch remote index
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

    # Detect already-installed skills (both global and workspace tiers)
    installed_names: set[str] = set()
    for skills_dir in (Path(GLOBAL_SKILLS_DIR), Path(USER_SKILLS_DIR)):
        if skills_dir.exists():
            installed_names.update(e.name for e in skills_dir.iterdir() if e.is_dir())

    pre_filter_tag = args.strip().lower() if args else ""

    # Step 2: Tag filter (skip if pre-filtered via args)
    if pre_filter_tag:
        filtered = [
            s for s in index if pre_filter_tag in [t.lower() for t in s.get("tags", [])]
        ]
        if not filtered:
            console.print(f"[yellow]No skills found with tag: {args.strip()}[/yellow]")
            # Show available tags
            tag_counter: Counter[str] = Counter()
            for s in index:
                for t in s.get("tags", []):
                    tag_counter[t.lower()] += 1
            if tag_counter:
                sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
                tags_str = ", ".join(f"{tag} ({count})" for tag, count in sorted_tags)
                console.print(f"[dim]Available tags: {tags_str}[/dim]")
            console.print()
            return
    else:
        # Build tag choices for interactive picker
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
            style=_PICKER_STYLE,
            qmark="❯",
        ).ask()

        if selected_tag is None:
            console.print()
            return

        if selected_tag == "__all__":
            filtered = index
        else:
            filtered = [
                s
                for s in index
                if selected_tag in [t.lower() for t in s.get("tags", [])]
            ]

    # Step 3: Skill selection checkbox
    all_installed = all(s["name"] in installed_names for s in filtered)
    if all_installed:
        console.print(
            "[green]All skills in this category are already installed.[/green]"
        )
        console.print()
        return

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
