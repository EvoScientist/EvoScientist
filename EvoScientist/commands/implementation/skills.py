from __future__ import annotations

from typing import ClassVar

from rich.table import Table

from ..base import Argument, Command, CommandContext
from ..manager import manager


class SkillsCommand(Command):
    """List installed skills."""

    name = "/skills"
    description = "List installed skills"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...cli.agent import _shorten_path
        from ...paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR
        from ...tools.skills_manager import list_skills

        skills = list_skills(include_system=True)
        if not skills:
            ctx.ui.append_system("No skills available.", style="dim")
            ctx.ui.append_system(
                "Install with: /install-skill <path-or-url>", style="dim"
            )
            ctx.ui.append_system(
                f"Global skills: {_shorten_path(str(GLOBAL_SKILLS_DIR))}",
                style="dim",
            )
            return

        sections = [
            (
                "Workspace Skills",
                [s for s in skills if s.source == "workspace"],
                "green",
            ),
            ("Global Skills", [s for s in skills if s.source == "global"], "cyan"),
            ("Built-in Skills", [s for s in skills if s.source == "builtin"], "blue"),
        ]

        for title, group, color in sections:
            if not group:
                continue
            table = Table(title=f"{title} ({len(group)})", show_header=True)
            table.add_column("Name", style=color)
            table.add_column("Description", style="dim")
            table.add_column("Tags", style="dim")
            for s in group:
                tags = "\n".join(f"· {t}" for t in s.tags[:4]) if s.tags else ""
                table.add_row(s.name, s.description, tags)
            ctx.ui.mount_renderable(table)

        ctx.ui.append_system(
            f"Global: {_shorten_path(str(GLOBAL_SKILLS_DIR))}  "
            f"Workspace: {_shorten_path(str(USER_SKILLS_DIR))}",
            style="dim",
        )


class InstallSkill(Command):
    """Add a skill from path or GitHub."""

    name: ClassVar[str] = "/install-skill"
    description: ClassVar[str] = "Add a skill from path or GitHub"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="source",
            type=str,
            description="Path or GitHub URL of the skill",
            required=True,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...cli.agent import _shorten_path
        from ...tools.skills_manager import install_skill

        raw = " ".join(args)
        local = "--local" in args
        source = raw.replace("--local", "").strip()

        if not source:
            ctx.ui.append_system(
                "Usage: /install-skill <path-or-url> [--local]", style="yellow"
            )
            ctx.ui.append_system("Examples:", style="dim")
            ctx.ui.append_system("  /install-skill ./my-skill", style="dim")
            ctx.ui.append_system(
                "  /install-skill https://github.com/user/repo/tree/main/skill-name",
                style="dim",
            )
            ctx.ui.append_system("  /install-skill user/repo@skill-name", style="dim")
            ctx.ui.append_system(
                "  /install-skill ./my-skill --local  (workspace only)", style="dim"
            )
            return

        from ...paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR

        dest = USER_SKILLS_DIR if local else GLOBAL_SKILLS_DIR
        ctx.ui.append_system(f"Installing skill from: {source}", style="dim")
        ctx.ui.append_system(
            f"Destination: {_shorten_path(str(dest))} "
            f"({'workspace' if local else 'global'})",
            style="dim",
        )
        result = install_skill(source, global_install=not local)
        if result.get("batch"):
            for item in result.get("installed", []):
                ctx.ui.append_system(f"Installed: {item['name']}", style="green")
                ctx.ui.append_system(
                    f"  Description: {item.get('description', '(none)')}", style="dim"
                )
            for item in result.get("failed", []):
                ctx.ui.append_system(
                    f"Failed: {item['name']} — {item['error']}", style="red"
                )
            installed_count = len(result.get("installed", []))
            if installed_count:
                ctx.ui.append_system(
                    f"{installed_count} skill(s) installed. Reload with /new to apply.",
                    style="dim",
                )
        elif result.get("success"):
            ctx.ui.append_system(f"Installed: {result['name']}", style="green")
            ctx.ui.append_system(
                f"Description: {result.get('description', '(none)')}", style="dim"
            )
            ctx.ui.append_system(f"Path: {_shorten_path(result['path'])}", style="dim")
            ctx.ui.append_system("Reload with /new to apply.", style="dim")
        else:
            ctx.ui.append_system(f"Failed: {result['error']}", style="red")


class InstallSkills(Command):
    """Browse and install skills."""

    name: ClassVar[str] = "/evoskills"
    description: ClassVar[str] = (
        "Browse and install EvoSkills (optional: /evoskills <tag>)"
    )
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="tag", type=str, description="Tag to filter skills by", required=False
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from pathlib import Path as _Path

        from ...paths import USER_SKILLS_DIR
        from ...tools.skills_manager import fetch_remote_skill_index, install_skill

        tag = args[0] if args else ""
        ctx.ui.append_system(
            f"Fetching skill index{' for tag: ' + tag if tag else ''}...", style="dim"
        )

        try:
            index = fetch_remote_skill_index()
        except Exception as e:
            ctx.ui.append_system(f"Failed to fetch skill index: {e}", style="red")
            ctx.ui.append_system(
                "Try: /install-skill EvoScientist/EvoSkills@skills", style="dim"
            )
            return

        if not index:
            ctx.ui.append_system("No skills found.", style="yellow")
            return

        # Detect installed skills (both global and workspace tiers)
        from ...paths import GLOBAL_SKILLS_DIR

        installed_names: set[str] = set()
        for skills_dir in (_Path(GLOBAL_SKILLS_DIR), _Path(USER_SKILLS_DIR)):
            if skills_dir.exists():
                installed_names.update(
                    e.name for e in skills_dir.iterdir() if e.is_dir()
                )

        selected_sources: list[str] | None = None

        # For non-interactive UIs (channels), if a tag is provided, we can auto-install all matching skills
        # instead of failing due to lack of interactive UI.
        is_channel = not ctx.ui.supports_interactive
        if is_channel and tag:
            tag_lower = tag.lower()
            matches = []
            for s in index:
                tags = [t.lower() for t in s.get("tags", [])]
                if tag_lower in tags:
                    matches.append(s)

            if not matches:
                ctx.ui.append_system(
                    f"No skills found with tag '{tag}'.", style="yellow"
                )
                return

            ctx.ui.append_system(
                f"Found {len(matches)} skill(s) with tag '{tag}'. Installing..."
            )
            selected_sources = [s["install_source"] for s in matches]
        else:
            # Wait for user interaction
            selected_sources = await ctx.ui.wait_for_skill_browse(
                index,
                installed_names,
                pre_filter_tag=tag,
            )

        if not selected_sources:
            if not is_channel:
                ctx.ui.append_system("Browse cancelled.", style="dim")
            return

        # Install selected skills
        installed_count = 0
        for source in selected_sources:
            result = install_skill(source, global_install=True)
            if result.get("batch"):
                for item in result.get("installed", []):
                    ctx.ui.append_system(f"Installed: {item['name']}", style="green")
                    installed_count += 1
            elif result.get("success"):
                ctx.ui.append_system(f"Installed: {result['name']}", style="green")
                installed_count += 1
            else:
                ctx.ui.append_system(
                    f"Failed: {result.get('error', 'unknown')}", style="red"
                )

        if installed_count > 0:
            ctx.ui.append_system(
                f"Successfully installed {installed_count} skill(s). Reload with /new to apply.",
                style="dim",
            )
        elif not is_channel:
            ctx.ui.append_system("No skills were installed.", style="yellow")


class UninstallSkill(Command):
    """Remove an installed skill."""

    name: ClassVar[str] = "/uninstall-skill"
    description: ClassVar[str] = "Remove an installed skill"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="name",
            type=str,
            description="Name of the skill to remove",
            required=True,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...tools.skills_manager import uninstall_skill

        name = args[0] if args else ""
        if not name:
            ctx.ui.append_system("Usage: /uninstall-skill <skill-name>", style="yellow")
            ctx.ui.append_system("Use /skills to see installed skills.", style="dim")
            return

        result = uninstall_skill(name)
        if result["success"]:
            ctx.ui.append_system(f"Uninstalled: {name}", style="green")
            ctx.ui.append_system("Reload with /new to apply.", style="dim")
        else:
            ctx.ui.append_system(f"Failed: {result['error']}", style="red")


# Register skill commands
manager.register(SkillsCommand())
manager.register(InstallSkill())
manager.register(InstallSkills())
manager.register(UninstallSkill())
