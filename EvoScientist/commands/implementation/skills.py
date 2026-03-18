from __future__ import annotations

from rich.table import Table

from ..base import Argument, Command, CommandContext
from ..manager import manager


class SkillsCommand(Command):
    """List installed skills."""

    name = "/skills"
    description = "List installed skills"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...cli.agent import _shorten_path
        from ...paths import USER_SKILLS_DIR
        from ...tools.skills_manager import list_skills

        skills = list_skills(include_system=True)
        if not skills:
            ctx.ui.append_system("No skills available.", style="dim")
            ctx.ui.append_system(
                "Install with: /install-skill <path-or-url>", style="dim"
            )
            ctx.ui.append_system(
                f"Skills directory: {_shorten_path(str(USER_SKILLS_DIR))}",
                style="dim",
            )
            return

        user_skills = [s for s in skills if s.source == "user"]
        system_skills = [s for s in skills if s.source == "system"]

        if user_skills:
            table = Table(title=f"User Skills ({len(user_skills)})", show_header=True)
            table.add_column("Name", style="green")
            table.add_column("Description", style="dim")
            table.add_column("Tags", style="dim")
            for s in user_skills:
                tags = "\n".join(f"· {t}" for t in s.tags[:4]) if s.tags else ""
                table.add_row(s.name, s.description, tags)
            ctx.ui.mount_renderable(table)

        if system_skills:
            table = Table(
                title=f"Built-in Skills ({len(system_skills)})", show_header=True
            )
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="dim")
            table.add_column("Tags", style="dim")
            for s in system_skills:
                tags = "\n".join(f"· {t}" for t in s.tags[:4]) if s.tags else ""
                table.add_row(s.name, s.description, tags)
            ctx.ui.mount_renderable(table)

        ctx.ui.append_system(
            f"User skills folder: {_shorten_path(str(USER_SKILLS_DIR))}",
            style="dim",
        )


class InstallSkillCommand(Command):
    """Add a skill from path or GitHub."""

    name = "/install-skill"
    description = "Add a skill from path or GitHub"
    arguments = [
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

        source = args[0] if args else ""
        if not source:
            ctx.ui.append_system("Usage: /install-skill <path-or-url>", style="yellow")
            ctx.ui.append_system("Examples:", style="dim")
            ctx.ui.append_system("  /install-skill ./my-skill", style="dim")
            ctx.ui.append_system(
                "  /install-skill https://github.com/user/repo/tree/main/skill-name",
                style="dim",
            )
            ctx.ui.append_system("  /install-skill user/repo@skill-name", style="dim")
            return

        ctx.ui.append_system(f"Installing skill from: {source}", style="dim")
        # For simplicity, calling install_skill directly (might block loop if slow?
        # But install_skill doesn't seem to be async)
        result = install_skill(source)
        if result["success"]:
            ctx.ui.append_system(f"Installed: {result['name']}", style="green")
            ctx.ui.append_system(
                f"Description: {result.get('description', '(none)')}",
                style="dim",
            )
            ctx.ui.append_system(f"Path: {_shorten_path(result['path'])}", style="dim")
            ctx.ui.append_system("Reload with /new to apply.", style="dim")
        else:
            ctx.ui.append_system(f"Failed: {result['error']}", style="red")


class InstallSkillsCommand(Command):
    """Browse and install skills."""

    name = "/install-skills"
    description = "Browse and install skills (optional: /install-skills <tag>)"
    arguments = [
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

        # Detect installed skills
        skills_dir = _Path(USER_SKILLS_DIR)
        installed_names: set[str] = set()
        if skills_dir.exists():
            installed_names = {e.name for e in skills_dir.iterdir() if e.is_dir()}

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
            result = install_skill(source)
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


class UninstallSkillCommand(Command):
    """Remove an installed skill."""

    name = "/uninstall-skill"
    description = "Remove an installed skill"
    arguments = [
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
manager.register(InstallSkillCommand())
manager.register(InstallSkillsCommand())
manager.register(UninstallSkillCommand())
