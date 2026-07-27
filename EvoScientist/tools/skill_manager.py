"""Skill management tool (LangChain @tool wrapper)."""

from typing import Literal

from langchain_core.tools import tool


@tool(parse_docstring=True)
def skill_manager(
    action: Literal["install", "list", "uninstall", "info", "browse"],
    source: str = "",
    name: str = "",
    tag: str = "",
    include_system: bool = False,
    skill_type: Literal["all", "expert", "utility"] = "all",
) -> str:
    """Manage user-installable skills: install from GitHub or local path, list available skills, browse remote skills, get details, or uninstall.

    Actions and required parameters:

    action="install" (requires source):
      Install a skill. The source can be:
      - GitHub shorthand: "owner/repo@skill-name" (e.g. "anthropics/skills@peft")
      - GitHub URL: "https://github.com/owner/repo/tree/main/skill-name"
      - Local path: "./my-skill" or "/path/to/skill"
      Nested skills are auto-resolved — if the skill is not at the repo root, subdirectories are searched automatically.

    action="list":
      List installed skills. By default only shows user-installed skills.
      Set include_system=True to also show built-in system skills.
      Set skill_type="expert" to show only expert skills (agent teams);
      set skill_type="utility" to exclude them; leave as "all" (default) for no filter.
      Built-in skills evolve over time, so use action="list" to see the current set.

    action="browse" (optional tag):
      Browse available skills from the EvoSkills repository (EvoScientist/EvoSkills).
      Set tag to filter by category (e.g. tag="core", tag="writing", tag="experiments").
      Returns skill names, descriptions, tags, and install sources you can pass to action="install".

    action="info" (requires name):
      Get details (description, source, path, tags) about a specific skill by name.
      Expert skills also surface their role, byline, capability tags, and default
      dispatch mode. Searches both user and system skills.

    action="uninstall" (requires name):
      Remove a user-installed skill by name. System skills cannot be uninstalled.

    Args:
        action: The operation to perform — "install", "list", "browse", "info", or "uninstall"
        source: Required for install — GitHub shorthand, GitHub URL, or local directory path
        name: Required for info and uninstall — the skill name (for example, one returned by action="list")
        tag: Optional for browse — filter by tag (e.g. "core", "writing", "experiments", "research")
        include_system: Only for list — set True to include built-in system skills in the output
        skill_type: Only for list — one of "all" (default; no filter), "expert" (agent-team skills), or "utility"

    Returns:
        Result message
    """
    from .skills_manager import (
        fetch_remote_skill_index,
        get_skill_info,
        install_skill,
        list_skills,
        uninstall_skill,
    )

    if action == "install":
        if not source:
            return (
                "Error: 'source' is required for install action. "
                "Provide a GitHub shorthand (e.g. source='owner/repo@skill-name'), "
                "a GitHub URL, or a local directory path."
            )
        result = install_skill(source)
        if result["success"]:
            return (
                f"Successfully installed skill: {result['name']}\n"
                f"Description: {result.get('description', '(none)')}\n"
                f"Path: {result['path']}\n\n"
                f"Read its SKILL.md for full instructions."
            )
        else:
            return f"Failed to install skill: {result['error']}"

    elif action == "list":
        skills = list_skills(include_system=include_system)
        # `skill_type="all"` (default) is the no-filter case. A specific
        # value ("expert" or "utility") narrows the list to that type.
        # Empty string is NOT a legal input — Gemini's function-declaration
        # schema rejects empty enum values (was the root cause of a live
        # smoke failure); the `Literal` above defends against it.
        if skill_type in ("expert", "utility"):
            skills = [s for s in skills if s.type == skill_type]
        if not skills:
            if skill_type in ("expert", "utility"):
                return f"No {skill_type} skills found."
            if include_system:
                return "No skills found."
            return "No user skills installed. Use action='install' to add skills, or set include_system=True to see built-in skills."
        user_skills = [s for s in skills if s.source in ("workspace", "global")]
        system_skills = [s for s in skills if s.source == "builtin"]
        lines = []
        if user_skills:
            lines.append(f"User Skills ({len(user_skills)}):")
            for skill in user_skills:
                tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
                lines.append(f"  - {skill.name}: {skill.description}{tags_str}")
        if system_skills:
            if lines:
                lines.append("")
            lines.append(f"System Skills ({len(system_skills)}):")
            for skill in system_skills:
                tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
                lines.append(f"  - {skill.name}: {skill.description}{tags_str}")
        return "\n".join(lines)

    elif action == "browse":
        try:
            index = fetch_remote_skill_index()
        except Exception as e:
            return f"Failed to fetch skill index: {e}"
        if tag:
            tag_lower = tag.lower()
            index = [
                s for s in index if tag_lower in [t.lower() for t in s.get("tags", [])]
            ]
        if not index:
            return f"No skills found{' with tag: ' + tag if tag else ''}."
        lines = [f"Available Skills ({len(index)}):"]
        for s in index:
            tags_str = " · ".join(s.get("tags", []))
            lines.append(f"  - {s['name']}: {s['description']}")
            if tags_str:
                lines.append(f"    Tags: {tags_str}")
            lines.append(
                f"    Install: skill_manager(action='install', source='{s['install_source']}')"
            )
        return "\n".join(lines)

    elif action == "uninstall":
        if not name:
            return (
                "Error: 'name' is required for uninstall action. "
                "Use action='list' first to see installed skill names."
            )
        result = uninstall_skill(name)
        if result["success"]:
            return f"Successfully uninstalled skill: {name}"
        else:
            return f"Failed to uninstall skill: {result['error']}"

    elif action == "info":
        if not name:
            return (
                "Error: 'name' is required for info action. "
                "Use action='list' with include_system=True to see all available skill names."
            )
        info = get_skill_info(name)
        if not info:
            return (
                f"Skill not found: {name}. "
                f"Use action='list' with include_system=True to see all available skills."
            )
        lines = [
            f"Name: {info.name}",
            f"Description: {info.description}",
            f"Source: {info.source}",
            f"Path: {info.path}",
        ]
        if info.tags:
            lines.append(f"Tags: {', '.join(info.tags)}")
        # Expert-skill surface (agent-teams v1): only shown when the
        # skill declared `type: expert` in its SKILL.md frontmatter.
        if info.type == "expert":
            lines.append("Type: expert")
            if info.role:
                lines.append(f"Role: {info.role}")
            if info.byline:
                lines.append(f"Byline: {info.byline}")
            if info.capability_tags:
                lines.append(f"Capability tags: {', '.join(info.capability_tags)}")
            if info.avatar_hint:
                lines.append(f"Avatar hint: {info.avatar_hint}")
            if info.default_dispatch:
                lines.append(f"Default dispatch: {info.default_dispatch}")
        return "\n".join(lines)

    else:
        return f"Unknown action: {action}. Use 'install', 'list', 'browse', 'uninstall', or 'info'."
