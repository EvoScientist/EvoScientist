"""Skill installation and management for EvoScientist.

This module provides functions for installing, listing, and uninstalling user skills.
Skills are installed to USER_SKILLS_DIR (defaults to <workspace>/skills/).

Supported installation sources:
- Local directory paths
- GitHub URLs (https://github.com/owner/repo or .../tree/branch/path)
- GitHub shorthand (owner/repo@skill-name)

Usage:
    from EvoScientist.tools.skills_manager import install_skill, list_skills, uninstall_skill

    # Install from local path
    install_skill("./my-skill")

    # Install from GitHub
    install_skill("https://github.com/user/repo/tree/main/my-skill")

    # List installed skills
    for skill in list_skills():
        print(skill["name"], skill["description"])

    # Uninstall a skill
    uninstall_skill("my-skill")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import paths

_logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """Information about an installed skill."""

    name: str
    description: str
    path: Path
    source: str  # "workspace", "global", or "builtin"
    tags: list[str] = field(default_factory=list)


def _normalize_tags(raw: object) -> list[str]:
    """Normalize a tags value to a list of strings."""
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _parse_skill_md(skill_md_path: Path, *, source: str = "") -> SkillInfo:
    """Parse SKILL.md frontmatter to extract name, description, and tags.

    SKILL.md format:
        ---
        name: skill-name
        description: A brief description...
        tags: [tag1, tag2]
        metadata:
          tags: [tag1, tag2]   # fallback location
        ---
        # Skill Title
        ...

    Args:
        skill_md_path: Path to the SKILL.md file.
        source: Origin label (e.g. "user", "system").

    Returns:
        SkillInfo with path set to the skill's parent directory.
    """
    parent = skill_md_path.parent
    content = skill_md_path.read_text(encoding="utf-8")

    def _info(name: str, description: str, tags: list[str] | None = None) -> SkillInfo:
        return SkillInfo(
            name=name,
            description=description,
            path=parent,
            source=source,
            tags=tags or [],
        )

    # Extract YAML frontmatter
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        # No frontmatter, use directory name
        return _info(parent.name, "(no description)")

    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1))
        if not isinstance(frontmatter, dict):
            return _info(parent.name, "(empty frontmatter)")
        # Tags: check top-level first, fall back to metadata.tags
        tags = _normalize_tags(frontmatter.get("tags"))
        if not tags:
            metadata = frontmatter.get("metadata")
            if isinstance(metadata, dict):
                tags = _normalize_tags(metadata.get("tags"))
        return _info(
            frontmatter.get("name", parent.name),
            frontmatter.get("description", "(no description)"),
            tags,
        )
    except yaml.YAMLError:
        return _info(parent.name, "(invalid frontmatter)")


def _parse_github_url(url: str) -> tuple[str, str | None, str | None]:
    """Parse a GitHub URL into (repo, ref, path).

    Supports formats:
        https://github.com/owner/repo
        https://github.com/owner/repo/tree/main/path/to/skill
        github.com/owner/repo/tree/branch/path
        owner/repo@skill-name  (shorthand from skills.sh)

    Returns:
        (repo, ref_or_none, path_or_none)
    """
    # Shorthand: owner/repo@path
    if "@" in url and "://" not in url:
        repo, path = url.split("@", 1)
        return repo.strip(), None, path.strip()

    # Strip protocol and github.com prefix
    cleaned = re.sub(r"^https?://", "", url)
    cleaned = re.sub(r"^github\.com/", "", cleaned)
    cleaned = cleaned.rstrip("/")

    # Match: owner/repo/tree/ref/path...
    m = re.match(r"^([^/]+/[^/]+)/tree/([^/]+)(?:/(.+))?$", cleaned)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Match: owner/repo (no tree)
    m = re.match(r"^([^/]+/[^/]+)$", cleaned)
    if m:
        return m.group(1), None, None

    raise ValueError(f"Cannot parse GitHub URL: {url}")


_CLONE_TIMEOUT = 120  # seconds


def _clone_repo(repo: str, ref: str | None, dest: str) -> None:
    """Shallow-clone a GitHub repo."""
    clone_url = f"https://github.com/{repo}.git"
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [clone_url, dest]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_CLONE_TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"git clone timed out after {_CLONE_TIMEOUT}s for {repo}"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")


def _is_github_url(source: str) -> bool:
    """Check if the source looks like a GitHub URL or shorthand."""
    if "github.com" in source.lower():
        return True
    if "://" in source:
        return False  # Non-GitHub URL
    # Check for owner/repo@skill shorthand
    if "@" in source and "/" in source.split("@")[0]:
        return True
    # Check for owner/repo format (but not local paths like ./foo or /foo)
    if "/" in source and not source.startswith((".", "/")):
        parts = source.split("/")
        # GitHub shorthand: exactly 2 parts, both non-empty, no extensions
        if len(parts) == 2 and all(parts) and "." not in parts[0]:
            return True
    return False


def _validate_skill_dir(path: Path) -> bool:
    """Check if a directory contains a valid skill (has SKILL.md)."""
    return (path / "SKILL.md").is_file()


def _scan_skill_dirs(root: Path) -> list[Path]:
    """Scan *root* up to 2 levels deep for directories containing SKILL.md.

    Level 1: direct children of *root*.
    Level 2: grandchildren inside non-skill child directories.

    Both levels are always scanned so mixed-depth repos are fully discovered.
    """
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _validate_skill_dir(child):
            found.append(child)
        else:
            # Non-skill directory — scan its children (level 2)
            found.extend(
                gc
                for gc in sorted(child.iterdir())
                if gc.is_dir() and _validate_skill_dir(gc)
            )
    return found


def _find_skill_in_tree(root: str, skill_name: str) -> Path | None:
    """Walk a directory tree to find a subdirectory named *skill_name* containing SKILL.md.

    Skips hidden directories (starting with '.').

    Returns:
        The absolute Path to the skill directory, or None.
    """
    for dirpath, dirnames, _files in os.walk(root):
        # Prune hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if os.path.basename(dirpath) == skill_name:
            candidate = Path(dirpath)
            if _validate_skill_dir(candidate):
                return candidate
    return None


# Allowed pattern for skill names: alphanumeric, hyphens, underscores
_VALID_SKILL_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


def _sanitize_name(name: str) -> str | None:
    """Validate and sanitize a skill name.

    Returns the cleaned name, or None if invalid.
    """
    name = name.strip()
    if not name or not _VALID_SKILL_NAME.match(name):
        return None
    # Block path traversal components
    if ".." in name or "/" in name or "\\" in name:
        return None
    return name


def install_skill(
    source: str,
    dest_dir: str | None = None,
    global_install: bool = True,
) -> dict:
    """Install a skill from a local path or GitHub URL.

    Args:
        source: Local directory path or GitHub URL/shorthand.
        dest_dir: Explicit destination directory (overrides global_install).
        global_install: If True (default), install to GLOBAL_SKILLS_DIR
            (~/.config/evoscientist/skills/). If False, install to the
            workspace-local USER_SKILLS_DIR.

    Returns:
        Dictionary with installation result:
        - success: bool
        - name: skill name (if successful)
        - path: installed path (if successful)
        - error: error message (if failed)
    """
    dest_dir = dest_dir or (
        str(paths.GLOBAL_SKILLS_DIR) if global_install else str(paths.USER_SKILLS_DIR)
    )
    os.makedirs(dest_dir, exist_ok=True)

    if _is_github_url(source):
        return _install_from_github(source, dest_dir)
    else:
        # Check if local path exists
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            # Fallback: try resolving as a virtual workspace path
            from ..paths import resolve_virtual_path

            try:
                if resolve_virtual_path(source).exists():
                    return _install_from_local(source, dest_dir)
            except Exception:
                pass

            # If not local and not a GitHub URL, try remote lookup in EvoSkills
            # This handles /install-skill skill-name shorthand
            try:
                index = fetch_remote_skill_index()
                for skill in index:
                    if skill["name"].lower() == source.lower():
                        _logger.info(
                            f"Skill '{source}' found in remote index. Installing..."
                        )
                        return _install_from_github(skill["install_source"], dest_dir)
            except Exception as e:
                _logger.warning(f"Failed to fetch remote index for fallback: {e}")

        return _install_from_local(source, dest_dir)


def _install_from_local(source: str, dest_dir: str) -> dict:
    """Install a skill from a local directory path."""
    source_path = Path(source).expanduser().resolve()

    if not source_path.exists():
        # Fallback: try resolving as a virtual workspace path
        from ..paths import resolve_virtual_path

        try:
            source_path = resolve_virtual_path(source)
        except Exception:
            pass
        if not source_path.exists():
            return {"success": False, "error": f"Path does not exist: {source}"}

    if not source_path.is_dir():
        return {"success": False, "error": f"Not a directory: {source}"}

    if not _validate_skill_dir(source_path):
        found = _scan_skill_dirs(source_path)
        if len(found) == 1:
            return _install_single_local(found[0], dest_dir)
        if found:
            return _batch_install_local(found, dest_dir)
        return {"success": False, "error": f"No SKILL.md found in: {source}"}

    return _install_single_local(source_path, dest_dir)


def _install_single_local(source_path: Path, dest_dir: str, *, ignore_fn=None) -> dict:
    """Install one skill directory into *dest_dir*."""
    skill_info = _parse_skill_md(source_path / "SKILL.md")
    skill_name = _sanitize_name(skill_info.name)
    if not skill_name:
        return {
            "success": False,
            "error": f"Invalid skill name in SKILL.md: {skill_info.name!r}",
        }

    target_path = (Path(dest_dir) / skill_name).resolve()
    if not str(target_path).startswith(str(Path(dest_dir).resolve())):
        return {
            "success": False,
            "error": f"Skill name escapes destination: {skill_info.name!r}",
        }

    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.copytree(source_path, target_path, ignore=ignore_fn)

    return {
        "success": True,
        "name": skill_name,
        "path": str(target_path),
        "description": skill_info.description,
    }


def _batch_install_local(
    skill_dirs: list[Path], dest_dir: str, *, ignore_fn=None
) -> dict:
    """Install multiple skill directories and return a batch result."""
    installed: list[dict] = []
    failed: list[dict] = []

    for sd in skill_dirs:
        result = _install_single_local(sd, dest_dir, ignore_fn=ignore_fn)
        if result["success"]:
            installed.append(result)
        else:
            failed.append({"name": sd.name, "error": result["error"]})

    return {
        "success": len(installed) > 0,
        "batch": True,
        "installed": installed,
        "failed": failed,
    }


def _install_from_github(source: str, dest_dir: str) -> dict:
    """Install a skill from a GitHub URL or shorthand."""
    try:
        repo, ref, path = _parse_github_url(source)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    with tempfile.TemporaryDirectory(prefix="evoscientist-skill-") as tmp:
        clone_dir = os.path.join(tmp, "repo")

        try:
            _clone_repo(repo, ref, clone_dir)
        except RuntimeError as e:
            return {"success": False, "error": str(e)}

        # Exclude .git from copies
        def ignore_git(dir_name: str, files: list[str]) -> list[str]:
            return [f for f in files if f == ".git"]

        # Determine the skill source directory
        if path:
            skill_source = Path(clone_dir) / path
        else:
            skill_source = Path(clone_dir)

        # Validate — if the direct path doesn't have SKILL.md, try auto-resolve
        if not skill_source.exists() or not _validate_skill_dir(skill_source):
            if skill_source.is_dir():
                found_dirs = _scan_skill_dirs(skill_source)
                if len(found_dirs) == 1:
                    skill_source = found_dirs[0]
                elif found_dirs:
                    return _batch_install_local(
                        found_dirs, dest_dir, ignore_fn=ignore_git
                    )

            # Still not resolved — try tree search by name hint
            if not _validate_skill_dir(skill_source):
                if path:
                    skill_name_hint = path.rstrip("/").rsplit("/", 1)[-1]
                    resolved = _find_skill_in_tree(clone_dir, skill_name_hint)
                    if resolved:
                        skill_source = resolved
                    else:
                        return {
                            "success": False,
                            "error": f"No SKILL.md found at '{path}' (also searched subdirectories) in: {source}",
                        }
                else:
                    return {
                        "success": False,
                        "error": f"No SKILL.md found in: {source}",
                    }

        # Single skill — install it
        result = _install_single_local(skill_source, dest_dir, ignore_fn=ignore_git)
        if result.get("success"):
            result["source"] = source
        return result


def list_skills(include_system: bool = False) -> list[SkillInfo]:
    """List all installed skills across all tiers.

    Priority order: workspace > global > builtin.
    Higher-priority skills shadow lower-priority skills with the same name.

    Args:
        include_system: If True, also include built-in (PyPI) skills.

    Returns:
        List of SkillInfo objects for each skill, deduplicated by name.
    """
    skills: list[SkillInfo] = []
    seen: set[str] = set()  # dedup by parsed skill name (not directory name)

    def _add_tier(skill_dir: Path, source: str, check_seen: bool = True) -> None:
        if not skill_dir.exists():
            return
        for entry in sorted(skill_dir.iterdir()):
            if entry.is_dir() and _validate_skill_dir(entry):
                info = _parse_skill_md(entry / "SKILL.md", source=source)
                if check_seen and info.name in seen:
                    continue
                skills.append(info)
                seen.add(info.name)

    # Tier 1: workspace-local skills (always highest priority, no dedup needed)
    _add_tier(Path(paths.USER_SKILLS_DIR), source="workspace", check_seen=False)

    # Tier 2: global skills (~/.config/evoscientist/skills/)
    _add_tier(Path(paths.GLOBAL_SKILLS_DIR), source="global")

    # Tier 3: built-in skills (optional)
    if include_system:
        from ..EvoScientist import SKILLS_DIR

        _add_tier(Path(SKILLS_DIR), source="builtin")

    return skills


def uninstall_skill(name: str) -> dict:
    """Uninstall a skill from workspace or global tier.

    Searches workspace first, then global. Built-in skills cannot be uninstalled.

    Args:
        name: Name of the skill to uninstall.

    Returns:
        Dictionary with result:
        - success: bool
        - error: error message (if failed)
    """
    # Validate name to prevent path traversal
    clean_name = _sanitize_name(name)
    if not clean_name:
        return {"success": False, "error": f"Invalid skill name: {name!r}"}

    # Search workspace tier first, then global tier
    search_dirs = [
        Path(paths.USER_SKILLS_DIR).resolve(),
        Path(paths.GLOBAL_SKILLS_DIR).resolve(),
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        target_path = (search_dir / clean_name).resolve()

        if not target_path.exists():
            # Try to find by skill name in SKILL.md (dir name may differ)
            for entry in search_dir.iterdir():
                if entry.is_dir() and _validate_skill_dir(entry):
                    info = _parse_skill_md(entry / "SKILL.md")
                    if info.name == clean_name:
                        target_path = entry.resolve()
                        break
            else:
                continue

        # Safety: resolved path must still be inside the search dir
        if not str(target_path).startswith(str(search_dir)):
            return {"success": False, "error": f"Invalid skill path: {name}"}

        shutil.rmtree(target_path)
        return {"success": True, "name": name}

    # Check if it's a built-in skill (read-only, cannot be uninstalled)
    from ..EvoScientist import SKILLS_DIR

    builtin_dir = Path(SKILLS_DIR)
    if builtin_dir.exists():
        for entry in builtin_dir.iterdir():
            if entry.is_dir() and _validate_skill_dir(entry):
                info = _parse_skill_md(entry / "SKILL.md")
                if info.name == clean_name or entry.name == clean_name:
                    return {
                        "success": False,
                        "error": f"'{name}' is a built-in skill and cannot be uninstalled.",
                    }

    return {"success": False, "error": f"Skill not found: {name}"}


def get_skill_info(name: str) -> SkillInfo | None:
    """Get information about a specific skill.

    Args:
        name: Name of the skill.

    Returns:
        SkillInfo if found, None otherwise.
    """
    for skill in list_skills(include_system=True):
        if skill.name == name:
            return skill
    return None


def list_skills_by_tag(
    tag: str,
    include_system: bool = False,
) -> list[SkillInfo]:
    """Filter installed skills by tag (case-insensitive).

    Args:
        tag: Tag to filter by.
        include_system: If True, also include system skills.

    Returns:
        List of matching SkillInfo objects.
    """
    tag_lower = tag.lower()
    return [
        s
        for s in list_skills(include_system=include_system)
        if tag_lower in [t.lower() for t in s.tags]
    ]


def get_all_tags(include_system: bool = False) -> list[tuple[str, int]]:
    """Return all tags and their counts, sorted by frequency then alphabetically.

    Args:
        include_system: If True, also include system skills.

    Returns:
        List of (tag, count) tuples sorted by count descending, then name ascending.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for skill in list_skills(include_system=include_system):
        for tag in skill.tags:
            counter[tag.lower()] += 1
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))


# ── Remote skill index ──────────────────────────────────────────────

_REMOTE_INDEX_CACHE: dict[str, tuple[float, list[dict]]] = {}
_REMOTE_INDEX_TTL = 600  # 10 minutes


def fetch_remote_skill_index(
    repo: str = "EvoScientist/EvoSkills",
    ref: str | None = None,
    path: str = "skills",
) -> list[dict]:
    """Fetch skill metadata from a GitHub repo via shallow clone.

    Clones the repo to a temp directory, scans for SKILL.md files,
    parses their frontmatter, and returns an index of available skills.
    Results are cached for 10 minutes.

    Args:
        repo: GitHub repo in owner/repo format.
        ref: Branch or tag (None for default branch).
        path: Subdirectory containing skills.

    Returns:
        List of dicts with keys: name, description, tags, install_source.
    """
    cache_key = f"{repo}:{ref or 'default'}:{path}"
    now = time.monotonic()
    cached = _REMOTE_INDEX_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _REMOTE_INDEX_TTL:
        return cached[1]

    index: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="evoscientist-browse-") as tmp:
        clone_dir = os.path.join(tmp, "repo")
        _clone_repo(repo, ref, clone_dir)

        skills_root = Path(clone_dir) / path if path else Path(clone_dir)
        if not skills_root.is_dir():
            return index

        found = _scan_skill_dirs(skills_root)
        for skill_dir in found:
            info = _parse_skill_md(skill_dir / "SKILL.md")
            # Compute relative path from clone root for install source
            rel = skill_dir.relative_to(Path(clone_dir))
            install_source = f"{repo}@{rel}"
            index.append(
                {
                    "name": info.name,
                    "description": info.description,
                    "tags": info.tags,
                    "install_source": install_source,
                }
            )

    _REMOTE_INDEX_CACHE[cache_key] = (now, index)
    return index
