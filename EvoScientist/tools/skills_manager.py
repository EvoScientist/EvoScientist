"""Skill installation and management for EvoScientist.

This module provides functions for installing, listing, and uninstalling user skills.
Skills are installed to GLOBAL_SKILLS_DIR by default (~/.evoscientist/skills/).
Pass global_install=False to install to the workspace skills tier (<workspace>/skills/) instead.

Supported installation sources:
- Local directory paths
- GitHub URLs (https://github.com/owner/repo or .../tree/branch/path)
- GitHub shorthand (owner/repo@skill-name)

Usage:
    from EvoScientist.paths import Workspace
    from EvoScientist.tools.skills_manager import install_skill, list_skills, uninstall_skill

    ws = Workspace("~/my-project")

    # Install from local path (global by default)
    install_skill("./my-skill", workspace=ws)

    # Install to workspace only
    install_skill("./my-skill", global_install=False, workspace=ws)

    # Install from GitHub
    install_skill("https://github.com/user/repo/tree/main/my-skill", workspace=ws)

    # List installed skills (source: "workspace", "global", or "builtin")
    for skill in list_skills(workspace=ws):
        print(skill.name, skill.source, skill.description)

    # Uninstall a skill
    uninstall_skill("my-skill", workspace=ws)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from .. import paths
from ..git_cli import GitNotFoundError, run_git
from ..git_cli import clone_repo as _clone_repo
from ..paths import Workspace, resolve_virtual_path

_logger = logging.getLogger(__name__)


# --- skills-changed publish primitive ---------------------------------------
#
# ``install_skill`` and ``uninstall_skill`` are called from six places today:
# three slash commands, two paths in the agent's ``skill_manager`` @tool, and
# the onboarding step. Consumers that cache skill data (currently the
# /expert completion popup) need to invalidate on every mutation, and wiring
# that at each caller site was the "vibes and hope" pattern reviewers rightly
# called out on PR #371 — the next caller silently reintroduces staleness.
#
# The invariant lives here instead: any subscriber registered via
# ``register_skills_changed_callback`` is fired after every install / uninstall
# call, so no caller has to remember.

_skills_changed_callbacks: list[Callable[[], None]] = []


def register_skills_changed_callback(callback: Callable[[], None]) -> None:
    """Register a callback fired after every ``install_skill`` /
    ``uninstall_skill`` call.

    Consumers that cache skill data — e.g. the ``/expert`` completion
    popup — subscribe once at import time so every mutation path
    (commands, agent ``skill_manager`` @tool, onboarding) invalidates
    their view. Callbacks are process-scoped and never unregistered
    automatically; tests should call ``_reset_skills_changed_callbacks``
    from a fixture to isolate state.
    """
    _skills_changed_callbacks.append(callback)


def _reset_skills_changed_callbacks() -> None:
    """Test hook: clear the callback list."""
    _skills_changed_callbacks.clear()


# (fingerprint, {name: description}) per workspace for slash completion; rebuilt
# when the skills directories change on disk, and cleared on every notify.
_skill_index_cache: dict[str, tuple[tuple, dict[str, str]]] = {}


def _notify_skills_changed() -> None:
    """Drop the cached skill index and fire every registered callback.

    Exceptions are logged but swallowed so a misbehaving subscriber can't
    break the install/uninstall return path.
    """
    _skill_index_cache.clear()
    for cb in _skills_changed_callbacks:
        try:
            cb()
        except Exception:
            _logger.warning(
                "skills-changed callback %r raised; continuing.",
                getattr(cb, "__qualname__", cb),
                exc_info=True,
            )


@dataclass
class SkillInfo:
    """Information about an installed skill."""

    name: str
    description: str
    path: Path
    source: str  # "workspace", "global", or "builtin"
    tags: list[str] = field(default_factory=list)
    # Expert-skill fields for the v1 agent-teams feature. All default
    # to utility-skill values so existing skills stay unchanged and their
    # SKILL.md files need no edits.
    type: str = "utility"  # "utility" (default) or "expert"
    role: str = ""  # one-line role summary shown to the orchestrator LLM
    byline: str = ""  # WebUI gallery byline
    capability_tags: list[str] = field(default_factory=list)  # WebUI chips
    avatar_hint: str = ""  # WebUI icon hint
    # SKILL.md body (post-frontmatter). Populated by ``_parse_skill_md`` so
    # the expert-container factory doesn't have to re-read the file on every
    # main-agent construction. Empty for skills built by hand or when the
    # source SKILL.md has no body content.
    body: str = ""
    # How this skill came to be classified as an expert:
    #   "expert_md"   — a sibling EXPERT.md is present (current contract)
    #   "frontmatter" — legacy ``type: expert`` in SKILL.md (deprecated)
    #   ""            — not an expert
    expert_source: str = ""
    # EXPERT.md body — the actor definition (persona + result envelope) that
    # becomes the expert's runtime prompt. Empty for non-experts and for
    # experts still on the legacy frontmatter path, whose runtime prompt is
    # still sourced from ``body``.
    expert_body: str = ""


# Shared frontmatter split regex. Captures three parts:
#   1. leading fence (``^---\s*\n``)
#   2. frontmatter YAML (``.*?``)
#   3. trailing fence + optional newline (``\n---\s*\n?``)
# Used by both ``_parse_skill_md`` (needs the YAML) and the expert-container
# factory (needs the body); a single source of truth for "what is the
# frontmatter block" so schema extensions don't drift across call sites.
_FRONTMATTER_SPLIT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _split_frontmatter_and_body(content: str) -> tuple[str, str]:
    """Return ``(frontmatter_yaml, body)`` for a SKILL.md text.

    ``frontmatter_yaml`` is the raw YAML block between the ``---`` fences
    (still a string, not parsed). ``body`` is the post-fence content with
    leading whitespace stripped. When there is no valid frontmatter block,
    ``frontmatter_yaml`` is ``""`` and the whole content becomes the body —
    matches the legacy expert-container behaviour of passing plain-body
    files through unchanged.
    """
    match = _FRONTMATTER_SPLIT_RE.match(content)
    if not match:
        return "", content.lstrip()
    body = content[match.end() :].lstrip()
    return match.group(1), body


# --- expert declaration -----------------------------------------------------
#
# A skill directory declares itself an expert by containing this file. The
# rule the skill system already lives by, applied one level down: a directory
# is a package, a contract file is a capability, presence is the declaration,
# content is the contract. SKILL.md's presence makes a directory a skill;
# EXPERT.md's presence makes it an actor.
#
# The file carries no frontmatter — the directory name is the expert's
# identity, its presence is the declaration, and its body (persona + result
# envelope) is the runtime prompt. Nothing about it is machine-read beyond
# "does it exist" and "what does it say".
#
# ``metadata.type: [skill, expert]`` in SKILL.md frontmatter is deliberately
# NOT consulted here. That field is a projection of this declaration for
# index-facing consumers (the remote skill index, gallery chips) which can't
# cheaply stat the directory; the runtime resolves presence directly, so
# reading it would create a second classifier that can drift from the file.
_EXPERT_FILENAME = "EXPERT.md"

# Legacy ``type: expert`` frontmatter is a deprecated fallback, so the warning
# is one-shot per skill name — ``_parse_skill_md`` runs on every ``list_skills``
# call (main-agent construction, /expert completion, GET /api/teams), and a
# per-parse warning would bury real diagnostics under repeats.
_legacy_expert_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    """Emit *message* at WARNING the first time *key* is seen this process."""
    if key in _legacy_expert_warned:
        return
    _legacy_expert_warned.add(key)
    _logger.warning(message, *args)


def _read_expert_md(skill_dir: Path) -> str | None:
    """Return the EXPERT.md body for *skill_dir*, or None when absent.

    Returns a string (possibly empty) whenever the file exists, so callers
    can distinguish "not an expert" (None) from "declared expert with an
    empty actor definition" (""). The latter is a skill-authoring bug the
    container paths refuse loudly rather than papering over.

    Frontmatter is stripped if present. The contract says EXPERT.md carries
    none, but an author who adds some should not have raw YAML leak into the
    expert's system prompt.
    """
    expert_path = skill_dir / _EXPERT_FILENAME
    try:
        if not expert_path.is_file():
            return None
        content = expert_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # Presence was the declaration, so an unreadable EXPERT.md is a
        # broken expert, not a plain skill. Return "" to keep the expert
        # classification and let the empty-body guards in the container
        # factories refuse it with a skill-name-bearing warning.
        _logger.warning(
            "Skill %r: could not read %s (%s); treating actor definition as empty.",
            skill_dir.name,
            expert_path,
            exc,
        )
        return ""
    _, body = _split_frontmatter_and_body(content)
    return body


def _normalize_tags(raw: object) -> list[str]:
    """Normalize a tags value to a list of strings."""
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


_MANIFEST_FILENAME = ".installed.yaml"

# Per-tier sidecar at ``<dest_dir>/.installed.yaml``. Each entry records the
# install source plus optional provenance (the upstream commit SHA at install
# time, omitted for non-git sources). Schema:
#
#     <skill-dir-name>:
#       source: <URL | shorthand | local path>
#       commit: <git SHA>
_ManifestEntry = dict[str, str]


def _manifest_path(dest_dir: str | Path) -> Path:
    return Path(dest_dir) / _MANIFEST_FILENAME


def _load_manifest(dest_dir: str | Path) -> dict[str, _ManifestEntry]:
    """Read the install manifest from *dest_dir*. Malformed entries are
    skipped, and any read error returns ``{}`` so a corrupt file never breaks
    installation or detection.
    """
    path = _manifest_path(dest_dir)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _logger.warning("Failed to read skills manifest at %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, _ManifestEntry] = {}
    for k, v in data.items():
        if not isinstance(v, dict) or not isinstance(v.get("source"), str):
            continue
        entry: _ManifestEntry = {"source": v["source"]}
        commit = v.get("commit")
        if isinstance(commit, str) and commit:
            entry["commit"] = commit
        out[str(k)] = entry
    return out


def _save_manifest(dest_dir: str | Path, manifest: dict[str, _ManifestEntry]) -> None:
    path = _manifest_path(dest_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _logger.warning("Failed to update skills manifest at %s: %s", path, exc)
        return

    # Atomic write: stage to a sibling temp file, fsync, then os.replace into
    # place. Prevents a half-written manifest from breaking detection if the
    # process dies mid-write.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        _logger.warning("Failed to update skills manifest at %s: %s", path, exc)
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _record_install(
    dest_dir: str | Path,
    name: str,
    source: str | None,
    *,
    commit: str | None = None,
) -> None:
    """Add or update a manifest entry. No-op when *source* is empty.

    *commit*, when provided, is recorded alongside the source so onboarding
    can detect when an upstream version has moved past what's installed.
    """
    if not source:
        return
    manifest = _load_manifest(dest_dir)
    new_entry: _ManifestEntry = {"source": source}
    if commit:
        new_entry["commit"] = commit
    if manifest.get(name) == new_entry:
        return
    manifest[name] = new_entry
    _save_manifest(dest_dir, manifest)


def _record_uninstall(dest_dir: str | Path, name: str) -> None:
    manifest = _load_manifest(dest_dir)
    if name in manifest:
        del manifest[name]
        _save_manifest(dest_dir, manifest)


def installed_provenance(*, workspace: Workspace) -> dict[str, dict[str, str | None]]:
    """Per-source provenance for currently-installed skills.

    Maps source URL/shorthand → ``{"commit": <sha or None>}``. When several
    children share a source (a pack), the first observed commit is returned —
    packs install all children from a single clone so this is unambiguous in
    practice. Used by onboarding to surface "update available" when the
    recorded commit no longer matches upstream.
    """
    out: dict[str, dict[str, str | None]] = {}
    for dest_dir in (workspace.skills_dir, paths.GLOBAL_SKILLS_DIR):
        dest = Path(dest_dir)
        if not dest.exists():
            continue
        for name, entry in _load_manifest(dest).items():
            if not (dest / name).is_dir():
                continue
            source = entry["source"]
            if source in out:
                continue
            out[source] = {"commit": entry.get("commit")}
    return out


def _parse_skill_md(skill_md_path: Path, *, source: str = "") -> SkillInfo:
    """Parse a skill directory into a :class:`SkillInfo`.

    Reads name / description / tags from SKILL.md frontmatter, and resolves
    whether the skill is also an **expert** — an actor with a persona and a
    result-envelope contract, dispatchable as its own agent.

    Expert classification is structural: a sibling ``EXPERT.md`` in the same
    directory declares the skill an expert, and that file's body is the
    expert's runtime prompt. SKILL.md stays pure knowledge — portable to any
    harness, and readable in-turn off the ``/skills/`` mount whether or not
    the skill also acts::

        my-expert/
          SKILL.md     # knowledge: when to use, workflow, references
          EXPERT.md    # actor: persona + result envelope  <- the declaration
          scripts/
          references/

    Skills predating that convention declare ``type: expert`` in SKILL.md
    frontmatter, alongside actor fields (``role`` / ``byline`` /
    ``capability_tags`` / ``avatar_hint`` / ``default_dispatch``). That path
    still resolves, with a deprecation warning, and is still the source of
    the WebUI gallery's decoration fields. New experts declare via EXPERT.md
    and set none of those fields; their gallery cards carry name and
    description only.

    ``default_dispatch`` is read from neither contract. Every expert is
    reachable both in-turn and in the background, and the orchestrator picks
    per task — a skill does not declare one dispatch shape for all time.

    When both are present, EXPERT.md wins on every axis (prompt source,
    dispatch) and the frontmatter actor fields are reported as ignored — a
    skill mid-migration must not behave as a blend of the two contracts.

    SKILL.md format::

        ---
        name: skill-name
        description: A brief description...
        allowed-tools: "read_file write_file think_tool"
        tags: [tag1, tag2]
        metadata:
          tags: [tag1, tag2]        # fallback location
          type: [skill, expert]     # index-facing only; never read here
        ---
        # Skill Title
        ...

    Args:
        skill_md_path: Path to the SKILL.md file.
        source: Origin label (e.g. "workspace", "global", "builtin").

    Returns:
        SkillInfo with path set to the skill's parent directory.
    """
    parent = skill_md_path.parent
    expert_body = _read_expert_md(parent)
    declares_actor = expert_body is not None
    try:
        content = skill_md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # ``_parse_skill_md`` runs during agent construction (via
        # ``_fold_expert_subagents`` → ``list_skills``), so an unreadable
        # SKILL.md in any tier must not abort the whole main-agent build or
        # ``GET /api/teams``. Degrade to an "(unreadable)" placeholder so
        # the entry stays visible in ``skill_manager list`` for debugging
        # but never advances into expert registration (empty body →
        # ``build_expert_subagent_specs`` skips it).
        _logger.warning(
            "Skill %r: could not read %s (%s); skipping.",
            parent.name,
            skill_md_path,
            exc,
        )
        # An unreadable SKILL.md doesn't undo an EXPERT.md declaration — the
        # actor definition is a separate file and may be perfectly intact —
        # so the expert classification is applied here too.
        return SkillInfo(
            name=parent.name,
            description="(unreadable)",
            path=parent,
            source=source,
            type="expert" if declares_actor else "utility",
            expert_source="expert_md" if declares_actor else "",
            expert_body=expert_body or "",
        )

    def _info(
        name: str,
        description: str,
        tags: list[str] | None = None,
        *,
        type_: str = "utility",
        role: str = "",
        byline: str = "",
        capability_tags: list[str] | None = None,
        avatar_hint: str = "",
        legacy_dispatch: str = "",
        body: str = "",
    ) -> SkillInfo:
        # EXPERT.md presence overrides whatever the frontmatter says about
        # actor-hood: the declaration is the file, and a skill mid-migration
        # (both present) must resolve to exactly one contract.
        if declares_actor:
            if (
                type_ == "expert"
                or role
                or byline
                or capability_tags
                or avatar_hint
                or legacy_dispatch
            ):
                _warn_once(
                    f"{parent}:mixed",
                    "Skill %r: %s is present, so the skill is an expert and its "
                    "actor definition comes from that file; legacy actor "
                    "frontmatter in SKILL.md (type / role / byline / "
                    "capability_tags / avatar_hint / default_dispatch) is "
                    "ignored and should be removed.",
                    name,
                    _EXPERT_FILENAME,
                )
            type_ = "expert"
            expert_source = "expert_md"
            # EXPERT.md is authoritative on every axis: the actor definition is
            # that file, so legacy decoration from SKILL.md frontmatter must not
            # leak onto SkillInfo. ``role`` would otherwise be prepended to the
            # EXPERT.md prompt (``_compose_system_prompt`` / the async head), and
            # a mismatched frontmatter ``name`` would desync the registry
            # identity from the skill directory.
            name = parent.name
            role = ""
            byline = ""
            capability_tags = []
            avatar_hint = ""
        elif type_ == "expert":
            _warn_once(
                f"{parent}:legacy",
                "Skill %r declares `type: expert` in SKILL.md frontmatter. That "
                "is deprecated: add a sibling %s carrying the persona and "
                "result envelope instead, and drop the actor fields from "
                "frontmatter. The frontmatter path still works for now.",
                name,
                _EXPERT_FILENAME,
            )
            expert_source = "frontmatter"
        else:
            expert_source = ""

        return SkillInfo(
            name=name,
            description=description,
            path=parent,
            source=source,
            tags=tags or [],
            type=type_,
            role=role,
            byline=byline,
            capability_tags=capability_tags or [],
            avatar_hint=avatar_hint,
            body=body,
            expert_source=expert_source,
            expert_body=expert_body or "",
        )

    frontmatter_yaml, body = _split_frontmatter_and_body(content)
    if not frontmatter_yaml:
        # No frontmatter, use directory name; still cache the body so
        # the expert-container factory doesn't have to re-read.
        return _info(parent.name, "(no description)", body=body)

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml)
        if not isinstance(frontmatter, dict):
            return _info(parent.name, "(empty frontmatter)", body=body)
        # Tags: check top-level first, fall back to metadata.tags
        tags = _normalize_tags(frontmatter.get("tags"))
        if not tags:
            metadata = frontmatter.get("metadata")
            if isinstance(metadata, dict):
                tags = _normalize_tags(metadata.get("tags"))
        # Expert-skill fields. Only trusted when explicit; unknown
        # ``type`` values fall back to "utility" so a typo doesn't
        # accidentally register a skill as an expert. Log the fallback
        # so authors can debug a "why isn't my expert appearing" case.
        raw_type = frontmatter.get("type")
        type_ = raw_type if raw_type in ("expert", "utility") else "utility"
        if raw_type is not None and raw_type != type_:
            _logger.warning(
                "Skill %r: unrecognized type %r in frontmatter; treating as utility.",
                frontmatter.get("name") or parent.name,
                raw_type,
            )
        role = frontmatter.get("role") or ""
        byline = frontmatter.get("byline") or ""
        capability_tags = _normalize_tags(frontmatter.get("capability_tags"))
        avatar_hint = frontmatter.get("avatar_hint") or ""
        # ``default_dispatch`` is no longer read: an expert is reachable both
        # in-turn and in the background, and the orchestrator picks per task
        # rather than the skill declaring one shape for all time. Still
        # detected here so the mixed-contract warning can name it among the
        # legacy actor fields an EXPERT.md skill should drop.
        legacy_dispatch = str(frontmatter.get("default_dispatch") or "")
        # ``.get("name", parent.name)`` only defaults on missing key; a
        # present-but-empty ``name:`` yields None, which would flow into
        # ``SkillInfo.name`` and slip past the ``_fold_expert_subagents``
        # collision guard. Coerce to the parent dir name in both cases.
        return _info(
            frontmatter.get("name") or parent.name,
            frontmatter.get("description") or "(no description)",
            tags,
            type_=type_,
            role=str(role),
            byline=str(byline),
            capability_tags=capability_tags,
            avatar_hint=str(avatar_hint),
            legacy_dispatch=legacy_dispatch,
            body=body,
        )
    except yaml.YAMLError as exc:
        _logger.warning(
            "Skill %r: invalid frontmatter YAML in %s (%s); using placeholder.",
            parent.name,
            skill_md_path,
            exc,
        )
        return _info(parent.name, "(invalid frontmatter)", body=body)


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


_LS_REMOTE_TIMEOUT = 5  # seconds — bounded, runs once per recommended pack


def _resolve_local_head(clone_dir: str) -> str | None:
    """Return the HEAD commit SHA of a freshly cloned working tree, or None
    when ``git rev-parse`` is unavailable or the call fails."""
    try:
        proc = run_git(["-C", clone_dir, "rev-parse", "HEAD"], timeout=10)
    except (GitNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def resolve_remote_head(
    source: str, *, timeout: float = _LS_REMOTE_TIMEOUT
) -> str | None:
    """Resolve the upstream commit SHA for *source* via ``git ls-remote``.

    Returns ``None`` for non-git sources, parse failures, network errors, or
    timeouts — callers should treat ``None`` as "unknown" and never infer
    "out of date" from it. Used by onboarding to detect upstream updates
    without re-cloning the repo.
    """
    if not _is_github_url(source):
        return None
    try:
        repo, ref, _ = _parse_github_url(source)
    except ValueError:
        return None

    target = ref or "HEAD"
    repo_url = f"https://github.com/{repo}.git"
    try:
        proc = run_git(["ls-remote", "--exit-code", repo_url, target], timeout=timeout)
    except (GitNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # Output: "<sha>\t<refname>" lines. Take the first SHA.
    for line in proc.stdout.splitlines():
        sha, _, _ = line.partition("\t")
        sha = sha.strip()
        if sha:
            return sha
    return None


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
    *,
    workspace: Workspace,
    work_dir: str | Path | None = None,
) -> dict:
    """Install a skill from a local path or GitHub URL.

    Args:
        source: Local directory path or GitHub URL/shorthand.
        dest_dir: Explicit destination directory (overrides global_install).
        global_install: If True (default), install to GLOBAL_SKILLS_DIR
            (~/.evoscientist/skills/). If False, install to the workspace
            skills tier.
        workspace: The workspace whose skills tier receives a local install.
        work_dir: Folder that virtual source paths (``/my-skill``) resolve
            against. Defaults to the workspace root.

    Returns:
        Dictionary with installation result:
        - success: bool
        - name: skill name (if successful)
        - path: installed path (if successful)
        - error: error message (if failed)
    """
    try:
        return _install_skill_impl(
            source,
            dest_dir,
            global_install,
            workspace=workspace,
            work_dir=Path(work_dir) if work_dir is not None else workspace.root,
        )
    finally:
        _notify_skills_changed()


def _install_skill_impl(
    source: str,
    dest_dir: str | None,
    global_install: bool,
    *,
    workspace: Workspace,
    work_dir: Path,
) -> dict:
    dest_dir = dest_dir or (
        str(paths.GLOBAL_SKILLS_DIR) if global_install else str(workspace.skills_dir)
    )
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as e:
        return {
            "success": False,
            "error": f"Cannot create install directory '{dest_dir}': {e}",
        }

    if _is_github_url(source):
        return _install_from_github(source, dest_dir)
    else:
        # Check if local path exists
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            # Fallback: try resolving as a virtual workspace path
            try:
                if resolve_virtual_path(work_dir, source).exists():
                    return _install_from_local(source, dest_dir, work_dir=work_dir)
            except Exception:
                pass

            # If not local and not a GitHub URL, try remote lookup in EvoSkills
            # This handles /install-skill skill-name shorthand. A path-shaped
            # source can never match an index name, so a mistyped path goes
            # straight to the local check below and keeps "Path does not exist".
            if not _looks_like_path(source):
                try:
                    index = fetch_remote_skill_index()
                    for skill in index:
                        if skill["name"].lower() == source.lower():
                            _logger.info(
                                f"Skill '{source}' found in remote index. Installing..."
                            )
                            # Record under the user-facing source (the shorthand
                            # name they typed) so detection works on re-runs.
                            return _install_from_github(
                                skill["install_source"], dest_dir, record_as=source
                            )
                except Exception as e:
                    # The source is not a local path either, so this failure
                    # (git missing, the macOS Command Line Tools stub, offline)
                    # is the real cause; "Path does not exist" would hide it.
                    return {"success": False, "error": str(e)}

        return _install_from_local(source, dest_dir, work_dir=work_dir)


def _looks_like_path(source: str) -> bool:
    """True for a filesystem-shaped source (``./x``, ``~/x``, ``a/b``), as
    opposed to a bare skill name such as ``paper-writing``."""
    return source.startswith((".", "~")) or "/" in source or os.sep in source


def _install_from_local(source: str, dest_dir: str, *, work_dir: Path) -> dict:
    """Install a skill from a local directory path."""
    source_path = Path(source).expanduser().resolve()

    if not source_path.exists():
        # Fallback: try resolving as a virtual workspace path
        try:
            source_path = resolve_virtual_path(work_dir, source)
        except Exception:
            pass
        if not source_path.exists():
            return {"success": False, "error": f"Path does not exist: {source}"}

    if not source_path.is_dir():
        return {"success": False, "error": f"Not a directory: {source}"}

    if not _validate_skill_dir(source_path):
        found = _scan_skill_dirs(source_path)
        if len(found) == 1:
            return _install_single_local(found[0], dest_dir, record_as=source)
        if found:
            return _batch_install_local(found, dest_dir, record_as=source)
        return {"success": False, "error": f"No SKILL.md found in: {source}"}

    return _install_single_local(source_path, dest_dir, record_as=source)


def _install_single_local(
    source_path: Path,
    dest_dir: str,
    *,
    ignore_fn=None,
    record_as: str | None = None,
    record_commit: str | None = None,
) -> dict:
    """Install one skill directory into *dest_dir*.

    *record_as*, when provided, is recorded in the install manifest so later
    detection can match against the user-facing source string (URL, shorthand,
    or path) — important for packs where the installed dir name doesn't
    resemble the source.

    *record_commit*, when provided, captures the upstream git SHA at install
    time so onboarding can compare against the current upstream and surface
    "update available" without re-cloning.
    """
    skill_info = _parse_skill_md(source_path / "SKILL.md")
    skill_name = _sanitize_name(skill_info.name)
    if not skill_name:
        return {
            "success": False,
            "error": f"Invalid skill name in SKILL.md: {skill_info.name!r}",
        }

    target_path = (Path(dest_dir) / skill_name).resolve()
    if not target_path.is_relative_to(Path(dest_dir).resolve()):
        return {
            "success": False,
            "error": f"Skill name escapes destination: {skill_info.name!r}",
        }

    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.copytree(source_path, target_path, ignore=ignore_fn)
    _record_install(dest_dir, skill_name, record_as, commit=record_commit)

    return {
        "success": True,
        "name": skill_name,
        "path": str(target_path),
        "description": skill_info.description,
    }


def _batch_install_local(
    skill_dirs: list[Path],
    dest_dir: str,
    *,
    ignore_fn=None,
    record_as: str | None = None,
    record_commit: str | None = None,
) -> dict:
    """Install multiple skill directories and return a batch result."""
    installed: list[dict] = []
    failed: list[dict] = []

    for sd in skill_dirs:
        result = _install_single_local(
            sd,
            dest_dir,
            ignore_fn=ignore_fn,
            record_as=record_as,
            record_commit=record_commit,
        )
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


def _install_from_github(
    source: str, dest_dir: str, *, record_as: str | None = None
) -> dict:
    """Install a skill from a GitHub URL or shorthand.

    *record_as* overrides what gets written to the install manifest. By default
    we record the same URL/shorthand the caller passed as *source* so detection
    works against the user-facing string.
    """
    record_as = record_as or source
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

        # Capture HEAD now, before any fallback paths might re-enter — every
        # child of a pack share the same upstream SHA so we record it once.
        commit = _resolve_local_head(clone_dir)

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
                        found_dirs,
                        dest_dir,
                        ignore_fn=ignore_git,
                        record_as=record_as,
                        record_commit=commit,
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
        result = _install_single_local(
            skill_source,
            dest_dir,
            ignore_fn=ignore_git,
            record_as=record_as,
            record_commit=commit,
        )
        if result.get("success"):
            result["source"] = source
        return result


def list_skills(
    include_system: bool = False, *, workspace: Workspace
) -> list[SkillInfo]:
    """List all installed skills across all tiers.

    Priority order: workspace > global > builtin.
    Higher-priority skills shadow lower-priority skills with the same name.

    Args:
        include_system: If True, also include built-in (PyPI) skills.
        workspace: The workspace whose skills tier is listed first.

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
    _add_tier(workspace.skills_dir, source="workspace", check_seen=False)

    # Tier 2: global skills (~/.evoscientist/skills/)
    _add_tier(Path(paths.GLOBAL_SKILLS_DIR), source="global")

    # Tier 3: built-in skills (optional)
    if include_system:
        from ..EvoScientist import SKILLS_DIR

        _add_tier(Path(SKILLS_DIR), source="builtin")

    return skills


def _skill_dirs_fingerprint(workspace: Workspace) -> tuple:
    """Cheap stat-only signature of the skills roots the skill index reads.

    Each SKILL.md adds its mtime and ctime, so a permission repair is a change.
    A root that cannot be listed contributes a fresh marker on every call, so
    the signature never matches and the index is rebuilt each time.
    """
    from ..EvoScientist import SKILLS_DIR

    roots = (workspace.skills_dir, Path(paths.GLOBAL_SKILLS_DIR), Path(SKILLS_DIR))
    parts: list[tuple] = []
    for root in roots:
        try:
            if not root.is_dir():
                parts.append((str(root), None))
                continue
            children = []
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                try:
                    stat = (entry / "SKILL.md").stat()
                    stamps = (stat.st_mtime_ns, stat.st_ctime_ns)
                except FileNotFoundError:
                    stamps = None
                children.append((entry.name, stamps))
            parts.append(tuple(children))
        except OSError:
            # A fresh object never compares equal, so a failing root forces a rescan.
            parts.append((str(root), object()))
    return tuple(parts)


_MAX_SKILL_MD_BYTES = 10 * 1024 * 1024

# The fence rule of the agent's skills middleware: a closing line must be exactly ``---``.
_INDEX_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _skill_name_and_description(raw: bytes) -> tuple[str, str] | None:
    """Return the frontmatter name and description, or None when the agent skips the file."""
    if len(raw) > _MAX_SKILL_MD_BYTES:
        return None
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    match = _INDEX_FRONTMATTER_RE.match(content)
    if match is None:
        return None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except (yaml.YAMLError, ValueError, RecursionError):
        # safe_load also raises these for a bad date or tag and for deep nesting.
        return None
    if not isinstance(frontmatter, dict):
        return None
    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not name or not description:
        return None
    return name, description


def _build_skill_index(workspace: Workspace) -> dict[str, str]:
    """Return ``{name: description}`` for the skills the agent can pin.

    Runs deepagents' ``ls`` + ``download_files`` calls over the agent's own
    merged skills backend.
    """
    from ..backends import MergedSkillsBackend
    from ..EvoScientist import SKILLS_DIR

    backend = MergedSkillsBackend(
        primary_dir=str(workspace.skills_dir),
        global_dir=str(paths.GLOBAL_SKILLS_DIR),
        secondary_dir=SKILLS_DIR,
    )
    skill_md_paths = [
        str(PurePosixPath(entry["path"]) / "SKILL.md")
        for entry in backend.ls("/").entries or []
        if entry.get("is_dir")
    ]
    index: dict[str, str] = {}
    # In path order, so the later folder wins a repeated name, as in the middleware.
    for response in backend.download_files(skill_md_paths):
        if response.error is None and response.content is not None:
            parsed = _skill_name_and_description(response.content)
            if parsed is not None:
                index[parsed[0]] = parsed[1]
    return index


def installed_skill_index(workspace: Workspace) -> dict[str, str]:
    """Return ``{name: description}`` for every skill the agent can load.

    Cached per workspace and rebuilt when the skills directories change.
    """
    fingerprint = _skill_dirs_fingerprint(workspace)
    cached = _skill_index_cache.get(workspace.key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    index = _build_skill_index(workspace)
    _skill_index_cache[workspace.key] = (fingerprint, index)
    return index


def pinned_skill_names(text: str, workspace: Workspace) -> list[str]:
    """Return the installed skills named by the leading ``/name`` words of *text*.

    Empty unless the first word names one, so a typo stays an unknown command.
    Scans fresh so a skill made a moment ago is never rejected.
    """
    from ..message_meta import parse_skill_slashes

    names = parse_skill_slashes(text)
    if not names:
        return []
    index = _build_skill_index(workspace)
    if names[0] not in index:
        return []
    return [name for name in names if name in index]


def list_expert_skills(
    include_system: bool = True, *, workspace: Workspace
) -> list[SkillInfo]:
    """List installed expert skills.

    Filters ``list_skills()`` output to entries classified as experts by
    ``_parse_skill_md`` — a sibling ``EXPERT.md`` (current contract), or
    legacy ``type: expert`` SKILL.md frontmatter. Read ``expert_source`` on
    the returned entries to tell the two apart; use
    ``expert_container.expert_prompt_body`` rather than ``.body`` to get an
    expert's runtime prompt, since the two contracts source it from
    different files.

    Defaults ``include_system=True`` because first-party experts ship under
    the builtin skills tier; user-installed experts still surface from the
    workspace and global tiers alongside them.

    Consumed by:
      - ``GET /api/teams`` (WebUI gallery listing).
      - Main-agent construction (folds expert skills into the
        ``subagents=[...]`` list so the ``task`` tool can dispatch to
        them for sync consult).
      - The ``skill_manager`` @tool's ``type=expert`` filter.

    Returns:
        List of ``SkillInfo`` for each expert skill.
    """
    return [
        s
        for s in list_skills(include_system=include_system, workspace=workspace)
        if s.type == "expert"
    ]


def is_expert_skill_dir(skill_dir: Path) -> bool:
    """Return True if the skill directory *skill_dir* declares an expert."""
    return _parse_skill_md(skill_dir / "SKILL.md").type == "expert"


def uninstall_skill(name: str, *, workspace: Workspace) -> dict:
    """Uninstall a skill from workspace or global tier.

    Searches workspace first, then global. Built-in skills cannot be uninstalled.

    Args:
        name: Name of the skill to uninstall.

    Returns:
        Dictionary with result:
        - success: bool
        - name: the name that was passed in (if successful)
        - expert: whether the removed directory declared an expert (if successful)
        - error: error message (if failed)
    """
    try:
        return _uninstall_skill_impl(name, workspace=workspace)
    finally:
        _notify_skills_changed()


def _uninstall_skill_impl(name: str, *, workspace: Workspace) -> dict:
    # Validate name to prevent path traversal
    clean_name = _sanitize_name(name)
    if not clean_name:
        return {"success": False, "error": f"Invalid skill name: {name!r}"}

    # Search workspace tier first, then global tier
    search_dirs = [
        workspace.skills_dir.resolve(),
        Path(paths.GLOBAL_SKILLS_DIR).resolve(),
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        target_path = (search_dir / clean_name).resolve()

        if not target_path.exists() or not _validate_skill_dir(target_path):
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
        if not target_path.is_relative_to(search_dir):
            return {"success": False, "error": f"Invalid skill path: {name}"}

        expert = is_expert_skill_dir(target_path)
        shutil.rmtree(target_path)
        _record_uninstall(search_dir, target_path.name)
        return {"success": True, "name": name, "expert": expert}

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


def get_skill_info(name: str, *, workspace: Workspace) -> SkillInfo | None:
    """Get information about a specific skill.

    Args:
        name: Name of the skill.

    Returns:
        SkillInfo if found, None otherwise.
    """
    for skill in list_skills(include_system=True, workspace=workspace):
        if skill.name == name:
            return skill
    return None


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
