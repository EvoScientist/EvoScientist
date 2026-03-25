"""@file mention parsing and injection for CLI and TUI input.

Usage::

    text, injected = resolve_file_mentions(user_input, workspace_dir)
    # text   — original input unchanged
    # injected — full prompt with file contents appended (or original if no mentions)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PATH_CHARS = r"A-Za-z0-9._~/\\:-"

FILE_MENTION_PATTERN = re.compile(r"@(?P<path>(?:\\.|[" + _PATH_CHARS + r"])+)")
"""Matches ``@path/to/file`` in user input.

Escaped spaces (``@my\\\\ folder/file``) are supported.  Bare ``@`` with no
path characters is not matched (uses ``+`` not ``*``).
"""

_EMAIL_PREFIX = re.compile(r"[a-zA-Z0-9._%+-]$")
"""If the character immediately before ``@`` matches this, it's an email address."""

# Files larger than this are referenced by path only (not embedded inline).
_MAX_EMBED_BYTES = 256 * 1024  # 256 KB

# Fuzzy search thresholds (ported from DeepAgents FuzzyFileController)
_MIN_FUZZY_SCORE = 15
_MIN_FUZZY_RATIO = 0.4

# Max files to index per workspace
_MAX_WORKSPACE_FILES = 1000


# ---------------------------------------------------------------------------
# Module-level file cache
# ---------------------------------------------------------------------------

_file_cache: dict[str, list[str]] = {}
"""workspace_dir -> sorted list of relative POSIX paths"""


def _get_workspace_files(root: Path) -> list[str]:
    """Glob workspace files up to 4 levels deep, skipping hidden entries."""
    files: list[str] = []
    for pattern in ["*", "*/*", "*/*/*", "*/*/*/*"]:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            # Skip any part that starts with '.'
            if any(part.startswith(".") for part in rel.parts):
                continue
            files.append(rel.as_posix())
            if len(files) >= _MAX_WORKSPACE_FILES:
                return files
    return files


def _get_cached_files(workspace_dir: str) -> list[str]:
    """Return cached file list for *workspace_dir*, scanning if necessary."""
    if workspace_dir not in _file_cache:
        _file_cache[workspace_dir] = _get_workspace_files(Path(workspace_dir))
    return _file_cache[workspace_dir]


def invalidate_file_cache(workspace_dir: str | None = None) -> None:
    """Invalidate the workspace file cache.

    Call when the workspace changes (e.g. ``/new``, ``/resume``).

    Args:
        workspace_dir: If given, invalidate only that workspace entry.
                       If ``None``, clear the entire cache.
    """
    if workspace_dir:
        _file_cache.pop(workspace_dir, None)
    else:
        _file_cache.clear()


# ---------------------------------------------------------------------------
# Fuzzy scoring (ported from DeepAgents FuzzyFileController)
# ---------------------------------------------------------------------------


def _fuzzy_score(query: str, candidate: str) -> float:
    """Score how well *query* matches *candidate* path.

    Four-level priority (higher = better match):

    1. Filename starts with query (150 base + length bonus)
    2. Filename contains query as substring (100–120)
    3. Full path contains query as substring (40–80)
    4. SequenceMatcher ratio on filename (15–30)

    Returns 0 when below ``_MIN_FUZZY_SCORE``.
    """
    q = query.lower()
    c = candidate.lower()
    filename = c.split("/")[-1]

    # Level 1: filename starts with query
    if filename.startswith(q):
        return 150 + len(q)

    # Level 2: filename contains query
    if q in filename:
        bonus = 20 if filename.startswith(q[:1]) else 0
        return 100 + bonus

    # Level 3: full path contains query
    if q in c:
        depth_bonus = max(0, 40 - candidate.count("/") * 5)
        return 40 + depth_bonus

    # Level 4: SequenceMatcher on filename
    ratio = SequenceMatcher(None, q, filename).ratio()
    if ratio >= _MIN_FUZZY_RATIO:
        return 15 + ratio * 15

    return 0


def _fuzzy_search(
    query: str,
    candidates: list[str],
    limit: int = 10,
) -> list[str]:
    """Return up to *limit* candidates from *candidates* ranked by fuzzy score.

    When *query* is empty, returns the first *limit* candidates sorted by
    depth then name (shallowest, alphabetical first).
    """
    if not query:
        # Tree order: group by top-level component, dir entry before its children,
        # root-level files sorted among top-level dirs alphabetically.
        def _tree_key(p: str) -> tuple:
            top = p.split("/")[0]  # first path component (no slash)
            is_file_entry = 0 if p.endswith("/") else 1  # dir entry sorts first
            return (top.lower(), is_file_entry, p.lower())

        return sorted(candidates, key=_tree_key)[:limit]

    scored = [
        (score, c)
        for c in candidates
        if (score := _fuzzy_score(query, c)) >= _MIN_FUZZY_SCORE
    ]
    return [c for _, c in sorted(scored, key=lambda x: -x[0])[:limit]]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str:
    """Return a Markdown snippet for embedding the file inline.

    Files larger than ``_MAX_EMBED_BYTES`` get a path-only reference with a
    hint to use the ``read_file`` tool instead.
    """
    size = path.stat().st_size
    if size > _MAX_EMBED_BYTES:
        size_kb = size // 1024
        return (
            f"\n### {path.name}\n"
            f"Path: `{path}`\n"
            f"Size: {size_kb} KB (too large to embed inline — "
            "use the read_file tool to view it)"
        )
    content = path.read_text(encoding="utf-8", errors="replace")
    return f"\n### {path.name}\nPath: `{path}`\n```\n{content}\n```"


def parse_file_mentions(
    text: str,
    cwd: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """Extract resolved ``@file`` paths from *text*.

    Args:
        text: Raw user input that may contain ``@path`` mentions.
        cwd:  Base directory for resolving relative paths.  Defaults to the
              process working directory.

    Returns:
        ``(files, warnings)`` — deduplicated list of resolved, existing
        ``Path`` objects (directories excluded) in order of first appearance,
        and a list of human-readable warning strings to be displayed by the
        caller.  Callers must display the warnings themselves using the
        appropriate UI mechanism (Rich console, Textual widget, etc.).
    """
    if cwd is None:
        cwd = Path.cwd()

    workspace_root = cwd.resolve()
    files: list[Path] = []
    warnings: list[str] = []
    seen: set[Path] = set()
    for match in FILE_MENTION_PATTERN.finditer(text):
        # Skip email addresses — character immediately before @ is alphanumeric
        before = text[: match.start()]
        if before and _EMAIL_PREFIX.search(before):
            continue

        raw = match.group("path")
        clean = raw.replace("\\ ", " ")

        try:
            p = Path(clean).expanduser()
            if not p.is_absolute():
                p = cwd / p
            resolved = p.resolve()
            if not resolved.exists() or not resolved.is_file():
                warnings.append(f"@file not found: {raw}")
                continue
            # Deduplicate: skip paths already seen in this message.
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(resolved)
            # Warn when the file lives outside the workspace root — it may
            # contain sensitive content (e.g. @~/.ssh/id_rsa).
            # Checked after dedup so a repeated mention only warns once.
            try:
                resolved.relative_to(workspace_root)
            except ValueError:
                warnings.append(
                    f"@{raw} is outside the workspace "
                    f"({workspace_root}) — embedding may expose sensitive files"
                )
        except (OSError, RuntimeError) as exc:
            warnings.append(f"invalid @file path {raw!r}: {exc}")

    return files, warnings


def resolve_file_mentions(
    text: str,
    workspace_dir: str | None = None,
) -> tuple[str, str, list[str]]:
    """Parse ``@file`` mentions and return *(original_text, final_prompt, warnings)*.

    *final_prompt* equals *original_text* when no valid mentions are found,
    otherwise it appends a ``## Referenced Files`` section with the file
    contents embedded as fenced code blocks.

    Args:
        text:          Raw user input.
        workspace_dir: Workspace root used for resolving relative paths.

    Returns:
        ``(original_text, final_prompt, warnings)`` — the first element is
        always the unchanged input; the second is the prompt to send to the
        agent; the third is a list of warning strings to display to the user.
    """
    cwd = Path(workspace_dir) if workspace_dir else None
    files, warnings = parse_file_mentions(text, cwd=cwd)

    if not files:
        return text, text, warnings

    parts = [text, "\n\n## Referenced Files\n"]
    for path in files:
        try:
            parts.append(_read_file(path))
        except (OSError, UnicodeDecodeError) as exc:
            parts.append(f"\n### {path.name}\n[Error reading file: {exc}]")

    return text, "\n".join(parts), warnings


# ---------------------------------------------------------------------------
# Autocomplete helpers (used by CLI completer and TUI)
# ---------------------------------------------------------------------------


def _type_hint(rel_path: str) -> str:
    """Return a short type label for *rel_path* (extension or ``'file'``)."""
    suffix = rel_path.rsplit(".", 1)[-1] if "." in rel_path.split("/")[-1] else ""
    return suffix or "file"


def complete_file_mention(
    text: str,
    workspace_dir: str | None = None,
) -> list[tuple[str, str]]:
    """Return candidate file paths for the ``@`` prefix at the end of *text*.

    Scans the workspace (up to 4 levels deep) and returns fuzzy-matched
    file/dir names relative to *workspace_dir* (or cwd).  Returns ``[]``
    when *text* does not end with an ``@``-started token.

    Args:
        text:          Current input text (up to cursor position).
        workspace_dir: Root directory to scan for completions.

    Returns:
        List of ``(completion_string, type_hint)`` tuples, e.g.
        ``[("@results/v2.json", "json"), ("@README.md", "md")]``.
        Directories have a trailing ``/`` and type hint ``"dir"``.
    """
    # Find the last @token
    match = re.search(r"@([^\s]*)$", text)
    if not match:
        return []

    partial = match.group(1).replace("\\ ", " ")
    base_str = workspace_dir or str(Path.cwd())
    base = Path(base_str)

    # If partial contains a path separator, check for subdirectory listing
    if partial.endswith("/"):
        # List directory contents
        sub = (base / partial.rstrip("/")).resolve()
        if not sub.is_dir():
            return []
        candidates_raw: list[str] = []
        try:
            for entry in sorted(sub.iterdir()):
                if entry.name.startswith("."):
                    continue
                rel = entry.relative_to(base)
                suffix = "/" if entry.is_dir() else ""
                candidates_raw.append(rel.as_posix() + suffix)
        except OSError:
            return []
        return [
            (f"@{r}", "dir" if r.endswith("/") else _type_hint(r))
            for r in candidates_raw[:10]
        ]

    # Fuzzy search over cached workspace files
    all_files = _get_cached_files(base_str)

    # Also add top-level directories (for dir completion)
    dir_candidates: list[str] = []
    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                dir_candidates.append(entry.name + "/")
    except OSError:
        pass

    combined = all_files + dir_candidates

    # Determine query: if partial has a slash, search within that subtree
    if "/" in partial:
        # Filter candidates to those starting with the directory prefix
        dir_prefix = partial.rsplit("/", 1)[0] + "/"
        file_query = partial.rsplit("/", 1)[1]
        subtree = [c for c in combined if c.startswith(dir_prefix)]
        results = _fuzzy_search(file_query, subtree)
    else:
        results = _fuzzy_search(partial, combined)

    return [(f"@{r}", "dir" if r.endswith("/") else _type_hint(r)) for r in results]
