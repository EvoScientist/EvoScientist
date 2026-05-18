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

FILE_MENTION_PATTERN = re.compile(
    r"@(?:"
    r'"(?P<dquoted>[^"\n]+)"'
    r"|'(?P<squoted>[^'\n]+)'"
    r"|(?P<bare>(?:\\.|[" + _PATH_CHARS + r"])+)"
    r")"
)
"""Matches ``@path/to/file`` in user input.

Three forms are supported, in priority order:

1. ``@"path with spaces.pdf"`` — explicit double-quoted path
2. ``@'path with spaces.pdf'`` — explicit single-quoted path
3. ``@bare/path`` — backslash-escaped spaces (``@my\\\\ folder/file``) work;
   raw unescaped spaces are handled via greedy expansion in
   :func:`parse_file_mentions`.

Bare ``@`` with no path characters is not matched (uses ``+`` not ``*``).
"""

_EMAIL_PREFIX = re.compile(r"[a-zA-Z0-9._%+-]$")
"""If the character immediately before ``@`` matches this, it's an email address."""

# Hard cap on tokens consumed during greedy expansion across whitespace.
_GREEDY_MAX_TOKENS = 20

# Trailing punctuation stripped before checking if a greedy candidate exists.
_GREEDY_TRAIL_PUNCT = ",;:!?)]}>"

# Files larger than this are referenced by path only (not embedded inline).
_MAX_EMBED_BYTES = 256 * 1024  # 256 KB

# Bytes to sample for binary detection (null byte check).
_BINARY_PROBE_BYTES = 8192

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
    # Binary detection: sample first bytes for null byte (covers all formats).
    with open(path, "rb") as fh:
        if b"\x00" in fh.read(_BINARY_PROBE_BYTES):
            return (
                f"\n### {path.name}\n"
                f"Path: `{path}`\n"
                "(binary file — use the read_file tool to view it)"
            )
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


def _resolve_path(raw: str, cwd: Path) -> Path | None:
    """Resolve *raw* to an existing file path, or ``None``.

    Honors backslash-escaped spaces and ``~`` expansion.  Returns ``None``
    when the path does not exist, is not a regular file, or raises
    ``OSError``/``RuntimeError`` during resolution.
    """
    clean = raw.replace("\\ ", " ")
    try:
        p = Path(clean).expanduser()
        if not p.is_absolute():
            p = cwd / p
        resolved = p.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved.is_file():
        return resolved
    return None


def _greedy_extend(
    text: str,
    raw: str,
    match_end: int,
    cwd: Path,
) -> tuple[str, Path, int] | None:
    """Try to extend *raw* across whitespace until the path resolves.

    Walks the text after *match_end*, capped at the next newline or the
    start of another ``@`` mention.  Tries the longest plausible suffix
    first and shrinks one token at a time, stripping trailing punctuation
    that is unlikely to be part of a filename.

    Returns ``(extended_raw, resolved_file, new_end_pos)`` on success,
    else ``None``.
    """
    rest = text[match_end:]

    # Hard boundaries that should never be crossed.
    boundary = len(rest)
    nl = rest.find("\n")
    if nl >= 0:
        boundary = nl
    next_at = re.search(r"\s@", rest[:boundary])
    if next_at:
        boundary = next_at.start()

    region = rest[:boundary]
    if not region or not region[0].isspace():
        return None

    tokens = list(re.finditer(r"\S+", region))
    if not tokens:
        return None

    # Try longest-first so we prefer the most specific match.
    for i in range(min(len(tokens), _GREEDY_MAX_TOKENS), 0, -1):
        end = tokens[i - 1].end()
        suffix = region[:end].rstrip(_GREEDY_TRAIL_PUNCT)
        if not suffix:
            continue
        candidate = raw + suffix
        resolved = _resolve_path(candidate, cwd)
        if resolved is not None:
            return candidate, resolved, match_end + len(suffix)

    return None


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
    # finditer would normally re-scan from each match's end, but greedy
    # expansion can consume bytes past that point.  Track a manual cursor
    # and skip matches that start before it.
    cursor = 0
    for match in FILE_MENTION_PATTERN.finditer(text):
        if match.start() < cursor:
            continue
        # Skip email addresses — character immediately before @ is alphanumeric
        before = text[: match.start()]
        if before and _EMAIL_PREFIX.search(before):
            continue

        dquoted = match.group("dquoted")
        squoted = match.group("squoted")
        bare = match.group("bare")
        quoted_raw = dquoted if dquoted is not None else squoted
        raw = quoted_raw if quoted_raw is not None else bare
        is_quoted = quoted_raw is not None

        resolved = _resolve_path(raw, cwd)
        end_pos = match.end()

        if resolved is None and not is_quoted:
            extended = _greedy_extend(text, raw, match.end(), cwd)
            if extended is not None:
                raw, resolved, end_pos = extended

        cursor = end_pos

        if resolved is None:
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


def _format_mention(rel_path: str) -> str:
    """Render *rel_path* as an ``@`` mention, quoting if it contains spaces."""
    if " " in rel_path:
        return f'@"{rel_path}"'
    return f"@{rel_path}"


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
        Paths containing spaces are returned in double-quoted form,
        e.g. ``@"my docs/file.pdf"``.
    """
    # Find the last @token.  Allow whitespace inside a quoted partial so
    # completion keeps working as the user types ``@"PRE`` → ``@"PREPING_ B``.
    quoted_match = re.search(r'@"([^"\n]*)$', text)
    if quoted_match:
        partial = quoted_match.group(1)
        quoted = True
    else:
        match = re.search(r"@([^\s\"']*)$", text)
        if not match:
            return []
        partial = match.group(1).replace("\\ ", " ")
        quoted = False

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
            (_format_mention(r), "dir" if r.endswith("/") else _type_hint(r))
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

    if quoted:
        # User opened a quoted mention — close it for them.
        return [
            (f'@"{r}"', "dir" if r.endswith("/") else _type_hint(r)) for r in results
        ]
    return [
        (_format_mention(r), "dir" if r.endswith("/") else _type_hint(r))
        for r in results
    ]
