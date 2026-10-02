"""Diff formatting for edit_file tool results.

Constructs unified diffs from old_string/new_string tool args and renders
them with Rich markup (color-coded lines, line numbers, gutter bars).
Works in both Rich CLI and Textual TUI (both render Rich markup natively).

Ported from upstream deepagents-cli widgets/diff.py + config.py.
"""

from __future__ import annotations

import difflib
import os
import re
import sys

# ---------------------------------------------------------------------------
# Charset detection (simplified from upstream config.py)
# ---------------------------------------------------------------------------


def _detect_unicode_support() -> bool:
    """Check if the terminal supports Unicode glyphs."""
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return True
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    return "utf" in lang.lower()


# Module-level glyph constants
_UNICODE = _detect_unicode_support()
GUTTER_BAR = "\u258c" if _UNICODE else "|"  # ▌ or |
BOX_VERTICAL = "\u2502" if _UNICODE else "|"  # │ or |
BOX_DOUBLE_HORIZ = "\u2550" if _UNICODE else "="  # ═ or =


# ---------------------------------------------------------------------------
# Markup escaping
# ---------------------------------------------------------------------------


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text.

    Prevents ``[`` and ``]`` from being interpreted as Rich tags.
    """
    return text.replace("[", r"\[").replace("]", r"\]")


# ---------------------------------------------------------------------------
# Diff formatting (produces Rich markup string)
# ---------------------------------------------------------------------------


def _build_stats_text(additions: int, deletions: int) -> str:
    """Build a ``+N -M`` stats string with Rich markup."""
    parts: list[str] = []
    if additions:
        parts.append(f"[green]+{additions}[/green]")
    if deletions:
        parts.append(f"[red]-{deletions}[/red]")
    return " ".join(parts)


def format_diff_rich(
    diff: str,
    max_lines: int | None = 100,
    title: str | None = None,
) -> str:
    """Format a unified diff with line numbers and colors.

    Args:
        diff: Unified diff string.
        max_lines: Maximum number of content lines to show before truncating.
            ``None`` means unlimited.
        title: Optional title shown above the diff (e.g. file path).

    Returns:
        Rich-markup formatted diff string.
    """
    if not diff:
        return "[dim]No changes detected[/dim]"

    lines = diff.splitlines()

    # Compute stats (skip +++ / --- headers)
    additions = sum(
        1 for ln in lines if ln.startswith("+") and not ln.startswith("+++")
    )
    deletions = sum(
        1 for ln in lines if ln.startswith("-") and not ln.startswith("---")
    )

    # Find max line number for column width
    max_line = 0
    for line in lines:
        if m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            max_line = max(max_line, int(m.group(1)), int(m.group(2)))
    width = max(3, len(str(max_line + len(lines))))

    formatted: list[str] = []

    # Title header (═══ title ═══)
    h = BOX_DOUBLE_HORIZ
    if title:
        formatted.append(
            f"[bold cyan]{h}{h}{h} {_escape_markup(title)} {h}{h}{h}[/bold cyan]"
        )
        formatted.append("")

    # Stats header
    stats_text = _build_stats_text(additions, deletions)
    if stats_text:
        formatted.extend([stats_text, ""])

    old_num = new_num = 0
    line_count = 0

    for line in lines:
        if max_lines is not None and line_count >= max_lines:
            formatted.append(f"\n[dim]... ({len(lines) - line_count} more lines)[/dim]")
            break

        # Skip file headers
        if line.startswith(("---", "+++")):
            continue

        # Hunk headers — update line numbers, don't display
        if m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            old_num, new_num = int(m.group(1)), int(m.group(2))
            continue

        content = line[1:] if line else ""
        escaped = _escape_markup(content)

        if line.startswith("-"):
            gutter = f"[red bold]{GUTTER_BAR}[/red bold]"
            ln = f"[dim]{old_num:>{width}}[/dim]"
            body = f"[on #2d1515]{escaped}[/on #2d1515]"
            formatted.append(f"{gutter}{ln} {body}")
            old_num += 1
            line_count += 1
        elif line.startswith("+"):
            gutter = f"[green bold]{GUTTER_BAR}[/green bold]"
            ln = f"[dim]{new_num:>{width}}[/dim]"
            body = f"[on #152d15]{escaped}[/on #152d15]"
            formatted.append(f"{gutter}{ln} {body}")
            new_num += 1
            line_count += 1
        elif line.startswith(" "):
            formatted.append(f"[dim]{BOX_VERTICAL}{old_num:>{width}}[/dim]  {escaped}")
            old_num += 1
            new_num += 1
            line_count += 1
        elif line.strip() == "...":
            formatted.append("[dim]...[/dim]")
            line_count += 1

    # Stats footer (matches upstream EnhancedDiff)
    if stats_text:
        formatted.extend(["", stats_text])

    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# High-level helper: build diff from edit_file tool args
# ---------------------------------------------------------------------------


def build_edit_diff(
    file_path: str,
    old_string: str,
    new_string: str,
    max_lines: int | None = None,
) -> str | None:
    """Construct a formatted diff from edit_file tool arguments.

    Args:
        file_path: Path shown in the diff header.
        old_string: Original text that was replaced.
        new_string: Replacement text.
        max_lines: Max content lines before truncation.  ``None`` (default)
            means no truncation — callers handle collapsing if needed.

    Returns:
        Rich-markup formatted diff, or ``None`` if inputs are equal/empty.
    """
    if old_string == new_string:
        return None
    if not old_string and not new_string:
        return None

    diff_lines = list(
        difflib.unified_diff(
            old_string.splitlines(),
            new_string.splitlines(),
            fromfile=file_path,
            tofile=file_path,
            lineterm="",
            n=3,
        )
    )

    if not diff_lines:
        return None

    diff_text = "\n".join(diff_lines)
    return format_diff_rich(diff_text, max_lines=max_lines, title=file_path)
