"""Regression guard against ad-hoc event-loop bridges (audit finding #8).

The owned async runtime (``EvoScientist.runtime``) is the *one* place agent
coroutines run. Every other module must reach it through ``submit`` /
``run_sync`` / ``spawn`` — never by spinning its own loop, calling
``asyncio.run``, or applying ``nest_asyncio``. This test greps the package for
those patterns and fails on any occurrence outside:

* ``runtime.py`` — owns the sole loop and the shutdown path;
* ``_winloop.py`` — the Windows policy helper (until folded into ``start()``);
* the explicit ``ALLOWLIST`` below — every *current* bridge site, enumerated at
  the time this guard landed. The allowlist may only **shrink** over time: as
  each migration stage (design §8) converts a site, its entry is removed here.
  The test also asserts every allowlisted file still contains a match, so a
  fully-converted file cannot be left stale in the list.

This is what keeps "bridge #16-through-#20" from ever being written.
"""

import re
from pathlib import Path

import pytest

# Matches the four ad-hoc-loop patterns the design's §7 guard specifies.
_PATTERN = re.compile(
    r"nest_asyncio|new_event_loop\(|asyncio\.run\(|run_until_complete\("
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "EvoScientist"

# Files that structurally own loop machinery — never counted as bridges.
_EXEMPT = frozenset(
    {
        "runtime.py",
        "_winloop.py",
    }
)

# Every CURRENT ad-hoc-bridge site, as of stage 1. Paths are POSIX-relative to
# the ``EvoScientist/`` package root. This list may only shrink: converting a
# site (stages 2-6) means deleting its entry here, and stage 6 empties it.
ALLOWLIST = frozenset(
    {
        "channels/base.py",
        "channels/feishu/channel.py",
        "channels/standalone.py",
        "channels/wechat/serve.py",
        "cli/channel.py",
        "cli/commands.py",
        "cli/interactive.py",
        "cli/tui_interactive.py",
        "config/onboard/channels.py",
        "mcp/client.py",
        "middleware/model_fallback.py",
        "stream/display.py",
    }
)


def _rel(path: Path) -> str:
    return path.relative_to(_PACKAGE_ROOT).as_posix()


def _matching_files() -> dict[str, list[str]]:
    """Map each package .py file with a match to its ``line_no: text`` hits."""
    hits: dict[str, list[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        found = [
            f"{n}: {line.rstrip()}"
            for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if _PATTERN.search(line)
        ]
        if found:
            hits[_rel(path)] = found
    return hits


def test_no_adhoc_event_loops_outside_allowlist():
    """No new ad-hoc loop bridges may appear outside the exempt/allowlisted set."""
    allowed = _EXEMPT | ALLOWLIST
    offenders = {
        rel: lines
        for rel, lines in _matching_files().items()
        if rel.rsplit("/", 1)[-1] not in _EXEMPT and rel not in allowed
    }
    assert not offenders, (
        "Ad-hoc event-loop bridge found outside runtime.py/_winloop.py and the "
        "allowlist. Route it through EvoScientist.runtime (submit/run_sync/"
        f"spawn) instead:\n{offenders}"
    )


@pytest.mark.parametrize("entry", sorted(ALLOWLIST))
def test_allowlist_entries_are_not_stale(entry):
    """Every allowlisted file must still contain a bridge pattern.

    When a migration stage converts a site, this fails until the now-clean
    file's entry is pruned from ``ALLOWLIST`` — enforcing monotonic shrink.
    """
    path = _PACKAGE_ROOT / entry
    assert path.exists(), f"allowlisted file no longer exists: {entry}"
    assert _PATTERN.search(path.read_text(encoding="utf-8")), (
        f"'{entry}' is allowlisted but no longer contains an ad-hoc loop "
        "pattern — remove it from ALLOWLIST."
    )
