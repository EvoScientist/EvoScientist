"""The uv tool environment EvoScientist runs from, read from uv's receipt.

``uv tool install`` records what it installed in ``<env>/uv-receipt.toml``:
the tool's own requirement (with its extras, or the directory, git or URL it
came from), every ``--with`` requirement, the constraints and the requested
Python. A reinstall that leaves any of them out drops it, so the MCP
installer's ``--with`` reinstall and ``EvoSci update`` both rebuild their
``uv tool install`` from this file.

The receipt belongs to uv; it is read defensively and an entry this module
cannot turn back into a requirement raises :class:`ReceiptError` instead of
being guessed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

RECEIPT_NAME = "uv-receipt.toml"
# The package's own name as uv writes it in the receipt (normalized).
PACKAGE = "evoscientist"
# Requirement keys that name where a package comes from instead of the index.
SOURCE_KEYS = ("directory", "git", "url", "path", "editable", "virtual")


class ReceiptError(Exception):
    """The receipt is missing, unreadable, or has an entry that cannot be
    turned back into a requirement."""


def normalize(name: str) -> str:
    """A package name in PEP 503 form, as uv writes it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def tool_env() -> Path | None:
    """The uv tool environment this interpreter runs in, or None.

    ``sys.prefix`` holds the receipt in a uv tool environment. ``VIRTUAL_ENV``
    cannot tell: the ``EvoSci`` entry point does not set it.
    """
    prefix = Path(sys.prefix)
    return prefix if (prefix / RECEIPT_NAME).is_file() else None


def _git_requirement(url: str) -> str:
    """``git+<repo>@<ref>#subdirectory=...`` from the receipt's git URL, which
    keeps the ref and the subdirectory in the query and the commit in the
    fragment."""
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    if "path" in query:
        raise ReceiptError(f"git source with a path inside the repository: {url}")
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    ref = next(
        (query[key][0] for key in ("branch", "tag", "rev") if key in query),
        parts.fragment,
    )
    spec = f"git+{base}" + (f"@{ref}" if ref else "")
    if "subdirectory" in query:
        spec += f"#subdirectory={query['subdirectory'][0]}"
    return spec


def requirement_args(entry: dict[str, Any]) -> list[str]:
    """The ``uv tool install`` arguments that add *entry* again: ``--with
    <requirement>`` (name, extras, version or source, marker), or
    ``--with-editable <path>``. Raises :class:`ReceiptError` for a source
    that cannot be written as a requirement."""
    return [
        "--with-editable" if "editable" in entry else "--with",
        requirement_string(entry),
    ]


def requirement_string(entry: dict[str, Any]) -> str:
    """*entry* as a PEP 508 requirement (an editable entry: its path)."""
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ReceiptError(f"requirement without a name: {entry!r}")
    if "editable" in entry:
        return str(entry["editable"])
    extras = entry.get("extras") or []
    head = name + (f"[{','.join(extras)}]" if extras else "")
    marker = entry.get("marker")
    tail = f" ; {marker}" if marker else ""
    if "git" in entry:
        return f"{head} @ {_git_requirement(str(entry['git']))}{tail}"
    if "url" in entry:
        url = str(entry["url"])
        if entry.get("subdirectory"):
            url += f"#subdirectory={entry['subdirectory']}"
        return f"{head} @ {url}{tail}"
    for key in ("path", "directory"):
        if key in entry:
            return f"{head} @ {Path(str(entry[key])).as_uri()}{tail}"
    if "virtual" in entry:
        raise ReceiptError(f"virtual source for {name}")
    return f"{head}{entry.get('specifier', '')}{tail}"


@dataclass(frozen=True)
class Receipt:
    """What ``uv tool install`` recorded for EvoScientist's environment."""

    path: Path
    own: dict[str, Any]  # the evoscientist requirement
    others: tuple[dict[str, Any], ...]  # every other requirement (--with)
    constraints: tuple[dict[str, Any], ...]
    python: str | None  # the requested Python, absent without --python

    @property
    def extras(self) -> list[str]:
        return list(self.own.get("extras") or [])

    @property
    def source(self) -> str | None:
        """Where EvoScientist itself came from when not from the index: the
        receipt's directory, git, URL or path value."""
        for key in SOURCE_KEYS:
            if key in self.own:
                return str(self.own[key])
        return None

    def with_args(self, *, skip: str | None = None) -> list[str]:
        """``--with`` / ``--with-editable`` arguments for every other
        requirement, except the one named *skip*."""
        args: list[str] = []
        for entry in self.others:
            if skip is not None and normalize(entry["name"]) == normalize(skip):
                continue
            args += requirement_args(entry)
        return args

    def constraint_lines(self) -> list[str]:
        """The recorded constraints, one requirement per line."""
        return [requirement_string(entry) for entry in self.constraints]


def read_receipt(env: Path) -> Receipt:
    """Read ``<env>/uv-receipt.toml``. Raises :class:`ReceiptError` when it
    cannot be read or has no evoscientist requirement."""
    import tomllib

    path = env / RECEIPT_NAME
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        tool = data["tool"]
        entries = tool["requirements"]
        constraints = tool.get("constraints", [])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReceiptError(f"cannot read {path}: {exc}") from exc
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and isinstance(e.get("name"), str) for e in entries
    ):
        raise ReceiptError(f"unexpected requirements in {path}")
    own = [e for e in entries if normalize(e["name"]) == PACKAGE]
    if len(own) != 1:
        raise ReceiptError(f"no single {PACKAGE} requirement in {path}")
    python = tool.get("python")
    return Receipt(
        path=path,
        own=own[0],
        others=tuple(e for e in entries if normalize(e["name"]) != PACKAGE),
        constraints=tuple(e for e in constraints if isinstance(e, dict)),
        python=python if isinstance(python, str) else None,
    )


def find_uv() -> str | None:
    """The uv to run: the one the install scripts recorded in
    ``<data dir>/tools/uv.json``, else ``uv`` on PATH."""
    from .setup._install import tools_dir

    try:
        record = json.loads((tools_dir() / "uv.json").read_text(encoding="utf-8"))
        recorded = Path(record["path"])
        if recorded.is_file():
            return str(recorded)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return shutil.which("uv")


def runs_on_managed_python(uv: str) -> bool:
    """True when this environment's base interpreter is one uv installed
    (``uv python dir``). False when uv cannot tell."""
    from .setup._install import is_under

    try:
        # uv writes UTF-8, not the locale's code page (Windows).
        result = subprocess.run(
            [uv, "python", "dir"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    managed_root = result.stdout.strip()
    if result.returncode != 0 or not managed_root:
        return False
    return is_under(Path(sys.base_prefix), Path(managed_root))


def python_request(receipt: Receipt) -> str:
    """The ``--python`` value for a reinstall: the receipt's, else this
    interpreter's X.Y (an install made without ``--python`` records none)."""
    return receipt.python or f"{sys.version_info.major}.{sys.version_info.minor}"
