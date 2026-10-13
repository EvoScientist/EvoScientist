"""MCP server registry — marketplace index from EvoSkills.

Provides MCP server definitions used by:
- ``/install-mcp`` (interactive browser and direct install)
- ``EvoSci onboard`` (initial setup wizard, filters by ``onboarding`` tag)
- ``EvoSci mcp install`` (CLI command)

Server definitions live in ``EvoSkills/mcp/`` as individual YAML files.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import uv_tool
from ..git_cli import clone_repo as _clone_repo

logger = logging.getLogger(__name__)

# Sentinel for function-local caches that legitimately store ``None``.
_UNSET = object()


# =============================================================================
# Data model
# =============================================================================


@dataclass
class MCPServerEntry:
    """Unified representation of an MCP server."""

    name: str
    label: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    # Connection
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] | None = None
    # Environment & dependencies
    env: dict[str, str] | None = None
    env_key: str | None = None
    env_hint: str = ""
    env_optional: bool = False
    pip_package: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


# =============================================================================
# Pip / dependency helpers
# =============================================================================


def _bare_package_name(package: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string.

    Strips extras (``[...]``), version specifiers (``>=``, ``==``, etc.),
    and environment markers (``; ...``).
    """
    return re.split(r"[\[><=!~;]", package, maxsplit=1)[0].strip()


def pip_install_hint() -> str:
    """Human-readable install command for error messages.

    In EvoScientist's uv tool environment it installs into that environment
    with ``uv pip``: a hand-written ``uv tool install --with`` would have to
    repeat the version, extras and constraints the receipt records, or it
    drops them.
    """
    uv = uv_tool.find_uv()
    if uv_tool.tool_env() is not None and uv:
        # PowerShell runs a quoted path only after the call operator.
        call = "& " if sys.platform == "win32" else ""
        return f'{call}"{uv}" pip install --python "{sys.executable}"'
    if shutil.which("uv"):
        return "uv pip install"
    return "pip install"


def _uv_tool_bin_dir() -> Path | None:
    """Return uv's tool bin directory — where ``uv tool install`` places
    symlinks (typically ``~/.local/bin``). ``None`` when uv isn't
    available or the query fails.

    Cached for the lifetime of the process.
    """
    cached = getattr(_uv_tool_bin_dir, "_cached", _UNSET)
    if cached is not _UNSET:
        return cached  # type: ignore[return-value]
    result: Path | None = None
    if shutil.which("uv"):
        try:
            proc = subprocess.run(
                ["uv", "tool", "dir", "--bin"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                path = proc.stdout.strip()
                if path:
                    result = Path(path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    _uv_tool_bin_dir._cached = result  # type: ignore[attr-defined]
    return result


def _uv_tool_bin(command: str) -> Path | None:
    """Return the executable path for *command* in ``uv tool dir --bin``
    if it exists, else ``None``."""
    tool_bin = _uv_tool_bin_dir()
    if tool_bin is None:
        return None
    cand = tool_bin / command
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    if os.name == "nt":
        cand_exe = cand.with_suffix(".exe")
        if cand_exe.is_file():
            return cand_exe
    return None


def _uv_tool_reinstall_cmd(
    uv: str, receipt: uv_tool.Receipt, package: str, constraints: Path | None
) -> list[str]:
    """The ``uv tool install`` that adds *package* and keeps what the receipt
    records: EvoScientist's version (or the source it came from) and extras,
    the Python, every ``--with`` requirement and, through *constraints*, the
    constraints. A package already in the receipt keeps its recorded spec."""
    from importlib.metadata import version

    if "editable" in receipt.own:
        own = ["--editable", uv_tool.requirement_string(receipt.own)]
    elif receipt.source is not None:
        own = [uv_tool.requirement_string(receipt.own)]
    else:
        extras = f"[{','.join(receipt.extras)}]" if receipt.extras else ""
        own = [f"EvoScientist{extras}=={version('EvoScientist')}"]
    cmd = [uv, "tool", "install", "-q", "--python", uv_tool.python_request(receipt)]
    # Stay on the kind of interpreter the environment has: uv prefers its
    # own Python when one is installed.
    cmd.append(
        "--managed-python"
        if uv_tool.runs_on_managed_python(uv)
        else "--no-managed-python"
    )
    if constraints is not None:
        # By name, run from its folder: uv splits a -c path at spaces
        # (astral-sh/uv#12639).
        cmd += ["-c", constraints.name]
    cmd += receipt.with_args()
    bare = uv_tool.normalize(_bare_package_name(package))
    if all(uv_tool.normalize(e["name"]) != bare for e in receipt.others):
        cmd += ["--with", package]
    return cmd + own


def _install_with_uv_tool_env(package: str) -> bool:
    """Install *package* into EvoScientist's uv tool environment via ``--with``.

    Only in a uv tool environment on macOS and Linux; otherwise returns False
    and the caller falls back. The package is recorded in uv's receipt, so
    ``EvoSci update`` keeps it. On Windows the running ``EvoSci`` keeps the
    environment's files open, and a replace that fails halfway leaves it
    broken, so the fallback (``uv pip install``) is used there.
    """
    if sys.platform == "win32":
        return False
    env = uv_tool.tool_env()
    uv = uv_tool.find_uv()
    if env is None or uv is None:
        return False
    constraints: Path | None = None
    try:
        receipt = uv_tool.read_receipt(env)
        lines = receipt.constraint_lines()
        if lines:
            fd, name = tempfile.mkstemp(prefix="evosci-constraints-", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            constraints = Path(name)
        cmd = _uv_tool_reinstall_cmd(uv, receipt, package, constraints)
    except uv_tool.ReceiptError as exc:
        logger.info(f"Not adding {package} with uv tool install --with: {exc}")
        return False
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=constraints.parent if constraints is not None else None,
        )
        if result.returncode == 0:
            import importlib

            importlib.invalidate_caches()
            return True
        logger.info(
            f"uv tool install --with {package} failed (exit {result.returncode}); "
            f"falling back: {result.stderr.strip()}"
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info(f"uv tool install --with {package} errored ({exc}); falling back")
    finally:
        if constraints is not None:
            constraints.unlink(missing_ok=True)
    return False


def _install_via_pip(package: str) -> bool:
    """Install *package* into the active venv via ``uv pip`` or ``pip``.

    Importable from the current interpreter on success. Not durable under
    ``uv sync`` — the reconcile will remove anything not in the lockfile.
    """
    commands: list[list[str]] = []
    if shutil.which("uv"):
        commands.append(
            ["uv", "pip", "install", "--python", sys.executable, "-q", package]
        )
    commands.append([sys.executable, "-m", "pip", "install", "-q", package])
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                import importlib

                importlib.invalidate_caches()
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def install_library(package: str) -> bool:
    """Install a package that will be imported from the active environment.

    Strategy:

    1. **uv tool env** (macOS and Linux): ``uv tool install`` again with the
       receipt's version, extras, Python, constraints and ``--with``
       packages plus ``--with <pkg>`` — the package lands in evosci's
       isolated env, is importable from it and recorded in the receipt.
    2. **Fallback**: ``uv pip install`` / ``pip install`` into the active
       venv. Not durable under ``uv sync``, but there's no better option
       for libraries outside a uv tool env.

    Does not use standalone ``uv tool install <pkg>`` — that creates an
    isolated env the caller can't import from.
    """
    if _install_with_uv_tool_env(package):
        return True
    return _install_via_pip(package)


def install_cli_tool(package: str, *, verify_command: str) -> bool:
    """Install a package whose primary deliverable is a CLI binary.

    Strategy:

    1. **uv tool env** (macOS and Linux): the same receipt-keeping
       ``uv tool install ... --with <pkg>`` as :func:`install_library` —
       durable via uv's receipt, binary resolvable via the tool's bin dir.
    2. **Standalone ``uv tool install <pkg>``**: binary symlinked under
       ``~/.local/bin`` (``uv tool dir --bin``); survives ``uv sync``.
    3. **Fallback**: ``uv pip install`` / ``pip install`` into the active
       venv. Not durable — wiped by ``uv sync`` — but covers the case
       where ``uv`` isn't available or the package has no console-script.

    *verify_command* is the CLI name expected to appear after step 2; if
    it's missing from ``uv tool dir --bin`` we fall through to step 3.
    """
    # No bin-dir verify for step 1: in a uv-tool env the binary lands in
    # evosci's own bin dir (next to sys.executable), not in `uv tool dir
    # --bin`. `_resolve_command_path` picks it up via its sys.executable
    # branch.
    if _install_with_uv_tool_env(package):
        return True
    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "tool", "install", "-q", package],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0:
                import importlib

                importlib.invalidate_caches()
                # `uv tool install` silently produces no bin when the
                # package lacks a console-script. Check the uv-tool bin
                # dir directly rather than PATH, which under `uv run`
                # resolves stale `.venv/bin/` copies first.
                if _uv_tool_bin(verify_command) is not None:
                    return True
                logger.info(
                    "uv tool install %s succeeded but %s missing from uv "
                    "tool bin dir; falling back to pip install into current venv",
                    package,
                    verify_command,
                )
            else:
                logger.info(
                    "uv tool install %s failed (exit %d); falling back to pip install",
                    package,
                    result.returncode,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.info("uv tool install %s errored (%s); falling back", package, exc)
    return _install_via_pip(package)


def _resolve_command_path(command: str) -> str:
    """Resolve a command to its full path after pip installation.

    Resolution order:

    1. Absolute path — return as-is.
    2. ``uv tool dir --bin`` — prefer this over any ``.venv/bin/``
       shadow. ``uv run`` puts the project venv first on PATH, so a
       stale venv copy of a package (from an earlier ``uv pip install``
       that's since been superseded by ``uv tool install``) would
       otherwise mask the durable location.
    3. PATH — ``shutil.which``.
    4. Current interpreter's ``bin/`` — handles uv tool envs where a
       newly installed binary isn't on PATH but lives alongside the
       tool's Python executable.

    Returns the full absolute path if found, otherwise the original string.
    """
    if os.path.isabs(command):
        return command
    tool_bin = _uv_tool_bin(command)
    if tool_bin is not None:
        return str(tool_bin)
    found = shutil.which(command)
    if found:
        return found
    candidate = Path(sys.executable).parent / command
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    if os.name == "nt":
        candidate_exe = candidate.with_suffix(".exe")
        if candidate_exe.is_file():
            return str(candidate_exe)
    return command


# =============================================================================
# Marketplace index (YAML files in EvoSkills/mcp/)
# =============================================================================

_MARKETPLACE_CACHE: dict[str, tuple[float, list[MCPServerEntry]]] = {}
_MARKETPLACE_TTL = 600  # 10 minutes


def parse_marketplace_yaml(path: Path) -> MCPServerEntry:
    """Parse a single marketplace YAML file into an MCPServerEntry."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    return MCPServerEntry(
        name=data.get("name", path.stem),
        label=data.get("label", data.get("name", path.stem)),
        description=data.get("description", ""),
        tags=tags,
        transport=data.get("transport", "stdio"),
        command=data.get("command"),
        args=data.get("args", []),
        url=data.get("url"),
        headers=data.get("headers"),
        env=data.get("env"),
        env_key=data.get("env_key"),
        env_hint=data.get("env_hint", ""),
        env_optional=data.get("env_optional", False),
        pip_package=data.get("pip_package"),
    )


def _scan_mcp_dir(mcp_root: Path) -> list[MCPServerEntry]:
    """Scan a directory for ``*.yaml`` MCP server definitions."""
    entries: list[MCPServerEntry] = []
    if not mcp_root.is_dir():
        return entries
    for yaml_file in sorted(mcp_root.glob("*.yaml")):
        try:
            entries.append(parse_marketplace_yaml(yaml_file))
        except Exception as exc:
            logger.warning(
                "Failed to parse marketplace MCP %s: %s", yaml_file.name, exc
            )
    return entries


def fetch_marketplace_index(
    repo: str = "EvoScientist/EvoSkills",
    ref: str | None = None,
    path: str = "mcp",
) -> list[MCPServerEntry]:
    """Fetch MCP server definitions from the marketplace.

    Shallow-clones the EvoSkills repo and scans ``{path}/*.yaml``.
    Results are cached for 10 minutes.
    """
    cache_key = f"{repo}:{ref or 'default'}:{path}"
    now = time.monotonic()
    cached = _MARKETPLACE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _MARKETPLACE_TTL:
        return cached[1]

    entries: list[MCPServerEntry] = []
    with tempfile.TemporaryDirectory(prefix="evoscientist-mcp-browse-") as tmp:
        clone_dir = os.path.join(tmp, "repo")
        _clone_repo(repo, ref, clone_dir)
        mcp_root = Path(clone_dir) / path if path else Path(clone_dir)
        entries = _scan_mcp_dir(mcp_root)

    _MARKETPLACE_CACHE[cache_key] = (now, entries)
    return entries


# =============================================================================
# Installation logic
# =============================================================================


def install_mcp_server(
    entry: MCPServerEntry,
    *,
    print_fn: Callable[[str, str], None] | None = None,
) -> bool:
    """Install a single MCP server to the user config.

    Handles:
    1. ``env_key``: prints hint, warns if env var is not set
    2. ``pip_package``: installs via pip/uv
    3. Calls ``add_mcp_server()`` to persist to ``mcp.yaml``

    Args:
        entry: Server definition to install.
        print_fn: Output callback ``(text, style)`` for status messages.

    Returns:
        True on success.
    """
    from .client import add_mcp_server

    if print_fn is None:

        def print_fn(text: str, style: str = "") -> None:
            from ..stream.console import console

            console.print(f"[{style}]{text}[/{style}]" if style else text)

    # Env key hints
    if entry.env_key:
        if entry.env_optional:
            print_fn(f"  {entry.env_hint}", "dim")
        else:
            print_fn(f"  \u26a0 Requires {entry.env_key}", "yellow")
            if entry.env_hint:
                print_fn(f"  {entry.env_hint}", "dim")
            if not os.environ.get(entry.env_key):
                print_fn(
                    f"  Set it before running EvoScientist: export {entry.env_key}=...",
                    "dim",
                )

    # Pip package
    if entry.pip_package:
        print_fn(f"  Installing {entry.pip_package}...", "dim")
        if entry.command:
            ok = install_cli_tool(entry.pip_package, verify_command=entry.command)
        else:
            ok = install_library(entry.pip_package)
        if not ok:
            print_fn(f"  Failed: {pip_install_hint()} {entry.pip_package}", "red")
            return False

    # Add to mcp.yaml
    try:
        if entry.url and entry.transport != "stdio":
            add_mcp_server(
                entry.name,
                entry.transport,
                url=entry.url,
                headers=entry.headers,
            )
        else:
            resolved_cmd = (
                _resolve_command_path(entry.command) if entry.command else entry.command
            )
            add_mcp_server(
                entry.name,
                entry.transport,
                command=resolved_cmd,
                args=entry.args,
                env=entry.env,
            )
        return True
    except Exception as exc:
        print_fn(f"  Failed to add {entry.name}: {exc}", "red")
        return False


def find_server_by_name(
    name: str, servers: list[MCPServerEntry]
) -> MCPServerEntry | None:
    """Case-insensitive name lookup in a server list."""
    name_lower = name.lower()
    return next((s for s in servers if s.name.lower() == name_lower), None)


def get_all_tags(servers: list[MCPServerEntry]) -> set[str]:
    """Collect all unique tags (lowercased) from a server list."""
    return {t.lower() for s in servers for t in s.tags}


def get_installed_names() -> set[str]:
    """Return the set of server names already in the user MCP config."""
    from .client import _load_user_config

    return set(_load_user_config().keys())


def install_mcp_servers(
    entries: list[MCPServerEntry],
    *,
    print_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Install multiple MCP servers, returning the count of successes."""
    count = 0
    for entry in entries:
        if install_mcp_server(entry, print_fn=print_fn):
            if print_fn:
                print_fn(f"Configured: {entry.name}", "green")
            count += 1
        elif print_fn:
            print_fn(f"Failed: {entry.name}", "red")
    return count
