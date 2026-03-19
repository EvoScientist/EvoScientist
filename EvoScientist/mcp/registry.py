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
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

logger = logging.getLogger(__name__)


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


def pip_install_hint() -> str:
    """Human-readable install command for error messages."""
    if shutil.which("uv"):
        return "uv pip install"
    return "pip install"


def install_pip_package(package: str) -> bool:
    """Silently install a pip package.

    Tries ``uv pip install`` first, then ``python -m pip install``.

    Returns True if installation succeeded.
    """
    commands: list[list[str]] = []
    if shutil.which("uv"):
        commands.append(["uv", "pip", "install", "-q", package])
    commands.append([sys.executable, "-m", "pip", "install", "-q", package])

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                import importlib

                importlib.invalidate_caches()
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


# =============================================================================
# Marketplace index (YAML files in EvoSkills/mcp/)
# =============================================================================

_MARKETPLACE_CACHE: dict[str, tuple[float, list[MCPServerEntry]]] = {}
_MARKETPLACE_TTL = 600  # 10 minutes

_CLONE_TIMEOUT = 120


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
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git clone timed out after {_CLONE_TIMEOUT}s for {repo}")
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")


def parse_marketplace_yaml(path: Path) -> MCPServerEntry:
    """Parse a single marketplace YAML file into an MCPServerEntry."""
    data = yaml.safe_load(path.read_text()) or {}
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
            logger.warning("Failed to parse marketplace MCP %s: %s", yaml_file.name, exc)
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
            from ..stream.display import console

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
        if not install_pip_package(entry.pip_package):
            print_fn(
                f"  Failed: {pip_install_hint()} {entry.pip_package}", "red"
            )
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
            add_mcp_server(
                entry.name,
                entry.transport,
                command=entry.command,
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
