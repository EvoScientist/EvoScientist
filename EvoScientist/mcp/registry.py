"""MCP server registry — built-in servers, marketplace index, and YAML import.

Provides a shared registry of MCP server definitions used by:
- ``/install-mcp`` (interactive browser and direct install)
- ``EvoSci onboard`` (initial setup wizard)
- ``EvoSci mcp install`` (CLI command)

Three sources:
1. **Built-in** — curated servers shipped with EvoScientist.
2. **Marketplace** — YAML files in ``EvoSkills/mcp/`` (fetched via git clone).
3. **Arbitrary YAML** — user-provided files in ``mcp.yaml`` format.
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
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Data model
# =============================================================================


@dataclass
class MCPServerEntry:
    """Unified representation of an MCP server from any source."""

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
# Built-in registry
# =============================================================================

BUILTIN_MCP_SERVERS: list[MCPServerEntry] = [
    MCPServerEntry(
        name="sequential-thinking",
        label="Sequential Thinking  (structured reasoning for non-reasoning models)",
        description="Chain-of-thought reasoning with sequential thinking steps",
        tags=["reasoning", "core"],
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    ),
    MCPServerEntry(
        name="docs-langchain",
        label="Docs by LangChain  (documentation for building agents)",
        description="Documentation for building agents with LangChain",
        tags=["docs", "core"],
        transport="streamable_http",
        url="https://docs.langchain.com/mcp",
    ),
    MCPServerEntry(
        name="perplexity",
        label="Perplexity  (AI-powered web search — requires PERPLEXITY_API_KEY)",
        description="AI-powered web search via Perplexity",
        tags=["search", "research"],
        transport="stdio",
        command="npx",
        args=["-y", "@perplexity-ai/mcp-server"],
        env={"PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"},
        env_key="PERPLEXITY_API_KEY",
        env_hint="export PERPLEXITY_API_KEY=pplx-... (get one at perplexity.ai/settings/api)",
    ),
    MCPServerEntry(
        name="context7",
        label="Context7  (fast documentation lookup — API key unlocks higher rate limits)",
        description="Fast documentation lookup for libraries and frameworks",
        tags=["docs", "search"],
        transport="stdio",
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
        env={"CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"},
        env_key="CONTEXT7_API_KEY",
        env_hint="export CONTEXT7_API_KEY=... (optional — unlocks higher rate limits)",
        env_optional=True,
    ),
    MCPServerEntry(
        name="deepwiki",
        label="DeepWiki  (search & read GitHub repo documentation)",
        description="Search and read GitHub repository documentation",
        tags=["docs", "research"],
        transport="streamable_http",
        url="https://mcp.deepwiki.com/mcp",
    ),
    MCPServerEntry(
        name="arxiv",
        label="ArXiv  (search & fetch academic papers from arXiv)",
        description="Search and fetch academic papers from arXiv",
        tags=["research"],
        transport="stdio",
        command="arxiv-mcp-server",
        args=[],
        pip_package="arxiv-mcp-server",
    ),
]


def get_builtin_servers() -> list[MCPServerEntry]:
    """Return a copy of the built-in server list."""
    return list(BUILTIN_MCP_SERVERS)


# =============================================================================
# Pip / dependency helpers (extracted from config/onboard.py)
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
# Marketplace index (remote YAML files in EvoSkills/mcp/)
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


def fetch_marketplace_index(
    repo: str = "EvoScientist/EvoSkills",
    ref: str | None = None,
    path: str = "mcp",
) -> list[MCPServerEntry]:
    """Fetch MCP server definitions from the marketplace repo.

    Clones the repo, scans ``{path}/*.yaml``, parses each file.
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
        if not mcp_root.is_dir():
            _MARKETPLACE_CACHE[cache_key] = (now, entries)
            return entries

        for yaml_file in sorted(mcp_root.glob("*.yaml")):
            try:
                entry = parse_marketplace_yaml(yaml_file)
                entries.append(entry)
            except Exception as exc:
                logger.warning("Failed to parse marketplace MCP %s: %s", yaml_file.name, exc)

    _MARKETPLACE_CACHE[cache_key] = (now, entries)
    return entries


def get_merged_registry() -> list[MCPServerEntry]:
    """Merge built-in + marketplace servers, deduplicating by name.

    Marketplace entries override built-in entries with the same name.
    """
    by_name: dict[str, MCPServerEntry] = {}
    for entry in BUILTIN_MCP_SERVERS:
        by_name[entry.name] = entry

    try:
        marketplace = fetch_marketplace_index()
        for entry in marketplace:
            by_name[entry.name] = entry
    except Exception as exc:
        logger.warning("Failed to fetch marketplace MCP index: %s", exc)

    return list(by_name.values())


# =============================================================================
# Arbitrary YAML import (mcp.yaml format)
# =============================================================================


def load_servers_from_yaml(path: str | Path) -> list[MCPServerEntry]:
    """Parse an arbitrary YAML file in ``mcp.yaml`` format.

    The file has the same structure as ``~/.config/evoscientist/mcp.yaml``::

        server-name:
          transport: stdio
          command: npx
          args: [-y, "@package/server"]
          ...

    Returns a list of MCPServerEntry with ``name`` injected from each key.
    """
    filepath = Path(path).expanduser().resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    data = yaml.safe_load(filepath.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")

    entries: list[MCPServerEntry] = []
    for name, config in data.items():
        if not isinstance(config, dict):
            logger.warning("Skipping non-dict entry %r in %s", name, path)
            continue

        transport = config.get("transport", "stdio")
        entries.append(
            MCPServerEntry(
                name=str(name),
                label=config.get("label", str(name)),
                description=config.get("description", _synthesize_description(transport, config)),
                tags=config.get("tags", ["imported"]),
                transport=transport,
                command=config.get("command"),
                args=config.get("args", []),
                url=config.get("url"),
                headers=config.get("headers"),
                env=config.get("env"),
                env_key=config.get("env_key"),
                env_hint=config.get("env_hint", ""),
                env_optional=config.get("env_optional", False),
                pip_package=config.get("pip_package"),
            )
        )

    return entries


def _synthesize_description(transport: str, config: dict) -> str:
    """Create a description from config fields when none is provided."""
    if transport == "stdio":
        return f"stdio server: {config.get('command', '?')}"
    return f"{transport} server: {config.get('url', '?')}"


# =============================================================================
# Installation logic
# =============================================================================


def install_mcp_server(
    entry: MCPServerEntry,
    *,
    print_fn: Callable[[str], None] | None = None,
) -> bool:
    """Install a single MCP server to the user config.

    Handles:
    1. ``env_key``: prints hint, warns if env var is not set
    2. ``pip_package``: installs via pip/uv
    3. Calls ``add_mcp_server()`` to persist to ``mcp.yaml``

    Args:
        entry: Server definition to install.
        print_fn: Output function (defaults to ``rich.console.print``).

    Returns:
        True on success.
    """
    from .client import add_mcp_server

    if print_fn is None:
        from ..stream.display import console

        print_fn = console.print

    # Env key hints
    if entry.env_key:
        if entry.env_optional:
            print_fn(f"  [dim]{entry.env_hint}[/dim]")
        else:
            print_fn(f"  [yellow]\u26a0 Requires {entry.env_key}[/yellow]")
            if entry.env_hint:
                print_fn(f"  [dim]{entry.env_hint}[/dim]")
            if not os.environ.get(entry.env_key):
                print_fn(
                    f"  [dim]Set it before running EvoScientist: export {entry.env_key}=...[/dim]"
                )

    # Pip package
    if entry.pip_package:
        print_fn(f"  [dim]Installing {entry.pip_package}...[/dim]")
        if not install_pip_package(entry.pip_package):
            print_fn(
                f"  [red]Failed: {pip_install_hint()} {entry.pip_package}[/red]"
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
        print_fn(f"  [red]Failed to add {entry.name}: {exc}[/red]")
        return False
