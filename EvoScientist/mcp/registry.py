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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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


def _is_uv_tool_env() -> bool:
    """Return True when running inside a ``uv tool install`` isolated environment.

    Detection: ``VIRTUAL_ENV`` is set and its path contains the uv tools
    directory segment (``/uv/tools/`` on Unix, ``\\uv\\tools\\`` on Windows).
    """
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    if not virtual_env:
        return False
    normalized = virtual_env.replace("\\", "/")
    return "/uv/tools/" in normalized


def _uv_tool_name() -> str | None:
    """Extract the uv tool name from ``VIRTUAL_ENV``.

    Returns ``None`` when not in a uv tool environment.
    The tool name is the last path component of ``VIRTUAL_ENV``
    (e.g. ``/home/user/.local/share/uv/tools/evoscientist`` → ``evoscientist``).
    """
    if not _is_uv_tool_env():
        return None
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    return Path(virtual_env).name or None


def _uv_tool_existing_requirements() -> list[str]:
    """Read existing ``--with`` requirements from the uv tool receipt.

    Returns package names (excluding the tool itself) so that
    ``uv tool install --with`` calls can preserve them.
    """
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    if not virtual_env:
        return []
    receipt = Path(virtual_env) / "uv-receipt.toml"
    if not receipt.is_file():
        return []
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return []
    try:
        data = tomllib.loads(receipt.read_text())
    except Exception:
        return []
    tool_name = _uv_tool_name() or ""
    reqs = data.get("tool", {}).get("requirements", [])
    return [
        r["name"]
        for r in reqs
        if isinstance(r, dict) and r.get("name") and r["name"] != tool_name
    ]


def pip_install_hint() -> str:
    """Human-readable install command for error messages."""
    if _is_uv_tool_env():
        return "uv tool install --reinstall evoscientist --with"
    if shutil.which("uv"):
        return "uv pip install"
    return "pip install"


def install_pip_package(package: str) -> bool:
    """Silently install a pip package.

    In a **uv tool environment**, uses ``uv tool install <tool> --with``
    which records the dependency in uv's receipt so it survives
    ``uv tool upgrade``.  Existing ``--with`` packages are preserved.

    Otherwise, when ``uv`` is available, uses
    ``uv pip install --python sys.executable``.
    Falls back to ``python -m pip install`` when uv is not available.

    Returns True if installation succeeded.
    """
    # ---- uv tool env: durable install via `uv tool install --with` ----
    if _is_uv_tool_env() and shutil.which("uv"):
        tool_name = _uv_tool_name()
        if tool_name:
            existing = _uv_tool_existing_requirements()
            cmd = ["uv", "tool", "install", tool_name, "-q"]
            for req in existing:
                cmd += ["--with", req]
            if package not in existing:
                cmd += ["--with", package]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=180
                )
                if result.returncode == 0:
                    import importlib

                    importlib.invalidate_caches()
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            # Fall through to legacy methods if uv tool install failed.

    # ---- Standard venv / conda / system Python ----
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


def _resolve_command_path(command: str) -> str:
    """Resolve a command to its full path after pip installation.

    Checks PATH first (standard case), then the bin directory of the current
    Python interpreter — handles uv tool envs where a newly installed binary
    is not on PATH but lives alongside the tool's Python executable.

    Returns the full absolute path if found, otherwise the original string.
    """
    if os.path.isabs(command):
        return command
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
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"git clone timed out after {_CLONE_TIMEOUT}s for {repo}"
        ) from e
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
