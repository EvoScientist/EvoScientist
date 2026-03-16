"""MCP (Model Context Protocol) client integration.

Loads MCP server configurations from YAML, connects via langchain-mcp-adapters,
and routes the resulting LangChain tools to the appropriate agents.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Regex for ${VAR} env var interpolation
ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Supported transport protocols
VALID_TRANSPORTS = {"stdio", "http", "streamable_http", "sse", "websocket"}

# URL-based transports (share the same connection shape)
_URL_TRANSPORTS = {"http", "streamable_http", "sse", "websocket"}


def _get_mcp_config_dir() -> Path:
    """Get the MCP configuration directory, respecting XDG_CONFIG_HOME."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "evoscientist"
    return Path.home() / ".config" / "evoscientist"


# User-level config path
USER_CONFIG_DIR = _get_mcp_config_dir()
USER_MCP_CONFIG = USER_CONFIG_DIR / "mcp.yaml"


# =============================================================================
# Environment variable interpolation
# =============================================================================


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR}`` patterns with environment variable values.

    Missing variables are replaced with an empty string and a warning is logged.
    """

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            logger.warning("MCP config: env var $%s is not set", var)
            return ""
        return val

    return ENV_VAR_RE.sub(_replace, value)


def _interpolate_value(value: Any) -> Any:
    """Recursively interpolate env vars in strings, dicts, and lists."""
    if isinstance(value, str):
        return _interpolate_env(value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v) for v in value]
    return value


# =============================================================================
# User config persistence
# =============================================================================


def _load_user_config() -> dict[str, Any]:
    """Load the user-level MCP config, returning an empty dict if absent."""
    if USER_MCP_CONFIG.is_file():
        try:
            data = yaml.safe_load(USER_MCP_CONFIG.read_text()) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_user_config(config: dict[str, Any]) -> None:
    """Write *config* to the user-level MCP config file."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_MCP_CONFIG.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )


# =============================================================================
# CRUD operations
# =============================================================================


def add_mcp_server(
    name: str,
    transport: str,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    tools: list[str] | None = None,
    expose_to: list[str] | None = None,
) -> dict[str, Any]:
    """Add or replace an MCP server in the user config.

    Returns the server entry that was written.
    """
    if transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSPORTS))}"
        )

    entry: dict[str, Any] = {"transport": transport}

    if transport == "stdio":
        if not command:
            raise ValueError("stdio transport requires a command")
        entry["command"] = command
        entry["args"] = args or []
        if env:
            entry["env"] = env
    else:
        if not url:
            raise ValueError(f"{transport} transport requires a url")
        entry["url"] = url
        if headers:
            entry["headers"] = headers

    if tools:
        entry["tools"] = tools
    if expose_to:
        entry["expose_to"] = expose_to

    user_cfg = _load_user_config()
    user_cfg[name] = entry
    _save_user_config(user_cfg)
    return entry


def edit_mcp_server(name: str, **fields: Any) -> dict[str, Any]:
    """Update fields on an existing MCP server entry.

    Only the provided *fields* are changed; everything else is preserved.
    Passing ``None`` for a field removes it.

    Returns the updated entry.

    Raises:
        KeyError: if *name* doesn't exist in the user config.
        ValueError: on invalid transport or missing required fields.
    """
    user_cfg = _load_user_config()
    if name not in user_cfg:
        raise KeyError(f"MCP server {name!r} not found in user config")

    entry = user_cfg[name]

    for key, value in fields.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value

    # Re-validate after edits
    transport = entry.get("transport", "")
    if transport and transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSPORTS))}"
        )
    if transport == "stdio" and not entry.get("command"):
        raise ValueError("stdio transport requires a command")
    if transport in _URL_TRANSPORTS and not entry.get("url"):
        raise ValueError(f"{transport} transport requires a url")

    user_cfg[name] = entry
    _save_user_config(user_cfg)
    return entry


def remove_mcp_server(name: str) -> bool:
    """Remove an MCP server from the user config.

    Returns True if removed, False if it didn't exist.
    """
    user_cfg = _load_user_config()
    if name not in user_cfg:
        return False
    del user_cfg[name]
    _save_user_config(user_cfg)
    return True


# =============================================================================
# CLI argument parsing
# =============================================================================


def _infer_transport(target: str) -> str:
    """Return transport type inferred from *target* URL scheme."""
    if target.startswith(("ws://", "wss://")):
        return "websocket"
    if target.startswith(("http://", "https://")):
        return "http"
    return "stdio"


def build_mcp_add_kwargs(
    name: str,
    target: str,
    extra_args: list[str] | None = None,
    transport: str | None = None,
    tools: list[str] | None = None,
    expose_to: list[str] | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Build kwargs dict for :func:`add_mcp_server` from structured parameters.

    If *transport* is ``None`` it is inferred from *target* (URL → ``http``,
    otherwise ``stdio``).
    """
    if transport is None:
        transport = _infer_transport(target)
    kwargs: dict = {"name": name, "transport": transport}
    if transport == "stdio":
        kwargs["command"] = target
        kwargs["args"] = list(extra_args) if extra_args else []
        if env:
            kwargs["env"] = env
    else:
        kwargs["url"] = target
        if headers:
            kwargs["headers"] = headers
    if tools:
        kwargs["tools"] = tools
    if expose_to:
        kwargs["expose_to"] = expose_to
    return kwargs


def build_mcp_edit_fields(
    transport: str | None = None,
    command: str | None = None,
    url: str | None = None,
    tools: str | None = None,
    expose_to: str | None = None,
    headers: list[str] | None = None,
    env: list[str] | None = None,
) -> dict:
    """Build fields dict for :func:`edit_mcp_server` from structured parameters.

    *tools* and *expose_to* accept the string ``"none"`` to clear the field,
    or a comma-separated list.  *headers* and *env* are lists of
    ``"Key:Value"`` / ``"KEY=VALUE"`` strings respectively.
    """
    fields: dict = {}
    if transport is not None:
        fields["transport"] = transport
    if command is not None:
        fields["command"] = command
    if url is not None:
        fields["url"] = url
    if tools is not None:
        fields["tools"] = (
            None
            if tools == "none"
            else [t.strip() for t in tools.split(",") if t.strip()]
        )
    if expose_to is not None:
        fields["expose_to"] = (
            None
            if expose_to == "none"
            else [a.strip() for a in expose_to.split(",") if a.strip()]
        )
    if headers:
        hdr: dict[str, str] = {}
        for h in headers:
            if ":" in h:
                k, v = h.split(":", 1)
                hdr[k.strip()] = v.strip()
        if hdr:
            fields["headers"] = hdr
    if env:
        env_dict: dict[str, str] = {}
        for e in env:
            if "=" in e:
                k, v = e.split("=", 1)
                env_dict[k.strip()] = v.strip()
        if env_dict:
            fields["env"] = env_dict
    return fields


def parse_mcp_add_args(tokens: list[str]) -> dict:
    """Parse CLI tokens for ``/mcp add`` into kwargs for :func:`add_mcp_server`.

    Syntax::

        <name> <command-or-url> [extra-args...]
            [--transport T] [--tools t1,t2] [--expose-to a1,a2]
            [--header Key:Value]... [--env KEY=VALUE]...

    Transport defaults to ``stdio`` for commands and ``http`` for URLs.
    """
    if len(tokens) < 2:
        raise ValueError(
            "Usage: <name> <command-or-url> [args...]\n"
            "  Options: --transport T  --tools t1,t2  --expose-to agent1,agent2  --header Key:Value  --env KEY=VALUE"
        )

    name = tokens[0]

    positional: list[str] = []
    transport: str | None = None
    tools: list[str] | None = None
    expose_to: list[str] | None = None
    headers: dict[str, str] = {}
    env: dict[str, str] = {}

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--transport", "-T") and i + 1 < len(tokens):
            transport = tokens[i + 1]
            i += 2
        elif tok == "--tools" and i + 1 < len(tokens):
            tools = [t.strip() for t in tokens[i + 1].split(",") if t.strip()]
            i += 2
        elif tok == "--expose-to" and i + 1 < len(tokens):
            expose_to = [a.strip() for a in tokens[i + 1].split(",") if a.strip()]
            i += 2
        elif tok == "--header" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if ":" in kv:
                k, v = kv.split(":", 1)
                headers[k.strip()] = v.strip()
            i += 2
        elif tok == "--env" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
            i += 2
        elif tok == "--env-ref" and i + 1 < len(tokens):
            env[tokens[i + 1]] = "${" + tokens[i + 1] + "}"
            i += 2
        elif tok == "--":
            i += 1  # skip -- separator (used by shells, not meaningful here)
        else:
            positional.append(tok)
            i += 1

    if not positional:
        raise ValueError("A command or URL is required after the server name")

    return build_mcp_add_kwargs(
        name=name,
        target=positional[0],
        extra_args=positional[1:] or None,
        transport=transport,
        tools=tools,
        expose_to=expose_to,
        headers=headers or None,
        env=env or None,
    )


def parse_mcp_edit_args(tokens: list[str]) -> tuple[str, dict]:
    """Parse CLI tokens for ``/mcp edit`` into (name, fields).

    Syntax::

        <name> [--transport T] [--command C] [--url U]
               [--tools t1,t2] [--tools none] [--expose-to a1,a2]
               [--header Key:Value]... [--env KEY=VALUE]...

    ``--tools none`` and ``--expose-to none`` clear those fields.
    """
    if not tokens:
        raise ValueError(
            "Usage: <name> [--transport T] [--command C] [--url U] "
            "[--tools t1,t2] [--expose-to a1,a2] [--header K:V] [--env K=V]"
        )

    name = tokens[0]

    # Parse tokens into raw values
    transport_val: str | None = None
    command_val: str | None = None
    url_val: str | None = None
    args_val: list[str] | None = None
    tools_val: str | None = None
    expose_to_val: str | None = None
    header_list: list[str] = []
    env_list: list[str] = []

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--transport" and i + 1 < len(tokens):
            transport_val = tokens[i + 1]
            i += 2
        elif tok == "--command" and i + 1 < len(tokens):
            command_val = tokens[i + 1]
            i += 2
        elif tok == "--url" and i + 1 < len(tokens):
            url_val = tokens[i + 1]
            i += 2
        elif tok == "--args" and i + 1 < len(tokens):
            args_val = tokens[i + 1].split(",")
            i += 2
        elif tok == "--tools" and i + 1 < len(tokens):
            tools_val = tokens[i + 1]
            i += 2
        elif tok == "--expose-to" and i + 1 < len(tokens):
            expose_to_val = tokens[i + 1]
            i += 2
        elif tok == "--header" and i + 1 < len(tokens):
            header_list.append(tokens[i + 1])
            i += 2
        elif tok == "--env" and i + 1 < len(tokens):
            env_list.append(tokens[i + 1])
            i += 2
        else:
            i += 1

    fields = build_mcp_edit_fields(
        transport=transport_val,
        command=command_val,
        url=url_val,
        tools=tools_val,
        expose_to=expose_to_val,
        headers=header_list or None,
        env=env_list or None,
    )
    if args_val is not None:
        fields["args"] = args_val

    if not fields:
        raise ValueError(
            "No fields to edit. Use --transport, --command, --url, --tools, --expose-to, etc."
        )

    return name, fields


# =============================================================================
# Config loading & merging
# =============================================================================


def load_mcp_config() -> dict[str, Any]:
    """Load MCP configuration from user config.

    Reads ``~/.config/evoscientist/mcp.yaml`` and interpolates ``${VAR}``
    environment variable references.

    Returns an empty dict if no servers are configured (MCP is optional).
    """
    if not USER_MCP_CONFIG.is_file():
        return {}

    try:
        data = yaml.safe_load(USER_MCP_CONFIG.read_text()) or {}
        if not isinstance(data, dict):
            return {}
    except Exception as exc:
        logger.warning("Failed to load MCP config %s: %s", USER_MCP_CONFIG, exc)
        return {}

    return _interpolate_value(data)


def _build_connections(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert YAML config to ``MultiServerMCPClient`` connections format.

    Unknown transports are skipped with a warning.
    """
    connections: dict[str, dict[str, Any]] = {}

    for name, server in config.items():
        transport = server.get("transport", "")

        if transport == "stdio":
            conn: dict[str, Any] = {
                "transport": "stdio",
                "command": server.get("command", ""),
                "args": server.get("args", []),
            }
            if "env" in server:
                conn["env"] = server["env"]
            connections[name] = conn

        elif transport in _URL_TRANSPORTS:
            conn = {
                "transport": transport,
                "url": server.get("url", ""),
            }
            if "headers" in server:
                conn["headers"] = server["headers"]
            connections[name] = conn

        else:
            logger.warning(
                "MCP server %r: unknown transport %r, skipping", name, transport
            )

    return connections


# =============================================================================
# Tool loading, filtering & routing
# =============================================================================


def _filter_tools(tools: list, allowed_names: list[str] | None) -> list:
    """Filter tools by allowlist with wildcard support.

    If *allowed_names* is ``None``, all tools pass through.

    Supports glob-style wildcards:
    - ``*`` matches any sequence of characters
    - ``?`` matches any single character
    - ``[seq]`` matches any character in seq
    - ``[!seq]`` matches any character not in seq

    Examples:
    - ``*_exa`` matches ``web_search_exa``, ``get_code_context_exa``
    - ``read_*`` matches ``read_file``, ``read_directory``
    - ``tool_[0-9]`` matches ``tool_1``, ``tool_2``, etc.
    """
    if allowed_names is None:
        return tools

    # Check if any pattern contains wildcard characters
    has_wildcards = any(
        any(char in pattern for char in "*?[]") for pattern in allowed_names
    )

    if not has_wildcards:
        # Fast path: exact matching with set lookup
        allowed_set = set(allowed_names)
        return [t for t in tools if t.name in allowed_set]

    # Wildcard matching: check each tool against all patterns
    filtered = []
    for tool in tools:
        if any(fnmatch.fnmatch(tool.name, pattern) for pattern in allowed_names):
            filtered.append(tool)
    return filtered


def _route_tools(
    config: dict[str, Any],
    server_tools: dict[str, list],
) -> dict[str, list]:
    """Group filtered tools by target agent.

    Args:
        config: Full MCP config dict (server name -> server settings).
        server_tools: server name -> list of LangChain tools from that server.

    Returns:
        Dict mapping agent name -> list of tools. Key ``"main"`` targets the
        main EvoScientist agent; other keys match subagent names.
    """
    by_agent: dict[str, list] = {}

    for server_name, tools in server_tools.items():
        server_cfg = config.get(server_name, {})

        # Apply tool name filter
        allowed = server_cfg.get("tools")  # None means all
        filtered = _filter_tools(tools, allowed)

        # Determine target agents
        expose_to = server_cfg.get("expose_to", ["main"])
        if isinstance(expose_to, str):
            expose_to = [expose_to]

        for agent_name in expose_to:
            by_agent.setdefault(agent_name, []).extend(filtered)

    return by_agent


async def _load_tools(config: dict[str, Any]) -> dict[str, list]:
    """Connect to MCP servers and retrieve tools.

    Returns a dict of server name -> list of LangChain tools.

    Raises:
        ImportError: if ``langchain-mcp-adapters`` is not installed.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        raise ImportError(
            "MCP servers are configured but langchain-mcp-adapters is not installed.\n"
            "Install with: pip install langchain-mcp-adapters"
        )

    connections = _build_connections(config)
    if not connections:
        return {}

    server_tools: dict[str, list] = {}
    client = MultiServerMCPClient(connections)  # type: ignore[invalid-argument-type]

    for server_name in connections:
        try:
            tools = await client.get_tools(server_name=server_name)
            server_tools[server_name] = tools
            logger.info("MCP server %r: loaded %d tool(s)", server_name, len(tools))
        except Exception as exc:
            logger.warning("MCP server %r: failed to load tools: %s", server_name, exc)
            server_tools[server_name] = []

    return server_tools


# =============================================================================
# SSH config validation & live checks
# =============================================================================


def is_ssh_server(server_config: dict[str, Any]) -> bool:
    """Return True if *server_config* describes an SSH MCP server."""
    env = server_config.get("env") or {}
    if env.get("SSH_HOST"):
        return True
    args = server_config.get("args") or []
    return any("ssh" in str(a).lower() for a in args)


def validate_ssh_config(name: str, server_config: dict[str, Any]) -> list[dict]:
    """Validate an MCP server config entry without making any network connections.

    Returns a list of check result dicts:
        [{"check": "config.transport", "status": "ok"|"warn"|"fail", "detail": "..."}]
    """
    results: list[dict] = []

    def _check(check: str, status: str, detail: str) -> None:
        results.append({"check": check, "status": status, "detail": detail})

    # --- transport ---
    transport = server_config.get("transport", "")
    if not transport:
        _check("config.transport", "fail", "transport field is missing")
        return results
    if transport not in VALID_TRANSPORTS:
        _check(
            "config.transport",
            "fail",
            f"unknown transport {transport!r}; valid: {', '.join(sorted(VALID_TRANSPORTS))}",
        )
        return results
    _check("config.transport", "ok", f"transport={transport!r}")

    # --- command / url ---
    if transport == "stdio":
        if not server_config.get("command"):
            _check("config.command", "fail", "stdio transport requires a command")
        else:
            _check("config.command", "ok", f"command={server_config['command']!r}")
    elif transport in _URL_TRANSPORTS:
        if not server_config.get("url"):
            _check("config.url", "fail", f"{transport} transport requires a url")
        else:
            _check("config.url", "ok", f"url={server_config['url']!r}")

    # --- detect SSH server ---
    is_ssh = is_ssh_server(server_config)
    env = server_config.get("env") or {}

    if not is_ssh:
        # No SSH-specific checks needed
        if not server_config.get("expose_to"):
            _check("config.expose_to", "warn", "expose_to not set; defaulting to main")
        else:
            _check("config.expose_to", "ok", f"expose_to={server_config['expose_to']!r}")
        return results

    # --- SSH-specific validation ---
    for var in ("SSH_HOST", "SSH_USER", "SSH_KEY_PATH"):
        val = env.get(var, "")
        if not val:
            _check(
                f"ssh.env.{var}",
                "fail",
                f"{var} is missing or empty in env",
            )
        else:
            _check(f"ssh.env.{var}", "ok", f"{var} is set")

    # SSH_KEY_PATH file existence
    key_path_raw = env.get("SSH_KEY_PATH", "")
    if key_path_raw:
        key_path = Path(key_path_raw).expanduser()
        if key_path.is_file():
            _check("ssh.key_file", "ok", f"key file exists: {key_path}")
        else:
            _check("ssh.key_file", "fail", f"key file not found: {key_path}")

    if not server_config.get("expose_to"):
        _check("config.expose_to", "warn", "expose_to not set; defaulting to main")
    else:
        _check("config.expose_to", "ok", f"expose_to={server_config['expose_to']!r}")

    return results


async def check_ssh_server(name: str, server_config: dict[str, Any]) -> list[dict]:
    """Run live connection and GPU checks for an MCP server.

    Steps:
    1. Runs ``validate_ssh_config()`` (config-only, no network).
    2. Connects via ``MultiServerMCPClient`` and lists tools.
    3. If SSH tools found, runs ``echo ok`` via ``ssh_execute``.
    4. If SSH works, queries ``nvidia-smi`` for GPU info.

    Each step uses ``asyncio.wait_for`` with a 30-second timeout.
    Returns extended check result list.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        return [
            {
                "check": "deps",
                "status": "fail",
                "detail": "langchain-mcp-adapters not installed",
            }
        ]

    results = validate_ssh_config(name, server_config)

    # Abort live checks if config has any failures
    if any(r["status"] == "fail" for r in results):
        return results

    # --- live connection ---
    connections = _build_connections({name: server_config})
    client = MultiServerMCPClient(connections)  # type: ignore[invalid-argument-type]

    try:
        try:
            tools = await asyncio.wait_for(
                client.get_tools(server_name=name), timeout=30
            )
            tool_names = [t.name for t in tools]
            results.append(
                {
                    "check": "connection",
                    "status": "ok",
                    "detail": f"connected; {len(tools)} tool(s): {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}",
                }
            )
        except asyncio.TimeoutError:
            results.append(
                {"check": "connection", "status": "fail", "detail": "connection timed out (30s)"}
            )
            return results
        except Exception as exc:
            results.append(
                {"check": "connection", "status": "fail", "detail": f"connection failed: {exc}"}
            )
            return results

        # --- SSH execute echo ---
        ssh_tools = {t.name: t for t in tools if "ssh" in t.name.lower()}
        execute_tool = ssh_tools.get("ssh_execute") or next(
            (t for t in tools if "execute" in t.name.lower()), None
        )

        if execute_tool is None:
            results.append(
                {"check": "ssh.execute", "status": "warn", "detail": "no ssh_execute tool found"}
            )
            return results

        try:
            echo_result = await asyncio.wait_for(
                execute_tool.ainvoke({"command": "echo ok"}), timeout=30
            )
            echo_ok = "ok" in str(echo_result).lower()
            results.append(
                {
                    "check": "ssh.execute",
                    "status": "ok" if echo_ok else "warn",
                    "detail": f"echo ok → {str(echo_result)[:80]}",
                }
            )
        except asyncio.TimeoutError:
            results.append(
                {"check": "ssh.execute", "status": "fail", "detail": "ssh_execute timed out (30s)"}
            )
            return results
        except Exception as exc:
            results.append(
                {"check": "ssh.execute", "status": "fail", "detail": f"ssh_execute failed: {exc}"}
            )
            return results

        # --- nvidia-smi GPU check ---
        try:
            gpu_result = await asyncio.wait_for(
                execute_tool.ainvoke(
                    {
                        "command": "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
                    }
                ),
                timeout=30,
            )
            gpu_str = str(gpu_result).strip()
            if gpu_str:
                results.append(
                    {"check": "ssh.gpu", "status": "ok", "detail": f"GPU(s): {gpu_str[:120]}"}
                )
            else:
                results.append(
                    {"check": "ssh.gpu", "status": "warn", "detail": "nvidia-smi returned no output"}
                )
        except asyncio.TimeoutError:
            results.append(
                {"check": "ssh.gpu", "status": "fail", "detail": "nvidia-smi timed out (30s)"}
            )
        except Exception as exc:
            results.append(
                {
                    "check": "ssh.gpu",
                    "status": "warn",
                    "detail": f"nvidia-smi not available: {exc}",
                }
            )

        return results
    finally:
        _close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if _close is not None:
            try:
                result = _close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass


async def aload_mcp_tools() -> dict[str, list]:
    """Async version of :func:`load_mcp_tools`.

    Prefer this when already inside an async context (e.g. Jupyter, async CLI).
    """
    config = load_mcp_config()
    if not config:
        return {}
    try:
        server_tools = await _load_tools(config)
    except Exception as exc:
        logger.warning("MCP tool loading failed: %s", exc)
        return {}
    return _route_tools(config, server_tools)


def load_mcp_tools() -> dict[str, list]:
    """Load MCP tools and return them grouped by target agent.

    This is the main synchronous entry point. It:
    1. Loads user config from ``~/.config/evoscientist/mcp.yaml``
    2. Connects to each configured MCP server
    3. Filters tools per server allowlist
    4. Routes tools to target agents

    Returns:
        Dict mapping agent name -> list of LangChain ``BaseTool`` objects.
        Key ``"main"`` = main agent. Other keys = subagent names.
        Returns empty dict if no MCP servers are configured.
    """
    config = load_mcp_config()
    if not config:
        return {}

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            # Inside an already-running event loop (e.g. Jupyter) —
            # nest_asyncio patches the loop so asyncio.run() works.
            import nest_asyncio

            nest_asyncio.apply()
        server_tools = asyncio.run(_load_tools(config))
    except Exception as exc:
        logger.warning("MCP tool loading failed: %s", exc)
        return {}

    return _route_tools(config, server_tools)
