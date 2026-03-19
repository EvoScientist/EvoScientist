"""MCP (Model Context Protocol) integration — external tool support.

See mcp/README.md for usage details.
"""

from .client import (
    VALID_TRANSPORTS,
    add_mcp_server,
    aload_mcp_tools,
    build_mcp_add_kwargs,
    build_mcp_edit_fields,
    edit_mcp_server,
    load_mcp_config,
    load_mcp_tools,
    parse_mcp_add_args,
    parse_mcp_edit_args,
    remove_mcp_server,
)
from .registry import (
    MCPServerEntry,
    fetch_marketplace_index,
    find_server_by_name,
    get_all_tags,
    get_installed_names,
    install_mcp_server,
    install_mcp_servers,
)

__all__ = [
    "VALID_TRANSPORTS",
    "MCPServerEntry",
    "add_mcp_server",
    "aload_mcp_tools",
    "build_mcp_add_kwargs",
    "build_mcp_edit_fields",
    "edit_mcp_server",
    "fetch_marketplace_index",
    "find_server_by_name",
    "get_all_tags",
    "get_installed_names",
    "install_mcp_server",
    "install_mcp_servers",
    "load_mcp_config",
    "load_mcp_tools",
    "parse_mcp_add_args",
    "parse_mcp_edit_args",
    "remove_mcp_server",
]
