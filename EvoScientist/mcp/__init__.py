"""MCP (Model Context Protocol) integration — external tool support.

See mcp/README.md for usage details.
"""

from .client import (
    load_mcp_config,
    load_mcp_tools,
    aload_mcp_tools,
    add_mcp_server,
    edit_mcp_server,
    remove_mcp_server,
    parse_mcp_add_args,
    parse_mcp_edit_args,
    build_mcp_add_kwargs,
    build_mcp_edit_fields,
    VALID_TRANSPORTS,
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
    "load_mcp_config",
    "load_mcp_tools",
    "aload_mcp_tools",
    "add_mcp_server",
    "edit_mcp_server",
    "remove_mcp_server",
    "parse_mcp_add_args",
    "parse_mcp_edit_args",
    "build_mcp_add_kwargs",
    "build_mcp_edit_fields",
    "VALID_TRANSPORTS",
    # Registry
    "MCPServerEntry",
    "fetch_marketplace_index",
    "find_server_by_name",
    "get_all_tags",
    "get_installed_names",
    "install_mcp_server",
    "install_mcp_servers",
]
