"""Typer application objects — no intra-package imports to avoid circular deps."""

import typer  # type: ignore[import-untyped]

app = typer.Typer(
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Config subcommand group
config_app = typer.Typer(help="Configuration management commands", invoke_without_command=True)
app.add_typer(config_app, name="config")

# MCP subcommand group
_MCP_HELP = """\
Configure and manage MCP servers

Examples:
  # Add a local MCP server (stdio auto-detected):
  EvoSci mcp add local-server python -- /path/to/server.py

  # Add an npx-based server:
  EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

  # Add an HTTP server (http auto-detected from URL):
  EvoSci mcp add docs-langchain https://docs.langchain.com/mcp

  # Add a stdio server with env vars (hardcoded):
  EvoSci mcp add my-server node --env API_KEY=xxx -- server.js

  # Add a server with runtime env ref (resolved from .env at startup):
  EvoSci mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search

  # Expose to a specific sub-agent (e.g. research-agent):
  EvoSci mcp add brave-search npx --env-ref BRAVE_API_KEY -e research-agent -- -y @modelcontextprotocol/server-brave-search

  # Expose to multiple agents:
  EvoSci mcp add local-server python -e main,research-agent,code-agent -- /path/to/server.py

  # Explicit transport override:
  EvoSci mcp add my-sse https://example.com/sse --transport sse

Sub-agents (-e): planner-agent | research-agent | code-agent | debug-agent | data-analysis-agent | writing-agent
"""
mcp_app = typer.Typer(help=_MCP_HELP, invoke_without_command=True)
app.add_typer(mcp_app, name="mcp")

# Channel subcommand group
channel_app = typer.Typer(help="Channel management commands")
app.add_typer(channel_app, name="channel")
