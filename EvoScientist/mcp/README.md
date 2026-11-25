# MCP Integration

Connect external tools and data sources to EvoScientist through [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers. Browse community servers in the [MCP Server Directory](https://github.com/modelcontextprotocol/servers).

## Quick Start

Add a server from your terminal, inside a running session, or directly in `~/.config/evoscientist/mcp.yaml`:

```bash
# Terminal (outside a session)
EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

# Inside a running EvoSci session (run /new afterwards to load it)
/mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
```

```yaml
# ~/.config/evoscientist/mcp.yaml (or $XDG_CONFIG_HOME/evoscientist/mcp.yaml)
sequential-thinking:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-sequential-thinking"]
```

## CLI & Slash Commands

Every `EvoSci mcp ...` command also works interactively as `/mcp ...`. Inside a session, use the long flag names: `/mcp add` accepts only `-T` as a short flag, and `/mcp edit` accepts none.

| Command | Action |
|---|---|
| `EvoSci mcp` / `EvoSci mcp list` | List configured servers |
| `EvoSci mcp install [name-or-tag]` | Browse or install servers from the built-in registry/marketplace |
| `EvoSci mcp config [name]` | Show full config for all servers or one server |
| `EvoSci mcp add <name> <cmd-or-url> [-- args...]` | Add a server |
| `EvoSci mcp edit <name> [flags]` | Update routing, tools, or URL without re-adding |
| `EvoSci mcp remove <name>` | Remove a server |

### `add` Flags

| Flag | Short (CLI) | What it does |
|---|---|---|
| `--transport` | `-T` | `stdio`, `http`, `streamable_http`, `sse`, or `websocket` (auto-detected: `ws(s)://` → `websocket`, `http(s)://` → `http`, otherwise `stdio`) |
| `--tools` | `-t` | Comma-separated tool allowlist with glob wildcards (omit for all tools) |
| `--expose-to` | `-e` | Comma-separated target agents (default: `main`) |
| `--header` | `-H` | HTTP header `Key:Value` (repeatable) |
| `--env` | | Subprocess env var `KEY=VALUE` for `stdio` (repeatable) |
| `--env-ref` | | Forward an existing host env var by name (repeatable) |

*(Put `--` before server arguments that start with `-` so `EvoSci` passes them through.)*

### `edit` Flags

`edit` takes `--tools` (`-t`), `--expose-to` (`-e`), `--header` (`-H`), `--env`, and `--transport` (no `-T` here) as above, plus:

| Flag | What it does |
|---|---|
| `--command` / `--url` | Replace the server's command (`stdio`) or URL |
| `--args a,b,c` | Replace the command's arguments *(`/mcp edit` inside a session only)* |
| `--tools none` / `--expose-to none` | Clear the list (all tools / back to `main`) |

### Examples

```bash
# Remote HTTP server (transport auto-detected)
EvoSci mcp add docs-langchain https://docs.langchain.com/mcp

# SSE endpoint routed only to research-agent
EvoSci mcp add research-sse https://example.com/sse --transport sse -e research-agent

# Subprocess with env var reference + sub-agent routing
EvoSci mcp add brave-search npx --env-ref BRAVE_API_KEY -e research-agent -- -y @modelcontextprotocol/server-brave-search

# HTTP server with auth header from env var
EvoSci mcp add brave-http http://localhost:8080/mcp -H "Authorization:Bearer ${BRAVE_API_KEY}"

# Tool allowlist with glob wildcards
EvoSci mcp add fs npx -t "read_*,write_*" -- -y @modelcontextprotocol/server-filesystem /workspace

# Update an existing server in place
EvoSci mcp edit fs --expose-to main,code-agent
EvoSci mcp edit fs --tools none

# Inside a session: same commands, long flags only
/mcp add brave-search npx --env-ref BRAVE_API_KEY --expose-to research-agent -- -y @modelcontextprotocol/server-brave-search
```

## Routing Tools to Specific Agents (`expose_to`)

By default, MCP tools only go to `main`. Route specialized servers straight to the sub-agents that need them—no need to edit `EvoScientist/subagents/*.yaml`.

Valid targets are `main` (the orchestrator, default) and any sub-agent id—the top-level keys in `EvoScientist/subagents/*.yaml`, such as `research-agent`, `code-agent`, or `data-analysis-agent`.

## Filtering Tools With Wildcards (`tools`)

Limit which tools a server exposes using standard glob patterns (`*`, `?`, `[abc]`, `[0-9]`, `[!0-9]`):

```yaml
# ~/.config/evoscientist/mcp.yaml
filesystem:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
  tools: ["read_*", "write_*", "list_*"]
  expose_to: [main, code-agent]

brave-search:
  transport: http
  url: "http://localhost:8080/mcp"
  headers:
    Authorization: "Bearer ${BRAVE_API_KEY}"
  expose_to: [research-agent]
```

Supported YAML fields: `transport` *(required)*, `command`/`args`/`env` (`stdio`), `url`/`headers` (`http`, `streamable_http`, `sse`, `websocket`), `tools`, and `expose_to`.

## How It Loads

On session startup (or `/new`), EvoScientist reads `mcp.yaml`, connects to each server, filters tools by `tools`, routes them by `expose_to`, and caches loaded tools by config signature. Servers that fail to connect log a warning and are skipped without blocking startup.

## Troubleshooting

- **`npx` not found:** EvoScientist installs a private Node.js automatically on first use. If that failed (offline/proxy), run `EvoSci setup` (add `--cn` in mainland China).
- **`${VAR}` / `--env-ref` resolves empty:** Export the variable before starting EvoScientist (and restart the background server via `EvoSci server stop` if it was started from another shell). Missing vars resolve to `""` with a warning.
- **Tools don't appear where expected:** Check `expose_to` agent names, make sure `tools` isn't filtering everything out, and run `/new` after editing `mcp.yaml`.
- **Windows `stdio` `[Errno 9] Bad file descriptor`:** Launch EvoScientist from a real console window (not `pythonw.exe`).

## Security

- Never hardcode tokens in `mcp.yaml`—always use `${VAR}` or `--env-ref KEY`.
- Scope filesystem servers to the narrowest directory needed, and restrict both `tools` and `expose_to` by least privilege.
