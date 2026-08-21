# Model Context Protocol Integration

> Connects external systems to [EvoScientist](https://github.com/EvoScientist/EvoScientist) via [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

> [!TIP]
> Explore more servers: [MCP Server Directory](https://github.com/modelcontextprotocol/servers)

## 📖 Contents

- [🔌 Quick Start](#-quick-start)
- [🛠️ Command Reference](#️-command-reference)
  - [Syntax](#syntax)
  - [Management](#management)
  - [Examples](#examples)
  - [Editing Servers](#editing-servers)
- [🔀 Tool Routing](#-tool-routing)
- [🔍 Tool Filtering with Wildcards](#-tool-filtering-with-wildcards)
  - [Wildcard Patterns](#wildcard-patterns)
- [📄 Config File](#-config-file)
  - [Config Fields](#config-fields)
  - [Supported Transports](#supported-transports)
- [⚙️ How It Works](#️-how-it-works)
- [🔧 Troubleshooting](#-troubleshooting)
- [🔒 Security](#-security)

## 🔌 Quick Start

### Option A: Terminal — before or outside an agent session

```bash
EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
```

### Option B: In-session — inside a running agent session

```bash
/mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
```

> [!NOTE]
> After adding a new server in-session, reload the agent session (`/new` in interactive mode) or restart the CLI agent to load the new config.

### Option C: Edit YAML directly - advanced customization

```yaml
# ~/.config/evoscientist/mcp.yaml
sequential-thinking:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-sequential-thinking"]
```

## 🛠️ Command Reference

### Syntax

```bash
EvoSci mcp add <name> <command-or-url> [args...] \
  [--transport <transport>] \
  [--tools <tool1,tool2,...>] \
  [--expose-to <agent1,agent2,...>] \
  [--header <Key:Value>]... \
  [--env <KEY=VALUE>]... \
  [--env-ref <KEY>]...
```

**Options:**

| Flag | Short | Description |
|------|-------|-------------|
| `--transport` | | Transport type (auto-inferred if omitted) |
| `--tools` | `-t` | Comma-separated tool allowlist, supports glob wildcards (omit = all tools) |
| `--expose-to` | `-e` | Comma-separated target agents (default: `main`) |
| `--header` | `-H` | HTTP header as `Key:Value` (repeatable) |
| `--env` | | Env var as `KEY=VALUE` for stdio subprocess (repeatable) |
| `--env-ref` | | Reference an existing env var by name (repeatable) |

> [!NOTE]
> - `--transport` is auto-inferred from target: `http(s)` URL → `http`, otherwise `stdio`.
> - `--` is recommended before server args that start with `-`, so they are passed to the MCP server command.

### Management

| Command | Description |
|---------|-------------|
| `EvoSci mcp` | List configured servers |
| `EvoSci mcp list` | List configured servers |
| `EvoSci mcp config` | Show detailed config for all servers |
| `EvoSci mcp config <name>` | Show detailed config for one server |
| `EvoSci mcp add ...` | Add a server |
| `EvoSci mcp edit ...` | Edit an existing server |
| `EvoSci mcp remove <name>` | Remove a server |

> [!TIP]
> All commands also work as interactive slash commands: `/mcp`, `/mcp list`, `/mcp config`, `/mcp add ...`, `/mcp edit ...`, `/mcp remove <name>`.

### Examples

```bash
# Local stdio server
EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

# Remote HTTP server (transport auto-detected)
EvoSci mcp add docs-langchain https://docs.langchain.com/mcp

# SSE endpoint (explicit transport override)
EvoSci mcp add research-sse https://example.com/sse --transport sse

# With tool routing to sub-agent + env var reference
EvoSci mcp add brave-search npx --env-ref BRAVE_API_KEY -e research-agent -- -y @modelcontextprotocol/server-brave-search

# With tool allowlist (glob wildcards)
EvoSci mcp add fs npx -t "read_*,write_*" -- -y @modelcontextprotocol/server-filesystem /workspace

# Context7 routed to multiple agents
EvoSci mcp add context7 npx -e main,research-agent,code-agent -- -y @upstash/context7-mcp@latest
```

<details>
<summary><strong>More examples</strong></summary>

```bash
# stdio transport (auto-detected from command)
EvoSci mcp add filesystem npx -- -y @modelcontextprotocol/server-filesystem /tmp

# http transport (auto-detected from URL)
EvoSci mcp add brave-search http://localhost:8080/mcp -H "Authorization:Bearer ${BRAVE_API_KEY}"

# sse transport, routed to a specific agent
EvoSci mcp add my-sse http://localhost:9090/sse --transport sse -e research-agent

# With tool allowlist (supports glob wildcards)
EvoSci mcp add fs npx -- -y @modelcontextprotocol/server-filesystem /tmp -t "read_*,write_*"
```

</details>

### Editing Servers

Update individual fields without re-adding:

```bash
# Change routing
EvoSci mcp edit filesystem --expose-to main,code-agent

# Set a tool allowlist (supports glob wildcards)
EvoSci mcp edit filesystem --tools "read_*,write_*"

# Clear a tool allowlist (pass all tools)
EvoSci mcp edit filesystem --tools none

# Change URL
EvoSci mcp edit my-api --url http://new-host:9090/mcp
```

## 🔀 Tool Routing

Use `expose_to` to control which agents receive each server's tools:

```yaml
postgres:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
  expose_to: [data-analysis-agent]

github:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  expose_to: [main, research-agent]
```

**Available agents:**

| Agent | Role |
|-------|------|
| `main` | Main orchestrator (default target) |
| `planner-agent` | Experiment planning |
| `research-agent` | Literature search |
| `code-agent` | Code writing |
| `debug-agent` | Debugging |
| `data-analysis-agent` | Data analysis |
| `writing-agent` | Report writing |

> [!NOTE]
> Tools routed to sub-agents are injected automatically — no need to edit any `EvoScientist/subagents/*.yaml` file.

## 🔍 Tool Filtering with Wildcards

Use the `tools` field to filter which tools from a server are exposed. Supports glob-style wildcards:

```yaml
# Exact matching (original behavior)
exa:
  transport: http
  url: https://mcp.exa.ai/mcp
  tools:
    - web_search_exa
    - get_code_context_exa
    - company_research_exa

# Wildcard: all tools ending with _exa
exa:
  transport: http
  url: https://mcp.exa.ai/mcp
  tools:
    - "*_exa"

# Multiple patterns
filesystem:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
  tools:
    - "read_*"
    - "write_*"
    - "list_*"

# Mix wildcards and exact matches
mixed:
  transport: http
  url: https://example.com/mcp
  tools:
    - "search_*"
    - "get_metadata"
```

### Wildcard Patterns

| Pattern | Matches | Example |
|---------|---------|---------|
| `*` | Any sequence of characters | `*_exa` matches `web_search_exa`, `get_code_context_exa` |
| `?` | Any single character | `tool_?` matches `tool_1`, `tool_2` but not `tool_10` |
| `[seq]` | Any character in sequence | `tool_[abc]` matches `tool_a`, `tool_b`, `tool_c` |
| `[0-9]` | Any character in range | `version_[0-9]` matches `version_0` through `version_9` |
| `[!seq]` | Any character NOT in sequence | `tool_[!0-9]` matches `tool_a` but not `tool_1` |

## 📄 Config File

> Path: `~/.config/evoscientist/mcp.yaml` (or `$XDG_CONFIG_HOME/evoscientist/mcp.yaml`)

### Config Fields

| Field | Required | Description |
|-------|----------|-------------|
| `transport` | Yes | `stdio`, `http`, `streamable_http`, `sse`, `websocket` |
| `command` | stdio only | Command to run (e.g. `npx`) |
| `args` | stdio only | Arguments list |
| `env` | No | Environment variables for subprocess |
| `url` | http/sse/ws | Server URL |
| `headers` | No | HTTP headers (e.g. auth tokens) |
| `tools` | No | Tool allowlist with glob wildcards (omit = all tools) |
| `expose_to` | No | Target agents (default: `["main"]`) |

> [!TIP]
> Use `${VAR}` in YAML values to reference environment variables. Missing variables are replaced with empty string and logged as a warning.

### Supported Transports

| Transport | Config Fields |
|-----------|---------------|
| `stdio` | `command`, `args`, `env` (optional) |
| `http` | `url`, `headers` (optional) |
| `streamable_http` | `url`, `headers` (optional) |
| `sse` | `url`, `headers` (optional) |
| `websocket` | `url` |

<details>
<summary><strong>Full YAML example with inline annotations</strong></summary>

```yaml
filesystem:
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
  tools: ["read_*", "write_*"]       # optional allowlist with wildcards (omit = all tools)
  expose_to: [main, code-agent]      # optional routing (omit = ["main"])

brave-search:
  transport: http
  url: "http://localhost:8080/mcp"
  headers:
    Authorization: "Bearer ${BRAVE_API_KEY}"
  expose_to: [research-agent]
```

</details>

## ⚙️ How It Works

1. On agent startup (`/new` or `create_cli_agent()`), reads `~/.config/evoscientist/mcp.yaml`
2. Connects to each server via the configured transport
3. Retrieves available tools from each server
4. Filters tools by `tools` allowlist (if set)
5. Routes tools to target agents by `expose_to`
6. Tools are injected into the agent's tool list automatically

> [!NOTE]
> MCP servers that fail to connect are skipped with a warning — they don't block startup. Tools are cached by config signature to avoid redundant loading.

## 🔧 Troubleshooting

<details open>
<summary><strong>Dependency missing (<code>langchain-mcp-adapters</code>)</strong></summary>

Startup warning about MCP adapter missing:

```bash
pip install langchain-mcp-adapters
```

</details>

<details>
<summary><strong><code>npx</code> not available</strong></summary>

stdio server fails to start — install Node.js and `npx`, or replace `npx` with a command available in your environment.

</details>

<details>
<summary><strong>Windows: stdio server fails with <code>[Errno 9] Bad file descriptor</code></strong></summary>

On Windows, the TUI redirects `sys.stderr` to an in-memory capture whose `fileno()` is not a real OS handle. The MCP SDK forwards that `stderr` to the stdio server subprocess, and `subprocess.Popen` rejects the invalid handle with `OSError: [Errno 9] Bad file descriptor` — so only stdio servers fail to load (HTTP/SSE servers are unaffected).

EvoScientist wraps the SDK's stdio client so that, whenever the configured `stderr` has no usable file descriptor, it falls back to the original console handle (`sys.__stderr__`, or `os.devnull` in GUI hosts). If you still see this error, run from a real console (not `pythonw.exe`) and check the server's own startup output.

</details>

<details>
<summary><strong><code>--env-ref</code> or <code>${VAR}</code> not resolving</strong></summary>

Auth header/env becomes empty — ensure the variable exists in your environment before launching EvoScientist. `${VAR}` interpolation happens at runtime; missing vars are replaced with empty strings and logged as warnings.

</details>

<details>
<summary><strong>Server connects but no tools appear</strong></summary>

Server exists in config but agent doesn't use MCP tools:
- Confirm endpoint/command is healthy outside EvoScientist
- Check auth headers/env
- Check whether `tools` filter is too strict

</details>

<details>
<summary><strong>Tool routing not taking effect</strong></summary>

Tool is loaded but not available where expected:
- Verify `expose_to` target names are correct (see [Available agents](#-tool-routing) above)
- Reload the session with `/new` after config changes
- Include `main` in `expose_to` if you need tools in the main agent

</details>

## 🔒 Security

> [!WARNING]
> Do not hardcode secrets in `mcp.yaml`. Use `${VAR}` in YAML or `--env-ref KEY` in CLI to reference environment variables.

- For filesystem-style servers, expose only minimum required paths
- Use `tools` allowlists to limit which tools are loaded
- Use `expose_to` to restrict which agents can access each server
- Prefer least privilege for both tool scope and agent access
