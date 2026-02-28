# Contributing to EvoScientist

This guide serves both **human contributors** and **AI coding agents** (Claude Code, Cursor, etc.). It provides the architectural context, patterns, and extension points needed to safely and effectively modify the codebase.

## Core Principles

1. **Less is more.** The smallest change that achieves the goal is the best change.
2. **Keep solutions simple, composable, and maintainable.**
3. **Extend existing abstractions before introducing new paths.** If a change requires a new path, explain why existing paths cannot be safely extended.

---

## Project Overview

EvoScientist is a multi-agent AI system for automated scientific experimentation and discovery. It orchestrates specialized sub-agents that plan experiments, search literature, write code, debug, analyze data, and draft reports.

| Fact | Value |
|------|-------|
| Language | Python 3.11+ |
| License | MIT |
| Framework | [DeepAgents](https://github.com/langchain-ai/deepagents) + [LangChain](https://python.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Default model | `claude-sonnet-4-5` (Anthropic) |
| Tests | ~830 across 35 files, no API keys needed |
| Config file | `~/.config/evoscientist/config.yaml` |

### Sub-Agents (defined in `EvoScientist/subagent.yaml`)

| Agent | Purpose |
|-------|---------|
| `planner-agent` | Creates and updates experimental plans (no web search, no implementation) |
| `research-agent` | Web research for methods, baselines, and datasets (Tavily search) |
| `code-agent` | Implements experiment code and runnable scripts |
| `debug-agent` | Reproduces failures, identifies root causes, applies minimal fixes |
| `data-analysis-agent` | Computes metrics, creates plots, summarizes insights |
| `writing-agent` | Drafts paper-ready Markdown experiment reports |

### Data Flow

```txt
User Input (CLI / TUI / 10 Channel Integrations)
    |
CLI (cli/) / TUI (cli/tui_*) / Channel Server (channels/)
    |
Main Agent (EvoScientist.py) -- create_deep_agent()
    +-- System Prompt (prompts.py)
    +-- Chat Model (llm/ -- multi-provider)
    +-- Middleware: Memory (middleware/memory.py)
    +-- Backend: CompositeBackend (backends.py)
    |     / --> CustomSandboxBackend (workspace read/write + execute)
    |     /skills/ --> MergedReadOnlyBackend (user > built-in)
    |     /memory/ --> FilesystemBackend (persistent cross-session)
    +-- MCP Tools (mcp/ -- optional, cached by config signature)
    |
task tool --> Delegates to Sub-Agents
    |
Stream Events --> Emitter --> Tracker --> State --> Rich Display / TUI
```

---

## Development Setup

### Prerequisites

- Python 3.11 or higher
- `uv` (recommended) or `pip`

### Install

```bash
cd EvoScientist

# Option A: uv (recommended)
uv sync --dev

# Option B: pip (requires pip >=25.0 for PEP 735 dependency groups)
pip install -e . --dependency-groups dev

# Channel extras (optional)
pip install -e ".[telegram]"      # single channel
pip install -e ".[all-channels]"  # all channels
```

### Configure

```bash
# Interactive wizard (recommended)
EvoSci onboard

# Manual alternative
# Edit ~/.config/evoscientist/config.yaml directly, or:
EvoSci config set anthropic_api_key sk-ant-...
EvoSci config set tavily_api_key tvly-...
```

**Config priority chain** (highest wins):

| Priority | Source | Example |
|----------|--------|---------|
| 1 | CLI arguments | `--model gpt-4o` |
| 2 | Environment variables | `ANTHROPIC_API_KEY=sk-...` |
| 3 | Config file | `~/.config/evoscientist/config.yaml` |
| 4 | Defaults | `provider: anthropic`, `model: claude-sonnet-4-5` |

Implementation: `config/settings.py` — `get_effective_config()` merges all four sources.

### Run

```bash
python -m EvoScientist          # interactive mode (daemon workspace)
python -m EvoScientist -p "..."  # single-shot query
EvoSci                           # alias (same as above)
langgraph dev                    # LangGraph dev server
```

### Entry Points (from `pyproject.toml`)

All four CLI aliases (`evoscientist`, `EvoScientist`, `evosci`, `EvoSci`) resolve to `EvoScientist.cli:main`.

### Workspace Modes

| Mode | Workspace | `/new` behavior | Use case |
|------|-----------|-----------------|----------|
| `daemon` (default) | `workspace/` | New thread, same workspace | Long-term development |
| `run` | `workspace/runs/<timestamp>/` | New thread + new workspace | Isolated experiments |

Workspace priority: `--workdir` > `--mode` > `default_workdir` config > `default_mode` config > cwd.

---

## Project Structure

```txt
EvoScientist/EvoScientist/
|-- __init__.py          # Lazy-loaded exports (__getattr__ pattern)
|-- __main__.py          # python -m EvoScientist entry
|-- EvoScientist.py      # Agent graph construction, create_cli_agent()
|-- backends.py          # CompositeBackend, CustomSandboxBackend, MergedReadOnlyBackend
|-- prompts.py           # 3-layer prompt: EXPERIMENT_WORKFLOW + DELEGATION_STRATEGY + RESEARCHER_INSTRUCTIONS
|-- paths.py             # Workspace path management: set_workspace_root(), resolve_virtual_path()
|-- sessions.py          # SQLite checkpoint persistence for LangGraph threads
|-- utils.py             # Subagent loader, shared helpers
|-- subagent.yaml        # 6 sub-agent definitions (prompts, tools, descriptions)
|
|-- cli/                 # CLI module
|   |-- _app.py          # Typer app instances (main + sub-apps)
|   |-- interactive.py   # Main interactive loop, Rich Live display
|   |-- commands.py      # Workspace handling, config CLI, MCP commands
|   |-- agent.py         # Agent loading and session workspace creation
|   |-- channel.py       # Queue-based channel integration (shares agent session)
|   |-- mcp_ui.py        # MCP server management UI
|   |-- skills_cmd.py    # /install-skill, /uninstall-skill, /skills commands
|   |-- clipboard.py     # Clipboard utilities
|   |-- _constants.py    # CLI constants
|   |-- tui_interactive.py  # Textual-based TUI (alternative to Rich CLI)
|   |-- tui_backends.py     # TUI backend adapters
|   |-- tui_runtime.py      # TUI runtime loop
|   +-- widgets/         # 8 Textual widgets (assistant, loading, subagent, system, thinking, todo, tool_call, user)
|
|-- config/              # Configuration module
|   |-- __init__.py      # Re-exports
|   |-- settings.py      # EvoScientistConfig dataclass, get_effective_config(), apply_config_to_env()
|   +-- onboard.py       # Interactive setup wizard (questionary-based)
|
|-- llm/                 # LLM provider module
|   |-- __init__.py      # Re-exports
|   +-- models.py        # MODELS registry, get_chat_model(), DEFAULT_MODEL
|
|-- middleware/           # Agent middleware
|   |-- __init__.py      # Re-exports, create_memory_middleware()
|   |-- memory.py        # EvoMemoryMiddleware (injection + extraction)
|   +-- tool_error_handler.py  # ToolErrorHandlerMiddleware
|
|-- tools/               # Custom tools
|   |-- __init__.py      # Re-exports
|   |-- search.py        # tavily_search, fetch_webpage_content
|   |-- think.py         # think_tool (structured reflection)
|   |-- skill_manager.py # skill_manager tool (install/list/uninstall)
|   +-- skills_manager.py  # Core skill install logic (local, GitHub, batch)
|
|-- mcp/                 # MCP integration
|   |-- __init__.py      # load_mcp_tools()
|   +-- client.py        # load_mcp_config(), MCP transport handling
|
|-- stream/              # Streaming display pipeline
|   |-- __init__.py      # Re-exports
|   |-- emitter.py       # StreamEventEmitter
|   |-- tracker.py       # ToolCallTracker (incremental JSON parsing)
|   |-- state.py         # StreamState, SubAgentState, sub-agent name resolution
|   |-- events.py        # Event types
|   |-- display.py       # Rich rendering
|   |-- formatter.py     # Output formatting
|   +-- utils.py         # Stream helpers
|
|-- channels/            # 10 messaging channels
|   |-- __init__.py
|   |-- base.py          # Channel ABC, IncomingMessage, OutgoingMessage, chunk_text()
|   |-- bus/             # Message bus (events, routing)
|   |-- capabilities.py  # ChannelCapabilities feature abstraction
|   |-- consumer.py      # Inbound message processing
|   |-- formatter.py     # UnifiedFormatter
|   |-- middleware.py     # TypingManager
|   |-- plugin.py        # ChannelPlugin, ChannelMeta, ReloadPolicy
|   |-- config.py        # Channel config
|   |-- retry.py         # Retry logic
|   |-- mixins.py        # Shared mixins
|   |-- standalone.py    # Standalone channel server
|   |-- imessage/        # macOS iMessage (imsg CLI + JSON-RPC)
|   |-- telegram/        # Telegram Bot API
|   |-- discord/         # Discord bot
|   |-- slack/           # Slack bot
|   |-- wechat/          # WeChat (WeCom / MP)
|   |-- dingtalk/        # DingTalk
|   |-- feishu/          # Feishu (Lark)
|   |-- email/           # Email (IMAP/SMTP)
|   |-- qq/              # QQ (botpy)
|   +-- signal/          # Signal (signal-cli)
|
+-- skills/              # Built-in skills (read-only to agent)
    |-- find-skills/     # Skill discovery
    +-- skill-creator/   # Skill creation wizard
```

Additional built-in skills (`agent-swarm-protocol`, `paper-planning`, `paper-review`, `paper-writing`) can be installed as user skills under `workspace/skills/`.

### Tests

35 test files under `tests/`, following `test_*.py` naming. Tests are placed near the affected domain:

```txt
tests/
|-- conftest.py                  # Shared fixtures
|-- test_backends.py             # CustomSandboxBackend, validate_command
|-- test_config.py               # EvoScientistConfig, get_effective_config
|-- test_llm.py                  # Model registry, get_chat_model
|-- test_stream_state.py         # StreamState, sub-agent name resolution
|-- test_stream_emitter.py       # StreamEventEmitter
|-- test_stream_tracker.py       # ToolCallTracker
|-- test_tools.py                # tavily_search, think_tool, skill_manager
|-- test_skills_manager.py       # Skill install/uninstall/batch
|-- test_memory_merge.py         # EvoMemoryMiddleware extraction
|-- test_mcp_client.py           # MCP config loading, tool routing
|-- test_agent_mcp_cache.py      # MCP caching by config signature
|-- test_sessions.py             # SQLite checkpointer
|-- test_paths.py                # Workspace path management
|-- test_telegram_channel.py     # Channel-specific tests
|-- test_discord_channel.py
|-- test_slack_channel.py
|-- test_wechat_channel.py
|-- test_feishu_channel.py
|-- test_dingtalk_channel.py
|-- test_bus_integration.py      # Message bus integration
|-- ...                          # (see tests/ for full list)
```

---

## Architecture Deep Dive

### Agent Construction Pipeline

The sequence from config to running agent (`EvoScientist.py`):

1. **`_ensure_config()`** — Loads and caches `EvoScientistConfig` from 4 sources, calls `apply_config_to_env()` to set API keys as env vars.
2. **`_ensure_chat_model()`** — Creates a LangChain chat model via `get_chat_model(model, provider)` from `llm/models.py`. Auto-enables extended thinking for Anthropic models.
3. **`_ensure_system_prompt()`** — Builds the 3-layer system prompt from `prompts.py`.
4. **Backend construction** — `CompositeBackend` with 3 routes (workspace, skills, memory).
5. **Middleware construction** — `ToolErrorHandlerMiddleware` + `EvoMemoryMiddleware`.
6. **MCP tools** — `_load_mcp_tools_cached()` loads tools from `~/.config/evoscientist/mcp.yaml`, cached by config signature hash.
7. **Sub-agent loading** — `load_subagents()` reads `subagent.yaml`, wires tools from registry, injects `ToolErrorHandlerMiddleware` into each sub-agent.
8. **`create_deep_agent(**kwargs)`** — DeepAgents constructs the LangGraph agent graph.
9. **`.with_config({"recursion_limit": 500})`** — Sets LangGraph recursion limit.

Two variants exist:
- **`EvoScientist_agent`** — Lazy-loaded default (no checkpointer). For `langgraph dev`, notebooks, LangSmith.
- **`create_cli_agent(workspace_dir)`** — With checkpointer for CLI multi-turn. Constructs fresh backends on every call so runtime `set_workspace_root()` changes are respected.

### Backend Architecture

`CompositeBackend` (`backends.py`) routes virtual paths to three backends:

| Route | Backend | Purpose |
|-------|---------|---------|
| `/` (default) | `CustomSandboxBackend` | Read/write workspace with shell execution |
| `/skills/` | `MergedReadOnlyBackend` | Read-only, user skills override built-in |
| `/memory/` | `FilesystemBackend` | Persistent memory, shared across sessions |

**`CustomSandboxBackend`** extends `FilesystemBackend` + `LocalShellBackend`:
- **Path sanitization** — `_resolve_path()` auto-corrects LLM hallucinated paths: strips `/workspace/` prefixes, system path prefixes (`/Users/`, `/home/`, etc.). See `_SYSTEM_PATH_PREFIXES` at `backends.py:21`.
- **Command validation** — `validate_command()` blocks directory traversal (`..`), dangerous commands (`sudo`, `chmod`, `chown`, `mkfs`, `dd`, `shutdown`, `reboot`), and dangerous patterns (`rm -rf /`). See `BLOCKED_COMMANDS` at `backends.py:35`.
- **Limits** — 300s execution timeout, 100KB output limit.

**`MergedReadOnlyBackend`** (`backends.py`) merges two directories for `ls` and `read_file`. User skills (`workspace/skills/`) take priority over built-in skills (`EvoScientist/skills/`).

### Configuration System

`config/settings.py`:
- **`EvoScientistConfig`** — Dataclass with all settings: API keys (Anthropic, OpenAI, Google, NVIDIA, Tavily, SiliconFlow, OpenRouter, custom, Ollama), LLM settings (provider, model), workspace settings (default_mode, default_workdir), UI settings (show_thinking, ui_backend), and channel-specific settings.
- **`get_effective_config(cli_overrides)`** — Merges 4 sources in priority order (CLI > env > file > defaults).
- **`apply_config_to_env(config)`** — Sets API keys as env vars for downstream libraries (LangChain, Tavily).
- **`load_config()` / `save_config()`** — YAML file I/O at `~/.config/evoscientist/config.yaml`.

`config/onboard.py`:
- **Interactive wizard** — Uses `questionary` to prompt for API keys, validates them, and saves to config file.

### Memory Middleware

`middleware/memory.py` — `EvoMemoryMiddleware` provides two mechanisms:

1. **Injection** (every LLM call): Reads `/memory/MEMORY.md` and appends its contents to the system prompt. The agent always has accumulated context.
2. **Extraction** (threshold-triggered): When conversation exceeds a configurable message count (default 20), uses an LLM call to extract structured facts (`ExtractedMemory` Pydantic model) and merges them into MEMORY.md sections (user_profile, research_preferences, experiment_conclusions, etc.).

Memory always uses the shared `MEMORY_DIR` regardless of workspace mode, ensuring cross-session persistence.

### Streaming Pipeline

`stream/` — The rendering pipeline for CLI output:

```txt
StreamEventEmitter (emitter.py)
    |  Converts raw LangGraph stream events into typed events
    v
ToolCallTracker (tracker.py)
    |  Incremental JSON parsing of tool call arguments
    v
StreamState / SubAgentState (state.py)
    |  Tracks active sub-agents, thinking panels, tool calls
    v
Rich rendering (display.py) / TUI widgets (cli/widgets/)
```

**Sub-agent name resolution** — `_get_subagent_name()` in `events.py` and `_resolve_subagent_name()` in `state.py` use a 6-level priority chain:
1. `metadata["lc_agent_name"]` — most reliable; filters out generic names
2. Task ID from namespace tuple → maps to announced task `tool_call_id`
3. Task ID from metadata fields
4. Cached real name (skips "sub-agent" entries)
5. Queue-based assignment from `_register_task_tool_call()`
6. Fallback "sub-agent" (NOT cached, allows re-resolution on later events)

### Channel Architecture

`channels/base.py` — `Channel` ABC defines the interface all channels implement:

- **`start()`** — Start listening for incoming messages (no callback; inbound messages arrive via the bus)
- **`_send_chunk(chat_id, formatted_text, raw_text, reply_to, metadata)`** — Send a single message chunk to the channel
- **Text chunking** — `chunk_text()` splits messages at code block boundaries > paragraph breaks > newlines > spaces > hard cut

**Dual-thread design**: Bus thread enqueues `ChannelMessage` on a thread-safe queue. Main CLI thread polls queue, processes with `_run_streaming` (Rich Live, real-time), sets response. Bus thread publishes outbound.

**Display format** (in `cli/interactive.py`):

```txt
❯ message content              <-- user input style (❯ bold blue)
[channel: Received from sender] <-- dim text, sender in cyan
-----------------
                                <-- Rich Live streaming (thinking, tools, markdown)
[channel: Replied to sender]    <-- dim text, sender in cyan
-----------------
```

---

## Key Patterns & Invariants

Each pattern includes WHY it exists, so contributors understand the reason behind the rule.

### 1. Lazy Loading via `__getattr__`

**Where**: `EvoScientist/__init__.py`, `EvoScientist/EvoScientist.py`

**Why**: Non-agent CLI commands (`EvoSci config list`, `EvoSci onboard`) must start instantly without importing heavy dependencies (DeepAgents, LangChain, LangGraph). The `__init__.py` `_EXPORTS` dict maps attribute names to `(module, attr)` tuples; first access triggers import and caches the result.

**Rule**: Never import `deepagents`, `langchain`, or `langgraph` at the top level of entry-point modules. Defer to function/method scope.

### 2. Fresh Backends on Every `create_cli_agent()` Call

**Where**: `EvoScientist.py:318` — `create_cli_agent()`

**Why**: Users can call `set_workspace_root()` between sessions (via `--workdir`, `/new`, `--mode=run`). A stale backend would point to the wrong directory. Fresh construction on every call ensures current paths are respected.

**Rule**: Never cache or reuse backend instances across agent creation calls.

### 3. MCP Tools Cached by Config Signature Hash

**Where**: `EvoScientist.py:100` — `_mcp_config_signature()`, `_load_mcp_tools_cached()`

**Why**: MCP tool loading spawns subprocesses and connects to servers — expensive. But on `/new` (new session), the agent is recreated. Caching by config signature avoids redundant reconnection when config hasn't changed.

**Rule**: When adding MCP-related config fields, ensure they're included in the serialized config signature.

### 4. Memory Always Uses Shared Directory

**Where**: `paths.py` — `MEMORY_DIR`, `EvoScientist.py:375`

**Why**: Memory must persist across sessions, even in `run` mode where each session gets an isolated workspace. The `/memory/` backend route always points to the shared `MEMORY_DIR`, never the per-session workspace.

**Rule**: Never route `/memory/` to a per-session directory.

### 5. Path Sanitization — Auto-Corrects Hallucinated Paths

**Where**: `backends.py:21` — `_SYSTEM_PATH_PREFIXES`, `_resolve_path()`

**Why**: LLMs frequently hallucinate absolute paths like `/Users/alice/project/file.py` or `/workspace/file.py` instead of using virtual paths. The backend strips these prefixes automatically, converting them to workspace-relative paths.

**Rule**: When adding new path handling, always use `_resolve_path()`. Never trust raw LLM-provided paths.

### 6. Command Validation — Blocks Traversal + Dangerous Commands

**Where**: `backends.py:28-43` — `BLOCKED_PATTERNS`, `BLOCKED_COMMANDS`, `validate_command()`

**Why**: The sandbox must prevent workspace escape and system damage. Commands are split into segments (handling `&&`, `||`, `;`, `|`) and each base command is checked against blocklists.

**Rule**: When extending the executor, always route through `validate_command()`. Add new dangerous commands to `BLOCKED_COMMANDS`.

### 7. Channel = Text Injection into CLI

**Where**: `cli/interactive.py` — `_process_channel_message()`

**Why**: Channel messages must look exactly as if the user typed them at the prompt. The CLI is the only rendering engine; channels are just different input sources. This ensures consistent behavior regardless of input source.

**Rules**:
- Use `_run_streaming` (Rich Live, real-time) — NOT `_astream_to_console` (static, end-of-stream).
- Labels use `dim` style; sender ID in `cyan`.
- After response, manually redraw the prompt via `sys.stdout.write` (prompt_toolkit doesn't know the terminal was modified by Rich Live).

### 8. Incremental JSON Parsing

**Where**: `stream/tracker.py` — `ToolCallTracker`

**Why**: Tool call arguments stream incrementally — `subagent_type` may be empty initially, then filled in later as more tokens arrive. The tracker parses partial JSON to extract fields as early as possible for responsive UI updates.

**Rule**: Never assume tool call args are complete on first event. Handle partial/empty fields gracefully.

### 9. Sub-Agent Name Resolution 6-Level Priority

**Where**: `stream/events.py` — `_get_subagent_name()`, `stream/state.py` — `_resolve_subagent_name()`

**Why**: Sub-agent identity comes from multiple unreliable sources. The 6-level priority chain ensures the most reliable source wins. Critically, the fallback "sub-agent" name is NOT cached, allowing re-resolution when a more reliable source arrives later.

**Rule**: When modifying sub-agent handling, preserve the fallback non-caching behavior.

### 10. Dual-Thread Channel Architecture

**Where**: `channels/bus/`, `cli/channel.py`, `cli/interactive.py`

**Why**: Channel I/O (network polling) must not block the CLI's Rich Live display loop. A dedicated bus thread handles inbound/outbound messages via a thread-safe queue, while the CLI thread polls the queue and processes messages with full streaming support.

**Rule**: Never perform blocking I/O in the CLI's main thread. Use the message bus queue.

---

## Extension Guides

Each guide follows: **where to add** → **interface to implement** → **how to wire** → **how to test**.

### Adding a New Tool

1. **Create the tool** in `EvoScientist/tools/` using the LangChain `@tool` decorator:

```python
# EvoScientist/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """One-line description shown to the agent.

    Args:
        query: What to process.
    """
    return f"Result for {query}"
```

2. **Re-export** from `EvoScientist/tools/__init__.py`:

```python
from .my_tool import my_tool

__all__ = [..., "my_tool"]
```

3. **Wire to main agent** — Add to `_build_base_kwargs()` in `EvoScientist.py`:

```python
base_tools = [think_tool, skill_manager, my_tool]
```

4. **Wire to sub-agents** (if needed) — Add to the tool registry and reference in `subagent.yaml`:

```python
# EvoScientist.py - in _build_base_kwargs()
tool_registry = {"think_tool": think_tool, "tavily_search": tavily_search, "my_tool": my_tool}
```

```yaml
# subagent.yaml
my-sub-agent:
  tools: [my_tool, think_tool]
```

5. **Test** — Add tests in `tests/test_tools.py` using `unittest.mock` for external services.

### Adding a New Channel

1. **Create the channel directory**: `EvoScientist/channels/mychannel/`

2. **Implement the `Channel` ABC** (`channels/base.py`):

```python
# EvoScientist/channels/mychannel/channel.py
from ..base import Channel, IncomingMessage

class MyChannel(Channel):
    name = "mychannel"

    async def start(self) -> None:
        """Start listening. Inbound messages arrive via the bus (set_bus)."""
        ...

    async def _send_chunk(
        self, chat_id: str, formatted_text: str, raw_text: str,
        reply_to: str | None, metadata: dict,
    ) -> None:
        """Send a single text chunk to the channel."""
        ...

    async def stop(self) -> None:
        """Graceful shutdown."""
        ...
```

3. **Add capabilities** — Return a `ChannelCapabilities` from your channel describing supported features (max text length, media support, etc.).

4. **Add config fields** — Add channel-specific settings to `EvoScientistConfig` in `config/settings.py`:

```python
mychannel_api_key: str = ""
mychannel_allowed_senders: str = ""
```

5. **Register** — Add to the channel manager in `channels/channel_manager.py` and the onboard wizard in `config/onboard.py`.

6. **Add optional dependency** — In `pyproject.toml`:

```toml
[project.optional-dependencies]
mychannel = ["mychannel-sdk>=1.0"]
```

7. **Test** — Add `tests/test_mychannel_channel.py`. Use a stub/mock for the external SDK.

### Adding a New Sub-Agent

1. **Add entry to `subagent.yaml`**:

```yaml
my-agent:
  description: "One-line description of what this agent does."
  tools: [think_tool]  # tools from the registry
  system_prompt: |
    You are the my-agent. Your role is...

    Guidelines:
    - ...

    When responding, include:
    - ...
```

Or reference a shared prompt:
```yaml
my-agent:
  description: "..."
  tools: [think_tool]
  system_prompt_ref: MY_PROMPT_CONSTANT  # must exist in prompts.py
```

2. **Add tools to registry** (if using new tools) — Update `tool_registry` in `_build_base_kwargs()` and `load_mcp_and_build_kwargs()` in `EvoScientist.py`.

3. **Test** — Verify the agent appears in loaded subagents. Add tests for any new tools it uses.

### Adding a New Middleware

1. **Subclass `AgentMiddleware`** from LangChain:

```python
# EvoScientist/middleware/my_middleware.py
from deepagents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from typing import Awaitable, Callable

class MyMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Modify the request before it reaches the LLM (sync)."""
        # Modify request here, then pass to handler
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Modify the request before it reaches the LLM (async)."""
        return await handler(request)
```

2. **Re-export** from `EvoScientist/middleware/__init__.py`.

3. **Wire into the pipeline** — Add to `_get_default_middleware()` and the middleware list in `create_cli_agent()` in `EvoScientist.py`:

```python
mw = [
    ToolErrorHandlerMiddleware(),
    MyMiddleware(),
    create_memory_middleware(memory_dir, extraction_model=model),
]
```

4. **Test** — Add `tests/test_my_middleware.py`. Mock the LLM and verify request/response modifications.

### Adding a New Config Field

1. **Add the field** to `EvoScientistConfig` in `config/settings.py`:

```python
@dataclass
class EvoScientistConfig:
    ...
    my_setting: str = "default_value"
```

2. **Add env var mapping** (if applicable) — Update `_ENV_MAP` in `config/settings.py` or `apply_config_to_env()`.

3. **Add to onboard wizard** (if user-facing) — Update `config/onboard.py` to prompt for the new field during `EvoSci onboard`.

4. **Use the field** — Access via `cfg = _ensure_config()` in `EvoScientist.py` or `get_effective_config()` elsewhere.

5. **Test** — Add test cases in `tests/test_config.py` for default value, env override, and file override.

### Adding a New LLM Provider

1. **Add model entries** to `_MODEL_ENTRIES` in `llm/models.py`:

```python
_MODEL_ENTRIES = [
    ...
    ("my-model", "my-provider/my-model-id", "myprovider"),
]
```

2. **Update `get_chat_model()`** in `llm/models.py` — Add provider-specific initialization if needed (API key handling, special kwargs).

3. **Add config fields** — Add `myprovider_api_key` to `EvoScientistConfig` and the env var mapping.

4. **Add dependency** — Add the LangChain provider package to `pyproject.toml`:

```toml
dependencies = [
    ...
    "langchain-myprovider>=1.0",
]
```

5. **Update onboard wizard** — Add API key prompt in `config/onboard.py`.

6. **Test** — Add provider tests in `tests/test_llm.py`. Mock the chat model constructor.

---

## Code Quality Standards

### Style

- **Indentation**: 4 spaces (no tabs).
- **Naming**: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- **Type hints**: Required for public APIs. Types go in function signatures, NOT in docstrings.
- **Docstrings**: Google-style. Focus on "why" rather than "what". Keep concise.

```python
def send_request(url: str, *, timeout: int = 30) -> dict[str, Any]:
    """Fetch data from the upstream API.

    Retries once on timeout before raising.

    Args:
        url: The API endpoint to call.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response body.

    Raises:
        httpx.TimeoutException: If both attempts time out.
    """
```

- **American English** spelling (e.g., "behavior", not "behaviour").
- **Single backticks** for inline code in docstrings (`` `code` ``), not Sphinx-style double backticks.

### Linting

```bash
ruff check .
```

Prefer inline `# noqa: RULE` for individual exceptions over `per-file-ignores` in `pyproject.toml`. Reserve `per-file-ignores` for categorical policy (e.g., `"tests/**" = ["S101"]`).

### Security

- No `eval()`, `exec()`, or `pickle` on user-controlled input.
- No bare `except:` — always catch specific exceptions.
- Proper resource cleanup (file handles, connections, threads).
- Remove unreachable/commented code before committing.
- Use `msg` variable for error messages before raising.

---

## Testing Requirements

### Running Tests

```bash
# All tests (~830 tests, no API keys needed)
pytest -v

# Single file
pytest tests/test_stream_state.py -v

# Single class
pytest tests/test_stream_state.py::TestResolveSubagentName -v

# With coverage
pytest --cov=EvoScientist --cov-report=term-missing
```

### Requirements

- Every behavior change needs a test. Every bug fix needs a regression test.
- Tests must pass on Python 3.11 and 3.12.
- All tests have a 30-second timeout (`pytest-timeout`).
- No API keys or network calls in unit tests — use `unittest.mock`.

### Test Patterns

- **Placement**: Tests near the affected domain — `EvoScientist/llm/...` → `tests/test_llm.py`.
- **Mocking**: Use `unittest.mock.patch` for external services, API calls, and file system operations.
- **Stub channels**: Use `StubChannel` (see `tests/test_channel_comprehensive.py`) for channel tests rather than real connections.
- **Fixtures**: See `tests/conftest.py` for shared fixtures (temp directories, mock configs, etc.).

### CI Workflows

Three GitHub Actions workflows in `.github/workflows/`:

| Workflow | What it checks |
|----------|---------------|
| `test.yml` | `pytest -v` on Python 3.11 + 3.12 matrix |
| `lint.yml` | `ruff check .` |
| `build.yml` | `python -m build` |

All workflows have `timeout-minutes` set to prevent hangs.

---

## Commit & PR Standards

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

- **Lowercase** titles (except proper nouns).
- **Required scope** — always include one.
- **Allowed types**: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`, `perf`, `ci`, `build`, `style`
- **Common scopes**: `llm`, `stream`, `cli`, `channels`, `config`, `mcp`, `tools`, `backends`, `middleware`, `skills`, `asyncio`, `deps`

Examples:
```
feat(llm): add provider fallback for NVIDIA models
fix(stream): guard empty events in sub-agent name resolution
test(asyncio): cover event loop reuse edge case
docs(contributing): add extension guides
```

### Pull Request Checklist

**Description must include**:
- Clear summary of what changed and why
- Why this is the minimal solution (`less is more`)
- How backward compatibility is preserved
- Linked issue(s), if applicable
- Validation evidence (output of `ruff check .` and `pytest -v`)
- Screenshots or terminal snippets for user-facing changes
- **AI disclaimer**: Mention how AI agents were involved in the contribution

**Before requesting review**:
- [ ] `ruff check .` passes
- [ ] `pytest -v` passes
- [ ] Main flow behavior unchanged unless explicitly documented
- [ ] No feature-specific channel/pipeline introduced without architectural justification
- [ ] Documentation updated if behavior changed
- [ ] Backward compatibility preserved for CLI commands, config keys, and common workflows

---

## Agent-First Workflow

EvoScientist is commonly developed with AI agents as collaborators.

### Before Coding

Make sure your agent can clearly explain:
- What this project does
- What writing/coding style this repository follows
- What constraints must not be broken

Read **this file** (`CONTRIBUTING.md`) as primary context.

### Before Opening a PR

Ask your agent to confirm:
- The implementation is the smallest viable change
- Existing behavior is preserved
- The feature works with both defaults and custom settings
- `ruff check .` and `pytest -v` both pass

---

## Security & Configuration

### API Keys

Never commit real API keys or secrets. Configure via:
- `EvoSci onboard` (interactive wizard)
- `EvoSci config set <key> <value>`
- Environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `NVIDIA_API_KEY`, `TAVILY_API_KEY`, etc.)

Config file: `~/.config/evoscientist/config.yaml`
MCP config: `~/.config/evoscientist/mcp.yaml`

### MCP Security

MCP config supports `${VAR}` environment variable interpolation for secrets — never hardcode tokens in `mcp.yaml`.

### Backend Security

The `CustomSandboxBackend` validates all shell commands and file paths. See [Key Patterns #5 and #6](#5-path-sanitization--auto-corrects-hallucinated-paths) for details on the security model.

---

## Key Dependencies & Resources

| Package | Version | Purpose | Docs |
|---------|---------|---------|------|
| `deepagents` | >=0.4.4 | Multi-agent orchestration framework | [GitHub](https://github.com/langchain-ai/deepagents) |
| `langchain` | >=1.2.10 | LLM framework | [Docs](https://python.langchain.com/) |
| `langchain-anthropic` | >=1.3.3 | Anthropic provider (Claude) | [Docs](https://python.langchain.com/docs/integrations/chat/anthropic/) |
| `langchain-openai` | >=0.3 | OpenAI provider (GPT) | [Docs](https://python.langchain.com/docs/integrations/chat/openai/) |
| `langchain-google-genai` | >=4.2 | Google GenAI provider (Gemini) | [Docs](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) |
| `langchain-nvidia-ai-endpoints` | >=0.3 | NVIDIA provider | [Docs](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) |
| `langchain-ollama` | >=1.0 | Ollama local models | [Docs](https://python.langchain.com/docs/integrations/chat/ollama/) |
| `langgraph-cli[inmem]` | >=0.4 | Graph infrastructure | [Docs](https://langchain-ai.github.io/langgraph/) |
| `langgraph-checkpoint-sqlite` | >=3.0.0 | SQLite persistence | [Docs](https://langchain-ai.github.io/langgraph/) |
| `langchain-mcp-adapters` | >=0.1 | MCP integration | [GitHub](https://github.com/langchain-ai/langchain-mcp-adapters) |
| `tavily-python` | >=0.7 | Web search API | [Docs](https://docs.tavily.com/) |
| `rich` | >=14.0 | Terminal formatting/Live display | [Docs](https://rich.readthedocs.io/) |
| `prompt-toolkit` | >=3.0 | Interactive CLI input | [Docs](https://python-prompt-toolkit.readthedocs.io/) |
| `textual` | >=0.80 | TUI framework | [Docs](https://textual.textualize.io/) |
| `typer` | >=0.12 | CLI framework | [Docs](https://typer.tiangolo.com/) |
| `questionary` | >=2.0.1 | Interactive prompts | [GitHub](https://github.com/tmbo/questionary) |
| `httpx` | >=0.27 | Async HTTP client | [Docs](https://www.python-httpx.org/) |
| `markdownify` | >=0.14 | HTML to Markdown | [PyPI](https://pypi.org/project/markdownify/) |
| `pyyaml` | >=6.0 | YAML parsing | [Docs](https://pyyaml.org/) |
| `pytest` | >=8.0 | Testing framework | [Docs](https://docs.pytest.org/) |
| `ruff` | >=0.5 | Linting | [Docs](https://docs.astral.sh/ruff/) |

### Textual Resources (for TUI development)

- [Guide](https://textual.textualize.io/guide/)
- [Widget gallery](https://textual.textualize.io/widget_gallery/)
- [CSS reference](https://textual.textualize.io/styles/)
- [Workers (async operations)](https://textual.textualize.io/guide/workers/)
- [Events (message passing)](https://textual.textualize.io/guide/events/)
- [Testing guide](https://textual.textualize.io/guide/testing/)
- [Blog: Anatomy of a Textual User Interface](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/)

---

## Questions & Support

If anything is unclear, open an issue at [GitHub Issues](https://github.com/EvoScientist/EvoScientist/issues) and describe:
- **Current behavior** — what you observe
- **Expected behavior** — what you expected
- **Reproducible context** — commands, logs, environment details
