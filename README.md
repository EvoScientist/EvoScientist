<!-- Add logo here -->
<div align="center">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset=".github/assets/logo-dark.svg">
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/logo-light.svg">
      <img alt="EvoScientist Logo" src=".github/assets/logo-dark.svg" width="80%">
    </picture>
</div>

<div align="center">
<a href="https://pypi.org/project/EvoScientist/"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-pypi-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-pypi-dark.svg">
  <img alt="PyPI v0.0.1" src=".github/assets/badge-pypi-light.svg" height="28">
</picture></a>&nbsp;
<a href="https://evoscientist.ai/"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-website-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-website-dark.svg">
  <img alt="Website" src=".github/assets/badge-website-light.svg" height="28">
</picture></a>&nbsp;
<a href="https://github.com/langchain-ai/deepagents"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-framework-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-framework-dark.svg">
  <img alt="Framework DeepAgents" src=".github/assets/badge-framework-light.svg" height="28">
</picture></a>&nbsp;
<a href="https://github.com/EvoScientist/EvoScientist/blob/main/LICENSE"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-license-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-license-dark.svg">
  <img alt="License MIT" src=".github/assets/badge-license-light.svg" height="28">
</picture></a>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Sans-Serif&pause=1000&color=64B5F6&center=true&vCenter=true&width=435&lines=Towards+Self-Evolving+AI+Scientists;Harness+Vibe-Driven+Research" alt="Typing SVG" /></a>

</div>

## 🔥 News
> TODO
- **[27 Sep 2025]** ⛳ Our preprint is now live on [arXiv] — check it out for details.

## Overview
> TODO

## 📖 Contents
- [🤖 Supported Models](#-supported-models)
- [⛏️ Installation](#️-installation)
- [🔑 API Key Configuration](#-api-key-configuration)
- [⚡ Quick Start](#-quick-start)
  - [CLI Inference](#cli-inference)
  - [Script Inference](#script-inference)
  - [Web Interface](#web-interface)
- [💬 Channels](#-channels)
- [🔌 MCP Integration](#-mcp-integration)
- [📊 Evaluation](#-evaluation)
- [📝 Citation](#-citation)
- [📚 Acknowledgments](#-acknowledgments)
- [📦 EvoScientist Team](#-evoscientist-team)
- [📜 License](#-license)

## 🤖 Supported Models

| Provider | Short Name | Model ID |
|----------|-----------|----------|
| Anthropic | `claude-opus-4-6` | `claude-opus-4-6` |
| Anthropic | `claude-opus-4-5` | `claude-opus-4-5-20251101` |
| Anthropic | `claude-sonnet-4-5` | `claude-sonnet-4-5-20250929` |
| Anthropic | `claude-haiku-4-5` | `claude-haiku-4-5-20251001` |
| OpenAI | `gpt-4o` | `gpt-4o` |
| OpenAI | `gpt-4o-mini` | `gpt-4o-mini` |
| OpenAI | `o1` | `o1` |
| OpenAI | `o1-mini` | `o1-mini` |
| Google | `gemini-3-pro` | `gemini-3-pro-preview` |
| Google | `gemini-3-flash` | `gemini-3-flash-preview` |
| Google | `gemini-2.5-pro` | `gemini-2.5-pro` |
| Google | `gemini-2.5-flash` | `gemini-2.5-flash` |
| Google | `gemini-2.5-flash-lite` | `gemini-2.5-flash-lite` |
| NVIDIA | `glm4.7` | `z-ai/glm4.7` |
| NVIDIA | `deepseek-v3.1` | `deepseek-ai/deepseek-v3.1-terminus` |
| NVIDIA | `nemotron-nano` | `nvidia/nemotron-3-nano-30b-a3b` |

You can also use any full model ID directly — the provider will be inferred automatically.

## ⛏️ Installation

> [!TIP]  
> Use [`uv`](https://pypi.org/project/uv) for installation — it's faster and more reliable than `pip`.
### For Development

```Shell
# Create and activate a conda environment
conda create -n EvoSci python=3.11 -y
conda activate EvoSci

# Install in development (editable) mode
pip install EvoScientist
# or
pip install -e .
```

### Option 1:
Install the latest version directly from GitHub for quick setup:
> TODO
### Option 2: 
If you plan to modify the code or contribute to the project, you can clone the repository and install it in editable mode:

> TODO

<details>
<summary> 🔄 Upgrade to the latest code base </summary>

```Shell
git pull
uv pip install -e .
```

</details>

## 🔑 API Key Configuration

EvoScientist requires API keys for LLM inference and web search. You can configure them in three ways:

### Option A: Interactive Setup Wizard (Recommended)

```Shell
EvoSci onboard
```

The wizard guides you through selecting a provider, entering API keys, choosing a model, and configuring workspace settings. Keys are validated automatically.

### Option B: Environment Variables (Global)

Set keys directly in your terminal session. Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist across sessions:

```Shell
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export TAVILY_API_KEY="your_tavily_api_key_here"

# Optional: OpenAI, Google, or NVIDIA provider
export OPENAI_API_KEY="your_openai_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### Option C: `.env` File (Project-level)

Create a `.env` file in the project root. This keeps keys scoped to the project and out of your shell history:

```Shell
cp .env.example .env
```

Then edit `.env` and fill in your keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> [!WARNING]
> Never commit `.env` files containing real API keys to version control. The `.env` file is already included in `.gitignore`.

| Key | Required | Description |
|-----|----------|-------------|
| `ANTHROPIC_API_KEY` | For Anthropic | Anthropic API key for Claude ([console.anthropic.com](https://console.anthropic.com/)) |
| `GOOGLE_API_KEY` | For Google | Google API key for Gemini models ([aistudio.google.com](https://aistudio.google.com/api-keys)) |
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key for GPT models ([platform.openai.com](https://platform.openai.com/)) |
| `NVIDIA_API_KEY` | For NVIDIA | NVIDIA API key for NIM models ([build.nvidia.com](https://build.nvidia.com/)) |
| `TAVILY_API_KEY` | Yes | Tavily API key for web search ([app.tavily.com](https://app.tavily.com/)) |

## ⚡ Quick Start

### CLI Inference  
You can perform inference directly from the command line using our CLI tool:

![demo](.github/assets/EvoScientist_cli.png)

```Shell
python -m EvoScientist 
```
or
```Shell
EvoSci # or EvoScientist
```
**Optional arguments:**

```
--mode <mode>      Workspace mode: 'daemon' (persistent) or 'run' (isolated per-session)
-n, --name <name>  Name for the run directory (requires --mode run; duplicates get _1, _2, …)
--workdir <path>   Override workspace directory for this session
--use-cwd          Use current working directory as workspace
--thread-id <id>   Resume a conversation thread
--no-thinking      Disable thinking display
--ui <backend>     UI backend: rich (default) or textual (beta)
-p, --prompt <q>   Single-shot mode: execute query and exit
```

> [!NOTE]
> In `--ui textual` mode, the built-in TUI commands are:
> `/help`, `/current`, `/new`, `/clear`, `/threads`, `/resume <id>`, `/delete <id>`, `/exit`.

![demo](.github/assets/EvoScientist_cli_help.png)

**Configuration commands:**

```Shell
EvoSci onboard                # Interactive setup wizard
EvoSci onboard --skip-validation  # Skip API key validation
EvoSci config                 # List all configuration values
EvoSci config get <key>       # Get a single value
EvoSci config set <key> <val> # Set a single value
EvoSci config reset --yes     # Reset to defaults
EvoSci config path            # Show config file path
```

**Interactive Commands:**

| Command | Description |
|---------|-------------|
| `/exit` | Quit the session |
| `/new` | Start a new session (new workspace + thread) |
| `/current` | Show current thread ID and workspace path |
| `/channel` | Start iMessage channel (shares agent session) |
| `/skills` | List installed user skills |
| `/install-skill <source>` | Install a skill from local path or GitHub |
| `/uninstall-skill <name>` | Uninstall a user-installed skill |
| `/mcp` | List configured MCP servers and tool routing |

**Skill Installation Examples:**

```bash
# Install from local path
/install-skill ./my-skill

# Install from GitHub URL
/install-skill https://github.com/owner/repo/tree/main/skill-name

# Install from GitHub shorthand
/install-skill owner/repo@skill-name
```

### Channels

EvoScientist integrates with 10 messaging platforms, allowing you to control the agent remotely from any chat app. All channels share the same agent core — messages from any platform go through the same processing pipeline.

| Channel | Transport | Public IP Required | Install Extra |
|:--------|:----------|:------------------:|:--------------|
| Telegram | Long Polling | No | `pip install evoscientist[telegram]` |
| Discord | WebSocket | No | `pip install evoscientist[discord]` |
| Slack | Socket Mode | No | `pip install evoscientist[slack]` |
| Feishu / Lark | HTTP Webhook | Yes | `pip install evoscientist[feishu]` |
| WeChat (WeCom / MP) | HTTP Webhook | Yes | `pip install evoscientist[wechat]` |
| DingTalk | WebSocket Stream | No | `pip install evoscientist[dingtalk]` |
| QQ | WebSocket | No | `pip install evoscientist[qq]` |
| Signal | JSON-RPC | No | `pip install evoscientist[signal]` |
| Email | IMAP + SMTP | No | `pip install evoscientist[email]` |
| iMessage | JSON-RPC (stdio) | No | macOS only, requires [imsg](https://github.com/anthropics/imsg) CLI |

**Quick start:**

```bash
# 1. Install channel dependencies
pip install evoscientist[telegram]   # or discord, slack, feishu, etc.

# 2. Configure via wizard or CLI
EvoSci onboard                      # interactive setup
# or
EvoSci config set channel_enabled telegram
EvoSci config set telegram_bot_token "123456:ABC-xxx"

# 3. Start
EvoSci serve                        # agent + all enabled channels
```

Multiple channels can run concurrently — comma-separate names in the config:

```yaml
channel_enabled: "telegram,discord,slack"
```

The channel can also be started interactively with `/channel` in the CLI session.

> [!NOTE]
> For per-channel setup guides, capability matrix, architecture details, and troubleshooting, see the **[Channel Integration Guide](./EvoScientist/channels)**.

### Runtime Directories

By default, the **workspace root** is the current working directory. Sub-directories
are created automatically:

```
<cwd>/
  memory/   # shared MEMORY.md (persistent across sessions)
  skills/   # user-installed skills
  runs/     # per-session workspaces (run mode only)
```

Use `--workdir` to set a different workspace root, or configure it via
`EvoSci config set default_workdir /path/to/workspace`.

Override individual paths via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVOSCIENTIST_WORKSPACE_DIR` | current directory | Root workspace directory |
| `EVOSCIENTIST_RUNS_DIR` | `<workspace>/runs` | Per-session run directories |
| `EVOSCIENTIST_MEMORY_DIR` | `<workspace>/memory` | Shared memory storage |
| `EVOSCIENTIST_SKILLS_DIR` | `<workspace>/skills` | User-installed skills |

### Script Inference
```python
from EvoScientist import EvoScientist_agent
from langchain_core.messages import HumanMessage
from EvoScientist.utils import format_messages

thread = {"configurable": {"thread_id": "1"}}
question = "Hi?"
last_len = 0

for state in EvoScientist_agent.stream(
    {"messages": [HumanMessage(content=question)]},
    config=thread,
    stream_mode="values",
):
    msgs = state["messages"]
    if len(msgs) > last_len:
        format_messages(msgs[last_len:]) 
        last_len = len(msgs)
```

<details>
<summary> Output </summary>

```json

╭─────────────────────────────────────────────────── 🧑 Human ────────────────────────────────────────────────────╮
│ Hi?                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────── 📝 AI ─────────────────────────────────────────────────────╮
│ Hi! I'm here to help you with experimental research tasks. I can assist with:                                   │
│                                                                                                                 │
│ - **Planning experiments** - designing stages, success criteria, and workflows                                  │
│ - **Running experiments** - implementing baselines, training models, analyzing results                          │
│ - **Research** - finding papers, methods, datasets, and baselines                                               │
│ - **Analysis** - computing metrics, creating visualizations, interpreting results                               │
│ - **Writing** - drafting experimental reports and documentation                                                 │
│                                                                                                                 │
│ What would you like to work on today?                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

### Web Interface  

> TODO


## 🔌 MCP Integration

EvoScientist connects to external systems via [MCP](https://modelcontextprotocol.io/) servers. Supports `stdio`, `http`, `streamable_http`, `sse`, and `websocket` transports.

```bash
# Add a server from the terminal
EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

# Or from inside an agent session
/mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
```

> [!NOTE]
> For command options, config fields, tool routing, wildcard filtering, and troubleshooting, see the **[MCP Integration Guide](./EvoScientist/mcp)**.

## 📊 Evaluation

> TODO

## 📝 Citation

If you find our paper and code useful in your research and applications, please cite using this BibTeX:

> TODO

## 📚 Acknowledgments

This project builds upon the following outstanding open-source works:

- [**Deep Agents**](https://github.com/langchain-ai/deepagents) — A framework for building AI agents that can interact with various tools and environments.
- [**Deep Agents UI**](https://github.com/langchain-ai/deep-agents-ui) — A user interface for visualising and managing Deep Agents.

We thank the authors for their valuable contributions to the open-source community.

## 📦 EvoScientist Team

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://x-izhang.github.io/">
          <img src="https://x-izhang.github.io/author/xi-zhang/avatar_hu13660783057866068725.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Xi Zhang"/>
          <br />
          <sub><b>Xi Zhang</b><sup>†</sup></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://youganglyu.github.io/">
          <img src="https://youganglyu.github.io/images/profile.png"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Yougang Lyu"/>
          <br />
          <sub><b>Yougang Lyu</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://din0s.me/">
          <img src="https://din0s.me/images/pk.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Dinos Papakostas"/>
          <br />
          <sub><b>Dinos Papakostas</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://muxincg2004.github.io/">
          <img src="https://muxincg2004.github.io/resume_avatar.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Ziheng Zhang"/>
          <br />
          <sub><b>Ziheng Zhang</b></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>

<sup>†</sup> Project Leader

For any enquiries or collaboration opportunities, please contact: [**EvoScientist.ai@gmail.com**](mailto:evoscientist.ai@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
