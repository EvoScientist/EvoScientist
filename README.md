<!-- Add logo here -->
<h1 align="center">
  <!-- <img src="./assets/CCD_icon_logo.png" alt="CCD Logo" height="27" style="position: relative; top: -2px;"/> -->
  <strong>EvoScientist</strong>
</h1>


<div align="center">

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&width=435&lines=Towards+Self-Evolving+AI+Scientists+for+End-to-End+Scientific+Discovery" alt="Typing SVG" /></a>

[![PyPI](https://img.shields.io/badge/PyPI-EvoScientist%20v0.0.1-3da9fc?style=for-the-badge&logo=python&logoColor=3da9fc)](https://pypi.org/project/EvoScientist/)
[![Project Page](https://img.shields.io/badge/Project-Page-ff8e3c?style=for-the-badge&logo=googlelens&logoColor=ff8e3c)]()
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxx-b31b1b?style=for-the-badge&logo=arxiv&logoColor=b31b1b)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()
<!-- [![Gradio Demo](https://img.shields.io/badge/Gradio-Online_Demo-FFCC00?style=for-the-badge&logo=gradio&logoColor=yellow&labelColor=grey)]()
[![Evaluation Split](https://img.shields.io/badge/HF-Test_Dataset-AECBFA?style=for-the-badge&logo=huggingface&logoColor=FFCC00&labelColor=grey)]() -->

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
- [📊 Evaluation](#-evaluation)
- [📝 Citation](#-citation)
- [📚 Acknowledgments](#-acknowledgments)
- [📦 Codebase Contributors](#-codebase-contributors)
- [📜 License](#-license)

## 🤖 Supported Models

| Provider | Short Name | Model ID |
|----------|-----------|----------|
| Anthropic | `claude-sonnet-4-5` | `claude-sonnet-4-5-20250929` |
| Anthropic | `claude-opus-4-5` | `claude-opus-4-5-20251101` |
| Anthropic | `claude-3-5-sonnet` | `claude-3-5-sonnet-20241022` |
| Anthropic | `claude-3-5-haiku` | `claude-3-5-haiku-20241022` |
| OpenAI | `gpt-4o` | `gpt-4o` |
| OpenAI | `gpt-4o-mini` | `gpt-4o-mini` |
| OpenAI | `o1` | `o1` |
| OpenAI | `o1-mini` | `o1-mini` |
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

# Optional: OpenAI or NVIDIA provider
export OPENAI_API_KEY="your_openai_api_key_here"
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
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key for GPT models ([platform.openai.com](https://platform.openai.com/)) |
| `NVIDIA_API_KEY` | For NVIDIA | NVIDIA API key for NIM models ([build.nvidia.com](https://build.nvidia.com/)) |
| `TAVILY_API_KEY` | Yes | Tavily API key for web search ([app.tavily.com](https://app.tavily.com/)) |

## ⚡ Quick Start

### CLI Inference  
You can perform inference directly from the command line using our CLI tool:

![demo](./assets/EvoScientist_cli.png)

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
--workdir <path>   Override workspace directory for this session
--use-cwd          Use current working directory as workspace
--thread-id <id>   Resume a conversation thread
--no-thinking      Disable thinking display
-p, --prompt <q>   Single-shot mode: execute query and exit
```

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
| `/thread` | Show current thread ID and workspace path |
| `/skills` | List installed user skills |
| `/install-skill <source>` | Install a skill from local path or GitHub |
| `/uninstall-skill <name>` | Uninstall a user-installed skill |

**Skill Installation Examples:**

```bash
# Install from local path
/install-skill ./my-skill

# Install from GitHub URL
/install-skill https://github.com/owner/repo/tree/main/skill-name

# Install from GitHub shorthand
/install-skill owner/repo@skill-name
```

### Runtime Directories

By default, the **workspace** is created under the current directory:

```
./workspace/
  memory/   # shared MEMORY.md (persistent across sessions)
  skills/   # user-installed skills
  runs/     # per-session workspaces
```

You can force workspace to be the current directory via `--use-cwd`.

Override individual paths via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVOSCIENTIST_WORKSPACE_DIR` | `./workspace` | Root workspace directory |
| `EVOSCIENTIST_RUNS_DIR` | `./workspace/runs` | Per-session run directories |
| `EVOSCIENTIST_MEMORY_DIR` | `./workspace/memory` | Shared memory storage |
| `EVOSCIENTIST_SKILLS_DIR` | `./workspace/skills` | User-installed skills |

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


## 📦 Codebase Contributors

<table>
  <tbody>
    <tr>
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
        <a href="https://x-izhang.github.io/">
          <img src="https://x-izhang.github.io/author/xi-zhang/avatar_hu13660783057866068725.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Xi Zhang"/>
          <br />
          <sub><b>Xi Zhang</b></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>

For any enquiries or collaboration opportunities, please contact: [**youganglyu@gmail.com**](mailto:youganglyu@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
