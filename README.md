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
</picture></a><a href="https://evoscientist.ai/"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-website-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-website-dark.svg">
  <img alt="Website" src=".github/assets/badge-website-light.svg" height="28">
</picture></a><a href="https://github.com/langchain-ai/deepagents"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-framework-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-framework-dark.svg">
  <img alt="Framework DeepAgents" src=".github/assets/badge-framework-light.svg" height="28">
</picture></a><a href="https://github.com/EvoScientist/EvoScientist/blob/main/LICENSE"><picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/badge-license-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/badge-license-dark.svg">
  <img alt="License MIT" src=".github/assets/badge-license-light.svg" height="28">
</picture></a>
</div>

---

<div align="center">
<a href="https://github.com/EvoScientist/EvoScientist"><img src="https://readme-typing-svg.demolab.com?font=Sans-Serif&pause=1000&color=64B5F6&center=true&vCenter=true&width=435&lines=Towards+Self-Evolving+AI+Scientists;Harness+Vibe+Research" alt="Typing SVG" /></a>
</div>

<div align="center">

**English | [简体中文](./README.zh-CN.md)**

</div>

***EvoScientist aims to harness vibe research by enabling self-evolving AI scientists that autonomously explore, generate insights, and iteratively improve.
It is designed to be opinionated and ready to use out of the box, offering a living research system that grows alongside evolving agent skills, toolsets, and memory bases.
Going beyond traditional human-in-the-loop systems, EvoScientist introduces an AI-in-human’s-loop paradigm, where AI acts as a research buddy that co-evolves with human researchers and internalises scholarly taste and scientific judgement.***

<h3>Unified Control, Different Surfaces</h3>
[TODO: Add a Demo to demonstrate the different interfaces (TUI, mobile) and how they connect to the same underlying proxy system.]
<!-- <a href="https://github.com/EvoScientist/EvoScientist">
<img width="100%" src="https://github.com/EvoScientist/EvoScientist/tree/main/.github/assets/EvoScientist_demo.gif?raw=true"></a> -->

## ✨ Features
- **🤖 Multi-Agent Team** — 6 sub-agents (plan, research, code, debug, analyze, write) working in concert.
- **🧠 Persistent Memory** — Context, preferences, and findings survive across sessions.
- **🔬 Scientific Workflow** — Intake → plan → execute → evaluate → write → verify.
- **🌐 Multi-Provider** — Anthropic, OpenAI, Google, NVIDIA — one config to switch.
- **📱 Multi-Channel** — CLI as the hub; Telegram, Discord, Slack, Feishu, WeChat, and more — one agent session.
- **🔌 MCP & Skills** — Plug in MCP servers or install skills from GitHub on the fly.

## 🔥 News
- **[27 Feb 2026]** ⛳ EvoScientist officially debuts!

## 📖 Table of Contents

- [📦 Installation](#-installation)
- [🔑 Configuration](#-configuration)
- [⚡ Quick Start](#-quick-start)
- [🔌 MCP Integration](#-mcp-integration)
- [📱 Channels](#-channels)
- [📚 Acknowledgments](#-acknowledgments)
- [🧪 EvoScientist Team](#-evoscientist-team)
- [🤝 Contributing](#-contributing)

## 📦 Installation

> [!NOTE]
> Requires **Python 3.11+**. A virtual environment is strongly recommended — EvoScientist experiments may install ML libraries (PyTorch, transformers, etc.) that can conflict with your system packages.

### Quick Install

```bash
pip install EvoScientist
```

### Development Install

```bash
git clone https://github.com/EvoScientist/EvoScientist.git
cd EvoScientist
pip install -e ".[dev]"
```

<details>
<summary>Using conda</summary>

```bash
conda create -n EvoSci python=3.11 -y
conda activate EvoSci
pip install -e ".[dev]"
```

</details>

<details>
<summary>Upgrade to latest</summary>

```bash
git pull && pip install -e ".[dev]"
```

</details>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🔑 Configuration

The easiest way to configure API keys is the interactive wizard:

```bash
EvoSci onboard
```

It walks you through provider selection, key validation, model choice, and workspace setup.

<details>
<summary>Manual configuration via environment variables</summary>

Set at least one LLM provider key and (optionally) a search key:

```bash
# Pick one LLM provider
export ANTHROPIC_API_KEY="sk-..."   # Claude — console.anthropic.com
export OPENAI_API_KEY="sk-..."      # GPT   — platform.openai.com
export GOOGLE_API_KEY="AI..."       # Gemini — aistudio.google.com/api-keys
export NVIDIA_API_KEY="nvapi-..."   # NIM   — build.nvidia.com

# Web search (optional)
export TAVILY_API_KEY="tvly-..."    # app.tavily.com
```

Or use `EvoSci config set` to persist keys in `~/.config/evoscientist/config.yaml`.

Alternatively, copy the example `.env` file for project-level configuration:

```bash
cp .env.example .env  # then fill in your keys
```

> [!WARNING]
> Never commit `.env` files with real keys. It is already in `.gitignore`.

</details>

<p align="right"><a href="#top">🔝Back to top</a></p>

## ⚡ Quick Start

```bash
EvoSci  # or EvoScientist — interactive mode
```

![demo](.github/assets/EvoScientist_cli.png)

> Run `EvoSci -h` for all CLI options.

![cli help](.github/assets/EvoScientist_cli_help.png)

<details>
<summary>Common examples</summary>

```bash
EvoSci -p "your question"        # single-shot mode
EvoSci -m run                     # isolated per-session workspace
EvoSci --ui textual               # alternative TUI backend
EvoSci serve                      # headless mode — channels only, no interactive prompt
```

</details>

<details>
<summary>In-session commands</summary>

| Command | Description |
| ------- | ----------- |
| `/new` | Start a new session |
| `/current` | Show thread ID and workspace path |
| `/channel` | Start a messaging channel |
| `/skills` | List installed skills |
| `/install-skill <src>` | Install skill from path or GitHub |
| `/mcp` | List MCP servers and tool routing |
| `/exit` | Quit |

</details>

<details>
<summary>Script Inference</summary>

```python
from EvoScientist import EvoScientist_agent
from langchain_core.messages import HumanMessage
from EvoScientist.utils import format_messages

thread = {"configurable": {"thread_id": "1"}}
last_len = 0

for state in EvoScientist_agent.stream(
    {"messages": [HumanMessage(content="Hi?")]},
    config=thread,
    stream_mode="values",
):
    msgs = state["messages"]
    if len(msgs) > last_len:
        format_messages(msgs[last_len:])
        last_len = len(msgs)
```

</details>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🔌 MCP Integration

Add external tools via [MCP](https://modelcontextprotocol.io/) servers with a single command:

```bash
# Usage
EvoSci mcp add <name> <command> [-- args...]

# Example
EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
```

> [!NOTE]
> For command options, config fields, tool routing, wildcard filtering, and troubleshooting, see the **[MCP Integration Guide](https://github.com/EvoScientist/EvoScientist/tree/main/EvoScientist/mcp#model-context-protocol-integration)**.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📱 Channels

Connect messaging platforms so they share the same agent session as the CLI:

```bash
# Usage
EvoSci channel setup <channel>

# Example
EvoSci channel setup telegram
```

Multiple channels can run concurrently — comma-separate names in the config:

```yaml
channel_enabled: "telegram,discord,slack"
```

The channel can also be started interactively with `/channel` in the CLI session.

> [!NOTE]
> For per-channel setup guides, capability matrix, architecture details, and troubleshooting, see the **[Channel Integration Guide](https://github.com/EvoScientist/EvoScientist/tree/main/EvoScientist/channels#channels)**.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📚 Acknowledgments

This project builds upon the following outstanding open-source works:

- [**LangChain**](https://github.com/langchain-ai/langchain) — A framework for building agents and LLM-powered applications.
- [**DeepAgents**](https://github.com/langchain-ai/deepagents) — The batteries-included agent harness.

We thank the authors for their valuable contributions to the open-source community.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🧪 EvoScientist Team

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
        <a href="https://muxincg2004.github.io/">
          <img src="https://muxincg2004.github.io/resume_avatar.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Ziheng Zhang"/>
          <br />
          <sub><b>Ziheng Zhang<sup>‡</sup></b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://din0s.me/">
          <img src="https://din0s.me/images/pk.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Dinos Papakostas"/>
          <br />
          <sub><b>Dinos Papakostas<sup>‡</sup></b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://youganglyu.github.io/">
          <img src="https://youganglyu.github.io/images/profile.png"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Yougang Lyu"/>
          <br />
          <sub><b>Yougang Lyu<sup>§</sup></b></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>

> <sup>†</sup> Project Leader <sup>‡</sup> Core Developer <sup>§</sup> Project Correspondent

For any enquiries or collaboration opportunities, please contact: [**EvoScientist.ai@gmail.com**](mailto:evoscientist.ai@gmail.com)

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🤝 Contributing

<img align="right" alt="EvoScientist Team" src=".github/assets/EvoScientist_team.png" width="20%" />

We welcome contributions from developers and researchers at all levels. Please refer to our [Contributing Guidelines](./CONTRIBUTING.md) to get started and help make EvoScientist more accessible.

### ❤️ Thanks go to these awesome contributors:

[![EvoScientist contributors](https://contrib.rocks/image?repo=EvoScientist/EvoScientist)](https://github.com/EvoScientist/EvoScientist/graphs/contributors)

### 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EvoScientist/EvoScientist&type=date&legend=top-left)](https://www.star-history.com/#EvoScientist/EvoScientist&type=date&legend=top-left)

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

<p align="right"><a href="#top">🔝Back to top</a></p>

---

<p align="center">
  Made with ❤️ by the EvoScientist team and the open source community for the AI scientist community.
</p>