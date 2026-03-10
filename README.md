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
</picture></a><a href="https://EvoScientist.github.io/"><picture>
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

**EvoScientist aims to harness vibe research by enabling self-evolving AI scientists that autonomously explore, generate insights, and iteratively improve.
It is designed to be opinionated and ready to use out of the box, offering a living research system that grows alongside evolving agent skills, toolsets, and memory bases.
Going beyond traditional human-in-the-loop systems, EvoScientist introduces an AI-in-human’s-loop paradigm, where AI acts as a research buddy that co-evolves with human researchers and internalizes scholarly taste and scientific judgment.**

<!-- <h3>Unified Control, Different Surfaces</h3>
[TODO: Add a Demo to demonstrate the different interfaces (TUI, mobile) and how they connect to the same underlying proxy system.] -->
<!-- <a href="https://github.com/EvoScientist/EvoScientist">
<img width="100%" src="https://github.com/EvoScientist/EvoScientist/tree/main/.github/assets/EvoScientist_demo.gif?raw=true"></a> -->

## ✨ Features
- **🤖 Multi-Agent Team** — 6 sub-agents (plan, research, code, debug, analyze, write) working in concert.
- **🧠 Persistent Memory** — Context, preferences, and findings survive across sessions.
- **🔬 Scientific Workflow** — Intake → plan → execute → evaluate → write → verify.
- **🌐 Multi-Provider** — Anthropic, OpenAI, Google, NVIDIA — one config to switch.
- **📱 Multi-Channel** — CLI as the hub; Telegram, Discord, Slack, Feishu, WeChat, and more — one agent session.
- **🔌 MCP & Skills** — Plug in MCP servers or install skills from GitHub on the fly.

## 🎯 ᯓ➤ Roadmap
- [x] 🖥️ Full-screen TUI and classic CLI interfaces
- [x] 📻 EvoMemory v1.0 shipped
- [x] ⚒️ 200+ predefined skills built in
- [x] 🧩 Built-in research-lifecycle skills shipped
- [x] 👋 Human-in-the-loop action approval
- [ ] 📺 Web app with workspace UI
- [ ] 📑 Technical report on the way
- [ ] 📹 Demo and tutorial in the works
- [ ] 📊 Benchmark suite to be released
- [ ] ⏰ Scheduled tasks for the core system planned

## 🔥 News
- **[?? Mar 2026]** ⛳ Technical Report is live! [**Check it out**](https://arxiv.org/abs/1234.56780) 👈
- **[?? Mar 2026]** 🚀 [**EvoScientist**](https://github.com/EvoScientist/EvoScientist) officially debuts!

## 📖 Table of Contents

- [📦 Installation](#-installation)
- [🔑 Configuration](#-configuration)
- [⚡ Quick Start](#-quick-start)
- [🔌 MCP Integration](#-mcp-integration)
- [📱 Channels](#-channels)
- [📚 Acknowledgments](#-acknowledgments)
- [🌍 Project Roles](#-project-roles)
- [🤝 Contributing](#-contributing)

## 📦 Installation

> [!TIP]
> Requires **Python 3.11+** (**< 3.14**). We recommend [**uv**](https://docs.astral.sh/uv/) or **conda** for dependency management and virtual environments.

<details>
<summary> 🪛 Install uv (if you don't have it)</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

### Quick Install

```bash
uv tool install EvoScientist
```

Or install into the current environment instead:

```bash
uv pip install EvoScientist
```

### Development Install

```bash
git clone https://github.com/EvoScientist/EvoScientist.git
cd EvoScientist
uv sync --dev
```

<details>
<summary> Using conda</summary>

```bash
conda create -n EvoSci python=3.11 -y
conda activate EvoSci
pip install -e ".[dev]"
```

</details>

<details>
<summary> Using PyPi</summary>

```bash
pip install EvoScientist          # quick install
pip install -e ".[dev]"           # development install
```

</details>

<details>
<summary> Optional: Channel dependencies</summary>

Messaging channel integrations require extra dependencies. Install only what you need:

```bash
uv pip install "EvoScientist[telegram]"     # Telegram
uv pip install "EvoScientist[discord]"      # Discord
uv pip install "EvoScientist[slack]"        # Slack
uv pip install "EvoScientist[wechat]"       # WeChat
uv pip install "EvoScientist[qq]"           # QQ
uv pip install "EvoScientist[all-channels]" # everything
```

</details>

<details>
<summary> Upgrade to the latest code base </summary>

```bash
git pull && uv sync --dev
```

</details>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🔑 Configuration

The easiest way to configure API keys is the interactive wizard:

```bash
EvoSci onboard
```
> [!TIP]
> It walks you through provider selection, key validation, model choice, and workspace mode.

<details>
<summary> 📟 Manual configuration via environment variables </summary>

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

> ⚠️ Never commit `.env` files with real keys. It is already in `.gitignore`.

</details>

<p align="right"><a href="#top">🔝Back to top</a></p>

## ⚡ Quick Start

```bash
EvoSci  # or EvoScientist — interactive mode (TUI by default)
```

![demo](.github/assets/EvoScientist_cli.png)

> Run `EvoSci -h` for all CLI options.

![cli help](.github/assets/EvoScientist_cli_help.png)

<details>
<summary>Common examples</summary>

```bash
EvoSci                            # interactive mode (TUI by default)
EvoSci -p "your question"        # single-shot mode
EvoSci --workdir /path/to/project # open in a specific directory
EvoSci -m run                     # isolated per-session workspace
EvoSci --ui cli                   # classic CLI (lightweight)
EvoSci serve                      # headless mode — channels only, no interactive prompt
```

</details>

<details>
<summary>Action Approval (HITL)</summary>

By default, shell commands (`execute` tool) require human approval before running. To skip approval prompts:

```bash
# Per-session: auto-approve via CLI flag
EvoSci --auto-approve
EvoSci -p "query" --auto-approve

# Persistent: set in config (applies to all future sessions)
EvoSci config set auto_approve true

# Or allow only specific command prefixes
EvoSci config set shell_allow_list "python,pip,pytest,ruff,git"
```

During a session you can also reply **3** (Approve all) at any approval prompt to auto-approve for the rest of that session.

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

> [!TIP]
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

> [!TIP]
> For per-channel setup guides, capability matrix, architecture details, and troubleshooting, see the **[Channel Integration Guide](https://github.com/EvoScientist/EvoScientist/tree/main/EvoScientist/channels#channels)**.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📚 Acknowledgments

This project builds upon the following outstanding open-source works:

- [**LangChain**](https://github.com/langchain-ai/langchain) — A framework for building agents and LLM-powered applications.
- [**DeepAgents**](https://github.com/langchain-ai/deepagents) — The batteries-included agent harness.

We thank the authors for their valuable contributions to the open-source community.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🌍 Project Roles

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://x-izhang.github.io/">
          <img src="https://x-izhang.github.io/author/xi-zhang/avatar_hu13660783057866068725.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Xi Zhang"/>
          <br />
          <sub><b>Xi Zhang</b><sup>†§</sup></sub>
        </a>
      </td>
      </td>
      <td align="center">
        <a href="https://youganglyu.github.io/">
          <img src="https://youganglyu.github.io/images/profile.png"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Yougang Lyu"/>
          <br />
          <sub><b>Yougang Lyu</b><sup>‡§</sup></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://din0s.me/">
          <img src="https://din0s.me/images/pk.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Dinos Papakostas"/>
          <br />
          <sub><b>Dinos Papakostas</b><sup>‡</sup></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://go0day.github.io/">
          <img src="https://go0day.github.io/authors/admin/avatar_hu_ee1051aceae96124.png"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Yuyue Zhao"/>
          <br />
          <sub><b>Yuyue Zhao</b><sup>‡</sup></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://muxincg2004.github.io/">
          <img src="https://muxincg2004.github.io/resume_avatar.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Ziheng Zhang"/>
          <br />
          <sub><b>Ziheng Zhang</b><sup>‡</sup></sub>
        </a>
      <td align="center">
        <a href="https://xiaohuiyan.github.io/">
          <img src="https://xiaohuiyan.github.io/img/me.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Xiaohui Yan"/>
          <br />
          <sub><b>Xiaohui Yan</b><sup>§</sup></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>

#### Contributors

Jan Piotrowski, Wiktor Cupiał, Xinhao Yi, Jakub Kaliski, Jakub Filipiuk, Shuyu Guo, Andreas Sauter, Wenxiang Hu, Jacopo Urbani, Zaiqiao Meng, Jun Luo, Lun Zhou

> <sup>†</sup>Project Lead & Engineering Lead <sup>‡</sup>Core Developer <sup>§</sup>Project Owner

> *Xiaoyi DeepResearch Team* and the wider open-source community contribute to this project.

For any inquiries or collaboration opportunities, please contact: [**EvoScientist.ai@gmail.com**](mailto:evoscientist.ai@gmail.com)

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🤝 Contributing

<img align="right" alt="EvoScientist Team" src=".github/assets/EvoScientist_team.png" width="20%" />

We welcome contributions from developers, researchers, and AI coding agents at all levels. Our [Contributing Guidelines](./CONTRIBUTING.md) are designed for both humans and AI agents — covering architecture, patterns, extension guides, and code standards to help you contribute safely and effectively.

### 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EvoScientist/EvoScientist&type=date&legend=top-left)](https://www.star-history.com/#EvoScientist/EvoScientist&type=date&legend=top-left)

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

<p align="right"><a href="#top">🔝Back to top</a></p>

---

<p align="center">
  Initiated and led by <a href="https://github.com/x-izhang">Xi Zhang</a>, built with the open-source community.
</p>
