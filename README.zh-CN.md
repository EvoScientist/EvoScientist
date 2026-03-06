 > [!WARNING]
 > 这是社区翻译版本，欢迎修正！

---

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

**[English](./README.md) | 简体中文**



</div>

**EvoScientist 旨在通过构建自我进化的 AI 科学家来驱动 Vibe Research——让 AI 自主探索、生成洞见并持续迭代优化。
它以开箱即用为设计理念，提供一个伴随智能体技能、工具集和记忆库共同成长的活跃研究系统。
EvoScientist 超越了传统的人在回路（Human-in-the-Loop）模式，引入了 AI 在人的回路中（AI-in-Human's-Loop）的新范式——AI 作为研究伙伴，与人类研究者共同进化，逐步内化学术品味与科学判断力。**

<!-- <h3>统一控制，多种界面</h3>
[TODO: 添加 Demo 以展示不同界面（TUI、移动端）及其如何连接到同一底层代理系统。] -->
<!-- <a href="https://github.com/EvoScientist/EvoScientist">
<img width="100%" src="https://github.com/EvoScientist/EvoScientist/tree/main/.github/assets/EvoScientist_demo.gif?raw=true"></a> -->

## ✨ 特性

- **🤖 多智能体协作** — 6 个子智能体（规划、调研、编码、调试、分析、写作）协同工作。
- **🧠 持久化记忆** — 上下文、偏好和研究发现跨会话保持。
- **🔬 科学工作流** — 需求采集 → 规划 → 执行 → 评估 → 撰写 → 验证。
- **🌐 多模型供应商** — Anthropic、OpenAI、Google、NVIDIA——一处配置，随时切换。
- **📱 多渠道接入** — CLI 为中心；Telegram、Discord、Slack、飞书、微信等——共享同一智能体会话。
- **🔌 MCP 与 Skills** — 即插即用 MCP 服务器，或从 GitHub 一键安装技能包。

## 🎯 ᯓ➤ 路线图

- [x] 🖥️ 基于 Rich 和 Textual 的 TUI 界面
- [x] 📻 EvoMemory v1.0 已上线
- [x] ⚒️ 200+ 预置技能已内置
- [ ] 📑 技术报告即将发布
- [ ] 📺 Demo 与教程正在制作中
- [ ] 📊 基准测试套件即将推出
- [ ] ⏰ 核心系统定时任务规划中
- [ ] 🧩 更多内置技能与集成即将上线

## 🔥 动态

- **[2026 年 2 月 27 日]** ⛳ EvoScientist 正式亮相！

## 📖 目录

- [📦 安装](#-安装)
- [🔑 配置](#-配置)
- [⚡ 快速上手](#-快速上手)
- [🔌 MCP 集成](#-mcp-集成)
- [📱 渠道接入](#-渠道接入)
- [📚 致谢](#-致谢)
- [🌍 项目角色](#-项目角色)
- [🤝 贡献](#-贡献)

## 📦 安装

> [!TIP]
> 需要 **Python 3.11+**（**< 3.14**）。推荐使用 [**uv**](https://docs.astral.sh/uv/) 或 **conda** 进行依赖管理和虚拟环境管理。

<details>
<summary>🪛 安装 uv（如果尚未安装）</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

### 快速安装

```bash
uv tool install EvoScientist
```

或安装到当前环境：

```bash
uv pip install EvoScientist
```

### 开发安装

```bash
git clone https://github.com/EvoScientist/EvoScientist.git
cd EvoScientist
uv sync --dev
```

<details>
<summary> 使用 conda</summary>

```bash
conda create -n EvoSci python=3.11 -y
conda activate EvoSci
pip install -e ".[dev]"
```

</details>

<details>
<summary> 使用 PyPi</summary>

```bash
pip install EvoScientist          # quick install
pip install -e ".[dev]"           # development install
```

</details>

<details>
<summary> 可选：渠道依赖</summary>

消息渠道集成需要额外依赖，按需安装即可：

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
<summary> 升级到最新代码库</summary>

```bash
git pull && uv sync --dev
```

</details>

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 🔑 配置

最简单的方式是使用交互式配置向导：

```bash
EvoSci onboard
```

向导将引导你完成供应商选择、密钥验证、模型选择和工作区设置。

<details>
<summary> 📟 通过环境变量手动配置</summary>

至少设置一个 LLM 供应商密钥，搜索密钥为可选项：

```bash
# 选择一个 LLM 供应商
export ANTHROPIC_API_KEY="sk-..."   # Claude — console.anthropic.com
export OPENAI_API_KEY="sk-..."      # GPT   — platform.openai.com
export GOOGLE_API_KEY="AI..."       # Gemini — aistudio.google.com/api-keys
export NVIDIA_API_KEY="nvapi-..."   # NIM   — build.nvidia.com

# 网络搜索（可选）
export TAVILY_API_KEY="tvly-..."    # app.tavily.com
```

也可以使用 `EvoSci config set` 将密钥持久化到 `~/.config/evoscientist/config.yaml`。

或者复制示例 `.env` 文件用于项目级配置：

```bash
cp .env.example .env  # 填入你的密钥
```

> ⚠️ 切勿将包含真实密钥的 `.env` 文件提交到版本库。该文件已在 `.gitignore` 中。

</details>

<p align="right"><a href="#top">🔝回到顶部</a></p>

## ⚡ 快速上手

```bash
EvoSci  # 或 EvoScientist — 交互模式
```

![demo](.github/assets/EvoScientist_cli.png)

> 运行 `EvoSci -h` 查看全部 CLI 选项。

![cli help](.github/assets/EvoScientist_cli_help.png)

<details>
<summary>常用示例</summary>

```bash
EvoSci -p "你的问题"              # 单次查询模式
EvoSci -m run                     # 隔离的会话级工作区
EvoSci --ui textual               # 替代 TUI 后端
EvoSci serve                      # 无头模式——仅渠道，无交互提示符
```

</details>

<details>
<summary>会话内命令</summary>

| 命令 | 说明 |
| ---- | ---- |
| `/new` | 开始新会话 |
| `/current` | 显示当前线程 ID 和工作区路径 |
| `/channel` | 启动消息渠道 |
| `/skills` | 列出已安装的技能包 |
| `/install-skill <src>` | 从本地路径或 GitHub 安装技能包 |
| `/mcp` | 列出 MCP 服务器和工具路由 |
| `/exit` | 退出 |

</details>

<details>
<summary>脚本调用</summary>

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

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 🔌 MCP 集成

通过 [MCP](https://modelcontextprotocol.io/) 服务器一条命令即可添加外部工具：

```bash
# 用法
EvoSci mcp add <name> <command> [-- args...]

# 示例
EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
```

> [!TIP]
> 关于命令选项、配置字段、工具路由、通配符过滤和故障排查，请参阅 **[MCP 集成指南](https://github.com/EvoScientist/EvoScientist/tree/main/EvoScientist/mcp#model-context-protocol-integration)**。

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 📱 渠道接入

连接消息平台，使其与 CLI 共享同一智能体会话：

```bash
# 用法
EvoSci channel setup <channel>

# 示例
EvoSci channel setup telegram
```

多个渠道可同时运行——在配置中用逗号分隔：

```yaml
channel_enabled: "telegram,discord,slack"
```

也可以在 CLI 会话中通过 `/channel` 交互式启动渠道。

> [!TIP]
> 关于各渠道设置指南、功能矩阵、架构详情和故障排查，请参阅 **[渠道集成指南](https://github.com/EvoScientist/EvoScientist/tree/main/EvoScientist/channels#channels)**。

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 📚 致谢

本项目基于以下优秀的开源项目构建：

- [**LangChain**](https://github.com/langchain-ai/langchain) — 构建智能体和 LLM 驱动应用的框架。
- [**DeepAgents**](https://github.com/langchain-ai/deepagents) — 开箱即用的智能体编排框架。

感谢以上项目作者对开源社区的宝贵贡献。

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 🌍 项目角色

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
      <td align="center">
        <a href="https://muxincg2004.github.io/">
          <img src="https://muxincg2004.github.io/resume_avatar.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Ziheng Zhang"/>
          <br />
          <sub><b>Ziheng Zhang</b><sup>‡</sup></sub>
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
        <a href="https://youganglyu.github.io/">
          <img src="https://youganglyu.github.io/images/profile.png"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Yougang Lyu"/>
          <br />
          <sub><b>Yougang Lyu</b><sup>§</sup></sub>
        </a>
      </td>
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

Jan Piotrowski, Wiktor Cupiał, Yuyue Zhao, Xinhao Yi, Jakub Kaliski, Jakub Filipiuk, Shuyu Guo, Andreas Sauter, Jacopo Urbani, Zaiqiao Meng, Lun Zhou

> <sup>†</sup>Project Lead & Engineering Lead <sup>‡</sup>Core Developer <sup>§</sup>Project Owner

> *Xiaoyi DeepResearch Team* 及更广泛的开源社区共同为本项目做出贡献。

如有任何咨询或合作意向，请联系：[**EvoScientist.ai@gmail.com**](mailto:evoscientist.ai@gmail.com)

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 🤝 贡献

<img align="right" alt="EvoScientist Team" src=".github/assets/EvoScientist_team.png" width="20%" />

我们欢迎各层次的开发者、研究者以及 AI 编程助手参与贡献。我们的 [贡献指南](./CONTRIBUTING.md) 同时面向人类和 AI Agent 编写，涵盖架构说明、设计模式、扩展指南和代码规范，帮助你安全高效地参与项目开发。

### 📈 Star 趋势

[![Star History Chart](https://api.star-history.com/svg?repos=EvoScientist/EvoScientist&type=date&legend=top-left)](https://www.star-history.com/#EvoScientist/EvoScientist&type=date&legend=top-left)

<p align="right"><a href="#top">🔝回到顶部</a></p>

## 📜 许可证

本项目基于 MIT 许可证开源——详情请见 [LICENSE](./LICENSE) 文件。

<p align="right"><a href="#top">🔝回到顶部</a></p>

---

<p align="center">
  由 <a href="https://github.com/x-izhang">Xi Zhang</a> 发起并主导，与开源社区共建。
</p>
