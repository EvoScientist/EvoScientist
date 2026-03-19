# Contributing to EvoScientist

We appreciate your interest and the time you spend helping improve EvoScientist. Please read the following guidelines before contributing.

## How you can contribute

- **Report bugs and request features:** open an issue using the provided templates. Make sure to use the correct template and labels.
- **Propose design changes:** use issues or discussion threads to outline the problem, alternatives, and trade-offs before implementing.
- **Contribute code or docs:** submit PRs that address an open issue. They must have a clear rationale and tests where applicable.

## What we are looking for in PRs

We aim to keep EvoScientist focused on core functionality that benefits the majority of users. PRs should only include:

- Bug fixes / improvements to existing features
- New features that were proposed in an issue and agreed upon with maintainers
- Documentation updates and examples
- Meaningful additions to the test suite

If you want to add a niche or specialized workflow, consider contributing to the [EvoSkills repository](https://github.com/EvoScientist/EvoSkills) instead.

## Development setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/EvoScientist.git
   cd EvoScientist
   ```

2. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync --dev
   ```

3. **Run the test suite** (no API keys needed):
   ```bash
   uv run pytest
   ```

4. **Run the linter:**
   ```bash
   uv run ruff check .
   ```

## Submitting a pull request

1. Create a branch from `main` with a descriptive name (e.g. `fix/session-crash`, `feat/export-csv`).
2. Make your changes, keeping commits focused and well-described.
3. Ensure `uv run ruff check .` and `uv run pytest` pass locally — these also run in CI.
4. Open a PR against `main` and fill in the PR template.
5. A maintainer will review your PR. Please be responsive to feedback.

## Code style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting. Run `uv run ruff check .` before pushing.
- Follow the existing code patterns and conventions in the area you're modifying.
- Keep changes minimal and focused on the task at hand.

---

## Project overview

EvoScientist is a multi-agent AI system for automated scientific experimentation and discovery. It orchestrates specialized sub-agents that plan experiments, search literature, write code, debug, analyze data, and draft reports.

| Fact | Value |
|------|-------|
| Language | Python 3.11+ |
| License | Apache 2.0 |
| Framework | [DeepAgents](https://github.com/langchain-ai/deepagents) + [LangChain](https://python.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Default model | `claude-sonnet-4-6` (Anthropic) |
| Tests | ~890 across 36 files, no API keys needed |
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

### Data flow

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

## Need help?

Reach us on [Discord](https://discord.gg/AZ9ZMXkunY) or WeChat (linked in README).
