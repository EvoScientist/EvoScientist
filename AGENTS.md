# EvoScientist

EvoScientist is a local-first, multi-agent research assistant built on Python 3.11+, [DeepAgents](https://github.com/langchain-ai/deepagents), LangChain, and LangGraph. A persistent background LangGraph server (port `6174`) serves the CLI, Textual TUI, Desktop/WebUI (port `4716`), messaging channels, scheduled tasks, and programmatic `stream-json`/SDK clients. The WebUI front-end and Desktop app live in a separate repo, [EvoScientist-WebUI](https://github.com/EvoScientist/EvoScientist-WebUI); this repo holds only their backend.

Two things to keep in mind on every change:

- **It must work out of the box on macOS, Linux, and native Windows.** Users install with one command or the desktop app, without wrangling Python, Node.js, or Git Bash. Don't regress Windows (file handles, `cmd.exe`/Git Bash quoting, event loops, UTF-8 config reads).
- **The single-user security model is deliberate.** The command guard, workspace sandbox, and read-only WebUI settings exist for real reasons. Read [SECURITY.md](SECURITY.md) before changing approval, server bind, CORS/origin, or settings-API behavior.

## Principles

Fight complexity and scope creep. Prefer small, obvious changes over clever machinery. If an abstraction only has one caller, inline it until there's a second. Follow existing patterns in the module you're touching, and don't silently swallow errors.

Don't add new CLI flags, config settings, LLM providers, or messaging channels, and don't change default behavior, unless the human asked for it. If the task seems to need one, propose it and wait.

Treat this file as strong defaults—explicit instructions from the human driving your session override anything here. If a rule here fights the task in front of you, say so before breaking it.

## Terminology

- **you** = the coding agent reading this file.
- **maintainers** = the EvoScientist core team.
- **operator / user** = the person running EvoScientist on their machine.
- **agent** = the EvoScientist runtime agent (main orchestrator + sub-agents).
- **background server** = the shared `langgraph dev` backend on port `6174` managed via `EvoSci server`.

## Ways to hurt yourself (hard rules)

1. **Never touch the live install's data or config.** `~/.config/evoscientist/` and `~/.evoscientist/` hold the operator's real API keys, ChatGPT OAuth tokens, sessions DB, memories, skills, and scheduled tasks. Reading them is fine if needed; never write to them, never point tests at them, and never clean them up.
2. **Never kill processes by broad pattern.** Do not run `pkill -f`, `pgrep | kill`, or blanket kills on `python`/`node`/`EvoSci`/`ccproxy`. The operator may be running a live EvoScientist session, background server (`6174`), WebUI (`4716`), or `ccproxy` (`8000`) while you work. Only stop PIDs you spawned yourself. That includes `EvoSci server stop`/`start` and `EvoSci deploy` on the default ports: they act on the operator's live server. Use the sandbox in [Running EvoScientist by hand](#running-evoscientist-by-hand) instead.
3. **Never edit `constraints.txt` by hand.** It is generated from `uv.lock` on release PRs only (`release_guard.sh` enforces this). Only touch `pyproject.toml` / `uv.lock` when a task requires a dependency change.
4. **Respect pinned dependency ceilings in `pyproject.toml`.** Every upper-bound cap in `pyproject.toml` has an inline comment explaining the upstream regression it guards against. Do not bump or unpin a capped dependency without verifying the original issue is fixed.
5. **Never weaken the settings API boundary.** The WebUI settings API refuses changes to anything that decides what the agent may run or who can reach it (approval modes, the shell allow list, bind hosts, browser origins, channel access). The list is `READ_ONLY_FIELDS` in `EvoScientist/langgraph_dev/settings_api.py`, covered by `tests/test_settings_api.py`. Don't remove entries; add new settings of that kind to it (see [SECURITY.md](SECURITY.md)).
6. **Never make live network or LLM calls in tests.** The entire test suite runs offline with zero API keys on Linux and Windows.
7. **Don't fight the test home redirect.** `EvoScientist.paths` resolves its folders at import, so `tests/conftest.py` points `HOME` at a throwaway folder before EvoScientist is imported—setting `EVOSCIENTIST_DATA_DIR` in a fixture is too late. Use `tmp_path` / `tmp_workspace` for files, and don't import `EvoScientist` from a pytest plugin or anything that loads before `conftest.py`.

## Hit every surface

The most common bug in this repo is fixing one entry point and forgetting the others. Before calling a user-facing change done, check which of these apply:

- **Interactive TUI** (`EvoScientist/cli/tui_*.py`, `cli/widgets/`)
- **Classic CLI & single-shot** (`EvoScientist/cli/interactive.py`, `-p`, `--prompt-file`, `--output-format stream-json`)
- **WebUI & Desktop backend** (`EvoScientist/deploy/`, `EvoScientist/langgraph_dev/`, `EvoSci deploy --json`)
- **Messaging channels & `EvoSci serve`** (`EvoScientist/channels/`)
- **Slash commands & CLI subcommands** (`EvoScientist/commands/`, `EvoScientist/cli/*_cmd.py`)
- **Config, env vars, & onboarding** (`EvoScientist/config/`)
- **Sync & async sub-agents** (`EvoScientist/subagents/`, `EvoScientist/gateway/`)
- **Cross-platform** (macOS Apple silicon, Linux, native Windows + Git Bash)
- **Docs** — check whether `README.md`, `docs/guides/`, or `docs/recipes/` became inaccurate (follow [Documentation](#documentation)).

The same bug on several surfaces is one problem: fix every surface in the same PR. "One problem per PR" is about not bundling *unrelated* changes.

## Verifying changes

First-time setup (includes optional channel extras so channel tests don't skip):

```bash
uv sync --dev --extra all-channels
```

Iterate fast on targeted tests while developing:

```bash
uv run pytest tests/path/to/test_relevant.py -v
```

Before pushing or finishing a task, run the linter, formatter check, and affected tests (CI runs `ruff check`, `ruff format --check`, and `pytest -v -n 4 --dist worksteal` across Ubuntu and Windows):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -n auto --dist worksteal   # full suite when changes cross modules
```

Keep tests deterministic and fast—no arbitrary `time.sleep` waits, and respect the per-test timeout in `pyproject.toml`.

If the operator installed the pre-commit hooks, `git commit` runs Ruff and may reformat files and abort the commit. Stage the reformatted files and commit again.

### Running EvoScientist by hand

To exercise a change in the real app, run it against a throwaway data folder, config folder, and ports so it never touches the operator's install or live server:

```bash
SANDBOX=$(mktemp -d)
export EVOSCIENTIST_DATA_DIR="$SANDBOX/data" XDG_CONFIG_HOME="$SANDBOX/config"
export EVOSCIENTIST_LANGGRAPH_DEV_PORT=16174 EVOSCIENTIST_WEBUI_PORT=14716 EVOSCIENTIST_CCPROXY_PORT=18000
uv run --no-sync EvoSci ...
```

The sandbox starts with no API keys. If the run needs a real model, ask the operator instead of copying keys out of their config. Stop any server you started there with `EvoSci server stop` from the same shell.

## Pull requests

- Never open a PR unless the user explicitly asks you to.
- One problem per PR. Keep unrelated refactors and cleanups separate.
- Title commits and PRs as Conventional Commits with the area as scope, e.g. `fix(channels): avoid echoing ask_user attachments`. Use `feat`, `fix`, `docs`, `test`, or `refactor`; `release: vX.Y.Z` is reserved for maintainers.
- Fill in [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).
- Explain the problem in a sentence or two, what changed, and **how you verified it** (reproduction + exact commands/observations, not just "tests pass").
- Never commit temporary plans, scratch notes, or PR-only screenshots.

## Documentation

Keep docs human-readable, accurate, and durable.

- **Don't append PR changelogs onto guides.** When behavior changes, rewrite or remove the outdated prose instead of bolting another paragraph onto the end.
- **Avoid content that goes stale immediately.** Don't hardcode version numbers, exact test/sub-agent counts, transient model lists, or internal class catalogs in prose when the code or CLI already owns them.
- **Lead with the mental model and common case.** Put copy-pasteable commands and tables first; tuck obscure edge cases into brief notes or collapsible `<details>` blocks.

## Where code lives

| Path | Responsibility |
|---|---|
| `EvoScientist/EvoScientist.py`, `prompts.py`, `backends.py` | Main agent assembly (`create_deep_agent`), system prompts, sandbox/memory/skills composite backends, action safety checks |
| `EvoScientist/paths.py` | Data, config, and workspace folder resolution (read once at import) |
| `EvoScientist/skills/` | Built-in skills shipped with the agent |
| `EvoScientist/cli/` | Entrypoint (`EvoSci`), Textual TUI, classic CLI, widgets, slash-command UI, CLI subcommands |
| `EvoScientist/commands/` | Slash-command definitions and handlers shared across interactive surfaces |
| `EvoScientist/config/` | `config.yaml` schema (`settings.py`), persistence (`store.py`), provider key resolution, interactive `onboard/` wizard |
| `EvoScientist/llm/` | Multi-provider chat model factory, model registry, fallback chains, retries, custom endpoints, ChatGPT OAuth |
| `EvoScientist/deploy/` | Background LangGraph server lifecycle, WebUI launcher, `--json` desktop protocol, Cloudflare tunnel |
| `EvoScientist/gateway/` | UI-agnostic runtime layer bridging local in-process runs and the shared LangGraph server |
| `EvoScientist/subagents/` | Built-in sub-agent YAML specs and expert containers |
| `EvoScientist/memory/` | Cross-session profile memory, observation extraction/linking, and AutoSkills distillation |
| `EvoScientist/middleware/` | Memory injection, context editing, summarization, tool selection, approval/HITL policies |
| `EvoScientist/channels/` | Unified message bus, middleware pipeline, and messaging platform adapters |
| `EvoScientist/mcp/` | `mcp.yaml` config, client lifecycle, wildcard tool filtering, and per-agent tool routing |
| `EvoScientist/cron/`, `proactive/` | Cron scheduled tasks and proactive channel follow-ups |
| `EvoScientist/stream/` | Native event emitter, tracker, and `stream-json` JSONL serialization |
| `EvoScientist/setup/`, `updater.py`, `uv_tool.py` | One-line installer helpers, `EvoSci setup`, and `EvoSci update` |
| `EvoScientist/tools/` | Built-in tools (web search, `think_tool`, skill manager, etc.) |
| `tests/` | Offline unit and integration test suite (no API keys required) |

## Taste

- Push platform, provider, and channel quirks to the edges (`llm/`, `channels/`, `deploy/`); keep core orchestration clean.
- Use modern Python 3.11+ typing (`X | Y`, `list[str]`, `dict[str, Any]`); `ruff check` enforces the rest.
- Keep comments sparse. Only write a code comment for a non-obvious constraint, upstream quirk, or invariant (e.g., why a dependency version is capped or why a Windows FD fallback exists).
- Preserve backward-compatible user config/data paths unless a migration is explicitly part of the task.
- Make error messages actionable: tell the operator *what failed* and *the exact command or setting to fix it*.
