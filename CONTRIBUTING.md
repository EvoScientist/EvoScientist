# Contributing to EvoScientist

Thanks for caring enough about EvoScientist to help make it better.

Like most fast-moving AI projects, **we are bottlenecked by review bandwidth, not by code generation**. Reviewing, testing, and maintaining a PR long-term costs far more time than writing one. Please read this before opening a PR so nobody's time gets wasted.

> **Using an AI coding agent on this repo?** Make sure it reads [AGENTS.md](AGENTS.md) first. You are still responsible for understanding, testing, and owning every line in your PR. Keep your own instructions out of the tracked `AGENTS.md` and `CLAUDE.md`: put them in your tool's local, untracked file (for Claude Code, `CLAUDE.local.md`, which is gitignored).

---

## What Belongs in a PR (and What Doesn't)

**Always welcome:**
- **Bug fixes** for real issues you actually hit (crashes, regressions, Windows/macOS quirks, broken provider/channel behavior)
- **Focused improvements** to existing core features
- **Documentation fixes, guides, and recipes** that make EvoScientist clearer to use
- **Tests** covering edge cases or regressions

**Discuss in an issue first (before writing code):**
- New core features, new CLI flags, or changes to how EvoScientist behaves by default
- New LLM providers or messaging channels
- Architectural or UI refactors

**Belongs elsewhere (likely closed if opened here):**
- **Domain-specific workflows or research pipelines** — contribute these as skills to [**EvoSkills**](https://github.com/EvoScientist/EvoSkills) instead of baking them into core.
- **Drive-by AI-generated refactors or "cleanup" PRs** with no clear bug or user impact.
- **Bundled mega-PRs** that mix a bug fix, a refactor, and a new feature together.

---

## High-Value Ways to Help (Without Even Opening a PR)

- **Confirm or reproduce open bugs** with your OS, Python version, provider, and minimal steps.
- **Flag duplicates or already-fixed issues** so maintainers can close stale tickets.
- **Test open PRs** that fix a bug you care about and comment with what happened on your machine—a real test report beats opening a competing PR.

---

## PR Rules

1. **One problem per PR.** Include the code, tests, and doc updates needed for that fix. Keep unrelated cleanups in separate PRs.
2. **Show that it works.** State how you reproduced the bug (or tested the change), the exact commands you ran, and what you observed. *"Tests pass"* alone is not verification. For TUI or WebUI changes, attach before/after screenshots or a short recording to the PR description (don't commit them to the repo).
3. **Respect the security model.** Read [SECURITY.md](SECURITY.md) before touching approvals, shell sandboxing, server bind hosts, or the WebUI settings API.
4. **Stay responsive and mergeable.** Rebase on `main` if conflicts appear and reply to review feedback.
5. **Follow our doc style.** If your PR updates docs, rewrite or trim affected sections directly—don't tack changelog paragraphs onto the end of guides.

---

## Local Development Setup

Requires Python **3.11+** (CI tests 3.11–3.13) and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/EvoScientist.git
cd EvoScientist

# 2. Install dev tools + optional channel dependencies (so channel tests run)
uv sync --dev --extra all-channels

# 3. Install pre-commit hooks (optional but recommended)
uv run pre-commit install
```

### Running Checks Locally

All tests run offline—no LLM API keys or network required. The test, lint, and format commands CI runs are in [AGENTS.md → Verifying changes](AGENTS.md#verifying-changes), along with the codebase map, the multi-surface checklist, and how to run EvoScientist without touching your own install. It's written for coding agents but works just as well for humans.

---

## Maintainer Note: Releases & `constraints.txt`

Normal PRs never touch `constraints.txt`. Only release PRs (`release: vX.Y.Z`) regenerate it from `uv.lock` so tag-based installs pin the exact tested dependencies:

```bash
uv lock
uv export --frozen --no-hashes --no-dev --no-emit-project --all-extras --no-header --no-annotate -o constraints.txt
```

`release-check.yml` fails a release PR whose `uv.lock` is stale or whose `constraints.txt` does not match it, and `publish.yml` / `docker.yml` run the same guard when the release is published and refuse to publish such a release. Releases are immutable: a refused or wrong release is fixed by a new patch version; its tag never moves and its version cannot be reused.

---

## Questions?

Come chat on [Discord](https://discord.gg/AZ9ZMXkunY) or WeChat (see [README.md](README.md#-contributing--community)).
