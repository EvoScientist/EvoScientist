# Contributing to EvoScientist

Thanks for caring enough about EvoScientist to help make it better.

Like most fast-moving AI projects, **we are bottlenecked by review bandwidth, not by code generation**. Reviewing, testing, and maintaining a PR long-term costs far more time than writing one. Please read this before opening a PR so nobody's time gets wasted.

> **Using an AI coding agent on this repo?** Make sure it reads [AGENTS.md](AGENTS.md) first. You are still responsible for understanding, testing, and owning every line in your PR.

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

Requires Python **3.11–3.13** and [uv](https://docs.astral.sh/uv/).

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

All tests run offline—no LLM API keys or network required.

```bash
# Fast iteration on a single file
uv run pytest tests/path/to/test_foo.py -v

# Full parallel test suite (mirrors CI)
uv run pytest -n auto --dist worksteal

# Lint + format check (both run in CI)
uv run ruff check .
uv run ruff format --check .
```

See [AGENTS.md](AGENTS.md) for the codebase map, multi-surface checklist, and local safety rules.

---

## Maintainer Note: Releases & `constraints.txt`

Normal PRs never touch `constraints.txt`. Only release PRs (`release: vX.Y.Z`) regenerate it from `uv.lock` so tag-based installs pin the exact tested dependencies:

```bash
uv lock
uv export --frozen --no-hashes --no-dev --no-emit-project --all-extras --no-header --no-annotate -o constraints.txt
```

`release-check.yml` rejects release PRs with a stale `uv.lock` or mismatched `constraints.txt`. Releases are immutable—fix a bad release with a new patch version rather than moving tags.

---

## Questions?

Come chat on [Discord](https://discord.gg/AZ9ZMXkunY) or WeChat (see [README.md](README.md#-contributing--community)).
