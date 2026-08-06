# Workspace-Relative Answer Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent EvoScientist's final answers from linking workspace artifacts with host absolute paths that the WebUI cannot resolve.

**Architecture:** Extend the existing shared `WRITING_GUIDELINES` prompt section so the main agent always emits workspace-relative POSIX Markdown links. Protect the behavior with a focused assertion against the fully assembled system prompt.

**Tech Stack:** Python 3.11+, pytest, Ruff

## Global Constraints

- Change the main EvoScientist system prompt only; do not modify `@evoscientist/webui` or add response post-processing.
- Workspace artifact links must be relative to the active workspace root and use POSIX `/` separators.
- Host absolute paths, `file://` URLs, and `sandbox:` URLs must be forbidden for workspace artifacts.

---

### Task 1: Constrain workspace artifact links in final answers

**Files:**
- Modify: `tests/test_prompts.py`
- Modify: `EvoScientist/prompts.py`

**Interfaces:**
- Consumes: `get_system_prompt(dangerous: bool = False, cwd: str | None = None) -> str`
- Produces: An assembled system prompt containing explicit workspace-link formatting rules.

- [ ] **Step 1: Write the failing regression test**

Add this method to `TestWritingGuidelines`:

```python
def test_requires_workspace_relative_artifact_links(self):
    result = get_system_prompt()
    assert "relative to the workspace root" in result
    assert "POSIX `/` separators" in result
    assert "host absolute paths" in result
    assert "`file://`" in result
    assert "`sandbox:`" in result
    assert "[report](reports/final_report.md)" in result
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
uv run pytest tests/test_prompts.py::TestWritingGuidelines::test_requires_workspace_relative_artifact_links -v
```

Expected: FAIL because the current system prompt does not contain the workspace-relative link rule.

- [ ] **Step 3: Add the minimal prompt rule**

Append these bullets to `WRITING_GUIDELINES` in `EvoScientist/prompts.py`:

```text
- When linking to a file in the active workspace, use a Markdown link target relative to the workspace root with POSIX `/` separators, for example `[report](reports/final_report.md)`.
- Never use host absolute paths, `file://` URLs, or `sandbox:` URLs for workspace files.
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
uv run pytest tests/test_prompts.py::TestWritingGuidelines::test_requires_workspace_relative_artifact_links -v
```

Expected: PASS.

- [ ] **Step 5: Run prompt regression tests and lint**

Run:

```powershell
uv run pytest tests/test_prompts.py -v
uv run ruff check EvoScientist/prompts.py tests/test_prompts.py
uv run ruff format --check EvoScientist/prompts.py tests/test_prompts.py
```

Expected: all commands exit successfully with no failures or formatting changes required.

- [ ] **Step 6: Review the diff and commit**

Run:

```powershell
git diff --check
git diff -- EvoScientist/prompts.py tests/test_prompts.py
git add EvoScientist/prompts.py tests/test_prompts.py docs/superpowers/plans/2026-08-06-workspace-relative-answer-links.md
git commit -m "fix: use workspace-relative artifact links"
```

Expected: one focused commit containing the regression test, prompt rule, and implementation plan.
