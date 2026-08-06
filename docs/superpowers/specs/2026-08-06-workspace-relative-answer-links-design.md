# Workspace-Relative Answer Links Design

## Problem

EvoScientist can write a Markdown artifact inside the active workspace and then reference that artifact in its final answer using the host's absolute filesystem path. The WebUI treats workspace file links as workspace-relative paths, strips the leading slash, and consequently looks for a duplicated path below the workspace. The file remains accessible from the FILE SYSTEM panel, but the answer link fails with `Path is not accessible.`

## Scope

Change EvoScientist's main system prompt only. Do not modify the separately published `@evoscientist/webui` package or add response post-processing middleware.

## Design

Add file-link rules to the shared `WRITING_GUIDELINES` section of the main system prompt:

- Links to files in the active workspace must use paths relative to the workspace root.
- Link targets must use POSIX `/` separators.
- Host absolute paths, `file://` URLs, and `sandbox:` URLs must not be used for workspace files.
- Include a concrete correct example so the model can copy the expected Markdown form.

The rule belongs in `WRITING_GUIDELINES`, rather than only in the experiment report step, because final answers may link to reports, figures, tables, logs, or other generated artifacts.

The writing sub-agent prompt remains unchanged. It drafts report content, while the main agent owns the final user-facing answer and link.

## Testing

Extend `tests/test_prompts.py` with a regression test against the assembled system prompt. The test will require the workspace-relative and POSIX-path instructions and the three forbidden link forms. The test must fail before the prompt change and pass afterward.

Run the focused prompt tests, followed by formatting/lint checks for the changed Python files.

## Success Criteria

- The assembled main system prompt explicitly instructs the model to emit workspace-relative POSIX Markdown links.
- The prompt explicitly forbids host absolute paths, `file://`, and `sandbox:` for workspace artifacts.
- Existing prompt tests and the new regression test pass.
- No WebUI or runtime behavior is changed in this patch.
