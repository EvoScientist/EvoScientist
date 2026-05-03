# Spec — Issue #174: feat: session audit log of all operations performed

> Source: https://github.com/EvoScientist/EvoScientist/issues/174

## Problem

After a research session ends, there is no persistent record of what the agent actually did. Tool invocations, file operations, shell commands, web searches, and sub-agent delegations are only visible by scrolling the TUI history — which disappears once the session ends. For long or automated sessions this makes post-hoc review impractical and there is no tamper-evident trail.

## Acceptance criteria

- [ ] Each session writes an append-only JSONL audit log to `~/.evoscientist/sessions/<thread_id>/audit.jsonl`
- [ ] Every `tool_call` event is logged with: `ts`, `kind="tool_call"`, `tool`, `args`, `tool_id`
- [ ] Every `tool_result` event is logged with: `ts`, `kind="tool_result"`, `tool`, `success`, `content` (truncated to 2000 chars)
- [ ] Every `subagent_start` / `subagent_end` event is logged with: `ts`, `kind`, `name`, `agent_id`
- [ ] Session metadata (start time, model, thread_id) is written as the first line on session open
- [ ] Session end time is appended as the last line on session close
- [ ] Log file is created lazily (only when the first loggable event arrives)
- [ ] Audit logging failures are silently swallowed — they must never crash the TUI
- [ ] A new `AuditLogger` class in `EvoScientist/audit.py` owns all I/O
- [ ] Tests cover: log creation, each event kind, truncation, error swallowing

## Approach

Intercept events in the existing `_stream_response` event loop in `tui_interactive.py`. After each `elif event_type == "tool_call"` / `"tool_result"` / `"subagent_start"` / `"subagent_end"` / `"done"` branch, call `audit_logger.log(event_type, event)`. The `AuditLogger` is instantiated once per conversation turn (or per session) and holds the open file handle.

The log path follows `DATA_DIR / "sessions" / thread_id / "audit.jsonl"` using the existing `paths.DATA_DIR`.

Chosen over post-hoc export because incremental writes survive crashes.

## Files likely touched

- `EvoScientist/audit.py` — new: `AuditLogger` class
- `EvoScientist/paths.py` — add `SESSIONS_DIR` constant
- `EvoScientist/cli/tui_interactive.py` — instantiate logger, call `log()` in event loop
- `tests/test_audit.py` — new: unit tests for AuditLogger

## Risk / blast radius

- **Scope**: small
- **Breaking**: no
- **Migration needed**: no
- All audit I/O is wrapped in try/except so failures are silent

## Test plan

- **Unit**: `tests/test_audit.py` — test log creation, each event kind, content truncation, error swallowing, session metadata lines
- **Integration**: not required (file I/O is simple enough for unit tests)
- **Visual**: none (no UI change)

## Out of scope

- `/audit` TUI command to display a human-readable summary (follow-up)
- `--audit-log` CLI flag (follow-up)
- Tamper-evidence / cryptographic signing (follow-up)
- CLI (non-TUI) audit logging (follow-up)
