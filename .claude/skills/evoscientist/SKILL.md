---
name: evoscientist
description: Use when invoking EvoScientist from Claude Code to start a run, inspect run state, read artifacts, diagnose environment issues, or resume a prior run
argument-hint: status [run-id] | report [run-id] | artifacts [run-id] | doctor | resume <run-id> | <task...>
---

# EvoScientist — Orchestration Skill

## Description

Use EvoScientist as an external research/coding engine through its orchestration-friendly CLI contract.

This skill is intentionally thin. It should not re-implement EvoScientist's internal workflow in prompt text. Instead, it should:

- validate the config,
- diagnose the environment,
- bootstrap a run,
- inspect status and artifacts,
- summarize a run,
- recover with explicit resume semantics.

## Trigger Conditions

Activate this skill when the user:
- asks to run EvoScientist as an external engine
- wants Claude Code or another coding agent to orchestrate EvoScientist
- asks to start, monitor, diagnose, summarize, or resume an EvoScientist run
- mentions "EvoScientist" together with run/status/artifacts/resume

## Instructions

### 1. Validate config first

Run:

```bash
EvoSci validate --json
```

If validation fails, stop and report the missing fields.

### 2. Diagnose environment if needed

Run:

```bash
EvoSci doctor --json
```

Use this when config validity alone is not enough and runtime readiness is unclear.

Doctor checks provider support, provider-specific credentials/auth settings, default workdir validity, and artifact root writability.

### 3. Launch a run

Run:

```bash
EvoSci -p "<prompt>" --json --output <artifact-root>
```

Record these fields from the JSON output:
- `run_id`
- `thread_id`
- `artifact_dir`
- `workspace_dir`
- `status`

Treat returned `status` as the latest orchestration state observed for that invocation.

### 4. Check status

Run:

```bash
EvoSci status <run-id> --json --output <artifact-root>
```

Use this as the primary source of truth for orchestration state.

### 5. Inspect artifact paths

Run:

```bash
EvoSci artifacts <run-id> --json --output <artifact-root>
```

Use returned paths instead of guessing file locations.

### 6. Summarize a run for the user

Run:

```bash
EvoSci report <run-id> --output <artifact-root>
```

If the run directory is missing, treat `Run not found: <run-id>` as a terminal lookup failure for that run id.

### 7. Resume with explicit semantics

Run:

```bash
EvoSci resume <run-id> --json --output <artifact-root>
```

Current semantics are:
- `restart_from_saved_run_context`

Do not describe this as exact suspended-turn continuation.

If resume fails after the run directory is found, treat a JSON payload shaped like `{"error": "resume unavailable", ...}` as meaning the saved run context is incomplete or malformed and inspect the `detail` field.

## Important Rules

- Do not hardcode EvoScientist internal stages unless they are already exposed by the CLI contract.
- Do not guess artifact file paths if `EvoSci artifacts --json` can tell you.
- Do not claim a run is resumable without checking the resume payload.
- Prefer machine-readable CLI output over manual repo inspection.
- If the run is blocked or failed, inspect status and artifacts before asking the user.

## Key Files

- `.claude/agents/EVOSCIENTIST_AGENTS.md` — orchestrator-facing runbook
- `artifacts/<run-id>/run.json`
- `artifacts/<run-id>/status.json`
- `artifacts/<run-id>/events.jsonl`

## Tools Required

- Bash
- Read
- JSON inspection via command output
