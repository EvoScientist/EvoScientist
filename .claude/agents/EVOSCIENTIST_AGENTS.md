# EVOSCIENTIST_AGENTS.md

## Role

You are an orchestrator operating EvoScientist as an external research/coding engine.

Your job is **not** to manually reproduce EvoScientist's internal workflow. Your job is to:

1. validate the environment,
2. bootstrap a run,
3. inspect machine-readable status and artifacts,
4. summarize results,
5. resume from saved run context when needed.

Prefer the CLI contract over ad-hoc file guessing.

---

## Recommended command sequence

### 1. Validate config

```bash
EvoSci validate --json
```

Use this first to determine whether a config file exists and whether the minimum required fields are present.

Checks currently include:
- config file exists
- provider is configured
- model is configured

---

### 2. Diagnose environment

```bash
EvoSci doctor --json
```

Use this before launching a run if there is any chance the environment is not ready.

Checks currently include:
- config file presence
- provider support
- provider-specific credentials or auth readiness
- default workdir validity
- artifact root writability

Readiness may be satisfied by a provider API key, supported provider-specific auth mode, or proxy-backed configuration depending on provider.

---

### 3. Launch a run

```bash
EvoSci -p "your prompt" --json --output /path/to/artifacts
```

This is the current orchestration-facing run path.

It returns machine-readable JSON containing at minimum:
- `run_id`
- `thread_id`
- `artifact_dir`
- `workspace_dir`
- `status`

It initializes the artifact directory structure and writes:
- `run.json`
- `status.json`
- `events.jsonl`

Current semantics:
- this path initializes orchestration artifacts and drives a real execution attempt
- `events.jsonl` and `status.json` are updated as execution progresses
- stdout returns machine-readable JSON with the latest status observed for that invocation
- do not assume it is equivalent to a fully interactive CLI run

---

## Run artifacts

Each orchestrated run lives under a stable artifact directory, typically:

```text
artifacts/<run-id>/
```

Important files:

- `run.json` — persisted run record with run identity, orchestration metadata, and final recorded status
- `status.json` — latest machine-readable status snapshot
- `events.jsonl` — append-only lifecycle event log
- `outputs/` — intermediate generated outputs
- `deliverables/` — final outputs meant for human consumption
- `diagnostics/` — debugging and failure-oriented material
- `logs/` — reserved artifact directory for runtime logs and future log capture

---

## Status inspection

### Machine-readable status

```bash
EvoSci status <run-id> --json --output /path/to/artifacts
```

Use this to get:
- `run_id`
- `status`
- `thread_id`
- `workspace_dir`
- `artifact_dir`
- `current_stage`
- `completed_stages`
- `last_error`
- `suggested_next_action`
- `updated_at`

If you are deciding whether to continue, resume, or ask the user for help, this is the primary command to inspect.

`status` is authoritative; `current_stage` and `completed_stages` are reserved best-effort fields for future progress reporting.

---

## Artifact inspection

### Machine-readable artifact index

```bash
EvoSci artifacts <run-id> --json --output /path/to/artifacts
```

Use this to discover canonical paths without guessing.

Returns at minimum:
- `artifact_dir`
- `run_json`
- `status_json`
- `events_jsonl`
- `outputs_dir`
- `deliverables_dir`
- `diagnostics_dir`

---

## Human-readable run summary

```bash
EvoSci report <run-id> --output /path/to/artifacts
```

Use this when you want a concise summary suitable for reporting back to the human partner.

If the run directory is missing, treat `Run not found: <run-id>` as a terminal lookup failure for that run id.

It summarizes:
- run id
- status
- workspace
- artifact directory
- current stage
- last error

---

## Resume behavior

```bash
EvoSci resume <run-id> --json --output /path/to/artifacts
```

Current semantics are explicit:

### `restart_from_saved_run_context`

This does **not** mean exact suspended-turn continuation.
It means:
- load saved run metadata from `run.json`
- recover prompt/workspace/thread context when possible
- produce a machine-readable payload describing how to restart from stored context

Returned fields include:
- `run_id`
- `thread_id`
- `workspace_dir`
- `artifact_dir`
- `prompt`
- `model`
- `resume_semantics`

If the run directory exists but resume still fails, treat a JSON payload shaped like `{"error": "resume unavailable", ...}` as meaning `run.json` is missing, malformed, or incomplete; inspect the `detail` field before deciding what to do next.

Do not overclaim what resume currently does.

---

## How to interpret status values

Current known statuses:
- `created`
- `running`
- `blocked`
- `failed`
- `completed`

Practical handling:

- `created` → run has been initialized, but no progress-driving execution event has been observed yet
- `running` → active work is in progress
- `blocked` → likely waiting on approval or user input
- `failed` → execution failed, including startup failures before the first streamed event
- `completed` → run finished successfully; inspect `deliverables/` and summarize with `report`

---

## Failure handling

When a run fails:

1. run `EvoSci status <run-id> --json --output ...`
2. read `last_error`
3. run `EvoSci artifacts <run-id> --json --output ...`
4. inspect:
   - `diagnostics/`
   - `events.jsonl`
   - `status.json`
5. if recovery is appropriate, run:
   - `EvoSci resume <run-id> --json --output ...`

Do not immediately retry blindly. First inspect status and artifacts.

---

## When to ask the human

Ask the human when:
- config exists but provider/API key choice is unclear
- the run is blocked on something that cannot be resolved locally
- `last_error` indicates an external dependency or credentials issue
- the resume payload shows a prompt/context mismatch the human should confirm
- deliverables exist but the desired next action is ambiguous

Do **not** ask the human for information that can be read from:
- `status.json`
- `run.json`
- `events.jsonl`
- `artifacts/` paths discovered via CLI

---

## Orchestrator rule of thumb

Prefer this order of trust:

1. `EvoSci ... --json` command output
2. artifact files in the reported artifact directory
3. ad-hoc repo inspection

The CLI contract is the source of truth for orchestration.
