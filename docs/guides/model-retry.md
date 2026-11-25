# Retrying Transient Model Failures

When an LLM provider hiccups mid-request, EvoScientist retries the call in place without re-running completed tools. Any partial text streamed by the failed attempt is erased cleanly before the retry streams its answer.

## What Gets Retried (and What Doesn't)

- **Retried:** HTTP `408`, `409`, `425`, `429`, `5xx`; SDK connection/timeout errors; provider errors inside a stream or `200` body; responses ending with `finish_reason: "error"`.
  - *(Note: a mistyped `base_url` or stopped local server looks like a dropped connection and will retry until the budget runs out—check the error reason in the retry line.)*
- **Not retried:** Auth/permission errors (`401`/`403`), quota exhaustion, malformed requests, output-token cutoffs, or context-window overflows (overflows go straight to conversation summarization and retry once after compacting).

## Wait Budgets

Each failed model call gets a retry wait budget based on whether the run is interactive or unattended:

| Config key | Default | Applies to |
|---|---|---|
| `model_retry_wait_seconds` | `60` | Interactive sessions (TUI, CLI, `serve`), plus background memory workers and the observation linker (even under `auto_mode`) |
| `auto_mode_retry_wait_seconds` | `1800` | Unattended runs (`--auto-mode`, `auto_mode: true`, scheduled tasks, AutoSkills) |

```bash
EvoSci config set auto_mode_retry_wait_seconds 600
EvoSci config set model_retry_wait_seconds 0        # disable retries for interactive runs & memory workers
```

**Which config counts:** The tier travels with the run, so a `-p --auto-mode` run gets the unattended budget whether it runs in-process or on the background server. Runs the server starts on its own (WebUI, background sub-agents, scheduled tasks) follow the **server's** `config.yaml`; after changing these keys, run `EvoSci server stop` so the server picks them up.

### How the Budget Works

- **Clock starts on first failure:** Time spent on the initial attempt does not count against the budget (so a long generation that drops near the end still gets retried). Backoff starts at `1s` and doubles up to `10s` (interactive) or `2m` (unattended), honoring `Retry-After` headers when they fit in the remaining budget (otherwise the call gives up at once).
- **One exhausted budget ends retries for the rest of that run:** If a single model call burns through its entire budget (including inside a sub-agent), retries turn off for remaining calls in that run so a downed provider doesn't stall the run repeatedly. Starting a new message (or answering an approval/question) resets a fresh budget.
- **Calls that recover don't penalize later calls:** If a call retries for 55s and succeeds, the next call still gets a full 60s budget.
- **Fallback chains (`model_fallbacks`):** One attempt tries the primary model followed by the chain, and retries if *any* model in the chain failed transiently (starting from the primary again, using the shortest retry wait). If the budget runs out, the final error from the last model in the chain is raised (`-p` exits `1` with a resume hint).

<details>
<summary>Edge cases</summary>

- The budget decides whether another attempt may *start*; it doesn't cut off an attempt that hangs.
- A value that isn't a whole number of seconds (e.g. `60s`) falls back to the default, with a warning in the log.
- Every attempt starts from the primary again, so a primary that fails permanently (wrong key, unknown model) is still called once per attempt.
- A fallback whose request is rejected as malformed ends the chain for that attempt.

</details>

## What You See

Every retry prints a single line (prefixed with the sub-agent name or `rubric_grader` when applicable) and emits a `model_retry` event under `--output-format stream-json`:

```text
Model request failed (503: ... starting up); retrying in 8 s (attempt 2)
```

Partial streamed output from the failed attempt is cleared from the current view, the final answer, and channel replies (the same cleanup happens when a `model_fallbacks` entry takes over mid-stream).
