# Grading a run against a rubric

`--rubric` gives a single-shot run a checklist of acceptance criteria. Each time the agent stops, a separate grader checks the work against the checklist. If a criterion is not met, the agent gets the gaps back and keeps working.

```bash
EvoSci -p "Reproduce table 2 and plot the ablation" --auto-mode --rubric "- results/table2.csv has one row per baseline
- figures/ablation.png exists"

EvoSci -p "Reproduce table 2" --auto-mode --rubric @criteria.md
```

`@path` reads the checklist from a file (relative to the current directory; `~` works). `--rubric` needs `-p`.

## What happens

- The grader runs after the agent stops, at most 3 times per run.
- It sees this run's conversation and can list and read files in the workspace. It cannot run commands, so a criterion such as "the tests pass" is judged from what the conversation shows.
- The rubric applies to this run only. Later turns on the same thread are not graded, and a run that is parked at an approval and then continued from another client (for example the WebUI) finishes without grading.
- Sub-agents run in-turn during a `--rubric` run instead of in the background.
- The grader's own file checks are shown as a sub-agent named `rubric_grader`.
- The grader uses the auxiliary model when one is configured, otherwise the main model.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | The last grading pass was satisfied |
| `3`  | The run finished without a satisfied verdict: the pass limit was reached, the grader failed, or the rubric could not be evaluated |
| `1`  | The run itself failed |

With `--output-format stream-json` each pass also emits `rubric_evaluation_start` and `rubric_evaluation` events; see [stream-json](stream-json.md).

## Limits

- A process started with `run_in_background` may still be running when the agent stops. The grader then reports what is missing and the agent can wait for it, but each early stop uses one of the 3 passes.
- One grader attempt may make at most 12 model calls; an attempt that needs more ends as a grader failure. A pass makes a second attempt when its verdict leaves criteria out, so a pass can use up to 24.
- `--rubric` is refused when single-shot runs are configured to use the `langgraph_server` gateway.
