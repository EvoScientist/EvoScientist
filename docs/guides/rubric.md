# Grading a run against a rubric

A rubric is a checklist of acceptance criteria. Each time the agent stops, a separate grader checks the work against it; if a criterion is not met, the agent gets the gaps back and keeps working. `--rubric` grades one single-shot run; `/rubric` grades the turns of a thread in the TUI or the interactive CLI (see [Interactive sessions](#interactive-sessions)).

```bash
EvoSci -p "Reproduce table 2 and plot the ablation" --auto-mode --rubric "- results/table2.csv has one row per baseline
- figures/ablation.png exists"

EvoSci -p "Reproduce table 2" --auto-mode --rubric @criteria.md
```

`@path` reads the checklist from a file (relative to the current directory; `~` works). `--rubric` needs `-p`.

## What happens

- The grader runs after the agent stops, at most 3 times per run.
- It sees this run's conversation and can list and read files in the workspace. It cannot run commands, so a criterion such as "the tests pass" is judged from what the conversation shows.
- The rubric applies to this run only. Later turns on the same thread are not graded unless the thread holds a `/rubric` checklist, and a run that is parked at an approval and then continued from another client (for example the WebUI) finishes without grading.
- Sub-agents run in-turn during a `--rubric` run instead of in the background.
- Only runs the CLI marks are graded (`configurable.rubric_run` in the run's config): a single-shot run started with `--rubric`, and in an interactive session the turns of a thread that holds a `/rubric` checklist. A `rubric` that another client sends in a run's input is dropped and the run is not graded, unless that client also sets the marker.
- The grader's own work is not shown: a line says a check is running, then the verdict follows.
- The grader uses the auxiliary model when one is configured, otherwise the main model.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | The last grading pass was satisfied |
| `3`  | The run finished without a satisfied verdict: the pass limit was reached, the grader failed, the rubric could not be evaluated, or the run stopped before it was graded (a cancelled approval or question, the approval round limit) |
| `1`  | The run itself failed |

With `--output-format stream-json` each pass also emits `rubric_evaluation_start` and `rubric_evaluation` events; see [stream-json](stream-json.md).

## Interactive sessions

In the TUI and the interactive CLI, `/rubric` stores checklists on the current thread:

| Command | Effect |
|---------|--------|
| `/rubric set <criteria>` or `/rubric set @file` | Grade every turn from your next message on, until cleared. Setting again replaces the checklist. |
| `/rubric next <criteria>` or `/rubric next @file` | Grade only the next task; the checklist is dropped once that grading reaches a verdict. |
| `/rubric show` (or `/rubric`) | Show both checklists and any change not yet applied. |
| `/rubric clear` | Remove both. |

The text can span lines; quotes inside it are kept as typed. `@file` is relative to the session's working folder; a name with spaces is written `@"file name.md"`. A change applies with your next message: it is carried into that run and stored on the thread by the grader, so it survives `/resume` and a restart, while a change you never followed with a message is not saved.

- When both checklists are set, the next task uses `next`; later turns use the sticky one.
- Each message you send gets up to 3 passes. A message that continues a deferred or cut-short task keeps that task's start, so the grader still reads from the message that started it, but its count starts again; the notification turn that finishes a deferred grading continues the same count.
- Background work: when the turn left sub-agent tasks or `run_in_background` processes of this thread still running, grading is deferred (`Acceptance check deferred: ...`) and runs in the turn their completion starts, on the original task. A process that never exits holds grading back only until your next message. A sub-agent task holds back the grading of the task that started it, and one that was already running when that task's first message was sent is not waited for; a task that another client (the WebUI) starts on the thread while the grading is deferred is waited for too. A task that never ends keeps its grading waiting; `/rubric clear` drops it.
- The status bar shows `rubric: on`, `rubric: next` or `rubric: on+next`.
- A thread's [goal](goal.md) keeps its criteria in the place of the `/rubric set` checklist, so `/rubric set` and `/rubric clear` cancel the goal (the command says so).
- Approvals work as usual; revisions can bring more of them, within the existing per-turn limits.
- Not graded: single-shot runs without `--rubric`, the WebUI, `EvoSci serve`, and a notification turn for a thread other than the current one. `/rubric` is refused in channel sessions and when the session executes its runs on the `langgraph_server` gateway.

## Limits

- In a single-shot run, a process started with `run_in_background` may still be running when the agent stops. The grader then reports what is missing and the agent can wait for it, but each early stop uses one of the 3 passes. Interactive sessions defer instead (above).
- A deferred grading waits for a notification; when none comes (the agent observed the exit itself in a turn cut short before grading, the CLI restarted, or the notification could not be delivered), it runs at your next message.
- The live status of a sub-agent task is read before grading; a status that cannot be read never holds a grading back, so a grading can run while the task is in fact still running when the dev server is unreachable.
- One grader attempt may make at most 12 model calls; an attempt that needs more ends as a grader failure. A pass makes a second attempt when its verdict leaves criteria out, so a pass can use up to 24.
- `--rubric` is refused when single-shot runs are configured to use the `langgraph_server` gateway.
