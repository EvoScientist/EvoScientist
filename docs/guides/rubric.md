# Grading a Run Against a Rubric

A **rubric** is a checklist of acceptance criteria. Whenever the agent finishes, a separate grader checks the conversation and workspace files against your checklist—if anything is missing, the agent gets the gaps back and keeps working (up to **3 passes per task**).

Use `--rubric` for one-off scripted runs, or `/rubric` inside interactive TUI/CLI threads.

## Single-Shot Runs (`--rubric`)

Requires `-p`. Pass criteria inline or read them from a file with `@path` (supports `~`):

```bash
EvoSci -p "Reproduce table 2 and plot the ablation" --auto-mode \
  --rubric "- results/table2.csv has one row per baseline
- figures/ablation.png exists"

EvoSci -p "Reproduce table 2" --auto-mode --rubric @criteria.md
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Grader returned **satisfied** |
| `3` | Run finished without a satisfied verdict: the 3-pass limit was reached, the grader failed, the rubric could not be evaluated, or the run stopped before grading (a cancelled approval/question, or the approval round limit) |
| `1` | The run itself errored |

With `--output-format stream-json`, each pass emits `rubric_evaluation_start` and `rubric_evaluation` events (see [`stream-json`](stream-json.md)).

### How the Grader Works

- **Model:** Uses your auxiliary model if configured, otherwise the main model (up to 12 model calls per attempt; 24 if a retry is needed for omitted criteria).
- **Read-only workspace access:** The grader can read conversation history and list/read workspace files, but **cannot execute commands**—so criteria like *"pytest passes"* are verified from command output in the conversation.
- **Quiet output:** The grader's internal steps aren't printed—only a brief status line and the final verdict.
- **Single-shot scope:** Only applies to that run, runs sub-agents in-turn instead of in the background, and requires the local runtime (refused on the `langgraph_server` gateway).

## Interactive Sessions

In the TUI and classic CLI, attach checklists directly to the current thread (saved across `/resume` and restarts once your next message runs):

| Command | What it does |
|---|---|
| `/rubric set <criteria>` or `@file` | Grades every turn starting from your next message until cleared. Setting it again replaces the checklist. |
| `/rubric next <criteria>` or `@file` | Grades **only** the next task, then clears itself (takes precedence over `set` for that one turn). |
| `/rubric` (or `/rubric show`) | Displays active and pending checklists. |
| `/rubric clear` | Removes both checklists. |

*(Multiline text works; quote paths with spaces like `@"my criteria.md"`. The status bar shows `rubric: on`, `rubric: next`, or `rubric: on+next`. Because [`/goal`](goal.md) shares the sticky slot, `/rubric set` and `/rubric clear` cancel an active goal.)*

Each message you send gets up to 3 passes. A message that continues a deferred or cut-short task keeps that task's start (the grader still reads from the message that started it), but its pass count starts again.

### Background Work & Deferral

- **Interactive sessions:** If the agent leaves sub-agents or `run_in_background` processes running on the thread, grading is **deferred** (`Acceptance check deferred: ...`) and runs automatically when their completion notification arrives.
  - A `run_in_background` process that never exits holds grading back only until your next message.
  - A sub-agent task that never ends keeps the grading waiting; `/rubric clear` drops it.
  - After a CLI restart, a process the grading waits for is usually reported as no longer tracked, and the turn that notice starts runs the grading (the process may still be running).- **Single-shot (`--rubric`) runs:** Do not defer—if the agent stops while a background command is still running, the grader flags the missing deliverable and the agent gets another pass to wait for it. Each early stop uses one of the 3 passes.

### Limitations

- Not active on WebUI, `EvoSci serve`, or channel sessions (`/rubric` is refused in channels and on the `langgraph_server` gateway).
- A run paused on an approval and resumed from another client (like the WebUI) finishes without grading.
- External clients cannot trigger grading unless `configurable.rubric_run` is explicitly set.
