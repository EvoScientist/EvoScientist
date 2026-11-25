# Working Toward a Goal (`/goal`)

`/goal` turns an interactive thread (TUI or classic CLI) into a self-checking loop: you state an objective, EvoScientist drafts concrete acceptance criteria for your approval, and then a grader checks every turn until the goal is actually satisfied.

```text
/goal Reproduce table 2 of the paper and plot the ablation
```

Want to supply your own checklist without drafting? Use [`/rubric set`](rubric.md#interactive-sessions) instead.

## Commands

| Command | What it does |
|---|---|
| `/goal <objective>` | Drafts acceptance criteria and opens them for your review. Once accepted, the objective is sent as your message and the agent starts working. A thread that already has a goal keeps it until you accept the new one. |
| `/goal` (or `/goal show`, `/goal status`) | Shows the current goal, status, note, and criteria. |
| `/goal amend <feedback>` | Revises an active or blocked goal and its criteria (goes through review again), then sends your feedback to continue. |
| `/goal pause` | Pauses grading and tells the agent to stop pursuing the goal while keeping it saved. |
| `/goal resume` | Resumes a paused goal and starts a turn immediately. |
| `/goal clear` | Deletes the goal and its criteria. |

*(Subcommands like `show`, `pause`, `resume`, `clear` only trigger when nothing follows them—so `/goal show me the ablation results` starts a new goal. Objectives can span multiple lines; `@file` mentions are not expanded, so just name the file for the agent to read.)*

## How the Review & Grading Loop Works

1. **Draft & review:** The main model inspects workspace files (up to 12 calls, following `model_fallbacks` on failure) and proposes a checklist. You can **accept**, **cancel**, or type feedback to redraft. Nothing enters the conversation until you accept.
   - In the TUI, an unanswered review times out after 5 minutes.
   - Under `--auto-mode`, the draft is auto-accepted with a visible warning (`Goal accepted without review (auto mode).`).
2. **Automatic grading:** Each time the agent finishes a turn, the grader checks the work against the criteria (up to **3 passes per task**, waiting for any thread background work—see [Grading With a Rubric](rubric.md)).
3. **Lifecycle states** (shown in the status bar):
   - `goal: active` — Grading after every turn. If 3 passes aren't enough, the goal stays `active` and the agent waits for your next message, `/goal amend`, or `/goal clear`.
   - `goal: blocked` — The agent reported it needs input from you; grading skips that turn and your next message unblocks it.
   - `goal: paused` — Saved but not actively graded.
   - `goal: done` — Criteria passed; grading stops while the goal stays inspectable via `/goal`.

## Interaction With `/rubric`

A goal uses the same thread slot as `/rubric set`:
- Running `/rubric set` or `/rubric clear` cancels the active goal on your next message (status bar shows `goal: ending` until then).
- `/rubric next` still grades a single one-off turn independently without completing the goal.
- Accepting or resuming a goal clears any pending unapplied `/rubric` changes (and accepting drops an unused `/rubric next`), with a warning beforehand.

## Limits & Surface Support

- **Supported surfaces:** TUI and classic interactive CLI only. Refused in messaging channels and on the `langgraph_server` gateway.
- **Switching surfaces:** If you open the same thread in WebUI, `EvoSci serve`, or `-p`, the goal is ignored there—but stays intact when you return to the TUI/CLI.
- **Length caps:** Objective ≤ 8,000 chars; objective + criteria combined ≤ 12,000 chars.
- **Interrupted review:** Closing the app or pressing Ctrl+C during a review discards the draft on your next message—just run `/goal` again. Answering the review from a surface without goal support discards it too.
- **Drafting errors:** Model retries and fallback switches while drafting aren't announced; a draft that fails is reported with its reason.
