# Working toward a goal

`/goal` gives the current thread an objective. EvoScientist drafts acceptance criteria for it, you review them, and from then on every turn of the thread is graded against those criteria until they are met. It is available in the TUI and the interactive CLI.

```text
/goal Reproduce table 2 of the paper and plot the ablation
```

## Commands

| Command | Effect |
|---------|--------|
| `/goal <objective>` | Draft acceptance criteria and ask you to review them. Once you accept, the objective enters the conversation as your message and the agent starts working. A thread that already has a goal keeps it until you accept the new one. |
| `/goal` (also `/goal show`, `/goal status`) | Show the goal, its status, its note and its criteria. |
| `/goal amend <feedback>` | Revise the goal and its criteria from your feedback, with the same review. Your feedback enters the conversation and the agent goes on. For a goal that is active or blocked. |
| `/goal pause` | Stop grading against the goal and tell the agent not to pursue it. The goal and its criteria are kept. |
| `/goal resume` | Continue a paused goal; the agent starts a turn at once. |
| `/goal clear` | Drop the goal and its criteria. |

`show`, `status`, `pause`, `resume` and `clear` are commands only when nothing follows them, so `/goal show me the money` sets a goal. The objective can span lines. An `@file` mention in it is not expanded: name the file and the agent reads it.

## The review

The review shows the objective and the drafted criteria. Accept them, cancel, or type what to change; with feedback the criteria are drafted again and shown again. Nothing enters the conversation until you accept. To use criteria you wrote yourself, use [`/rubric set`](rubric.md#interactive-sessions) instead.

In the TUI an unanswered review is cancelled after 5 minutes. With `--auto-mode` there is nobody to ask, so the draft is adopted as it is.

## While a goal is active

- Each time the agent stops, the grader checks the work against the criteria, exactly as for `/rubric set`: at most 3 passes per task, and grading waits for this thread's background work (see [Grading a run against a rubric](rubric.md)).
- When the criteria pass, the goal becomes `complete` and grading stops. The goal stays visible with `/goal` until you clear it or set a new one.
- When 3 passes do not get there, the goal stays `active` and the agent waits for you: send a message to continue, `/goal amend` to change the goal, or `/goal clear` to drop it.
- The agent can report that it needs you. The goal then shows as `blocked` with the reason, that turn is not graded, and your next message unblocks it.
- The status bar shows `goal: active`, `paused`, `blocked` or `done`.

## Together with `/rubric`

A goal's criteria take the place of the `/rubric set` checklist. `/rubric set` and `/rubric clear` therefore cancel the goal, with your next message; until then `/goal` and the status bar (`goal: ending`) say so. `/rubric next` still grades one task on its own and does not complete the goal. Accepting or resuming a goal drops `/rubric` changes that were still waiting for your next message, and accepting also drops a `/rubric next` checklist that was not used yet.

## Limits

- `/goal` is refused in channel sessions and when the session executes its runs on the `langgraph_server` gateway.
- On other surfaces (the WebUI, `EvoSci serve`, single-shot runs) a thread's goal is ignored: turns are not graded against it and the agent is not told about it. The goal is still there when you come back to the thread in the TUI or the interactive CLI.
- The criteria are drafted by the main model, which may list and read files in the workspace. A draft may make at most 12 model calls.
- An objective is limited to 8,000 characters; the objective and its criteria together to 12,000.
- A review that was left open (the app was closed, or the turn was stopped with Ctrl+C) is discarded by your next message; run `/goal` again. The same happens, without a message, when the review is answered from a surface that does not support goals.
- Model retries during drafting are not announced; a draft that fails is reported with its reason.
