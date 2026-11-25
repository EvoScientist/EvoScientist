# `stream-json` Output Protocol

`--output-format stream-json` turns a single-shot (`-p`) EvoScientist run into a clean, line-delimited JSON (JSONL) event stream on `stdout`—designed for scripts, SDKs, and external agent runtimes.

```bash
EvoSci -p "Summarize the attention mechanism into notes.md" \
  --output-format stream-json \
  --auto-mode \
  --workdir /path/to/task
```

## Stream Contract

- **`stdout` is pure JSONL:** One complete JSON object per line—parse line by line with `json.loads`. Nothing human-formatted is ever written to `stdout`.
- **`stderr` is human logs:** Status messages, resume hints, and error panels go to `stderr`. Treat `stderr` as diagnostics, not protocol.
- **Forward-compatible dispatch:** Every object has a `type` field. Ignore unrecognized `type` values and unknown fields so new event types never break your parser.
- **Unattended by default:** `stream-json` implies `--auto-mode` so runs proceed straight to a terminal `done` (or `error`) event. Pass `--no-auto-mode` to emit `interrupt` / `ask_user` events instead—note that the process then ends immediately (`… → interrupt → done → EOF`) rather than blocking on `stdin` (resuming via `--resume` is experimental).

## Event Types

| `type` | Fields | Meaning |
|---|---|---|
| `thinking` | `content`, `id` | Model reasoning text |
| `text` | `content` | Assistant output chunk |
| `tool_call` | `name`, `args`, `id` | Tool invocation |
| `tool_result` | `name`, `content`, `success`, `id` | Tool output (`id` matches `tool_call`; `content` truncated to ~2,000 chars) |
| `subagent_start` | `name`, `description` | Sub-agent delegation began |
| `subagent_tool_call` | `subagent`, `name`, `args`, `id` | Tool call inside a sub-agent |
| `subagent_tool_result` | `subagent`, `name`, `content`, `success`, `id` | Tool result inside a sub-agent |
| `subagent_text` | `subagent`, `content`, `instance_id` | Text emitted by a sub-agent |
| `subagent_end` | `name`, `paused` *(when true)* | Sub-agent finished (or paused for approval/question) |
| `tool_selection` | `tools` | Adaptive tool selector chose visible tools |
| `summarization_start` | — | Context compaction started |
| `summarization` | `content` | Context compaction summary |
| `usage_stats` | `input_tokens`, `output_tokens` | Token counts (may appear multiple times; sum them) |
| `interrupt` | `interrupt_id`, `action_requests`, `review_configs` | HITL approval interrupt (`--no-auto-mode`) |
| `ask_user` | `interrupt_id`, `questions`, `tool_call_id` | Clarifying question (`--no-auto-mode`) |
| `rubric_evaluation_start` | `iteration` | Rubric grading pass began (`--rubric`) |
| `rubric_evaluation` | `iteration`, `result`, `explanation`, `criteria`, `unverified` | Rubric verdict (`satisfied`, `needs_revision`, `max_iterations_reached`, `failed`, or `grader_error`) |
| `rubric_evaluation_deferred` | `tasks`, `processes` | Interactive `/rubric` or `/goal` only (never emitted by `-p`): grading deferred until background work finishes |
| `model_retry` | `attempt`, `wait_seconds`, `reason`, `budget_left_seconds`, `cause`, `instance_id`, `subagent`, `discarded` | Transient failure scheduled for retry (`cause`: `exception` or `finish_reason`) |
| `model_output_discarded` | `instance_id`, `discarded` | Partial output dropped when a fallback model took over without a retry notice |
| `error` | `message` | Unhandled run error |
| `done` | `content`, `response`, `last_pass_response` | End of stream; `response` is the authoritative final text |

### Field Notes

- **`tool_result.success`:** `false` when the tool errored, `execute` exited non-zero (even `grep` with no matches), `code_interpreter` raised, or output (other than `write_todos`) begins with `[FAILED]` / `[TOOL ERROR]` or shows an error in its first 3 lines.
- **`discarded` on retry/fallback events:** Object `{text_chars, thinking_chars, tool_call_ids}` telling UIs what streamed output from the failed attempt to roll back. `subagent` is empty for the main agent, or `rubric_grader` for the grader.
- **`done.last_pass_response`:** Present on `--rubric` runs—contains only the text written after the final revision feedback (while `response` concatenates all passes separated by blank lines).

## Example Transcript

```jsonl
{"type": "thinking", "content": "I should write the notes file.", "id": 0}
{"type": "tool_call", "name": "write_file", "args": {"file_path": "notes.md", "content": "..."}, "id": "call_1"}
{"type": "tool_result", "name": "write_file", "content": "wrote 412 bytes", "success": true, "id": "call_1"}
{"type": "text", "content": "Done — notes.md now summarizes attention."}
{"type": "usage_stats", "input_tokens": 5123, "output_tokens": 388}
{"type": "done", "content": "Done — notes.md now summarizes attention.", "response": "Done — notes.md now summarizes attention."}
```

## Implementer Checklist

- Read `stdout` line by line (never buffer as a single JSON array).
- Parse each line as a full JSON object—fields like `args`, `action_requests`, and `questions` are nested objects/arrays.
- Correlate `tool_call` and `tool_result` via `id`; accumulate `text` for live display, and treat `done.response` as final.
- Sum `usage_stats` across the run, and ignore unknown event `type`s safely.
