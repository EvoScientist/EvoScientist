# `stream-json` output protocol

`EvoSci --output-format stream-json` runs a single-shot (`-p`) session and emits
EvoScientist's **native event stream** as line-delimited JSON (JSONL) on stdout —
one self-describing JSON object per line. This is the integration surface for
programmatic clients that drive EvoScientist headlessly (for example, an agent
runtime that assigns work and renders live progress).

```bash
EvoSci -p "Summarize the attention mechanism into notes.md" \
       --output-format stream-json \
       --auto-mode \
       --workdir /path/to/task
```

## Stream contract

- **stdout is pure JSONL.** Every line is one complete JSON object. Parse it with
  a line reader + `json.loads` per line. Nothing else is written to stdout.
- **stderr carries everything human** — status lines ("Loading agent…"),
  separators, the resume hint, and error panels. A consumer should treat stderr
  as logs, not protocol.
- **Each object has a `type` field** used for dispatch. A consumer that does not
  recognize a `type` should **ignore that line** rather than fail — new event
  types may be added over time, and the protocol is forward-compatible by design.
- **The stream ends with a `done` event** carrying the final response text. On an
  unhandled failure an `error` event is emitted instead/also.
- **Unattended by default.** `stream-json` is headless, so `--auto-mode` is
  enabled automatically: approval and `ask_user` gates are auto-handled and the
  run proceeds straight to its `done` event. Pass `--no-auto-mode` to opt out —
  a human-in-the-loop `interrupt` / `ask_user` is then emitted as a normal event
  and the single-shot run ends right after it (`… → interrupt → done → EOF`). It
  does **not** block waiting for input, but it also stops before finishing the
  task; answering the event requires re-invoking with `--resume` (experimental).

## Event types

Each line is a self-contained JSON object with a `type` field. Most fields are
scalars, but some events carry nested payloads (e.g. `args`, `action_requests`,
`questions`) — parse each line as a full object, not a flat key/value map. The
fields beyond `type` are listed below.

| `type`                 | Fields                                                      | Meaning |
|------------------------|------------------------------------------------------------|---------|
| `thinking`             | `content`, `id`                                            | Model reasoning text |
| `text`                 | `content`                                                 | Assistant output text |
| `tool_call`            | `name`, `args`, `id`                                      | Tool invocation |
| `tool_result`          | `name`, `content`, `success`, `id`                       | Tool result (`id` matches the `tool_call`) |
| `subagent_start`       | `name`, `description`                                     | Sub-agent delegation begins |
| `subagent_tool_call`   | `subagent`, `name`, `args`, `id`                          | Tool call inside a sub-agent |
| `subagent_tool_result` | `subagent`, `name`, `content`, `success`, `id`            | Tool result inside a sub-agent |
| `subagent_text`        | `subagent`, `content`, `instance_id`                      | Text from a sub-agent |
| `subagent_end`         | `name`                                                    | Sub-agent delegation completes |
| `tool_selection`       | `tools`                                                   | Tool-selector middleware picked tools |
| `summarization_start`  | —                                                         | Context summarization begins |
| `summarization`        | `content`                                                 | Context summarization output |
| `usage_stats`          | `input_tokens`, `output_tokens`                           | Token usage |
| `interrupt`            | `interrupt_id`, `action_requests`, `review_configs`      | HITL approval interrupt |
| `ask_user`             | `interrupt_id`, `questions`, `tool_call_id`              | Agent-initiated clarifying question |
| `error`                | `message`                                                | Error during the run |
| `done`                 | `content`, `response`                                     | Final response; end of stream |

## Example transcript

```jsonl
{"type": "thinking", "content": "I should write the notes file.", "id": 0}
{"type": "tool_call", "name": "write_file", "args": {"path": "notes.md", "content": "..."}, "id": "call_1"}
{"type": "tool_result", "name": "write_file", "content": "wrote 412 bytes", "success": true, "id": "call_1"}
{"type": "text", "content": "Done — notes.md now summarizes attention."}
{"type": "usage_stats", "input_tokens": 5123, "output_tokens": 388}
{"type": "done", "content": "Done — notes.md now summarizes attention.", "response": "Done — notes.md now summarizes attention."}
```

## Notes for client implementers

- Read stdout line by line; do not assume a single JSON document.
- Accumulate `text` events for the running assistant message; the terminal
  `done.response` is the authoritative final text.
- `tool_call` / `tool_result` correlate by `id`.
- Token usage may arrive across multiple `usage_stats` events; sum them.
- Treat unknown `type` values and unknown fields as non-fatal.
