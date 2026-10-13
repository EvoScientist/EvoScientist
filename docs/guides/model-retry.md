# Retrying transient model failures

A model call that fails for a transient reason — a cold-start 503, a gateway 502, a 429, a dropped connection, a timeout, or a response the provider ended with `finish_reason: "error"` — is retried in place. Completed tool calls are not run again, and the half answer a failed attempt streamed is removed before the retry's answer appears.

## What counts as transient

HTTP 408, 409, 425, 429 and 5xx; connection and timeout errors from the provider SDKs; provider errors reported inside a stream or inside a 200 body; a response that ends with `finish_reason: "error"`. Not retried: authentication and permission errors, context-window overflow, a response cut by the output limit, quota exhaustion, malformed requests, and anything else that would fail the same way again. With a `model_fallbacks` chain, one attempt is the primary model followed by the chain; the wait happens only when every model failed transiently.

## Budget

| Setting | Default | Applies to |
|---|---|---|
| `model_retry_wait_seconds` | 60 | interactive sessions (TUI, CLI, `serve`) |
| `auto_mode_retry_wait_seconds` | 1800 | unattended runs (`--auto-mode`, `auto_mode: true`, scheduled tasks) |

The budget is a deadline per model call: waits start at 1 s and double up to a cap of 10 s (interactive) or 2 minutes (unattended), a `Retry-After` header is honoured, and the time a failing attempt itself takes counts too. `0` switches the retry off; a value that is not a whole number of seconds (`60s`) falls back to the default, with a warning in the log. When the budget runs out the last error is raised and the run ends with it (`-p` exits with code 1 and prints the resume hint).

```bash
EvoSci config set auto_mode_retry_wait_seconds 600
EvoSci config set model_retry_wait_seconds 0        # interactive sessions: no retry
```

The tier travels with the run: a `-p --auto-mode` run gets the unattended budget whether the agent runs in-process or on the LangGraph server, while runs the server starts on its own (WebUI, background sub-agents, scheduled tasks) follow the server's configuration; scheduled tasks always use the unattended budget.

## What you see

Each retry prints one line — `Model request failed (503: ... starting up); retrying in 8 s (attempt 2)` — in the TUI, the CLI and the log, and a `model_retry` event in `--output-format stream-json`; a sub-agent's or the acceptance grader's retry is prefixed with its name. The partial text of the failed attempt disappears from the screen (text that already scrolled off the terminal stays), from the final answer and from channel replies. With a `model_fallbacks` chain the same happens when the next model takes over from one that failed mid-answer.
