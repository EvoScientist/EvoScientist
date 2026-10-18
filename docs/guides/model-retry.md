# Retrying transient model failures

A model call that fails for a transient reason — a cold-start 503, a gateway 502, a 429, a dropped connection, a timeout, or a response the provider ended with `finish_reason: "error"` — is retried in place. Completed tool calls are not run again, and the half answer a failed attempt streamed is removed before the retry's answer appears.

## What counts as transient

HTTP 408, 409, 425, 429 and 5xx; connection and timeout errors from the provider SDKs; provider errors reported inside a stream or inside a 200 body; a response that ends with `finish_reason: "error"`. Not retried: authentication and permission errors, context-window overflow, a response cut by the output limit, quota exhaustion, malformed requests, and anything else that would fail the same way again. With a `model_fallbacks` chain, one attempt is the primary model followed by the chain, and it is retried when at least one of those models failed transiently: a fallback entry that fails for good (a wrong key, an unknown model) does not switch the retry off, and when several models failed transiently the shortest of their waits is taken, so a rate-limited model can be called again before its `Retry-After` has passed. Every attempt starts from the primary again, so a model that keeps failing for a permanent reason is still called once per attempt. A fallback whose request is rejected as malformed ends the chain, and the attempt is still retried when an earlier model failed transiently; the same error from the primary is not retried. A context-window overflow is never retried, from the primary or a fallback: it goes straight to the conversation's summarization, which compacts the conversation and tries once more.

## Budget

| Setting | Default | Applies to |
|---|---|---|
| `model_retry_wait_seconds` | 60 | interactive sessions (TUI, CLI, `serve`) |
| `auto_mode_retry_wait_seconds` | 1800 | unattended runs (`--auto-mode`, `auto_mode: true`, scheduled tasks) |

A model call's budget starts when that call first fails: the waits and the retried attempts count against it, the time the first attempt took does not, so a long answer that drops near its end is still retried. The budget decides whether another attempt may start, not how long an attempt may run: an attempt that hangs is not cut by it and can run far past it. Waits start at 1 s and double up to a cap of 10 s (interactive) or 2 minutes (unattended); a model that sends a `Retry-After` header is given that wait instead when it fits in the budget left; otherwise the call gives up at once. `0` switches the retry off; a value that is not a whole number of seconds (`60s`) falls back to the default, with a warning in the log. When the budget runs out, the error the last attempt ended with is raised and the run ends with it; with a fallback chain that is the last model's error, for example a fallback's 401 after retry lines that named the primary's 503 (`-p` exits with code 1 and prints the resume hint).

```bash
EvoSci config set auto_mode_retry_wait_seconds 600
EvoSci config set model_retry_wait_seconds 0        # interactive sessions: no retry
```

A run waits about one budget in all. Once a model call has used its whole budget (its next backoff wait no longer fits in what is left; a Retry-After longer than what is left ends only that call), later model calls in the same run are tried once and not retried, and a call that is still retrying at that point stops at its next wait. When a sub-agent's call uses the budget up, its task fails and the main agent carries on, but the main agent's next call is not retried, so with the provider still down the run ends there instead of waiting a second budget. The cost: a run whose provider comes back only after more than one budget no longer gets through by having the main agent wait and dispatch the sub-agent again. The next message starts a new run, with a full budget.

A wrong `base_url`, or a local model server that is not running, looks like a dropped connection and is retried for the whole budget. Each retry line names the reason.

The tier travels with the run: a `-p --auto-mode` run gets the unattended budget whether the agent runs in-process or on the LangGraph server, while runs the server starts on its own (WebUI, background sub-agents, scheduled tasks) follow the server's configuration and keep a budget per model call; scheduled tasks always use the unattended budget.

## What you see

Each retry prints one line — `Model request failed (503: ... starting up); retrying in 8 s (attempt 2)` — in the TUI, the CLI and the log, and a `model_retry` event in `--output-format stream-json`; a sub-agent's or the acceptance grader's retry is prefixed with its name. The partial text of the failed attempt disappears from the screen (text that already scrolled off the terminal stays), from the final answer and from channel replies. With a `model_fallbacks` chain the same happens when the next model takes over from one that failed mid-answer.
