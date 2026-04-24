## Summary

This PR narrows the channel stop/resume fix to the core production paths:

- scope channel `/stop` and `/cancel` to the current chat session instead of a process-wide stop flag
- allow stop to cancel both queued work and active channel responses
- keep `/resume` in channel mode useful by replaying recent saved history back to the user

Compared with the previous version, this change drops the extra global `/stop` slash command and trims some channel-only presentation/test surface so the PR stays focused on the channel bugfix.

## What Changed

- track queued/active channel requests per `channel + chat_id`
- add per-request stream cancel scopes in the shared streaming display
- wire scoped cancellation through Rich CLI, Textual TUI, and serve mode channel handlers
- keep HITL and ask_user waits stoppable from the channel fast-path
- replay the recent saved conversation after channel `/resume`

## Simplifications In This Revision

- removed the standalone `StopCommand` implementation
- kept channel `/resume` output minimal instead of expanding more session metadata in-channel
- trimmed duplicate tests while preserving coverage for scoped cancellation, queued cancellation, HITL release, serve-mode handling, and resume history replay

## Testing

```bash
uv run pytest tests/test_stream_cancel.py tests/test_bus_integration.py tests/test_channel_command_ui.py tests/test_resume_command.py tests/test_serve_agent_holder.py
```
