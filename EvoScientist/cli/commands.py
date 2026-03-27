"""Typer command registrations — onboard, config, mcp, main callback."""

import logging
import os
import queue
import re
from datetime import datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated, Any

import typer  # type: ignore[import-untyped]
from rich.markup import escape
from rich.table import Table

from ..paths import ensure_dirs, set_workspace_root
from ..stream.display import console
from ._app import app, channel_app, config_app, mcp_app
from ._constants import build_metadata
from .agent import (
    _create_session_workspace,
    _deduplicate_run_name,
    _load_agent,
    _shorten_path,
)
from .channel import (
    ChannelMessage,
    _channels_stop,
    _message_queue,
    _set_channel_response,
    _start_channels_bus_mode,
    channel_ask_user_prompt,
    channel_hitl_prompt,
)
from .interactive import cmd_interactive, cmd_run
from .mcp_ui import (
    _mcp_add_server_from_kwargs,
    _mcp_edit_server_fields,
    _mcp_list_servers,
    _mcp_remove_server,
    _show_mcp_config,
)
from .tui_runtime import run_streaming

# =============================================================================
# Onboard command
# =============================================================================


@app.command()
def onboard(
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip API key validation during setup"
    ),
):
    """Interactive setup wizard for EvoScientist

    Guides you through configuring API keys, model selection,
    workspace settings, and agent parameters.
    """
    from ..config import run_onboard

    run_onboard(skip_validation=skip_validation)


# =============================================================================
# Channel setup command
# =============================================================================


@channel_app.command("setup")
def channel_setup():
    """Interactive channel configuration wizard.

    Guides you through selecting and configuring messaging channels
    (Telegram, Discord, or iMessage).
    """
    import asyncio

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    from ..config import load_config, save_config
    from ..config.onboard import _step_channels

    config = load_config()
    updates = _step_channels(config)
    if updates:
        for key, value in updates.items():
            setattr(config, key, value)
        save_config(config)
        console.print("[green]Channel configuration saved.[/green]")
    else:
        console.print("[dim]No changes made.[/dim]")


# =============================================================================
# Compact helper
# =============================================================================


class CompactResult:
    """Structured result from compact_conversation.

    Attributes:
        status: "noop" (nothing to compact), "ok" (compacted), or "error".
        message: Short human-readable message (used as fallback / TUI text).
        messages_compacted: Number of messages summarized (0 for noop/error).
        messages_kept: Number of messages unchanged.
        tokens_before: Total tokens before compaction.
        tokens_after: Total tokens after compaction.
        tokens_summarized: Tokens in the summarized portion (before).
        tokens_summary: Tokens in the summary message (after).
        pct_decrease: Percentage decrease.
    """

    __slots__ = (
        "message",
        "messages_compacted",
        "messages_kept",
        "pct_decrease",
        "status",
        "tokens_after",
        "tokens_before",
        "tokens_summarized",
        "tokens_summary",
    )

    def __init__(
        self,
        status: str,
        message: str,
        *,
        messages_compacted: int = 0,
        messages_kept: int = 0,
        tokens_before: int = 0,
        tokens_after: int = 0,
        tokens_summarized: int = 0,
        tokens_summary: int = 0,
        pct_decrease: int = 0,
    ):
        self.status = status
        self.message = message
        self.messages_compacted = messages_compacted
        self.messages_kept = messages_kept
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.tokens_summarized = tokens_summarized
        self.tokens_summary = tokens_summary
        self.pct_decrease = pct_decrease

    def __str__(self) -> str:
        return self.message


def render_compact_result(result: CompactResult):  # -> rich.text.Text
    """Render a CompactResult as styled Rich Text.

    Uses the same visual language as the token usage display:
    cyan for numbers, green for savings, dim for labels.
    """
    from rich.text import Text

    output = Text()

    if result.status == "noop":
        output.append("○ ", style="dim")
        output.append("Nothing to compact", style="dim")
        if result.tokens_before > 0:
            output.append(" — conversation is ~", style="dim")
            output.append(f"{result.tokens_before:,}", style="cyan")
            output.append(" tokens, within retention budget", style="dim")
        elif result.message:
            # Extract reason from message (e.g. "no messages")
            output.append(
                f" — {result.message.split('—')[-1].strip()}"
                if "—" in result.message
                else "",
                style="dim",
            )
        return output

    if result.status == "error":
        output.append("✗ ", style="red")
        output.append(result.message, style="red")
        return output

    # status == "ok"
    output.append("✓ ", style="green")
    output.append("Compacted ", style="dim")
    output.append(f"{result.messages_compacted}", style="bold")
    output.append(" messages", style="dim")
    output.append("  [", style="dim")
    output.append(f"{result.tokens_before:,}", style="cyan")
    output.append(" → ", style="dim")
    output.append(f"{result.tokens_after:,}", style="green")
    output.append(" tokens", style="dim")
    output.append(f"  ↓{result.pct_decrease}%", style="green bold")
    output.append("]", style="dim")

    # Second line: detail breakdown
    output.append("\n  ", style="")
    output.append("Summarized: ", style="dim")
    output.append(f"{result.tokens_summarized:,}", style="cyan")
    output.append(" → ", style="dim")
    output.append(f"{result.tokens_summary:,}", style="green")
    output.append("  │  ", style="dim")
    output.append("Kept: ", style="dim")
    output.append(f"{result.messages_kept}", style="cyan")
    output.append(" messages unchanged", style="dim")

    return output


async def compact_conversation(agent: Any, thread_id: str | None) -> CompactResult:
    """Compact the conversation by summarizing old messages.

    Reads the agent's checkpointed state, creates a temporary
    ``SummarizationMiddleware``, generates a summary, and writes
    the compacted state back via ``aupdate_state``.

    Returns a structured ``CompactResult``.
    """
    if not agent or not thread_id:
        return CompactResult("noop", "Nothing to compact — start a conversation first.")

    from langchain_core.messages.utils import count_tokens_approximately

    config = {"configurable": {"thread_id": thread_id}}

    try:
        state_snapshot = await agent.aget_state(config)
    except Exception as exc:
        return CompactResult("error", f"Failed to read state: {exc}")

    messages = state_snapshot.values.get("messages", [])
    if not messages:
        return CompactResult(
            "noop", "Nothing to compact — no messages in conversation."
        )

    from deepagents.middleware.summarization import (
        SummarizationEvent,
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    from ..EvoScientist import _ensure_chat_model, _get_default_backend

    try:
        model = _ensure_chat_model()
    except Exception as exc:
        return CompactResult(
            "error", f"Compaction requires a working model configuration: {exc}"
        )

    backend = _get_default_backend()

    defaults = compute_summarization_defaults(model)
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
    )

    # Rebuild effective message list accounting for prior compaction
    event = state_snapshot.values.get("_summarization_event")
    effective = middleware._apply_event_to_messages(messages, event)

    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff == 0:
        conv_tokens = count_tokens_approximately(effective)
        return CompactResult(
            "noop",
            f"Nothing to compact — conversation (~{conv_tokens:,} tokens) "
            f"is within the retention budget.",
            tokens_before=conv_tokens,
        )

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)

    tokens_summarized = count_tokens_approximately(to_summarize)
    tokens_kept = count_tokens_approximately(to_keep)
    tokens_before = tokens_summarized + tokens_kept

    # Skip if savings would be negligible — compacting ≤2 messages with
    # <2% of total tokens prevents the infinite 1-message-at-a-time loop
    # that occurs when the conversation sits just above the keep budget.
    _MIN_COMPACT_MESSAGES = 3
    _MIN_COMPACT_TOKEN_FRACTION = 0.02
    if (
        len(to_summarize) < _MIN_COMPACT_MESSAGES
        and tokens_summarized < tokens_before * _MIN_COMPACT_TOKEN_FRACTION
    ):
        return CompactResult(
            "noop",
            f"Nothing to compact — only {len(to_summarize)} message(s) "
            f"({tokens_summarized:,} tokens) would be summarized, "
            f"not worth the overhead.",
            tokens_before=tokens_before,
        )

    # Generate summary (LLM call)
    summary = await middleware._acreate_summary(to_summarize)

    # Inject thread_id into LangGraph contextvar so _get_thread_id() finds it
    # (compact runs outside a runnable context, so get_config() would fail
    # and the middleware would generate a random "session_xxx" filename instead
    # of reusing the real thread_id).
    from langgraph.config import var_child_runnable_config

    _token = var_child_runnable_config.set(config)

    # Offload old messages to backend
    file_path: str | None = None
    try:
        file_path = await middleware._aoffload_to_backend(backend, to_summarize)
    except Exception:
        pass  # non-fatal — proceed without offloaded history
    finally:
        var_child_runnable_config.reset(_token)

    summary_msg = middleware._build_new_messages_with_path(summary, file_path)[0]

    # Compute token savings
    tokens_summary = count_tokens_approximately([summary_msg])
    tokens_after = tokens_summary + tokens_kept
    pct = (
        round((tokens_before - tokens_after) / tokens_before * 100)
        if tokens_before > 0
        else 0
    )

    # Append savings note to summary message for model awareness
    savings_note = (
        f"\n\n{len(to_summarize)} messages were compacted "
        f"({tokens_summarized:,} → {tokens_summary:,} tokens). "
        f"Total context: {tokens_before:,} → {tokens_after:,} tokens "
        f"({pct}% decrease), "
        f"{len(to_keep)} messages unchanged."
    )
    summary_msg.content += savings_note

    state_cutoff = middleware._compute_state_cutoff(event, cutoff)

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
    }

    await agent.aupdate_state(config, {"_summarization_event": new_event})

    return CompactResult(
        "ok",
        f"Compacted {len(to_summarize)} messages "
        f"({tokens_before:,} → {tokens_after:,} tokens, {pct}% decrease)",
        messages_compacted=len(to_summarize),
        messages_kept=len(to_keep),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_summarized=tokens_summarized,
        tokens_summary=tokens_summary,
        pct_decrease=pct,
    )


# =============================================================================
# Serve helpers
# =============================================================================

_serve_logger = logging.getLogger(__name__)


def _serve_process_message(
    msg: ChannelMessage,
    *,
    agent: Any,
    thread_id: str,
    model: str | None,
    workspace_dir: str,
    show_thinking: bool,
) -> None:
    """Process a single channel message in headless serve mode.

    Headless equivalent of interactive.py's ``_process_channel_message``.
    No CLI prompt manipulation — just log lines for monitoring.
    """
    import asyncio

    from .channel import _bus_loop

    console.print(
        f"[dim][{msg.channel_type}] {msg.sender}: {escape(msg.content[:80])}[/dim]"
    )

    # -- channel callback helpers (same pattern as interactive.py) --

    def _send_to_channel(coro, label: str, timeout: int = 15) -> None:
        loop = _bus_loop
        if not loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)
        except Exception as e:
            _serve_logger.debug(f"{label} send failed: {e}")

    def _send_thinking(thinking: str) -> None:
        ch = msg.channel_ref
        if ch and ch.send_thinking:
            _send_to_channel(
                ch.send_thinking_message(
                    sender=msg.chat_id,
                    thinking=thinking,
                    metadata=msg.metadata,
                ),
                "Thinking",
            )

    def _send_todo(items: list[dict]) -> None:
        from ..channels.consumer import _format_todo_list

        if msg.channel_ref:
            _send_to_channel(
                msg.channel_ref.send_todo_message(
                    sender=msg.chat_id,
                    content=_format_todo_list(items),
                    metadata=msg.metadata,
                ),
                "Todo",
            )

    def _send_media(file_path: str) -> None:
        if msg.channel_ref:
            _send_to_channel(
                msg.channel_ref.send_media(
                    recipient=msg.chat_id,
                    file_path=file_path,
                    metadata=msg.metadata,
                ),
                "Media",
                timeout=30,
            )

    def _hitl_prompt(action_requests: list) -> list[dict] | None:
        return channel_hitl_prompt(action_requests, msg)

    def _ask_user_prompt(ask_user_data: dict) -> dict:
        return channel_ask_user_prompt(ask_user_data, msg)

    meta = build_metadata(workspace_dir, model)
    try:
        response = run_streaming(
            ui_backend="cli",
            agent=agent,
            message=msg.content,
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=True,
            metadata=meta,
            on_thinking=_send_thinking,
            on_todo=_send_todo,
            on_file_write=_send_media,
            hitl_prompt_fn=_hitl_prompt,
            ask_user_prompt_fn=_ask_user_prompt,
        )
    except Exception as e:
        response = f"Error: {e}"
        console.print(f"[red]Serve error: {e}[/red]")

    _set_channel_response(msg.msg_id, response)
    console.print(f"[dim][{msg.channel_type}] Replied to {msg.sender}[/dim]")


# =============================================================================
# Serve command (headless mode)
# =============================================================================


@app.command()
def serve(
    no_thinking: bool = typer.Option(
        False, "--no-thinking", help="Disable thinking relay to channels"
    ),
    workdir: str | None = typer.Option(
        None, "--workdir", help="Override workspace directory"
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve all tool executions without prompting",
    ),
    ask_user: bool = typer.Option(
        False,
        "--ask-user",
        help="Enable agent to ask clarifying questions about your research preferences",
    ),
):
    """Run EvoScientist in headless mode -- channels only, no interactive prompt.

    Starts all configured channels and processes messages via the agent.
    Press Ctrl+C to shut down.
    """
    import nest_asyncio  # type: ignore[import-untyped]

    nest_asyncio.apply()

    from ..config import apply_config_to_env, get_effective_config

    cli_overrides = {}
    if auto_approve:
        cli_overrides["auto_approve"] = True
    if ask_user:
        cli_overrides["enable_ask_user"] = True
    config = get_effective_config(cli_overrides)
    apply_config_to_env(config)

    # Auto-start ccproxy if any provider uses OAuth mode
    _ccproxy_proc_serve = None
    if config.anthropic_auth_mode == "oauth" or config.openai_auth_mode == "oauth":
        try:
            from ..ccproxy_manager import maybe_start_ccproxy, stop_ccproxy

            _ccproxy_proc_serve = maybe_start_ccproxy(config)
            if _ccproxy_proc_serve:
                import atexit

                atexit.register(stop_ccproxy, _ccproxy_proc_serve)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    if not config.channel_enabled:
        console.print("[red]No channels configured.[/red]")
        console.print("[dim]Run [bold]evosci channel setup[/bold] first.[/dim]")
        raise typer.Exit(1)

    effective_channel_thinking = config.channel_send_thinking and (not no_thinking)
    if workdir:
        ws = os.path.abspath(os.path.expanduser(workdir))
    elif config.default_workdir:
        ws = os.path.abspath(os.path.expanduser(config.default_workdir))
    else:
        ws = os.getcwd()
    os.makedirs(ws, exist_ok=True)
    set_workspace_root(ws)
    ensure_dirs()

    console.print("[dim]Loading agent...[/dim]")
    agent = _load_agent(workspace_dir=ws, config=config)
    from ..sessions import generate_thread_id

    tid = generate_thread_id()

    _start_channels_bus_mode(
        config,
        agent,
        tid,
        send_thinking=effective_channel_thinking,
    )
    console.print("[green]Serve mode started (bus mode).[/green]")

    console.print(f"[dim]Thread: {tid}[/dim]")
    console.print(f"[dim]Workspace: {_shorten_path(ws)}[/dim]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    try:
        while True:
            try:
                msg = _message_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            _serve_process_message(
                msg,
                agent=agent,
                thread_id=tid,
                model=config.model,
                workspace_dir=ws,
                show_thinking=effective_channel_thinking,
            )
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
    finally:
        _channels_stop()
        console.print("[dim]Stopped.[/dim]")


# =============================================================================
# Config commands
# =============================================================================


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands"""
    if ctx.invoked_subcommand is None:
        config_list()


@config_app.command("list")
def config_list():
    """List all configuration values"""
    from ..config import get_config_path, list_config

    config_data = list_config()

    table = Table(title="EvoScientist Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Mask API keys
    def format_value(key: str, value: Any) -> str:
        if "api_key" in key and value:
            return "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
        if value == "":
            return "[dim](not set)[/dim]"
        return str(value)

    for key, value in config_data.items():
        table.add_row(key, format_value(key, value))

    console.print(table)
    console.print(f"\n[dim]Config file: {get_config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Configuration key to get")):
    """Get a single configuration value"""
    from ..config import get_config_value

    value = get_config_value(key)
    if value is None:
        console.print(f"[red]Unknown key: {key}[/red]")
        raise typer.Exit(1)

    # Mask API keys
    if "api_key" in key and value:
        display_value = "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
    elif value == "":
        display_value = "(not set)"
    else:
        display_value = str(value)

    console.print(f"[cyan]{key}[/cyan]: {display_value}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a single configuration value"""
    from ..config import set_config_value

    if set_config_value(key, value):
        console.print(f"[green]Set {escape(key)}[/green]")
    else:
        console.print(f"[red]Invalid key: {escape(key)}[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset configuration to defaults"""
    from ..config import get_config_path, reset_config

    config_path = get_config_path()

    if not config_path.exists():
        console.print("[yellow]No config file to reset.[/yellow]")
        return

    if not yes:
        confirm = typer.confirm("Reset configuration to defaults?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    reset_config()
    console.print("[green]Configuration reset to defaults.[/green]")


@config_app.command("path")
def config_path():
    """Show the configuration file path"""
    from ..config import get_config_path

    path = get_config_path()
    exists = path.exists()
    status = "[green]exists[/green]" if exists else "[dim]not created yet[/dim]"
    console.print(f"{path} ({status})")


# =============================================================================
# MCP commands
# =============================================================================


@mcp_app.callback(invoke_without_command=True)
def mcp_callback(ctx: typer.Context):
    """MCP server management commands"""
    if ctx.invoked_subcommand is None:
        mcp_list()


@mcp_app.command("list")
def mcp_list():
    """List configured MCP servers"""
    _mcp_list_servers()


@mcp_app.command("config")
def mcp_config(
    name: str | None = typer.Argument(None, help="Server name (omit to show all)"),
):
    """Show detailed configuration for MCP servers

    \b
    Examples:
      evosci mcp config             # Show all servers in detail
      evosci mcp config filesystem  # Show one server
    """
    status = _show_mcp_config(name or "", show_blank_line=False)
    if status == "empty":
        console.print(
            "[dim]Add one with:[/dim] EvoSci mcp add <name> <transport> <command-or-url> [args...]"
        )
        return
    if status == "missing":
        raise typer.Exit(1)


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name")],
    target: Annotated[str, typer.Argument(help="Command (stdio) or URL (http/sse)")],
    args: Annotated[
        list[str] | None, typer.Argument(help="Extra args for stdio command")
    ] = None,
    transport: Annotated[
        str | None,
        typer.Option("--transport", "-T", help="Transport type (default: auto-detect)"),
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            help="Comma-separated tool allowlist (supports wildcards: *_exa, read_*)",
        ),
    ] = None,
    expose_to: Annotated[
        str | None,
        typer.Option("--expose-to", "-e", help="Comma-separated target agents"),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option("--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    ] = None,
    env_ref: Annotated[
        list[str] | None,
        typer.Option(
            "--env-ref", help="Env var name as ${NAME} runtime ref (repeatable)"
        ),
    ] = None,
):
    """Add an MCP server to user config

    \b
    Transport is auto-detected: URLs default to http, commands default to stdio.

    \b
    Examples:
      evosci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
      evosci mcp add docs-langchain https://docs.langchain.com/mcp
      evosci mcp add my-sse https://example.com/sse --transport sse -e research-agent
      evosci mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search
    """
    from ..mcp import build_mcp_add_kwargs

    # Merge env and env_ref into a single dict
    env_dict: dict[str, str] = {}
    for e in env or []:
        if "=" in e:
            k, v = e.split("=", 1)
            env_dict[k.strip()] = v.strip()
    for ref in env_ref or []:
        env_dict[ref] = "${" + ref + "}"

    kwargs = build_mcp_add_kwargs(
        name=name,
        target=target,
        extra_args=list(args) if args else None,
        transport=transport,
        tools=[t.strip() for t in tools.split(",") if t.strip()] if tools else None,
        expose_to=[a.strip() for a in expose_to.split(",") if a.strip()]
        if expose_to
        else None,
        headers={
            k.strip(): v.strip()
            for h in (header or [])
            for k, v in [h.split(":", 1)]
            if ":" in h
        }
        or None,
        env=env_dict or None,
    )

    if not _mcp_add_server_from_kwargs(kwargs, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("edit")
def mcp_edit(
    name: Annotated[str, typer.Argument(help="Server name to edit")],
    transport: Annotated[
        str | None, typer.Option("--transport", help="New transport type")
    ] = None,
    command: Annotated[
        str | None, typer.Option("--command", help="New command (stdio)")
    ] = None,
    url: Annotated[
        str | None, typer.Option("--url", help="New URL (http/sse/websocket)")
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            help="Comma-separated tool allowlist, supports wildcards ('none' to clear)",
        ),
    ] = None,
    expose_to: Annotated[
        str | None,
        typer.Option(
            "--expose-to",
            "-e",
            help="Comma-separated target agents ('none' to clear)",
        ),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option("--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    ] = None,
):
    """Edit an existing MCP server in user config

    \b
    Examples:
      evosci mcp edit filesystem --expose-to main,code-agent
      evosci mcp edit filesystem -t read_file,write_file
      evosci mcp edit my-api --url http://new-host:9090/mcp
      evosci mcp edit my-api --tools none
    """
    from ..mcp import build_mcp_edit_fields

    fields = build_mcp_edit_fields(
        transport=transport,
        command=command,
        url=url,
        tools=tools,
        expose_to=expose_to,
        headers=header,
        env=env,
    )

    if not _mcp_edit_server_fields(name, fields, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name to remove"),
):
    """Remove an MCP server from user config"""
    if not _mcp_remove_server(name, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("install")
def mcp_install(
    source: Annotated[
        str | None, typer.Argument(help="Server name or tag filter")
    ] = None,
):
    """Browse and install MCP servers from the registry and marketplace

    \b
    Examples:
      evosci mcp install                       # Interactive browser
      evosci mcp install search                # Filter by 'search' tag
      evosci mcp install sequential-thinking   # Install by name
    """
    from .mcp_install_cmd import _cmd_install_mcp

    _cmd_install_mcp(source or "")


# =============================================================================
# Main callback (default behavior)
# =============================================================================


def _version_callback(value: bool):
    if value:
        typer.echo(f"EvoScientist {_pkg_version('EvoScientist')}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None,
        "-V",
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    mode: str | None = typer.Option(
        None,
        "-m",
        "--mode",
        help="Workspace mode: 'daemon' (persistent, default) or 'run' (isolated per-session)",
    ),
    name: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="Name for this run (used as directory name instead of timestamp; requires --mode run)",
    ),
    prompt: str | None = typer.Option(
        None, "-p", "--prompt", help="Query to execute (single-shot mode)"
    ),
    thread_id: str | None = typer.Option(
        None, "--thread-id", help="Thread ID for conversation persistence"
    ),
    workdir: str | None = typer.Option(
        None, "--workdir", help="Override workspace directory for this session"
    ),
    use_cwd: bool = typer.Option(
        False, "--use-cwd", help="Use current working directory as workspace"
    ),
    no_thinking: bool = typer.Option(
        False, "--no-thinking", help="Disable thinking display"
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve all tool executions without prompting",
    ),
    ask_user: bool = typer.Option(
        False,
        "--ask-user",
        help="Enable agent to ask clarifying questions about your research preferences",
    ),
    auth_mode: str | None = typer.Option(
        None,
        "--auth-mode",
        help="Auth mode for Anthropic/OpenAI: api_key (default) or oauth (ccproxy).",
    ),
    ui: str | None = typer.Option(
        None,
        "--ui",
        help="UI backend: tui (default) or cli.",
    ),
):
    """EvoScientist Agent - AI-powered research & code execution CLI"""
    # If a subcommand was invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    # Load and apply configuration
    from ..config import apply_config_to_env, get_effective_config

    # Build CLI overrides dict
    cli_overrides = {}
    if mode:
        cli_overrides["default_mode"] = mode
    if workdir:
        cli_overrides["default_workdir"] = workdir
    if no_thinking:
        cli_overrides["show_thinking"] = False
    if ui:
        cli_overrides["ui_backend"] = ui
    if auto_approve:
        cli_overrides["auto_approve"] = True
    if ask_user:
        cli_overrides["enable_ask_user"] = True
    if auth_mode:
        if auth_mode not in ("api_key", "oauth"):
            raise typer.BadParameter("--auth-mode must be 'api_key' or 'oauth'")
        cli_overrides["anthropic_auth_mode"] = auth_mode
        cli_overrides["openai_auth_mode"] = auth_mode

    config = get_effective_config(cli_overrides)
    apply_config_to_env(config)

    # Auto-start ccproxy if any provider uses OAuth mode
    _ccproxy_proc = None
    if config.anthropic_auth_mode == "oauth" or config.openai_auth_mode == "oauth":
        try:
            from ..ccproxy_manager import maybe_start_ccproxy, stop_ccproxy

            _ccproxy_proc = maybe_start_ccproxy(config)
            if _ccproxy_proc:
                import atexit

                atexit.register(stop_ccproxy, _ccproxy_proc)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    show_thinking = config.show_thinking if not no_thinking else False
    effective_channel_thinking = config.channel_send_thinking and (not no_thinking)

    # Validate mutually exclusive options
    if workdir and use_cwd:
        raise typer.BadParameter("Use either --workdir or --use-cwd, not both.")

    if mode and (workdir or use_cwd):
        raise typer.BadParameter(
            "--mode cannot be combined with --workdir or --use-cwd"
        )

    if mode and mode not in ("run", "daemon"):
        raise typer.BadParameter("--mode must be 'run' or 'daemon'")
    if ui and ui.lower() not in ("cli", "tui"):
        raise typer.BadParameter("--ui must be 'tui' or 'cli'")

    # --name only makes sense in run mode
    if name and not (
        mode == "run"
        or (not mode and not workdir and not use_cwd and config.default_mode == "run")
    ):
        raise typer.BadParameter("--name can only be used with --mode run")

    # Sanitize run name: allow alphanumeric, hyphens, underscores
    if name:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise typer.BadParameter(
                "--name may only contain letters, digits, hyphens, and underscores"
            )

    # Resolve effective mode from config (CLI mode already applied via overrides)
    effective_mode: str | None = (
        None  # None means explicit --workdir/--use-cwd was used
    )

    # Resolve workspace directory for this session
    # Priority: --workdir > --mode (explicit) > default_workdir > default_mode > cwd
    # --use-cwd is kept for backward compat but is now the default behavior
    if use_cwd:
        workspace_dir = os.getcwd()
        set_workspace_root(workspace_dir)
        workspace_fixed = True
    elif workdir:
        workspace_dir = os.path.abspath(os.path.expanduser(workdir))
        os.makedirs(workspace_dir, exist_ok=True)
        set_workspace_root(workspace_dir)
        workspace_fixed = True
    elif mode:
        # Explicit --mode overrides default_workdir
        effective_mode = mode
        workspace_root = config.default_workdir or os.getcwd()
        workspace_root = os.path.abspath(os.path.expanduser(workspace_root))
        set_workspace_root(workspace_root)
        if effective_mode == "run":
            runs_dir = Path(workspace_root, "runs")
            session_id = (
                _deduplicate_run_name(name, runs_dir)
                if name
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            workspace_dir = os.path.join(runs_dir, session_id)
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = False
        else:  # daemon
            workspace_dir = workspace_root
            workspace_fixed = True
    elif config.default_workdir:
        # Use configured default workdir with configured mode
        workspace_root = os.path.abspath(os.path.expanduser(config.default_workdir))
        set_workspace_root(workspace_root)
        effective_mode = config.default_mode
        if effective_mode == "run":
            runs_dir = Path(workspace_root, "runs")
            session_id = (
                _deduplicate_run_name(name, runs_dir)
                if name
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            workspace_dir = os.path.join(runs_dir, session_id)
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = False
        else:  # daemon
            workspace_dir = workspace_root
            workspace_fixed = True
    else:
        effective_mode = config.default_mode
        workspace_root = os.getcwd()
        set_workspace_root(workspace_root)
        if effective_mode == "run":
            workspace_dir = _create_session_workspace(name)
            workspace_fixed = False
        else:  # daemon mode (default) — use current directory
            workspace_dir = workspace_root
            workspace_fixed = True

    # Ensure memory and skills subdirs exist in workspace
    ensure_dirs()

    if prompt:
        # Single-shot mode: wrap in persistent checkpointer
        import asyncio

        from ..sessions import generate_thread_id, get_checkpointer

        async def _single_shot():
            async with get_checkpointer() as checkpointer:
                console.print("[dim]Loading agent...[/dim]")
                agent = _load_agent(
                    workspace_dir=workspace_dir,
                    checkpointer=checkpointer,
                    config=config,
                )
                tid = thread_id or generate_thread_id()
                cmd_run(
                    agent,
                    prompt,
                    thread_id=tid,
                    show_thinking=show_thinking,
                    workspace_dir=workspace_dir,
                    model=config.model,
                    ui_backend=config.ui_backend,
                )

        import nest_asyncio  # type: ignore[import-untyped]

        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_single_shot())
    else:
        # Interactive mode (default) — checkpointer managed inside cmd_interactive
        cmd_interactive(
            show_thinking=show_thinking,
            channel_send_thinking=effective_channel_thinking,
            workspace_dir=workspace_dir,
            workspace_fixed=workspace_fixed,
            mode=effective_mode,
            model=config.model,
            provider=config.provider,
            run_name=name,
            thread_id=thread_id,
            ui_backend=config.ui_backend,
            config=config,
        )


def _configure_logging():
    """Configure logging with warning symbols for better visibility."""
    from rich.logging import RichHandler

    class DimWarningHandler(RichHandler):
        """Custom handler that renders warnings in dim style."""

        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                # Use Rich console to print dim warning
                msg = record.getMessage()
                console.print(
                    f"[dim yellow]\u26a0\ufe0f  Warning:[/dim yellow] [dim]{escape(msg)}[/dim]"
                )
            else:
                super().emit(record)

    # Configure root logger to use our handler for WARNING and above
    handler = DimWarningHandler(
        console=console, show_time=False, show_path=False, show_level=False
    )
    handler.setLevel(logging.WARNING)

    # Apply to root logger (catches all loggers including deepagents)
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)

    # Suppress noisy schema warnings from langchain_google_genai
    # (e.g. "Key '$schema' is not supported in schema, ignoring")
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
