from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..base import Command, CommandContext
from ..manager import manager


class ChannelCommand(Command):
    """Configure messaging channels."""

    name = "/channel"
    description = "Configure messaging channels"

    def needs_agent(self, args: list[str]) -> bool:
        # ``status`` and ``stop`` are introspection / teardown; they
        # must work even when the agent load is still in flight or has
        # failed.  Only start/add flows feed ``ctx.agent`` into
        # ``_start_channels_bus_mode``.
        subcmd = args[0].lower() if args else ""
        return subcmd not in {"status", "stop"}

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        import EvoScientist.cli.channel as _ch_mod

        from ...cli.channel import (
            _channels_is_running,
            _channels_running_list,
            _channels_stop,
            _start_channels_bus_mode,
        )
        from ...config import load_config

        subcmd = args[0].lower() if args else ""

        if subcmd == "status" or (not subcmd and _channels_is_running()):
            running = _channels_running_list()
            if running and _ch_mod._manager:
                detailed = _ch_mod._manager.get_detailed_status()
                table = Table(
                    title="Channel Status",
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("Channel")
                table.add_column("Status")
                table.add_column("Details", style="dim")
                for name, info in detailed.items():
                    ok = info.get("running", False)
                    detail = (
                        f"sent: {info.get('sent', 0)}, recv: {info.get('received', 0)}"
                    )
                    status = (
                        "[green]\u25cf online[/green]"
                        if ok
                        else "[red]\u25cb offline[/red]"
                    )
                    table.add_row(name, status, detail)
                ctx.ui.mount_renderable(table)
            else:
                ctx.ui.append_system(
                    "No messaging channels are currently active.", style="yellow"
                )
            return

        if subcmd == "stop":
            target = args[1] if len(args) > 1 else ""
            if not _channels_is_running():
                ctx.ui.append_system("No channels are running.", style="dim")
            else:
                if target:
                    _channels_stop(target)
                    ctx.ui.append_system(f"Channel '{target}' stopped.", style="green")
                else:
                    _channels_stop()
                    ctx.ui.append_system("All channels stopped.", style="green")
            return

        config = load_config()
        send_thinking = config.channel_send_thinking

        # If args specifies channels, use them, otherwise use config
        # Handle both space-separated and comma-separated inputs
        requested = []
        for a in args:
            requested.extend([t.strip() for t in a.split(",") if t.strip()])

        if _channels_is_running():
            if not requested:
                ctx.ui.append_system(
                    "Channels already running. Specify type to add: /channel <type>",
                    style="yellow",
                )
                return

            ctx.ui.append_system(f"Adding channel(s): {', '.join(requested)}...")
            from ...cli.channel import _add_channel_to_running_bus

            try:
                for ct in requested:
                    _add_channel_to_running_bus(ct, config, send_thinking=send_thinking)
                # Sync CLI-side globals so _auto_start_channel observes the
                # latest agent/thread refs (see cli/channel.py _cli_agent /
                # _cli_thread_id). The pre-migration inline helper set these
                # in the same branch.
                _ch_mod._cli_agent = ctx.agent
                _ch_mod._cli_thread_id = ctx.thread_id
                ctx.ui.append_system("Channel(s) added successfully.", style="green")
            except Exception as e:
                ctx.ui.append_system(f"Failed to add channels: {e}", style="red")
            return

        ctx.ui.append_system("Starting channels...")
        original = config.channel_enabled
        if requested:
            config.channel_enabled = ",".join(requested)

        if not config.channel_enabled:
            ctx.ui.append_system(
                "No channels enabled in config. Use /channel <type> to start one.",
                style="yellow",
            )
            ctx.ui.append_system(
                "Types: telegram, discord, slack, feishu, dingtalk, wechat, email, imessage",
                style="dim",
            )
            return

        try:
            # We need to make sure we don't block the UI.
            # _start_channels_bus_mode might start threads/loops.
            _start_channels_bus_mode(
                config,
                ctx.agent,
                ctx.thread_id,
                send_thinking=send_thinking,
            )
            # Sync CLI-side globals so _auto_start_channel observes the
            # latest agent/thread refs (see cli/channel.py _cli_agent /
            # _cli_thread_id). The pre-migration inline helper set these
            # in the same branch.
            _ch_mod._cli_agent = ctx.agent
            _ch_mod._cli_thread_id = ctx.thread_id

            # Show status panel
            if _ch_mod._manager:
                detailed = _ch_mod._manager.get_detailed_status()
                all_ok = all(info.get("running", False) for info in detailed.values())
                lines = []
                for name, info in detailed.items():
                    ok = info.get("running", False)
                    line = Text()
                    if ok:
                        line.append("\u25cf ", style="green")
                        line.append(name, style="bold")
                    else:
                        line.append("\u25cb ", style="red")
                        line.append(name, style="bold red")

                    detail = (
                        f"sent: {info.get('sent', 0)}, recv: {info.get('received', 0)}"
                    )
                    line.append(f"  {detail}", style="dim")
                    lines.append(line)

                body = Text("\n").join(lines)
                border = "green" if all_ok else "yellow"
                ctx.ui.mount_renderable(
                    Panel(
                        body,
                        title="[bold]Channels[/bold]",
                        border_style=border,
                        expand=False,
                    )
                )
        except Exception as e:
            ctx.ui.append_system(f"Failed to start channels: {e}", style="red")
        finally:
            config.channel_enabled = original


# Register Channel command
manager.register(ChannelCommand())
