"""Slash commands for Research Radar: /radar, /radar run, /radar history, etc."""

from __future__ import annotations

from typing import ClassVar

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..base import Argument, Command, CommandContext
from ..manager import manager


def _render_radar_update(date_str: str, update) -> Panel:
    """Build a Rich Panel from a RadarUpdate."""
    from rich.console import Group
    from rich.markdown import Markdown

    parts = []
    if update.summary_text:
        parts.append(Markdown(update.summary_text))

    if update.papers:
        parts.append(Text(f"\nPapers ({len(update.papers)})", style="bold magenta"))
        for p in update.papers:
            parts.append(Text(f"\n  {p.title}", style="bold"))
            parts.append(Text(f"  {p.url}", style="dim underline"))
            parts.append(Text(f"  {p.relevance}", style=""))

    if update.trends:
        parts.append(Text("\nTrends", style="bold cyan"))
        for t in update.trends:
            parts.append(Text(f"  - {t}"))

    if update.ideas:
        parts.append(Text("\nIdeas", style="bold green"))
        for idea in update.ideas:
            parts.append(Text(f"\n  {idea.title}", style="bold"))
            parts.append(Text(f"  {idea.motivation}", style="dim"))

    if update.suggested_actions:
        parts.append(Text("\nSuggested Actions", style="bold yellow"))
        for a in update.suggested_actions:
            parts.append(Text(f"  - {a}"))

    if not parts:
        parts.append(Text("No results", style="dim"))

    parts.append(
        Text(
            '\nGive feedback in chat: "track similar work", '
            '"deprioritize X", "expand idea N"',
            style="dim italic",
        )
    )

    return Panel(
        Group(*parts),
        title=f"[Research Radar] {date_str}",
        border_style="cyan",
        padding=(1, 2),
    )


class RadarCommand(Command):
    """Research Radar: view updates, trigger scans, manage scheduling."""

    name = "/radar"
    description = "Research Radar — paper monitoring and updates"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="subcommand",
            type=str,
            description="run | history | enable | disable (default: show latest)",
            required=False,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        sub = args[0] if args else ""

        if sub == "run":
            await self._run(ctx)
        elif sub == "history":
            await self._history(ctx)
        elif sub == "enable":
            await self._enable(ctx)
        elif sub == "disable":
            await self._disable(ctx)
        elif sub == "cron":
            await self._cron(ctx, args[1:])
        elif sub == "status":
            await self._status(ctx)
        elif sub == "help":
            await self._help(ctx)
        elif sub and any(c.isdigit() for c in sub):
            await self._show_by_date(ctx, sub)
        else:
            await self._show_latest(ctx)

    async def _show_by_date(self, ctx: CommandContext, date_str: str) -> None:
        from ...radar import load_radar_update

        update = load_radar_update(date_str)
        if update is None:
            ctx.ui.append_system(f"No report found for '{date_str}'.", style="red")
            return
        ctx.ui.mount_renderable(_render_radar_update(date_str, update))

    async def _show_latest(self, ctx: CommandContext) -> None:
        from ...radar import load_latest_radar

        result = load_latest_radar()
        if result is None:
            ctx.ui.append_system(
                "No radar reports yet. Run /radar run to trigger a scan.",
                style="yellow",
            )
            return
        date_str, update = result
        ctx.ui.mount_renderable(_render_radar_update(date_str, update))

    async def _run(self, ctx: CommandContext) -> None:
        from ...radar import is_paper_navigator_installed, run_radar_now

        if not is_paper_navigator_installed():
            ctx.ui.append_system(
                "paper-navigator skill not installed. "
                "Run: /install-skill paper-navigator",
                style="red",
            )
            return
        ctx.ui.append_system("Running Research Radar scan (takes around one minute)...")
        try:
            result = await run_radar_now()
        except Exception as exc:
            ctx.ui.append_system(f"Radar scan failed: {exc}", style="red")
            return
        if result:
            ts, update = result
            ctx.ui.mount_renderable(_render_radar_update(ts, update))
        else:
            ctx.ui.append_system(
                "Radar scan produced no results. Check that RESEARCH_TASTE.md "
                "has research interests populated.",
                style="yellow",
            )

    async def _history(self, ctx: CommandContext) -> None:
        from ...radar import list_radar_history, load_radar_update

        entries = list_radar_history()
        if not entries:
            ctx.ui.append_system("No radar reports found.", style="yellow")
            return

        # Try interactive picker, fall back to table
        use_picker = ctx.ui.supports_interactive and hasattr(
            ctx.ui, "wait_for_radar_pick"
        )
        if use_picker:
            try:
                selected = await ctx.ui.wait_for_radar_pick(
                    entries, title=">>> Select radar report <<<"
                )
                if selected is None:
                    return
                update = load_radar_update(selected)
                if update is None:
                    ctx.ui.append_system(
                        f"Could not load report for {selected}.", style="red"
                    )
                    return
                ctx.ui.mount_renderable(_render_radar_update(selected, update))
                return
            except Exception:
                pass  # fall through to table

        table = Table(title="Radar History", show_header=True)
        table.add_column("Date", style="bold")
        table.add_column("Papers", justify="right")
        table.add_column("Summary", style="dim", max_width=50)
        for e in entries:
            table.add_row(e["date"], str(e["paper_count"]), e["summary"])
        ctx.ui.mount_renderable(table)
        ctx.ui.append_system(
            "Use /radar <date> to view a specific report (e.g. /radar 2026-06-12_134934)",
            style="dim",
        )

    async def _status(self, ctx: CommandContext) -> None:
        from ...config.settings import get_effective_config
        from ...radar import (
            get_scheduler_next_run,
            is_scheduler_running,
            list_radar_history,
        )

        cfg = get_effective_config()
        tz = cfg.radar_timezone or "UTC"
        history = list_radar_history()
        last = history[0]["date"] if history else "never"
        running = is_scheduler_running()
        next_run = get_scheduler_next_run()

        lines = [
            f"Enabled:    {'yes' if cfg.radar_enabled else 'no'}",
            f"Schedule:   {cfg.radar_schedule}",
            f"Timezone:   {tz}",
            f"Last run:   {last}",
            f"Reports:    {len(history)}",
            f"Scheduler:  {'running' if running else 'stopped'}",
        ]
        if next_run:
            lines.append(f"Next run:   {next_run.isoformat()}")

        ctx.ui.append_system("\n".join(lines))

    async def _enable(self, ctx: CommandContext) -> None:
        from ...config.settings import get_effective_config, load_config, save_config
        from ...radar import start_radar_scheduler

        config = load_config()
        config.radar_enabled = True
        save_config(config)

        effective = get_effective_config()
        start_radar_scheduler(effective)
        ctx.ui.append_system(
            f"Research Radar enabled. Schedule: {effective.radar_schedule}",
            style="green",
        )

    async def _cron(self, ctx: CommandContext, args: list[str]) -> None:
        from ...config.settings import get_effective_config, load_config, save_config
        from ...radar import start_radar_scheduler, stop_radar_scheduler

        if not args:
            cfg = get_effective_config()
            tz = cfg.radar_timezone or "UTC"
            status = "enabled" if cfg.radar_enabled else "disabled"
            ctx.ui.append_system(
                f"Schedule: {cfg.radar_schedule}  (timezone: {tz}, {status})"
            )
            return

        expr = " ".join(args)
        config = load_config()
        config.radar_schedule = expr
        save_config(config)

        effective = get_effective_config()
        if effective.radar_enabled:
            stop_radar_scheduler()
            start_radar_scheduler(effective)
        ctx.ui.append_system(f"Radar schedule updated: {expr}", style="green")

    async def _help(self, ctx: CommandContext) -> None:
        lines = [
            "/radar            Show the latest radar report",
            "/radar run        Trigger an immediate radar scan",
            "/radar history    Browse past reports (interactive picker)",
            "/radar status     Show enabled state, schedule, and last run",
            "/radar cron [exp] Show or set the cron schedule",
            "/radar enable     Enable scheduled scans",
            "/radar disable    Disable scheduled scans",
            "/radar help       Show this help",
            "",
            "CLI commands:  EvoSci radar enable | disable | run | show | history | schedule",
            "",
            "Cron syntax: MIN HOUR DOM MON DOW",
            "  0 9 * * 1    = weekly Monday 9am",
            "  0 9 * * *    = daily 9am",
            "  0 */6 * * *  = every 6 hours",
            "",
            "Radar reads your interests from RESEARCH_TASTE.md and searches for",
            "matching papers, trends, and research ideas.",
            "",
            "Feedback (type in chat after viewing a report):",
            '  "This paper is relevant, track similar work"',
            '  "Deprioritize application papers"',
            '  "Expand idea 2 into a research proposal"',
            '  "Monitor author X / this citation cluster"',
            "Feedback updates RESEARCH_TASTE.md and shapes future scans.",
        ]
        ctx.ui.append_system("\n".join(lines))

    async def _disable(self, ctx: CommandContext) -> None:
        from ...config.settings import load_config, save_config
        from ...radar import stop_radar_scheduler

        config = load_config()
        config.radar_enabled = False
        save_config(config)

        stop_radar_scheduler()
        msg = "Research Radar disabled."
        ctx.ui.append_system(msg, style="green")


manager.register(RadarCommand())
