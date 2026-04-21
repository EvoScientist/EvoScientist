"""Helper for printing the session-exit Goodbye message and resume hint."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape


def print_resume_hint(
    thread_id: str | None,
    console: Console | None = None,
) -> None:
    """Print ``Goodbye!`` and, when available, a resume hint for *thread_id*."""
    out = console or Console()
    out.print("[dim]Goodbye![/dim]")
    if thread_id:
        out.print()
        out.print("[dim]Resume this session with:[/dim]")
        out.print(f"[cyan]EvoSci --resume {escape(thread_id)}[/cyan]")
