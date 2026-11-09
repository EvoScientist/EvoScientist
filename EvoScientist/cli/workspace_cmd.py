"""``EvoSci workspace``: the projects the background server serves.

Everything a project owns (conversations, scheduled tasks, memory, AutoSkills
proposals) is kept outside its folder, keyed on the folder's path, so a
moved or renamed folder starts as a new, empty project. ``move`` carries the
project over.
"""

from __future__ import annotations

import os

import typer  # type: ignore[import-untyped]
from rich.markup import escape

from ..stream.console import console
from ._app import workspace_app


@workspace_app.command("move")
def workspace_move(
    old: str = typer.Argument(..., help="The folder the project was in"),
    new: str = typer.Argument(..., help="The folder it is in now"),
) -> None:
    """Carry a project over to the folder you moved or renamed it to: its
    conversations, scheduled tasks, memory and AutoSkills proposals follow
    it. Moves no files. A folder copied, not moved, is a new project."""
    from ..config import get_effective_config
    from ..langgraph_dev import manager, registration
    from ..paths import PATH_ERRORS, Workspace

    try:
        old_project, new_project = Workspace(old), Workspace(new)
    except PATH_ERRORS as exc:
        raise typer.BadParameter(str(exc)) from exc
    shown_old = os.path.abspath(os.path.expanduser(old))
    if not new_project.root.is_dir():
        console.print(f"[red]{escape(str(new_project.root))} does not exist.[/red]")
        raise typer.Exit(1)
    # No sign-in check: a move calls no model.
    config = get_effective_config()
    try:
        with console.status(
            "[dim]Starting the background agent server...[/dim]", spinner="dots"
        ):
            # No project to register: the old folder is gone.
            result = manager.serve_project(config, None)
    except (registration.ServerUnusableError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if not result.available or result.url is None:
        console.print(
            "[red]The background agent server is not available"
            + (f": {escape(result.reason)}" if result.reason else ".")
            + "[/red]"
        )
        raise typer.Exit(1)
    try:
        with console.status(
            f"[dim]Moving {escape(shown_old)}...[/dim]", spinner="dots"
        ):
            report = registration.move_folder(old_project, new_project, url=result.url)
    except (registration.RegistrationError, registration.OlderServerError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    _print_report(shown_old, new_project.root, report)


def _print_report(old, new, report: dict) -> None:
    if report.get("resumed"):
        console.print(
            f"[green]✓[/green] Finished moving {escape(str(old))} to "
            f"{escape(str(new))}."
        )
    else:
        console.print(
            f"[green]✓[/green] Moved {escape(str(old))} to {escape(str(new))}: "
            f"{report.get('threads', 0)} conversation(s), "
            f"{report.get('tasks', 0)} scheduled task(s)"
            + (", its memory" if report.get("memory") else "")
            + (
                f", {report['proposals']} AutoSkills proposal(s)"
                if report.get("proposals")
                else ""
            )
            + "."
        )
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]⚠ {escape(str(warning))}[/yellow]")
    console.print(
        "[dim]An EvoSci serve still running for the old folder keeps its old "
        "proactive conversations until it is started again in the new one.[/dim]"
    )
