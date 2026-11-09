"""``EvoSci service``: what ``EvoSci server`` was called in 0.4.0 to 0.4.2,
kept for one release. Each command says so and runs its ``EvoSci server``
one.
"""

from __future__ import annotations

import typer  # type: ignore[import-untyped]

from ..stream.console import console
from ._app import service_app
from .server_cmd import server_start, server_status, server_stop


@service_app.callback()
def service() -> None:
    """Deprecated: use `EvoSci server`."""
    console.print(
        "[yellow]⚠ EvoSci service is deprecated and goes in the next release; "
        "use [bold]EvoSci server[/bold], which manages ccproxy too.[/yellow]"
    )


@service_app.command("start")
def service_start(
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="The project it registers (default: default_workdir, else "
        "the current folder)",
    ),
) -> None:
    """Deprecated: `EvoSci server start`."""
    server_start(workdir)


@service_app.command("stop")
def service_stop(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Stop it even while the server runs work"
    ),
) -> None:
    """Deprecated: `EvoSci server stop`."""
    server_stop(yes)


@service_app.command("status")
def service_status() -> None:
    """Deprecated: `EvoSci server status`."""
    server_status()
