"""Shared constants and utilities for CLI and TUI modules."""

from datetime import UTC, datetime

from ..sessions import AGENT_NAME

WELCOME_SLOGANS = [
    "Ready for vibe research? What do you want cooking?",
    "Science doesn't sleep. Neither do your sub-agents.",
    "From hypothesis to paper — let's cook.",
    "Your research kitchen is ready. What's on the menu?",
    "Experiments don't run themselves. Oh wait — they do now.",
    "Drop a question. We'll bring the citations.",
    "Vibe first. Discovery follows.",
    "What breakthrough are we cooking today?",
    "Harness the vibe. Start the research.",
    "Ideas in. Discoveries out.",
]

# ASCII art logo — shared by both Rich CLI and Textual TUI banners.
LOGO_LINES = (
    r" ███████╗ ██╗   ██╗  ██████╗  ███████╗  ██████╗ ██╗ ███████╗ ███╗   ██╗ ████████╗ ██╗ ███████╗ ████████╗",
    r" ██╔════╝ ██║   ██║ ██╔═══██╗ ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║ ██╔════╝ ╚══██╔══╝",
    r" █████╗   ██║   ██║ ██║   ██║ ███████╗ ██║      ██║ █████╗   ██╔██╗ ██║    ██║    ██║ ███████╗    ██║   ",
    r" ██╔══╝   ╚██╗ ██╔╝ ██║   ██║ ╚════██║ ██║      ██║ ██╔══╝   ██║╚██╗██║    ██║    ██║ ╚════██║    ██║   ",
    r" ███████╗  ╚████╔╝  ╚██████╔╝ ███████║ ╚██████╗ ██║ ███████╗ ██║ ╚████║    ██║    ██║ ███████║    ██║   ",
    r" ╚══════╝   ╚═══╝    ╚═════╝  ╚══════╝  ╚═════╝ ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚══════╝    ╚═╝   ",
)

# Blue gradient: deep navy -> royal blue -> sky blue -> cyan
LOGO_GRADIENT = ["#1a237e", "#1565c0", "#1e88e5", "#42a5f5", "#64b5f6", "#90caf9"]


def build_metadata(workspace_dir: str | None, model: str | None) -> dict:
    """Build metadata dict for LangGraph checkpoint persistence."""
    return {
        "agent_name": AGENT_NAME,
        "updated_at": datetime.now(UTC).isoformat(),
        "workspace_dir": workspace_dir or "",
        "model": model or "",
    }
