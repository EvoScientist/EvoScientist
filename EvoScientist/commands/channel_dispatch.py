from __future__ import annotations

from io import StringIO
from typing import Any

from rich.console import Console

from .base import CommandContext
from .manager import manager

_SUPPORTED_CHANNEL_COMMANDS = {"/gpu"}


class _CaptureChannelCommandUI:
    """Non-interactive command UI that captures output as plain text."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    @property
    def supports_interactive(self) -> bool:
        return False

    @property
    def output(self) -> str:
        return "\n".join(part for part in self._parts if part).strip()

    def append_system(self, text: str, style: str = "dim") -> None:
        del style
        self._parts.append(text)

    def mount_renderable(self, renderable: Any) -> None:
        with StringIO() as buffer:
            console = Console(
                file=buffer,
                force_terminal=False,
                width=120,
                color_system=None,
            )
            console.print(renderable)
            text = buffer.getvalue().rstrip()

        if text:
            self._parts.append(f"```\n{text}\n```")

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        del threads, current_thread
        self.append_system(f"{title}\nUse /resume <id> to continue.")
        return None

    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None:
        del index, installed_names, pre_filter_tag
        self.append_system(
            "Interactive skill browsing is not supported in channel commands."
        )
        return None

    async def wait_for_mcp_browse(
        self, servers: list, installed_names: set[str], pre_filter_tag: str
    ) -> list | None:
        del servers, installed_names, pre_filter_tag
        self.append_system(
            "Interactive MCP browsing is not supported in channel commands."
        )
        return None

    def clear_chat(self) -> None:
        self.append_system("Clear chat is not supported in channel commands.")

    def request_quit(self) -> None:
        self.append_system("Quit is ignored in channel commands.")

    def start_new_session(self) -> None:
        self.append_system("New session requested.")

    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None:
        del workspace_dir
        self.append_system(f"Session resumed: {thread_id}")

    async def flush(self) -> None:
        return


async def execute_supported_channel_command(
    command_str: str,
    *,
    agent: Any,
    thread_id: str,
    workspace_dir: str | None = None,
    checkpointer: Any = None,
    config: Any = None,
) -> str | None:
    """Execute a built-in channel command and return its outbound text."""

    normalized = command_str.strip().lower()
    if normalized not in _SUPPORTED_CHANNEL_COMMANDS:
        return None

    ui = _CaptureChannelCommandUI()
    ctx = CommandContext(
        agent=agent,
        thread_id=thread_id,
        ui=ui,
        workspace_dir=workspace_dir,
        checkpointer=checkpointer,
        config=config,
    )

    handled = await manager.execute(command_str, ctx)
    if not handled:
        return None
    return ui.output or "No response"
