"""Shared command runtime utilities for non-TUI interactive surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any

from rich.console import Console

from ..commands.base import CommandContext, CommandUI
from ..commands.manager import manager as command_manager

_COMMANDS_REGISTERED = False

_NATIVE_ACTION_BY_COMMAND = {
    "/new": "new_thread",
    "/resume": "switch_thread",
    "/clear": "clear_thread",
    "/threads": "show_threads",
    "/skills": "show_skills",
    "/evoskills": "browse_skills",
    "/install-skill": "install_skill",
    "/uninstall-skill": "uninstall_skill",
    "/mcp": "manage_mcp",
    "/install-mcp": "browse_mcp",
    "/channel": "manage_channels",
    "/current": "show_current",
    "/compact": "compact_session",
    "/exit": "exit_session",
}


def _ensure_commands_registered() -> None:
    global _COMMANDS_REGISTERED
    if _COMMANDS_REGISTERED:
        return
    # Import side effects register all command implementations.
    from ..commands import implementation as _impl  # noqa: F401

    _COMMANDS_REGISTERED = True


@dataclass
class CommandCatalogEntry:
    name: str
    description: str
    aliases: list[str]
    arguments: list[dict[str, Any]]
    native_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "aliases": self.aliases,
            "arguments": self.arguments,
            "nativeAction": self.native_action,
        }


@dataclass
class CommandExecutionResult:
    executed: bool
    command: str
    outputs: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    resolved_thread_id: str
    workspace_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "executed": self.executed,
            "command": self.command,
            "outputs": self.outputs,
            "actions": self.actions,
            "threadId": self.resolved_thread_id,
            "workspaceDir": self.workspace_dir,
        }


class RecordingCommandUI(CommandUI):
    """CommandUI adapter that records output and requested UI actions."""

    def __init__(self) -> None:
        self.outputs: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self._resolved_thread_id: str | None = None
        self._resolved_workspace_dir: str | None = None

    @property
    def supports_interactive(self) -> bool:
        return False

    @property
    def resolved_thread_id(self) -> str | None:
        return self._resolved_thread_id

    @property
    def resolved_workspace_dir(self) -> str | None:
        return self._resolved_workspace_dir

    def append_system(self, text: str, style: str = "dim") -> None:
        content = str(text).strip()
        if not content:
            return
        self.outputs.append(
            {
                "kind": "text",
                "text": content,
                "style": style,
            }
        )

    def mount_renderable(self, renderable: Any) -> None:
        with StringIO() as buffer:
            console = Console(
                file=buffer,
                force_terminal=False,
                width=120,
                color_system=None,
            )
            console.print(renderable)
            content = buffer.getvalue().strip()

        if not content:
            return
        self.outputs.append({"kind": "renderable", "text": content})

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        self.append_system(
            "Interactive thread picker unavailable here. Use explicit thread selection.",
            style="yellow",
        )
        return None

    async def wait_for_skill_browse(
        self,
        index: list[dict],
        installed_names: set[str],
        pre_filter_tag: str,
    ) -> list[str] | None:
        self.append_system(
            "Interactive skill browser unavailable here. Use /install-skill instead.",
            style="yellow",
        )
        return None

    async def wait_for_mcp_browse(
        self,
        servers: list,
        installed_names: set[str],
        pre_filter_tag: str,
    ) -> list | None:
        self.append_system(
            "Interactive MCP browser unavailable here. Use /mcp install instead.",
            style="yellow",
        )
        return None

    def clear_chat(self) -> None:
        self.actions.append({"type": "clear_chat"})

    def request_quit(self) -> None:
        self.actions.append({"type": "request_quit"})

    def force_quit(self) -> None:
        self.actions.append({"type": "force_quit"})

    def start_new_session(self) -> None:
        self.actions.append({"type": "start_new_session"})

    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None:
        self._resolved_thread_id = thread_id
        self._resolved_workspace_dir = workspace_dir
        self.actions.append(
            {
                "type": "resume_session",
                "threadId": thread_id,
                "workspaceDir": workspace_dir,
            }
        )

    async def flush(self) -> None:
        return None

    # Optional helpers used by /compact in richer UIs.
    def start_compacting_indicator(self) -> None:
        self.append_system("Compacting conversation...", style="dim")

    def stop_compacting_indicator(self) -> None:
        return None

    def update_status_after_compact(self, tokens_after: int) -> None:
        self.actions.append({"type": "compact_status", "tokensAfter": tokens_after})


def build_command_catalog() -> list[dict[str, Any]]:
    _ensure_commands_registered()

    catalog: list[dict[str, Any]] = []
    for command in sorted(
        command_manager.get_all_commands(), key=lambda item: item.name
    ):
        entry = CommandCatalogEntry(
            name=command.name,
            description=command.description,
            aliases=list(command.alias),
            arguments=[
                {
                    "name": argument.name,
                    "required": bool(argument.required),
                    "description": argument.description,
                    "type": getattr(argument.type, "__name__", str(argument.type)),
                }
                for argument in command.arguments
            ],
            native_action=_NATIVE_ACTION_BY_COMMAND.get(command.name),
        )
        catalog.append(entry.to_dict())
    return catalog


async def execute_command_line(
    command_line: str,
    *,
    thread_id: str,
    agent: Any,
    workspace_dir: str | None,
    checkpointer: Any = None,
    input_tokens_hint: int | None = None,
) -> CommandExecutionResult:
    _ensure_commands_registered()

    ui = RecordingCommandUI()
    ctx = CommandContext(
        agent=agent,
        thread_id=thread_id,
        ui=ui,
        workspace_dir=workspace_dir,
        checkpointer=checkpointer,
        input_tokens_hint=input_tokens_hint,
    )
    executed = await command_manager.execute(command_line, ctx)
    resolved_thread_id = ui.resolved_thread_id or ctx.thread_id
    resolved_workspace_dir = ui.resolved_workspace_dir or ctx.workspace_dir
    return CommandExecutionResult(
        executed=executed,
        command=command_line,
        outputs=ui.outputs,
        actions=ui.actions,
        resolved_thread_id=resolved_thread_id,
        workspace_dir=resolved_workspace_dir,
    )
