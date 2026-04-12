from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass
class Argument:
    """Definition of a command argument."""

    name: str
    type: type
    description: str
    required: bool = True


@runtime_checkable
class CommandUI(Protocol):
    """Protocol for UI operations that commands can perform."""

    @property
    def supports_interactive(self) -> bool: ...

    def append_system(self, text: str, style: str = "dim") -> None: ...
    def mount_renderable(self, renderable: Any) -> None: ...

    # Optional interactive operations
    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None: ...
    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None: ...
    async def wait_for_mcp_browse(
        self, servers: list, installed_names: set[str], pre_filter_tag: str
    ) -> list | None: ...
    def clear_chat(self) -> None: ...
    def request_quit(self) -> None: ...
    def force_quit(self) -> None: ...
    def start_new_session(self) -> None: ...
    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None: ...
    async def flush(self) -> None: ...


@dataclass
class CommandContext:
    """Context passed to commands during execution."""

    agent: Any
    thread_id: str
    ui: CommandUI
    workspace_dir: str | None = None
    checkpointer: Any = None
    config: Any = None
    # Real LLM input token count from last usage_metadata (includes system
    # prompt + tool schemas).  Used by /compact for accurate display.
    input_tokens_hint: int | None = None


class Command(ABC):
    """Base class for all EvoScientist slash commands."""

    name: str
    alias: ClassVar[list[str]] = []
    description: str
    arguments: ClassVar[list[Argument]] = []

    @abstractmethod
    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        """Execute the command with given context and arguments."""
        pass
