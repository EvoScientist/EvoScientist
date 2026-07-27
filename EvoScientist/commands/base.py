from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..gateway import GraphGateway
    from ..runtime import AsyncRuntime


@dataclass
class Argument:
    """Definition of a command argument."""

    name: str
    type: type
    description: str
    required: bool = True


@dataclass
class SubCommand:
    """A subcommand of a parent slash command."""

    name: str
    description: str
    arguments: list[Argument] = field(default_factory=list)


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
    async def wait_for_model_pick(
        self,
        entries: list[tuple[str, str, str]],
        current_model: str | None,
        current_provider: str | None,
    ) -> tuple[str, str] | None: ...
    def clear_chat(self) -> None: ...
    def request_quit(self) -> None: ...
    def force_quit(self) -> None: ...
    async def start_new_session(self) -> None: ...
    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None: ...
    async def flush(self) -> None: ...


@dataclass
class ChannelRuntime:
    """Mutable handle to the agent + thread bound to running channels.

    Also holds session-scoped bindings mutated by slash commands — the
    ``active_teams`` list backs the ``/expert`` command, feeding into
    ``RunRequest.configurable_extra`` at stream call time.
    """

    agent: Any = None
    thread_id: str | None = None
    active_teams: list[str] = field(default_factory=list)

    def bind(self, agent: Any, thread_id: str) -> None:
        self.agent = agent
        self.thread_id = thread_id

    def clear(self) -> None:
        # ``active_teams`` is session-scoped and reset explicitly by ``/new``
        # (session.py) and ``/expert clear`` — not tied to channel lifecycle.
        # Clearing here on channel shutdown would silently dismiss the user's
        # invited experts, which they never asked for.
        self.agent = None
        self.thread_id = None


def active_teams_configurable_extra(
    runtime: ChannelRuntime | None,
) -> dict[str, Any] | None:
    """Build ``RunRequest.configurable_extra`` from a channel runtime.

    Returns ``{"active_teams": [...]}`` when the runtime has invited
    experts, or ``None`` when there is no runtime or no active invites —
    lets stream call sites forward the field unconditionally without
    each duplicating the "read runtime slot, build dict, drop when
    empty" three-liner.
    """
    if runtime is None:
        return None
    invited = list(runtime.active_teams)
    return {"active_teams": invited} if invited else None


@dataclass
class CommandContext:
    """Context passed to commands during execution."""

    agent: Any
    thread_id: str
    ui: CommandUI
    workspace_dir: str | None = None
    checkpointer: Any = None
    config: Any = None
    channel_runtime: ChannelRuntime | None = None
    graph_gateway: GraphGateway | None = None
    async_runtime: AsyncRuntime | None = None
    command_error: str | None = None
    # Real LLM input token count from last usage_metadata (includes system
    # prompt + tool schemas).  Used by /compact for accurate display.
    input_tokens_hint: int | None = None


class Command(ABC):
    """Base class for all EvoScientist slash commands."""

    name: str
    alias: ClassVar[list[str]] = []
    description: str
    arguments: ClassVar[list[Argument]] = []
    category: ClassVar[str] = "General"
    subcommands: ClassVar[list[SubCommand]] = []
    # When False, callers may dispatch this command without waiting for
    # the background agent load to finish — important so recovery
    # commands like ``/mcp add`` can run even when the MCP load is
    # failing and ``_await_agent_ready`` would hang.
    requires_agent: ClassVar[bool] = False

    def needs_agent(self, args: list[str]) -> bool:
        """Whether this specific invocation needs the agent.

        Default returns :attr:`requires_agent`.  Override when a command
        has a mix of agent-using and agent-free subcommands (e.g.
        ``/channel start`` vs ``/channel status``).
        """
        return self.requires_agent

    def get_completions(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Return completions for args typed after the command name.

        Default walks :attr:`subcommands` for the first positional token
        only.  Override for deeper levels (e.g. server names, thread IDs).
        """
        if not self.subcommands:
            return []
        if len(tokens) <= 1:
            prefix = tokens[0].lower() if tokens else ""
            matches = [
                (sc.name, sc.description)
                for sc in self.subcommands
                if sc.name.startswith(prefix)
            ]
            # Exact match — subcommand already complete, hide popup
            if len(matches) == 1 and matches[0][0] == prefix:
                return []
            return matches
        # partial + trailing space: /mcp a  → still show "add"
        if len(tokens) == 2 and tokens[1] == "":
            prefix = tokens[0].lower()
            if any(sc.name == prefix for sc in self.subcommands):
                return []
            return [
                (sc.name, sc.description)
                for sc in self.subcommands
                if sc.name.startswith(prefix)
            ]
        return []

    @abstractmethod
    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        """Execute the command with given context and arguments."""
        pass
