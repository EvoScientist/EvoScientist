"""Plugin-based channel interface.

A ChannelPlugin is a declarative object with optional adapter slots.
The framework inspects which slots are filled and auto-assembles
the message processing pipeline.

The ``Channel`` base class extends ``ChannelPlugin``, so all channel
implementations are automatically ChannelPlugin instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .capabilities import ChannelCapabilities


# ── Channel metadata ─────────────────────────────────────────────────

@dataclass
class ChannelMeta:
    """Channel metadata for registry and UI."""

    id: str
    label: str
    description: str = ""
    docs_path: str = ""
    system_image: str = ""  # icon name


# ── Adapter Protocols (slots) ────────────────────────────────────────

@runtime_checkable
class ConfigAdapter(Protocol):
    """Account configuration management."""

    def list_account_ids(self, config: Any) -> list[str]: ...
    def resolve_account(self, config: Any, account_id: str | None = None) -> Any: ...
    def is_enabled(self, account: Any, config: Any) -> bool: ...
    def is_configured(self, account: Any, config: Any) -> bool: ...


@runtime_checkable
class SecurityAdapter(Protocol):
    """DM policy and security warnings."""

    def resolve_dm_policy(self, ctx: Any) -> str: ...  # "open" | "allowlist" | "pairing"
    def collect_warnings(self, ctx: Any) -> list[str]: ...


@runtime_checkable
class GroupAdapter(Protocol):
    """Per-group policy resolution."""

    def resolve_require_mention(self, ctx: Any) -> bool | None: ...
    def resolve_tool_policy(self, ctx: Any) -> dict[str, Any] | None: ...
    def resolve_intro_hint(self, ctx: Any) -> str | None: ...


@runtime_checkable
class MentionAdapter(Protocol):
    """Bot mention detection and stripping."""

    def strip_mentions(self, text: str, ctx: Any) -> str: ...


@runtime_checkable
class OutboundAdapter(Protocol):
    """Outbound message delivery."""

    delivery_mode: str  # "direct" | "gateway" | "hybrid"

    async def send_text(self, ctx: Any) -> bool: ...
    async def send_media(self, ctx: Any) -> bool: ...


@runtime_checkable
class ThreadingAdapter(Protocol):
    """Reply threading behavior."""

    def resolve_reply_to_mode(self, ctx: Any) -> str: ...  # "off" | "first" | "all"


@runtime_checkable
class StreamingAdapter(Protocol):
    """Edit-in-place streaming output."""

    async def edit_message(self, chat_id: str, message_id: str, text: str) -> bool: ...


@runtime_checkable
class DirectoryAdapter(Protocol):
    """Contact/group directory queries."""

    async def list_peers(self, ctx: Any) -> list[dict]: ...
    async def list_groups(self, ctx: Any) -> list[dict]: ...
    async def list_group_members(self, ctx: Any) -> list[dict]: ...


@runtime_checkable
class StatusAdapter(Protocol):
    """Health probing and status reporting."""

    async def probe_account(self, ctx: Any) -> Any: ...
    async def audit_account(self, ctx: Any) -> Any: ...
    def collect_status_issues(self, accounts: list) -> list[dict]: ...


@runtime_checkable
class HeartbeatAdapter(Protocol):
    """Channel heartbeat / readiness checks."""

    async def check_ready(self, ctx: Any) -> tuple[bool, str]: ...


@runtime_checkable
class ActionsAdapter(Protocol):
    """Message actions (react, edit, delete, poll, etc.)."""

    def list_actions(self) -> list[str]: ...
    async def handle_action(self, action: str, ctx: Any) -> Any: ...


@runtime_checkable
class PairingAdapter(Protocol):
    """DM pairing flow."""

    id_label: str

    def normalize_entry(self, entry: str) -> str: ...
    async def notify_approval(self, ctx: Any) -> None: ...


@runtime_checkable
class OnboardingAdapter(Protocol):
    """Interactive setup wizard hooks."""

    async def wizard_steps(self, ctx: Any) -> list[dict]: ...
    async def validate_step(self, step: str, value: Any) -> str | None: ...


# ── Reload policy ────────────────────────────────────────────────────

@dataclass
class ReloadPolicy:
    """Declares which config prefixes trigger a channel reload."""

    config_prefixes: list[str] = field(default_factory=list)
    noop_prefixes: list[str] = field(default_factory=list)


# ── ChannelPlugin ────────────────────────────────────────────────────

class ChannelPlugin:
    """Declarative channel plugin with optional adapter slots.

    Replaces the monolithic Channel base class.  Each slot is optional —
    the framework adapts behavior based on which are present.

    Usage::

        class MyPlugin(ChannelPlugin):
            id = "my_channel"
            meta = ChannelMeta(id="my_channel", label="My Channel")
            capabilities = ChannelCapabilities(...)

            def __init__(self):
                self.outbound = MyOutboundAdapter()
                self.config_adapter = MyConfigAdapter()

            async def start(self, config, account_id=None):
                ...

            async def stop(self, account_id=None):
                ...
    """

    id: str = ""
    meta: ChannelMeta | None = None
    capabilities: ChannelCapabilities = ChannelCapabilities()

    # Optional adapter slots — fill what you need
    # Default: SingleAccountConfigAdapter so every plugin has multi-account
    # support out of the box (returns a single "default" account).
    config_adapter: ConfigAdapter | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self) -> None:
        # Provide default SingleAccountConfigAdapter if not overridden
        if self.config_adapter is None:
            from .config import SingleAccountConfigAdapter
            self.config_adapter = SingleAccountConfigAdapter()
    security: SecurityAdapter | None = None
    groups: GroupAdapter | None = None
    mentions: MentionAdapter | None = None
    outbound: OutboundAdapter | None = None
    threading: ThreadingAdapter | None = None
    streaming: StreamingAdapter | None = None
    directory: DirectoryAdapter | None = None
    status: StatusAdapter | None = None
    heartbeat: HeartbeatAdapter | None = None
    actions: ActionsAdapter | None = None
    pairing: PairingAdapter | None = None
    onboarding: OnboardingAdapter | None = None

    # Lifecycle
    reload: ReloadPolicy | None = None

    # Connection management
    async def start(self, config: Any, account_id: str | None = None) -> None:
        """Start the channel (or a specific account)."""

    async def stop(self, account_id: str | None = None) -> None:
        """Stop the channel (or a specific account)."""

    def filled_slots(self) -> list[str]:
        """Return names of adapter slots that are not None."""
        slot_names = [
            "config_adapter", "security", "groups", "mentions", "outbound",
            "threading", "streaming", "directory", "status", "heartbeat",
            "actions", "pairing", "onboarding",
        ]
        return [s for s in slot_names if getattr(self, s, None) is not None]
