"""Inline MCP server browser widget for /install-mcp in TUI.

Two-phase keyboard-driven widget built on the shared picker engine
(``picker_base.TagCheckboxBrowserBase``):
  Phase 1 — tag picker (arrow keys + Enter to select, or Esc for all)
  Phase 2 — server checkbox (arrow keys to navigate, Space to toggle, Enter to confirm)

Posts ``MCPBrowserWidget.Confirmed`` with selected MCPServerEntry objects,
or ``MCPBrowserWidget.Cancelled`` on Esc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.message import Message

from .picker_base import TagCheckboxBrowserBase

if TYPE_CHECKING:
    from ...mcp.registry import MCPServerEntry


class MCPBrowserWidget(TagCheckboxBrowserBase):
    """Inline MCP server browser — mounts in chat, keyboard-driven."""

    _INSTALLED_SUFFIX: ClassVar[str] = "  (configured)"
    _PHASE2_TITLE: ClassVar[str] = "Select MCP servers to install"
    _PHASE2_CONFIRM_LABEL: ClassVar[str] = "install"

    class Confirmed(Message):
        """Posted when user confirms server selection."""

        def __init__(self, entries: list[MCPServerEntry]) -> None:
            super().__init__()
            self.entries = entries

    class Cancelled(Message):
        """Posted when user cancels."""

    def _item_name(self, item: Any) -> str:
        return item.name

    def _item_tags(self, item: Any) -> list[str]:
        return item.tags

    def _item_desc(self, item: Any) -> str:
        return item.description or item.label

    def _post_confirmed(self, items: list[Any]) -> None:
        self.post_message(self.Confirmed(items))

    def _post_cancelled(self) -> None:
        self.post_message(self.Cancelled())
