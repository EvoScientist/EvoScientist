"""Inline skill browser widget for /evoskills in TUI.

Two-phase keyboard-driven widget built on the shared picker engine
(``picker_base.TagCheckboxBrowserBase``):
  Phase 1 — tag picker (arrow keys + Enter to select, or Esc for all)
  Phase 2 — skill checkbox (arrow keys to navigate, Space to toggle, Enter to confirm)

Posts ``SkillBrowserWidget.Confirmed`` with selected install sources,
or ``SkillBrowserWidget.Cancelled`` on Esc.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.message import Message

from .picker_base import TagCheckboxBrowserBase


class SkillBrowserWidget(TagCheckboxBrowserBase):
    """Inline skill browser — mounts in chat, keyboard-driven."""

    _INSTALLED_SUFFIX: ClassVar[str] = "  (installed)"
    _PHASE2_TITLE: ClassVar[str] = "Select skills to install"
    _PHASE2_CONFIRM_LABEL: ClassVar[str] = "install"

    class Confirmed(Message):
        """Posted when user confirms skill selection."""

        def __init__(self, install_sources: list[str]) -> None:
            super().__init__()
            self.install_sources = install_sources

    class Cancelled(Message):
        """Posted when user cancels."""

    def _item_name(self, item: Any) -> str:
        return item["name"]

    def _item_tags(self, item: Any) -> list[str]:
        return item.get("tags", [])

    def _item_desc(self, item: Any) -> str:
        return item["description"]

    def _post_confirmed(self, items: list[Any]) -> None:
        self.post_message(self.Confirmed([s["install_source"] for s in items]))

    def _post_cancelled(self) -> None:
        self.post_message(self.Cancelled())
