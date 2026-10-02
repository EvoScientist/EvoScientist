"""History-based auto-suggest for Textual TUI Input widget.

Reads prompt_toolkit FileHistory format so Rich CLI and TUI share the same
history file at ~/.evoscientist/history.
"""

from __future__ import annotations

import time
from pathlib import Path

from textual.suggester import Suggester


class HistorySuggester(Suggester):
    """Suggest completions from a shared prompt_toolkit FileHistory file."""

    def __init__(self, history_file: Path) -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self._history_file = history_file
        self._entries: list[str] = self._load_history()

    def _load_history(self) -> list[str]:
        """Parse prompt_toolkit FileHistory format into a list (newest-first).

        Format: blocks separated by blank lines. Each block has a ``#`` comment
        line (timestamp) followed by one or more ``+`` prefixed content lines.
        """
        if not self._history_file.exists():
            return []

        entries: list[str] = []
        current_lines: list[str] = []
        try:
            for raw in self._history_file.read_text(encoding="utf-8").splitlines():
                if raw.startswith("+"):
                    current_lines.append(raw[1:])
                elif raw.startswith("#"):
                    # Comment/timestamp line — flush any accumulated entry
                    if current_lines:
                        entries.append("\n".join(current_lines))
                        current_lines = []
                else:
                    # Blank or unknown line — flush
                    if current_lines:
                        entries.append("\n".join(current_lines))
                        current_lines = []

            if current_lines:
                entries.append("\n".join(current_lines))
        except OSError:
            return []

        # Newest-first for matching
        entries.reverse()
        return entries

    async def get_suggestion(self, value: str) -> str | None:
        """Return the full history entry whose prefix matches *value*."""
        if not value:
            return None
        for entry in self._entries:
            if entry.startswith(value) and entry != value:
                return entry
        return None

    def append_entry(self, text: str) -> None:
        """Record a new entry in-memory and persist to the history file."""
        text = text.strip()
        if not text:
            return

        # Prepend so it's found first on next suggestion lookup
        self._entries.insert(0, text)

        # Append to file in prompt_toolkit FileHistory format
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_file.open("a", encoding="utf-8") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n# {ts}\n")
                for line in text.split("\n"):
                    f.write(f"+{line}\n")
        except OSError:
            pass
