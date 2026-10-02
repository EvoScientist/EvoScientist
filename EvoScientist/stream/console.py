"""Shared Rich ``console`` singleton.

Lives in its own lightweight module so that importing just the ``console``
from ``EvoScientist.stream`` does not pull in the heavy rendering/event
machinery (``stream.events`` → ``langchain_core.messages``).
"""

from __future__ import annotations

import os
import sys

from rich.console import Console  # type: ignore[import-untyped]

console = Console(
    legacy_windows=(sys.platform == "win32"),
    no_color=os.getenv("NO_COLOR") is not None,
)
