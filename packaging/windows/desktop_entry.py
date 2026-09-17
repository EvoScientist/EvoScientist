"""PyInstaller entry for the windowed desktop shell (``EvoScientist.exe``).

Kept trivial: all logic lives in :mod:`EvoScientist.desktop`. This is only the
frozen-app entry point PyInstaller analyzes.
"""

from __future__ import annotations

import sys

from EvoScientist.desktop.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
