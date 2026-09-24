"""PyInstaller entry for the bundled langgraph CLI (``langgraph.exe``).

The desktop shell starts the backend via ``subprocess.Popen(["langgraph",
"dev", ...])`` through ``manager._langgraph_exe``, which looks for a
``langgraph``/``langgraph.exe`` next to ``sys.executable`` first. Building this
second executable into the same onedir (beside ``EvoScientist.exe``) is what
makes that resolution succeed in the frozen app — no PATH, no separate install.
"""

from __future__ import annotations

from langgraph_cli.cli import cli

if __name__ == "__main__":
    cli()
