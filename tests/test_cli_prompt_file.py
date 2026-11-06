"""Tests for the ``--prompt-file`` CLI flag.

``--prompt-file`` reads the query from a file instead of the command line,
which bypasses the Windows CreateProcess 32767-character command-line limit
for very long prompts (e.g. full papers embedded in the query).
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from EvoScientist.cli._app import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(result) -> str:
    """Strip ANSI escapes (Rich color codes break literal substring lookups)."""
    return _ANSI_RE.sub("", result.stdout + (result.stderr or ""))


def _plain_help() -> str:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    return _ANSI_RE.sub("", result.stdout)


def test_prompt_file_flag_listed_in_help():
    assert "--prompt-file" in _plain_help()


def test_prompt_file_missing_path_errors(tmp_path):
    """A nonexistent --prompt-file path fails fast with a clear message."""
    missing = tmp_path / "no-such-prompt.txt"
    result = runner.invoke(app, ["--prompt-file", str(missing)])
    assert result.exit_code != 0
    assert "--prompt-file not found" in _plain(result)
