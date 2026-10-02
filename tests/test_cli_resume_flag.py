"""Tests for the ``--resume`` CLI flag (alias of ``--thread-id``)."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from EvoScientist.cli._app import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_help() -> str:
    """Render ``--help`` and strip ANSI escapes.

    Rich inserts per-character color codes (e.g. ``\\x1b[36m-\\x1b[0m\\x1b[36m-name``),
    which break literal substring lookups like ``"--resume" in stdout``.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    return _ANSI_RE.sub("", result.stdout)


def test_resume_flag_listed_in_help():
    plain = _plain_help()
    assert "--resume" in plain
    assert "--thread-id" in plain


def test_thread_id_flag_still_works():
    """Backwards compatibility: --thread-id should remain a valid flag."""
    assert "--thread-id" in _plain_help()
