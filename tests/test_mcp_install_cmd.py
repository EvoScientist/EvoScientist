"""Tests for EvoScientist.cli.mcp_install_cmd."""

import io
from unittest.mock import patch

from rich.console import Console


class TestCmdInstallMcp:
    def test_missing_git_prints_the_git_message(self, no_git):
        from EvoScientist.cli.mcp_install_cmd import _cmd_install_mcp
        from EvoScientist.mcp.registry import _MARKETPLACE_CACHE

        _MARKETPLACE_CACHE.clear()
        out = io.StringIO()
        with patch(
            "EvoScientist.cli.mcp_install_cmd.console",
            Console(file=out, width=500, color_system=None),
        ):
            _cmd_install_mcp("")

        assert "Failed to fetch server index: git was not found on PATH." in (
            out.getvalue()
        )
