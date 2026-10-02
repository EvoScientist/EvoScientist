"""iMessage environment probe/diagnostics.

Provides utilities to detect and verify the imsg CLI environment.
"""

import asyncio
import shutil
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result of iMessage environment probe."""

    available: bool = False
    cli_path: str | None = None
    cli_version: str | None = None
    rpc_supported: bool = False
    error: str | None = None


def find_cli(cli_path: str = "imsg") -> str | None:
    """Find the imsg CLI binary.

    Args:
        cli_path: Path or command name to search

    Returns:
        Full path to CLI or None if not found
    """
    return shutil.which(cli_path)


async def get_cli_version(cli_path: str) -> str | None:
    """Get the imsg CLI version.

    Args:
        cli_path: Path to CLI binary

    Returns:
        Version string or None
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            cli_path,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        return stdout.decode().strip() or None
    except Exception:
        return None


async def check_rpc_support(cli_path: str) -> bool:
    """Check if CLI supports RPC mode.

    Args:
        cli_path: Path to CLI binary

    Returns:
        True if RPC is supported
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            cli_path,
            "rpc",
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5.0)
        return proc.returncode == 0
    except Exception:
        return False


async def probe_imessage(
    cli_path: str = "imsg",
    timeout_ms: int = 10000,
) -> ProbeResult:
    """Probe the iMessage environment.

    Args:
        cli_path: Path or name of imsg CLI
        timeout_ms: Timeout in milliseconds

    Returns:
        ProbeResult with environment details
    """
    result = ProbeResult()

    # Find CLI
    found_path = find_cli(cli_path)
    if not found_path:
        result.error = f"imsg CLI not found: {cli_path}"
        return result

    result.cli_path = found_path
    result.available = True

    # Get version
    result.cli_version = await get_cli_version(found_path)

    # Check RPC support
    result.rpc_supported = await check_rpc_support(found_path)

    return result
