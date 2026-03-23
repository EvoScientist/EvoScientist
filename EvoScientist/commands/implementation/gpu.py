from __future__ import annotations

import asyncio
import subprocess

from ..base import Command, CommandContext
from ..manager import manager

_GPU_CHECK_TIMEOUT_SECONDS = 30


def _format_code_block(text: str) -> str:
    body = text.rstrip() or "(empty output)"
    return f"```text\n{body}\n```"


def _run_nvidia_smi(nvidia_smi_path: str = "nvidia-smi") -> str:
    try:
        result = subprocess.run(
            [nvidia_smi_path],
            capture_output=True,
            check=True,
            text=True,
            timeout=_GPU_CHECK_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return "GPU check failed: `nvidia-smi` was not found on this machine."
    except subprocess.TimeoutExpired:
        return "GPU check failed: `nvidia-smi` timed out."
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or exc.stdout or str(exc)).strip()
        return f"GPU check failed:\n{_format_code_block(error_text)}"

    return _format_code_block(result.stdout)


class GPUCommand(Command):
    """Show host GPU status."""

    name = "/gpu"
    description = "Show host GPU status via nvidia-smi"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        if args:
            ctx.ui.append_system("Usage: /gpu", style="yellow")
            return

        output = await asyncio.to_thread(_run_nvidia_smi)
        ctx.ui.append_system(output)


manager.register(GPUCommand())
