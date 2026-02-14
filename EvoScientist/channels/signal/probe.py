"""Signal credential validation."""
import logging
logger = logging.getLogger(__name__)


async def validate_signal(
    phone_number: str,
    cli_path: str = "signal-cli",
    rpc_port: int = 7583,
) -> tuple[bool, str]:
    """Validate Signal setup by checking signal-cli availability."""
    import asyncio, subprocess

    if not phone_number:
        return False, "phone_number is required"

    # Check signal-cli binary
    loop = asyncio.get_event_loop()
    def _check():
        try:
            result = subprocess.run(
                [cli_path, "--version"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return True, f"signal-cli {result.stdout.strip()}"
            return False, "signal-cli returned error"
        except FileNotFoundError:
            return False, f"signal-cli not found at '{cli_path}'"
        except Exception as e:
            return False, f"Error: {e}"

    return await loop.run_in_executor(None, _check)
