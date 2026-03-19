"""JSON-RPC client for imsg CLI.

Communicates with the imsg CLI via JSON-RPC over stdio,
similar to OpenClaw's approach.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RpcError:
    """RPC error response."""

    code: int | None = None
    message: str | None = None
    data: Any = None


@dataclass
class RpcNotification:
    """RPC notification (no id, server-initiated)."""

    method: str
    params: Any = None


class ImsgRpcClient:
    """JSON-RPC client for imsg CLI.

    Spawns `imsg rpc` as a subprocess and communicates via stdin/stdout.

    Example:
        client = ImsgRpcClient(cli_path="/usr/local/bin/imsg")
        await client.start()
        result = await client.request("send", {"to": "+1234", "text": "Hello"})
        await client.stop()
    """

    def __init__(
        self,
        cli_path: str = "imsg",
        db_path: str | None = None,
        on_notification: Callable[[RpcNotification], None] | None = None,
    ):
        self.cli_path = cli_path
        self.db_path = db_path
        self.on_notification = on_notification

        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._closed = asyncio.Event()

    async def start(self) -> None:
        """Start the imsg rpc subprocess."""
        if self._process is not None:
            return

        args = [self.cli_path, "rpc"]
        if self.db_path:
            args.extend(["--db", self.db_path])

        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())
        logger.info(f"Started imsg rpc (pid={self._process.pid})")

    async def stop(self) -> None:
        """Stop the imsg rpc subprocess."""
        if self._process is None:
            return

        if self._process.stdin:
            self._process.stdin.close()

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=2.0)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()

        self._fail_all_pending(Exception("RPC client stopped"))
        self._process = None
        self._closed.set()
        logger.info("Stopped imsg rpc")

    async def wait_closed(self) -> None:
        """Wait for the client to close."""
        await self._closed.wait()

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            The result from the RPC response

        Raises:
            Exception: If request fails or times out
        """
        if self._process is None or self._process.stdin is None:
            raise Exception("RPC client not running")

        request_id = self._next_id
        self._next_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        line = json.dumps(payload) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending.pop(request_id, None)
            raise Exception(f"RPC request timeout: {method}") from None

    async def _read_loop(self) -> None:
        """Read and process responses from stdout."""
        if self._process is None or self._process.stdout is None:
            return

        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break

                self._handle_line(line.decode().strip())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading from imsg: {e}")
                break

        self._fail_all_pending(Exception("RPC connection closed"))
        self._closed.set()

    async def _stderr_loop(self) -> None:
        """Log stderr output."""
        if self._process is None or self._process.stderr is None:
            return

        while True:
            try:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.warning(f"imsg stderr: {line.decode().strip()}")
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _handle_line(self, line: str) -> None:
        """Handle a single JSON-RPC response line."""
        if not line:
            return

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse RPC response: {e}")
            return

        # Check if it's a response (has id)
        if "id" in data and data["id"] is not None:
            request_id = data["id"]
            future = self._pending.pop(request_id, None)
            if future is None:
                return

            if data.get("error"):
                error = data["error"]
                msg = error.get("message", "RPC error")
                future.set_exception(Exception(msg))
            else:
                future.set_result(data.get("result"))
            return

        # It's a notification
        if "method" in data:
            notification = RpcNotification(
                method=data["method"],
                params=data.get("params"),
            )
            if self.on_notification:
                try:
                    self.on_notification(notification)
                except Exception as e:
                    logger.error(f"Notification handler error: {e}")

    def _fail_all_pending(self, error: Exception) -> None:
        """Fail all pending requests with an error."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
