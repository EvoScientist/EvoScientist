from __future__ import annotations

import asyncio
from typing import Any

from .base import CommandUI


class ChannelCommandUI(CommandUI):
    """CommandUI implementation for messaging channels with output buffering."""

    _TEXT_CHUNK_LIMIT = 3500

    @property
    def supports_interactive(self) -> bool:
        return False

    def __init__(
        self,
        channel_msg: Any,
        append_system_callback: Any = None,
        start_new_session_callback: Any = None,
        handle_session_resume_callback: Any = None,
    ):
        self.msg = channel_msg
        self.append_system_callback = append_system_callback
        self.start_new_session_callback = start_new_session_callback
        self.handle_session_resume_callback = handle_session_resume_callback
        self._system_buffer: list[str] = []

    def append_system(self, text: str, style: str = "dim") -> None:
        if self.append_system_callback:
            self.append_system_callback(text, style)

        # Buffer the text for grouped delivery to the channel
        # We ignore style for grouping but keep it for individual lines if needed
        self._system_buffer.append(text)

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(parts) if parts else ""
        return str(content).strip()

    async def _send_text_chunks(self, text: str) -> None:
        """Flush long plain-text payloads in channel-safe chunks."""
        text = (text or "").strip()
        if not text:
            return

        pending = text
        while pending:
            chunk = pending[: self._TEXT_CHUNK_LIMIT]
            if len(pending) > self._TEXT_CHUNK_LIMIT:
                split_at = chunk.rfind("\n")
                if split_at > 0:
                    chunk = chunk[:split_at]
            chunk = chunk.rstrip()
            if not chunk:
                chunk = pending[: self._TEXT_CHUNK_LIMIT]
            self.append_system(chunk)
            await self.flush()
            pending = pending[len(chunk) :].lstrip("\n")

    async def flush(self) -> None:
        """Send all buffered system messages as a single grouped message."""
        if not self._system_buffer:
            return

        grouped_text = "\n".join(self._system_buffer)
        self._system_buffer = []

        from ..channels.base import OutboundMessage
        from ..cli.channel import _bus_loop

        loop = _bus_loop
        if not loop:
            return

        outbound = OutboundMessage(
            channel=self.msg.channel_type,
            chat_id=self.msg.chat_id,
            content=grouped_text,
            reply_to=self.msg.message_id,
            metadata=self.msg.metadata,
        )

        if self.msg.bus_ref:
            coro = self.msg.bus_ref.publish_outbound(outbound)
        else:
            coro = self.msg.channel_ref.send(outbound)

        asyncio.run_coroutine_threadsafe(coro, loop)

    def mount_renderable(self, renderable: Any) -> None:
        # Convert Rich renderable to text/markdown for channel
        from io import StringIO

        from rich.console import Console

        # Increase width to 120 to avoid wrapping tables like /threads
        with StringIO() as f:
            console = Console(
                file=f, force_terminal=False, width=120, color_system=None
            )
            console.print(renderable)
            text = f.getvalue()

        from ..channels.base import OutboundMessage
        from ..cli.channel import _bus_loop

        loop = _bus_loop
        if not loop:
            return

        # Flush any pending system messages first to preserve order
        if self._system_buffer:
            asyncio.run_coroutine_threadsafe(self.flush(), loop)

        outbound = OutboundMessage(
            channel=self.msg.channel_type,
            chat_id=self.msg.chat_id,
            content=f"```\n{text}\n```",
            reply_to=self.msg.message_id,
            metadata=self.msg.metadata,
        )

        if self.msg.bus_ref:
            coro = self.msg.bus_ref.publish_outbound(outbound)
        else:
            coro = self.msg.channel_ref.send(outbound)

        asyncio.run_coroutine_threadsafe(coro, loop)

    async def wait_for_thread_pick(
        self, threads: list[dict], current_thread: str, title: str
    ) -> str | None:
        from ..sessions import _format_relative_time

        lines = [title, "Available sessions:"]
        for idx, thread in enumerate(threads, start=1):
            thread_id = thread.get("thread_id", "") or ""
            marker = " *" if thread_id == current_thread else ""
            message_count = int(thread.get("message_count", 0) or 0)
            preview = (thread.get("preview", "") or "").strip()
            if len(preview) > 90:
                preview = preview[:87] + "..."
            metadata_bits = []
            if message_count:
                label = "msg" if message_count == 1 else "msgs"
                metadata_bits.append(f"{message_count} {label}")
            relative = _format_relative_time(thread.get("updated_at"))
            if relative:
                metadata_bits.append(relative)
            metadata = f" ({' | '.join(metadata_bits)})" if metadata_bits else ""
            lines.append(f"{idx}. {thread_id}{marker}{metadata}")
            if preview:
                lines.append(f"   {preview}")
        lines.append("")
        lines.append("Reply with `/resume <id>` to continue one of these sessions.")

        await self._send_text_chunks("\n".join(lines))
        return None

    async def wait_for_skill_browse(
        self, index: list[dict], installed_names: set[str], pre_filter_tag: str
    ) -> list[str] | None:
        self.append_system(
            "Interactive skill browsing not supported in channels. Use /install-skill <name> instead."
        )
        await self.flush()
        return None

    def clear_chat(self) -> None:
        self.append_system("Clear chat not supported in channels.")

    def request_quit(self) -> None:
        self.append_system("Quit command ignored in channel.")

    def force_quit(self) -> None:
        self.request_quit()

    def start_new_session(self) -> None:
        if self.start_new_session_callback:
            self.start_new_session_callback()
        else:
            self.append_system(
                "New session requested. Please restart the channel link or use /new if supported."
            )

    async def handle_session_resume(
        self, thread_id: str, workspace_dir: str | None = None
    ) -> None:
        if self.handle_session_resume_callback:
            await self.handle_session_resume_callback(thread_id, workspace_dir)
        from ..sessions import get_thread_messages

        lines = [f"Resumed session: {thread_id}"]
        if workspace_dir:
            lines.append(f"Workspace: {workspace_dir}")

        messages = await get_thread_messages(thread_id)
        display = [m for m in messages if getattr(m, "type", None) in ("human", "ai")]

        if not display:
            lines.append("No saved messages in this session.")
            await self._send_text_chunks("\n".join(lines))
            return

        HISTORY_WINDOW = 20
        if len(display) > HISTORY_WINDOW:
            skipped = len(display) - HISTORY_WINDOW
            display = display[-HISTORY_WINDOW:]
            lines.append(f"Conversation history (showing last {HISTORY_WINDOW}, skipped {skipped}):")
        else:
            lines.append("Conversation history:")

        for message in display:
            text = self._extract_message_text(message)
            if not text:
                continue
            if getattr(message, "type", None) == "human":
                lines.append(f"User: {text}")
            else:
                lines.append(f"EvoScientist: {text}")

        await self._send_text_chunks("\n".join(lines))
