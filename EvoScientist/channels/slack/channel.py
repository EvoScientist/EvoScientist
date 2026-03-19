"""Slack channel implementation using slack-sdk Socket Mode."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from ..base import Channel, ChannelError, RawIncoming
from ..capabilities import SLACK as SLACK_CAPS
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)


@dataclass
class SlackConfig(BaseChannelConfig):
    bot_token: str = ""
    app_token: str = ""
    text_chunk_limit: int = 4096


class SlackChannel(Channel):
    """Slack channel using slack-sdk Socket Mode."""

    name = "slack"

    capabilities = SLACK_CAPS
    _ready_attrs = ("_web_client",)
    _mention_pattern = r"<@{bot_id}>\s*"

    def __init__(self, config: SlackConfig):
        super().__init__(config)
        self._socket_client = None
        self._web_client = None
        self._typing_message_ts: dict[str, str] = {}

    async def start(self) -> None:
        if not self.config.bot_token:
            raise ChannelError("Slack bot token is required")
        if not self.config.app_token:
            raise ChannelError(
                "Slack app token is required for Socket Mode (starts with xapp-)"
            )

        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.socket_mode.request import SocketModeRequest
            from slack_sdk.socket_mode.response import SocketModeResponse
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError:
            raise ChannelError(
                "slack-sdk or aiohttp not installed. "
                "Install with: pip install evoscientist[slack]"
            ) from None

        self._web_client = AsyncWebClient(
            token=self.config.bot_token,
            proxy=self._get_proxy(),
        )

        # Get bot user ID for filtering own messages
        try:
            auth = await asyncio.wait_for(
                self._web_client.auth_test(),
                timeout=15,
            )
            self._bot_user_id = auth["user_id"]
        except TimeoutError:
            raise ChannelError(
                "Slack auth_test timed out — check network and bot token"
            ) from None
        except Exception as e:
            raise ChannelError(f"Failed to authenticate Slack bot: {e}") from e

        self._socket_client = SocketModeClient(
            app_token=self.config.app_token,
            web_client=self._web_client,
        )

        async def _event_handler(
            client: SocketModeClient,
            req: SocketModeRequest,
        ) -> None:
            # Acknowledge immediately
            resp = SocketModeResponse(envelope_id=req.envelope_id)
            await client.send_socket_mode_response(resp)

            logger.debug(f"Slack socket event: type={req.type}")

            if req.type == "events_api":
                event = req.payload.get("event", {})
                event_type = event.get("type", "")
                if event_type == "message" and "subtype" not in event:
                    is_dm = event.get("channel_type") == "im"
                    await self._on_message(
                        event,
                        is_group=not is_dm,
                        was_mentioned=is_dm,
                    )
                elif event_type == "app_mention":
                    await self._on_message(
                        event,
                        is_group=True,
                        was_mentioned=True,
                    )

        self._socket_client.socket_mode_request_listeners.append(_event_handler)
        try:
            await asyncio.wait_for(
                self._socket_client.connect(),
                timeout=30,
            )
        except TimeoutError:
            raise ChannelError(
                "Slack Socket Mode connection timed out — "
                "check app token (must start with xapp-) and "
                "ensure Socket Mode is enabled in your Slack app settings"
            ) from None
        self._running = True
        logger.info("Slack channel started (Socket Mode)")

    async def _cleanup(self) -> None:
        if self._socket_client:
            await self._socket_client.close()
            logger.info("Slack channel stopped")

    # ── Typing indicator (override base) ────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        """Send typing indicator via Slack.

        Slack's Web API and Socket Mode do not expose a dedicated
        typing-indicator endpoint for bot tokens. We approximate
        the experience by posting a short-lived status message that
        is deleted once the real reply is sent (handled by
        ``stop_typing``).  When the status post fails we silently
        fall back to no indicator.
        """
        if not self._web_client:
            return
        try:
            resp = await self._web_client.chat_postMessage(
                channel=chat_id,
                text="\u2026",  # "…" ellipsis as minimal typing hint
            )
            ts = resp.get("ts")
            if ts:
                self._typing_message_ts[chat_id] = ts
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        """Cancel typing loop and clean up the status message."""
        # Delete the ephemeral "…" message if we posted one
        ts = self._typing_message_ts.pop(chat_id, None)
        if ts and self._web_client:
            try:
                await self._web_client.chat_delete(channel=chat_id, ts=ts)
            except Exception:
                pass
        await super().stop_typing(chat_id)

    # ── Send (template method overrides) ──────────────────────────

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        kwargs = {"channel": chat_id}
        # Always route to thread if thread_ts is present in metadata,
        # not just for the first chunk (reply_to is only set for chunk 0).
        if metadata:
            thread_ts = metadata.get("thread_ts")
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

        async def _send(text):
            await self._web_client.chat_postMessage(text=text, **kwargs)

        await self._send_with_format_fallback(_send, formatted_text, raw_text)

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file through Slack."""
        channel_id = self._resolve_media_chat_id(recipient, metadata)
        await self._web_client.files_upload_v2(
            channel=channel_id,
            file=file_path,
            initial_comment=caption or None,
        )
        return True

    def _get_bot_identifier(self) -> str | None:
        return getattr(self, "_bot_user_id", None)

    # ── ACK Reactions ───────────────────────────────────────────────

    async def _send_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "eyes"
    ) -> None:
        """Add an emoji reaction to acknowledge receipt."""
        if self._web_client and message_id:
            try:
                await self._web_client.reactions_add(
                    channel=chat_id,
                    timestamp=message_id,
                    name=emoji,
                )
            except Exception as e:
                logger.debug(f"Slack ACK reaction failed: {e}")

    async def _remove_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "eyes"
    ) -> None:
        """Remove the ACK reaction after replying."""
        if self._web_client and message_id:
            try:
                await self._web_client.reactions_remove(
                    channel=chat_id,
                    timestamp=message_id,
                    name=emoji,
                )
            except Exception as e:
                logger.debug(f"Slack remove ACK reaction failed: {e}")

    async def _on_message(
        self,
        event: dict,
        *,
        is_group: bool = False,
        was_mentioned: bool = True,
    ) -> None:
        """Handle an incoming Slack message event."""
        user_id = event.get("user", "")

        # Skip bot's own messages
        if user_id == getattr(self, "_bot_user_id", None):
            logger.debug("Skipping own bot message")
            return

        # Skip bot messages (e.g. from other bots)
        if event.get("bot_id"):
            logger.debug(f"Skipping bot message from bot_id={event.get('bot_id')}")
            return

        channel_id = event.get("channel", "")

        text = event.get("text", "")

        annotations: list[str] = []
        media_paths: list[str] = []

        # Handle file attachments
        if self.config.include_attachments:
            files = event.get("files", [])
            for file_info in files:
                file_size = file_info.get("size", 0)
                filename = file_info.get("name", "unknown")

                url = file_info.get("url_private_download") or file_info.get(
                    "url_private"
                )
                if url and self._web_client:
                    headers = {"Authorization": f"Bearer {self.config.bot_token}"}
                    local_path, annotation = await self._download_attachment(
                        url,
                        f"{file_info.get('id', 'unknown')}_{filename}",
                        headers=headers,
                        file_size=file_size,
                    )
                    if local_path:
                        media_paths.append(local_path)
                    if annotation:
                        annotations.append(annotation)

        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts") or ts
        try:
            timestamp = datetime.fromtimestamp(float(ts)) if ts else datetime.now()
        except (ValueError, TypeError):
            timestamp = datetime.now()

        await self._enqueue_raw(
            RawIncoming(
                sender_id=user_id,
                chat_id=channel_id,
                text=text,
                media_files=media_paths,
                content_annotations=annotations,
                timestamp=timestamp,
                message_id=ts,
                metadata={"chat_id": channel_id, "thread_ts": thread_ts},
                is_group=is_group,
                was_mentioned=was_mentioned,
            )
        )
        logger.info(
            f"Slack message queued: sender={user_id}, "
            f"channel={channel_id}, content={text[:50]}"
        )
