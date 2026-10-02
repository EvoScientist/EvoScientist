"""DeepSeek chat model integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_deepseek import ChatDeepSeek

from .openai_compat import OpenAICompatContentMixin

logger = logging.getLogger(__name__)

DEEPSEEK_THINKING_DISABLED = {"type": "disabled"}


def is_deepseek_thinking_disabled(
    extra_body: Mapping[str, object] | None,
) -> bool:
    """Return whether a request body explicitly disables DeepSeek thinking."""
    if not extra_body:
        return False
    thinking = extra_body.get("thinking")
    return isinstance(thinking, Mapping) and thinking.get("type") == "disabled"


def _inject_reasoning_content(
    messages: list[BaseMessage],
    payload: dict[str, object],
) -> dict[str, object]:
    """Copy captured DeepSeek reasoning into serialized assistant messages."""
    reasoning = [
        message.additional_kwargs.get("reasoning_content")
        for message in messages
        if isinstance(message, AIMessage)
    ]
    serialized = payload.get("messages")
    if not isinstance(serialized, list):
        return payload

    ai_index = 0
    for message in serialized:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        value = reasoning[ai_index] if ai_index < len(reasoning) else None
        if value:
            message["reasoning_content"] = value
        elif "reasoning_content" not in message:
            message["reasoning_content"] = ""
        ai_index += 1
    return payload


class EvoChatDeepSeek(OpenAICompatContentMixin, ChatDeepSeek):
    """ChatDeepSeek with EvoScientist's media and history compatibility."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if is_deepseek_thinking_disabled(self.extra_body):
            return payload

        try:
            messages = self._convert_input(input_).to_messages()
        except Exception:
            logger.warning(
                "DeepSeek reasoning passback: input conversion failed",
                exc_info=True,
            )
            return payload

        return _inject_reasoning_content(messages, payload)
