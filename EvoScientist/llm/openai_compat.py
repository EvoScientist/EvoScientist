"""Reusable behavior for OpenAI-compatible chat model integrations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from .patches import _OpenAICompatContent


class OpenAICompatContentMixin:
    """Normalize message content before calling an OpenAI-compatible model."""

    def _content_compat(self) -> _OpenAICompatContent:
        compat = self.__dict__.get("_evosci_content_compat")
        if not isinstance(compat, _OpenAICompatContent):
            profile = getattr(self, "profile", None)
            compat = _OpenAICompatContent(
                profile if isinstance(profile, Mapping) else None,
                hoist_tool_media=True,
            )
            self.__dict__["_evosci_content_compat"] = compat
        return compat

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._content_compat().invoke(
            super()._generate,  # type: ignore[attr-defined]
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self._content_compat().ainvoke(
            super()._agenerate,  # type: ignore[attr-defined]
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from self._content_compat().stream(
            super()._stream,  # type: ignore[attr-defined]
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._content_compat().astream(
            super()._astream,  # type: ignore[attr-defined]
            messages,
            stop=stop,
            run_manager=run_manager,
            stream_usage=stream_usage,
            **kwargs,
        ):
            yield chunk
