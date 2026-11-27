"""Reusable behavior for OpenAI-compatible chat model integrations."""

from __future__ import annotations

import functools
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
        # The normalizer is built once in as_content_compat_model and
        # carried in __dict__, so a copy shares what the model it was
        # copied from learned about the media its endpoint refuses.
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
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._content_compat().astream(
            super()._astream,  # type: ignore[attr-defined]
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            yield chunk


@functools.cache
def _content_compat_class(cls: type) -> type:
    """The mixin applied to *cls*, built once per base class."""
    return type(f"Evo{cls.__name__}", (OpenAICompatContentMixin, cls), {})


def as_content_compat_model(model: Any, *, hoist_tool_media: bool = True) -> Any:
    """Give *model* the OpenAI-compatible content normalization.

    The normalization is a mixin whose methods read ``self``, so a copy of
    the model sends its own settings: a copy of a model whose methods were
    replaced by closures over the original's bound methods sends the
    original's. The normalizer itself is built once, here, and carried in
    the model's ``__dict__``, so a copy shares what the model it was
    copied from learned about the media its endpoint refuses.
    """
    cls = type(model)
    if getattr(cls, "_generate", None) is None:
        # Not a chat model, or one whose methods live on the instance: a
        # mixin cannot take those over.
        return model
    if not issubclass(cls, OpenAICompatContentMixin):
        model.__class__ = _content_compat_class(cls)
    profile = getattr(model, "profile", None)
    model.__dict__["_evosci_content_compat"] = _OpenAICompatContent(
        profile if isinstance(profile, Mapping) else None,
        hoist_tool_media,
    )
    return model
