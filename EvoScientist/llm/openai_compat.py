"""Reusable behavior for OpenAI-compatible chat model integrations."""

from __future__ import annotations

import functools
import weakref
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
        profile = getattr(self, "profile", None)
        effective = profile if isinstance(profile, Mapping) else None
        compat = self.__dict__.get("_evosci_content_compat")
        owner = self.__dict__.get("_evosci_content_compat_owner")
        # A copy carries both, so the owner tells them apart: what one model
        # learns about the media its endpoint refuses must not reach the model
        # it was copied from.
        if (
            not isinstance(compat, _OpenAICompatContent)
            or not isinstance(owner, weakref.ReferenceType)
            or owner() is not self
        ):
            compat = _OpenAICompatContent(
                effective,
                hoist_tool_media=self.__dict__.get("_evosci_hoist_tool_media", True),
            )
            self.__dict__["_evosci_content_compat"] = compat
            self.__dict__["_evosci_content_compat_owner"] = weakref.ref(self)
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
    original's.
    """
    cls = type(model)
    if getattr(cls, "_generate", None) is None:
        # Not a chat model, or one whose methods live on the instance: a
        # mixin cannot take those over.
        return model
    if not issubclass(cls, OpenAICompatContentMixin):
        model.__class__ = _content_compat_class(cls)
    model.__dict__["_evosci_hoist_tool_media"] = hoist_tool_media
    compat = model.__dict__.get("_evosci_content_compat")
    owner = model.__dict__.get("_evosci_content_compat_owner")
    if (
        isinstance(compat, _OpenAICompatContent)
        and isinstance(owner, weakref.ReferenceType)
        and owner() is model
    ):
        # Re-applying with a different flag must reach a normalizer that is
        # already serving this model's requests. A copy shares the original's
        # normalizer until its first request, and must leave it alone.
        compat.hoist_tool_media = hoist_tool_media
    return model
