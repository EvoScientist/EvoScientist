"""Transient-failure classification on exceptions real provider SDKs raise."""

from __future__ import annotations

import asyncio
import json
import ssl
from types import SimpleNamespace

import aiohttp
import httpx
import httpx2
import pytest
import requests
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from EvoScientist.middleware.error_normalization import ErrorNormalizationMiddleware
from EvoScientist.middleware.model_retry import (
    describe_failure,
    has_error_finish,
    is_transient,
    retry_after_seconds,
)

SSE = {"content-type": "text/event-stream"}


class _Req:
    def __init__(self, model):
        self.model = model


class _Stream2(httpx2.AsyncByteStream):
    def __init__(self, chunks, exc=None):
        self._chunks, self._exc = chunks, exc

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._exc:
            raise self._exc

    async def aclose(self):
        pass


class _Stream1(httpx.AsyncByteStream):
    def __init__(self, chunks, exc=None):
        self._chunks, self._exc = chunks, exc

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._exc:
            raise self._exc

    async def aclose(self):
        pass


def _oa_chunk(text: str) -> bytes:
    body = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    return ("data: " + json.dumps(body) + "\n\n").encode()


def _openai(handler, *, streaming=True):
    return ChatOpenAI(
        model="m",
        api_key="sk-test-xxxxxxxxxxxxxxxx",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        streaming=streaming,
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _anthropic(handler):
    import anthropic

    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key="sk-ant-test-xxxxxxxx",
        max_retries=0,
        streaming=True,
    )
    model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
        api_key="sk-ant-test-xxxxxxxx",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return model


def _openrouter(handler, *, streaming=True):
    import openrouter
    import openrouter.utils as oru
    from langchain_openrouter import ChatOpenRouter

    model = ChatOpenRouter(
        model="anthropic/claude-haiku-5.5",
        api_key="sk-or-test-xxxxxxxxxx",
        max_retries=0,
        streaming=streaming,
    )
    model.client = openrouter.OpenRouter(
        retry_config=oru.RetryConfig("none", oru.BackoffStrategy(1, 1, 1, 1), False),
        api_key="sk-or-test-xxxxxxxxxx",
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return model


_AN_HEAD = [
    (
        "event: message_start\ndata: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "c",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        )
        + "\n\n"
    ).encode()
]


async def _astream_all(model):
    async for _ in model.astream("hi"):
        pass


async def _ainvoke(model):
    await model.ainvoke("hi")


async def _through_normalization(model, how):
    """The exception the retry layer sees: raised by the SDK and normalized once."""
    mw = ErrorNormalizationMiddleware()

    async def handler(_request):
        return await how(model)

    try:
        await mw.awrap_model_call(_Req(model), handler)
    except Exception as exc:
        return exc
    return None


def _raise(exc):
    def handler(request):
        raise exc.__class__(str(exc), request=request)

    return handler


def _json(status, body):
    return lambda r: httpx2.Response(status, json=body)


def _sse(chunks, exc=None):
    return lambda r: httpx2.Response(200, headers=SSE, stream=_Stream2(chunks, exc))


def _sse1(chunks, exc=None):
    return lambda r: httpx.Response(200, headers=SSE, stream=_Stream1(chunks, exc))


def _error_chunk(code):
    return (
        "data: "
        + json.dumps({"error": {"code": code, "message": "Provider returned error"}})
        + "\n\n"
    ).encode()


_OR_NO_CHOICES = (
    "data: "
    + json.dumps(
        {
            "id": "g",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "error": {"code": 502, "message": "Provider returned error"},
        }
    )
    + "\n\n"
).encode()

CASES = [
    (
        "openai 503 cold start",
        True,
        lambda: _openai(_json(503, {"error": {"message": "starting up", "code": 503}})),
        _astream_all,
    ),
    (
        "openai 502 html",
        True,
        lambda: _openai(
            lambda r: httpx2.Response(502, text="<html>502 Proxy Error</html>")
        ),
        _astream_all,
    ),
    (
        "openai 429",
        True,
        lambda: _openai(_json(429, {"error": {"message": "slow down"}})),
        _astream_all,
    ),
    (
        "openai 429 insufficient_quota",
        False,
        lambda: _openai(
            _json(
                429,
                {
                    "error": {
                        "message": "quota",
                        "code": "insufficient_quota",
                        "type": "insufficient_quota",
                    }
                },
            )
        ),
        _astream_all,
    ),
    (
        "openai 401",
        False,
        lambda: _openai(_json(401, {"error": {"message": "bad key"}})),
        _astream_all,
    ),
    (
        "openai 400 context",
        False,
        lambda: _openai(
            _json(
                400,
                {
                    "error": {
                        "message": "maximum context length exceeded",
                        "code": "context_length_exceeded",
                    }
                },
            )
        ),
        _astream_all,
    ),
    (
        "openai connect refused",
        True,
        lambda: _openai(_raise(httpx2.ConnectError("refused"))),
        _astream_all,
    ),
    (
        "openai mid-stream disconnect",
        True,
        lambda: _openai(
            _sse([_oa_chunk("Half")], httpx2.RemoteProtocolError("peer closed"))
        ),
        _astream_all,
    ),
    (
        "openai SSE error code 502",
        True,
        lambda: _openai(_sse([_oa_chunk("Half"), _error_chunk(502)])),
        _astream_all,
    ),
    (
        "openai SSE error code '502'",
        True,
        lambda: _openai(_sse([_oa_chunk("Half"), _error_chunk("502")])),
        _astream_all,
    ),
    (
        "openai non-stream 200 + error body",
        True,
        lambda: _openai(
            _json(200, {"error": {"code": 502, "message": "Provider returned error"}}),
            streaming=False,
        ),
        _ainvoke,
    ),
    (
        "openai UnsupportedProtocol wrapped retryable",
        False,
        lambda: _openai(_raise(httpx2.UnsupportedProtocol("ftp not supported"))),
        _astream_all,
    ),
    (
        "anthropic 529",
        True,
        lambda: _anthropic(
            _json(
                529,
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
            )
        ),
        _astream_all,
    ),
    (
        "anthropic mid-stream overloaded",
        True,
        lambda: _anthropic(
            _sse(
                [
                    *_AN_HEAD,
                    b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n',
                ]
            )
        ),
        _astream_all,
    ),
    (
        "anthropic mid-stream invalid_request",
        False,
        lambda: _anthropic(
            _sse(
                [
                    *_AN_HEAD,
                    b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error","message":"bad"}}\n\n',
                ]
            )
        ),
        _astream_all,
    ),
    (
        "anthropic mid-stream disconnect",
        True,
        lambda: _anthropic(_sse(_AN_HEAD, httpx2.RemoteProtocolError("peer closed"))),
        _astream_all,
    ),
    (
        "openrouter 503",
        True,
        lambda: _openrouter(
            lambda r: httpx.Response(
                503,
                headers={"content-type": "application/json"},
                json={"error": {"code": 503, "message": "down"}},
            )
        ),
        _astream_all,
    ),
    (
        "openrouter mid-stream disconnect",
        True,
        lambda: _openrouter(
            _sse1([_oa_chunk("Half")], httpx.RemoteProtocolError("peer closed"))
        ),
        _astream_all,
    ),
    (
        "openrouter error chunk without choices",
        True,
        lambda: _openrouter(_sse1([_oa_chunk("Half"), _OR_NO_CHOICES])),
        _astream_all,
    ),
]


@pytest.mark.parametrize(
    ("label", "expected", "make", "how"), CASES, ids=[c[0] for c in CASES]
)
def test_classification_on_real_sdk_exceptions(label, expected, make, how):
    exc = asyncio.run(_through_normalization(make(), how))
    assert exc is not None, label
    assert is_transient(exc) is expected, f"{label}: {type(exc).__name__}: {exc}"


def _ssl_key():
    from aiohttp.client_reqrep import ConnectionKey

    return ConnectionKey("h", 443, True, True, None, None, None)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: requests.exceptions.ReadTimeout("read timed out"), True),
        (lambda: requests.exceptions.ConnectionError("connection reset"), True),
        (lambda: requests.exceptions.SSLError("certificate verify failed"), False),
        (lambda: aiohttp.ServerDisconnectedError(), True),
        (lambda: aiohttp.ServerTimeoutError("timeout"), True),
        (lambda: aiohttp.ClientSSLError(_ssl_key(), ssl.SSLError(1, "bad")), False),
        (
            lambda: aiohttp.ClientConnectorCertificateError(
                _ssl_key(), ssl.SSLCertVerificationError(1, "self-signed")
            ),
            False,
        ),
        (lambda: aiohttp.InvalidURL("not a url"), False),
        (lambda: requests.exceptions.InvalidURL("not a url"), False),
    ],
    ids=[
        "requests-timeout",
        "requests-reset",
        "requests-ssl",
        "aiohttp-disconnect",
        "aiohttp-timeout",
        "aiohttp-ssl",
        "aiohttp-certificate",
        "aiohttp-invalid-url",
        "requests-invalid-url",
    ],
)
def test_the_nvidia_clients_transport_errors(make, expected):
    """The NVIDIA client raises these raw, past the normalization layer."""
    exc = make()

    async def _how(_model):
        raise exc

    normalized = asyncio.run(_through_normalization(SimpleNamespace(), _how))
    assert is_transient(normalized) is expected


def test_an_implicit_context_is_not_followed():
    # An implicit __context__ is not evidence: a chain's earlier failures
    # count only when the chain lists them itself.
    try:
        try:
            raise httpx.ConnectError("refused")
        except httpx.ConnectError:
            raise PermissionError("fallback rejected")  # noqa: B904 - implicit context on purpose
    except PermissionError as exc:
        chained = exc
    from EvoScientist.llm.errors import ProviderStreamError

    outer = ProviderStreamError(
        "openai", "openai.AuthenticationError", "rejected", status_code=None
    )
    outer.__cause__ = chained
    assert is_transient(outer) is False


def test_graph_bubble_up_and_context_overflow_are_never_transient():
    from langchain_core.exceptions import ContextOverflowError
    from langgraph.errors import GraphInterrupt

    assert is_transient(GraphInterrupt()) is False
    assert is_transient(ContextOverflowError("too long")) is False


def test_structured_output_error_with_error_finished_message_is_transient():
    from langchain.agents.structured_output import StructuredOutputValidationError
    from langchain_core.messages import AIMessage

    errored = AIMessage(
        content="", id="lc_run--g1", response_metadata={"finish_reason": "error"}
    )
    exc = StructuredOutputValidationError(
        "GraderResponse", ValueError("no json"), errored
    )
    assert is_transient(exc) is True
    assert has_error_finish(exc) is True
    fine = AIMessage(
        content="{}", id="lc_run--g2", response_metadata={"finish_reason": "stop"}
    )
    assert (
        is_transient(
            StructuredOutputValidationError("GraderResponse", ValueError("bad"), fine)
        )
        is False
    )


class _Carrier(Exception):
    """An exception carrying whatever response attributes a test needs."""

    def __init__(self, **attrs):
        super().__init__("carrier")
        self.__dict__.update(attrs)


def _with_headers(**headers):
    return _Carrier(response=SimpleNamespace(headers=httpx.Headers(headers)))


def test_retry_after_header_locations_and_forms():
    assert retry_after_seconds(_with_headers(**{"retry-after": "7"})) == 7.0
    assert retry_after_seconds(_with_headers(**{"retry-after-ms": "2500"})) == 2.5
    assert retry_after_seconds(_with_headers(**{"retry-after-ms": "2500.0"})) == 2.5
    assert (
        retry_after_seconds(
            _with_headers(**{"retry-after-ms": "bad", "retry-after": "30"})
        )
        == 30.0
    )
    assert retry_after_seconds(_with_headers(**{"retry-after": "0"})) is None
    assert (
        retry_after_seconds(
            _with_headers(**{"retry-after-ms": "0", "retry-after": "5"})
        )
        == 5.0
    )
    assert (
        retry_after_seconds(
            _with_headers(**{"retry-after": "Thu, 01 Jan 1970 00:00:00 GMT"})
        )
        is None
    )
    assert retry_after_seconds(_with_headers(**{"retry-after": "soon"})) is None
    assert (
        retry_after_seconds(_Carrier(headers=httpx.Headers({"retry-after": "3"})))
        == 3.0
    )
    assert (
        retry_after_seconds(
            _Carrier(
                raw_response=SimpleNamespace(
                    headers=httpx.Headers({"retry-after": "4"})
                )
            )
        )
        == 4.0
    )
    assert retry_after_seconds(RuntimeError("nothing")) is None


def test_describe_failure_prefers_normalized_fields():
    from EvoScientist.llm.errors import ProviderStreamError

    exc = ProviderStreamError(
        "openai", "openai.InternalServerError", "starting up", status_code=503
    )
    assert describe_failure(exc) == "503: starting up"
    assert describe_failure(RuntimeError("x" * 300)).endswith("...")


def test_body_level_4xx_is_excluded_before_any_positive_signal():
    """A deterministic status inside a 200 body or an OpenRouter message wins over ``server_error``."""
    assert (
        is_transient(
            ValueError({"code": 401, "type": "server_error", "message": "key"})
        )
        is False
    )
    assert is_transient(ValueError({"code": "403", "type": "server_error"})) is False
    assert is_transient(ValueError({"code": 503, "type": "server_error"})) is True
    assert is_transient(ValueError("Provider returned error (code: 403)")) is False
    assert is_transient(ValueError("Provider returned error (code: 502)")) is True
    assert (
        is_transient(ValueError({"code": "not-a-status", "type": "server_error"}))
        is True
    )
