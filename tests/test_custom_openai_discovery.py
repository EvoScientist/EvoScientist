"""Tests for EvoScientist.llm.custom_openai_discovery.

Covers the sync ``list_custom_openai_models`` (onboarding wizard) and the
async ``discover_custom_openai_models`` (model pickers). Both must never
raise — the picker falls back to the static defaults when the endpoint is
down or misbehaving. The key is sent only when one was configured; a keyless
local endpoint is still probed, and ``OPENAI_API_KEY`` is never attached.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
from unittest.mock import Mock, patch

import httpx
import pytest

from EvoScientist.llm import custom_openai_discovery as discovery
from EvoScientist.llm.custom_openai_discovery import (
    discover_custom_openai_models,
    list_custom_openai_models,
)

_LISTING = {"data": [{"id": "fake-agent"}, {"id": "other-model"}]}


def _response(status_code: int = 200, body: object = None, *, json_error: bool = False):
    content = (
        b"{not json"
        if json_error
        else json.dumps(body if body is not None else _LISTING).encode()
    )
    return httpx.Response(status_code, stream=httpx.ByteStream(content))


def _patch_transport(handler):
    client = httpx.AsyncClient
    return patch.object(
        httpx,
        "AsyncClient",
        side_effect=lambda **kwargs: client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )


def _patch_request(response=None, *, side_effect=None):
    seen: dict[str, object] = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        if side_effect is not None:
            raise side_effect
        return response

    return _patch_transport(handler), seen


class TestListCustomOpenaiModels:
    """Sync probe - the wizard calls this while questionary is blocking."""

    def test_empty_base_url_returns_empty_without_http(self):
        patcher, seen = _patch_request(_response())
        with patcher:
            assert list_custom_openai_models("") == []
            assert list_custom_openai_models(None) == []
        assert seen == {}

    def test_200_returns_ids(self):
        patcher, seen = _patch_request(_response())
        with patcher:
            names = list_custom_openai_models("https://llm.test/v1", "sk-test")
        assert names == ["fake-agent", "other-model"]
        assert seen["url"] == "https://llm.test/v1/models"
        assert seen["authorization"] == "Bearer sk-test"

    def test_empty_key_sends_no_authorization(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-sho...leak")
        patcher, seen = _patch_request(_response())
        with patcher:
            names = list_custom_openai_models("https://llm.test/v1", "")
        assert names == ["fake-agent", "other-model"]
        assert seen["authorization"] is None

    def test_whitespace_key_sends_no_authorization(self):
        patcher, seen = _patch_request(_response())
        with patcher:
            list_custom_openai_models("https://llm.test/v1", "   ")
        assert seen["authorization"] is None

    def test_non_200_returns_empty(self):
        for status in (401, 404, 500):
            patcher, _ = _patch_request(_response(status))
            with patcher:
                assert list_custom_openai_models("https://llm.test/v1", "sk") == []

    def test_timeout_returns_empty(self):
        patcher, _ = _patch_request(side_effect=httpx.TimeoutException("timed out"))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_connect_error_returns_empty(self):
        patcher, _ = _patch_request(side_effect=httpx.ConnectError("refused"))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_malformed_json_returns_empty(self):
        patcher, _ = _patch_request(_response(json_error=True))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_missing_data_returns_empty(self):
        patcher, _ = _patch_request(_response(body={"unexpected": []}))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_data_not_a_list_returns_empty(self):
        patcher, _ = _patch_request(_response(body={"data": "nope"}))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_drops_entries_without_id(self):
        body = {"data": [{"id": "fake-agent"}, {"id": ""}, {}, "bare"]}
        patcher, _ = _patch_request(_response(body=body))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == ["fake-agent"]

    def test_trailing_slash_stripped_from_url(self):
        patcher, seen = _patch_request(_response(body={"data": []}))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1/") == []
        assert seen["url"] == "https://llm.test/v1/models"

    def test_a_timed_out_probe_leaves_no_worker_running(self):
        """An endpoint that never answers does not leave a thread behind.

        The deadline cancels the request, so the worker ends with it. A probe
        that only stopped waiting would leave the thread and its connection
        behind, and every further probe would leave another one.
        """
        workers: list[threading.Thread] = []

        async def never_answers(request):
            workers.append(threading.current_thread())
            await asyncio.sleep(3600)

        with _patch_transport(never_answers):
            assert list_custom_openai_models("https://llm.test/v1", timeout=0.2) == []

        assert len(workers) == 1
        workers[0].join(timeout=1)
        assert not workers[0].is_alive()


class TestDiscoverCustomOpenaiModels:
    """Async probe — contract: never raise, return list[str]."""

    async def test_empty_base_url_returns_empty_without_http(self):
        names = await discover_custom_openai_models("")
        assert names == []

    async def test_none_base_url_returns_empty(self):
        names = await discover_custom_openai_models(None)
        assert names == []

    async def test_200_returns_ids_and_sends_key(self):
        patcher, seen = _patch_request(_response())
        with patcher:
            names = await discover_custom_openai_models(
                "https://llm.test/v1", "sk-test"
            )
        assert names == ["fake-agent", "other-model"]
        assert seen == {
            "url": "https://llm.test/v1/models",
            "authorization": "Bearer sk-test",
        }

    async def test_empty_key_sends_no_authorization(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "unused-openai-key")
        patcher, seen = _patch_request(_response())
        with patcher:
            names = await discover_custom_openai_models("https://llm.test/v1", "")
        assert names == ["fake-agent", "other-model"]
        assert seen["authorization"] is None

    @pytest.mark.parametrize("status", [401, 404, 500])
    async def test_non_200_returns_empty(self, status):
        patcher, _ = _patch_request(_response(status))
        with patcher:
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    @pytest.mark.parametrize(
        "error", [httpx.TimeoutException("timed out"), httpx.ConnectError("refused")]
    )
    async def test_http_errors_return_empty(self, error):
        patcher, _ = _patch_request(side_effect=error)
        with patcher:
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    @pytest.mark.parametrize(
        "response_kwargs",
        [
            {"json_error": True},
            {"body": {"object": "list"}},
            {"body": [{"id": "fake-agent"}]},
        ],
    )
    async def test_malformed_listing_returns_empty(self, response_kwargs):
        patcher, _ = _patch_request(_response(**response_kwargs))
        with patcher:
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_trailing_slash_stripped_from_url(self):
        patcher, seen = _patch_request(_response(body={"data": []}))
        with patcher:
            await discover_custom_openai_models("https://llm.test/v1/")
        assert seen["url"] == "https://llm.test/v1/models"

    async def test_a_parse_that_crosses_the_deadline_still_returns(self, monkeypatch):
        """A parse that ends past the deadline is still a result.

        The parse runs inline and cannot be interrupted, so once it has
        finished the listing is known; discarding it would drop a valid
        listing for no saving.
        """
        loop = asyncio.get_running_loop()
        clock = [100.0]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        original = discovery._picker_names

        def slow_parse(body):
            names = original(body)
            # The parse ends just after the deadline the caller set.
            clock[0] += 5.0
            return names

        monkeypatch.setattr(discovery, "_picker_names", slow_parse)
        patcher, _ = _patch_request(_response())
        with patcher:
            names = await discover_custom_openai_models("https://llm.test/v1")
        assert names == ["fake-agent", "other-model"]


async def test_settings_and_picker_list_the_same_normalized_ids(monkeypatch):
    import openai

    from EvoScientist.config.provider_keys import check_custom_openai

    body = {"data": [{"id": " first "}, {"id": "first"}, {"id": "  "}]}
    body["data"].extend({"id": f"model-{i}"} for i in range(510))
    # Both paths strip and de-duplicate the same way; the picker shows at
    # most 500 rows, and the key check reports what the endpoint listed.
    listed = ["first", *(f"model-{i}" for i in range(510))]
    expected = listed[:500]
    url = "https://llm.test/v1"

    def list_models(request):
        assert str(request.url) == f"{url}/models"
        assert request.headers["Authorization"] == "Bearer sk-test"
        return _response(body=body)

    sdk_client = openai.OpenAI
    transport = httpx.MockTransport(list_models)

    def make_client(**kwargs):
        return sdk_client(http_client=httpx.Client(transport=transport), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", make_client)
    with _patch_transport(list_models):
        discovered = await discover_custom_openai_models(url, "sk-test")
    checked = check_custom_openai("sk-test", url)

    assert discovered == expected
    assert checked.status == "valid"
    assert list(checked.models) == listed


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("declared_length", [True, False])
async def test_oversized_listing_stops_before_reading_the_rest(
    monkeypatch, caplog, declared_length
):
    content = json.dumps(_LISTING).encode()
    # The complete payload is valid JSON; size alone must keep it from parsing.
    stream = _TrackedStream([content[:16], content[16:], b" "])
    headers = {"Content-Length": str(len(content) + 1)} if declared_length else {}
    monkeypatch.setattr(discovery, "_MAX_BODY_BYTES", 16)
    parse = Mock(wraps=discovery._names_from_body)
    monkeypatch.setattr(discovery, "_names_from_body", parse)
    patcher, seen = _patch_request(httpx.Response(200, stream=stream, headers=headers))
    with patcher, caplog.at_level(logging.DEBUG, logger=discovery.__name__):
        assert await discover_custom_openai_models("https://llm.test/v1") == []

    assert seen["url"] == "https://llm.test/v1/models"
    # The read stops at the chunk that crosses the limit, so the rest of the
    # body is never read.
    assert stream.reads == 2
    assert stream.closed
    parse.assert_not_called()
    assert any(
        record.levelno == logging.DEBUG and record.name == discovery.__name__
        for record in caplog.records
    )


async def test_reads_a_compressed_listing(monkeypatch):
    """A listing served compressed is read, not skipped.

    A server that compresses the body still serves the models, and the
    limit applies to the decoded bytes -- the ones that get parsed.
    """
    content = json.dumps(_LISTING).encode()
    compressed = gzip.compress(content)
    stream = _TrackedStream([compressed])

    def compressed_listing(request):
        return httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})

    with _patch_transport(compressed_listing):
        assert await discover_custom_openai_models("https://llm.test/v1") == [
            "fake-agent",
            "other-model",
        ]
    assert stream.closed


async def test_a_compressed_listing_past_the_limit_is_dropped(monkeypatch):
    content = json.dumps(_LISTING).encode()
    monkeypatch.setattr(discovery, "_MAX_BODY_BYTES", len(content) - 1)
    stream = _TrackedStream([gzip.compress(content)])

    def compressed_listing(request):
        return httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})

    with _patch_transport(compressed_listing):
        assert await discover_custom_openai_models("https://llm.test/v1") == []
    assert stream.closed


async def test_accepts_a_listing_at_the_byte_limit(monkeypatch):
    content = json.dumps(_LISTING).encode()
    monkeypatch.setattr(discovery, "_MAX_BODY_BYTES", len(content))
    stream = _TrackedStream([content[:10], content[10:]])
    patcher, _ = _patch_request(httpx.Response(200, stream=stream))
    with patcher:
        assert await discover_custom_openai_models("https://llm.test/v1") == [
            "fake-agent",
            "other-model",
        ]
    assert stream.closed


def test_a_late_success_cannot_mutate_the_returned_list(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []
    late_result: list[str] = []

    async def late_listing(*args, **kwargs):
        workers.append(threading.current_thread())
        started.set()
        if release.wait(timeout=5):
            late_result.append("late")
        return late_result

    monkeypatch.setattr(discovery, "discover_custom_openai_models", late_listing)
    try:
        names = list_custom_openai_models("https://llm.test/v1", timeout=0)
        assert started.wait(timeout=1)
        assert workers[0].is_alive()
        assert names == []
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=2)

    assert not workers[0].is_alive()
    assert late_result == ["late"]
    assert names == []
