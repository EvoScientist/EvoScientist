"""Tests for the direct ChatGPT sign-in route. Nothing here touches the network."""

import json
import logging
import os
import threading
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import langchain_openai.chat_models.base as _openai_base
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chatgpt_oauth import (
    _ChatGPTOAuthRefreshError,
    _ChatGPTToken,
    _FileChatGPTOAuthTokenProvider,
)

from EvoScientist.llm import chatgpt_oauth
from EvoScientist.llm.chatgpt_oauth import (
    DEFAULT_INSTRUCTIONS,
    PORT_FORWARD_HINT,
    PORT_IN_USE,
    SIGN_IN_HINT,
    SIGN_IN_INCOMPLETE,
    NotSignedInError,
    _LockedTokenProvider,
    build_model,
    login,
    status,
    token_path,
)
from EvoScientist.middleware.utils import disable_streaming, disable_thinking

RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
REASONING = {"effort": "high", "summary": "auto", "context": "all_turns"}
PICK_SCHEMA = {
    "title": "pick",
    "type": "object",
    "additionalProperties": False,
    "properties": {"tools": {"type": "array", "items": {"type": "string"}}},
    "required": ["tools"],
}


def _token(**overrides):
    fields = {
        "access_token": "tok-access",
        "refresh_token": "tok-refresh",
        "expires_at": datetime.now(UTC) + timedelta(days=1),
        "account_id": "acct_1",
        "plan_type": "plus",
    }
    fields.update(overrides)
    return _ChatGPTToken(**fields)


def _write_token(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _FileChatGPTOAuthTokenProvider(path=path).save(_token())


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """An empty data dir: nobody is signed in."""
    monkeypatch.setattr("EvoScientist.paths.DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def signed_in(data_dir):
    """A data dir holding a valid, unexpired token file."""
    _write_token(token_path())
    return token_path()


def _sse(events) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def _response(output, status="completed"):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-sol",
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def text_stream(text: str = "ok") -> bytes:
    """A Responses stream carrying one assistant text message."""
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return _sse(
        [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": _response([], status="in_progress"),
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {**message, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 3,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 0,
                "item": message,
            },
            {
                "type": "response.completed",
                "sequence_number": 5,
                "response": _response([message]),
            },
        ]
    )


class Wire:
    """Mock transport that records requests and answers with one SSE body."""

    def __init__(self, body: bytes | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.body = body if body is not None else text_stream()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.body
        )

    def clients(self) -> dict:
        transport = httpx.MockTransport(self._handle)
        return {
            "http_client": httpx.Client(transport=transport),
            "http_async_client": httpx.AsyncClient(transport=transport),
        }

    @property
    def body_json(self) -> dict:
        return json.loads(self.requests[-1].content)


# --- request shape -----------------------------------------------------------


def test_request_goes_to_codex_backend_with_our_fields_only(signed_in, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-must-not-leak")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    wire = Wire()
    model = build_model("gpt-6-sol", reasoning=REASONING, **wire.clients())

    reply = model.invoke([SystemMessage("Be terse."), HumanMessage("hi")])

    assert reply.text == "ok"
    request = wire.requests[-1]
    assert str(request.url) == RESPONSES_URL
    assert request.headers["authorization"] == "Bearer tok-access"
    assert request.headers["chatgpt-account-id"] == "acct_1"
    assert request.headers["originator"] == "langchain"
    body = wire.body_json
    assert sorted(body) == [
        "input",
        "instructions",
        "model",
        "reasoning",
        "store",
        "stream",
    ]
    assert body["model"] == "gpt-6-sol"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["instructions"] == "Be terse."
    assert body["reasoning"] == REASONING
    assert [item.get("role") for item in body["input"]] == ["user"]
    assert b"sk-env-must-not-leak" not in request.content
    assert "sk-env-must-not-leak" not in str(request.headers)


def test_default_instructions_only_when_no_system_message(signed_in):
    wire = Wire()
    model = build_model("gpt-6-sol", **wire.clients())

    model.invoke([HumanMessage("hi")])

    assert wire.body_json["instructions"] == DEFAULT_INSTRUCTIONS
    assert DEFAULT_INSTRUCTIONS == "You are a helpful assistant."


# --- copies with streaming disabled (the tool selector's model) --------------


def test_streaming_disabled_copy_still_returns_a_result(signed_in):
    wire = Wire()
    model = build_model("gpt-6-sol", reasoning=REASONING, **wire.clients())
    selector_model = disable_streaming(disable_thinking(model))

    assert selector_model.disable_streaming is True
    assert selector_model.invoke([HumanMessage("hi")]).text == "ok"
    assert wire.body_json["stream"] is True
    assert "reasoning" not in wire.body_json


async def test_streaming_disabled_copy_returns_structured_output(signed_in):
    wire = Wire(text_stream('{"tools": ["ls"]}'))
    model = build_model("gpt-6-sol", reasoning=REASONING, **wire.clients())
    selector_model = disable_streaming(disable_thinking(model))

    picked = await selector_model.with_structured_output(
        PICK_SCHEMA, method="json_schema"
    ).ainvoke([SystemMessage("Pick tools."), HumanMessage("list a folder")])

    assert picked == {"tools": ["ls"]}
    assert wire.body_json["text"]["format"]["type"] == "json_schema"


# --- build_model -------------------------------------------------------------


def test_build_model_without_token_raises_and_creates_nothing(data_dir):
    with pytest.raises(NotSignedInError) as exc_info:
        build_model("gpt-6-sol")

    assert str(exc_info.value) == SIGN_IN_HINT
    assert list(data_dir.iterdir()) == []


def test_build_model_does_not_read_the_token(signed_in, monkeypatch):
    def _boom(self):
        raise AssertionError("token read during construction")

    monkeypatch.setattr(_LockedTokenProvider, "get_token", _boom)

    build_model("gpt-6-sol")


def test_build_model_emits_no_experimental_warning(signed_in, monkeypatch):
    monkeypatch.setattr(
        "langchain_openai.chat_models.codex._experimental_warning_emitted", False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_model("gpt-6-sol")

    assert [w for w in caught if "experimental" in str(w.message)] == []


@pytest.mark.parametrize("model_id", ["gpt-6.1-sol", "gpt-6-luna", "gpt-9-unlisted"])
def test_build_model_uses_the_route_context_window(signed_in, model_id):
    from EvoScientist.llm.context_window import get_context_window

    model = build_model(model_id)

    assert model.profile["max_input_tokens"] == 272_000
    assert get_context_window(model.model_copy()) == 272_000


def test_sign_in_model_keeps_the_route_context_window(
    signed_in, whole_tool_args_patch, monkeypatch
):
    from EvoScientist.llm import get_chat_model

    monkeypatch.setenv("EVOSCIENTIST_OPENAI_AUTH_MODE", "oauth")

    assert (
        get_chat_model("gpt-6.1-sol", provider="openai").profile["max_input_tokens"]
        == 272_000
    )


# --- token provider ----------------------------------------------------------


def test_token_is_read_under_our_lock(signed_in):
    provider = _LockedTokenProvider(signed_in)
    seen = {}
    inner_get = provider._inner.get_token

    def _spy():
        seen["locked"] = provider._lock.is_locked
        return inner_get()

    provider._inner.get_token = _spy

    assert provider.get_token().access_token == "tok-access"
    assert seen["locked"] is True
    assert provider._lock.is_locked is False


def test_lock_file_is_not_the_one_upstream_uses(signed_in):
    provider = _LockedTokenProvider(signed_in)

    assert Path(provider._lock.lock_file).name == "chatgpt.lock"
    assert Path(provider._lock.lock_file).parent == signed_in.parent


def test_missing_token_file_is_not_signed_in(data_dir):
    with pytest.raises(NotSignedInError):
        _LockedTokenProvider(token_path()).get_token()

    assert list(data_dir.iterdir()) == []


def test_unreadable_token_file_is_not_signed_in(signed_in):
    signed_in.write_text("{not json")

    with pytest.raises(NotSignedInError):
        _LockedTokenProvider(signed_in).get_token()


def test_other_refresh_failures_pass_through(signed_in):
    provider = _LockedTokenProvider(signed_in)

    def _network_down():
        raise RuntimeError("OAuth request failed with status 503")

    provider._inner.get_token = _network_down

    with pytest.raises(RuntimeError) as exc_info:
        provider.get_token()

    assert type(exc_info.value) is RuntimeError


@pytest.fixture
def expired(signed_in):
    """A stored token past its expiry, so the next read has to refresh."""
    _FileChatGPTOAuthTokenProvider(path=signed_in).save(
        _token(expires_at=datetime.now(UTC) - timedelta(days=1))
    )
    return signed_in


def _token_endpoint(monkeypatch, handler):
    """Answer upstream's refresh request with ``handler`` instead of the network."""
    import langchain_openai.chatgpt_oauth as upstream

    real_client = httpx.Client

    def _client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(upstream.httpx, "Client", _client)


# The first body is what auth.openai.com answers for a refresh token it rejects.
@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (
            401,
            {
                "error": {
                    "message": "Could not validate your refresh token. "
                    "Please try signing in again.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_refresh_token",
                }
            },
        ),
        (400, {"error": "invalid_grant", "error_description": "Token revoked."}),
    ],
)
def test_refresh_rejected_by_the_token_endpoint_signs_the_user_out(
    expired, monkeypatch, status_code, body
):
    _token_endpoint(monkeypatch, lambda request: httpx.Response(status_code, json=body))

    with pytest.raises(NotSignedInError):
        _LockedTokenProvider(expired).get_token()

    assert not expired.exists()
    assert status().signed_in is False


@pytest.mark.parametrize(
    ("status_code", "text"),
    [
        (500, "try later"),
        (403, "try later"),
        (429, "try later"),
        (500, "upstream said: failed with status 401"),
        (502, "invalid_grant"),
        (
            500,
            "OAuth request to https://auth.openai.com/oauth/token failed with status 401",
        ),
        (500, "ChatGPT refresh token is no longer valid (`invalid_grant`)"),
    ],
)
def test_refresh_failure_that_is_not_a_rejection_keeps_the_token(
    expired, monkeypatch, status_code, text
):
    _token_endpoint(monkeypatch, lambda request: httpx.Response(status_code, text=text))

    with pytest.raises(RuntimeError) as exc_info:
        _LockedTokenProvider(expired).get_token()

    assert not isinstance(exc_info.value, NotSignedInError)
    assert status().signed_in is True


def test_refresh_network_failure_keeps_the_token(expired, monkeypatch):
    def _down(request):
        raise httpx.ConnectError("no route", request=request)

    _token_endpoint(monkeypatch, _down)

    with pytest.raises(httpx.ConnectError):
        _LockedTokenProvider(expired).get_token()

    assert status().signed_in is True


def test_rejected_refresh_keeps_a_token_file_put_there_by_hand(expired, monkeypatch):
    def _copied_in_meanwhile(request):
        # Written directly, as a copy would be: no lock is taken.
        stored = json.loads(expired.read_text())
        expired.write_text(json.dumps({**stored, "refresh_token": "tok-refresh-new"}))
        return httpx.Response(401, json={"error": {"code": "invalid_refresh_token"}})

    _token_endpoint(monkeypatch, _copied_in_meanwhile)

    with pytest.raises(NotSignedInError):
        _LockedTokenProvider(expired).get_token()

    assert status().signed_in is True


def test_rejected_token_that_cannot_be_removed_names_the_file(expired, monkeypatch):
    def _denied(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(chatgpt_oauth.os, "unlink", _denied)
    _token_endpoint(
        monkeypatch,
        lambda request: httpx.Response(
            401, json={"error": {"code": "invalid_refresh_token"}}
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _LockedTokenProvider(expired).get_token()

    assert not isinstance(exc_info.value, NotSignedInError)
    assert str(expired) in str(exc_info.value)
    assert "sign in again" in str(exc_info.value)


@pytest.mark.parametrize(
    "body",
    [
        {"token_type": "Bearer", "expires_in": 3600},
        {"access_token": "new", "expires_in": 0},
        {"access_token": "new", "expires_in": "soon"},
        {"access_token": "new", "expires_in": "invalid_grant"},
    ],
)
def test_a_malformed_refresh_answer_is_not_a_rejection(signed_in, monkeypatch, body):
    """Upstream raises its refresh error for these too; the token may still be good."""
    _FileChatGPTOAuthTokenProvider(path=signed_in).save(
        _token(expires_at=datetime.now(UTC) + timedelta(minutes=2))
    )
    _token_endpoint(monkeypatch, lambda request: httpx.Response(200, json=body))

    with pytest.raises(_ChatGPTOAuthRefreshError):
        _LockedTokenProvider(signed_in).get_token()

    assert status().signed_in is True


def _expiry_out_of_range(good: str) -> str:
    return json.dumps({**json.loads(good), "expires_at": 1e30})


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda good: "[1, 2]",
        lambda good: '"text"',
        lambda good: "null",
        lambda good: "{}",
        _expiry_out_of_range,
    ],
    ids=["list", "string", "null", "empty object", "expiry out of range"],
)
def test_token_file_with_the_wrong_shape_is_not_signed_in(signed_in, rewrite):
    signed_in.write_text(rewrite(signed_in.read_text()))

    with pytest.raises(NotSignedInError):
        _LockedTokenProvider(signed_in).get_token()

    state = status()
    assert (state.signed_in, state.problem) == (False, "token file is unreadable")


def test_a_new_sign_in_replaces_the_token_a_running_provider_holds(signed_in):
    """A long-lived process must not keep the old account until its token expires."""
    provider = _LockedTokenProvider(signed_in)
    assert provider.get_token().account_id == "acct_1"

    _FileChatGPTOAuthTokenProvider(path=signed_in).save(
        _token(access_token="tok-access-2", account_id="acct_2")
    )

    token = provider.get_token()
    assert (token.access_token, token.account_id) == ("tok-access-2", "acct_2")


async def test_async_token_read_runs_off_the_event_loop(signed_in):
    provider = _LockedTokenProvider(signed_in)
    seen = {}
    inner_get = provider._inner.get_token

    def _spy():
        seen["thread"] = threading.get_ident()
        return inner_get()

    provider._inner.get_token = _spy

    assert await provider.aget_access_token() == "tok-access"
    assert seen["thread"] != threading.get_ident()


def test_upstream_no_fcntl_warning_is_dropped(caplog):
    upstream = logging.getLogger("langchain_openai.chatgpt_oauth")

    with caplog.at_level(logging.WARNING, logger="langchain_openai.chatgpt_oauth"):
        upstream.warning("fcntl is unavailable on this platform; token store at x")
        upstream.warning("something else went wrong")

    assert [r.getMessage() for r in caplog.records] == ["something else went wrong"]


# --- status ------------------------------------------------------------------


def test_status_without_token(data_dir):
    state = status()

    assert (state.signed_in, state.plan, state.problem) == (False, None, None)


def test_status_with_token(signed_in):
    state = status()

    assert (state.signed_in, state.plan, state.problem) == (True, "plus", None)


def test_status_with_unreadable_token(signed_in):
    signed_in.write_text("{not json")

    state = status()

    assert (state.signed_in, state.problem) == (False, "token file is unreadable")


# --- login -------------------------------------------------------------------


def test_login_stores_the_token_and_reports_the_plan(data_dir, monkeypatch, capsys):
    calls = []

    def _fake_login(*, store_path, open_browser):
        calls.append((store_path, open_browser))
        _write_token(store_path)

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _fake_login)

    state = login(open_browser=True)

    assert [open_browser for _, open_browser in calls] == [True]
    assert (state.signed_in, state.plan) == (True, "plus")
    assert PORT_FORWARD_HINT not in capsys.readouterr().out
    if os.name == "posix":
        assert token_path().parent.stat().st_mode & 0o777 == 0o700


def _auth_files(path: Path) -> list[str]:
    return sorted(p.name for p in path.parent.iterdir() if not p.name.endswith(".lock"))


def test_login_swaps_the_token_in_under_the_lock_a_refresh_holds(data_dir, monkeypatch):
    """Sign-in and refresh must never write the token file at the same time."""
    from filelock import FileLock, Timeout

    lock_file = str(token_path().with_name("chatgpt.lock"))
    real_replace = os.replace
    written_to = []
    swapped_in = []

    def _fake_login(*, store_path, open_browser):
        written_to.append(store_path)
        _write_token(store_path)

    def _replace(src, dst):
        if Path(dst) == token_path():
            probe = FileLock(lock_file, timeout=0)
            try:
                probe.acquire()
            except Timeout:
                swapped_in.append("lock held")
            else:
                probe.release()
                swapped_in.append("lock free")
        real_replace(src, dst)

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _fake_login)
    monkeypatch.setattr(chatgpt_oauth.os, "replace", _replace)

    assert login(open_browser=True).signed_in is True

    assert written_to != [token_path()]
    assert swapped_in == ["lock held"]
    assert _auth_files(token_path()) == ["chatgpt.json"]
    assert list(token_path().parent.glob("*.new*")) == []


def test_a_failed_sign_in_leaves_another_one_in_flight_alone(data_dir, monkeypatch):
    """Each attempt stores beside the token file under its own name."""

    def _second_attempt_fails(*, store_path, open_browser):
        store_path.write_text("{")
        raise RuntimeError("ChatGPT OAuth callback state mismatch.")

    def _first_attempt(*, store_path, open_browser):
        _write_token(store_path)
        # Another sign-in starts and fails before this one is swapped in.
        monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _second_attempt_fails)
        with pytest.raises(RuntimeError, match="state mismatch"):
            login(open_browser=False)

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _first_attempt)

    assert login(open_browser=False).signed_in is True
    assert _auth_files(token_path()) == ["chatgpt.json"]


def test_login_finds_an_unusable_lock_before_the_browser_step(data_dir, monkeypatch):
    class _Unusable:
        def __enter__(self):
            raise PermissionError(13, "Permission denied")

        def __exit__(self, *exc_info):
            return False

    started = []
    monkeypatch.setattr(chatgpt_oauth, "_token_lock", lambda path: _Unusable())
    monkeypatch.setattr(
        chatgpt_oauth, "login_chatgpt", lambda **kwargs: started.append(kwargs)
    )

    with pytest.raises(PermissionError):
        login(open_browser=True)

    assert started == []


def test_login_succeeds_even_if_its_leftovers_cannot_be_removed(data_dir, monkeypatch):
    real_unlink = Path.unlink

    def _unlink(self, *args, **kwargs):
        if ".new" in self.name:
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(
        chatgpt_oauth,
        "login_chatgpt",
        lambda *, store_path, open_browser: _write_token(store_path),
    )
    monkeypatch.setattr(Path, "unlink", _unlink)

    assert login(open_browser=True).signed_in is True


def test_failed_login_keeps_the_token_already_stored(signed_in, monkeypatch):
    before = signed_in.read_text()

    def _half_done(*, store_path, open_browser):
        store_path.write_text("{")
        raise RuntimeError("ChatGPT OAuth callback state mismatch.")

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _half_done)

    with pytest.raises(RuntimeError, match="state mismatch"):
        login(open_browser=True)

    assert signed_in.read_text() == before
    assert _auth_files(signed_in) == ["chatgpt.json"]


def test_login_without_a_browser_prints_the_port_forward_hint(
    data_dir, monkeypatch, capsys
):
    monkeypatch.setattr(
        chatgpt_oauth,
        "login_chatgpt",
        lambda *, store_path, open_browser: _write_token(store_path),
    )

    login(open_browser=False)

    assert PORT_FORWARD_HINT in capsys.readouterr().out


def test_login_reports_a_busy_callback_port_in_our_words(data_dir, monkeypatch):
    def _port_taken(*, store_path, open_browser):
        try:
            raise OSError(48, "Address already in use")
        except OSError as exc:
            raise RuntimeError("Could not bind ... pass `port=`") from exc

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _port_taken)

    with pytest.raises(RuntimeError) as exc_info:
        login(open_browser=True)

    assert str(exc_info.value) == PORT_IN_USE


@pytest.mark.parametrize(
    "upstream_error",
    [
        _ChatGPTOAuthRefreshError(
            "ChatGPT refresh token is no longer valid (`invalid_grant`). "
            "Re-run `login_chatgpt()` to obtain a new token."
        ),
        RuntimeError(
            "Token response did not include a refresh token and no prior "
            "refresh token was available; re-run `login_chatgpt()`."
        ),
    ],
)
def test_login_never_tells_the_user_to_call_upstreams_function(
    data_dir, monkeypatch, upstream_error
):
    def _fails(*, store_path, open_browser):
        raise upstream_error

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _fails)

    with pytest.raises(RuntimeError) as exc_info:
        login(open_browser=True)

    assert str(exc_info.value) == SIGN_IN_INCOMPLETE


def test_login_passes_other_failures_through(data_dir, monkeypatch):
    def _state_mismatch(*, store_path, open_browser):
        raise RuntimeError("ChatGPT OAuth callback state mismatch.")

    monkeypatch.setattr(chatgpt_oauth, "login_chatgpt", _state_mismatch)

    with pytest.raises(RuntimeError, match="state mismatch"):
        login(open_browser=True)


# --- parallel tool calls delivered whole -------------------------------------


def tool_stream(calls, *, deltas: bool, on_added: int = 0) -> bytes:
    """A Responses stream with function calls.

    ``deltas=False`` mimics the backend for parallel calls: the arguments only
    arrive in ``function_call_arguments.done`` and ``output_item.done``.
    ``on_added`` characters of the arguments already arrive in ``output_item.added``
    (-1 for all of them), as some servers send; the deltas carry the rest.
    """
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _response([], status="in_progress"),
        }
    ]
    items = []
    for index, (name, args) in enumerate(calls):
        arguments = json.dumps(args)
        item = {
            "type": "function_call",
            "id": f"fc_{index}",
            "call_id": f"call_{index}",
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }
        items.append(item)
        events.append(
            {
                "type": "response.output_item.added",
                "sequence_number": len(events),
                "output_index": index,
                "item": {
                    **item,
                    "arguments": arguments[:on_added] if on_added >= 0 else arguments,
                    "status": "in_progress",
                },
            }
        )
        if deltas:
            events.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": len(events),
                    "item_id": f"fc_{index}",
                    "output_index": index,
                    "delta": arguments[on_added:] if on_added >= 0 else "",
                }
            )
        events.append(
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": len(events),
                "item_id": f"fc_{index}",
                "output_index": index,
                "arguments": arguments,
            }
        )
        events.append(
            {
                "type": "response.output_item.done",
                "sequence_number": len(events),
                "output_index": index,
                "item": item,
            }
        )
    events.append(
        {
            "type": "response.completed",
            "sequence_number": len(events),
            "response": _response(items),
        }
    )
    return _sse(events)


CALLS = [
    ("task", {"description": "write a.py", "subagent_type": "code"}),
    ("task", {"description": "write b.md", "subagent_type": "debug"}),
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Delegate a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "subagent_type": {"type": "string"},
                },
                "required": ["description", "subagent_type"],
            },
        },
    }
]


# Captured at import, before any test can install the process-wide patch.
_UPSTREAM_CONVERT = _openai_base._convert_responses_chunk_to_generation_chunk


@pytest.fixture
def whole_tool_args_patch(monkeypatch):
    from EvoScientist.llm.patches import _patch_openai_responses_whole_tool_args

    monkeypatch.setattr(
        _openai_base, "_convert_responses_chunk_to_generation_chunk", _UPSTREAM_CONVERT
    )
    _patch_openai_responses_whole_tool_args()


def _tool_calls(message):
    return [(call["name"], call["args"]) for call in message.tool_calls]


def test_tool_arguments_sent_only_in_done_events_are_kept(
    signed_in, whole_tool_args_patch
):
    wire = Wire(tool_stream(CALLS, deltas=False))
    model = build_model("gpt-6-sol", **wire.clients()).bind_tools(TOOLS)

    assert _tool_calls(model.invoke([HumanMessage("go")])) == CALLS


async def test_tool_arguments_sent_only_in_done_events_are_kept_async(
    signed_in, whole_tool_args_patch
):
    wire = Wire(tool_stream(CALLS, deltas=False))
    model = build_model("gpt-6-sol", **wire.clients()).bind_tools(TOOLS)

    assert _tool_calls(await model.ainvoke([HumanMessage("go")])) == CALLS


def test_streamed_tool_arguments_are_not_doubled(signed_in, whole_tool_args_patch):
    wire = Wire(tool_stream(CALLS, deltas=True))
    model = build_model("gpt-6-sol", **wire.clients()).bind_tools(TOOLS)

    assert _tool_calls(model.invoke([HumanMessage("go")])) == CALLS


@pytest.mark.parametrize(
    ("deltas", "on_added"),
    [(False, -1), (False, 0), (False, 5), (True, 0), (True, 5)],
    ids=[
        "whole-on-added",
        "done-only",
        "part-on-added-rest-in-done",
        "deltas",
        "part-on-added-rest-in-deltas",
    ],
)
def test_streamed_arguments_are_whole_and_not_doubled(
    signed_in, whole_tool_args_patch, deltas, on_added
):
    wire = Wire(tool_stream(CALLS, deltas=deltas, on_added=on_added))
    model = build_model("gpt-6-sol", **wire.clients()).bind_tools(TOOLS)

    message = None
    for chunk in model.stream([HumanMessage("go")]):
        message = chunk if message is None else message + chunk

    # These strings are what the next turn sends back.
    assert [c["args"] for c in message.tool_call_chunks] == [
        json.dumps(args) for _, args in CALLS
    ]


def test_whole_tool_args_patch_is_applied_once(whole_tool_args_patch):
    import langchain_openai.chat_models.base as base

    from EvoScientist.llm.patches import _patch_openai_responses_whole_tool_args

    first = base._convert_responses_chunk_to_generation_chunk
    _patch_openai_responses_whole_tool_args()

    assert base._convert_responses_chunk_to_generation_chunk is first


def test_whole_tool_args_patch_survives_an_upstream_rename(monkeypatch):
    import langchain_openai.chat_models.base as base

    from EvoScientist.llm.patches import _patch_openai_responses_whole_tool_args

    monkeypatch.delattr(base, "_convert_responses_chunk_to_generation_chunk")

    _patch_openai_responses_whole_tool_args()

    assert not hasattr(base, "_convert_responses_chunk_to_generation_chunk")


def test_provider_tag_is_openai(signed_in):
    from EvoScientist.llm.errors import _provider_from_model

    assert _provider_from_model(build_model("gpt-6-sol")) == "openai"


def test_upstream_names_this_module_relies_on_exist():
    from langchain_openai.chat_models.codex import _ChatOpenAICodex
    from langchain_openai.chatgpt_oauth import login_chatgpt

    assert issubclass(chatgpt_oauth.EvoChatOpenAICodex, _ChatOpenAICodex)
    assert callable(login_chatgpt)
    assert hasattr(_FileChatGPTOAuthTokenProvider, "_read_from_disk")
    assert hasattr(_FileChatGPTOAuthTokenProvider(path=Path("unused")), "_cached")
    assert issubclass(_ChatGPTOAuthRefreshError, RuntimeError)


def test_a_call_dropped_from_the_record_is_left_to_upstream(monkeypatch):
    from types import SimpleNamespace

    from EvoScientist.llm import patches

    seen = []
    monkeypatch.setattr(
        _openai_base,
        "_convert_responses_chunk_to_generation_chunk",
        lambda chunk, *a, **k: seen.append(chunk),
    )
    monkeypatch.setattr(patches, "_WHOLE_TOOL_ARGS_RECORD", 2)
    patches._patch_openai_responses_whole_tool_args()
    convert = _openai_base._convert_responses_chunk_to_generation_chunk

    def added(item_id):
        item = SimpleNamespace(type="function_call", id=item_id, arguments='{"x": 1}')
        return SimpleNamespace(type="response.output_item.added", item=item)

    for item_id in ("a", "b", "c"):  # "a" falls out of a record of two
        convert(added(item_id))
    done = SimpleNamespace(
        type="response.function_call_arguments.done",
        item_id="a",
        arguments='{"x": 1}',
        output_index=0,
        sequence_number=9,
    )
    convert(done)

    assert seen[-1] is done


def test_a_reopened_call_starts_a_fresh_record(monkeypatch):
    from types import SimpleNamespace

    from EvoScientist.llm import patches

    seen = []
    monkeypatch.setattr(
        _openai_base,
        "_convert_responses_chunk_to_generation_chunk",
        lambda chunk, *a, **k: seen.append(chunk),
    )
    monkeypatch.setattr(patches, "_WHOLE_TOOL_ARGS_RECORD", 1)
    patches._patch_openai_responses_whole_tool_args()
    convert = _openai_base._convert_responses_chunk_to_generation_chunk

    def added(item_id, arguments):
        item = SimpleNamespace(type="function_call", id=item_id, arguments=arguments)
        return SimpleNamespace(type="response.output_item.added", item=item)

    convert(added("a", '{"x": 1}'))
    convert(added("b", '{"y": 2}'))  # pushes "a" out; its stream is then cut
    convert(added("a", ""))  # a later stream reuses the id
    convert(
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="a",
            arguments='{"z": 3}',
            output_index=0,
            sequence_number=9,
        )
    )

    assert seen[-1].type == "response.function_call_arguments.delta"
    assert seen[-1].delta == '{"z": 3}'


def test_later_deltas_of_a_dropped_call_are_not_recorded(monkeypatch):
    from types import SimpleNamespace

    from EvoScientist.llm import patches

    seen = []
    monkeypatch.setattr(
        _openai_base,
        "_convert_responses_chunk_to_generation_chunk",
        lambda chunk, *a, **k: seen.append(chunk),
    )
    monkeypatch.setattr(patches, "_WHOLE_TOOL_ARGS_RECORD", 2)
    patches._patch_openai_responses_whole_tool_args()
    convert = _openai_base._convert_responses_chunk_to_generation_chunk

    def added(item_id, arguments):
        item = SimpleNamespace(type="function_call", id=item_id, arguments=arguments)
        return SimpleNamespace(type="response.output_item.added", item=item)

    def delta(item_id, text):
        return SimpleNamespace(
            type="response.function_call_arguments.delta", item_id=item_id, delta=text
        )

    convert(added("a", '{"s": "'))
    convert(added("b", '{"y": 2}'))
    convert(added("c", '{"z": 3}'))  # pushes "a" out
    convert(delta("a", "{"))
    convert(delta("b", "}"))
    done = SimpleNamespace(
        type="response.function_call_arguments.done",
        item_id="a",
        arguments='{"s": "{"}',
        output_index=0,
        sequence_number=9,
    )
    convert(done)

    assert seen[-1] is done
