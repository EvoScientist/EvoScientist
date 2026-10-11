"""The retry loop: budget as a deadline, backoff, Retry-After, events, per-run key."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt

from EvoScientist.backends import MODEL_RETRY_RUN_KEY
from EvoScientist.llm.errors import ModelErrorFinishError, ProviderStreamError
from EvoScientist.middleware.model_retry import (
    INTERACTIVE_CAP_SECONDS,
    MIN_WAIT_SECONDS,
    UNATTENDED_CAP_SECONDS,
    ModelCallBoundaryMiddleware,
    ModelRetryMiddleware,
)

OK = ModelResponse(
    result=[
        AIMessage("fine", id="lc_run--ok", response_metadata={"finish_reason": "stop"})
    ],
    structured_response=None,
)


class _Carrier(Exception):
    def __init__(self, headers):
        super().__init__("carrier")
        self.response = SimpleNamespace(headers=headers)


def _transient(status=503, headers=None):
    exc = ProviderStreamError(
        "openai", "openai.InternalServerError", "starting up", status_code=status
    )
    if headers is not None:
        exc.__cause__ = _Carrier(headers)
    return exc


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Harness:
    """A middleware with a fake clock and a recording sleep; ``request`` records emitted events."""

    def __init__(
        self, *, budget=60.0, unattended=False, rng=lambda: 0.5, honor_run_key=True
    ):
        self.clock = _Clock()
        self.sleeps: list[float] = []
        self.events: list[dict] = []

        async def _sleep(seconds):
            self.sleeps.append(seconds)
            self.clock.now += seconds

        self.mw = ModelRetryMiddleware(
            wait_budget_seconds=budget,
            unattended=unattended,
            honor_run_key=honor_run_key,
            clock=self.clock,
            sleep=_sleep,
            rng=rng,
        )
        self.request = SimpleNamespace(
            runtime=SimpleNamespace(stream_writer=self.events.append), model=None
        )

    def handler(self, outcomes, *, cost=0.0):
        """Each call pops one outcome (an exception to raise or a response); ``cost`` advances the clock."""
        queue = list(outcomes)

        async def _handler(_request):
            self.clock.now += cost
            outcome = queue.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return _handler


async def test_transient_failure_is_retried_then_succeeds():
    h = _Harness()
    result = await h.mw.awrap_model_call(
        h.request, h.handler([_transient(), _transient(), OK])
    )
    assert result is OK
    assert h.sleeps == [1.0, 2.0]
    assert [e["attempt"] for e in h.events] == [2, 3]
    assert h.events[0]["cause"] == "exception"
    assert "message_id" not in h.events[0]
    assert h.events[0]["reason"] == "503: starting up"
    assert h.events[0]["budget_left_seconds"] == 59.0


async def test_non_transient_error_is_raised_at_once():
    h = _Harness()
    auth = ProviderStreamError(
        "openai", "openai.AuthenticationError", "bad key", status_code=401
    )
    with pytest.raises(ProviderStreamError):
        await h.mw.awrap_model_call(h.request, h.handler([auth, OK]))
    assert h.sleeps == []
    assert h.events == []


async def test_graph_bubble_up_passes_through_untouched():
    h = _Harness()
    with pytest.raises(GraphInterrupt):
        await h.mw.awrap_model_call(h.request, h.handler([GraphInterrupt(), OK]))
    assert h.sleeps == []


async def test_backoff_doubles_with_jitter_and_caps():
    h = _Harness(budget=1000.0, rng=lambda: 1.0)  # +10 % jitter
    outcomes = [_transient()] * 6 + [OK]
    await h.mw.awrap_model_call(h.request, h.handler(outcomes))
    assert h.sleeps == [
        1.1,
        2.2,
        4.4,
        8.8,
        INTERACTIVE_CAP_SECONDS,
        INTERACTIVE_CAP_SECONDS,
    ]


async def test_unattended_tier_caps_at_two_minutes():
    h = _Harness(budget=3600.0, unattended=True)
    outcomes = [_transient()] * 9 + [OK]
    await h.mw.awrap_model_call(h.request, h.handler(outcomes))
    assert h.sleeps[-1] == UNATTENDED_CAP_SECONDS
    assert h.sleeps[0] == 1.0


async def test_deadline_counts_failed_attempt_time_and_gives_up():
    h = _Harness(budget=60.0)
    outcomes = [_transient(), _transient(), _transient(), OK]
    with pytest.raises(ProviderStreamError):
        await h.mw.awrap_model_call(h.request, h.handler(outcomes, cost=25.0))
    # attempts end at t=25, 51, 78: waits of 1 s and 2 s fit, the 4 s wait at t=78 does not
    assert h.sleeps == [1.0, 2.0]
    assert h.clock.now == 78.0


async def test_retry_after_is_honoured_up_to_the_budget_not_the_cap():
    h = _Harness(budget=60.0)
    long_wait = _transient(headers=httpx.Headers({"retry-after": "30"}))
    await h.mw.awrap_model_call(h.request, h.handler([long_wait, OK]))
    assert h.sleeps == [30.0]


async def test_retry_after_beyond_budget_gives_up_now():
    h = _Harness(budget=20.0)
    too_long = _transient(headers=httpx.Headers({"retry-after": "30"}))
    with pytest.raises(ProviderStreamError):
        await h.mw.awrap_model_call(h.request, h.handler([too_long, OK]))
    assert h.sleeps == []
    assert h.events == []


async def test_retry_after_zero_falls_back_to_backoff():
    h = _Harness(budget=60.0)
    zero = _transient(headers=httpx.Headers({"retry-after": "0"}))
    await h.mw.awrap_model_call(h.request, h.handler([zero, zero, OK]))
    assert h.sleeps == [1.0, 2.0]
    assert all(s >= MIN_WAIT_SECONDS for s in h.sleeps)


async def test_zero_budget_calls_once_and_still_rejects_error_finish():
    h = _Harness(budget=0.0)
    with pytest.raises(ProviderStreamError):
        await h.mw.awrap_model_call(h.request, h.handler([_transient(), OK]))
    assert h.sleeps == []
    assert h.events == []
    broken = ModelResponse(
        result=[
            AIMessage(
                "", id="lc_run--bad", response_metadata={"finish_reason": "error"}
            )
        ],
        structured_response=None,
    )
    with pytest.raises(ModelErrorFinishError):
        await h.mw.awrap_model_call(h.request, h.handler([broken]))


async def test_error_finished_response_is_retried_with_finish_reason_cause():
    h = _Harness()
    broken = ModelResponse(
        result=[
            AIMessage(
                "Half", id="lc_run--bad", response_metadata={"finish_reason": "error"}
            )
        ],
        structured_response=None,
    )
    result = await h.mw.awrap_model_call(h.request, h.handler([broken, OK]))
    assert result is OK
    assert h.events[0]["cause"] == "finish_reason"


async def test_error_finish_exception_from_inner_layers_sets_the_cause():
    h = _Harness()
    inner = ModelErrorFinishError(
        AIMessage("", id="lc_run--inner", response_metadata={"finish_reason": "error"})
    )
    wrapped = ProviderStreamError("openrouter", "x", "finish_reason=error")
    wrapped.__cause__ = inner
    await h.mw.awrap_model_call(h.request, h.handler([wrapped, OK]))
    assert h.events[0]["cause"] == "finish_reason"


async def test_run_key_overrides_construction_tier(monkeypatch):
    from EvoScientist.middleware import model_retry as mod

    h = _Harness(budget=0.0, unattended=False)
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {
            "configurable": {
                MODEL_RETRY_RUN_KEY: {"unattended": True, "wait_budget_seconds": 1800}
            }
        },
    )
    outcomes = [_transient()] * 9 + [OK]
    await h.mw.awrap_model_call(h.request, h.handler(outcomes))
    assert h.sleeps[-1] == UNATTENDED_CAP_SECONDS


async def test_run_key_zero_switches_off_and_is_ignored_when_not_honoured(monkeypatch):
    from EvoScientist.middleware import model_retry as mod

    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {
            "configurable": {
                MODEL_RETRY_RUN_KEY: {"unattended": False, "wait_budget_seconds": 0}
            }
        },
    )
    h = _Harness(budget=60.0)
    with pytest.raises(ProviderStreamError):
        await h.mw.awrap_model_call(h.request, h.handler([_transient(), OK]))
    fixed = _Harness(budget=60.0, unattended=True, honor_run_key=False)
    result = await fixed.mw.awrap_model_call(
        fixed.request, fixed.handler([_transient(), OK])
    )
    assert result is OK


def test_sync_path_mirrors_the_async_loop():
    sleeps: list[float] = []
    clock = _Clock()

    def _sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    mw = ModelRetryMiddleware(
        wait_budget_seconds=60.0,
        unattended=False,
        clock=clock,
        sync_sleep=_sleep,
        rng=lambda: 0.5,
    )
    outcomes = [_transient(), OK]

    def handler(_request):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    request = SimpleNamespace(
        runtime=SimpleNamespace(stream_writer=lambda payload: None), model=None
    )
    assert mw.wrap_model_call(request, handler) is OK
    assert sleeps == [1.0]


async def test_warning_lines(caplog):
    h = _Harness(budget=1.5)
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.model_retry"):
        with pytest.raises(ProviderStreamError):
            await h.mw.awrap_model_call(
                h.request, h.handler([_transient(), _transient(), OK])
            )
    messages = [r.getMessage() for r in caplog.records]
    assert any("retrying in 1 s (attempt 2" in m for m in messages)
    assert any("giving up" in m for m in messages)


async def test_wait_that_overruns_the_deadline_gives_up_without_another_attempt(
    caplog,
):
    h = _Harness(budget=5.0)

    async def _suspended_sleep(seconds):
        h.sleeps.append(seconds)
        h.clock.now += seconds + 30.0  # the process was suspended mid-wait

    h.mw._sleep = _suspended_sleep
    calls = 0

    async def _handler(_request):
        nonlocal calls
        calls += 1
        raise _transient()

    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.model_retry"):
        with pytest.raises(ProviderStreamError):
            await h.mw.awrap_model_call(h.request, _handler)
    assert calls == 1
    assert [e["attempt"] for e in h.events] == [2]
    assert "giving up" in caplog.text


# --- the boundary middleware -------------------------------------------------


def _boundary():
    events: list[dict] = []
    request = SimpleNamespace(
        runtime=SimpleNamespace(stream_writer=events.append), model=None
    )
    return ModelCallBoundaryMiddleware(), request, events


async def test_boundary_marks_a_clean_call():
    mw, request, events = _boundary()

    async def _handler(_request):
        return OK

    assert await mw.awrap_model_call(request, _handler) is OK
    assert [e["type"] for e in events] == ["model_call", "model_call"]
    assert [e["phase"] for e in events] == ["start", "end"]
    assert events[0]["call"] == events[1]["call"]
    assert events[1]["status"] == "clean"


async def test_boundary_marks_an_error_finished_call_and_returns_it():
    mw, request, events = _boundary()
    errored = ModelResponse(
        result=[AIMessage("half", response_metadata={"finish_reason": "error"})],
        structured_response=None,
    )

    async def _handler(_request):
        return errored

    assert await mw.awrap_model_call(request, _handler) is errored
    assert events[1]["status"] == "error"


async def test_boundary_marks_a_failed_call_and_reraises():
    mw, request, events = _boundary()

    async def _handler(_request):
        raise ProviderStreamError("openai", "x", "down", status_code=503)

    with pytest.raises(ProviderStreamError):
        await mw.awrap_model_call(request, _handler)
    assert [e["phase"] for e in events] == ["start", "end"]
    assert events[1]["status"] == "failed"


async def test_boundary_call_ids_differ_between_calls():
    mw, request, events = _boundary()

    async def _handler(_request):
        return OK

    await mw.awrap_model_call(request, _handler)
    await mw.awrap_model_call(request, _handler)
    assert events[0]["call"] != events[2]["call"]


def test_boundary_sync_path_mirrors_the_async_one():
    mw, request, events = _boundary()
    with pytest.raises(RuntimeError):
        mw.wrap_model_call(request, lambda _r: (_ for _ in ()).throw(RuntimeError("x")))
    assert mw.wrap_model_call(request, lambda _r: OK) is OK
    assert [(e["phase"], e.get("status")) for e in events] == [
        ("start", None),
        ("end", "failed"),
        ("start", None),
        ("end", "clean"),
    ]


async def test_boundary_without_a_stream_writer_is_silent():
    mw = ModelCallBoundaryMiddleware()
    request = SimpleNamespace(runtime=None, model=None)

    async def _handler(_request):
        return OK

    assert await mw.awrap_model_call(request, _handler) is OK


async def test_retry_event_names_the_agent_it_runs_in(monkeypatch):
    from EvoScientist.middleware import model_retry as mod

    h = _Harness()
    monkeypatch.setattr(
        mod, "get_config", lambda: {"metadata": {"lc_agent_name": "research-agent"}}
    )
    await h.mw.awrap_model_call(h.request, h.handler([_transient(), OK]))
    assert h.events[0]["agent"] == "research-agent"


async def test_retry_event_agent_is_empty_outside_a_run():
    h = _Harness()
    await h.mw.awrap_model_call(h.request, h.handler([_transient(), OK]))
    assert h.events[0]["agent"] == ""
