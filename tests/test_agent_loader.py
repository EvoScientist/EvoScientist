"""Tests for ``cli/_agent_loader``."""

from __future__ import annotations

import asyncio

import pytest

from EvoScientist.cli._agent_loader import BackgroundAgentLoader, MCPProgressTracker

# ──────────────────────────────────────────────────────────────────────
# MCPProgressTracker
# ──────────────────────────────────────────────────────────────────────


class TestMCPProgressTracker:
    def test_prime_empty_when_no_config(self, monkeypatch):
        import EvoScientist.mcp as mcp_pkg

        monkeypatch.setattr(mcp_pkg, "load_mcp_config", lambda: {})
        t = MCPProgressTracker()
        t.prime()
        assert t.progress == {}

    def test_prime_seeds_pending_entries(self, monkeypatch):
        import EvoScientist.mcp as mcp_pkg

        monkeypatch.setattr(mcp_pkg, "load_mcp_config", lambda: {"a": {}, "b": {}})
        t = MCPProgressTracker()
        t.prime()
        assert t.progress == {"a": ("pending", ""), "b": ("pending", "")}

    def test_prime_swallows_config_errors(self, monkeypatch):
        import EvoScientist.mcp as mcp_pkg

        def _boom():
            raise RuntimeError("config broken")

        monkeypatch.setattr(mcp_pkg, "load_mcp_config", _boom)
        t = MCPProgressTracker()
        t.prime()
        assert t.progress == {}

    def test_record_maps_events(self):
        t = MCPProgressTracker()
        assert t.record("start", "srv", "") == "pending"
        assert t.record("success", "srv", "5") == "ok"
        assert t.record("error", "srv", "timeout") == "error"
        assert t.record("bogus", "srv", "") is None
        assert t.progress == {"srv": ("error", "timeout")}

    def test_start_does_not_overwrite_existing_state(self):
        t = MCPProgressTracker()
        t.record("success", "srv", "3")
        t.record("start", "srv", "")
        assert t.progress["srv"] == ("ok", "3")

    def test_snapshot_is_independent_copy(self):
        t = MCPProgressTracker()
        t.record("success", "srv", "1")
        snap = t.snapshot()
        t.record("error", "srv", "oops")
        assert snap == [("ok", "1")]

    def test_totals(self):
        t = MCPProgressTracker()
        t.record("start", "a", "")
        t.record("success", "b", "1")
        t.record("error", "c", "boom")
        done, total = t.totals()
        assert (done, total) == (2, 3)


# ──────────────────────────────────────────────────────────────────────
# BackgroundAgentLoader
# ──────────────────────────────────────────────────────────────────────


def _make_loader_fn(agent_value="AGENT", fail_with=None, capture=None):
    """Build a sync loader that records ``on_mcp_progress`` + kwargs."""

    def _loader(*, on_mcp_progress=None, **kwargs):
        if capture is not None:
            capture["on_mcp_progress"] = on_mcp_progress
            capture.setdefault("kwargs", []).append(kwargs)
            if on_mcp_progress is not None:
                on_mcp_progress("start", "srv", "")
                on_mcp_progress("success", "srv", "1")
        if fail_with is not None:
            raise fail_with
        return agent_value

    return _loader


def _run(coro):
    return asyncio.run(coro)


class TestBackgroundAgentLoaderStart:
    def test_start_creates_task_and_forwards_kwargs(self):
        captured: dict = {}
        loader = BackgroundAgentLoader(_make_loader_fn(capture=captured))

        async def _go():
            loader.start(workspace_dir="/ws", checkpointer="CK")
            assert loader.task is not None
            assert loader.is_pending
            await loader.await_ready()

        _run(_go())
        assert captured["kwargs"][0] == {"workspace_dir": "/ws", "checkpointer": "CK"}

    def test_start_bumps_load_id(self):
        loader = BackgroundAgentLoader(_make_loader_fn())

        async def _go():
            assert loader._load_id == 0
            loader.start()
            assert loader._load_id == 1
            loader.start()
            assert loader._load_id == 2
            await loader.await_ready()

        _run(_go())

    def test_start_cancels_in_flight_prior_task(self):
        import time

        def _blocking(*, on_mcp_progress=None):
            time.sleep(0.05)
            return "LATE"

        async def _go():
            loader = BackgroundAgentLoader(_blocking)
            loader.start()
            first_task = loader.task
            # Supersede immediately; asyncio.to_thread wrapper gets cancelled.
            loader._loader_fn = _make_loader_fn("FRESH")
            loader.start()
            agent = await loader.await_ready()
            assert agent == "FRESH"
            # Let the first thread drain so its done callback (gated) fires.
            await asyncio.sleep(0.1)
            assert first_task.cancelled() or first_task.done()

        _run(_go())


class TestBackgroundAgentLoaderCallbacks:
    def test_progress_hook_sees_events_in_order(self):
        events: list[tuple[str, str, str]] = []
        loader = BackgroundAgentLoader(
            _make_loader_fn(capture={}),
            on_progress=lambda e, s, d: events.append((e, s, d)),
        )

        async def _go():
            loader.start()
            await loader.await_ready()

        _run(_go())
        assert events == [("start", "srv", ""), ("success", "srv", "1")]

    def test_stale_progress_events_are_dropped(self):
        """A progress event fired after a newer `start` must not reach the hook."""
        import time

        seen: list[str] = []

        # Loader 1 sleeps so its progress event fires AFTER load 2 starts.
        def slow_loader(*, on_mcp_progress=None):
            time.sleep(0.08)
            if on_mcp_progress is not None:
                on_mcp_progress("success", "from-slow", "1")
            return "slow-agent"

        def fast_loader(*, on_mcp_progress=None):
            if on_mcp_progress is not None:
                on_mcp_progress("success", "from-fast", "1")
            return "fast-agent"

        loader = BackgroundAgentLoader(
            slow_loader, on_progress=lambda e, s, d: seen.append(s)
        )

        async def _go():
            loader.start()
            # Supersede before the slow thread's event fires.
            await asyncio.sleep(0.01)
            loader._loader_fn = fast_loader
            loader.start()
            await loader.await_ready()
            # Let the superseded thread finish (its event is gated out).
            await asyncio.sleep(0.1)

        _run(_go())
        assert "from-fast" in seen
        assert "from-slow" not in seen

    def test_success_callback_fires_on_completion(self):
        got = []
        loader = BackgroundAgentLoader(
            _make_loader_fn("MY_AGENT"),
            on_success=lambda a: got.append(a),
        )

        async def _go():
            loader.start()
            await loader.await_ready()
            await asyncio.sleep(0)  # let done-callback run

        _run(_go())
        assert got == ["MY_AGENT"]

    def test_failure_callback_fires_on_error(self):
        err = RuntimeError("load failed")
        got_failures = []
        got_successes = []
        loader = BackgroundAgentLoader(
            _make_loader_fn(fail_with=err),
            on_success=lambda a: got_successes.append(a),
            on_failure=lambda e: got_failures.append(e),
        )

        async def _go():
            loader.start()
            with pytest.raises(RuntimeError, match="load failed"):
                await loader.await_ready()
            await asyncio.sleep(0)

        _run(_go())
        assert got_failures == [err]
        assert got_successes == []


class TestBackgroundAgentLoaderAwaitReady:
    def test_returns_cached_agent_without_reawaiting(self):
        captured: dict = {}
        loader = BackgroundAgentLoader(_make_loader_fn("A", capture=captured))

        async def _go():
            loader.start()
            assert await loader.await_ready() == "A"
            assert await loader.await_ready() == "A"

        _run(_go())
        assert len(captured["kwargs"]) == 1

    def test_raises_if_started_not_called(self):
        loader = BackgroundAgentLoader(_make_loader_fn())

        async def _go():
            with pytest.raises(RuntimeError, match="before start"):
                await loader.await_ready()

        _run(_go())

    def test_reraises_real_error_on_subsequent_awaits(self):
        """After a failure, ``await_ready`` must keep raising the real exception —
        not the "before start()" sentinel — until ``start`` is called again."""

        def _fail(*, on_mcp_progress=None):
            raise RuntimeError("bad MCP config")

        loader = BackgroundAgentLoader(_fail)

        async def _go():
            loader.start()
            with pytest.raises(RuntimeError, match="bad MCP config"):
                await loader.await_ready()
            with pytest.raises(RuntimeError, match="bad MCP config"):
                await loader.await_ready()

        _run(_go())

    def test_needs_restart_flags_failed_load_for_retry(self):
        calls = {"n": 0}

        def flaky(*, on_mcp_progress=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first attempt failed")
            return "SECOND"

        loader = BackgroundAgentLoader(flaky)

        async def _go():
            assert loader.needs_restart  # never started
            loader.start()
            with pytest.raises(RuntimeError):
                await loader.await_ready()
            assert loader.needs_restart  # failed, caller may retry
            loader.start()
            assert await loader.await_ready() == "SECOND"
            assert not loader.needs_restart  # success → no retry

        _run(_go())


class TestBackgroundAgentLoaderAdopt:
    def test_adopt_seats_external_agent(self):
        loader = BackgroundAgentLoader(_make_loader_fn())
        loader.adopt("EXTERNAL")
        assert loader.agent == "EXTERNAL"
        assert not loader.is_pending

    def test_adopt_supersedes_in_flight_load(self):
        """A late background completion must not overwrite an adopted agent."""
        import time

        def _slow(*, on_mcp_progress=None):
            time.sleep(0.08)
            return "FROM_BACKGROUND"

        loader = BackgroundAgentLoader(_slow)

        async def _go():
            loader.start()
            await asyncio.sleep(0.01)
            loader.adopt("FROM_MODEL")
            # Give the background thread time to finish and fire its
            # done-callback; the generation token should make it a no-op.
            await asyncio.sleep(0.1)
            assert loader.agent == "FROM_MODEL"

        _run(_go())


class TestBackgroundAgentLoaderIsPending:
    def test_false_before_start(self):
        loader = BackgroundAgentLoader(_make_loader_fn())
        assert not loader.is_pending

    def test_false_after_completion(self):
        loader = BackgroundAgentLoader(_make_loader_fn())

        async def _go():
            loader.start()
            await loader.await_ready()

        _run(_go())
        assert not loader.is_pending

    def test_true_between_start_and_completion(self):
        import time

        def _wait_loader(*, on_mcp_progress=None):
            time.sleep(0.05)
            return "ok"

        loader = BackgroundAgentLoader(_wait_loader)

        async def _go():
            loader.start()
            assert loader.is_pending
            await loader.await_ready()
            assert not loader.is_pending

        _run(_go())
