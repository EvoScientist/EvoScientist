"""Tests for ``EvoScientist.middleware.configurable_model``.

Verifies that the middleware reads ``model`` / ``model_provider`` from
the active ``RunnableConfig.configurable`` (via ``langgraph.config.get_config``)
and overrides ``request.model`` accordingly, without breaking the no-override
pass-through path or the per-instance cache.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from EvoScientist.middleware.configurable_model import (
    ConfigurableModelMiddleware,
    _read_model_override,
)
from tests.conftest import run_async as _run


@contextmanager
def _patched_config(configurable: dict | object | None):
    """Patch ``langgraph.config.get_config`` to return a controlled value.

    Pass ``None`` to simulate "outside a runnable context" (raises
    ``RuntimeError`` like the real ``get_config()`` does).
    Pass a dict for ``configurable`` to expose just that key.
    Pass any other object to simulate a malformed config.
    """
    import langgraph.config as _lg_cfg

    if configurable is None:
        # Simulate get_config raising outside a runnable context.
        with patch.object(
            _lg_cfg,
            "get_config",
            side_effect=RuntimeError("Called get_config outside of a runnable context"),
        ):
            yield
    elif isinstance(configurable, dict):
        with patch.object(
            _lg_cfg,
            "get_config",
            return_value={"configurable": configurable},
        ):
            yield
    else:
        with patch.object(
            _lg_cfg,
            "get_config",
            return_value=configurable,
        ):
            yield


def _make_request():
    """Build a minimal ``ModelRequest`` stub.

    ``request.override(model=...)`` returns a new request whose ``model``
    field reflects the override.
    """
    req = MagicMock()

    def _override(**kwargs):
        new = MagicMock()
        new.model = kwargs.get("model", req.model)
        return new

    req.override = MagicMock(side_effect=_override)
    return req


# =============================================================================
# 1. _read_model_override — input parsing
# =============================================================================


class TestReadModelOverride:
    """Verify the helper that pulls (model, provider) from active config."""

    def test_returns_override_when_both_present(self):
        with _patched_config({"model": "gpt-5", "model_provider": "openai"}):
            assert _read_model_override() == ("gpt-5", "openai")

    def test_provider_optional(self):
        with _patched_config({"model": "claude-haiku-4-5"}):
            assert _read_model_override() == ("claude-haiku-4-5", None)

    def test_no_configurable_key(self):
        with _patched_config({}):
            assert _read_model_override() == (None, None)

    def test_outside_runnable_context(self):
        """``get_config`` raises outside a runnable — middleware must no-op."""
        with _patched_config(None):
            assert _read_model_override() == (None, None)

    def test_empty_string_treated_as_absent(self):
        with _patched_config({"model": "", "model_provider": ""}):
            assert _read_model_override() == (None, None)

    def test_non_string_ignored(self):
        with _patched_config({"model": 42, "model_provider": object()}):
            assert _read_model_override() == (None, None)

    def test_non_dict_configurable_safe(self):
        with _patched_config({"configurable": "not-a-dict"}):
            # Inner ``configurable`` is the wrong type — patched_config
            # wraps it again so we end up with {"configurable": {"configurable": "..."}}
            # which has no model/model_provider keys → no override.
            assert _read_model_override() == (None, None)

    def test_non_dict_config_safe(self):
        with _patched_config("garbage"):
            assert _read_model_override() == (None, None)


# =============================================================================
# 2. ConfigurableModelMiddleware — pass-through behavior
# =============================================================================


class TestPassThrough:
    """When no override is present, the middleware must not touch the request."""

    def test_sync_no_override_passes_request_unchanged(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        sentinel = object()
        handler = MagicMock(return_value=sentinel)
        with _patched_config({}):
            result = mw.wrap_model_call(req, handler)
        assert result is sentinel
        handler.assert_called_once_with(req)
        req.override.assert_not_called()

    def test_async_no_override_passes_request_unchanged(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()

        async def handler(r):
            assert r is req
            return "ok"

        with _patched_config({}):
            result = _run(mw.awrap_model_call(req, handler))
        assert result == "ok"
        req.override.assert_not_called()

    def test_outside_runnable_context_passes_through(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        handler = MagicMock(return_value="ok")
        with _patched_config(None):
            mw.wrap_model_call(req, handler)
        handler.assert_called_once_with(req)


# =============================================================================
# 3. ConfigurableModelMiddleware — override behavior
# =============================================================================


class TestModelOverride:
    """When override present, middleware resolves model and overrides request."""

    def test_sync_override_calls_get_chat_model_and_overrides(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        new_model = MagicMock(name="resolved_chat_model")
        handler = MagicMock(return_value="response")

        with (
            _patched_config({"model": "gpt-5", "model_provider": "openai"}),
            patch(
                "EvoScientist.llm.get_chat_model", return_value=new_model
            ) as mock_get,
        ):
            mw.wrap_model_call(req, handler)

        mock_get.assert_called_once_with(model="gpt-5", provider="openai")
        req.override.assert_called_once_with(model=new_model)
        # Handler must receive the OVERRIDDEN request, not the original.
        called_with = handler.call_args[0][0]
        assert called_with is not req
        assert called_with.model is new_model

    def test_async_override_path_parity(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        new_model = MagicMock()

        async def handler(r):
            assert r.model is new_model
            return "ok"

        with (
            _patched_config(
                {"model": "claude-opus-4-7", "model_provider": "anthropic"}
            ),
            patch(
                "EvoScientist.llm.get_chat_model", return_value=new_model
            ) as mock_get,
        ):
            result = _run(mw.awrap_model_call(req, handler))

        assert result == "ok"
        mock_get.assert_called_once_with(model="claude-opus-4-7", provider="anthropic")

    def test_provider_omitted_passed_as_none(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        new_model = MagicMock()
        handler = MagicMock(return_value="ok")

        with (
            _patched_config({"model": "gpt-5"}),
            patch(
                "EvoScientist.llm.get_chat_model", return_value=new_model
            ) as mock_get,
        ):
            mw.wrap_model_call(req, handler)

        mock_get.assert_called_once_with(model="gpt-5", provider=None)


# =============================================================================
# 4. ConfigurableModelMiddleware — caching
# =============================================================================


class TestCache:
    """Two consecutive calls with same (model, provider) should hit cache."""

    def test_cache_hit_avoids_second_get_chat_model(self):
        mw = ConfigurableModelMiddleware()
        req1 = _make_request()
        req2 = _make_request()
        new_model = MagicMock()
        handler = MagicMock(return_value="ok")

        with (
            _patched_config({"model": "gpt-5", "model_provider": "openai"}),
            patch(
                "EvoScientist.llm.get_chat_model", return_value=new_model
            ) as mock_get,
        ):
            mw.wrap_model_call(req1, handler)
            mw.wrap_model_call(req2, handler)

        # First call resolves via factory, second hits the cache.
        assert mock_get.call_count == 1

    def test_cache_miss_on_different_provider(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        handler = MagicMock(return_value="ok")

        with patch(
            "EvoScientist.llm.get_chat_model", side_effect=[MagicMock(), MagicMock()]
        ) as mock_get:
            with _patched_config(
                {"model": "claude-sonnet-4-6", "model_provider": "anthropic"}
            ):
                mw.wrap_model_call(req, handler)
            with _patched_config(
                {"model": "claude-sonnet-4-6", "model_provider": "custom-anthropic"}
            ):
                mw.wrap_model_call(req, handler)

        assert mock_get.call_count == 2

    def test_independent_instances_have_independent_caches(self):
        mw_a = ConfigurableModelMiddleware()
        mw_b = ConfigurableModelMiddleware()
        req = _make_request()
        handler = MagicMock(return_value="ok")

        with (
            _patched_config({"model": "gpt-5", "model_provider": "openai"}),
            patch(
                "EvoScientist.llm.get_chat_model",
                side_effect=[MagicMock(), MagicMock()],
            ) as mock_get,
        ):
            mw_a.wrap_model_call(req, handler)
            mw_b.wrap_model_call(req, handler)

        # Different instances must each resolve once.
        assert mock_get.call_count == 2


# =============================================================================
# 5. Resilience — get_chat_model raising
# =============================================================================


class TestResolveFailure:
    """If get_chat_model raises, middleware must fall back to original model."""

    def test_sync_falls_back_when_resolve_raises(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()
        handler = MagicMock(return_value="ok")

        with (
            _patched_config({"model": "doesnotexist", "model_provider": "openai"}),
            patch(
                "EvoScientist.llm.get_chat_model",
                side_effect=ValueError("unknown model"),
            ),
        ):
            mw.wrap_model_call(req, handler)

        # Override never happened — handler called with original request.
        handler.assert_called_once_with(req)
        req.override.assert_not_called()

    def test_async_falls_back_when_resolve_raises(self):
        mw = ConfigurableModelMiddleware()
        req = _make_request()

        called = []

        async def handler(r):
            called.append(r)
            return "ok"

        with (
            _patched_config({"model": "doesnotexist", "model_provider": "openai"}),
            patch(
                "EvoScientist.llm.get_chat_model",
                side_effect=ValueError("unknown model"),
            ),
        ):
            result = _run(mw.awrap_model_call(req, handler))

        assert result == "ok"
        assert called == [req]


# =============================================================================
# 6. Integration — real langgraph contextvar (no get_config mock)
# =============================================================================


class TestRunnableContextVarIntegration:
    """Set the actual ``var_child_runnable_config`` contextvar that LangGraph
    populates per node, then verify the middleware reads through to it.

    This catches breakage of the ``langgraph.config.get_config()`` contract
    that pure-mock tests would miss (e.g. if get_config is moved to a
    different module, or the contextvar mechanism changes).
    """

    def test_real_contextvar_drives_override(self):
        """Without mocking get_config, set the contextvar and verify override."""
        from langchain_core.runnables.config import var_child_runnable_config

        mw = ConfigurableModelMiddleware()
        req = _make_request()
        new_model = MagicMock(name="resolved")
        handler = MagicMock(return_value="ok")

        token = var_child_runnable_config.set(
            {"configurable": {"model": "gpt-5.5", "model_provider": "openai"}}
        )
        try:
            with patch(
                "EvoScientist.llm.get_chat_model", return_value=new_model
            ) as mock_get:
                mw.wrap_model_call(req, handler)
        finally:
            var_child_runnable_config.reset(token)

        mock_get.assert_called_once_with(model="gpt-5.5", provider="openai")
        req.override.assert_called_once_with(model=new_model)

    def test_real_contextvar_unset_passes_through(self):
        """When no contextvar is set, get_config() raises → no override."""
        from langchain_core.runnables.config import var_child_runnable_config

        # Defensive: ensure no leftover contextvar from another test.
        token = var_child_runnable_config.set(None)
        try:
            mw = ConfigurableModelMiddleware()
            req = _make_request()
            handler = MagicMock(return_value="ok")
            mw.wrap_model_call(req, handler)
        finally:
            var_child_runnable_config.reset(token)

        handler.assert_called_once_with(req)
        req.override.assert_not_called()
