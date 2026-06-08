"""Tests for the auxiliary-model resolver and its middleware scoping.

Covers ``EvoScientist.EvoScientist._ensure_auxiliary_chat_model`` (fallback to
the main model when unset) and the wiring in ``_get_default_middleware`` that
routes the main agent's tool selector to the auxiliary model while keeping
context editing — and async sub-agents — on the main model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import EvoScientist.EvoScientist as E


@pytest.fixture(autouse=True)
def _reset_model_caches(monkeypatch):
    """Isolate the module-level model caches per test."""
    monkeypatch.setattr(E, "_chat_model", None, raising=False)
    monkeypatch.setattr(E, "_chat_model_key", None, raising=False)
    monkeypatch.setattr(E, "_auxiliary_chat_model", None, raising=False)
    monkeypatch.setattr(E, "_auxiliary_chat_model_key", None, raising=False)


def _cfg(**over):
    base = {
        "model": "main-m",
        "provider": "main-p",
        "auxiliary_model": "",
        "auxiliary_provider": "",
    }
    base.update(over)
    return SimpleNamespace(**base)


class TestAuxiliaryResolver:
    def test_empty_returns_main_instance(self, monkeypatch):
        main = object()
        monkeypatch.setattr(E, "_ensure_config", lambda config=None: _cfg())
        monkeypatch.setattr(E, "_ensure_chat_model", lambda: main)
        assert E._ensure_auxiliary_chat_model() is main

    def test_aux_equal_to_main_reuses_main_instance(self, monkeypatch):
        main = object()
        monkeypatch.setattr(
            E,
            "_ensure_config",
            lambda config=None: _cfg(
                auxiliary_model="main-m", auxiliary_provider="main-p"
            ),
        )
        monkeypatch.setattr(E, "_ensure_chat_model", lambda: main)
        assert E._ensure_auxiliary_chat_model() is main

    def test_set_builds_auxiliary(self, monkeypatch):
        fake = object()
        get_chat_model = MagicMock(return_value=fake)
        monkeypatch.setattr(
            E,
            "_ensure_config",
            lambda config=None: _cfg(
                auxiliary_model="aux-m", auxiliary_provider="aux-p"
            ),
        )
        monkeypatch.setattr("EvoScientist.llm.get_chat_model", get_chat_model)
        assert E._ensure_auxiliary_chat_model() is fake
        get_chat_model.assert_called_once_with(model="aux-m", provider="aux-p")

    def test_empty_provider_falls_back_to_main_provider(self, monkeypatch):
        get_chat_model = MagicMock(return_value=object())
        monkeypatch.setattr(
            E,
            "_ensure_config",
            lambda config=None: _cfg(auxiliary_model="aux-m", auxiliary_provider=""),
        )
        monkeypatch.setattr("EvoScientist.llm.get_chat_model", get_chat_model)
        E._ensure_auxiliary_chat_model()
        get_chat_model.assert_called_once_with(model="aux-m", provider="main-p")

    def test_set_chat_model_resets_aux_cache(self, monkeypatch):
        monkeypatch.setattr(E, "_auxiliary_chat_model", object(), raising=False)
        monkeypatch.setattr(E, "_auxiliary_chat_model_key", ("x", "y"), raising=False)
        monkeypatch.setattr(
            "EvoScientist.llm.get_chat_model", MagicMock(return_value=object())
        )
        E.set_chat_model("new-m", "new-p")
        assert E._auxiliary_chat_model is None
        assert E._auxiliary_chat_model_key is None


def _mock_cfg():
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    return cfg


class TestAuxiliaryMiddlewareScope:
    """``_get_default_middleware`` routes only the right components to aux."""

    def _capture(self):
        cap: dict[str, object] = {}

        def fake_tool_selector(*args, model=None, **kwargs):
            cap["tool_selector"] = model
            return [MagicMock()]

        def fake_context_editing(model=None, *args, **kwargs):
            cap["context_editing"] = model
            return MagicMock()

        return cap, fake_tool_selector, fake_context_editing

    def test_main_agent_tool_selector_aux_context_editing_main(self):
        cap, fake_ts, fake_ce = self._capture()
        main_model, aux_model = object(), object()
        with (
            patch.object(E, "_ensure_config", return_value=_mock_cfg()),
            patch.object(E, "_ensure_chat_model", return_value=main_model),
            patch.object(E, "_ensure_auxiliary_chat_model", return_value=aux_model),
            patch(
                "EvoScientist.middleware.create_tool_selector_middleware",
                side_effect=fake_ts,
            ),
            patch(
                "EvoScientist.middleware.create_context_editing_middleware",
                side_effect=fake_ce,
            ),
        ):
            E._get_default_middleware()

        assert cap["tool_selector"] is aux_model
        assert cap["context_editing"] is main_model

    def test_async_subagent_tool_selector_stays_main(self):
        cap, fake_ts, fake_ce = self._capture()
        main_model, aux_model = object(), object()
        with (
            patch.object(E, "_ensure_config", return_value=_mock_cfg()),
            patch.object(E, "_ensure_chat_model", return_value=main_model),
            patch.object(E, "_ensure_auxiliary_chat_model", return_value=aux_model),
            patch(
                "EvoScientist.middleware.create_tool_selector_middleware",
                side_effect=fake_ts,
            ),
            patch(
                "EvoScientist.middleware.create_context_editing_middleware",
                side_effect=fake_ce,
            ),
        ):
            E._get_default_middleware(for_async_subagent=True)

        assert cap["tool_selector"] is main_model
        assert cap["context_editing"] is main_model

    def test_pure_path_tool_selector_uses_threaded_main_when_aux_empty(self):
        cap, fake_ts, fake_ce = self._capture()
        cfg = _mock_cfg()
        cfg.model = "new-main"
        cfg.provider = "new-provider"
        cfg.auxiliary_model = ""
        cfg.auxiliary_provider = ""
        main_model = object()

        with (
            patch.object(E, "_ensure_config", side_effect=AssertionError),
            patch.object(E, "_ensure_chat_model", side_effect=AssertionError),
            patch.object(E, "_ensure_auxiliary_chat_model", side_effect=AssertionError),
            patch(
                "EvoScientist.middleware.create_tool_selector_middleware",
                side_effect=fake_ts,
            ),
            patch(
                "EvoScientist.middleware.create_context_editing_middleware",
                side_effect=fake_ce,
            ),
        ):
            E._get_default_middleware(cfg=cfg, chat_model=main_model)

        assert cap["tool_selector"] is main_model
        assert cap["context_editing"] is main_model

    def test_pure_path_tool_selector_builds_aux_from_threaded_config(self):
        cap, fake_ts, fake_ce = self._capture()
        cfg = _mock_cfg()
        cfg.model = "new-main"
        cfg.provider = "new-provider"
        cfg.auxiliary_model = "new-aux"
        cfg.auxiliary_provider = "aux-provider"
        main_model, aux_model = object(), object()

        with (
            patch.object(E, "_ensure_config", side_effect=AssertionError),
            patch.object(E, "_ensure_chat_model", side_effect=AssertionError),
            patch.object(E, "_ensure_auxiliary_chat_model", side_effect=AssertionError),
            patch(
                "EvoScientist.llm.get_chat_model", return_value=aux_model
            ) as get_model,
            patch(
                "EvoScientist.middleware.create_tool_selector_middleware",
                side_effect=fake_ts,
            ),
            patch(
                "EvoScientist.middleware.create_context_editing_middleware",
                side_effect=fake_ce,
            ),
        ):
            E._get_default_middleware(cfg=cfg, chat_model=main_model)

        get_model.assert_called_once_with(model="new-aux", provider="aux-provider")
        assert cap["tool_selector"] is aux_model
        assert cap["context_editing"] is main_model
