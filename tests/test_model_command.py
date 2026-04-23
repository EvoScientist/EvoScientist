"""Tests for the /model command and extract_model_and_provider helper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import run_async as _run


class TestExtractModelAndProvider:
    """Unit tests for the argument parser helper."""

    def test_known_model_no_provider(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(["claude-sonnet-4-6"])
        assert name == "claude-sonnet-4-6"
        assert prov == "anthropic"

    def test_known_model_with_provider_override(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(["claude-sonnet-4-6", "openrouter"])
        assert name == "claude-sonnet-4-6"
        assert prov == "openrouter"

    def test_unknown_model_no_provider_raises(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        with pytest.raises(ValueError, match="Unknown model"):
            extract_model_and_provider(["nonexistent-model-xyz"])

    def test_unknown_model_with_provider_still_raises(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        # Unknown models are always rejected, even with an explicit provider
        with pytest.raises(ValueError, match="Unknown model"):
            extract_model_and_provider(["my-custom-model", "custom-openai"])

    def test_provider_override_on_known_model(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        # Known model with explicit provider override uses the override
        name, prov = extract_model_and_provider(["claude-sonnet-4-6", "openrouter"])
        assert name == "claude-sonnet-4-6"
        assert prov == "openrouter"


class TestModelCommandUnknownModel:
    """Verify error message for unknown models."""

    def test_unknown_model_shows_error(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=cfg,
        ):
            _run(cmd.execute(ctx, ["nonexistent-model-xyz"]))

        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Unknown model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


class TestModelCommandPickerCancelled:
    """Verify no-op when the interactive picker is cancelled."""

    def test_picker_returns_none(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_model_pick = AsyncMock(return_value=None)
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=cfg,
        ):
            _run(cmd.execute(ctx, []))

        # No model switch should have happened
        ui.append_system.assert_not_called()


class TestModelCommandSwitch:
    """Verify a successful model switch updates config and rebuilds agent."""

    def test_switch_known_model(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        new_agent = MagicMock()

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.EvoScientist.set_chat_model",
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=new_agent,
            ),
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6"]))

        # Config should be updated
        assert cfg.model == "claude-opus-4-6"
        assert cfg.provider == "anthropic"

        # Agent should be replaced on context
        assert ctx.agent == new_agent

        # Success message shown
        ui.append_system.assert_called_once()
        msg = ui.append_system.call_args[0][0]
        assert "claude-opus-4-6" in msg
        assert "anthropic" in msg

    def test_switch_with_save_flag(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch("EvoScientist.EvoScientist.set_chat_model"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch("EvoScientist.config.settings.set_config_value") as mock_save,
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6", "--save"]))

        # Config file should be updated
        mock_save.assert_any_call("model", "claude-opus-4-6")
        mock_save.assert_any_call("provider", "anthropic")

        # Success message should mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" in msg

    def test_switch_without_save_flag_does_not_persist(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch("EvoScientist.EvoScientist.set_chat_model"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch("EvoScientist.config.settings.set_config_value") as mock_save,
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6"]))

        # Config file should NOT be updated
        mock_save.assert_not_called()

        # Message should not mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" not in msg


class TestModelCommandFailure:
    """Verify error handling when set_chat_model raises."""

    def test_set_chat_model_error(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch(
                "EvoScientist.EvoScientist.set_chat_model",
                side_effect=RuntimeError("API key missing"),
            ) as mock_set,
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6"]))

        mock_set.assert_called_once()
        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Failed to switch model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


class TestEnsureChatModelCacheInvalidation:
    """Regression tests for issue #179: /model switch lagged by one step.

    Root cause: ``_ensure_chat_model()`` returned a cached ``_chat_model``
    without checking whether ``cfg.model`` / ``cfg.provider`` had changed
    since the cache was populated. ``ModelCommand._apply_model`` builds
    a new agent *before* ``set_chat_model`` runs, so the new agent was
    bound to the *previous* cached model.

    Fix: track a ``(model, provider)`` key alongside ``_chat_model`` and
    rebuild on mismatch.
    """

    def test_cache_rebuilds_when_config_model_changes(self):
        """After cfg.model changes, _ensure_chat_model must return a new instance."""
        import EvoScientist.EvoScientist as mod

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        m1 = MagicMock(name="model-1")
        m2 = MagicMock(name="model-2")

        # Snapshot module state and restore after the test.
        saved = (
            mod._chat_model,
            mod._chat_model_key,
            mod._config,
            mod._EvoScientist_agent,
        )
        try:
            mod._chat_model = None
            mod._chat_model_key = None
            mod._config = cfg
            mod._EvoScientist_agent = None

            with patch("EvoScientist.llm.get_chat_model", side_effect=[m1, m2]) as gm:
                first = mod._ensure_chat_model()
                assert first is m1
                # Same config → cache hit, no rebuild.
                again = mod._ensure_chat_model()
                assert again is m1
                assert gm.call_count == 1

                # Simulate /model switch writing the new choice into cfg.
                cfg.model = "minimax-m2.7"
                cfg.provider = "openrouter"

                second = mod._ensure_chat_model()
                # Must be the NEW model instance, not the cached one.
                assert second is m2
                assert second is not first
                assert gm.call_count == 2
                # Second call used the new model name + provider.
                _, kwargs = gm.call_args
                assert kwargs == {"model": "minimax-m2.7", "provider": "openrouter"}
        finally:
            (
                mod._chat_model,
                mod._chat_model_key,
                mod._config,
                mod._EvoScientist_agent,
            ) = saved

    def test_cache_rebuilds_when_only_provider_changes(self):
        """Same model name, different provider (openrouter vs anthropic) must rebuild."""
        import EvoScientist.EvoScientist as mod

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        m1 = MagicMock(name="anthropic-model")
        m2 = MagicMock(name="openrouter-model")

        saved = (
            mod._chat_model,
            mod._chat_model_key,
            mod._config,
            mod._EvoScientist_agent,
        )
        try:
            mod._chat_model = None
            mod._chat_model_key = None
            mod._config = cfg
            mod._EvoScientist_agent = None

            with patch("EvoScientist.llm.get_chat_model", side_effect=[m1, m2]) as gm:
                assert mod._ensure_chat_model() is m1
                cfg.provider = "openrouter"
                assert mod._ensure_chat_model() is m2
                assert gm.call_count == 2
        finally:
            (
                mod._chat_model,
                mod._chat_model_key,
                mod._config,
                mod._EvoScientist_agent,
            ) = saved

    def test_set_chat_model_updates_key(self):
        """set_chat_model must keep _chat_model_key in sync to avoid
        an extra rebuild on the very next _ensure_chat_model() call."""
        import EvoScientist.EvoScientist as mod

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        m_set = MagicMock(name="explicit-set-model")

        saved = (
            mod._chat_model,
            mod._chat_model_key,
            mod._config,
            mod._EvoScientist_agent,
        )
        try:
            mod._chat_model = None
            mod._chat_model_key = None
            mod._config = cfg
            mod._EvoScientist_agent = None

            with patch("EvoScientist.llm.get_chat_model", return_value=m_set) as gm:
                mod.set_chat_model("minimax-m2.7", provider="openrouter")
                assert mod._chat_model is m_set
                assert mod._chat_model_key == ("minimax-m2.7", "openrouter")

                # Align cfg to what set_chat_model was called with.
                cfg.model = "minimax-m2.7"
                cfg.provider = "openrouter"
                # Now _ensure_chat_model should NOT rebuild (key matches cfg).
                assert mod._ensure_chat_model() is m_set
                assert gm.call_count == 1
        finally:
            (
                mod._chat_model,
                mod._chat_model_key,
                mod._config,
                mod._EvoScientist_agent,
            ) = saved


class TestApplyModelIntegration:
    """End-to-end regression for #179: `_apply_model` must produce an agent
    bound to the NEW model, not a stale cached one.

    Exercises the real chain:
        ``_apply_model → _load_agent → _ensure_config → _ensure_chat_model``

    Only ``_load_agent`` is replaced by a minimal fake that mirrors the
    exact two globals ``create_cli_agent`` mutates (``_ensure_config``
    + ``_ensure_chat_model``) — so the bug path is fully exercised
    without having to spin up deepagents, MCP tools, middleware, and
    subagent YAML. ``get_chat_model`` returns a distinct sentinel per
    ``(model, provider)`` pair so we can assert on identity.
    """

    def test_new_agent_is_bound_to_newly_selected_model(self):
        import EvoScientist.EvoScientist as mod
        from EvoScientist.commands.implementation.model import ModelCommand

        sentinels: dict[tuple[str, str | None], MagicMock] = {}

        def _fake_get_chat_model(model, provider=None):
            key = (model, provider)
            if key not in sentinels:
                sentinels[key] = MagicMock(name=f"chat_model[{model}|{provider}]")
                sentinels[key]._bound_model = model
                sentinels[key]._bound_provider = provider
            return sentinels[key]

        def _fake_load_agent(
            workspace_dir=None, checkpointer=None, config=None, *, on_mcp_progress=None
        ):
            # Replicates the two create_cli_agent side effects that
            # reveal the bug: ``_ensure_config`` writes the new cfg,
            # then ``_ensure_chat_model`` must rebuild to match it.
            mod._ensure_config(config)
            agent = MagicMock(name="fake-agent")
            agent._bound_model = mod._ensure_chat_model()
            return agent

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.workspace_dir = "/tmp/test_integration"
        ctx.checkpointer = None

        saved = (
            mod._chat_model,
            mod._chat_model_key,
            mod._config,
            mod._EvoScientist_agent,
        )
        try:
            # Prime: _chat_model already holds the OLD (default) model —
            # this is the state that caused the off-by-one in production.
            mod._config = cfg
            mod._chat_model = _fake_get_chat_model("claude-sonnet-4-6", "anthropic")
            mod._chat_model_key = ("claude-sonnet-4-6", "anthropic")
            mod._EvoScientist_agent = None
            old_model = mod._chat_model

            with (
                patch(
                    "EvoScientist.llm.get_chat_model",
                    side_effect=_fake_get_chat_model,
                ),
                patch(
                    "EvoScientist.cli.agent._load_agent",
                    side_effect=_fake_load_agent,
                ),
                # ``_ensure_config`` calls ``apply_config_to_env`` which reads
                # a dozen attributes off the config.  The SimpleNamespace we
                # use as cfg only has ``model`` / ``provider``, so stub the
                # env propagation out — it's orthogonal to this test.
                patch("EvoScientist.EvoScientist.apply_config_to_env"),
            ):
                cmd = ModelCommand()
                _run(cmd._apply_model(ctx, "minimax-m2.7", "openrouter"))

            # The agent produced by _apply_model must be bound to the
            # NEWLY requested model, not the previously cached one.
            assert ctx.agent._bound_model is not old_model
            assert ctx.agent._bound_model._bound_model == "minimax-m2.7"
            assert ctx.agent._bound_model._bound_provider == "openrouter"

            # Global state reflects the switch end-to-end.
            assert mod._chat_model_key == ("minimax-m2.7", "openrouter")
            assert mod._chat_model is sentinels[("minimax-m2.7", "openrouter")]
            assert cfg.model == "minimax-m2.7"
            assert cfg.provider == "openrouter"

            # User-visible success message.
            msg = ctx.ui.append_system.call_args[0][0]
            assert "minimax-m2.7" in msg
            assert "openrouter" in msg
        finally:
            (
                mod._chat_model,
                mod._chat_model_key,
                mod._config,
                mod._EvoScientist_agent,
            ) = saved


class TestModelCommandLoadAgentFailure:
    """Verify the transactional ordering: when ``_load_agent`` raises,
    nothing downstream (``set_chat_model``, ``cfg`` mutation,
    ``set_config_value``) should happen.

    This is the core guarantee of the refactor that established
    "build agent first, commit state only on success". Without this test
    the ordering could silently regress (e.g. if ``_apply_model`` were
    reordered to call ``set_chat_model`` first)."""

    def test_load_agent_error_is_transactional(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                side_effect=RuntimeError("agent build failed"),
            ) as mock_load,
            patch(
                "EvoScientist.EvoScientist.set_chat_model",
            ) as mock_set,
            patch(
                "EvoScientist.config.settings.set_config_value",
            ) as mock_save,
        ):
            # Pass ``--save`` to strengthen the assertion: if the ordering
            # ever regresses, ``set_config_value`` would be called with
            # stale data.
            _run(cmd.execute(ctx, ["claude-opus-4-6", "--save"]))

        # _load_agent was attempted (transactional first step).
        mock_load.assert_called_once()
        # Downstream side-effects must NOT have happened.
        mock_set.assert_not_called()
        mock_save.assert_not_called()
        # Config must be untouched.
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.provider == "anthropic"
        # User sees a red error message.
        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Failed to switch model" in call_args[0][0]
        assert call_args[1]["style"] == "red"
