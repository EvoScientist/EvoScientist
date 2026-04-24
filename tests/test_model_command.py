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

    def test_ollama_provider_accepts_arbitrary_name(self):
        """Ollama models are locally-installed — the registry doesn't know
        them. The ``ollama`` provider must pass any name through verbatim."""
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(["llama3.3:8b", "ollama"])
        assert name == "llama3.3:8b"
        assert prov == "ollama"

    def test_ollama_provider_accepts_dotted_tag(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(["qwen3-coder-next:latest", "ollama"])
        assert name == "qwen3-coder-next:latest"
        assert prov == "ollama"


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


@pytest.fixture
def evo_module_state():
    """Snapshot and restore ``EvoScientist.EvoScientist`` module globals.

    The chat-model cache tests mutate ``_chat_model`` / ``_chat_model_key``
    / ``_config`` / ``_EvoScientist_agent`` directly.  This fixture
    guarantees all four are restored — even if a test body grows an early
    return — so later tests in the suite see a clean module state.
    """
    import EvoScientist.EvoScientist as mod

    snapshot = (
        mod._chat_model,
        mod._chat_model_key,
        mod._config,
        mod._EvoScientist_agent,
    )
    try:
        yield mod
    finally:
        (
            mod._chat_model,
            mod._chat_model_key,
            mod._config,
            mod._EvoScientist_agent,
        ) = snapshot


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

    def test_cache_rebuilds_when_config_model_changes(self, evo_module_state):
        """After cfg.model changes, _ensure_chat_model must return a new instance."""
        mod = evo_module_state

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        m1 = MagicMock(name="model-1")
        m2 = MagicMock(name="model-2")

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

    def test_cache_rebuilds_when_only_provider_changes(self, evo_module_state):
        """Same model name, different provider (openrouter vs anthropic) must rebuild."""
        mod = evo_module_state

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        m1 = MagicMock(name="anthropic-model")
        m2 = MagicMock(name="openrouter-model")

        mod._chat_model = None
        mod._chat_model_key = None
        mod._config = cfg
        mod._EvoScientist_agent = None

        with patch("EvoScientist.llm.get_chat_model", side_effect=[m1, m2]) as gm:
            assert mod._ensure_chat_model() is m1
            cfg.provider = "openrouter"
            assert mod._ensure_chat_model() is m2
            assert gm.call_count == 2

    def test_set_chat_model_updates_key(self, evo_module_state):
        """set_chat_model must keep _chat_model_key in sync to avoid
        an extra rebuild on the very next _ensure_chat_model() call."""
        mod = evo_module_state

        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        m_set = MagicMock(name="explicit-set-model")

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

    def test_set_chat_model_is_no_op_when_key_already_matches(self, evo_module_state):
        """set_chat_model must NOT rebuild when the cache already holds the
        requested (model, provider).

        Under the /model flow, ``_load_agent`` already rebuilt ``_chat_model``
        via ``_ensure_chat_model`` before ``set_chat_model`` is reached, so
        the subsequent set should be idempotent — reusing the same Python
        instance (and thus the same underlying HTTP client) that ``ctx.agent``
        is already bound to.
        """
        mod = evo_module_state

        existing = MagicMock(name="existing-model")
        mod._chat_model = existing
        mod._chat_model_key = ("minimax-m2.7", "openrouter")
        mod._config = SimpleNamespace(model="minimax-m2.7", provider="openrouter")
        mod._EvoScientist_agent = None

        with patch("EvoScientist.llm.get_chat_model") as gm:
            returned = mod.set_chat_model("minimax-m2.7", provider="openrouter")

        # Fast path: returned the EXISTING instance, no rebuild.
        assert returned is existing
        assert mod._chat_model is existing
        gm.assert_not_called()


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

    def test_new_agent_is_bound_to_newly_selected_model(self, evo_module_state):
        from EvoScientist.commands.implementation.model import ModelCommand
        from EvoScientist.config.settings import EvoScientistConfig

        mod = evo_module_state

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

        cfg = EvoScientistConfig(model="claude-sonnet-4-6", provider="anthropic")
        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.workspace_dir = "/tmp/test_integration"
        ctx.checkpointer = None

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


class TestApplyModelLoadAgentFailureTransactional:
    """Regression: if ``create_cli_agent`` raises AFTER partially mutating
    module globals via ``_ensure_config`` + ``_ensure_chat_model``,
    ``_apply_model`` must roll those back so the session stays on the
    original model.

    Complements :class:`TestModelCommandLoadAgentFailure`, which tests
    the early-failure path where ``_load_agent`` never reaches
    ``create_cli_agent`` and no globals get mutated.
    """

    def test_globals_restored_after_create_cli_agent_partial_mutation(
        self, evo_module_state
    ):
        from EvoScientist.commands.implementation.model import ModelCommand
        from EvoScientist.config.settings import EvoScientistConfig

        mod = evo_module_state
        sentinels: dict[tuple[str, str | None], MagicMock] = {}

        def _fake_get_chat_model(model, provider=None):
            key = (model, provider)
            sentinels.setdefault(key, MagicMock(name=f"chat_model[{model}|{provider}]"))
            return sentinels[key]

        def _fake_load_agent(
            workspace_dir=None,
            checkpointer=None,
            config=None,
            *,
            on_mcp_progress=None,
        ):
            # Mimic ``create_cli_agent``: mutate globals via
            # ``_ensure_config`` + ``_ensure_chat_model``, then raise
            # (as if middleware construction or deepagents wiring failed).
            mod._ensure_config(config)
            mod._ensure_chat_model()
            raise RuntimeError("middleware build failed")

        cfg = EvoScientistConfig(model="claude-sonnet-4-6", provider="anthropic")
        old_model = _fake_get_chat_model("claude-sonnet-4-6", "anthropic")
        old_agent = MagicMock(name="old-default-agent")

        mod._config = cfg
        mod._chat_model = old_model
        mod._chat_model_key = ("claude-sonnet-4-6", "anthropic")
        mod._EvoScientist_agent = old_agent

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.workspace_dir = "/tmp/test_rollback"
        ctx.checkpointer = None

        with (
            patch(
                "EvoScientist.llm.get_chat_model",
                side_effect=_fake_get_chat_model,
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                side_effect=_fake_load_agent,
            ),
        ):
            cmd = ModelCommand()
            _run(cmd._apply_model(ctx, "minimax-m2.7", "openrouter"))

        # All four globals restored to their pre-call state.
        assert mod._config is cfg
        assert mod._chat_model is old_model
        assert mod._chat_model_key == ("claude-sonnet-4-6", "anthropic")
        assert mod._EvoScientist_agent is old_agent
        # The ``cfg`` object itself was not mutated.
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.provider == "anthropic"
        # User sees an error message.
        msg = ctx.ui.append_system.call_args[0][0]
        assert "Failed to switch model" in msg


class TestApplyModelSetChatModelFailureTransactional:
    """Regression (CodeRabbit review on PR #187): if ``set_chat_model``
    raises *after* ``_load_agent`` has already mutated module globals,
    those globals must be restored. Without the rollback the session
    ends up half-switched — new ``_config`` / ``_chat_model`` committed,
    but no successful agent to back them.

    Complements :class:`TestApplyModelLoadAgentFailureTransactional`
    which covers the earlier failure site.
    """

    def test_globals_restored_when_set_chat_model_raises(self, evo_module_state):
        from EvoScientist.commands.implementation.model import ModelCommand
        from EvoScientist.config.settings import EvoScientistConfig

        mod = evo_module_state
        sentinels: dict[tuple[str, str | None], MagicMock] = {}

        def _fake_get_chat_model(model, provider=None):
            key = (model, provider)
            sentinels.setdefault(key, MagicMock(name=f"chat_model[{model}|{provider}]"))
            return sentinels[key]

        def _fake_load_agent(
            workspace_dir=None,
            checkpointer=None,
            config=None,
            *,
            on_mcp_progress=None,
        ):
            # Mimic the real ``create_cli_agent``: mutate globals via
            # ``_ensure_config`` + ``_ensure_chat_model``, then succeed.
            mod._ensure_config(config)
            mod._ensure_chat_model()
            return MagicMock(name="new-agent")

        cfg = EvoScientistConfig(model="claude-sonnet-4-6", provider="anthropic")
        old_model = _fake_get_chat_model("claude-sonnet-4-6", "anthropic")
        old_agent = MagicMock(name="old-default-agent")

        mod._config = cfg
        mod._chat_model = old_model
        mod._chat_model_key = ("claude-sonnet-4-6", "anthropic")
        mod._EvoScientist_agent = old_agent

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.workspace_dir = "/tmp/test_rollback_set"
        ctx.checkpointer = None

        with (
            patch(
                "EvoScientist.llm.get_chat_model",
                side_effect=_fake_get_chat_model,
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                side_effect=_fake_load_agent,
            ),
            patch(
                "EvoScientist.EvoScientist.set_chat_model",
                side_effect=RuntimeError("API key missing at commit step"),
            ),
        ):
            cmd = ModelCommand()
            _run(cmd._apply_model(ctx, "minimax-m2.7", "openrouter"))

        # All four globals restored — the new agent was built, but the
        # commit step (set_chat_model) failed, so the session must remain
        # on the original model.
        assert mod._config is cfg
        assert mod._chat_model is old_model
        assert mod._chat_model_key == ("claude-sonnet-4-6", "anthropic")
        assert mod._EvoScientist_agent is old_agent
        # cfg itself must not have been mutated (happens after the commit).
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.provider == "anthropic"
        # User sees an error message.
        msg = ctx.ui.append_system.call_args[0][0]
        assert "Failed to switch model" in msg


class TestModelCommandOllamaPicker:
    """Verify Ollama discovery augments the picker entries and the sentinel
    is always present when Ollama is configured."""

    def _make_ctx_and_cfg(self, *, ollama_base_url: str | None):
        cfg = SimpleNamespace(
            model="claude-sonnet-4-6",
            provider="anthropic",
            ollama_base_url=ollama_base_url,
        )
        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_model_pick = AsyncMock(return_value=None)
        ctx = MagicMock()
        ctx.ui = ui
        return ctx, cfg, ui

    def test_picker_entries_include_detected_ollama_models(self):
        """When Ollama is reachable, detected models appear in entries with
        provider='ollama' and the Custom sentinel is appended."""
        from EvoScientist.commands.implementation.model import ModelCommand

        ctx, cfg, ui = self._make_ctx_and_cfg(ollama_base_url="http://localhost:11434")

        async def fake_discover(base_url, *, timeout):
            return ["llama3.3:latest", "qwen3:8b"]

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.llm.ollama_discovery.discover_ollama_models",
                side_effect=fake_discover,
            ),
        ):
            _run(ModelCommand().execute(ctx, []))

        entries = ui.wait_for_model_pick.call_args[0][0]
        ollama_rows = [(n, mid, p) for (n, mid, p) in entries if p == "ollama"]
        assert ("llama3.3:latest", "llama3.3:latest", "ollama") in ollama_rows
        assert ("qwen3:8b", "qwen3:8b", "ollama") in ollama_rows
        assert (
            "Custom Ollama model...",
            "__custom_ollama__",
            "ollama",
        ) in ollama_rows

    def test_picker_entries_include_sentinel_when_discovery_empty(self):
        """Daemon unreachable / no models pulled — sentinel is the user's
        escape hatch and must always be present."""
        from EvoScientist.commands.implementation.model import ModelCommand

        ctx, cfg, ui = self._make_ctx_and_cfg(ollama_base_url="http://localhost:11434")

        async def fake_discover(base_url, *, timeout):
            return []

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.llm.ollama_discovery.discover_ollama_models",
                side_effect=fake_discover,
            ),
        ):
            _run(ModelCommand().execute(ctx, []))

        entries = ui.wait_for_model_pick.call_args[0][0]
        ollama_rows = [(n, mid, p) for (n, mid, p) in entries if p == "ollama"]
        assert ollama_rows == [
            ("Custom Ollama model...", "__custom_ollama__", "ollama")
        ]

    def test_picker_skips_ollama_section_when_not_configured(self):
        """ollama_base_url unset → no discovery call, no ollama entries,
        no sentinel (issue non-goal: no implicit localhost detection)."""
        from EvoScientist.commands.implementation.model import ModelCommand

        ctx, cfg, ui = self._make_ctx_and_cfg(ollama_base_url="")

        discovery = AsyncMock(return_value=["should-never-appear"])

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.llm.ollama_discovery.discover_ollama_models",
                discovery,
            ),
        ):
            _run(ModelCommand().execute(ctx, []))

        discovery.assert_not_called()
        entries = ui.wait_for_model_pick.call_args[0][0]
        assert not any(p == "ollama" for (_, _, p) in entries)

    def test_picker_handles_cfg_without_ollama_base_url_attr(self):
        """getattr(cfg, 'ollama_base_url', None) fallback: old configs
        (or SimpleNamespace test fixtures) may not carry the attribute
        at all. Must not raise AttributeError, must not probe."""
        from EvoScientist.commands.implementation.model import ModelCommand

        # Deliberately omit ollama_base_url from the namespace.
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_model_pick = AsyncMock(return_value=None)
        ctx = MagicMock()
        ctx.ui = ui

        discovery = AsyncMock(return_value=["should-never-appear"])

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.llm.ollama_discovery.discover_ollama_models",
                discovery,
            ),
        ):
            _run(ModelCommand().execute(ctx, []))

        discovery.assert_not_called()
        entries = ui.wait_for_model_pick.call_args[0][0]
        assert not any(p == "ollama" for (_, _, p) in entries)

    def test_picker_sentinel_result_is_treated_as_cancel(self):
        """Defense-in-depth: if the widget ever returns the sentinel name
        itself (shouldn't happen — it should substitute the typed name),
        dispatch treats it as a cancel and does NOT call _apply_model."""
        from EvoScientist.commands.implementation.model import ModelCommand

        ctx, cfg, ui = self._make_ctx_and_cfg(ollama_base_url="http://localhost:11434")
        ui.wait_for_model_pick = AsyncMock(return_value=("__custom_ollama__", "ollama"))

        async def fake_discover(base_url, *, timeout):
            return []

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.llm.ollama_discovery.discover_ollama_models",
                side_effect=fake_discover,
            ),
            patch("EvoScientist.cli.agent._load_agent") as load_agent,
        ):
            _run(ModelCommand().execute(ctx, []))

        load_agent.assert_not_called()
        assert cfg.model == "claude-sonnet-4-6"  # unchanged

    def test_picker_applies_detected_ollama_model(self):
        """User picks a live-detected Ollama model → _apply_model is invoked
        with (name, "ollama") and the agent is rebuilt."""
        from EvoScientist.commands.implementation.model import ModelCommand

        ctx, cfg, ui = self._make_ctx_and_cfg(ollama_base_url="http://localhost:11434")
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()
        ui.wait_for_model_pick = AsyncMock(return_value=("llama3.3", "ollama"))

        async def fake_discover(base_url, *, timeout):
            return ["llama3.3"]

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.llm.ollama_discovery.discover_ollama_models",
                side_effect=fake_discover,
            ),
            patch("EvoScientist.EvoScientist.set_chat_model"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
        ):
            _run(ModelCommand().execute(ctx, []))

        assert cfg.model == "llama3.3"
        assert cfg.provider == "ollama"
