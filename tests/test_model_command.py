"""Tests for the /model command and extract_model_and_provider helper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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

    async def test_unknown_model_shows_error(self):
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
            await cmd.execute(ctx, ["nonexistent-model-xyz"])

        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Unknown model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


class TestModelCommandPickerCancelled:
    """Verify no-op when the interactive picker is cancelled."""

    async def test_picker_returns_none(self):
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
            await cmd.execute(ctx, [])

        # No model switch should have happened
        ui.append_system.assert_not_called()


class TestModelCommandSwitch:
    """Verify a successful model switch updates config and rebuilds agent."""

    async def test_switch_known_model(self):
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
            patch("EvoScientist.EvoScientist._build_chat_model"),
            patch("EvoScientist.EvoScientist.set_active_config") as set_cfg,
            patch("EvoScientist.EvoScientist.set_chat_model_instance"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=new_agent,
            ),
        ):
            await cmd.execute(ctx, ["claude-opus-4-8"])

        # The switch is committed via set_active_config(temp_cfg), not by
        # mutating the original cfg object in place.
        set_cfg.assert_called_once()
        committed = set_cfg.call_args[0][0]
        assert committed.model == "claude-opus-4-8"
        assert committed.provider == "anthropic"

        # Agent should be replaced on context
        assert ctx.agent == new_agent

        # Success message shown
        ui.append_system.assert_called_once()
        msg = ui.append_system.call_args[0][0]
        assert "claude-opus-4-8" in msg
        assert "anthropic" in msg

    async def test_switch_with_save_flag(self):
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
            patch("EvoScientist.EvoScientist._build_chat_model"),
            patch("EvoScientist.EvoScientist.set_active_config"),
            patch("EvoScientist.EvoScientist.set_chat_model_instance"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch("EvoScientist.config.settings.set_config_value") as mock_save,
        ):
            await cmd.execute(ctx, ["claude-opus-4-8", "--save"])

        # Config file should be updated
        mock_save.assert_any_call("model", "claude-opus-4-8")
        mock_save.assert_any_call("provider", "anthropic")

        # Success message should mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" in msg

    async def test_switch_without_save_flag_does_not_persist(self):
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
            patch("EvoScientist.EvoScientist._build_chat_model"),
            patch("EvoScientist.EvoScientist.set_active_config"),
            patch("EvoScientist.EvoScientist.set_chat_model_instance"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch("EvoScientist.config.settings.set_config_value") as mock_save,
        ):
            await cmd.execute(ctx, ["claude-opus-4-8"])

        # Config file should NOT be updated
        mock_save.assert_not_called()

        # Message should not mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" not in msg


class TestModelCommandFailure:
    """Verify error handling when chat-model construction raises."""

    async def test_build_chat_model_error(self):
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
                "EvoScientist.EvoScientist._build_chat_model",
                side_effect=RuntimeError("API key missing"),
            ) as mock_build,
        ):
            await cmd.execute(ctx, ["claude-opus-4-8"])

        mock_build.assert_called_once()
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
    """End-to-end regression for #179 + #183: `_apply_model` must produce an
    agent bound to the NEW model, threaded in via the pure ``create_cli_agent``
    path.

    Exercises the real chain:
        ``_apply_model → _build_chat_model → _load_agent(chat_model=...) →
        set_active_config / set_chat_model_instance``

    Only ``_load_agent`` is faked (to avoid spinning up deepagents, MCP tools,
    middleware, and subagent YAML); it binds the ``chat_model`` it receives.
    ``get_chat_model`` returns a distinct sentinel per ``(model, provider)``
    pair so we can assert on identity.
    """

    async def test_new_agent_is_bound_to_newly_selected_model(self, evo_module_state):
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
            workspace_dir=None,
            checkpointer=None,
            config=None,
            chat_model=None,
            *,
            on_mcp_progress=None,
        ):
            # The pure path threads the freshly built chat model in; bind it
            # directly instead of re-deriving via _ensure_chat_model.
            agent = MagicMock(name="fake-agent")
            agent._bound_model = chat_model
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
            await cmd._apply_model(ctx, "minimax-m2.7", "openrouter")

        # The agent produced by _apply_model must be bound to the
        # NEWLY requested model, threaded in via chat_model=.
        assert ctx.agent._bound_model is not old_model
        assert ctx.agent._bound_model._bound_model == "minimax-m2.7"
        assert ctx.agent._bound_model._bound_provider == "openrouter"

        # Global state reflects the switch end-to-end, committed via the setters.
        assert mod._chat_model_key == ("minimax-m2.7", "openrouter")
        assert mod._chat_model is sentinels[("minimax-m2.7", "openrouter")]
        # The switch is applied to the LIVE cfg in place and ``_config`` stays
        # bound to that same object, so callers holding the active config by
        # reference (e.g. serve's ``agent_holder["config"]``) observe the swap.
        assert mod._config is cfg
        assert cfg.model == "minimax-m2.7"
        assert cfg.provider == "openrouter"

        # User-visible success message.
        msg = ctx.ui.append_system.call_args[0][0]
        assert "minimax-m2.7" in msg
        assert "openrouter" in msg


class TestApplyModelPreservesConfigByReference:
    """Regression for the din0s review on #267: serve mode (and any long-lived
    caller) holds the active config object by reference via
    ``agent_holder["config"]``.  The pure-path commit must apply the switch to
    that LIVE object in place — not rebind ``_config`` to a fresh ``temp_cfg``
    copy — otherwise a later workspace-changing ``/resume`` reloads the agent
    from the stale startup config and silently reverts the ``/model`` switch.

    Critically this must hold across *repeated* switches: the earlier
    "also mutate cfg but still rebind to temp_cfg" remedy only survives one
    switch (the held object stops being the active ``_config`` after the first).
    """

    async def test_held_config_reference_tracks_repeated_switches(
        self, evo_module_state
    ):
        from EvoScientist.commands.implementation.model import ModelCommand
        from EvoScientist.config.settings import EvoScientistConfig

        mod = evo_module_state

        def _fake_get_chat_model(model, provider=None):
            m = MagicMock(name=f"chat_model[{model}|{provider}]")
            m._bound_model = model
            return m

        def _fake_load_agent(
            workspace_dir=None,
            checkpointer=None,
            config=None,
            chat_model=None,
            *,
            on_mcp_progress=None,
        ):
            return MagicMock(name="fake-agent")

        cfg = EvoScientistConfig(model="claude-sonnet-4-6", provider="anthropic")
        mod._config = cfg
        mod._chat_model = None
        mod._chat_model_key = None
        mod._EvoScientist_agent = None

        # Simulate serve capturing the startup config object once (commands.py:
        # ``agent_holder = {... "config": config}``) and never re-reading it.
        agent_holder = {"config": cfg}

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.workspace_dir = "/tmp/test_byref"
        ctx.checkpointer = None

        cmd = ModelCommand()
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
            for model, provider in [
                ("claude-opus-4-8", "anthropic"),
                ("minimax-m2.7", "openrouter"),
                ("claude-sonnet-4-6", "anthropic"),
            ]:
                await cmd._apply_model(ctx, model, provider)
                # The held reference must reflect the LATEST switch on every
                # iteration — not just the first — and stay the active config.
                assert agent_holder["config"].model == model
                assert agent_holder["config"].provider == provider
                assert mod._config is agent_holder["config"]


class TestModelCommandLoadAgentFailure:
    """Verify the transactional ordering: when ``_load_agent`` raises,
    nothing downstream (``set_chat_model``, ``cfg`` mutation,
    ``set_config_value``) should happen.

    This is the core guarantee of the refactor that established
    "build agent first, commit state only on success". Without this test
    the ordering could silently regress (e.g. if ``_apply_model`` were
    reordered to call ``set_chat_model`` first)."""

    async def test_load_agent_error_is_transactional(self):
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
            patch("EvoScientist.EvoScientist._build_chat_model"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                side_effect=RuntimeError("agent build failed"),
            ) as mock_load,
            patch(
                "EvoScientist.EvoScientist.set_active_config",
            ) as mock_set_cfg,
            patch(
                "EvoScientist.EvoScientist.set_chat_model_instance",
            ) as mock_set_model,
            patch(
                "EvoScientist.config.settings.set_config_value",
            ) as mock_save,
        ):
            # Pass ``--save`` to strengthen the assertion: if the ordering
            # ever regresses, ``set_config_value`` would be called with
            # stale data.
            await cmd.execute(ctx, ["claude-opus-4-8", "--save"])

        # _load_agent was attempted (transactional first step).
        mock_load.assert_called_once()
        # Commit setters and downstream side-effects must NOT have happened.
        mock_set_cfg.assert_not_called()
        mock_set_model.assert_not_called()
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
    """Regression for #183: when agent construction fails, the session stays on
    the original model with no snapshot/restore.

    Because ``create_cli_agent(config=..., chat_model=...)`` is now pure — it
    writes none of the four config/model globals — and ``_apply_model`` commits
    only after a successful build, a failing ``_load_agent`` leaves all four
    globals untouched. This replaces the old snapshot/restore rollback test.

    Complements :class:`TestModelCommandLoadAgentFailure`, which asserts the
    downstream setters never run on failure.
    """

    async def test_globals_unchanged_when_load_agent_raises(self, evo_module_state):
        from EvoScientist.commands.implementation.model import ModelCommand
        from EvoScientist.config.settings import EvoScientistConfig

        mod = evo_module_state

        def _fake_load_agent(
            workspace_dir=None,
            checkpointer=None,
            config=None,
            chat_model=None,
            *,
            on_mcp_progress=None,
        ):
            # The pure path writes no globals; mimic a failure partway through
            # agent wiring (middleware build, deepagents, MCP reconnect, ...).
            raise RuntimeError("middleware build failed")

        cfg = EvoScientistConfig(model="claude-sonnet-4-6", provider="anthropic")
        old_model = MagicMock(name="old-model")
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
                "EvoScientist.EvoScientist._build_chat_model",
                return_value=MagicMock(name="new-model"),
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                side_effect=_fake_load_agent,
            ),
        ):
            cmd = ModelCommand()
            await cmd._apply_model(ctx, "minimax-m2.7", "openrouter")

        # All four globals are unchanged — nothing was committed.
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

    async def test_picker_entries_include_detected_ollama_models(self):
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
            await ModelCommand().execute(ctx, [])

        entries = ui.wait_for_model_pick.call_args[0][0]
        ollama_rows = [(n, mid, p) for (n, mid, p) in entries if p == "ollama"]
        assert ("llama3.3:latest", "llama3.3:latest", "ollama") in ollama_rows
        assert ("qwen3:8b", "qwen3:8b", "ollama") in ollama_rows
        assert (
            "Custom Ollama model...",
            "__custom_ollama__",
            "ollama",
        ) in ollama_rows

    async def test_picker_entries_include_sentinel_when_discovery_empty(self):
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
            await ModelCommand().execute(ctx, [])

        entries = ui.wait_for_model_pick.call_args[0][0]
        ollama_rows = [(n, mid, p) for (n, mid, p) in entries if p == "ollama"]
        assert ollama_rows == [
            ("Custom Ollama model...", "__custom_ollama__", "ollama")
        ]

    async def test_picker_skips_ollama_section_when_not_configured(self):
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
            await ModelCommand().execute(ctx, [])

        discovery.assert_not_called()
        entries = ui.wait_for_model_pick.call_args[0][0]
        assert not any(p == "ollama" for (_, _, p) in entries)

    async def test_picker_handles_cfg_without_ollama_base_url_attr(self):
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
            await ModelCommand().execute(ctx, [])

        discovery.assert_not_called()
        entries = ui.wait_for_model_pick.call_args[0][0]
        assert not any(p == "ollama" for (_, _, p) in entries)

    async def test_picker_sentinel_result_is_treated_as_cancel(self):
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
            await ModelCommand().execute(ctx, [])

        load_agent.assert_not_called()
        assert cfg.model == "claude-sonnet-4-6"  # unchanged

    async def test_picker_applies_detected_ollama_model(self):
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
            patch("EvoScientist.EvoScientist._build_chat_model"),
            patch("EvoScientist.EvoScientist.set_active_config") as set_cfg,
            patch("EvoScientist.EvoScientist.set_chat_model_instance"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
        ):
            await ModelCommand().execute(ctx, [])

        # Committed via set_active_config(temp_cfg); original cfg untouched.
        set_cfg.assert_called_once()
        committed = set_cfg.call_args[0][0]
        assert committed.model == "llama3.3"
        assert committed.provider == "ollama"
