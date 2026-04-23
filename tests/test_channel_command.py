"""Tests for the /channel command (ChannelCommand)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.conftest import run_async as _run


def _ctx():
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(
        agent=object(), thread_id="tid-42", ui=ui, workspace_dir="/ws"
    ), ui


class TestNeedsAgent:
    """status + stop must not require the agent (recovery from broken load)."""

    def test_status_does_not_need_agent(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        assert ChannelCommand().needs_agent(["status"]) is False

    def test_stop_does_not_need_agent(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        assert ChannelCommand().needs_agent(["stop"]) is False

    def test_stop_with_target_does_not_need_agent(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        assert ChannelCommand().needs_agent(["stop", "telegram"]) is False

    def test_start_subcommand_needs_agent(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        assert ChannelCommand().needs_agent(["telegram"]) is True

    def test_no_args_needs_agent(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        # no subcmd → falls through to start flow, needs agent
        assert ChannelCommand().needs_agent([]) is True


class TestStartPath:
    """Start flow must propagate agent/thread_id globals."""

    def test_start_sets_cli_agent_globals(self, monkeypatch):
        import EvoScientist.cli.channel as _ch_mod
        from EvoScientist.commands.implementation.channel import ChannelCommand

        ctx, _ui = _ctx()
        config = SimpleNamespace(
            channel_enabled="telegram",
            channel_send_thinking=True,
        )

        # ``monkeypatch`` auto-restores on teardown so a raised assertion
        # can't leak ``None`` globals into subsequent tests.
        monkeypatch.setattr(_ch_mod, "_cli_agent", None)
        monkeypatch.setattr(_ch_mod, "_cli_thread_id", None)
        with (
            patch(
                "EvoScientist.cli.channel._channels_is_running",
                return_value=False,
            ),
            patch(
                "EvoScientist.cli.channel._start_channels_bus_mode",
            ),
            patch(
                "EvoScientist.config.load_config",
                return_value=config,
            ),
        ):
            _run(ChannelCommand().execute(ctx, ["telegram"]))
        assert _ch_mod._cli_agent is ctx.agent
        assert _ch_mod._cli_thread_id == "tid-42"

    def test_start_propagates_send_thinking(self):
        """send_thinking flag must reach _start_channels_bus_mode."""
        from EvoScientist.commands.implementation.channel import ChannelCommand

        ctx, _ui = _ctx()
        config = SimpleNamespace(
            channel_enabled="telegram",
            channel_send_thinking=False,
        )
        captured = {}

        def _fake_start(cfg, agent, thread_id, *, send_thinking=None):
            captured["agent"] = agent
            captured["thread_id"] = thread_id
            captured["send_thinking"] = send_thinking

        with (
            patch(
                "EvoScientist.cli.channel._channels_is_running",
                return_value=False,
            ),
            patch(
                "EvoScientist.cli.channel._start_channels_bus_mode",
                _fake_start,
            ),
            patch(
                "EvoScientist.config.load_config",
                return_value=config,
            ),
        ):
            _run(ChannelCommand().execute(ctx, ["telegram"]))
        assert captured["agent"] is ctx.agent
        assert captured["thread_id"] == "tid-42"
        assert captured["send_thinking"] is False


class TestAddToRunningPath:
    def test_add_to_running_sets_cli_agent_globals(self, monkeypatch):
        import EvoScientist.cli.channel as _ch_mod
        from EvoScientist.commands.implementation.channel import ChannelCommand

        ctx, _ui = _ctx()
        config = SimpleNamespace(
            channel_enabled="telegram",
            channel_send_thinking=False,
        )

        monkeypatch.setattr(_ch_mod, "_cli_agent", None)
        monkeypatch.setattr(_ch_mod, "_cli_thread_id", None)
        with (
            patch(
                "EvoScientist.cli.channel._channels_is_running",
                return_value=True,
            ),
            patch(
                "EvoScientist.cli.channel._add_channel_to_running_bus",
            ),
            patch(
                "EvoScientist.config.load_config",
                return_value=config,
            ),
        ):
            _run(ChannelCommand().execute(ctx, ["discord"]))
        assert _ch_mod._cli_agent is ctx.agent
        assert _ch_mod._cli_thread_id == "tid-42"

    def test_add_to_running_propagates_send_thinking(self):
        """Adding to a running bus must honor config.channel_send_thinking."""
        from EvoScientist.commands.implementation.channel import ChannelCommand

        ctx, _ui = _ctx()
        config = SimpleNamespace(
            channel_enabled="telegram",
            channel_send_thinking=True,
        )
        captured = {}

        def _fake_add(channel_type, cfg, *, send_thinking=None):
            captured["channel_type"] = channel_type
            captured["send_thinking"] = send_thinking

        with (
            patch(
                "EvoScientist.cli.channel._channels_is_running",
                return_value=True,
            ),
            patch(
                "EvoScientist.cli.channel._add_channel_to_running_bus",
                _fake_add,
            ),
            patch(
                "EvoScientist.config.load_config",
                return_value=config,
            ),
        ):
            _run(ChannelCommand().execute(ctx, ["discord"]))
        assert captured["channel_type"] == "discord"
        assert captured["send_thinking"] is True


class TestStatusPath:
    def test_status_without_running_channels(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        ctx, ui = _ctx()
        config = SimpleNamespace(channel_enabled="", channel_send_thinking=False)
        with (
            patch(
                "EvoScientist.cli.channel._channels_is_running",
                return_value=False,
            ),
            patch(
                "EvoScientist.cli.channel._channels_running_list",
                return_value=[],
            ),
            patch(
                "EvoScientist.config.load_config",
                return_value=config,
            ),
        ):
            _run(ChannelCommand().execute(ctx, ["status"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("No messaging channels" in m for m in msgs)
