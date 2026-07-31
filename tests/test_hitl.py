"""Tests for HITL (Human-in-the-Loop) approval mechanism."""

import asyncio
from unittest.mock import MagicMock, patch

from langgraph.types import Interrupt

from EvoScientist.stream.emitter import StreamEvent, StreamEventEmitter
from EvoScientist.stream.state import StreamState
from tests.stream_v3_fakes import (
    FakeV3Agent,
    collect_events,
    message_delta,
    protocol_event,
)

# =============================================================================
# StreamEventEmitter.interrupt()
# =============================================================================


class TestInterruptEmitter:
    def test_interrupt_event_structure(self):
        ev = StreamEventEmitter.interrupt(
            "main",
            [{"name": "execute", "args": {"command": "ls"}, "id": "tc1"}],
            [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
        )
        assert isinstance(ev, StreamEvent)
        assert ev.type == "interrupt"
        assert ev.data["type"] == "interrupt"
        assert ev.data["interrupt_id"] == "main"
        assert len(ev.data["action_requests"]) == 1
        assert ev.data["action_requests"][0]["name"] == "execute"
        assert len(ev.data["review_configs"]) == 1

    def test_interrupt_defaults_review_configs(self):
        ev = StreamEventEmitter.interrupt("default", [{"name": "execute"}])
        assert ev.data["review_configs"] == []

    def test_interrupt_multiple_action_requests(self):
        reqs = [
            {"name": "execute", "args": {"command": "ls"}},
            {"name": "write_file", "args": {"path": "/out.txt"}},
        ]
        ev = StreamEventEmitter.interrupt("main", reqs)
        assert len(ev.data["action_requests"]) == 2


# =============================================================================
# StreamState.handle_event("interrupt")
# =============================================================================


class TestStreamStateInterrupt:
    def test_handle_interrupt_sets_pending(self):
        state = StreamState()
        event = {
            "type": "interrupt",
            "interrupt_id": "main",
            "action_requests": [{"name": "execute", "args": {"command": "ls"}}],
            "review_configs": [],
        }
        result = state.handle_event(event)
        assert result == "interrupt"
        assert state.pending_interrupt is not None
        assert state.pending_interrupt["action_requests"][0]["name"] == "execute"

    def test_pending_interrupt_starts_none(self):
        state = StreamState()
        assert state.pending_interrupt is None

    def test_interrupt_does_not_affect_other_state(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "hello"})
        state.handle_event(
            {
                "type": "interrupt",
                "interrupt_id": "main",
                "action_requests": [{"name": "execute"}],
                "review_configs": [],
            }
        )
        assert state.response_text == "hello"
        assert state.pending_interrupt is not None

    def test_done_after_interrupt_preserves_pending(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "interrupt",
                "interrupt_id": "main",
                "action_requests": [{"name": "execute"}],
                "review_configs": [],
            }
        )
        state.handle_event({"type": "done", "response": ""})
        # pending_interrupt should still be set
        assert state.pending_interrupt is not None


# =============================================================================
# _resolve_hitl_approval
# =============================================================================


class TestResolveHitlApproval:
    def test_empty_requests_auto_approves(self):
        from EvoScientist.stream.display import _resolve_hitl_approval

        result = _resolve_hitl_approval({"action_requests": []})
        assert result == [{"type": "approve"}]

    def test_session_auto_approve(self):
        import EvoScientist.stream.display as disp

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = True
            result = disp._resolve_hitl_approval(
                {
                    "action_requests": [
                        {"name": "execute", "args": {"command": "rm -rf /"}}
                    ],
                }
            )
            assert result == [{"type": "approve"}]
        finally:
            disp._session_auto_approve = original

    def test_config_auto_approve(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = True
            mock_cfg.shell_allow_list = ""
            mock_cfg.dangerous_mode = False
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {
                        "action_requests": [
                            {"name": "execute", "args": {"command": "rm"}}
                        ],
                    }
                )
            assert result == [{"type": "approve"}]
        finally:
            disp._session_auto_approve = original

    def test_non_execute_tool_auto_approves(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = ""
            mock_cfg.dangerous_mode = False
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {
                        "action_requests": [
                            {"name": "write_file", "args": {"path": "/out.txt"}}
                        ],
                    }
                )
            assert result == [{"type": "approve"}]
        finally:
            disp._session_auto_approve = original

    def test_execute_with_matching_allow_list(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = "ls,cat,python"
            mock_cfg.dangerous_mode = False
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {
                        "action_requests": [
                            {"name": "execute", "args": {"command": "ls -la"}}
                        ],
                    }
                )
            assert result == [{"type": "approve"}]
        finally:
            disp._session_auto_approve = original

    def test_execute_not_in_allow_list_prompts(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = "ls,cat"
            mock_cfg.dangerous_mode = False
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                with patch(
                    "EvoScientist.stream.display._prompt_hitl_approval"
                ) as mock_prompt:
                    mock_prompt.return_value = [{"type": "approve"}]
                    result = _resolve_hitl_approval(
                        {
                            "action_requests": [
                                {"name": "execute", "args": {"command": "rm -rf /"}}
                            ],
                        }
                    )
            assert result == [{"type": "approve"}]
            mock_prompt.assert_called_once()
        finally:
            disp._session_auto_approve = original

    def test_run_in_background_not_in_allow_list_prompts(self):
        """run_in_background must NOT auto-approve — it runs shell like execute."""
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = "ls,cat"
            mock_cfg.dangerous_mode = False
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                with patch(
                    "EvoScientist.stream.display._prompt_hitl_approval"
                ) as mock_prompt:
                    mock_prompt.return_value = [{"type": "approve"}]
                    result = _resolve_hitl_approval(
                        {
                            "action_requests": [
                                {
                                    "name": "run_in_background",
                                    "args": {"command": "rm -rf /"},
                                }
                            ],
                        }
                    )
            assert result == [{"type": "approve"}]
            mock_prompt.assert_called_once()  # prompted, not silently approved
        finally:
            disp._session_auto_approve = original

    def test_run_in_background_in_allow_list_auto_approves(self):
        """An allow-listed command still auto-approves for run_in_background."""
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = "python"
            mock_cfg.dangerous_mode = False
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {
                        "action_requests": [
                            {
                                "name": "run_in_background",
                                "args": {"command": "python train.py"},
                            }
                        ],
                    }
                )
            assert result == [{"type": "approve"}]
        finally:
            disp._session_auto_approve = original

    def test_delete_only_request_prompts_not_auto_approved(self):
        """delete is armed in HITL_INTERRUPT_ON but has no `command` arg, so the
        shell_allow_list carve-out must never silently clear it (C1)."""
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = ""
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                with patch(
                    "EvoScientist.stream.display._prompt_hitl_approval"
                ) as mock_prompt:
                    mock_prompt.return_value = [{"type": "approve"}]
                    result = _resolve_hitl_approval(
                        {
                            "action_requests": [
                                {"name": "delete", "args": {"file_path": "/f.txt"}}
                            ],
                        }
                    )
            assert result == [{"type": "approve"}]
            mock_prompt.assert_called_once()  # must be prompted, not auto-approved
        finally:
            disp._session_auto_approve = original


# =============================================================================
# Config fields
# =============================================================================


class TestHitlConfig:
    def test_auto_approve_default(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig()
        assert cfg.auto_approve is False

    def test_auto_mode_default(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig()
        assert cfg.auto_mode is False

    def test_shell_allow_list_default(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig()
        assert cfg.shell_allow_list == ""

    def test_auto_approve_set(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig(auto_approve=True)
        assert cfg.auto_approve is True

    def test_auto_mode_set(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig(auto_mode=True)
        assert cfg.auto_mode is True

    def test_shell_allow_list_set(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig(shell_allow_list="ls,cat,python")
        assert cfg.shell_allow_list == "ls,cat,python"


# =============================================================================
# Interrupt event parsing in stream_agent_events
# =============================================================================


class TestInterruptEventParsing:
    async def test_interrupt_from_updates_mode(self):
        """__interrupt__ in updates mode yields interrupt event."""
        interrupt_data = {
            "__interrupt__": [
                Interrupt(
                    value={
                        "action_requests": [
                            {"name": "execute", "args": {"command": "ls"}, "id": "tc1"}
                        ],
                        "review_configs": [
                            {
                                "action_name": "execute",
                                "allowed_decisions": ["approve", "reject"],
                            }
                        ],
                    },
                    id="main",
                )
            ]
        }

        agent = FakeV3Agent(
            [
                message_delta("thinking..."),
                protocol_event("updates", interrupt_data),
            ]
        )
        events = await collect_events(agent, message="test", thread_id="thread-1")

        types = [e["type"] for e in events]
        assert "interrupt" in types

        interrupt_ev = next(e for e in events if e["type"] == "interrupt")
        assert len(interrupt_ev["action_requests"]) == 1
        assert interrupt_ev["action_requests"][0]["name"] == "execute"
        assert interrupt_ev["interrupt_id"] == "main"

    async def test_updates_without_interrupt_skipped(self):
        """Regular updates mode data is skipped as before."""
        agent = FakeV3Agent(
            [
                protocol_event("updates", {"some_node": {"key": "value"}}),
            ]
        )
        events = await collect_events(agent, message="test", thread_id="thread-1")

        types = [e["type"] for e in events]
        assert "interrupt" not in types
        # Should only have done event
        assert types == ["done"]


# =============================================================================
# Channel consumer HITL helpers
# =============================================================================


class TestConsumerHitlHelpers:
    def test_parse_approval_approve(self):
        from EvoScientist.channels.interaction import parse_approval_reply

        for text in ("1", "y", "yes", "approve", "ok", " 1 ", "  Y  "):
            assert parse_approval_reply(text) == "approve", f"Failed for: {text!r}"

    def test_parse_approval_reject(self):
        from EvoScientist.channels.interaction import parse_approval_reply

        for text in ("2", "n", "no", "reject"):
            assert parse_approval_reply(text) == "reject", f"Failed for: {text!r}"

    def test_parse_approval_auto(self):
        from EvoScientist.channels.interaction import parse_approval_reply

        for text in ("3", "a", "auto", "approve all"):
            assert parse_approval_reply(text) == "auto", f"Failed for: {text!r}"

    def test_parse_approval_unrecognized(self):
        from EvoScientist.channels.interaction import parse_approval_reply

        assert parse_approval_reply("hello world") is None
        assert parse_approval_reply("") is None
        assert parse_approval_reply("maybe") is None

    def test_format_approval_prompt(self):
        from EvoScientist.channels.interaction import format_approval_prompt

        prompt = format_approval_prompt(
            [
                {"name": "execute", "args": {"command": "ls -la"}},
            ]
        )
        assert "Approval Required" in prompt
        assert "execute" in prompt
        assert "ls -la" in prompt
        assert "1=Approve" in prompt
        assert "2=Reject" in prompt

    def test_format_approval_prompt_multiple(self):
        from EvoScientist.channels.interaction import format_approval_prompt

        prompt = format_approval_prompt(
            [
                {"name": "execute", "args": {"command": "ls"}},
                {"name": "write_file", "args": {"path": "/out.txt"}},
            ]
        )
        assert "1. execute: ls" in prompt
        assert "2. write_file: /out.txt" in prompt

    def test_should_auto_approve_non_execute(self):
        from EvoScientist.channels.interaction import config_auto_approve

        assert config_auto_approve([{"name": "write_file", "args": {}}]) is True

    def test_should_auto_approve_empty(self):
        from EvoScientist.channels.interaction import config_auto_approve

        assert config_auto_approve([]) is True

    def test_should_auto_approve_execute_no_allowlist(self):
        from EvoScientist.channels.interaction import config_auto_approve

        # With default config (auto_approve=False, shell_allow_list=""),
        # execute should NOT auto-approve
        mock_cfg = MagicMock()
        mock_cfg.auto_approve = False
        mock_cfg.shell_allow_list = ""
        mock_cfg.dangerous_mode = False
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = config_auto_approve(
                [
                    {"name": "execute", "args": {"command": "rm -rf /"}},
                ]
            )
        assert result is False

    def test_should_auto_approve_run_in_background_no_allowlist(self):
        """Channel path must NOT auto-approve run_in_background (same as execute)."""
        from EvoScientist.channels.interaction import config_auto_approve

        mock_cfg = MagicMock()
        mock_cfg.auto_approve = False
        mock_cfg.shell_allow_list = ""
        mock_cfg.dangerous_mode = False
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = config_auto_approve(
                [
                    {"name": "run_in_background", "args": {"command": "rm -rf /"}},
                ]
            )
        assert result is False

    def test_should_auto_approve_config_true(self):
        from EvoScientist.channels.interaction import config_auto_approve

        mock_cfg = MagicMock()
        mock_cfg.auto_approve = True
        mock_cfg.dangerous_mode = False
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = config_auto_approve(
                [
                    {"name": "execute", "args": {"command": "rm -rf /"}},
                ]
            )
        assert result is True

    def test_should_auto_approve_allowlist_match(self):
        from EvoScientist.channels.interaction import config_auto_approve

        mock_cfg = MagicMock()
        mock_cfg.auto_approve = False
        mock_cfg.shell_allow_list = "ls,python"
        mock_cfg.dangerous_mode = False
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = config_auto_approve(
                [
                    {"name": "execute", "args": {"command": "ls -la"}},
                ]
            )
        assert result is True

    def test_should_auto_approve_delete_not_cleared(self):
        """delete has no `command` arg, so shell_allow_list must never clear it
        the way it clears execute (C1)."""
        from EvoScientist.channels.interaction import config_auto_approve

        mock_cfg = MagicMock()
        mock_cfg.auto_approve = False
        mock_cfg.shell_allow_list = "ls,python"
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = config_auto_approve(
                [
                    {"name": "delete", "args": {"file_path": "/f.txt"}},
                ]
            )
        assert result is False


# =============================================================================
# Channel reply-interception mechanism (channel.py PendingReplyRegistry)
# =============================================================================
# The CLI bridge routes prompt replies through the shared asyncio-based
# ``PendingReplyRegistry`` on the bus loop (replacing the old threading.Event
# ``_pending_hitl`` globals). ``_bus_inbound_consumer`` feeds it via
# ``try_resolve`` ahead of normal enqueue.


class TestChannelReplyRegistry:
    async def test_register_and_resolve_reply(self):
        from EvoScientist.cli import channel as channel_mod

        reg = channel_mod._reply_registry
        reg.clear()

        async def _resolver():
            await asyncio.sleep(0.01)  # let wait() register first
            assert reg.try_resolve("telegram:chat123", "1") is True

        got, _ = await asyncio.gather(
            reg.wait("telegram:chat123", timeout=1.0), _resolver()
        )
        assert got == "1"
        assert "telegram:chat123" not in reg

    def test_try_resolve_no_pending(self):
        from EvoScientist.cli import channel as channel_mod

        channel_mod._reply_registry.clear()
        # No pending wait — should not intercept.
        resolved = channel_mod._reply_registry.try_resolve("discord:no_pending", "y")
        assert resolved is False

    async def test_reply_timeout_returns_none(self):
        from EvoScientist.cli import channel as channel_mod

        reg = channel_mod._reply_registry
        reg.clear()
        # No reply delivered — wait should time out and clean up.
        got = await reg.wait("telegram:timeout_chat", timeout=0.02)
        assert got is None
        assert "telegram:timeout_chat" not in reg


# =============================================================================
# _resolve_hitl_approval with custom prompt_fn
# =============================================================================


class TestResolveHitlApprovalWithPromptFn:
    def test_prompt_fn_called_for_execute(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = ""
            mock_cfg.dangerous_mode = False
            custom_decisions = [{"type": "approve"}]
            mock_fn = MagicMock(return_value=custom_decisions)
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {
                        "action_requests": [
                            {"name": "execute", "args": {"command": "rm -rf /"}}
                        ]
                    },
                    prompt_fn=mock_fn,
                )
            assert result == custom_decisions
            mock_fn.assert_called_once()
        finally:
            disp._session_auto_approve = original

    def test_prompt_fn_not_called_for_auto_approve(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = True
            mock_cfg.dangerous_mode = False
            mock_fn = MagicMock()
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {
                        "action_requests": [
                            {"name": "execute", "args": {"command": "rm"}}
                        ]
                    },
                    prompt_fn=mock_fn,
                )
            assert result == [{"type": "approve"}]
            mock_fn.assert_not_called()
        finally:
            disp._session_auto_approve = original

    def test_prompt_fn_not_called_for_non_execute(self):
        import EvoScientist.stream.display as disp
        from EvoScientist.stream.display import _resolve_hitl_approval

        original = disp._session_auto_approve
        try:
            disp._session_auto_approve = False
            mock_cfg = MagicMock()
            mock_cfg.auto_approve = False
            mock_cfg.shell_allow_list = ""
            mock_cfg.dangerous_mode = False
            mock_fn = MagicMock()
            with patch(
                "EvoScientist.config.settings.load_config", return_value=mock_cfg
            ):
                result = _resolve_hitl_approval(
                    {"action_requests": [{"name": "write_file", "args": {}}]},
                    prompt_fn=mock_fn,
                )
            assert result == [{"type": "approve"}]
            mock_fn.assert_not_called()
        finally:
            disp._session_auto_approve = original


# =============================================================================
# _build_hitl_interrupt_on
# =============================================================================


class TestInterruptOnWiring:
    """interrupt_on must be passed natively and gated on auto_approve."""

    def test_hitl_interrupt_on_helper_gates_on_auto_approve(self):
        from EvoScientist.EvoScientist import _build_hitl_interrupt_on

        assert _build_hitl_interrupt_on(auto_approve=True) is None

    def test_hitl_interrupt_on_helper_returns_shell_tools(self):
        from EvoScientist.EvoScientist import _build_hitl_interrupt_on

        cfg = _build_hitl_interrupt_on(auto_approve=False)
        assert cfg == {
            "execute": True,
            "run_in_background": True,
            "schedule_task": True,
            "delete": True,
        }

    def test_auto_mode_implies_auto_approve_so_nothing_is_armed(self):
        """auto_mode must imply auto_approve from ANY source (not just the CLI
        flag), so a config-file / direct-construction auto_mode run arms no
        interrupt and never prompts."""
        from EvoScientist.config.settings import EvoScientistConfig
        from EvoScientist.EvoScientist import _build_hitl_interrupt_on

        cfg = EvoScientistConfig(auto_mode=True)
        assert cfg.auto_approve is True
        assert _build_hitl_interrupt_on(auto_approve=cfg.auto_approve) is None

    def test_hitl_interrupt_on_reaches_create_deep_agent(self):
        """The kwarg must actually reach ``create_deep_agent`` — not just the
        pure helper — so a future edit that drops it or re-adds a bare
        ``HumanInTheLoopMiddleware`` append gets caught."""
        import EvoScientist.EvoScientist as es_mod
        from EvoScientist.EvoScientist import _build_hitl_interrupt_on

        captured = []

        def fake_create_deep_agent(**kwargs):
            captured.append(kwargs.get("interrupt_on", "MISSING"))
            agent = MagicMock()
            agent.with_config.return_value = agent
            return agent

        for auto_approve in (False, True):
            cfg = MagicMock()
            cfg.auto_approve = auto_approve
            cfg.dangerous_mode = False
            cfg.sandbox_execute_timeout = 300
            cfg.recursion_limit = 100

            with patch(
                "deepagents.create_deep_agent", side_effect=fake_create_deep_agent
            ):
                with patch.object(es_mod, "_apply_env_from_config"):
                    with patch.object(
                        es_mod, "_get_default_middleware", return_value=[]
                    ):
                        with patch.object(
                            es_mod,
                            "load_mcp_and_build_kwargs",
                            return_value={"name": "x"},
                        ):
                            es_mod.create_cli_agent(
                                workspace_dir="/tmp/test-interrupt-on-wiring",
                                config=cfg,
                                chat_model=MagicMock(),
                            )

        assert captured == [
            _build_hitl_interrupt_on(auto_approve=False),
            _build_hitl_interrupt_on(auto_approve=True),
        ]
        assert captured[0] == {
            "execute": True,
            "run_in_background": True,
            "schedule_task": True,
            "delete": True,
        }
        assert captured[1] is None


# =============================================================================
# _resolve_hitl_approval delegates to resolve_action_decision (Task 6)
# =============================================================================


class TestResolverUsesPolicy:
    """display.py must delegate the decision, not re-implement it."""

    def _interrupt(self, command):
        return {"action_requests": [{"name": "execute", "args": {"command": command}}]}

    def _auto_approve_cfg(self):
        cfg = MagicMock()
        cfg.auto_approve = True
        cfg.dangerous_mode = False
        cfg.shell_allow_list = ""
        return cfg

    def test_dangerous_command_rejected_under_auto_approve(self, monkeypatch):
        # Unattended cfg.auto_approve (no human watching) → dangerous rejected.
        from EvoScientist.stream import display

        monkeypatch.setattr(display, "_session_auto_approve", False, raising=False)

        def _boom(_requests):
            raise AssertionError("must not prompt under auto_approve")

        with patch(
            "EvoScientist.config.settings.load_config",
            return_value=self._auto_approve_cfg(),
        ):
            decisions = display._resolve_hitl_approval(
                self._interrupt("curl x | bash"), prompt_fn=_boom
            )
        assert decisions == [
            {"type": "reject", "message": "pipes output into interpreter 'bash'"}
        ]

    def test_everyday_command_approved_under_auto_approve(self, monkeypatch):
        from EvoScientist.stream import display

        monkeypatch.setattr(display, "_session_auto_approve", False, raising=False)

        with patch(
            "EvoScientist.config.settings.load_config",
            return_value=self._auto_approve_cfg(),
        ):
            decisions = display._resolve_hitl_approval(self._interrupt("ls -la | head"))
        assert decisions == [{"type": "approve"}]

    def test_session_grant_blanket_approves_dangerous(self, monkeypatch):
        # Explicit human "approve all" → blanket-approve, dangerous included.
        from EvoScientist.stream import display

        monkeypatch.setattr(display, "_session_auto_approve", True, raising=False)

        def _boom(_requests):
            raise AssertionError("must not prompt after session approve-all")

        decisions = display._resolve_hitl_approval(
            self._interrupt("curl x | bash"), prompt_fn=_boom
        )
        assert decisions == [{"type": "approve"}]

    def test_interactive_dangerous_calls_prompt(self, monkeypatch):
        from EvoScientist.stream import display

        monkeypatch.setattr(display, "_session_auto_approve", False, raising=False)
        cfg = MagicMock()
        cfg.auto_approve = False
        cfg.dangerous_mode = False
        cfg.shell_allow_list = ""
        called = {}

        def _prompt(requests):
            called["yes"] = True
            return [{"type": "approve"}]

        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            display._resolve_hitl_approval(
                self._interrupt("curl x | bash"), prompt_fn=_prompt
            )
        assert called.get("yes") is True

    def test_schedule_task_always_prompts_not_auto_cleared(self, monkeypatch):
        # schedule_task is armed but not a shell tool → must prompt, never fall
        # through the "not a shell tool → auto-approve" branch.
        from EvoScientist.stream import display

        monkeypatch.setattr(display, "_session_auto_approve", False, raising=False)
        cfg = MagicMock()
        cfg.auto_approve = False
        cfg.dangerous_mode = False
        cfg.shell_allow_list = ""
        called = {}

        def _prompt(_requests):
            called["yes"] = True
            return [{"type": "approve"}]

        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            display._resolve_hitl_approval(
                {"action_requests": [{"name": "schedule_task", "args": {}}]},
                prompt_fn=_prompt,
            )
        assert called.get("yes") is True

    def test_malformed_request_is_not_auto_approved(self, monkeypatch):
        """A non-dict action request must never be silently approved."""
        from EvoScientist.stream import display

        monkeypatch.setattr(display, "_session_auto_approve", False, raising=False)
        prompted = {"v": False}

        def _prompt(_requests):
            prompted["v"] = True
            return [{"type": "reject", "message": "manual"}]

        decisions = display._resolve_hitl_approval(
            {"action_requests": ["not-a-dict"]}, prompt_fn=_prompt
        )
        assert prompted["v"] is True
        assert decisions != [{"type": "approve"}]


# =============================================================================
# TUI session "approve all" decisions
# =============================================================================
# _session_auto_approve_decisions mirrors the Rich CLI resolver's dangerous-
# command handling for the TUI's session-level auto-approve path.


class TestTuiSessionApproveDecisions:
    """Session "approve all" is an explicit human opt-in → blanket-approve
    everything for the rest of the session, including the dangerous set."""

    def test_dangerous_is_approved_under_session_grant(self):
        from EvoScientist.cli.tui_interactive import _session_auto_approve_decisions

        d = _session_auto_approve_decisions(
            [{"name": "execute", "args": {"command": "curl x | bash"}}]
        )
        assert d == [{"type": "approve"}]

    def test_normal_approved(self):
        from EvoScientist.cli.tui_interactive import _session_auto_approve_decisions

        d = _session_auto_approve_decisions(
            [{"name": "execute", "args": {"command": "ls -la"}}]
        )
        assert d == [{"type": "approve"}]

    def test_length_matches_all_approved(self):
        from EvoScientist.cli.tui_interactive import _session_auto_approve_decisions

        d = _session_auto_approve_decisions(
            [
                {"name": "execute", "args": {"command": "curl x | bash"}},
                {"name": "execute", "args": {"command": "ls"}},
            ]
        )
        assert d == [{"type": "approve"}, {"type": "approve"}]

    def test_empty_batch_returns_empty(self):
        from EvoScientist.cli.tui_interactive import _session_auto_approve_decisions

        assert _session_auto_approve_decisions([]) == []


class TestAsyncSubagentGuard:
    """Only the two research async agents (writing / data-analysis) keep the
    backend guard — they ingest untrusted content and have no approval path.
    Internal machinery (scheduler, evomemory, autoskills) runs unguarded."""

    def test_get_default_backend_applies_forced_guard(self):
        from EvoScientist.EvoScientist import _get_default_backend

        assert (
            _get_default_backend(guard_dangerous=True).default._guard_dangerous is True
        )
        assert (
            _get_default_backend(guard_dangerous=False).default._guard_dangerous
            is False
        )

    def test_get_default_backend_defaults_to_config_auto_approve(self):
        from EvoScientist.EvoScientist import _ensure_config, _get_default_backend

        # No explicit guard → follows cfg.auto_approve (Task 3 behaviour preserved).
        backend = _get_default_backend()
        assert backend.default._guard_dangerous == _ensure_config().auto_approve

    @staticmethod
    def _factory_kwargs_for(name: str) -> dict:
        """Run the async factory for ``name`` and capture the backend kwargs."""
        from unittest.mock import MagicMock, patch

        import EvoScientist.EvoScientist as ev
        from EvoScientist.subagents import _factory

        captured: dict = {}

        def _spy_backend(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch.object(ev, "_get_default_backend", _spy_backend),
            patch(
                "EvoScientist.utils.load_subagents",
                return_value=[{"name": name, "system_prompt": "x", "tools": []}],
            ),
            patch.object(ev, "_load_mcp_tools_cached", return_value={}),
            patch.object(ev, "_get_default_middleware", return_value=[]),
            patch.object(ev, "_ensure_general_purpose_subagent", lambda subs: None),
            patch.object(ev, "_inject_subagent_middleware", lambda subs: None),
            patch.object(ev, "_ensure_chat_model", return_value=MagicMock()),
            patch.object(ev, "_ensure_auxiliary_chat_model", return_value=MagicMock()),
            patch("deepagents.create_deep_agent", return_value=MagicMock()),
        ):
            _factory.build_async_subagent_graph(name)

        return captured

    def test_async_factory_guards_research_agents(self):
        # Research async agents keep both the dangerous-command guard AND the
        # delete refusal on (no interactive approval path → relay to orchestrator).
        for name in ("writing-agent", "data-analysis-agent"):
            kw = self._factory_kwargs_for(name)
            assert kw.get("guard_dangerous") is True
            assert kw.get("refuse_delete") is True

    def test_async_factory_does_not_guard_internal_agents(self):
        # Scheduler and any other internal async graph run unguarded in any mode.
        kw = self._factory_kwargs_for("scheduler")
        assert kw.get("guard_dangerous") is False
        assert kw.get("refuse_delete") is False


class TestOrchestratorRelayGuidance:
    """The orchestrator needs the recovery path spelled out in its prompt."""

    def test_delegation_prompt_explains_blocked_command_relay(self):
        from EvoScientist.prompts import DELEGATION_STRATEGY

        text = DELEGATION_STRATEGY.lower()
        assert "update_async_task" in text
        assert "blocked" in text


class TestHitlResumeKeying:
    """The HITL resume payload must be keyed by interrupt_id so parallel
    sub-agent interrupts don't hit langgraph's multi-pending-interrupt crash."""

    def test_build_hitl_resume_is_id_keyed(self):
        from EvoScientist.backends import build_hitl_resume

        cmd = build_hitl_resume("abc123", [{"type": "approve"}])
        assert cmd.resume == {"abc123": {"decisions": [{"type": "approve"}]}}

    def test_two_parallel_subagent_interrupts_drain_without_crash(self):
        """Integration guard against the exact regression: 2 declarative
        sub-agents each calling execute leave 2 pending interrupts; a flat
        resume raises 'multiple pending interrupts'. Resuming one id at a time
        (what build_hitl_resume produces) drains them cleanly."""
        import uuid

        from deepagents import create_deep_agent
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel,
        )
        from langchain_core.messages import AIMessage, ToolCall
        from langgraph.checkpoint.memory import InMemorySaver

        from EvoScientist.backends import build_hitl_resume

        class _SM(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        def _mk(s):
            return _SM(responses=s)

        top = AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="task",
                    args={"description": "A", "subagent_type": "agent-a"},
                    id="t1",
                ),
                ToolCall(
                    name="task",
                    args={"description": "B", "subagent_type": "agent-b"},
                    id="t2",
                ),
            ],
        )
        tf = AIMessage(content="done")
        sub = AIMessage(
            content="",
            tool_calls=[ToolCall(name="execute", args={"command": "echo hi"}, id="e1")],
        )
        sf = AIMessage(content="sub done")
        agent = create_deep_agent(
            model=_mk([top] + [tf] * 6),
            subagents=[
                {
                    "name": "agent-a",
                    "description": "a",
                    "system_prompt": "a",
                    "model": _mk([sub, sf, sf]),
                },
                {
                    "name": "agent-b",
                    "description": "b",
                    "system_prompt": "b",
                    "model": _mk([sub, sf, sf]),
                },
            ],
            interrupt_on={"execute": True},
            checkpointer=InMemorySaver(),
        )
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        res = agent.invoke({"messages": [("user", "go")]}, config=cfg)
        ints = res.get("__interrupt__", [])
        assert len(ints) == 2  # the regression precondition

        for _ in range(5):
            ints = res.get("__interrupt__", [])
            if not ints:
                break
            res = agent.invoke(
                build_hitl_resume(ints[0].id, [{"type": "approve"}]), config=cfg
            )
        assert not res.get("__interrupt__", [])  # drained, no crash
