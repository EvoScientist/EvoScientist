"""Tests for HITL (Human-in-the-Loop) approval mechanism."""

import asyncio
from unittest.mock import MagicMock, patch

from EvoScientist.stream.emitter import StreamEvent, StreamEventEmitter
from EvoScientist.stream.state import StreamState

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
# _matches_shell_allow_list
# =============================================================================


class TestMatchesShellAllowList:
    def test_matches_prefix(self):
        from EvoScientist.stream.display import _matches_shell_allow_list

        assert _matches_shell_allow_list("ls -la", ["ls", "cat"]) is True
        assert _matches_shell_allow_list("cat file.txt", ["ls", "cat"]) is True

    def test_no_match(self):
        from EvoScientist.stream.display import _matches_shell_allow_list

        assert _matches_shell_allow_list("rm -rf /", ["ls", "cat"]) is False

    def test_empty_allow_list(self):
        from EvoScientist.stream.display import _matches_shell_allow_list

        assert _matches_shell_allow_list("ls", []) is False

    def test_whitespace_handling(self):
        from EvoScientist.stream.display import _matches_shell_allow_list

        assert _matches_shell_allow_list("  ls -la", ["ls"]) is True

    def test_exact_match(self):
        from EvoScientist.stream.display import _matches_shell_allow_list

        assert _matches_shell_allow_list("python", ["python"]) is True

    def test_partial_word_match(self):
        from EvoScientist.stream.display import _matches_shell_allow_list

        # "ls" prefix matches "lsof" — this is by design (prefix matching)
        assert _matches_shell_allow_list("lsof", ["ls"]) is True


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


# =============================================================================
# Config fields
# =============================================================================


class TestHitlConfig:
    def test_auto_approve_default(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig()
        assert cfg.auto_approve is False

    def test_shell_allow_list_default(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig()
        assert cfg.shell_allow_list == ""

    def test_auto_approve_set(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig(auto_approve=True)
        assert cfg.auto_approve is True

    def test_shell_allow_list_set(self):
        from EvoScientist.config.settings import EvoScientistConfig

        cfg = EvoScientistConfig(shell_allow_list="ls,cat,python")
        assert cfg.shell_allow_list == "ls,cat,python"


# =============================================================================
# Interrupt event parsing in stream_agent_events
# =============================================================================


class TestInterruptEventParsing:
    def _run_async(self, coro):
        """Run async code with a fresh event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_interrupt_from_updates_mode(self):
        """__interrupt__ in updates mode yields interrupt event."""
        from langchain_core.messages import AIMessageChunk

        from EvoScientist.stream.events import stream_agent_events

        mock_agent = MagicMock()

        ai_chunk = AIMessageChunk(content="thinking...", id="msg1")

        interrupt_data = {
            "__interrupt__": [
                {
                    "value": {
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
                    "ns": ["main"],
                    "resumable": True,
                }
            ]
        }

        chunks = [
            ((), "messages", (ai_chunk, {})),
            ((), "updates", interrupt_data),
        ]

        async def fake_astream(*a, **kw):
            for c in chunks:
                yield c

        mock_agent.astream = fake_astream

        events = []

        async def collect():
            async for ev in stream_agent_events(mock_agent, "test", "thread-1"):
                events.append(ev)

        self._run_async(collect())

        types = [e["type"] for e in events]
        assert "interrupt" in types

        interrupt_ev = next(e for e in events if e["type"] == "interrupt")
        assert len(interrupt_ev["action_requests"]) == 1
        assert interrupt_ev["action_requests"][0]["name"] == "execute"
        assert interrupt_ev["interrupt_id"] == "main"

    def test_updates_without_interrupt_skipped(self):
        """Regular updates mode data is skipped as before."""
        from EvoScientist.stream.events import stream_agent_events

        mock_agent = MagicMock()

        chunks = [
            ((), "updates", {"some_node": {"key": "value"}}),
        ]

        async def fake_astream(*a, **kw):
            for c in chunks:
                yield c

        mock_agent.astream = fake_astream

        events = []

        async def collect():
            async for ev in stream_agent_events(mock_agent, "test", "thread-1"):
                events.append(ev)

        self._run_async(collect())

        types = [e["type"] for e in events]
        assert "interrupt" not in types
        # Should only have done event
        assert types == ["done"]


# =============================================================================
# Channel consumer HITL helpers
# =============================================================================


class TestConsumerHitlHelpers:
    def test_parse_approval_approve(self):
        from EvoScientist.channels.consumer import _parse_approval_reply

        for text in ("1", "y", "yes", "approve", "ok", " 1 ", "  Y  "):
            assert _parse_approval_reply(text) == "approve", f"Failed for: {text!r}"

    def test_parse_approval_reject(self):
        from EvoScientist.channels.consumer import _parse_approval_reply

        for text in ("2", "n", "no", "reject"):
            assert _parse_approval_reply(text) == "reject", f"Failed for: {text!r}"

    def test_parse_approval_auto(self):
        from EvoScientist.channels.consumer import _parse_approval_reply

        for text in ("3", "a", "auto", "approve all"):
            assert _parse_approval_reply(text) == "auto", f"Failed for: {text!r}"

    def test_parse_approval_unrecognized(self):
        from EvoScientist.channels.consumer import _parse_approval_reply

        assert _parse_approval_reply("hello world") is None
        assert _parse_approval_reply("") is None
        assert _parse_approval_reply("maybe") is None

    def test_format_approval_prompt(self):
        from EvoScientist.channels.consumer import _format_approval_prompt

        prompt = _format_approval_prompt(
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
        from EvoScientist.channels.consumer import _format_approval_prompt

        prompt = _format_approval_prompt(
            [
                {"name": "execute", "args": {"command": "ls"}},
                {"name": "write_file", "args": {"path": "/out.txt"}},
            ]
        )
        assert "1. execute: ls" in prompt
        assert "2. write_file: /out.txt" in prompt

    def test_should_auto_approve_non_execute(self):
        from EvoScientist.channels.consumer import _should_auto_approve

        assert _should_auto_approve([{"name": "write_file", "args": {}}]) is True

    def test_should_auto_approve_empty(self):
        from EvoScientist.channels.consumer import _should_auto_approve

        assert _should_auto_approve([]) is True

    def test_should_auto_approve_execute_no_allowlist(self):
        from EvoScientist.channels.consumer import _should_auto_approve

        # With default config (auto_approve=False, shell_allow_list=""),
        # execute should NOT auto-approve
        mock_cfg = MagicMock()
        mock_cfg.auto_approve = False
        mock_cfg.shell_allow_list = ""
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = _should_auto_approve(
                [
                    {"name": "execute", "args": {"command": "rm -rf /"}},
                ]
            )
        assert result is False

    def test_should_auto_approve_config_true(self):
        from EvoScientist.channels.consumer import _should_auto_approve

        mock_cfg = MagicMock()
        mock_cfg.auto_approve = True
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = _should_auto_approve(
                [
                    {"name": "execute", "args": {"command": "rm -rf /"}},
                ]
            )
        assert result is True

    def test_should_auto_approve_allowlist_match(self):
        from EvoScientist.channels.consumer import _should_auto_approve

        mock_cfg = MagicMock()
        mock_cfg.auto_approve = False
        mock_cfg.shell_allow_list = "ls,python"
        with patch("EvoScientist.config.settings.load_config", return_value=mock_cfg):
            result = _should_auto_approve(
                [
                    {"name": "execute", "args": {"command": "ls -la"}},
                ]
            )
        assert result is True


# =============================================================================
# Channel HITL intercept mechanism (channel.py)
# =============================================================================


class TestChannelHitlIntercept:
    def test_register_and_set_hitl_reply(self):
        from EvoScientist.cli.channel import (
            _pop_hitl_reply,
            _register_hitl_wait,
            _try_set_hitl_reply,
        )

        event = _register_hitl_wait("telegram", "chat123")
        assert not event.is_set()

        # Simulate reply arriving
        intercepted = _try_set_hitl_reply("telegram", "chat123", "1")
        assert intercepted is True
        assert event.is_set()

        reply = _pop_hitl_reply("telegram", "chat123")
        assert reply == "1"

    def test_try_set_hitl_reply_no_pending(self):
        from EvoScientist.cli.channel import _try_set_hitl_reply

        # No pending HITL — should not intercept
        assert _try_set_hitl_reply("discord", "no_pending", "y") is False

    def test_pop_hitl_reply_no_pending(self):
        from EvoScientist.cli.channel import _pop_hitl_reply

        assert _pop_hitl_reply("discord", "no_pending") is None

    def test_hitl_reply_timeout(self):
        from EvoScientist.cli.channel import (
            _pop_hitl_reply,
            _register_hitl_wait,
        )

        event = _register_hitl_wait("telegram", "timeout_chat")
        # Don't set reply — simulate timeout
        replied = event.wait(timeout=0.01)
        assert replied is False
        # Pop should still return None (reply was never set)
        reply = _pop_hitl_reply("telegram", "timeout_chat")
        assert reply is None


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
