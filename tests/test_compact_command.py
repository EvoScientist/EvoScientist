"""Tests for the /compact command (compact_conversation helper)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import run_async as _run


class TestCompactGuards:
    """Guard conditions that return early without touching the middleware."""

    def test_no_agent(self):
        from EvoScientist.cli.commands import compact_conversation

        result = _run(compact_conversation(agent=None, thread_id="abc"))
        assert result.status == "noop"
        assert "Nothing to compact" in result.message

    def test_no_thread_id(self):
        from EvoScientist.cli.commands import compact_conversation

        result = _run(compact_conversation(agent=MagicMock(), thread_id=None))
        assert result.status == "noop"
        assert "Nothing to compact" in result.message

    def test_empty_messages(self):
        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        snapshot = SimpleNamespace(values={"messages": []})
        agent.aget_state = AsyncMock(return_value=snapshot)

        result = _run(compact_conversation(agent=agent, thread_id="tid-1"))
        assert result.status == "noop"
        assert "no messages" in result.message

    def test_state_read_failure(self):
        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        agent.aget_state = AsyncMock(side_effect=RuntimeError("DB gone"))

        result = _run(compact_conversation(agent=agent, thread_id="tid-1"))
        assert result.status == "error"
        assert "Failed to read state" in result.message


class TestCompactCutoffZero:
    """When cutoff == 0, conversation is within retention budget."""

    def test_nothing_to_compact_short_conversation(self):
        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        msgs = [MagicMock() for _ in range(3)]
        snapshot = SimpleNamespace(values={"messages": msgs})
        agent.aget_state = AsyncMock(return_value=snapshot)

        mock_middleware_inst = MagicMock()
        mock_middleware_inst._apply_event_to_messages.return_value = msgs
        mock_middleware_inst._determine_cutoff_index.return_value = 0

        mock_middleware_cls = MagicMock(return_value=mock_middleware_inst)
        model = SimpleNamespace(profile={"max_input_tokens": 1000})

        with (
            patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=model),
            patch(
                "EvoScientist.EvoScientist._get_default_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deepagents.middleware.summarization.SummarizationMiddleware",
                mock_middleware_cls,
            ),
            patch(
                "deepagents.middleware.summarization.compute_summarization_defaults",
                return_value={"keep": ("messages", 6)},
            ),
            patch(
                "langchain_core.messages.utils.count_tokens_approximately",
                return_value=500,
            ),
        ):
            result = _run(compact_conversation(agent=agent, thread_id="tid-1"))

        assert result.status == "noop"
        assert "within the retention budget" in result.message
        assert result.tokens_before == 500


class TestCompactNegligibleSavings:
    """When cutoff > 0 but savings are too small to be worth it."""

    def test_skip_when_few_messages_and_low_tokens(self):
        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        msgs = [MagicMock() for _ in range(15)]
        snapshot = SimpleNamespace(
            values={"messages": msgs, "_summarization_event": None}
        )
        agent.aget_state = AsyncMock(return_value=snapshot)

        mock_middleware_inst = MagicMock()
        mock_middleware_inst._apply_event_to_messages.return_value = msgs
        mock_middleware_inst._determine_cutoff_index.return_value = 1
        # 1 message to summarize (200 tokens), 14 to keep (22000 tokens)
        mock_middleware_inst._partition_messages.return_value = (msgs[:1], msgs[1:])

        mock_middleware_cls = MagicMock(return_value=mock_middleware_inst)
        model = SimpleNamespace(profile={"max_input_tokens": 50_000})

        # effective=22200 (44%), to_summarize=200, to_keep=22000 → 200/22200 < 2%
        token_values = iter([22_200, 200, 22_000])

        with (
            patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=model),
            patch(
                "EvoScientist.EvoScientist._get_default_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deepagents.middleware.summarization.SummarizationMiddleware",
                mock_middleware_cls,
            ),
            patch(
                "deepagents.middleware.summarization.compute_summarization_defaults",
                return_value={"keep": ("messages", 6)},
            ),
            patch(
                "langchain_core.messages.utils.count_tokens_approximately",
                side_effect=lambda x: next(token_values),
            ),
        ):
            result = _run(compact_conversation(agent=agent, thread_id="tid-1"))

        assert result.status == "noop"
        assert "not worth" in result.message
        # No LLM call should have been made
        mock_middleware_inst._acreate_summary.assert_not_called()

    def test_still_compacts_when_few_messages_but_high_tokens(self):
        """2 messages but they account for >2% of tokens — should compact."""
        from langchain_core.messages import HumanMessage

        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        msgs = [MagicMock() for _ in range(10)]
        snapshot = SimpleNamespace(
            values={"messages": msgs, "_summarization_event": None}
        )
        agent.aget_state = AsyncMock(return_value=snapshot)
        agent.aupdate_state = AsyncMock()

        summary_msg = HumanMessage(content="Summary")

        mock_middleware_inst = MagicMock()
        mock_middleware_inst._apply_event_to_messages.return_value = msgs
        mock_middleware_inst._determine_cutoff_index.return_value = 2
        mock_middleware_inst._partition_messages.return_value = (msgs[:2], msgs[2:])
        mock_middleware_inst._acreate_summary = AsyncMock(return_value="Summary")
        mock_middleware_inst._aoffload_to_backend = AsyncMock(return_value=None)
        mock_middleware_inst._build_new_messages_with_path.return_value = [summary_msg]
        mock_middleware_inst._compute_state_cutoff.return_value = 2

        mock_middleware_cls = MagicMock(return_value=mock_middleware_inst)
        model = SimpleNamespace(profile={"max_input_tokens": 40_000})

        # effective=20000 (50%), to_summarize=5000, to_keep=15000 → 25% > 2%
        token_values = iter([20_000, 5_000, 15_000, 500])

        with (
            patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=model),
            patch(
                "EvoScientist.EvoScientist._get_default_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deepagents.middleware.summarization.SummarizationMiddleware",
                mock_middleware_cls,
            ),
            patch(
                "deepagents.middleware.summarization.compute_summarization_defaults",
                return_value={"keep": ("messages", 6)},
            ),
            patch(
                "langchain_core.messages.utils.count_tokens_approximately",
                side_effect=lambda x: next(token_values),
            ),
        ):
            result = _run(compact_conversation(agent=agent, thread_id="tid-1"))

        assert result.status == "ok"
        agent.aupdate_state.assert_awaited_once()


class TestCompactSuccess:
    """Normal compaction flow."""

    def test_manual_threshold_blocks_low_context_compaction(self):
        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        msgs = [MagicMock() for _ in range(20)]
        snapshot = SimpleNamespace(
            values={"messages": msgs, "_summarization_event": None}
        )
        agent.aget_state = AsyncMock(return_value=snapshot)

        mock_middleware_inst = MagicMock()
        mock_middleware_inst._apply_event_to_messages.return_value = msgs
        mock_middleware_cls = MagicMock(return_value=mock_middleware_inst)
        model = SimpleNamespace(profile={"max_input_tokens": 100_000})

        with (
            patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=model),
            patch(
                "EvoScientist.EvoScientist._get_default_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deepagents.middleware.summarization.SummarizationMiddleware",
                mock_middleware_cls,
            ),
            patch(
                "deepagents.middleware.summarization.compute_summarization_defaults",
                return_value={"keep": ("messages", 6)},
            ),
            patch(
                "langchain_core.messages.utils.count_tokens_approximately",
                return_value=30_000,
            ),
        ):
            result = _run(compact_conversation(agent=agent, thread_id="tid-1"))

        assert result.status == "noop"
        assert "40%" in result.message
        assert result.context_percent == 30
        mock_middleware_inst._determine_cutoff_index.assert_not_called()
        mock_middleware_inst._acreate_summary.assert_not_called()

    def test_successful_compaction(self):
        from langchain_core.messages import HumanMessage

        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        msgs = [MagicMock() for _ in range(20)]
        snapshot = SimpleNamespace(
            values={"messages": msgs, "_summarization_event": None}
        )
        agent.aget_state = AsyncMock(return_value=snapshot)
        agent.aupdate_state = AsyncMock()

        summary_msg = HumanMessage(content="Summary of conversation")
        to_summarize = msgs[:15]
        to_keep = msgs[15:]

        mock_middleware_inst = MagicMock()
        mock_middleware_inst._apply_event_to_messages.return_value = msgs
        mock_middleware_inst._determine_cutoff_index.return_value = 15
        mock_middleware_inst._partition_messages.return_value = (to_summarize, to_keep)
        mock_middleware_inst._acreate_summary = AsyncMock(return_value="Summary text")
        mock_middleware_inst._aoffload_to_backend = AsyncMock(
            return_value="/conversation_history/tid.md"
        )
        mock_middleware_inst._build_new_messages_with_path.return_value = [summary_msg]
        mock_middleware_inst._compute_state_cutoff.return_value = 15

        mock_middleware_cls = MagicMock(return_value=mock_middleware_inst)
        model = SimpleNamespace(profile={"max_input_tokens": 10_000})

        # effective=6000 (60%), then summarize/keep/summary accounting
        token_values = iter([6000, 5000, 1000, 200])

        with (
            patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=model),
            patch(
                "EvoScientist.EvoScientist._get_default_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deepagents.middleware.summarization.SummarizationMiddleware",
                mock_middleware_cls,
            ),
            patch(
                "deepagents.middleware.summarization.compute_summarization_defaults",
                return_value={"keep": ("messages", 6)},
            ),
            patch(
                "langchain_core.messages.utils.count_tokens_approximately",
                side_effect=lambda x: next(token_values),
            ),
        ):
            result = _run(compact_conversation(agent=agent, thread_id="tid-1"))

        assert result.status == "ok"
        assert result.messages_compacted == 15
        assert result.messages_kept == 5
        assert result.tokens_before == 6000
        assert result.tokens_after == 1200
        assert result.pct_decrease == 80
        # context_percent reflects usage AFTER compact (12%), not before (60%)
        assert result.context_percent == 12
        assert result.summary_text == "Summary text"
        agent.aupdate_state.assert_awaited_once()

        # Verify the event structure passed to aupdate_state
        call_args = agent.aupdate_state.call_args
        event_data = call_args[0][1]
        assert "_summarization_event" in event_data
        assert event_data["_summarization_event"]["cutoff_index"] == 15

    def test_offload_failure_non_fatal(self):
        """Offload failure should not prevent compaction."""
        from langchain_core.messages import HumanMessage

        from EvoScientist.cli.commands import compact_conversation

        agent = MagicMock()
        msgs = [MagicMock() for _ in range(10)]
        snapshot = SimpleNamespace(
            values={"messages": msgs, "_summarization_event": None}
        )
        agent.aget_state = AsyncMock(return_value=snapshot)
        agent.aupdate_state = AsyncMock()

        summary_msg = HumanMessage(content="Summary")

        mock_middleware_inst = MagicMock()
        mock_middleware_inst._apply_event_to_messages.return_value = msgs
        mock_middleware_inst._determine_cutoff_index.return_value = 7
        mock_middleware_inst._partition_messages.return_value = (msgs[:7], msgs[7:])
        mock_middleware_inst._acreate_summary = AsyncMock(return_value="Summary")
        mock_middleware_inst._aoffload_to_backend = AsyncMock(
            side_effect=RuntimeError("write failed")
        )
        mock_middleware_inst._build_new_messages_with_path.return_value = [summary_msg]
        mock_middleware_inst._compute_state_cutoff.return_value = 7

        mock_middleware_cls = MagicMock(return_value=mock_middleware_inst)
        model = SimpleNamespace(profile={"max_input_tokens": 2_000})

        with (
            patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=model),
            patch(
                "EvoScientist.EvoScientist._get_default_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deepagents.middleware.summarization.SummarizationMiddleware",
                mock_middleware_cls,
            ),
            patch(
                "deepagents.middleware.summarization.compute_summarization_defaults",
                return_value={"keep": ("messages", 6)},
            ),
            patch(
                "langchain_core.messages.utils.count_tokens_approximately",
                return_value=1000,
            ),
        ):
            result = _run(compact_conversation(agent=agent, thread_id="tid-1"))

        assert result.status == "ok"
        agent.aupdate_state.assert_awaited_once()

        # file_path should be None in the event
        event_data = agent.aupdate_state.call_args[0][1]
        assert event_data["_summarization_event"]["file_path"] is None


class TestRenderCompactResult:
    """Test the Rich rendering of CompactResult."""

    def test_render_noop(self):
        from EvoScientist.cli.commands import CompactResult, render_compact_result

        result = CompactResult("noop", "Nothing to compact", tokens_before=500)
        text = render_compact_result(result)
        plain = text.plain
        assert "Nothing to compact" in plain
        assert "500" in plain

    def test_render_noop_no_tokens(self):
        from EvoScientist.cli.commands import CompactResult, render_compact_result

        result = CompactResult(
            "noop", "Nothing to compact — no messages in conversation."
        )
        text = render_compact_result(result)
        assert "Nothing to compact" in text.plain

    def test_render_error(self):
        from EvoScientist.cli.commands import CompactResult, render_compact_result

        result = CompactResult("error", "Failed to read state: DB gone")
        text = render_compact_result(result)
        assert "Failed to read state" in text.plain


class TestCompactCommandUI:
    """TUI-specific compact progress indicator behavior."""

    def test_command_uses_tui_indicator_when_available(self):
        from EvoScientist.cli.commands import CompactResult
        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.session import CompactCommand

        class _UI:
            supports_interactive = True

            def __init__(self) -> None:
                self.system_messages: list[str] = []
                self.renderables: list[object] = []
                self.started = 0
                self.stopped = 0
                self.updated_tokens: list[int] = []

            def append_system(self, text: str, style: str = "dim") -> None:
                self.system_messages.append(text)

            def mount_renderable(self, renderable):
                self.renderables.append(renderable)

            async def start_compacting_indicator(self) -> None:
                self.started += 1

            async def stop_compacting_indicator(self) -> None:
                self.stopped += 1

            def update_status_after_compact(self, tokens_after: int) -> None:
                self.updated_tokens.append(tokens_after)

        ui = _UI()
        # input_tokens_hint must be set for update_status_after_compact to fire
        # (without it, tokens_after is message-level and the unit would be wrong)
        ctx = CommandContext(
            agent=MagicMock(), thread_id="tid-1", ui=ui, input_tokens_hint=5000
        )
        result = CompactResult(
            "ok",
            "Compacted",
            tokens_after=1200,
            summary_text="summary body",
        )

        with (
            patch(
                "EvoScientist.cli.commands.compact_conversation",
                AsyncMock(return_value=result),
            ),
            patch(
                "EvoScientist.cli.commands.render_compact_result",
                return_value="result-panel",
            ),
            patch(
                "EvoScientist.cli.commands.build_compact_summary_renderable",
                return_value="summary-panel",
            ),
        ):
            _run(CompactCommand().execute(ctx, []))

        assert ui.started == 1
        assert ui.stopped == 1
        assert ui.system_messages == []
        assert ui.renderables == ["result-panel", "summary-panel"]
        assert ui.updated_tokens == [1200]

    def test_render_ok(self):
        from EvoScientist.cli.commands import CompactResult, render_compact_result

        result = CompactResult(
            "ok",
            "Compacted",
            messages_compacted=15,
            messages_kept=5,
            tokens_before=6000,
            tokens_after=1200,
            tokens_summarized=5000,
            tokens_summary=200,
            pct_decrease=80,
            context_window=10_000,
            context_percent=60,
        )
        text = render_compact_result(result)
        plain = text.plain
        assert "15" in plain
        assert "6,000" in plain
        assert "1,200" in plain
        assert "80%" in plain
        assert "5 messages unchanged" in plain
        assert "60% used" in plain

    def test_build_compact_summary_renderable(self):
        from EvoScientist.cli.commands import (
            CompactResult,
            build_compact_summary_renderable,
        )

        result = CompactResult("ok", "Compacted", summary_text="Summary body")
        renderable = build_compact_summary_renderable(result)

        assert renderable is not None
        assert renderable.summary_text == "Summary body"

    def test_str_fallback(self):
        from EvoScientist.cli.commands import CompactResult

        result = CompactResult("ok", "hello world")
        assert str(result) == "hello world"
