"""Stage 1 tests for ``channels.interaction`` — the shared interaction grammar.

Two jobs:

1. **Grammar** — table-driven coverage of the reply grammar (approval
   letters, ask_user choice letters + the "Other" sub-flow, stop/cancel
   commands, and the auto-approve policy).  The ``/stop`` rows pin the
   drift fix (G3): both flows classify ``/stop`` *before* parsing a reply.
2. **Prompt-format goldens** — byte-for-byte assertions that the extracted
   formatters produce exactly the text the two drivers emitted before the
   extraction.
"""

from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.channels import interaction as I

# ═══════════════════════════════════════════════════════════════════════
# Stop / cancel grammar (G3 — the drift fix, shared by BOTH flows)
# ═══════════════════════════════════════════════════════════════════════


class TestStopCommand:
    @pytest.mark.parametrize(
        "text",
        ["/stop", "/cancel", " /stop ", "/STOP", "/Cancel", "\t/stop\n"],
    )
    def test_stop_recognized(self, text):
        assert I.is_stop_command(text) is True

    @pytest.mark.parametrize(
        "text",
        ["stop", "cancel", "/stopp", "1", "", None, "please /stop"],
    )
    def test_stop_not_recognized(self, text):
        assert I.is_stop_command(text) is False

    @pytest.mark.parametrize("text", ["cancel", "CANCEL", " Cancel ", "\tcancel"])
    def test_cancel_recognized(self, text):
        assert I.is_cancel_reply(text) is True

    @pytest.mark.parametrize("text", ["/cancel", "cancelled", "c", "", None])
    def test_cancel_not_recognized(self, text):
        assert I.is_cancel_reply(text) is False


# ═══════════════════════════════════════════════════════════════════════
# Approval reply grammar
# ═══════════════════════════════════════════════════════════════════════


class TestParseApprovalReply:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # approve
            ("1", "approve"),
            ("y", "approve"),
            ("yes", "approve"),
            ("approve", "approve"),
            ("ok", "approve"),
            (" 1 ", "approve"),
            ("  Y  ", "approve"),
            ("YES", "approve"),
            # reject
            ("2", "reject"),
            ("n", "reject"),
            ("no", "reject"),
            ("reject", "reject"),
            ("REJECT", "reject"),
            # auto / approve-all
            ("3", "auto"),
            ("a", "auto"),
            ("auto", "auto"),
            ("approve all", "auto"),
            ("APPROVE ALL", "auto"),
            # unrecognized
            ("hello world", None),
            ("", None),
            ("maybe", None),
            ("4", None),
        ],
    )
    def test_parse(self, text, expected):
        assert I.parse_approval_reply(text) == expected

    def test_button_values_normalize_to_decisions(self):
        # R3: Feishu/QQ buttons deliver their `value` ("1"/"2"/"3") through
        # the same reply path, so the shared parser must map them identically
        # to a typed reply.
        buttons = I.approval_prompt_metadata(None, with_buttons=True)["buttons"]
        values = [b["value"] for b in buttons]
        assert values == ["1", "2", "3"]
        assert [I.parse_approval_reply(v) for v in values] == [
            "approve",
            "reject",
            "auto",
        ]

    def test_approve_decisions_length(self):
        assert I.approve_decisions([{"name": "a"}, {"name": "b"}]) == [
            {"type": "approve"},
            {"type": "approve"},
        ]
        # empty request list still yields a single approve (Command shape)
        assert I.approve_decisions([]) == [{"type": "approve"}]


class TestDecisionFeedback:
    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            ("approve", I.APPROVED_FEEDBACK),
            ("auto", I.APPROVED_AUTO_FEEDBACK),
            ("reject", I.REJECTED_FEEDBACK),
            (None, I.UNRECOGNIZED_FEEDBACK),
        ],
    )
    def test_feedback(self, decision, expected):
        assert I.decision_feedback(decision) == expected


# ═══════════════════════════════════════════════════════════════════════
# ask_user choice grammar (letters + "Other")
# ═══════════════════════════════════════════════════════════════════════


class TestParseChoiceAnswer:
    CHOICES: ClassVar = [{"value": "CIFAR-10"}, {"value": "ImageNet"}]

    def test_letter_selects_choice(self):
        assert I.parse_choice_answer("A", self.CHOICES) == ("answer", "CIFAR-10")
        assert I.parse_choice_answer("b", self.CHOICES) == ("answer", "ImageNet")

    def test_other_letter(self):
        # Two choices -> "Other" is C.
        assert I.parse_choice_answer("C", self.CHOICES) == ("other", None)
        assert I.parse_choice_answer("c", self.CHOICES) == ("other", None)

    def test_out_of_range_letter_is_literal(self):
        # Z is a single alpha char but past the choice range -> literal answer.
        assert I.parse_choice_answer("Z", self.CHOICES) == ("answer", "Z")

    def test_multichar_reply_is_literal(self):
        assert I.parse_choice_answer("CIFAR-10", self.CHOICES) == (
            "answer",
            "CIFAR-10",
        )

    def test_no_choices_other_is_a(self):
        assert I.parse_choice_answer("A", []) == ("other", None)


# ═══════════════════════════════════════════════════════════════════════
# ApprovalPolicy (config rule + session registry + session key)
# ═══════════════════════════════════════════════════════════════════════


class TestApprovalPolicy:
    def test_session_key(self):
        assert I.ApprovalPolicy.session_key("telegram", "c1") == "telegram:c1"

    def test_grant_and_is_granted(self):
        p = I.ApprovalPolicy()
        assert p.is_session_granted("tg:c1") is False
        p.grant_session("tg:c1")
        assert p.is_session_granted("tg:c1") is True
        p.clear_sessions()
        assert p.is_session_granted("tg:c1") is False

    def test_auto_decision_session_granted(self):
        p = I.ApprovalPolicy()
        p.grant_session("tg:c1")
        reqs = [{"name": "execute", "args": {"command": "rm -rf /"}}]
        # Session grant short-circuits config entirely.
        assert p.auto_decision("tg:c1", reqs) == [{"type": "approve"}]

    def test_auto_decision_config_true(self):
        p = I.ApprovalPolicy()
        cfg = MagicMock()
        cfg.auto_approve = True
        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            reqs = [{"name": "execute", "args": {"command": "rm -rf /"}}]
            assert p.auto_decision("tg:c1", reqs) == [{"type": "approve"}]

    def test_auto_decision_needs_prompt(self):
        p = I.ApprovalPolicy()
        cfg = MagicMock()
        cfg.auto_approve = False
        cfg.shell_allow_list = ""
        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            reqs = [{"name": "execute", "args": {"command": "rm -rf /"}}]
            assert p.auto_decision("tg:c1", reqs) is None


class TestConfigAutoApprove:
    def test_empty(self):
        assert I.config_auto_approve([]) is True

    def test_non_execute(self):
        assert I.config_auto_approve([{"name": "write_file", "args": {}}]) is True

    def test_execute_no_allowlist(self):
        cfg = MagicMock()
        cfg.auto_approve = False
        cfg.shell_allow_list = ""
        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            assert (
                I.config_auto_approve(
                    [{"name": "execute", "args": {"command": "rm -rf /"}}]
                )
                is False
            )

    def test_execute_allowlist_match(self):
        cfg = MagicMock()
        cfg.auto_approve = False
        cfg.shell_allow_list = "ls,python"
        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            assert (
                I.config_auto_approve(
                    [{"name": "execute", "args": {"command": "ls -la"}}]
                )
                is True
            )

    def test_run_in_background_not_allowlisted(self):
        cfg = MagicMock()
        cfg.auto_approve = False
        cfg.shell_allow_list = "ls,cat"
        with patch("EvoScientist.config.settings.load_config", return_value=cfg):
            assert (
                I.config_auto_approve(
                    [{"name": "run_in_background", "args": {"command": "rm -rf /"}}]
                )
                is False
            )

    def test_fail_closed_on_config_error(self):
        with patch(
            "EvoScientist.config.settings.load_config", side_effect=RuntimeError("boom")
        ):
            assert (
                I.config_auto_approve([{"name": "execute", "args": {"command": "ls"}}])
                is False
            )


# ═══════════════════════════════════════════════════════════════════════
# Prompt-format goldens — byte-for-byte vs the pre-extraction drivers.
# ═══════════════════════════════════════════════════════════════════════
#
# The literals below are exactly what ``consumer.py`` / ``cli/channel.py``
# emitted before extraction (whitespace, emoji, and phrasing preserved).


class TestApprovalPromptGolden:
    def test_single_no_buttons(self):
        got = I.format_approval_prompt(
            [{"name": "execute", "args": {"command": "ls -la"}}]
        )
        assert got == (
            "⚠️ Approval Required\n"
            "\n"
            "  1. execute: ls -la\n"
            "\n"
            "Reply: 1=Approve, 2=Reject, 3=Approve all\n"
            "(Auto-reject in 2 min if no reply)"
        )

    def test_multiple_no_buttons(self):
        got = I.format_approval_prompt(
            [
                {"name": "execute", "args": {"command": "ls"}},
                {"name": "write_file", "args": {"path": "/out.txt"}},
            ]
        )
        assert got == (
            "⚠️ Approval Required\n"
            "\n"
            "  1. execute: ls\n"
            "  2. write_file: /out.txt\n"
            "\n"
            "Reply: 1=Approve, 2=Reject, 3=Approve all\n"
            "(Auto-reject in 2 min if no reply)"
        )

    def test_with_buttons_drops_text_instruction(self):
        got = I.format_approval_prompt(
            [{"name": "execute", "args": {"command": "ls -la"}}],
            with_buttons=True,
        )
        assert got == "⚠️ Approval Required\n\n  1. execute: ls -la"

    def test_no_command_falls_back_to_name(self):
        got = I.format_approval_prompt([{"name": "ask_user", "args": {}}])
        assert got == (
            "⚠️ Approval Required\n"
            "\n"
            "  1. ask_user\n"
            "\n"
            "Reply: 1=Approve, 2=Reject, 3=Approve all\n"
            "(Auto-reject in 2 min if no reply)"
        )

    def test_metadata_no_buttons(self):
        assert I.approval_prompt_metadata({"k": "v"}, with_buttons=False) == {"k": "v"}

    def test_metadata_with_buttons(self):
        md = I.approval_prompt_metadata({"k": "v"}, with_buttons=True)
        assert md["k"] == "v"
        assert md["buttons"] == [
            {"text": "Approve", "value": "1", "type": "primary"},
            {"text": "Reject", "value": "2", "type": "danger"},
            {"text": "Approve all", "value": "3"},
        ]


class TestQuestionPromptGolden:
    def test_single_text_required(self):
        got = I.format_question_prompt(
            {"question": "What dataset?", "type": "text"}, 0, 1
        )
        assert got == (
            "❓ Quick check-in from EvoScientist\n"
            "\n"
            "1. What dataset?\n"
            "\n"
            "Reply with your answer, or 'cancel'."
        )

    def test_single_text_optional(self):
        got = I.format_question_prompt(
            {"question": "Notes?", "type": "text", "required": False}, 0, 1
        )
        assert got == (
            "❓ Quick check-in from EvoScientist\n"
            "\n"
            "1. Notes? (optional)\n"
            "\n"
            "Reply with your answer, or 'cancel'. Leave empty to skip."
        )

    def test_multi_choice_optional_numbered(self):
        got = I.format_question_prompt(
            {
                "question": "Which?",
                "type": "multiple_choice",
                "choices": [{"value": "A"}, {"value": "B"}],
                "required": False,
            },
            1,
            3,
        )
        assert got == (
            "❓ Question 2/3\n"
            "\n"
            "2. Which? (optional)\n"
            "   A. A\n"
            "   B. B\n"
            "   C. Other\n"
            "\n"
            "Reply with a letter (A/B/C), or 'cancel'."
        )

    def test_multi_choice_required(self):
        got = I.format_question_prompt(
            {
                "question": "Pick one",
                "type": "multiple_choice",
                "choices": [{"value": "CIFAR-10"}, {"value": "ImageNet"}],
            },
            0,
            1,
        )
        assert got == (
            "❓ Quick check-in from EvoScientist\n"
            "\n"
            "1. Pick one\n"
            "   A. CIFAR-10\n"
            "   B. ImageNet\n"
            "   C. Other\n"
            "\n"
            "Reply with a letter (A/B/C), or 'cancel'."
        )
