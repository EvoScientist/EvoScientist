"""Tests the shared interaction grammar for ``channels.interaction``."""

from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.channels import interaction as I

# ═══════════════════════════════════════════════════════════════════════
# Stop / cancel grammar
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
        # Feishu/QQ buttons deliver their `value` ("1"/"2"/"3") through
        # the same reply path, so the shared parser must map them
        # identically to a typed reply.
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
# Prompt-format checks.
# ═══════════════════════════════════════════════════════════════════════


class TestApprovalPromptFormat:
    def test_lists_each_action_with_reply_options(self):
        got = I.format_approval_prompt(
            [
                {"name": "execute", "args": {"command": "ls"}},
                {"name": "write_file", "args": {"path": "/out.txt"}},
            ]
        )
        assert "execute: ls" in got
        assert "write_file: /out.txt" in got
        # The offered options must match what parse_approval_reply accepts.
        for option in ("1=Approve", "2=Reject", "3=Approve all"):
            assert option in got

    def test_with_buttons_drops_text_instruction(self):
        got = I.format_approval_prompt(
            [{"name": "execute", "args": {"command": "ls -la"}}],
            with_buttons=True,
        )
        assert "execute: ls -la" in got
        assert "1=Approve" not in got  # buttons replace the typed-reply hint

    def test_no_command_falls_back_to_name(self):
        got = I.format_approval_prompt([{"name": "ask_user", "args": {}}])
        assert "ask_user" in got

    def test_metadata_no_buttons(self):
        assert I.approval_prompt_metadata({"k": "v"}, with_buttons=False) == {"k": "v"}

    def test_metadata_with_buttons(self):
        md = I.approval_prompt_metadata({"k": "v"}, with_buttons=True)
        assert md["k"] == "v"
        # Button values must be replies parse_approval_reply understands.
        assert [b["value"] for b in md["buttons"]] == ["1", "2", "3"]


class TestQuestionPromptFormat:
    def test_single_question_offers_cancel(self):
        got = I.format_question_prompt(
            {"question": "What dataset?", "type": "text"}, 0, 1
        )
        assert "What dataset?" in got
        assert "cancel" in got

    def test_optional_question_is_marked_and_skippable(self):
        got = I.format_question_prompt(
            {"question": "Notes?", "type": "text", "required": False}, 0, 1
        )
        assert "(optional)" in got
        assert "skip" in got.lower()

    def test_multi_question_header_shows_position(self):
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
        assert "2/3" in got

    def test_choices_get_letters_plus_other(self):
        got = I.format_question_prompt(
            {
                "question": "Pick one",
                "type": "multiple_choice",
                "choices": [{"value": "CIFAR-10"}, {"value": "ImageNet"}],
            },
            0,
            1,
        )
        # Displayed letters must match what the choice parser accepts, with
        # the "Other" free-form option appended after the real choices.
        assert "A. CIFAR-10" in got
        assert "B. ImageNet" in got
        assert "C. Other" in got


class TestChoiceNormalization:
    """Choices arrive from model tool args — plain strings must not crash."""

    def test_prompt_renders_plain_string_choices(self):
        got = I.format_question_prompt(
            {
                "question": "Pick one",
                "type": "multiple_choice",
                "choices": ["CIFAR-10", "ImageNet"],
            },
            0,
            1,
        )
        assert "A. CIFAR-10" in got
        assert "B. ImageNet" in got
        assert "C. Other" in got

    def test_parse_returns_plain_string_choice(self):
        kind, value = I.parse_choice_answer("b", ["CIFAR-10", "ImageNet"])
        assert (kind, value) == ("answer", "ImageNet")

    def test_mixed_dict_and_string_choices(self):
        choices = [{"value": "CIFAR-10"}, "ImageNet"]
        got = I.format_question_prompt(
            {"question": "Pick", "type": "multiple_choice", "choices": choices},
            0,
            1,
        )
        assert "A. CIFAR-10" in got
        assert "B. ImageNet" in got
        kind, value = I.parse_choice_answer("a", choices)
        assert (kind, value) == ("answer", "CIFAR-10")
