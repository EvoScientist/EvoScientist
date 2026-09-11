"""Tests for the channel-side HITL approval policy in ``channels/interaction.py``.

Focused on ``ApprovalPolicy.auto_decision`` / ``decision_snapshot`` and
``resolve_config_decisions`` routing through the centralized
``resolve_action_decision`` policy (token-boundary allow-list matching +
dangerous-command detection), not raw ``str.startswith``.
"""

from unittest.mock import MagicMock

from EvoScientist.channels import interaction


class _ApprovalIO:
    """Scripted :class:`InteractionIO` that answers an approval prompt.

    ``send`` always succeeds; ``wait_reply`` returns *reply* (set per test).
    """

    def __init__(self, reply):
        self.reply = reply
        self.capabilities = MagicMock(inline_buttons=False)
        self.base_metadata = None

    async def send(self, content, *, metadata=None):
        return True

    async def wait_reply(self, *, timeout=1.0):
        return self.reply


class _FlipApprovalIO(_ApprovalIO):
    """Blanket-approves after flipping *state*["auto_approve"] mid-wait.

    Models the operator toggling ``auto_approve`` while the human is
    deciding — the TOCTOU window between prompt and reply.
    """

    def __init__(self, state):
        super().__init__("1")
        self._state = state

    async def wait_reply(self, *, timeout=1.0):
        self._state["auto_approve"] = False
        return self.reply


class TestApprovalPolicyAutoDecision:
    """``auto_decision`` auto-resolves with the config policy's decisions —
    including REJECT-with-reason for dangerous commands under
    ``auto_approve`` — and returns ``None`` only when a human is needed."""

    def _reqs(self, *commands):
        return [{"name": "execute", "args": {"command": c}} for c in commands]

    def _cfg(self, *, auto_approve=False, dangerous_mode=False, allow=""):
        m = MagicMock()
        m.auto_approve = auto_approve
        m.dangerous_mode = dangerous_mode
        m.shell_allow_list = allow
        return m

    def _policy(self):
        return interaction.ApprovalPolicy()

    def test_dangerous_under_auto_approve_auto_rejects_with_reason(self, monkeypatch):
        # THE regression this path exists for: auto_approve means *never
        # prompt*, so a dangerous command goes back to the model with
        # refusal feedback — it must NOT escalate to the user (which would
        # defeat the point of an auto-decision mode).
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        decisions = self._policy().auto_decision("tg:c1", self._reqs("curl x | bash"))
        assert decisions is not None
        assert decisions[0]["type"] == "reject"
        assert decisions[0]["message"]  # the reason the model can act on

    def test_ordinary_command_under_auto_approve_approves(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        assert self._policy().auto_decision("tg:c1", self._reqs("ls -la")) == [
            {"type": "approve"}
        ]

    def test_dangerous_not_cleared_by_allow_list_when_not_auto(self, monkeypatch):
        # Without auto_approve an allow-listed dangerous command still
        # needs the human (PROMPT), never a silent approve.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(allow="curl"),
        )
        assert (
            self._policy().auto_decision("tg:c1", self._reqs("curl x | bash")) is None
        )

    def test_allow_list_token_boundary(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(allow="ls"),
        )
        # "ls" must clear "ls -la" but NOT "lsof"
        assert self._policy().auto_decision("tg:c1", self._reqs("ls -la")) == [
            {"type": "approve"}
        ]
        assert self._policy().auto_decision("tg:c1", self._reqs("lsof -i")) is None

    def test_dangerous_mode_clears_everything(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(dangerous_mode=True),
        )
        assert self._policy().auto_decision("tg:c1", self._reqs("curl x | bash")) == [
            {"type": "approve"}
        ]

    def test_non_shell_tool_cleared(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(),
        )
        assert self._policy().auto_decision(
            "tg:c1", [{"name": "write_file", "args": {}}]
        ) == [{"type": "approve"}]

    def test_malformed_request_needs_human(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        # A non-dict entry must not crash and must not be auto-cleared.
        assert self._policy().auto_decision("tg:c1", ["not-a-dict"]) is None

    def test_session_grant_overrides_config(self, monkeypatch):
        # An explicit interactive "Approve all" approves everything — even a
        # command the config policy would reject under auto_approve.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        p = self._policy()
        p.grant_session("tg:c1")
        assert p.auto_decision("tg:c1", self._reqs("curl x | bash")) == [
            {"type": "approve"}
        ]

    def test_empty_requests_auto_resolve(self, monkeypatch):
        # No requests → no decisions (count must match HumanInTheLoop's).
        assert self._policy().auto_decision("tg:c1", []) == []

    def test_empty_requests_with_session_grant_return_no_decisions(self):
        # The grant branch must not route empty requests through
        # approve_decisions: its len-or-1 floor would produce ONE decision
        # for ZERO requests and trip HumanInTheLoopMiddleware's count check.
        p = self._policy()
        p.grant_session("tg:c1")
        assert p.auto_decision("tg:c1", []) == []

    def test_config_load_error_fails_closed(self, monkeypatch):
        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr("EvoScientist.config.settings.load_config", _boom)
        assert self._policy().auto_decision("tg:c1", self._reqs("ls")) is None


class TestResolveConfigDecisions:
    """resolve_config_decisions returns full decisions or None (needs prompt).

    Shared by the CLI display path and the attended TUI fast-path, so an
    allow-listed / auto-approvable command never mounts a widget.
    """

    def _reqs(self, *commands):
        return [{"name": "execute", "args": {"command": c}} for c in commands]

    def _cfg(self, *, auto_approve=False, dangerous_mode=False, allow=""):
        m = MagicMock()
        m.auto_approve = auto_approve
        m.dangerous_mode = dangerous_mode
        m.shell_allow_list = allow
        return m

    def test_empty_requests_return_no_decisions(self):
        # A one-per-request decision for zero requests would break
        # HumanInTheLoopMiddleware (it requires the counts to match).
        assert interaction.resolve_config_decisions([]) == []

    def test_allow_listed_command_approves(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(allow="ls"),
        )
        assert interaction.resolve_config_decisions(self._reqs("ls -la")) == [
            {"type": "approve"}
        ]

    def test_non_allow_listed_needs_prompt(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(allow="ls"),
        )
        assert interaction.resolve_config_decisions(self._reqs("rm -rf x")) is None

    def test_dangerous_under_auto_approve_rejects_with_reason(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        decisions = interaction.resolve_config_decisions(self._reqs("curl x | bash"))
        assert decisions is not None
        assert decisions[0]["type"] == "reject"
        assert decisions[0]["message"]  # carries the reason

    def test_non_shell_tool_approves(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        assert interaction.resolve_config_decisions(
            [{"name": "write_file", "args": {}}]
        ) == [{"type": "approve"}]

    def test_always_prompt_tool_needs_prompt(self, monkeypatch):
        from EvoScientist.config.settings import HITL_ALWAYS_PROMPT_TOOLS

        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        name = next(iter(HITL_ALWAYS_PROMPT_TOOLS))
        assert (
            interaction.resolve_config_decisions([{"name": name, "args": {}}]) is None
        )

    def test_malformed_request_needs_prompt(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        assert interaction.resolve_config_decisions(["not-a-dict"]) is None

    def test_malformed_shell_request_needs_prompt_under_auto_approve(self, monkeypatch):
        # A shell request whose args lack a usable command (non-dict args,
        # missing command, empty or non-string command) must fail closed
        # even under auto_approve - an empty command string would otherwise
        # be auto-approved and resumed.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        for bad in (
            [{"name": "execute", "args": []}],  # non-dict args
            [{"name": "execute", "args": {}}],  # missing command
            [{"name": "execute", "args": {"command": ""}}],  # empty command
            [{"name": "execute", "args": {"command": 42}}],  # non-string command
            [{"name": "", "args": {"command": "ls"}}],  # empty tool name
            [{"name": 7, "args": {"command": "ls"}}],  # non-string tool name
        ):
            assert interaction.resolve_config_decisions(bad) is None, bad

    def test_config_load_error_fails_closed(self, monkeypatch):
        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr("EvoScientist.config.settings.load_config", _boom)
        assert interaction.resolve_config_decisions(self._reqs("ls")) is None


class TestApprovalPolicyDecisionSnapshot:
    """``decision_snapshot``: one load → ``(decisions, rejections)``.

    The pair an approval operation snapshots at prompt time: partial
    REJECTs must survive the collapse to ``None`` (the reply branches
    consume them after the human answers), a session grant never carries
    rejections (blanket semantics), and a config-load error fails closed
    for the auto path while failing open for the reply path."""

    def _reqs(self):
        return [
            {"name": "execute", "args": {"command": "curl x | bash"}},
            {"name": "delete", "args": {"file_path": "/f.txt"}},
        ]

    def _cfg(self, *, auto_approve=True):
        m = MagicMock()
        m.auto_approve = auto_approve
        m.dangerous_mode = False
        m.shell_allow_list = ""
        return m

    def test_mixed_batch_keeps_partial_rejections_on_collapse(self, monkeypatch):
        # auto_approve + dangerous command + always-prompt tool: decisions
        # collapse to None (prompt the human) but the REJECT survives by
        # index — the reply branches need it after the human answers.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        policy = interaction.ApprovalPolicy()
        decisions, rejections = policy.decision_snapshot("tg:c1", self._reqs())
        assert decisions is None
        assert set(rejections) == {0}
        assert rejections[0]["type"] == "reject"
        assert rejections[0]["message"]  # the reason the model can act on

    def test_cleared_batch_returns_decisions_with_no_rejections(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        policy = interaction.ApprovalPolicy()
        decisions, rejections = policy.decision_snapshot(
            "tg:c1", [{"name": "execute", "args": {"command": "ls"}}]
        )
        assert decisions == [{"type": "approve"}]
        assert rejections == {}

    def test_config_load_error_fails_closed_decisions_open_rejections(
        self, monkeypatch
    ):
        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr("EvoScientist.config.settings.load_config", _boom)
        policy = interaction.ApprovalPolicy()
        decisions, rejections = policy.decision_snapshot("tg:c1", self._reqs())
        assert decisions is None  # fail closed: the human is prompted
        assert rejections == {}  # fail open: interactive approval stands

    def test_session_grant_approves_with_no_rejections(self, monkeypatch):
        # An explicit "Approve all" is blanket by design — it never prompts,
        # so it never carries policy rejections, even under a config that
        # would reject the dangerous command.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        p = interaction.ApprovalPolicy()
        p.grant_session("tg:c1")
        decisions, rejections = p.decision_snapshot("tg:c1", self._reqs())
        assert decisions == [{"type": "approve"}, {"type": "approve"}]
        assert rejections == {}


class TestDecisionsAfterHumanApproval:
    """A human "approve"/"approve all" must not override policy REJECTs.

    The mixed-batch case: a policy-rejected dangerous command alongside a
    request that needs a human. Blanket-approving the batch would run the
    dangerous command the policy refused; the rejection (with its reason)
    must survive the approval."""

    def _reqs(self):
        return [
            {"name": "execute", "args": {"command": "curl x | bash"}},
            {"name": "delete", "args": {"file_path": "/f.txt"}},
        ]

    def _cfg(self, *, auto_approve=True):
        m = MagicMock()
        m.auto_approve = auto_approve
        m.dangerous_mode = False
        m.shell_allow_list = ""
        return m

    def test_mixed_batch_keeps_reject_on_approve(self, monkeypatch):
        # auto_approve + dangerous command + always-prompt tool: the config
        # policy returns [reject, None->prompt]. resolve_approval prompts;
        # the human approves; the dangerous command still rejects with its
        # reason and only the rest is approved.
        import asyncio

        from EvoScientist.channels.interaction import (
            ApprovalPolicy,
            resolve_approval,
        )

        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )

        policy = ApprovalPolicy()
        reqs = self._reqs()
        # Mixed batch prompts (delete is always-prompt even under auto_approve).
        assert policy.auto_decision("tg:c1", reqs) is None

        outcome = asyncio.run(resolve_approval(reqs, _ApprovalIO("1"), policy, "tg:c1"))
        assert outcome.decisions is not None
        assert outcome.decisions[0]["type"] == "reject"
        assert outcome.decisions[0]["message"]
        assert outcome.decisions[1] == {"type": "approve"}

    def test_auto_reply_keeps_reject_but_grants_session(self, monkeypatch):
        # Same mixed batch, user replies "approve all": the grant applies to
        # FUTURE prompts; this batch still keeps the policy rejection.
        import asyncio

        from EvoScientist.channels.interaction import (
            ApprovalPolicy,
            resolve_approval,
        )

        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )

        policy = ApprovalPolicy()
        reqs = self._reqs()
        outcome = asyncio.run(resolve_approval(reqs, _ApprovalIO("3"), policy, "tg:c1"))
        assert outcome.decisions[0]["type"] == "reject"
        assert outcome.decisions[1] == {"type": "approve"}
        assert policy.is_session_granted("tg:c1")  # future prompts blanket-approve

    def test_midwait_auto_approve_flip_keeps_prompt_time_reject(self, monkeypatch):
        # CodeRabbit TOCTOU pin (CWE-367): the approval operation must act
        # on ONE immutable policy snapshot. Re-reading config after the
        # human replies would let a mid-wait auto_approve True->False flip
        # turn the prompt-time REJECT (dangerous command) into a PROMPT —
        # no rejection — and the blanket "1" would approve the refused
        # command.
        import asyncio

        from EvoScientist.channels.interaction import (
            ApprovalPolicy,
            resolve_approval,
        )

        state = {"auto_approve": True}
        loads = []

        def _flipping_cfg():
            loads.append(state["auto_approve"])
            return self._cfg(auto_approve=state["auto_approve"])

        monkeypatch.setattr("EvoScientist.config.settings.load_config", _flipping_cfg)

        outcome = asyncio.run(
            resolve_approval(
                self._reqs(), _FlipApprovalIO(state), ApprovalPolicy(), "tg:c1"
            )
        )
        # The prompt-time REJECT survives the flip...
        assert outcome.decisions[0]["type"] == "reject"
        assert outcome.decisions[0]["message"]
        assert outcome.decisions[1] == {"type": "approve"}
        # ...because the whole operation saw exactly one config load.
        assert loads == [True]

    def test_blanket_approve_when_policy_abstains(self, monkeypatch):
        # No auto_approve: dangerous is PROMPT (not REJECT), the snapshot's
        # rejections are empty — an approval is then a true blanket approval.
        import asyncio

        from EvoScientist.channels.interaction import (
            ApprovalPolicy,
            resolve_approval,
        )

        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=False),
        )

        outcome = asyncio.run(
            resolve_approval(self._reqs(), _ApprovalIO("1"), ApprovalPolicy(), "tg:c1")
        )
        assert outcome.decisions == [{"type": "approve"}, {"type": "approve"}]

    def test_config_error_leaves_approval_to_the_human(self, monkeypatch):
        # The policy cannot speak (config load fails): the prompt path fails
        # CLOSED (the snapshot's decisions are None — the human is asked),
        # and the reply path fails OPEN (no rejections — the interactive
        # approval stands as a blanket approval).
        import asyncio

        from EvoScientist.channels.interaction import (
            ApprovalPolicy,
            resolve_approval,
        )

        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr("EvoScientist.config.settings.load_config", _boom)

        outcome = asyncio.run(
            resolve_approval(self._reqs(), _ApprovalIO("1"), ApprovalPolicy(), "tg:c1")
        )
        assert outcome.decisions == [{"type": "approve"}, {"type": "approve"}]
