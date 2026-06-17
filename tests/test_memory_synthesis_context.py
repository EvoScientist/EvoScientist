"""Tests for synthesis context building and claim/release deduplication.

This PR adds:
- _synthesis_context_digest (hash-based)
- _claim_synthesis_context / _release_synthesis_context (deduplication)
- SynthesisReviewDecision model validation
- build_synthesis_context excluding covered observations
"""

from __future__ import annotations

from EvoScientist.memory import synthesis as memory_synthesis
from EvoScientist.memory.synthesis import (
    SynthesisAction,
    SynthesisArchiveDecision,
    SynthesisCreateDecision,
    SynthesisReviewDecision,
    SynthesisSkipDecision,
    SynthesisUpdateDecision,
    _claim_synthesis_context,
    _release_synthesis_context,
    _synthesis_context_digest,
)
from EvoScientist.memory.types import MemoryScope, MemoryType


# ---------------------------------------------------------------------------
# _synthesis_context_digest
# ---------------------------------------------------------------------------


class TestSynthesisContextDigest:
    def _make_context(self, obs_ids=("O-abc",)):
        return {
            "project_id": "P-test",
            "uncovered_observations": [
                {
                    "id": obs_id,
                    "path": f"/memories/observations/global/{obs_id}.md",
                    "memory_type": "semantic",
                    "scope": "global",
                    "summary": "Test summary",
                    "snippet": "Test snippet",
                }
                for obs_id in obs_ids
            ],
            "memory_inventory": {"active_knowledge_count": 0, "seed_observation_count": len(obs_ids)},
        }

    def test_returns_16_char_hex_string(self):
        context = self._make_context()
        digest = _synthesis_context_digest(context)
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic(self):
        context = self._make_context()
        assert _synthesis_context_digest(context) == _synthesis_context_digest(context)

    def test_different_contexts_differ(self):
        c1 = self._make_context(("O-abc",))
        c2 = self._make_context(("O-xyz",))
        assert _synthesis_context_digest(c1) != _synthesis_context_digest(c2)


# ---------------------------------------------------------------------------
# _claim_synthesis_context / _release_synthesis_context
# ---------------------------------------------------------------------------


class TestClaimReleaseContext:
    def setup_method(self):
        """Clear the global set before each test."""
        from EvoScientist.memory.worker_activity import reset_memory_worker_status_for_tests
        with memory_synthesis._active_synthesis_lock:
            memory_synthesis._active_synthesis_contexts.clear()
        reset_memory_worker_status_for_tests()

    def teardown_method(self):
        """Clear the global set after each test."""
        from EvoScientist.memory.worker_activity import reset_memory_worker_status_for_tests
        with memory_synthesis._active_synthesis_lock:
            memory_synthesis._active_synthesis_contexts.clear()
        reset_memory_worker_status_for_tests()

    def test_first_claim_succeeds(self):
        assert _claim_synthesis_context(project_id="P-test", context_digest="d1") is True

    def test_second_claim_fails(self):
        _claim_synthesis_context(project_id="P-test", context_digest="d1")
        assert _claim_synthesis_context(project_id="P-test", context_digest="d1") is False

    def test_different_digests_can_both_claim(self):
        assert _claim_synthesis_context(project_id="P-test", context_digest="d1") is True
        assert _claim_synthesis_context(project_id="P-test", context_digest="d2") is True

    def test_different_project_ids_independent(self):
        assert _claim_synthesis_context(project_id="P-a", context_digest="d1") is True
        assert _claim_synthesis_context(project_id="P-b", context_digest="d1") is True

    def test_release_allows_reclaim(self, tmp_path):
        _claim_synthesis_context(project_id="P-test", context_digest="d1")
        # Use the module's release helper; it also calls mark_synthesis_finished
        from EvoScientist.memory.worker_activity import reset_memory_worker_status_for_tests
        reset_memory_worker_status_for_tests()
        _release_synthesis_context(project_id="P-test", context_digest="d1")
        # After release, the same context can be claimed again
        assert _claim_synthesis_context(project_id="P-test", context_digest="d1") is True

    def test_release_nonexistent_is_noop(self):
        # Should not raise
        _release_synthesis_context(project_id="P-test", context_digest="no-such")


# ---------------------------------------------------------------------------
# SynthesisReviewDecision model
# ---------------------------------------------------------------------------


class TestSynthesisReviewDecision:
    def test_empty_decisions_is_valid(self):
        review = SynthesisReviewDecision(decisions=[])
        assert review.decisions == []

    def test_skip_decision(self):
        decision = SynthesisSkipDecision(
            action=SynthesisAction.SKIP,
            rationale="No durable findings.",
        )
        review = SynthesisReviewDecision(decisions=[decision])
        assert len(review.decisions) == 1
        assert isinstance(review.decisions[0], SynthesisSkipDecision)

    def test_create_decision(self):
        decision = SynthesisCreateDecision(
            action=SynthesisAction.CREATE,
            rationale="Good finding.",
            summary="Test knowledge",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.GLOBAL,
            knowledge="This is a test knowledge record.",
            supporting_observation_ids=["O-abc1234567890123"],
        )
        review = SynthesisReviewDecision(decisions=[decision])
        assert isinstance(review.decisions[0], SynthesisCreateDecision)
        assert review.decisions[0].memory_type == MemoryType.SEMANTIC
        assert review.decisions[0].scope == MemoryScope.GLOBAL

    def test_update_decision(self):
        decision = SynthesisUpdateDecision(
            action=SynthesisAction.UPDATE,
            rationale="Broader evidence.",
            target_knowledge_id="K-cafebabe12345678",
            summary="Updated knowledge",
            memory_type=MemoryType.PROCEDURAL,
            knowledge="Updated procedure content.",
            supporting_observation_ids=["O-abc1234567890123", "O-def1234567890123"],
        )
        assert decision.target_knowledge_id == "K-cafebabe12345678"
        assert len(decision.supporting_observation_ids) == 2

    def test_archive_decision(self):
        decision = SynthesisArchiveDecision(
            action=SynthesisAction.ARCHIVE,
            rationale="Superseded.",
            target_knowledge_id="K-abc1234567890123",
            archive_reason="This was replaced.",
        )
        assert decision.target_knowledge_id == "K-abc1234567890123"
        assert decision.archive_reason == "This was replaced."

    def test_archive_decision_no_archive_reason(self):
        decision = SynthesisArchiveDecision(
            action=SynthesisAction.ARCHIVE,
            rationale="Superseded.",
            target_knowledge_id="K-abc1234567890123",
        )
        assert decision.archive_reason is None

    def test_no_op_reason(self):
        review = SynthesisReviewDecision(
            decisions=[],
            no_op_reason="No uncovered observations needed synthesis.",
        )
        assert review.no_op_reason is not None

    def test_multiple_decisions(self):
        skip = SynthesisSkipDecision(
            action=SynthesisAction.SKIP, rationale="Skip this."
        )
        create = SynthesisCreateDecision(
            action=SynthesisAction.CREATE,
            rationale="New finding.",
            summary="New knowledge",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.PROJECT,
            knowledge="Episodic event captured.",
            supporting_observation_ids=["O-abc1234567890123"],
        )
        review = SynthesisReviewDecision(decisions=[skip, create])
        assert len(review.decisions) == 2


# ---------------------------------------------------------------------------
# build_synthesis_context
# ---------------------------------------------------------------------------


class TestBuildSynthesisContext:
    def _record_obs(
        self,
        memory_dir,
        *,
        project_id="P-test",
        summary="Test finding",
        observation="The finding.",
        why_it_matters="Future reference.",
        scope=None,
    ):
        from EvoScientist.memory.observations import record_observation_file
        from EvoScientist.memory.types import MemoryScope, MemorySourceType, MemoryType

        return record_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
            memory_type=MemoryType.SEMANTIC,
            summary=summary,
            observation=observation,
            why_it_matters=why_it_matters,
            scope=scope or MemoryScope.GLOBAL,
            source_type=MemorySourceType.SUBAGENT,
            source_session_id="t-1",
            source_agent="test",
        )

    def test_returns_none_for_empty_seeds(self, tmp_path):
        context = memory_synthesis.build_synthesis_context(
            memory_dir=tmp_path / "memories",
            project_id="P-test",
            seed_observation_ids=(),
        )
        assert context is None

    def test_returns_none_when_all_seeds_missing(self, tmp_path):
        context = memory_synthesis.build_synthesis_context(
            memory_dir=tmp_path / "memories",
            project_id="P-test",
            seed_observation_ids=("O-nonexistent1234567",),
        )
        assert context is None

    def test_returns_context_for_valid_seed(self, tmp_path):
        memories = tmp_path / "memories"
        obs = self._record_obs(memories)
        context = memory_synthesis.build_synthesis_context(
            memory_dir=memories,
            project_id="P-test",
            seed_observation_ids=(obs["observation_id"],),
        )
        assert context is not None
        assert context["project_id"] == "P-test"
        assert len(context["uncovered_observations"]) == 1
        assert context["uncovered_observations"][0]["id"] == obs["observation_id"]

    def test_dedupes_seed_ids(self, tmp_path):
        memories = tmp_path / "memories"
        obs = self._record_obs(memories)
        context = memory_synthesis.build_synthesis_context(
            memory_dir=memories,
            project_id="P-test",
            seed_observation_ids=(
                obs["observation_id"],
                obs["observation_id"],
            ),
        )
        assert context is not None
        assert len(context["uncovered_observations"]) == 1

    def test_excludes_covered_observations(self, tmp_path):
        """Observations already cited by active knowledge are not included."""
        from EvoScientist.memory.knowledge import record_knowledge_file

        memories = tmp_path / "memories"
        covered_obs = self._record_obs(memories, summary="Covered observation")
        uncovered_obs = self._record_obs(
            memories,
            summary="Uncovered observation",
            observation="Separate finding.",
            why_it_matters="New context.",
        )

        # Create knowledge that cites covered_obs
        record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Active knowledge covering obs",
            knowledge="Synthesized from covered obs.",
            supporting_observation_ids=[covered_obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
        )

        context = memory_synthesis.build_synthesis_context(
            memory_dir=memories,
            project_id="P-test",
            seed_observation_ids=(
                covered_obs["observation_id"],
                uncovered_obs["observation_id"],
            ),
        )
        assert context is not None
        ids = {o["id"] for o in context["uncovered_observations"]}
        assert covered_obs["observation_id"] not in ids
        assert uncovered_obs["observation_id"] in ids

    def test_memory_inventory_accurate(self, tmp_path):
        from EvoScientist.memory.knowledge import record_knowledge_file

        memories = tmp_path / "memories"
        obs = self._record_obs(memories)
        obs2 = self._record_obs(
            memories,
            summary="Second observation",
            observation="Second finding.",
            why_it_matters="More context.",
        )
        # One active knowledge record
        record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Knowledge for inventory test",
            knowledge="Some knowledge.",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
        )

        context = memory_synthesis.build_synthesis_context(
            memory_dir=memories,
            project_id="P-test",
            seed_observation_ids=(obs["observation_id"], obs2["observation_id"]),
        )
        assert context is not None
        inventory = context["memory_inventory"]
        assert inventory["active_knowledge_count"] == 1
        assert inventory["seed_observation_count"] == 2

    def test_context_within_max_chars(self, tmp_path):
        memories = tmp_path / "memories"
        obs = self._record_obs(memories, summary="A " * 200, observation="B " * 200, why_it_matters="C " * 50)
        context = memory_synthesis.build_synthesis_context(
            memory_dir=memories,
            project_id="P-test",
            seed_observation_ids=(obs["observation_id"],),
            max_chars=1000,
        )
        # Should return a context even when shrinkage is needed
        assert context is not None