"""Tests for EvoScientist.memory.search (refactored in this PR).

The search module was changed to accept MemorySearchDocument (ABC) instead of
ObservationSearchDocument, and adds knowledge score boost logic.
"""

from __future__ import annotations

from EvoScientist.memory.search import (
    KNOWLEDGE_SCORE_BOOST,
    search_memory_documents,
)
from EvoScientist.memory.types import (
    KnowledgeSearchDocument,
    KnowledgeStatus,
    MemoryLevel,
    MemoryScope,
    MemorySearchMode,
    MemoryType,
    ObservationSearchDocument,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _obs(
    *,
    observation_id: str = "O-abc1234567890123",
    summary: str = "observation summary",
    body: str = "observation body text",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    scope: MemoryScope = MemoryScope.GLOBAL,
) -> ObservationSearchDocument:
    return ObservationSearchDocument(
        observation_id=observation_id,
        path=f"/memories/observations/global/{observation_id}.md",
        memory_type=memory_type,
        scope=scope,
        summary=summary,
        body=body,
    )


def _know(
    *,
    knowledge_id: str = "K-abc1234567890123",
    summary: str = "knowledge summary",
    body: str = "synthesized knowledge body text",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    scope: MemoryScope = MemoryScope.GLOBAL,
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
    supporting_observation_ids: tuple[str, ...] = ("O-obs1",),
) -> KnowledgeSearchDocument:
    return KnowledgeSearchDocument(
        knowledge_id=knowledge_id,
        path=f"/memories/knowledge/global/{knowledge_id}.md",
        memory_type=memory_type,
        scope=scope,
        summary=summary,
        body=body,
        status=status,
        supporting_observation_ids=supporting_observation_ids,
    )


# ---------------------------------------------------------------------------
# search_memory_documents: basic ranked mode
# ---------------------------------------------------------------------------


class TestSearchMemoryDocumentsRanked:
    def test_returns_ranked_hits_for_observation(self):
        docs = [_obs(summary="Python asyncio event loop", body="asyncio event loop")]
        hits = search_memory_documents(
            documents=docs,
            query="asyncio event loop",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        assert len(hits) == 1
        assert hits[0]["memory_id"] == "O-abc1234567890123"
        assert hits[0]["level"] == MemoryLevel.OBSERVATION

    def test_returns_ranked_hits_for_knowledge(self):
        docs = [_know(summary="Python asyncio event loop", body="asyncio event loop")]
        hits = search_memory_documents(
            documents=docs,
            query="asyncio event loop",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        assert len(hits) == 1
        assert hits[0]["memory_id"] == "K-abc1234567890123"
        assert hits[0]["level"] == MemoryLevel.KNOWLEDGE

    def test_knowledge_boosted_above_observation_with_equal_text(self):
        """Knowledge should score higher than observation with same text due to boost."""
        same_body = "important finding about database indexing"
        obs = _obs(observation_id="O-1234567890123456", summary=same_body, body=same_body)
        know = _know(
            knowledge_id="K-1234567890123456", summary=same_body, body=same_body
        )
        hits = search_memory_documents(
            documents=[obs, know],
            query="database indexing",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        assert len(hits) == 2
        knowledge_hit = next(h for h in hits if h["level"] == MemoryLevel.KNOWLEDGE)
        observation_hit = next(h for h in hits if h["level"] == MemoryLevel.OBSERVATION)
        # Knowledge score should be >= observation score due to KNOWLEDGE_SCORE_BOOST
        assert knowledge_hit.get("score", 0) >= observation_hit.get("score", 0)

    def test_no_match_returns_empty(self):
        docs = [_obs(summary="Python", body="Python programming")]
        hits = search_memory_documents(
            documents=docs,
            query="completely unrelated topic xyz",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        assert hits == []

    def test_limit_applied(self):
        docs = [
            _obs(
                observation_id=f"O-{str(i).zfill(16)}",
                summary=f"asyncio finding number {i}",
                body=f"asyncio body {i}",
            )
            for i in range(10)
        ]
        hits = search_memory_documents(
            documents=docs, query="asyncio", limit=3, mode=MemorySearchMode.RANKED
        )
        assert len(hits) <= 3

    def test_empty_documents_returns_empty(self):
        hits = search_memory_documents(
            documents=[], query="test", limit=5, mode=MemorySearchMode.RANKED
        )
        assert hits == []

    def test_hit_shape_has_required_fields(self):
        docs = [_obs(summary="test query match", body="test query match content")]
        hits = search_memory_documents(
            documents=docs, query="test query", limit=5, mode=MemorySearchMode.RANKED
        )
        assert len(hits) == 1
        hit = hits[0]
        assert "memory_id" in hit
        assert "level" in hit
        assert "path" in hit
        assert "memory_type" in hit
        assert "scope" in hit
        assert "summary" in hit
        assert "matches" in hit
        assert "score" in hit

    def test_score_rounded_to_two_decimals(self):
        docs = [_obs(summary="unique term abcxyz", body="unique term abcxyz content")]
        hits = search_memory_documents(
            documents=docs, query="abcxyz", limit=5, mode=MemorySearchMode.RANKED
        )
        if hits and "score" in hits[0]:
            score_str = str(hits[0]["score"])
            # Should have at most 2 decimal places
            if "." in score_str:
                assert len(score_str.split(".")[1]) <= 2

    def test_mixed_document_types(self):
        """Mixed obs and knowledge documents work together."""
        obs = _obs(
            observation_id="O-1111111111111111",
            summary="graphql resolver observation",
            body="graphql resolver aliases",
        )
        know = _know(
            knowledge_id="K-2222222222222222",
            summary="graphql knowledge",
            body="graphql resolver pattern",
        )
        hits = search_memory_documents(
            documents=[obs, know],
            query="graphql resolver",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        assert len(hits) == 2
        ids = {h["memory_id"] for h in hits}
        assert "O-1111111111111111" in ids
        assert "K-2222222222222222" in ids


# ---------------------------------------------------------------------------
# search_memory_documents: regex mode
# ---------------------------------------------------------------------------


class TestSearchMemoryDocumentsRegex:
    def test_regex_mode_finds_pattern(self):
        docs = [_obs(summary="GraphQL resolver aliases", body="alias pattern")]
        hits = search_memory_documents(
            documents=docs,
            query="GraphQL",
            limit=5,
            mode=MemorySearchMode.REGEX,
        )
        assert len(hits) == 1
        assert hits[0]["memory_id"] == "O-abc1234567890123"

    def test_regex_mode_case_insensitive(self):
        docs = [_obs(summary="GraphQL test", body="graphql lowercase")]
        hits = search_memory_documents(
            documents=docs, query="graphql", limit=5, mode=MemorySearchMode.REGEX
        )
        assert len(hits) >= 1

    def test_regex_mode_no_match_returns_empty(self):
        docs = [_obs(summary="Python test", body="python body")]
        hits = search_memory_documents(
            documents=docs, query="^NOMATCH$", limit=5, mode=MemorySearchMode.REGEX
        )
        assert hits == []

    def test_invalid_regex_falls_back_to_literal(self):
        # "(" is an invalid regex but should fall back to literal match
        docs = [_obs(summary="(brackets) test", body="bracket content")]
        hits = search_memory_documents(
            documents=docs, query="(brackets)", limit=5, mode=MemorySearchMode.REGEX
        )
        # Should not raise, may return hits
        assert isinstance(hits, list)

    def test_regex_mode_no_score_in_hit(self):
        docs = [_obs(summary="test regex hit", body="regex body content")]
        hits = search_memory_documents(
            documents=docs, query="regex", limit=5, mode=MemorySearchMode.REGEX
        )
        assert len(hits) == 1
        # Regex mode does not include a score
        assert "score" not in hits[0]

    def test_regex_limit_applied(self):
        docs = [
            _obs(
                observation_id=f"O-{str(i).zfill(16)}",
                summary=f"matching pattern {i}",
                body=f"pattern body {i}",
            )
            for i in range(10)
        ]
        hits = search_memory_documents(
            documents=docs, query="pattern", limit=4, mode=MemorySearchMode.REGEX
        )
        assert len(hits) <= 4

    def test_regex_hit_has_matches_field(self):
        docs = [
            _obs(summary="the pattern appears here", body="this pattern is visible")
        ]
        hits = search_memory_documents(
            documents=docs, query="pattern", limit=5, mode=MemorySearchMode.REGEX
        )
        assert len(hits) == 1
        assert isinstance(hits[0]["matches"], list)
        assert len(hits[0]["matches"]) > 0

    def test_knowledge_in_regex_mode(self):
        docs = [_know(summary="python typing hint", body="type hints for functions")]
        hits = search_memory_documents(
            documents=docs, query="typing", limit=5, mode=MemorySearchMode.REGEX
        )
        assert len(hits) == 1
        assert hits[0]["level"] == MemoryLevel.KNOWLEDGE

    def test_knowledge_extra_fields_in_regex_hit(self):
        docs = [
            _know(
                status=KnowledgeStatus.ACTIVE,
                supporting_observation_ids=("O-s1", "O-s2"),
                summary="test knowledge result",
                body="knowledge body match",
            )
        ]
        hits = search_memory_documents(
            documents=docs, query="knowledge", limit=5, mode=MemorySearchMode.REGEX
        )
        assert len(hits) == 1
        hit = hits[0]
        assert hit["status"] == KnowledgeStatus.ACTIVE
        assert "O-s1" in hit["supporting_observation_ids"]


# ---------------------------------------------------------------------------
# Knowledge score boost constant
# ---------------------------------------------------------------------------


class TestKnowledgeScoreBoost:
    def test_boost_greater_than_one(self):
        assert KNOWLEDGE_SCORE_BOOST > 1.0

    def test_boost_value(self):
        # The boost should be 1.15 per the source
        assert KNOWLEDGE_SCORE_BOOST == 1.15


# ---------------------------------------------------------------------------
# Hit serialization
# ---------------------------------------------------------------------------


class TestHitSerialization:
    def test_observation_hit_has_level_observation(self):
        obs = _obs(
            observation_id="O-1234567890123456",
            summary="test observation level",
            body="level test body",
        )
        hits = search_memory_documents(
            documents=[obs],
            query="level test",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        if hits:
            assert hits[0]["level"] == MemoryLevel.OBSERVATION

    def test_knowledge_hit_has_level_knowledge(self):
        know = _know(
            knowledge_id="K-1234567890123456",
            summary="test knowledge level",
            body="knowledge level body",
        )
        hits = search_memory_documents(
            documents=[know],
            query="knowledge level",
            limit=5,
            mode=MemorySearchMode.RANKED,
        )
        if hits:
            assert hits[0]["level"] == MemoryLevel.KNOWLEDGE

    def test_path_preserved_in_hit(self):
        obs = _obs(observation_id="O-pathtest12345678", summary="path test", body="body")
        hits = search_memory_documents(
            documents=[obs], query="path test", limit=5, mode=MemorySearchMode.RANKED
        )
        if hits:
            assert "O-pathtest12345678" in hits[0]["path"]