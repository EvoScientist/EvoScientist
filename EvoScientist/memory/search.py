"""Search helpers for file-backed observation memory."""

from __future__ import annotations

import math
import re
from collections import Counter

from .types import (
    ObservationSearchDocument,
    ObservationSearchHit,
    ObservationSearchMode,
)

MIN_TOKEN_CHARS = 3
ID_MATCH_WEIGHT = 5.0
SUMMARY_MATCH_WEIGHT = 3.0
BODY_MATCH_WEIGHT = 1.0
METADATA_MATCH_WEIGHT = 0.5
IDF_SMOOTHING = 0.5
IDF_OFFSET = 1.0
DEFAULT_MATCH_LINES = 3
DEFAULT_MATCH_CHARS = 240

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _compile_query_pattern(query: str) -> re.Pattern[str]:
    """Compile a case-insensitive regex, falling back to literal matching."""
    try:
        return re.compile(query, flags=re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(query), flags=re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    """Return simple lowercase search tokens."""
    return [
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= MIN_TOKEN_CHARS
    ]


def _document_tokens(document: ObservationSearchDocument) -> set[str]:
    """Return unique tokens used for IDF calculation."""
    return set(
        _tokens(
            " ".join(
                [
                    document.observation_id,
                    document.summary,
                    str(document.memory_type),
                    str(document.scope),
                    document.body,
                ]
            )
        )
    )


def _token_idf(documents: list[ObservationSearchDocument]) -> dict[str, float]:
    """Compute smoothed IDF over the current observation corpus."""
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(_document_tokens(document))
    document_count = len(documents)
    return {
        token: math.log((document_count + 1) / (count + IDF_SMOOTHING)) + IDF_OFFSET
        for token, count in document_frequency.items()
    }


def _ranked_score(
    *,
    query_tokens: set[str],
    document: ObservationSearchDocument,
    idf: dict[str, float],
) -> float:
    """Score a document with named token-overlap weights."""
    id_tokens = set(_tokens(document.observation_id))
    summary_tokens = set(_tokens(document.summary))
    body_tokens = set(_tokens(document.body))
    metadata_tokens = set(_tokens(f"{document.memory_type} {document.scope}"))

    score = 0.0
    for token in query_tokens:
        token_weight = idf.get(token, 0.0)
        if token in id_tokens:
            score += ID_MATCH_WEIGHT * token_weight
        if token in summary_tokens:
            score += SUMMARY_MATCH_WEIGHT * token_weight
        if token in body_tokens:
            score += BODY_MATCH_WEIGHT * token_weight
        elif token in metadata_tokens:
            score += METADATA_MATCH_WEIGHT * token_weight
    return score


def _match_snippet(
    text: str,
    match: re.Match[str],
    *,
    max_chars: int,
) -> str:
    """Return compact context around a regex match."""
    context = max_chars // 3
    start = max(0, match.start() - context)
    end = min(len(text), match.end() + (max_chars - context))
    return " ".join(text[start:end].split())[:max_chars]


def _regex_matching_lines(
    *,
    body: str,
    summary: str,
    pattern: re.Pattern[str],
    max_lines: int = DEFAULT_MATCH_LINES,
    max_chars: int = DEFAULT_MATCH_CHARS,
) -> list[str]:
    """Return compact grep-like matching lines."""
    matches: list[str] = []
    if pattern.search(summary):
        matches.append(summary[:max_chars])
    candidates = [line.strip() for line in body.splitlines() if line.strip()]
    for line in candidates:
        if pattern.search(line):
            matches.append(line[:max_chars])
            if len(matches) >= max_lines:
                return matches
    body_match = pattern.search(body)
    if body_match and len(matches) < max_lines:
        matches.append(_match_snippet(body, body_match, max_chars=max_chars))
    if matches:
        return matches
    if summary:
        return [summary[:max_chars]]
    return [(candidates[0] if candidates else "")[:max_chars]]


def _ranked_matching_lines(
    *,
    body: str,
    query_tokens: set[str],
    max_lines: int = DEFAULT_MATCH_LINES,
    max_chars: int = DEFAULT_MATCH_CHARS,
) -> list[str]:
    """Return compact lines that explain a ranked match."""
    matches: list[str] = []

    scored_lines: list[tuple[int, int, str]] = []
    for index, line in enumerate(raw_line.strip() for raw_line in body.splitlines()):
        if not line:
            continue
        overlap = len(query_tokens & set(_tokens(line)))
        if overlap:
            scored_lines.append((-overlap, index, line[:max_chars]))
    for _, _, line in sorted(scored_lines):
        if line not in matches:
            matches.append(line)
        if len(matches) >= max_lines:
            return matches

    return matches[:max_lines]


def _observation_haystack(document: ObservationSearchDocument) -> str:
    """Return searchable text for regex search."""
    return "\n".join(
        [
            document.observation_id,
            document.summary,
            str(document.memory_type),
            str(document.scope),
            document.body,
        ]
    )


def _regex_search_documents(
    *,
    documents: list[ObservationSearchDocument],
    query: str,
    limit: int,
) -> list[ObservationSearchHit]:
    """Search observations with grep-like regex semantics."""
    pattern = _compile_query_pattern(query)
    hits: list[ObservationSearchHit] = []
    for document in documents:
        if pattern.search(_observation_haystack(document)) is None:
            continue
        hit: ObservationSearchHit = {
            "observation_id": document.observation_id,
            "path": document.path,
            "memory_type": document.memory_type,
            "scope": document.scope,
            "summary": document.summary,
            "matches": _regex_matching_lines(
                body=document.body,
                summary=document.summary,
                pattern=pattern,
            ),
        }
        if document.related_observations:
            hit["related_observations"] = list(document.related_observations)
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def _ranked_search_documents(
    *,
    documents: list[ObservationSearchDocument],
    query: str,
    limit: int,
) -> list[ObservationSearchHit]:
    """Search observations with token-overlap ranking."""
    if not documents:
        return []
    query_tokens = set(_tokens(query.replace("|", " ")))
    if not query_tokens:
        return _regex_search_documents(documents=documents, query=query, limit=limit)

    idf = _token_idf(documents)
    scored = [
        (
            _ranked_score(
                query_tokens=query_tokens,
                document=document,
                idf=idf,
            ),
            index,
            document,
        )
        for index, document in enumerate(documents)
    ]
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    positive_ranked = [item for item in ranked if item[0] > 0]
    if not positive_ranked:
        return []
    selected = positive_ranked[:limit]

    hits: list[ObservationSearchHit] = []
    for score, _, document in selected:
        hit: ObservationSearchHit = {
            "observation_id": document.observation_id,
            "path": document.path,
            "memory_type": document.memory_type,
            "scope": document.scope,
            "summary": document.summary,
            "matches": _ranked_matching_lines(
                body=document.body,
                query_tokens=query_tokens,
            ),
            "score": round(score, 2),
        }
        if document.related_observations:
            hit["related_observations"] = list(document.related_observations)
        hits.append(hit)
    return hits


def search_documents(
    *,
    documents: list[ObservationSearchDocument],
    query: str,
    limit: int,
    mode: ObservationSearchMode,
) -> list[ObservationSearchHit]:
    """Search parsed observation documents."""
    if mode == ObservationSearchMode.REGEX:
        return _regex_search_documents(documents=documents, query=query, limit=limit)
    return _ranked_search_documents(documents=documents, query=query, limit=limit)
