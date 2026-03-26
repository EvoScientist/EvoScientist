#!/usr/bin/env python3
"""Traverse citation graphs via Semantic Scholar API.

Supports forward citations (who cited this), backward citations
(references), and co-citation discovery.
"""

import argparse
import json
import os
import sys
import time

import httpx

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,authors,year,citationCount,influentialCitationCount,tldr,isOpenAccess,openAccessPdf"

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


def _headers() -> dict:
    h = {"User-Agent": "EvoScientist/1.0 (paper-navigator)"}
    key = os.environ.get("S2_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def _request_with_retry(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, params=params, headers=_headers(), timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise SystemExit(f"Error: {e}") from e
    return {}


def _normalize_paper_id(raw: str) -> str:
    """Normalize paper ID: strip URL prefixes, add ArXiv: prefix if needed."""
    raw = raw.strip()
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/",
                    "https://arxiv.org/pdf/", "http://arxiv.org/pdf/"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].rstrip(".pdf")
            return f"ArXiv:{raw}"
    if raw.startswith("10."):
        return f"DOI:{raw}"
    return raw


def get_citations(paper_id: str, limit: int = 20) -> list[dict]:
    """Forward citations: papers that cite this paper."""
    params = {"fields": S2_FIELDS, "limit": min(limit, 1000)}
    with httpx.Client() as client:
        data = _request_with_retry(client, f"{S2_BASE}/paper/{paper_id}/citations", params)
    return [c["citingPaper"] for c in data.get("data", []) if c.get("citingPaper")]


def get_references(paper_id: str, limit: int = 20) -> list[dict]:
    """Backward citations: papers this paper references."""
    params = {"fields": S2_FIELDS, "limit": min(limit, 1000)}
    with httpx.Client() as client:
        data = _request_with_retry(client, f"{S2_BASE}/paper/{paper_id}/references", params)
    return [r["citedPaper"] for r in data.get("data", []) if r.get("citedPaper")]


def get_co_citations(paper_id: str, limit: int = 15) -> list[dict]:
    """Co-citation: papers frequently cited alongside this paper.

    Strategy: get forward citations, collect their references,
    find most common papers (excluding the seed).
    """
    # Get papers that cite the seed
    citers = get_citations(paper_id, limit=50)
    if not citers:
        return []

    # Sample up to 10 citers to stay within rate limits
    sample = citers[:10]
    ref_counts: dict[str, dict] = {}

    with httpx.Client() as client:
        for citer in sample:
            citer_id = citer.get("paperId")
            if not citer_id:
                continue
            try:
                params = {"fields": S2_FIELDS, "limit": 100}
                data = _request_with_retry(client, f"{S2_BASE}/paper/{citer_id}/references", params)
                for r in data.get("data", []):
                    ref = r.get("citedPaper", {})
                    rid = ref.get("paperId")
                    if rid and rid != paper_id:
                        if rid not in ref_counts:
                            ref_counts[rid] = {"paper": ref, "count": 0}
                        ref_counts[rid]["count"] += 1
                time.sleep(0.5)  # Rate limit courtesy
            except Exception:
                continue

    # Sort by co-citation frequency
    sorted_refs = sorted(ref_counts.values(), key=lambda x: x["count"], reverse=True)
    return [r["paper"] for r in sorted_refs[:limit]]


def format_paper(p: dict, idx: int) -> str:
    title = p.get("title", "Unknown")
    year = p.get("year", "?")
    citations = p.get("citationCount", 0)
    authors = p.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors[:3])
    if len(authors) > 3:
        author_str += " et al."

    tldr = ""
    if p.get("tldr") and p["tldr"].get("text"):
        tldr = f"\n  > {p['tldr']['text']}"

    pid = p.get("paperId", "")
    ext = p.get("externalIds", {})
    arxiv = ext.get("ArXiv", "")
    id_str = f"S2:`{pid[:8]}…`"
    if arxiv:
        id_str = f"arXiv:`{arxiv}`"

    return f"{idx}. **{title}** — {author_str} ({year}) — ⭐{citations} — {id_str}{tldr}"


def main():
    parser = argparse.ArgumentParser(description="Traverse citation graphs via Semantic Scholar")
    parser.add_argument("--paper-id", "-p", required=True, help="Paper ID (S2, ArXiv:, DOI:, or URL)")
    parser.add_argument("--direction", "-d", required=True,
                        choices=["forward", "backward", "co-citation"],
                        help="Traversal direction")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Max results (default 20)")
    parser.add_argument("--min-citations", type=int, default=0,
                        help="Minimum citation count filter")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    paper_id = _normalize_paper_id(args.paper_id)

    direction_labels = {
        "forward": "Forward Citations (papers citing this paper)",
        "backward": "Backward Citations (papers referenced by this paper)",
        "co-citation": "Co-cited Papers (frequently cited alongside this paper)",
    }

    print(f"# {direction_labels[args.direction]}\n", file=sys.stderr)
    print(f"Seed paper: `{paper_id}`\n", file=sys.stderr)

    if args.direction == "forward":
        papers = get_citations(paper_id, args.limit)
    elif args.direction == "backward":
        papers = get_references(paper_id, args.limit)
    else:
        print("⚠️  Co-citation requires multiple API calls (may be slow)…\n", file=sys.stderr)
        papers = get_co_citations(paper_id, args.limit)

    # Filter by min citations
    if args.min_citations > 0:
        papers = [p for p in papers if (p.get("citationCount") or 0) >= args.min_citations]

    # Sort by citations
    papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    papers = papers[:args.limit]

    if not papers:
        print("No papers found.", file=sys.stderr)
        sys.exit(0)

    if args.json:
        print(json.dumps(papers, indent=2))
        return

    print(f"# {direction_labels[args.direction]}\n")
    print(f"Seed: `{paper_id}` | Results: **{len(papers)}**\n")
    for i, p in enumerate(papers, 1):
        print(format_paper(p, i))
    print()


if __name__ == "__main__":
    main()
