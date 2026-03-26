#!/usr/bin/env python3
"""Detect trending papers via Semantic Scholar.

Finds papers with high citation velocity (citations per month)
within a recent time period.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import httpx

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,authors,year,citationCount,influentialCitationCount,tldr,isOpenAccess,openAccessPdf,publicationDate"

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


def _citation_velocity(paper: dict) -> float:
    """Calculate citations per month since publication."""
    pub_date = paper.get("publicationDate")
    citations = paper.get("citationCount", 0)
    if not pub_date or not citations:
        return 0.0
    try:
        pub = datetime.strptime(pub_date, "%Y-%m-%d")
        months = max((datetime.now() - pub).days / 30.0, 1.0)
        return citations / months
    except (ValueError, TypeError):
        return 0.0


def find_trending(query: str, period_days: int = 90, min_citations: int = 5,
                  limit: int = 20) -> list[dict]:
    """Search for papers and rank by citation velocity."""
    now = datetime.now()
    year_min = now.year - max(period_days // 365, 1)

    params: dict = {
        "query": query,
        "limit": 100,  # Fetch more to filter
        "fields": S2_FIELDS,
        "year": f"{year_min}-",
    }

    with httpx.Client() as client:
        data = _request_with_retry(client, f"{S2_BASE}/paper/search", params)

    papers = data.get("data", [])

    # Filter by min citations
    papers = [p for p in papers if (p.get("citationCount") or 0) >= min_citations]

    # Calculate velocity and sort
    for p in papers:
        p["_velocity"] = _citation_velocity(p)

    papers.sort(key=lambda p: p["_velocity"], reverse=True)
    return papers[:limit]


def format_paper(p: dict, idx: int) -> str:
    title = p.get("title", "Unknown")
    year = p.get("year", "?")
    citations = p.get("citationCount", 0)
    velocity = p.get("_velocity", 0)
    pub_date = p.get("publicationDate", "")

    authors = p.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors[:3])
    if len(authors) > 3:
        author_str += " et al."

    tldr = ""
    if p.get("tldr") and p["tldr"].get("text"):
        tldr = f"\n  > {p['tldr']['text']}"

    ext = p.get("externalIds", {})
    arxiv = ext.get("ArXiv", "")
    id_str = f"arXiv:`{arxiv}`" if arxiv else f"S2:`{p.get('paperId', '')[:12]}…`"

    return (f"{idx}. **{title}**\n"
            f"  {author_str} ({year}) | Published: {pub_date}\n"
            f"  ⭐ {citations} citations | 🔥 **{velocity:.1f} cit/month** | {id_str}{tldr}\n")


def main():
    parser = argparse.ArgumentParser(description="Detect trending papers via Semantic Scholar")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--period", "-p", type=int, default=90,
                        help="Look back period in days (default 90)")
    parser.add_argument("--min-citations", type=int, default=5,
                        help="Minimum citations filter (default 5)")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Max results (default 20)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    papers = find_trending(args.query, args.period, args.min_citations, args.limit)

    if not papers:
        print(f"No trending papers found for '{args.query}'", file=sys.stderr)
        sys.exit(0)

    if args.json:
        print(json.dumps(papers, indent=2, default=str))
        return

    print(f"# Trending Papers: \"{args.query}\"\n")
    print(f"Period: last **{args.period}** days | Min citations: {args.min_citations}\n")
    for i, p in enumerate(papers, 1):
        print(format_paper(p, i))


if __name__ == "__main__":
    main()
