#!/usr/bin/env python3
"""Search for authors and their papers via Semantic Scholar API."""

import argparse
import json
import os
import sys
import time

import httpx

S2_BASE = "https://api.semanticscholar.org/graph/v1"
AUTHOR_FIELDS = "authorId,name,affiliations,paperCount,citationCount,hIndex"
PAPER_FIELDS = "paperId,externalIds,title,year,citationCount,tldr,isOpenAccess,openAccessPdf"

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


def search_author(name: str) -> list[dict]:
    """Search for authors by name."""
    with httpx.Client() as client:
        data = _request_with_retry(client, f"{S2_BASE}/author/search",
                                   {"query": name, "fields": AUTHOR_FIELDS, "limit": 5})
    return data.get("data", [])


def get_author_papers(author_id: str, limit: int = 20) -> list[dict]:
    """Get papers by a specific author."""
    with httpx.Client() as client:
        data = _request_with_retry(
            client,
            f"{S2_BASE}/author/{author_id}/papers",
            {"fields": PAPER_FIELDS, "limit": min(limit, 1000)}
        )
    return data.get("data", [])


def format_author(a: dict) -> str:
    name = a.get("name", "Unknown")
    aid = a.get("authorId", "")
    affiliations = ", ".join(a.get("affiliations", [])[:2]) or "N/A"
    papers = a.get("paperCount", 0)
    citations = a.get("citationCount", 0)
    h_index = a.get("hIndex", "?")
    return (f"**{name}** (ID: `{aid}`)\n"
            f"  Affiliation: {affiliations} | Papers: {papers} | "
            f"Citations: {citations} | h-index: {h_index}")


def format_paper(p: dict, idx: int) -> str:
    title = p.get("title", "Unknown")
    year = p.get("year", "?")
    citations = p.get("citationCount", 0)
    ext = p.get("externalIds", {})
    arxiv = ext.get("ArXiv", "")
    id_str = f"arXiv:`{arxiv}`" if arxiv else f"S2:`{p.get('paperId', '')[:12]}…`"

    tldr = ""
    if p.get("tldr") and p["tldr"].get("text"):
        tldr = f"\n  > {p['tldr']['text']}"

    return f"{idx}. **{title}** ({year}) — ⭐{citations} — {id_str}{tldr}"


def main():
    parser = argparse.ArgumentParser(description="Search authors via Semantic Scholar")
    parser.add_argument("--name", help="Author name to search")
    parser.add_argument("--author-id", help="S2 author ID (skip name search)")
    parser.add_argument("--papers", action="store_true", help="List author's papers")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Max papers (default 20)")
    parser.add_argument("--sort-by", choices=["citations", "year"], default="year")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.name and not args.author_id:
        print("Error: --name or --author-id required", file=sys.stderr)
        sys.exit(1)

    # Resolve author ID
    author_id = args.author_id
    if not author_id:
        authors = search_author(args.name)
        if not authors:
            print(f"No author found for '{args.name}'", file=sys.stderr)
            sys.exit(0)

        if args.json and not args.papers:
            print(json.dumps(authors, indent=2))
            return

        print("# Author Search Results\n")
        for a in authors:
            print(format_author(a))
            print()

        author_id = authors[0]["authorId"]
        if not args.papers:
            return
        print("---\n")

    # Get papers
    if args.papers or args.author_id:
        papers = get_author_papers(author_id, args.limit)

        if args.sort_by == "citations":
            papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
        else:
            papers.sort(key=lambda p: p.get("year", 0), reverse=True)

        papers = papers[:args.limit]

        if args.json:
            print(json.dumps(papers, indent=2))
            return

        print(f"# Papers by Author `{author_id}`\n")
        print(f"Showing **{len(papers)}** papers (sorted by {args.sort_by})\n")
        for i, p in enumerate(papers, 1):
            print(format_paper(p, i))
        print()


if __name__ == "__main__":
    main()
