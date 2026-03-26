#!/usr/bin/env python3
"""Search academic papers via Semantic Scholar API.

Returns paper metadata including title, authors, year, citation count,
TLDR summary, and open-access PDF links.
"""

import argparse
import json
import os
import sys
import time

import httpx

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,authors,year,citationCount,influentialCitationCount,tldr,isOpenAccess,openAccessPdf,publicationVenue,abstract"

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


def _headers() -> dict:
    h = {"User-Agent": "EvoScientist/1.0 (paper-navigator)"}
    key = os.environ.get("S2_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def _request_with_retry(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    """GET with retry on 429/5xx."""
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


def search(query: str, limit: int = 10, year_min: int | None = None,
           year_max: int | None = None, open_access_only: bool = False) -> list[dict]:
    """Search S2 for papers matching query."""
    params: dict = {
        "query": query,
        "limit": min(limit, 100),
        "fields": S2_FIELDS,
    }
    if year_min or year_max:
        lo = year_min or ""
        hi = year_max or ""
        params["year"] = f"{lo}-{hi}"
    if open_access_only:
        params["openAccessPdf"] = ""

    with httpx.Client() as client:
        data = _request_with_retry(client, f"{S2_BASE}/paper/search", params)
    return data.get("data", [])


def get_paper(paper_id: str) -> dict:
    """Get single paper details by S2 paper ID or external ID."""
    with httpx.Client() as client:
        return _request_with_retry(client, f"{S2_BASE}/paper/{paper_id}",
                                   {"fields": S2_FIELDS})


def _truncate(text: str | None, max_len: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def format_paper(p: dict, idx: int | None = None) -> str:
    """Format a single paper as Markdown."""
    prefix = f"### {idx}. " if idx is not None else "### "
    title = p.get("title", "Unknown")
    year = p.get("year", "?")
    citations = p.get("citationCount", 0)
    influential = p.get("influentialCitationCount", 0)

    authors = p.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors[:5])
    if len(authors) > 5:
        author_str += f" et al. ({len(authors)} authors)"

    venue = ""
    if p.get("publicationVenue"):
        venue = p["publicationVenue"].get("name", "")

    tldr = ""
    if p.get("tldr") and p["tldr"].get("text"):
        tldr = f"\n> **TLDR:** {p['tldr']['text']}"

    abstract = ""
    if not tldr and p.get("abstract"):
        abstract = f"\n> {_truncate(p['abstract'])}"

    pdf = ""
    if p.get("openAccessPdf") and p["openAccessPdf"].get("url"):
        pdf = f"\n📄 [Open Access PDF]({p['openAccessPdf']['url']})"

    paper_id = p.get("paperId", "")
    ext_ids = p.get("externalIds", {})
    arxiv_id = ext_ids.get("ArXiv", "")
    doi = ext_ids.get("DOI", "")

    ids_line = f"S2: `{paper_id}`"
    if arxiv_id:
        ids_line += f" | arXiv: `{arxiv_id}`"
    if doi:
        ids_line += f" | DOI: `{doi}`"

    return f"""{prefix}{title}
**{author_str}** ({year}) — {venue}
Citations: **{citations}** (influential: {influential})
{ids_line}{tldr}{abstract}{pdf}
"""


def main():
    parser = argparse.ArgumentParser(description="Search papers via Semantic Scholar")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--year-min", type=int, help="Minimum publication year")
    parser.add_argument("--year-max", type=int, help="Maximum publication year")
    parser.add_argument("--open-access-only", action="store_true", help="Only return OA papers")
    parser.add_argument("--sort-by", choices=["citations", "year", "relevance"],
                        default="relevance", help="Sort order")
    parser.add_argument("--paper-id", help="Get single paper by ID instead of searching")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.paper_id:
        paper = get_paper(args.paper_id)
        if args.json:
            print(json.dumps(paper, indent=2))
        else:
            print(format_paper(paper))
        return

    papers = search(args.query, args.limit, args.year_min, args.year_max, args.open_access_only)

    if not papers:
        print(f"No papers found for '{args.query}'", file=sys.stderr)
        sys.exit(0)

    if args.sort_by == "citations":
        papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    elif args.sort_by == "year":
        papers.sort(key=lambda p: p.get("year", 0), reverse=True)

    if args.json:
        print(json.dumps(papers, indent=2))
        return

    print(f"# Search Results: \"{args.query}\"\n")
    print(f"Found **{len(papers)}** papers\n")
    for i, p in enumerate(papers, 1):
        print(format_paper(p, i))


if __name__ == "__main__":
    main()
