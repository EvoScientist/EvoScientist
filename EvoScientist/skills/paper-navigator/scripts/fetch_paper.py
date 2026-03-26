#!/usr/bin/env python3
"""Fetch full paper text using Semantic Scholar metadata + Jina Reader.

Resolves paper ID to a URL, then uses Jina Reader (r.jina.ai) to
convert the paper page to clean Markdown.
"""

import argparse
import os
import sys
import time

import httpx

S2_BASE = "https://api.semanticscholar.org/graph/v1"
JINA_PREFIX = "https://r.jina.ai/"

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


def _s2_headers() -> dict:
    h = {"User-Agent": "EvoScientist/1.0 (paper-navigator)"}
    key = os.environ.get("S2_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def _jina_headers() -> dict:
    headers = {"Accept": "text/markdown"}
    key = os.environ.get("JINA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _normalize_paper_id(raw: str) -> str:
    raw = raw.strip()
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/",
                    "https://arxiv.org/pdf/", "http://arxiv.org/pdf/"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].rstrip(".pdf")
            return f"ArXiv:{raw}"
    if raw.startswith("10."):
        return f"DOI:{raw}"
    return raw


def resolve_paper_url(paper_id: str) -> tuple[str, dict]:
    """Resolve paper ID to best available URL + metadata."""
    pid = _normalize_paper_id(paper_id)

    fields = "paperId,externalIds,title,authors,year,citationCount,tldr,isOpenAccess,openAccessPdf,abstract"

    with httpx.Client() as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.get(f"{S2_BASE}/paper/{pid}",
                                  params={"fields": fields},
                                  headers=_s2_headers(), timeout=30)
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAYS[attempt])
                        continue
                resp.raise_for_status()
                meta = resp.json()
                break
            except httpx.HTTPError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                raise SystemExit(f"Error resolving paper: {e}") from e
        else:
            meta = {}

    # Determine best URL
    url = ""
    # Prefer OA PDF
    if meta.get("openAccessPdf") and meta["openAccessPdf"].get("url"):
        url = meta["openAccessPdf"]["url"]
    # Fallback: arXiv abstract page (Jina handles HTML well)
    elif meta.get("externalIds", {}).get("ArXiv"):
        arxiv_id = meta["externalIds"]["ArXiv"]
        url = f"https://arxiv.org/abs/{arxiv_id}"
    # Fallback: DOI
    elif meta.get("externalIds", {}).get("DOI"):
        doi = meta["externalIds"]["DOI"]
        url = f"https://doi.org/{doi}"

    return url, meta


def fetch_via_jina(url: str, limit_chars: int = 50000) -> str:
    """Fetch URL content as Markdown via Jina Reader."""
    jina_url = f"{JINA_PREFIX}{url}"

    with httpx.Client() as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.get(jina_url, headers=_jina_headers(), timeout=60,
                                  follow_redirects=True)
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAYS[attempt])
                        continue
                resp.raise_for_status()
                text = resp.text
                if len(text) > limit_chars:
                    text = text[:limit_chars] + f"\n\n---\n*[Truncated at {limit_chars} characters]*"
                return text
            except httpx.HTTPError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                return f"Error fetching paper: {e}"
    return ""


def format_metadata(meta: dict) -> str:
    """Format paper metadata header."""
    title = meta.get("title", "Unknown")
    authors = meta.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors[:5])
    if len(authors) > 5:
        author_str += f" et al. ({len(authors)} authors)"
    year = meta.get("year", "?")
    citations = meta.get("citationCount", 0)

    tldr = ""
    if meta.get("tldr") and meta["tldr"].get("text"):
        tldr = f"\n> **TLDR:** {meta['tldr']['text']}\n"

    ext = meta.get("externalIds", {})
    ids = []
    if ext.get("ArXiv"):
        ids.append(f"arXiv: `{ext['ArXiv']}`")
    if ext.get("DOI"):
        ids.append(f"DOI: `{ext['DOI']}`")
    ids.append(f"S2: `{meta.get('paperId', '')}`")

    return (f"# {title}\n\n"
            f"**{author_str}** ({year}) | Citations: {citations}\n"
            f"{' | '.join(ids)}\n"
            f"{tldr}\n---\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch paper full text via Jina Reader")
    parser.add_argument("--paper-id", "-p", help="Paper ID (S2, ArXiv:, DOI:, or URL)")
    parser.add_argument("--url", "-u", help="Direct URL to fetch (skip S2 lookup)")
    parser.add_argument("--limit-chars", type=int, default=50000,
                        help="Max output characters (default 50000)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only show metadata, skip full text")
    args = parser.parse_args()

    if not args.paper_id and not args.url:
        print("Error: --paper-id or --url required", file=sys.stderr)
        sys.exit(1)

    if args.url:
        # Direct URL mode
        url = args.url
        meta = {}
        # Try to get metadata from S2 if it looks like an arXiv URL
        if "arxiv.org" in url:
            arxiv_id = url.split("/abs/")[-1].split("/pdf/")[-1].rstrip(".pdf")
            try:
                _, meta = resolve_paper_url(f"ArXiv:{arxiv_id}")
            except Exception:
                pass
    else:
        url, meta = resolve_paper_url(args.paper_id)

    if meta:
        print(format_metadata(meta))

    if args.metadata_only:
        if meta.get("abstract"):
            print(f"## Abstract\n\n{meta['abstract']}\n")
        return

    if not url:
        print("Error: no accessible URL found for this paper", file=sys.stderr)
        if meta.get("abstract"):
            print(f"\n## Abstract (full text not available)\n\n{meta['abstract']}\n")
        sys.exit(1)

    print(f"*Fetching from: {url}*\n", file=sys.stderr)
    content = fetch_via_jina(url, args.limit_chars)
    print(content)


if __name__ == "__main__":
    main()
