#!/usr/bin/env python3
"""Monitor arXiv for new papers by category or keywords.

Uses the arXiv API to fetch recent papers from specific categories
or matching keywords.
"""

import argparse
import json
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


_UA = {"User-Agent": "EvoScientist/1.0 (paper-navigator)"}


def _request_with_retry(client: httpx.Client, url: str, params: dict) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, params=params, headers=_UA, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise SystemExit(f"Error: {e}") from e
    return ""


def _parse_entries(xml_text: str) -> list[dict]:
    """Parse arXiv Atom XML into paper dicts."""
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", NS)
    papers = []
    for entry in entries:
        title = entry.findtext("atom:title", "", NS).replace("\n", " ").strip()
        summary = entry.findtext("atom:summary", "", NS).strip()
        published = entry.findtext("atom:published", "", NS)
        updated = entry.findtext("atom:updated", "", NS)

        # Extract arXiv ID from the id URL
        id_url = entry.findtext("atom:id", "", NS)
        arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url

        authors = []
        for author in entry.findall("atom:author", NS):
            name = author.findtext("atom:name", "", NS)
            if name:
                authors.append(name)

        categories = []
        for cat in entry.findall("atom:category", NS):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        # PDF link
        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")

        # Comment (often contains page count, conference info)
        comment = ""
        comment_el = entry.find("arxiv:comment", NS)
        if comment_el is not None and comment_el.text:
            comment = comment_el.text.strip()

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "summary": summary[:300],
            "categories": categories,
            "published": published,
            "updated": updated,
            "pdf_url": pdf_url,
            "comment": comment,
        })
    return papers


def fetch_by_categories(categories: list[str], days: int = 1, limit: int = 50) -> list[dict]:
    """Fetch recent papers from specific arXiv categories."""
    # arXiv API query: cat:cs.CL OR cat:cs.AI
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    # Date filter via submittedDate
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    date_range = f"[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"

    query = f"({cat_query}) AND submittedDate:{date_range}"
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": min(limit, 200),
    }

    with httpx.Client() as client:
        xml_text = _request_with_retry(client, ARXIV_API, params)
    return _parse_entries(xml_text)


def fetch_by_keywords(keywords: list[str], days: int = 7, limit: int = 50) -> list[dict]:
    """Fetch recent papers matching keywords."""
    # Search in title and abstract
    kw_parts = []
    for kw in keywords:
        kw = kw.strip()
        if " " in kw:
            kw_parts.append(f'ti:"{kw}" OR abs:"{kw}"')
        else:
            kw_parts.append(f"ti:{kw} OR abs:{kw}")

    kw_query = " OR ".join(f"({p})" for p in kw_parts)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    date_range = f"[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"

    query = f"({kw_query}) AND submittedDate:{date_range}"
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": min(limit, 200),
    }

    with httpx.Client() as client:
        xml_text = _request_with_retry(client, ARXIV_API, params)
    return _parse_entries(xml_text)


def format_paper(p: dict, idx: int) -> str:
    title = p["title"]
    arxiv_id = p["arxiv_id"]
    authors = ", ".join(p["authors"][:3])
    if len(p["authors"]) > 3:
        authors += " et al."
    cats = ", ".join(p["categories"][:3])
    published = p["published"][:10] if p["published"] else "?"
    summary = p["summary"][:200]
    if len(p["summary"]) > 200:
        summary = summary.rsplit(" ", 1)[0] + "…"

    comment = f"\n  📝 {p['comment']}" if p["comment"] else ""

    return (f"{idx}. **{title}**\n"
            f"  {authors} | {published} | [{cats}]\n"
            f"  arXiv:`{arxiv_id}` | [PDF]({p['pdf_url']}){comment}\n"
            f"  > {summary}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor arXiv for new papers")
    parser.add_argument("--categories", "-c", help="Comma-separated arXiv categories (e.g. cs.CL,cs.AI)")
    parser.add_argument("--keywords", "-k", help="Comma-separated keywords to search")
    parser.add_argument("--days", "-d", type=int, default=3, help="Look back N days (default 3)")
    parser.add_argument("--limit", "-l", type=int, default=30, help="Max results (default 30)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.categories and not args.keywords:
        print("Error: --categories or --keywords required", file=sys.stderr)
        sys.exit(1)

    papers = []
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        papers = fetch_by_categories(cats, args.days, args.limit)
    else:
        kws = [k.strip() for k in args.keywords.split(",")]
        papers = fetch_by_keywords(kws, args.days, args.limit)

    if not papers:
        print("No new papers found.", file=sys.stderr)
        sys.exit(0)

    if args.json:
        print(json.dumps(papers, indent=2))
        return

    mode = f"categories: {args.categories}" if args.categories else f"keywords: {args.keywords}"
    print(f"# arXiv Monitor: {mode}\n")
    print(f"Last **{args.days}** days | Found **{len(papers)}** papers\n")
    for i, p in enumerate(papers, 1):
        print(format_paper(p, i))


if __name__ == "__main__":
    main()
