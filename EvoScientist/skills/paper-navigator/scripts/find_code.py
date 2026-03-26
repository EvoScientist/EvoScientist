#!/usr/bin/env python3
"""Find code implementations for papers via Papers With Code API."""

import argparse
import json
import sys
import time

import httpx

PWC_BASE = "https://paperswithcode.com/api/v1"

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


_UA = {"User-Agent": "EvoScientist/1.0 (paper-navigator)"}


def _request_with_retry(client: httpx.Client, url: str, params: dict | None = None) -> dict | list:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, params=params, headers=_UA, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {}
            raise
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise SystemExit(f"Error: {e}") from e
    return {}


def _find_paper_id(client: httpx.Client, arxiv_id: str | None = None,
                   title: str | None = None) -> str | None:
    """Find PwC paper ID by arXiv ID or title search."""
    if arxiv_id:
        # Clean arXiv ID
        arxiv_id = arxiv_id.strip().replace("ArXiv:", "").replace("arxiv:", "")
        for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/"]:
            if arxiv_id.startswith(prefix):
                arxiv_id = arxiv_id[len(prefix):]

        # Search by arXiv ID
        data = _request_with_retry(client, f"{PWC_BASE}/papers/",
                                   {"arxiv_id": arxiv_id})
        if isinstance(data, dict) and data.get("results"):
            return data["results"][0].get("id")

    if title:
        data = _request_with_retry(client, f"{PWC_BASE}/papers/",
                                   {"q": title})
        if isinstance(data, dict) and data.get("results"):
            return data["results"][0].get("id")

    return None


def find_code(arxiv_id: str | None = None, title: str | None = None,
              limit: int = 5) -> list[dict]:
    """Find code repos for a paper."""
    with httpx.Client() as client:
        paper_id = _find_paper_id(client, arxiv_id, title)
        if not paper_id:
            return []

        data = _request_with_retry(
            client,
            f"{PWC_BASE}/papers/{paper_id}/repositories/",
        )

        repos = data.get("results", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        # Sort by stars
        repos.sort(key=lambda r: r.get("stars", 0), reverse=True)
        return repos[:limit]


def format_repo(r: dict, idx: int) -> str:
    url = r.get("url", "")
    stars = r.get("stars", 0)
    framework = r.get("framework", "unknown")
    is_official = r.get("is_official", False)
    desc = r.get("description", "")[:150]

    official_tag = " 🏷️ **Official**" if is_official else ""

    return (f"{idx}. [{url}]({url}){official_tag}\n"
            f"   ⭐ {stars} | Framework: {framework}\n"
            f"   {desc}\n")


def main():
    parser = argparse.ArgumentParser(description="Find code implementations via Papers With Code")
    parser.add_argument("--arxiv-id", "-a", help="arXiv ID (e.g. 1706.03762)")
    parser.add_argument("--title", "-t", help="Paper title to search")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Max repos (default 5)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.arxiv_id and not args.title:
        print("Error: --arxiv-id or --title required", file=sys.stderr)
        sys.exit(1)

    repos = find_code(args.arxiv_id, args.title, args.limit)

    if not repos:
        query = args.arxiv_id or args.title
        print(f"No code found for '{query}'", file=sys.stderr)
        sys.exit(0)

    if args.json:
        print(json.dumps(repos, indent=2))
        return

    query = args.arxiv_id or args.title
    print(f"# Code Implementations: {query}\n")
    print(f"Found **{len(repos)}** repositories\n")
    for i, r in enumerate(repos, 1):
        print(format_repo(r, i))


if __name__ == "__main__":
    main()
