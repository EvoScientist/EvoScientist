#!/usr/bin/env python3
"""Search for datasets via Papers With Code API."""

import argparse
import json
import sys
import time

import httpx

PWC_BASE = "https://paperswithcode.com/api/v1"

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


_UA = {"User-Agent": "EvoScientist/1.0 (paper-navigator)"}


def _request_with_retry(client: httpx.Client, url: str, params: dict | None = None) -> dict:
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


def search_datasets(query: str, limit: int = 10) -> list[dict]:
    """Search PwC datasets."""
    with httpx.Client() as client:
        data = _request_with_retry(client, f"{PWC_BASE}/datasets/", {"q": query})
    results = data.get("results", [])
    return results[:limit]


def format_dataset(d: dict, idx: int) -> str:
    name = d.get("name", "Unknown")
    did = d.get("id", "")
    full_name = d.get("full_name", "")
    num_papers = d.get("num_papers", "?")
    homepage = d.get("homepage", "")
    desc = (d.get("description") or "")[:200]
    if len(d.get("description", "") or "") > 200:
        desc = desc.rsplit(" ", 1)[0] + "..."

    modalities = d.get("modalities", [])
    modality_str = ", ".join(modalities[:3]) if modalities else ""

    lines = [f"{idx}. **{name}**"]
    if full_name and full_name != name:
        lines[0] += f" ({full_name})"
    lines.append(f"   Papers: {num_papers} | ID: `{did}`")
    if modality_str:
        lines.append(f"   Modalities: {modality_str}")
    if homepage:
        lines.append(f"   Homepage: {homepage}")
    if desc:
        lines.append(f"   > {desc}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Search datasets via Papers With Code")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    datasets = search_datasets(args.query, args.limit)

    if not datasets:
        print(f"No datasets found for '{args.query}'", file=sys.stderr)
        sys.exit(0)

    if args.json:
        print(json.dumps(datasets, indent=2))
        return

    print(f"# Datasets: \"{args.query}\"\n")
    print(f"Found **{len(datasets)}** datasets\n")
    for i, d in enumerate(datasets, 1):
        print(format_dataset(d, i))


if __name__ == "__main__":
    main()
