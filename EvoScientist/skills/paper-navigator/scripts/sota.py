#!/usr/bin/env python3
"""Query SOTA leaderboards via Papers With Code API."""

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


def _slugify(text: str) -> str:
    """Convert text to PwC URL slug format."""
    return text.lower().strip().replace(" ", "-").replace("_", "-")


def search_tasks(query: str, limit: int = 5) -> list[dict]:
    """Search for tasks/benchmarks."""
    with httpx.Client() as client:
        data = _request_with_retry(client, f"{PWC_BASE}/tasks/", {"q": query})
    results = data.get("results", [])
    return results[:limit]


def get_task_datasets(task_id: str) -> list[dict]:
    """Get datasets for a task."""
    with httpx.Client() as client:
        data = _request_with_retry(client, f"{PWC_BASE}/tasks/{task_id}/datasets/")
    return data.get("results", []) if isinstance(data, dict) else []


def get_sota(task_id: str, dataset_id: str | None = None, limit: int = 10) -> list[dict]:
    """Get SOTA results for a task (optionally filtered by dataset)."""
    with httpx.Client() as client:
        if dataset_id:
            url = f"{PWC_BASE}/evaluations/"
            data = _request_with_retry(client, url,
                                       {"task": task_id, "dataset": dataset_id})
        else:
            # Get first dataset for the task
            datasets = get_task_datasets(task_id)
            if not datasets:
                return []
            dataset_id = datasets[0].get("id", "")
            url = f"{PWC_BASE}/evaluations/"
            data = _request_with_retry(client, url,
                                       {"task": task_id, "dataset": dataset_id})

    results = data.get("results", []) if isinstance(data, dict) else []
    return results[:limit]


def format_task(t: dict, idx: int) -> str:
    name = t.get("name", "Unknown")
    tid = t.get("id", "")
    desc = (t.get("description") or "")[:150]
    return f"{idx}. **{name}** (`{tid}`)\n   {desc}\n"


def format_evaluation(e: dict, idx: int) -> str:
    method = e.get("method", "Unknown")
    metrics = e.get("metrics", {})
    paper = e.get("paper", "")
    code = e.get("code_links", [])

    metrics_str = " | ".join(f"{k}: **{v}**" for k, v in metrics.items()) if isinstance(metrics, dict) else str(metrics)
    code_str = " 💻" if code else ""

    return f"{idx}. {method} — {metrics_str}{code_str}\n   Paper: {paper}\n"


def main():
    parser = argparse.ArgumentParser(description="Query SOTA leaderboards via Papers With Code")
    parser.add_argument("--task", "-t", required=True, help="Task name to search")
    parser.add_argument("--dataset", "-d", help="Dataset name (optional)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--list-tasks", action="store_true", help="List matching tasks only")
    parser.add_argument("--list-datasets", action="store_true", help="List datasets for the task")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    # Search for tasks
    tasks = search_tasks(args.task, limit=5)
    if not tasks:
        print(f"No tasks found for '{args.task}'", file=sys.stderr)
        sys.exit(0)

    if args.list_tasks:
        if args.json:
            print(json.dumps(tasks, indent=2))
            return
        print(f"# Tasks matching: \"{args.task}\"\n")
        for i, t in enumerate(tasks, 1):
            print(format_task(t, i))
        return

    task_id = tasks[0].get("id", "")
    task_name = tasks[0].get("name", args.task)

    if args.list_datasets:
        datasets = get_task_datasets(task_id)
        if args.json:
            print(json.dumps(datasets, indent=2))
            return
        print(f"# Datasets for: {task_name}\n")
        for i, d in enumerate(datasets[:20], 1):
            print(f"{i}. **{d.get('name', '')}** (`{d.get('id', '')}`)")
        return

    # Get SOTA
    dataset_id = _slugify(args.dataset) if args.dataset else None
    results = get_sota(task_id, dataset_id, args.limit)

    if not results:
        print(f"No SOTA results found for '{task_name}'", file=sys.stderr)
        # Fallback: show available datasets
        datasets = get_task_datasets(task_id)
        if datasets:
            print("\nAvailable datasets:", file=sys.stderr)
            for d in datasets[:5]:
                print(f"  - {d.get('name', '')} ({d.get('id', '')})", file=sys.stderr)
        sys.exit(0)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    ds_label = f" on {args.dataset}" if args.dataset else ""
    print(f"# SOTA: {task_name}{ds_label}\n")
    for i, e in enumerate(results, 1):
        print(format_evaluation(e, i))


if __name__ == "__main__":
    main()
