"""Web search tools.

Provides ``tavily_search`` and ``fetch_webpage_content`` for the research agent,
using Tavily for URL discovery and fetching full webpage content.
"""

import asyncio
from typing import Literal

import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient
from typing_extensions import Annotated

# Lazy initialization - only create client when needed
_tavily_client = None


def _get_tavily_client() -> TavilyClient:
    """Get or create the Tavily client (lazy initialization)."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


async def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns
    full webpage content as markdown for comprehensive research.

    Args:
        query: Search query to execute

    Returns:
        Formatted search results with full webpage content in markdown
    """

    def _sync_search() -> dict:
        return _get_tavily_client().search(
            query,
            max_results=max_results,
            topic=topic,
        )

    try:
        # Run Tavily search asynchronously
        search_results = await asyncio.to_thread(_sync_search)

        # Fetch full content for each URL concurrently
        results = search_results.get("results", [])
        if not results:
            return f"No results found for '{query}'"

        # Fetch all webpages concurrently
        fetch_tasks = [fetch_webpage_content(r["url"]) for r in results]
        contents = await asyncio.gather(*fetch_tasks)

        # Format results
        result_texts = []
        for result, content in zip(results, contents):
            result_text = f"""## {result["title"]}
**URL:** {result["url"]}

{content}

---
"""
            result_texts.append(result_text)

        return f"""Found {len(result_texts)} result(s) for '{query}':

{"".join(result_texts)}"""

    except Exception as e:
        return f"Search failed: {str(e)}"
