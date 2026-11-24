"""Custom OpenAI-compatible endpoint probing — shared by onboard and the model pickers.

A ``custom-openai`` endpoint serves whatever models the user deployed (vLLM,
LM Studio, an internal proxy). Those ids cannot be enumerated in
``_MODEL_ENTRIES``. The setup wizard and every model picker need to hit
``GET {base_url}/models`` to see what is actually there.

``base_url`` already includes the ``/v1`` part, matching
``check_custom_openai``. The key is sent only when the caller has one, so a
keyless local server is probed and ``OPENAI_API_KEY`` is never forwarded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import httpx

logger = logging.getLogger(__name__)

# A picker mounts one row per model, so a huge listing would make it unusable.
_MAX_MODELS = 500

# Leave room for metadata-rich catalogs while bounding the download and parse.
_MAX_BODY_BYTES = 2 * 1024 * 1024


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _names_from_body(body: object) -> list[str] | None:
    """Return unique, stripped IDs in listing order, or None for a non-list response."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return None
    names: list[str] = []
    seen: set[str] = set()
    for item in body["data"]:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        name = str(item["id"]).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _picker_names(body: object) -> list[str]:
    """The IDs a picker shows: the listing's, capped at the row limit."""
    return (_names_from_body(body) or [])[:_MAX_MODELS]


def list_custom_openai_models(
    base_url: str | None,
    api_key: str | None = None,
    *,
    timeout: float = 5.0,
) -> list[str]:
    """Return model IDs or ``[]`` to a synchronous caller.

    A daemon worker runs the async probe, which cancels an overdue download.
    The caller waits at most ``timeout`` plus a short grace period.
    """
    if not base_url:
        return []

    names: list[str] = []

    def fetch() -> None:
        try:
            found = asyncio.run(
                discover_custom_openai_models(base_url, api_key, timeout=timeout)
            )
        except BaseException:
            found = []
        names.extend(found)

    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        # Allow loop startup and request cleanup without waiting indefinitely.
        worker.join(0.5)
    # A worker finishing after the grace period must not change the result.
    return list(names)


async def _download_listing(
    base_url: str, api_key: str | None, timeout: float
) -> bytearray | None:
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "GET", base_url.rstrip("/") + "/models", headers=_auth_headers(api_key)
        ) as resp,
    ):
        if resp.status_code != 200:
            return None
        # The limit is on the body that gets parsed, so it is measured on the
        # decoded bytes: a compressed listing that decodes past the limit is
        # dropped for the same reason an oversized plain body is.
        content = bytearray()
        async for chunk in resp.aiter_bytes():
            if len(content) + len(chunk) > _MAX_BODY_BYTES:
                logger.debug(
                    "Skipping custom OpenAI model list: body exceeds the %d-byte limit",
                    _MAX_BODY_BYTES,
                )
                return None
            content.extend(chunk)
        return content


async def discover_custom_openai_models(
    base_url: str | None,
    api_key: str | None = None,
    *,
    timeout: float = 1.5,
) -> list[str]:
    """Return model IDs, or ``[]`` on failure or a download past the deadline.

    The download has a total timeout, not just HTTPX's inactivity timeout.
    Parsing the size-limited body is synchronous and cannot be interrupted,
    so a listing that finishes parsing after the deadline is still returned:
    it is already known, and dropping it would save nothing.
    """
    if not base_url:
        return []

    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        content = await asyncio.wait_for(
            _download_listing(base_url, api_key, timeout), timeout=timeout
        )
        if content is None or loop.time() >= deadline:
            return []
        return _picker_names(json.loads(content))
    except Exception:
        return []
