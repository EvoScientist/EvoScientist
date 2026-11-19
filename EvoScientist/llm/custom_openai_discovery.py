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

import threading

# A picker mounts one row per model, so a huge listing would make it unusable.
_MAX_MODELS = 500


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _names_from_body(body: object) -> list[str]:
    """Model ids from an OpenAI ``/models`` body: stripped, unique, capped."""
    from ..config.provider_keys import _listed_models

    listed = _listed_models(body)
    if not listed:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for raw in listed:
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= _MAX_MODELS:
            break
    return names


def list_custom_openai_models(
    base_url: str | None,
    api_key: str | None = None,
    *,
    timeout: float = 5.0,
) -> list[str]:
    """The endpoint's model ids, or ``[]`` on any failure.

    Never raises: the wizard must still open when the endpoint is down. The
    call has a TOTAL deadline, not just HTTPX's inactivity timeout, because
    HTTPX's timeout only limits the wait between chunks.

    The probe runs in a daemon thread, because the wizard is blocking and
    the request must not. That thread runs the async probe, whose deadline
    CANCELS the request, so the thread ends with the request.
    """
    if not base_url:
        return []

    names: list[str] = []

    def fetch() -> None:
        import asyncio

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
        # The probe's own deadline cancels the request, so the thread ends
        # when the cancellation lands. This is the grace for that -- and a
        # listing that arrives inside it is the answer.
        worker.join(0.5)
    return names


async def discover_custom_openai_models(
    base_url: str | None,
    api_key: str | None = None,
    *,
    timeout: float = 1.5,
) -> list[str]:
    """The endpoint's model ids, or ``[]`` on any failure.

    Never raises: the picker must always open, even when the endpoint is
    down. An empty ``base_url`` returns ``[]`` without an HTTP call. The
    timeout is a TOTAL deadline, so a trickling endpoint cannot keep the
    picker waiting.
    """
    if not base_url:
        return []

    try:
        import asyncio

        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await asyncio.wait_for(
                client.get(
                    base_url.rstrip("/") + "/models",
                    headers=_auth_headers(api_key),
                ),
                timeout=timeout,
            )
        if resp.status_code != 200:
            return []
        return _names_from_body(resp.json())
    except Exception:
        return []
