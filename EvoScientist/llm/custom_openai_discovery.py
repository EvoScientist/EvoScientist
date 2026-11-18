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

# A picker mounts one row per model, so a huge listing would make it
# unusable. The cap keeps the list usable and the response bounded.
_MAX_MODELS = 500


def _models_url(base_url: str) -> str:
    """The endpoint's ``/models`` URL, built from the URL's parts."""
    from ..config.custom_endpoints import models_url

    return models_url(base_url)


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _names_from_body(body: object) -> list[str]:
    """Model ids from an OpenAI ``/models`` body, or ``[]`` if it isn't one.

    ``_listed_models`` lives next to the key checker. Importing it here (not
    from ``llm.registry``) keeps the registry free of the config package.

    The ids are stripped and de-duplicated in the order the endpoint listed
    them: a blank id is not a model, and the same model twice would show up
    twice in a picker. The list is capped at ``_MAX_MODELS``: a picker mounts
    one row per model, so an endpoint that serves tens of thousands of ids
    would leave the user with an unusable list.
    """
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
    """Sync probe for the onboarding wizard.

    Returns the endpoint's model ids, or ``[]`` on any failure (no base URL,
    non-200, timeout, connect error, non-JSON, or a body that is not a model
    list). Never raises — the wizard must still open when the endpoint is down.

    The call has a TOTAL deadline, not just HTTPX's inactivity timeout.
    HTTPX's timeout limits the wait for each chunk, so an endpoint that
    sends a chunk every few seconds without ever finishing would keep the
    wizard waiting indefinitely.

    The probe runs in a daemon thread, because the wizard is blocking and
    the request must not. That thread runs the async probe, whose deadline
    CANCELS the request: when the deadline passes the request is aborted
    and the thread ends with it. A probe that only stopped waiting would
    leave the thread and its connection behind, and every further probe
    would leave another one.

    The listing is bounded: an endpoint that serves tens of thousands of ids
    would otherwise fill the picker with one row each and make it unusable.
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
        # when the cancellation lands. This is the grace for that.
        worker.join(0.5)
        return []
    return names


async def discover_custom_openai_models(
    base_url: str | None,
    api_key: str | None = None,
    *,
    timeout: float = 1.5,
) -> list[str]:
    """Async probe for the model pickers.

    Same contract as :func:`list_custom_openai_models`: a list of model ids,
    or ``[]`` on any failure. Never raises — the picker must always open,
    even when the endpoint is down. An empty ``base_url`` returns ``[]``
    without an HTTP call.

    The timeout is a TOTAL deadline, not just an inactivity timeout: an
    endpoint that trickles a byte at a time would otherwise keep the
    connection alive indefinitely and the picker would never open.
    """
    if not base_url:
        return []

    try:
        import asyncio

        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await asyncio.wait_for(
                client.get(
                    _models_url(base_url),
                    headers=_auth_headers(api_key),
                ),
                timeout=timeout,
            )
        if resp.status_code != 200:
            return []
        return _names_from_body(resp.json())
    except Exception:
        return []
