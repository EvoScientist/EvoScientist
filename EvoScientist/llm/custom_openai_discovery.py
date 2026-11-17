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


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


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
    twice in a picker.
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
    """
    if not base_url:
        return []

    try:
        import httpx

        resp = httpx.get(
            _models_url(base_url),
            headers=_auth_headers(api_key),
            timeout=timeout,
        )
        if resp.status_code != 200:
            return []
        return _names_from_body(resp.json())
    except Exception:
        return []


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
    """
    if not base_url:
        return []

    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                _models_url(base_url),
                headers=_auth_headers(api_key),
            )
        if resp.status_code != 200:
            return []
        return _names_from_body(resp.json())
    except Exception:
        return []
