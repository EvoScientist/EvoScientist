"""Ollama server probing — shared by onboard wizard and /model picker.

Ollama models are whatever the user has ``ollama pull``ed locally; they
cannot be enumerated in ``_MODEL_ENTRIES``. Both the setup wizard and the
interactive model picker need to hit ``GET {base_url}/api/tags`` to see
what is actually installed.

This module keeps a single implementation of that probe. ``onboard.py``
uses the sync variant (it drives a synchronous questionary flow);
``/model`` uses the async variant (command dispatch is already ``async``
and benefits from ``httpx.AsyncClient``).
"""

from __future__ import annotations


def validate_ollama_connection(base_url: str) -> tuple[bool, str, list[str]]:
    """Sync probe. Returns ``(is_reachable, human_msg, model_names)``.

    Used by the onboarding wizard. Verbatim semantics from the original
    implementation that lived in ``config/onboard.py`` — 5 s timeout,
    swallows all exceptions into a ``(False, error_msg, [])`` tuple.
    """
    if not base_url:
        return True, "Skipped (no URL provided)", []

    try:
        import httpx

        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            names = [m.get("name", "?") for m in models]
            if names:
                preview = ", ".join(names[:5])
                return True, f"Connected — {len(names)} model(s): {preview}", names
            return True, "Connected (no models pulled yet)", []
        return False, f"HTTP {resp.status_code}", []
    except Exception as e:
        return False, f"Cannot reach Ollama: {e}", []


async def discover_ollama_models(
    base_url: str | None, *, timeout: float = 1.5
) -> list[str]:
    """Async probe for the ``/model`` picker.

    Returns the list of installed model names, or an empty list on any
    failure (unreachable daemon, timeout, HTTP error, malformed JSON).
    Never raises — the picker must always open, even when the daemon
    is down.

    If ``base_url`` is falsy, returns ``[]`` immediately without making
    an HTTP call. Implicit ``localhost:11434`` probing is deliberately
    out of scope (see ``.issue_ollama_model_picker.md`` non-goals);
    ``ollama_base_url`` must be explicitly configured to activate.
    """
    if not base_url:
        return []

    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []
