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

import json
import threading

import httpx

# A picker mounts one row per model, so a huge listing would make it unusable.
_MAX_MODELS = 500

# The body is parsed and walked before the names are counted, and that work
# has to fit the deadline. A picker shows at most _MAX_MODELS rows, so a
# body past this carries ids no picker would show: it is dropped rather
# than parsed.
_MAX_BODY_BYTES = 2 * 1024 * 1024


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

    The work after the response -- parsing the body and taking the names
    out of it -- runs in a thread of its own, and the wait for it is what
    the rest of the deadline bounds. That thread may outlive the call when
    the body is too big to parse in time; it finishes on its own and its
    result is dropped.
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
    # A copy: the worker may still be filling its own buffer, and a caller
    # that kept the list would see it fill itself in after the return.
    return list(names)


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

        # One deadline for the whole call: the download AND the parse. A
        # listing of a million ids is 20 MB, and parsing that takes longer
        # than the download -- so a deadline that covered only the request
        # would let the picker wait on the parse.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await asyncio.wait_for(
                client.get(
                    base_url.rstrip("/") + "/models",
                    headers=_auth_headers(api_key),
                ),
                timeout=max(0.0, deadline - loop.time()),
            )
        if resp.status_code != 200:
            return []
        # A body past the cap is not a picker: it is not parsed at all.
        if len(resp.content) > _MAX_BODY_BYTES:
            return []

        # The parse and the names out of it are the other half of the
        # deadline: walking a near-cap body costs as much as parsing it.
        remaining = deadline - loop.time()
        if remaining <= 0:
            return []
        return await _names_within(resp.content, remaining)
    except Exception:
        return []


async def _names_within(content: bytes, timeout: float) -> list[str]:
    """The names in *content*, parsed off the event loop within *timeout*.

    A parse cannot be interrupted, so what the timeout bounds is the WAIT for
    it: a parse that does not fit is dropped and its result thrown away.
    Waiting for a cancelled parse instead would let the body decide how long
    the picker stays open.

    The parsing thread is a daemon of its own rather than the loop's
    executor, so a loop that closes while the parse runs does not wait for
    it -- an executor thread would hold the caller until the parse ended.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    finished = asyncio.Event()
    parsed: list[list[str]] = []

    def parse() -> None:
        try:
            names = _names_from_body(json.loads(content))
        except Exception:
            names = []
        parsed.append(names)
        try:
            loop.call_soon_threadsafe(finished.set)
        except RuntimeError:
            # The loop is closed; the parse's result goes with it.
            pass

    threading.Thread(target=parse, daemon=True).start()
    try:
        await asyncio.wait_for(finished.wait(), timeout=timeout)
    except TimeoutError:
        return []
    return list(parsed[0]) if parsed else []
