"""How a custom endpoint's base URL and its key are paired.

A custom endpoint serves whatever the user deployed, and its key belongs to
one endpoint. The environment and the saved config each name a URL and a key,
and they are resolved independently -- so a merge can end up with the
environment's URL and the saved key, and a probe would send that key to an
endpoint it does not belong to.

The rule: the key that goes out is the one that belongs to the endpoint being
used.

The runtime reads a custom endpoint's URL and key from the environment
directly, so it never mixes the two. The configuration merge does, and this
module is what keeps the merged pair honest.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

#: The config fields that name a custom endpoint and the key that goes with
#: it: ``(base URL field, key field, URL env var, key env var)``.
CUSTOM_ENDPOINT_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "custom_openai_base_url",
        "custom_openai_api_key",
        "CUSTOM_OPENAI_BASE_URL",
        "CUSTOM_OPENAI_API_KEY",
    ),
    (
        "custom_anthropic_base_url",
        "custom_anthropic_api_key",
        "CUSTOM_ANTHROPIC_BASE_URL",
        "CUSTOM_ANTHROPIC_API_KEY",
    ),
)


def same_endpoint(first: str, second: str) -> bool:
    """Whether two base URLs name the same endpoint.

    Only the PATH's trailing slash is ignored: ``/v1`` and ``/v1/`` are the
    same endpoint, but a different host, port, query or fragment is a
    different endpoint. A URL that cannot even be parsed is not an endpoint
    this can compare.
    """
    if not first or not second:
        return False
    try:
        left = urlsplit(first)
        right = urlsplit(second)
    except ValueError:
        # An unmatched bracket raises here. It is not an endpoint to
        # compare, and it must not stop the caller.
        return False
    return (
        left.scheme == right.scheme
        and left.netloc == right.netloc
        and left.path.rstrip("/") == right.path.rstrip("/")
        and left.query == right.query
        and left.fragment == right.fragment
    )


def pair_custom_endpoint_credentials(
    *,
    env_base_url: str,
    env_api_key: str,
    saved_base_url: str,
    saved_api_key: str,
) -> tuple[str, str]:
    """The base URL and the key to use for a custom endpoint.

    - the environment names a **different** endpoint than the saved config:
      the environment's URL and key go together, and the saved key is left
      behind -- it belongs to the saved URL;
    - otherwise the saved config names the endpoint (the environment may name
      the same one): the environment's key wins, the same precedence the
      runtime gives it, so an expired saved key does not shadow a live one.
    """
    env_base_url = env_base_url or ""
    env_api_key = env_api_key or ""
    saved_base_url = saved_base_url or ""
    saved_api_key = saved_api_key or ""
    if env_base_url and not same_endpoint(env_base_url, saved_base_url):
        return env_base_url, env_api_key
    if saved_base_url:
        return saved_base_url, (env_api_key or saved_api_key)
    return env_base_url, env_api_key


def models_url(base_url: str) -> str:
    """The endpoint's ``/models`` URL, built from the URL's parts.

    Appending ``"/models"`` to the string would put it inside a query when
    the base URL carries one: ``http://x/v1?tenant=a`` would become
    ``http://x/v1?tenant=a/models``, whose query is ``tenant=a/models`` -- a
    different endpoint. The path is extended instead, and the query and the
    fragment are kept as they are.
    """
    try:
        parts = urlsplit(base_url)
    except ValueError:
        # A URL that cannot even be parsed cannot be extended either.
        return f"{base_url.rstrip('/')}/models"
    path = f"{parts.path.rstrip('/')}/models"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
