"""Fail fast when a routed provider has no API key of its own.

Routed providers (e.g. ``minimax`` via ChatAnthropic, ``novita`` via
ChatOpenAI) set a custom ``base_url`` but rely on the underlying client
for authentication.  When the provider's own key env var is empty and the
caller did not pass a non-empty ``api_key=`` explicitly, the underlying
client silently reads ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` and sends
*that* key to the third-party endpoint — a security bug and a confusing 401.

Args:
    provider: Name of the routed provider (e.g. ``"minimax"``).
    api_key_env: Environment variable name for the provider's own key.
    kwargs: Keyword arguments passed through to ``get_chat_model``.
    routed_via: Human-readable name of the underlying client
        (``"Anthropic"`` or ``"OpenAI"``), used in the error message.

Raises:
    ValueError: with a clear message naming the missing env var.
"""
    if os.environ.get(api_key_env, ""):
        return
    if kwargs.get("api_key"):
        return
    raise ValueError(
        f"The '{provider}' provider requires an API key. Set the "
        f"{api_key_env} environment variable, or pass api_key=... explicitly "
        f"to get_chat_model(). Without it, the underlying {routed_via} client "
        f"would silently fall back to the default vendor key and send it to "
        f"the third-party endpoint."
    )
