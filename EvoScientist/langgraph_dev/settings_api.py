"""The settings API behind ``/api/settings``: read and write ``config.yaml``.

Kept apart from the Starlette routes in :mod:`.http` so it can be tested
without a server. Every function here does blocking file and network I/O;
the routes run them in a thread.

- :func:`read_settings` builds the ``GET`` body. Secret values are never
  returned: each reads as ``{"set": ..., "hint": ...}``.
- :func:`patch_settings` validates a ``PATCH`` body, checks changed API keys
  online (except the custom endpoints'), and writes through
  :func:`~EvoScientist.config.settings.set_config_values`.
- :func:`check_key_request` runs one key check without saving anything. A
  stored key is only sent to its stored base URL.
- :func:`capture_restart_baseline` records the file as the server read it
  when it built its graphs; :func:`pending_restart` names the fields that
  changed since then.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from dataclasses import fields
from typing import Any
from urllib.parse import urlparse

from EvoScientist import local_secret
from EvoScientist.config import store
from EvoScientist.config.provider_keys import (
    KEY_SPECS,
    PROVIDERS,
    KeyCheck,
    KeySpec,
    configured_base_url,
    run_check,
)
from EvoScientist.config.settings import (
    _ENV_MAPPINGS,
    API_KEY_FIELDS,
    SECRET_FIELDS,
    EvoScientistConfig,
    SettingsError,
    _config_field_type,
    _config_to_dict,
    config_from_mapping,
    env_origin,
    get_config_dir,
    get_effective_config,
    load_config,
    secret_hint,
    set_config_values,
    validate_setting,
)

# Fields that decide what the agent may run and who can reach the server.
# The WebUI may be reachable from other machines, so they change only with
# ``EvoSci config set``.
READ_ONLY_FIELDS: frozenset[str] = frozenset(
    {
        "dangerous_mode",
        "auto_approve",
        "auto_mode",
        "shell_allow_list",
        "langgraph_dev_host",
        "webui_host",
        "backend_allowed_origins",
        # Run by the Signal channel (``subprocess.Popen``).
        "signal_cli_path",
        # Who reaches the agent through a channel.
        "dm_policy",
        "require_mention",
        "allowed_channels",
        "discord_allowed_channels",
        "slack_allowed_channels",
        "wechat_personal_dm_policy",
        "wechat_personal_group_policy",
        "wechat_personal_group_allowed",
    }
) | frozenset(
    f.name for f in fields(EvoScientistConfig) if f.name.endswith("_allowed_senders")
)

_READ_ONLY_MESSAGE = "read-only over this API; use EvoSci config set"


def _config_fields() -> list[str]:
    return [f.name for f in fields(EvoScientistConfig)]


# =============================================================================
# Restart baseline
# =============================================================================

_SUBAGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "subagents")
_baseline: dict[str, str] | None = None


def _digest(data: Any) -> str:
    """A short one-way digest; the baseline never holds a secret's value."""
    if not isinstance(data, bytes):
        data = repr(data).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _reaches_server(name: str) -> bool:
    """Whether the server reads ``name`` when it builds its graphs.

    The launcher's drift fingerprint keeps the list of fields that never
    reach the server; ``default_workdir`` is on it, but the server serves
    the workspace it was started with, so a change needs a restart too.
    """
    from EvoScientist.langgraph_dev.manager import (
        _FINGERPRINT_EXCLUDED_FIELDS,
        _FINGERPRINT_EXCLUDED_PREFIXES,
    )

    if name == "default_workdir":
        return True
    return not (
        name in _FINGERPRINT_EXCLUDED_FIELDS
        or name.startswith(_FINGERPRINT_EXCLUDED_PREFIXES)
    )


def _file_digests(mapping: dict[str, Any]) -> dict[str, str]:
    """Per-field digests of the file's values (defaults for what it lacks),
    plus ``mcp.yaml``, the packaged sub-agent specs and the version."""
    values = _config_to_dict(config_from_mapping(mapping))
    digests = {
        name: _digest(value) for name, value in values.items() if _reaches_server(name)
    }
    mcp_yaml = os.path.join(get_config_dir(), "mcp.yaml")
    try:
        with open(mcp_yaml, "rb") as f:
            digests["mcp.yaml"] = _digest(f.read())
    except OSError:
        digests["mcp.yaml"] = _digest(None)
    specs = hashlib.sha256()
    try:
        for name in sorted(os.listdir(_SUBAGENTS_DIR)):
            if name.endswith(".yaml"):
                specs.update(name.encode("utf-8"))
                with open(os.path.join(_SUBAGENTS_DIR, name), "rb") as f:
                    specs.update(f.read())
    except OSError:
        pass
    digests["subagents"] = specs.hexdigest()[:16]
    try:
        version = importlib.metadata.version("EvoScientist")
    except importlib.metadata.PackageNotFoundError:
        version = None
    digests["version"] = _digest(version)
    return digests


def capture_restart_baseline() -> None:
    """Record the file as the server reads it to build its graphs.

    Called before the graphs are built; the first call wins. A file that
    does not parse or cannot be read is recorded as the defaults the server
    then runs with.
    """
    global _baseline
    if _baseline is not None:
        return
    try:
        mapping = store.read_config_mapping()
    except (store.ConfigParseError, OSError):
        mapping = {}
    _baseline = _file_digests(mapping)


def pending_restart(mapping: dict[str, Any], shadowed: set[str]) -> list[str]:
    """Fields whose saved value changed since the server started.

    ``shadowed`` fields are left out: the environment overrides them, so a
    restart would not apply the saved value either.
    """
    capture_restart_baseline()
    assert _baseline is not None
    now = _file_digests(mapping)
    return sorted(
        name
        for name, digest in now.items()
        if digest != _baseline.get(name) and name not in shadowed
    )


# =============================================================================
# Sources
# =============================================================================


def config_sources(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Where each field's value comes from in this server.

    A mapped environment variable wins over the file. When our own code set
    it (see :func:`~EvoScientist.config.settings.record_env_origin`), its
    recorded source is reported: ``file`` and ``default`` are copies, not
    overrides; ``cli``, ``ccproxy`` and ``dotenv`` (with the ``.env`` path)
    override the saved value, so ``shadowed`` is true. A variable our code
    did not set is the user's: ``env``, shadowed. Without a variable, the
    source is ``file`` when the file holds the field, else ``default``.
    """
    sources: dict[str, dict[str, Any]] = {}
    for name in _config_fields():
        env_key = _ENV_MAPPINGS.get(name)
        if env_key and os.environ.get(env_key):
            origin = env_origin(env_key)
            if origin is None:
                sources[name] = {"source": "env", "shadowed": True}
            else:
                source = origin["source"]
                entry: dict[str, Any] = {"source": source}
                if "path" in origin:
                    entry["path"] = origin["path"]
                entry["shadowed"] = source not in ("file", "default")
                sources[name] = entry
        elif mapping.get(name) is not None:
            sources[name] = {"source": "file", "shadowed": False}
        else:
            sources[name] = {"source": "default", "shadowed": False}
    return sources


# =============================================================================
# GET
# =============================================================================


# Scheme (after any leading whitespace, which a saved value keeps), then
# everything up to the last ``@`` (the user and password, which
# people write with a raw ``/``, ``?`` or ``#`` that would end ``urlparse``'s
# netloc early), then the host. An ``@`` in a path is masked too: over-masking
# only changes what ``GET`` shows.
_USERINFO_URL = re.compile(
    r"^(\s*[A-Za-z][A-Za-z0-9+.\-]*://)(.*)@([^@/?#]*)(.*)$", re.DOTALL
)


def _hide_userinfo(value: Any) -> Any:
    """``value`` with the user and password of a URL replaced by ``***``, so
    a proxy such as ``http://<user>:<password>@host:3128`` keeps its password."""
    if not isinstance(value, str):
        return value
    match = _USERINFO_URL.match(value)
    if not match:
        return value
    scheme, _userinfo, host, rest = match.groups()
    return f"{scheme}***@{host}{rest}"


def _masked(name: str, value: Any) -> Any:
    if name not in SECRET_FIELDS:
        return _hide_userinfo(value)
    if not value:
        return {"set": False}
    hint = secret_hint(name, str(value))
    return {"set": True, "hint": hint} if hint else {"set": True}


def setup_needed(config: EvoScientistConfig) -> bool:
    """Whether the selected provider cannot run with this configuration.

    Follows the runtime: a key from the environment or a ``.env`` counts,
    Ollama needs no key, OAuth mode needs none, and only the custom
    endpoints need a base URL (the others have a default address).
    """
    spec = PROVIDERS.get(config.provider)
    if spec is None:
        return True
    if spec.needs_base_url and not getattr(config, spec.base_url_field or "", ""):
        return True
    if not spec.needs_key:
        return False
    if (config.provider == "anthropic" and config.anthropic_auth_mode == "oauth") or (
        config.provider == "openai" and config.openai_auth_mode == "oauth"
    ):
        return False
    return not getattr(config, spec.key_field or "", "")


def _provider_view(spec: KeySpec) -> dict[str, Any]:
    return {
        "label": spec.label,
        "key_field": spec.key_field,
        "base_url_field": spec.base_url_field,
        "needs_key": spec.needs_key,
        "needs_base_url": spec.needs_base_url,
        "has_check": spec.has_check,
        "lists_models": spec.lists_models,
    }


def read_settings() -> dict[str, Any]:
    """The ``GET /api/settings`` body.

    Besides setting names, ``restart_fields`` can name three things that are
    not settings: ``mcp.yaml`` (the MCP server file), ``subagents`` (the
    bundled sub-agent definitions) and ``version`` (the installed package).
    """
    # First: it merges a workspace ``.env`` into the environment, as the
    # graphs' own config reads do, so ``sources`` sees what they see.
    effective = get_effective_config()
    raw = store.read_config_bytes()
    config_error = None
    try:
        mapping = store.parse_config_mapping(raw)
    except store.ConfigParseError as e:
        mapping = {}
        config_error = {
            "error": "config_unreadable",
            "path": str(e.path),
            "detail": e.detail,
        }
    values = _config_to_dict(config_from_mapping(mapping))
    sources = config_sources(mapping)
    shadowed = {name for name, s in sources.items() if s["shadowed"]}
    if config_error is None:
        restart_fields = pending_restart(mapping, shadowed)
        needed = setup_needed(effective)
    else:
        restart_fields = []
        needed = False
    return {
        "settings": {name: _masked(name, value) for name, value in values.items()},
        "sources": sources,
        "read_only": sorted(READ_ONLY_FIELDS),
        "providers": {name: _provider_view(spec) for name, spec in PROVIDERS.items()},
        "setup_needed": needed,
        "restart_required": bool(restart_fields),
        "restart_fields": restart_fields,
        "revision": store.revision_of(raw),
        "config_error": config_error,
        "instance": local_secret.instance_id(),
    }


# =============================================================================
# PATCH
# =============================================================================


def _key_spec_for(field: str, provider: str) -> KeySpec | None:
    """The check for a key field. Some providers share one field with
    different checks (``dashscope`` and ``dashscope-code``), so the provider
    the configuration selects wins when it uses this field."""
    spec = KEY_SPECS.get(provider)
    if spec is not None and spec.key_field == field:
        return spec
    for spec in KEY_SPECS.values():
        if spec.key_field == field:
            return spec
    return None


def _changed_base_url(spec: KeySpec, changes: dict[str, Any], effective: Any) -> str:
    """The base URL the saved configuration will use for ``spec``."""
    if spec.base_url_field in changes:
        # A new value, or ``None`` (removed): then the environment's or the
        # provider's default, not the value about to be removed.
        return (
            str(changes[spec.base_url_field] or "").strip()
            or os.environ.get(spec.base_url_env or "", "").strip()
            or (spec.default_base_url or "")
        )
    return configured_base_url(spec, effective)


def _check_changed_keys(
    changes: dict[str, Any], current: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Check every API key that changes; return ``(errors, warnings)``.

    Runs before the lock is taken, so the lock is never held for a network
    call. An invalid key rejects the write; an inconclusive check (offline,
    timeout) or none available keeps it with a warning. Keys of the custom
    endpoints are not checked here: their check proves the endpoint, not the
    key (``test-key`` runs it on request).
    """
    errors: dict[str, str] = {}
    warnings: dict[str, str] = {}
    effective = None
    for name, value in changes.items():
        if name not in API_KEY_FIELDS or not value or value == current.get(name):
            continue
        if effective is None:
            effective = get_effective_config()
        provider = changes.get("provider") or effective.provider
        spec = _key_spec_for(name, provider)
        if spec is None or spec.lists_models:
            continue
        base_url = _changed_base_url(spec, changes, effective)
        result = run_check(spec, str(value), base_url)
        if result.status == "invalid":
            errors[name] = result.message
        elif result.status in ("inconclusive", "unavailable"):
            warnings[name] = _scrub(result.message, str(value))
    return errors, warnings


_EXPECTED = {
    bool: "expected true or false",
    int: "expected an integer",
    float: "expected a number",
}


def _json_type_error(name: str, value: Any) -> str | None:
    """Why a JSON value other than text does not fit ``name``'s type, or
    ``None``; text is coerced as ``EvoSci config set`` coerces it."""
    field = next((f for f in fields(EvoScientistConfig) if f.name == name), None)
    if field is None or isinstance(value, str):
        return None
    field_type = _config_field_type(name, field.type)
    if field_type is bool:
        ok = isinstance(value, bool)
    elif field_type is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif field_type is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    else:
        ok = False
    return None if ok else _EXPECTED.get(field_type, "expected a string")


def patch_settings(body: Any) -> tuple[int, dict[str, Any]]:
    """Apply a ``PATCH /api/settings`` body; return ``(status, payload)``.

    ``""`` or GET's ``{"set": ...}`` for a secret keeps the stored one, and a
    URL sent back with ``***@`` keeps the stored user and password. A JSON
    value other than text must fit the field's type; text is coerced as
    ``EvoSci config set`` coerces it.
    """
    if not isinstance(body, dict):
        return 400, {"errors": {"body": "expected a JSON object"}}
    body = dict(body)
    revision = body.pop("revision", store.UNSET)
    if (
        revision is not store.UNSET
        and revision is not None
        and not isinstance(revision, str)
    ):
        return 400, {"errors": {"revision": "expected a string or null"}}

    errors: dict[str, str] = {}
    changes: dict[str, Any] = {}
    for name, value in body.items():
        if name in READ_ONLY_FIELDS:
            errors[name] = _READ_ONLY_MESSAGE
        elif name in SECRET_FIELDS and (value == "" or isinstance(value, dict)):
            continue  # "" or GET's {"set": ...} keeps the stored secret
        elif value is None:
            changes[name] = None
        elif reason := _json_type_error(name, value):
            errors[name] = reason
        else:
            try:
                changes[name], _ = validate_setting(name, value)
            except ValueError as e:
                errors[name] = str(e)
    if errors:
        return 400, {"errors": errors}

    try:
        current = store.read_config_mapping()
    except store.ConfigParseError as e:
        return 409, _unreadable(e)
    for name in list(changes):
        # A page that sends back what GET showed (``http://***@host``), edited
        # or not, keeps the stored user and password, as ``""`` keeps a secret.
        stored = current.get(name)
        new = _USERINFO_URL.match(body[name]) if isinstance(body[name], str) else None
        old = _USERINFO_URL.match(stored) if isinstance(stored, str) else None
        if new is None or old is None or new.group(2) != "***":
            continue
        restored = f"{new.group(1)}{old.group(2)}@{new.group(3)}{new.group(4)}"
        if restored == stored:
            del changes[name]
        else:
            body[name] = changes[name] = restored
    if not changes:
        # Nothing to change (``{}``, or only ``""`` secrets): write nothing,
        # so the revision and the file's comments stay as they are.
        return _unchanged(revision)
    key_errors, warnings = _check_changed_keys(changes, current)
    if key_errors:
        return 400, {"errors": key_errors}

    try:
        new_revision, setting_warnings = set_config_values(
            {name: body[name] for name in changes}, expected_revision=revision
        )
    except SettingsError as e:
        return 400, {"errors": e.errors}
    except store.RevisionMismatchError as e:
        return 409, {"error": "revision_mismatch", "revision": e.current}
    except store.ConfigParseError as e:
        return 409, _unreadable(e)
    except store.ConfigLockedError as e:
        return 409, {"error": "config_locked", "detail": str(e)}
    except OSError as e:
        # For example a Windows replace that kept failing while another
        # program had the file open; nothing was written.
        return 500, {"error": "config_write_failed", "detail": str(e)}
    warnings.update(setting_warnings)
    return _patched(new_revision, sorted(changes), warnings)


def _patched(
    revision: str | None, changed: list[str], warnings: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    mapping = store.read_config_mapping()
    shadowed = {n for n, s in config_sources(mapping).items() if s["shadowed"]}
    restart_fields = pending_restart(mapping, shadowed)
    return 200, {
        "revision": revision,
        "changed": changed,
        "restart_required": bool(restart_fields),
        "restart_fields": restart_fields,
        "warnings": warnings,
    }


def _unchanged(expected_revision: Any) -> tuple[int, dict[str, Any]]:
    """The answer to a ``PATCH`` with nothing to write; a stale ``revision``
    is still refused, as for any other ``PATCH``."""
    current = store.revision_of(store.read_config_bytes())
    if expected_revision is not store.UNSET and expected_revision != current:
        return 409, {"error": "revision_mismatch", "revision": current}
    return _patched(current, [], {})


def _unreadable(e: store.ConfigParseError) -> dict[str, Any]:
    return {"error": "config_unreadable", "path": str(e.path), "detail": e.detail}


# =============================================================================
# test-key
# =============================================================================


def _scrub(message: str, key: str) -> str:
    """Never echo the key, even inside an SDK's error text."""
    return message.replace(key, "***") if key else message


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _same_url(a: str, b: str) -> bool:
    return a.strip().rstrip("/") == b.strip().rstrip("/")


def check_key_request(body: Any) -> tuple[int, dict[str, Any]]:
    """Run one key check without saving; return ``(status, payload)``.

    Without ``key``, the stored effective key (``config.yaml``, the
    environment or a ``.env``) is tested, and only against the stored base
    URL. A variable the server copied from the file at start reads as the
    file's current value, so a key saved since then is the one tested.
    Without ``base_url``, the stored base URL or the provider's default is
    used.
    """
    if not isinstance(body, dict):
        return 400, {"errors": {"body": "expected a JSON object"}}
    name = body.get("provider")
    spec = KEY_SPECS.get(name) if isinstance(name, str) else None
    if spec is None:
        return 400, {"errors": {"provider": "unknown provider"}}

    base_url = body.get("base_url")
    if base_url is not None:
        if spec.base_url_field is None:
            return 400, {"errors": {"base_url": "this provider takes no base URL"}}
        if not isinstance(base_url, str) or not _is_http_url(base_url):
            return 400, {"errors": {"base_url": "must be an http or https URL"}}

    key = body.get("key")
    if key is not None and not isinstance(key, str):
        return 400, {"errors": {"key": "expected a string"}}
    effective = None
    if key is None or base_url is None:
        effective = get_effective_config()
        saved = load_config()
        for field, env_key in _ENV_MAPPINGS.items():
            origin = env_origin(env_key)
            if origin is not None and origin["source"] in ("file", "default"):
                setattr(effective, field, getattr(saved, field))
    if key is None:
        key = (
            str(getattr(effective, spec.key_field, "") or "") if spec.key_field else ""
        )
        if (
            key
            and base_url is not None
            and not _same_url(base_url, configured_base_url(spec, effective))
        ):
            # The stored key goes only to the address it is configured for;
            # a caller trying another address sends the key it means to use.
            return 400, {
                "errors": {"key": "required with a base_url other than the stored one"}
            }
    if base_url is None:
        base_url = configured_base_url(spec, effective)

    result: KeyCheck = run_check(spec, key, base_url)
    payload: dict[str, Any] = {
        "ok": result.ok,
        "status": result.status,
        "message": _scrub(result.message, key),
    }
    if spec.lists_models and result.status == "valid":
        payload["models"] = list(result.models)
    return 200, payload
