"""Shared internal helpers for EvoMemory storage, search, and lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from langgraph.config import get_config

# --- memory document parsing -------------------------------------------------


def parse_memory_document(text: str) -> tuple[dict[str, object], str] | None:
    """Split memory markdown text into (frontmatter metadata, body).

    Returns ``None`` when the text lacks a leading ``---`` YAML frontmatter
    block or the frontmatter does not parse to a mapping.
    """
    if not text.startswith("---\n"):
        return None
    try:
        frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
        metadata = yaml.safe_load(frontmatter)
    except (ValueError, yaml.YAMLError):
        return None
    if not isinstance(metadata, dict):
        return None
    return {key: value for key, value in metadata.items() if isinstance(key, str)}, body


def read_memory_document(path: Path) -> tuple[dict[str, object], str] | None:
    """Read a memory markdown file and parse it into (metadata, body).

    Returns ``None`` when the file is missing, undecodable, or has no valid
    frontmatter block.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return parse_memory_document(text)


def document_body(text: str) -> str:
    """Return a memory document's markdown body without its YAML frontmatter."""
    if not text.startswith("---\n"):
        return text.strip()
    try:
        _frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
    except ValueError:
        return text.strip()
    return body.strip()


# --- LangGraph runtime config access -----------------------------------------


def current_configurable() -> Mapping[str, object]:
    """Return the active LangGraph run's ``configurable`` mapping, or empty."""
    try:
        config = get_config()
    except RuntimeError:
        return {}
    configurable = config.get("configurable", {})
    return configurable if isinstance(configurable, dict) else {}


def config_str(configurable: Mapping[str, object], key: str) -> str | None:
    """Return a non-empty string value from a configurable mapping, else None."""
    value = configurable.get(key)
    return value if isinstance(value, str) and value else None


# --- hashing & serialization -------------------------------------------------


def short_hash(text: str, *, n: int = 16) -> str:
    """Return a deterministic short hash fragment for generated ids and paths."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def stable_json(value: object) -> str:
    """Serialize a value deterministically for hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def pretty_json(value: object) -> str:
    """Serialize a value readably for worker prompts."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


# --- id normalization --------------------------------------------------------


def dedupe_ids(
    ids: Iterable[str],
    *,
    require_prefix: str | None = None,
) -> tuple[str, ...]:
    """Strip, drop blanks, and dedupe ids in first-seen order.

    When ``require_prefix`` is set, ids that do not start with it are dropped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw_id in ids:
        clean_id = raw_id.strip()
        if not clean_id or clean_id in seen:
            continue
        if require_prefix is not None and not clean_id.startswith(require_prefix):
            continue
        seen.add(clean_id)
        out.append(clean_id)
    return tuple(out)
