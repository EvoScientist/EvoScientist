"""Reading acceptance-checklist text given inline or as ``@path``."""

from __future__ import annotations

from pathlib import Path


class RubricTextError(ValueError):
    """The checklist text could not be read or is empty."""


def read_rubric_text(value: str, *, base_dir: Path | None = None) -> str:
    """Return the checklist text, reading ``@path`` from disk.

    A relative ``@path`` resolves against ``base_dir`` when given, else the
    working directory. The message of a raised error starts with the reason
    ("could not read ..." / "is empty") so callers can prefix their option name.
    """
    text = value
    if value.startswith("@"):
        raw = value[1:]
        path = Path(raw)
        if base_dir is not None and not path.is_absolute() and not raw.startswith("~"):
            path = Path(base_dir) / path
        try:
            # expanduser raises RuntimeError for an unknown "~user".
            text = path.expanduser().read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, RuntimeError) as exc:
            reason = getattr(exc, "strerror", None) or str(exc)
            raise RubricTextError(f"could not read {path}: {reason}") from exc
    text = text.strip()
    if not text:
        raise RubricTextError("is empty")
    return text
