"""Reading and writing ``config.yaml``: one lock, strict reads, atomic writes.

Every writer of the config file goes through this module:

- :func:`config_lock` is one cross-process lock for the file. It is
  re-entrant on the same thread, so ``EvoSci onboard`` can hold it for the
  whole wizard while its own saves take it again; any other writer waits
  about a second and then gets :class:`ConfigLockedError`.
- :func:`read_config_mapping` refuses a file that exists but does not parse
  (:class:`ConfigParseError`) instead of handing back defaults, so a
  read-modify-write never replaces the user's configuration with defaults.
- :func:`write_config_bytes` writes a temporary file and ``os.replace``-s it
  over the target, so a reader never sees a truncated file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock, Timeout

_LOCK_TIMEOUT = 1.0
_REPLACE_ATTEMPTS = 5
_REPLACE_DELAY = 0.2


class ConfigError(RuntimeError):
    """The config file cannot be read or written right now."""


class ConfigParseError(ConfigError):
    """``config.yaml`` exists but is not a YAML mapping.

    Carries the line and column only, never the parser's message: PyYAML
    quotes part of the failing line, which could hold a key.
    """

    def __init__(self, path: Path, line: int | None, column: int | None) -> None:
        self.path = path
        self.line = line
        self.column = column
        where = f" (line {line}, column {column})" if line is not None else ""
        super().__init__(
            f"{path} does not parse{where}. Fix it, or run `EvoSci config reset` "
            "to start over with defaults."
        )

    @property
    def detail(self) -> str:
        if self.line is None:
            return "not a YAML mapping"
        return f"line {self.line}, column {self.column}"


class ConfigLockedError(ConfigError):
    """Another process holds the config lock (usually ``EvoSci onboard``)."""

    def __init__(self) -> None:
        super().__init__(
            "The configuration is being edited by another EvoScientist process, "
            "for example `EvoSci onboard` running in a terminal. Try again when "
            "it has finished."
        )


class RevisionMismatchError(ConfigError):
    """The file changed since the caller read it."""

    def __init__(self, current: str | None) -> None:
        self.current = current
        super().__init__("The configuration changed since it was read.")


def config_path() -> Path:
    """The config file path (looked up on every call, so tests can move it)."""
    from . import settings

    return settings.get_config_path()


def _ensure_config_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass


@contextmanager
def config_lock(timeout: float = _LOCK_TIMEOUT) -> Iterator[None]:
    """Hold the config lock; :class:`ConfigLockedError` after ``timeout`` s.

    One shared lock object per path (``is_singleton``): a second ``FileLock``
    on the same path in the same thread would raise filelock's deadlock
    error instead of re-entering.
    """
    path = config_path()
    _ensure_config_dir(path)
    lock = FileLock(str(path.with_name(path.name + ".lock")), is_singleton=True)
    try:
        lock.acquire(timeout=timeout)
    except Timeout:
        raise ConfigLockedError() from None
    try:
        yield
    finally:
        lock.release()


def read_config_bytes() -> bytes | None:
    """The file's bytes, or ``None`` when there is no file."""
    try:
        return config_path().read_bytes()
    except FileNotFoundError:
        return None


def revision_of(data: bytes | None) -> str | None:
    """SHA-256 of the file's bytes; ``None`` when there is no file."""
    return None if data is None else hashlib.sha256(data).hexdigest()


def parse_config_mapping(data: bytes | None) -> dict[str, Any]:
    """Parse the file's bytes; :class:`ConfigParseError` if not a mapping.

    No file, an empty file and a comments-only file are an empty mapping.
    """
    if data is None:
        return {}
    try:
        loaded = yaml.safe_load(data.decode("utf-8"))
    except yaml.MarkedYAMLError as e:
        mark = e.problem_mark or e.context_mark
        raise ConfigParseError(
            config_path(),
            mark.line + 1 if mark else None,
            mark.column + 1 if mark else None,
        ) from None
    except (yaml.YAMLError, UnicodeDecodeError):
        raise ConfigParseError(config_path(), None, None) from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigParseError(config_path(), None, None)
    return loaded


def read_config_mapping() -> dict[str, Any]:
    """The file as a mapping; :class:`ConfigParseError` if it does not parse."""
    return parse_config_mapping(read_config_bytes())


def dump_config_mapping(data: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(data),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def _replace(src: str, dst: str) -> None:
    # On Windows an antivirus scanner or another process that has the file
    # open makes os.replace fail for a moment; retry briefly.
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY)


def write_config_bytes(data: bytes) -> None:
    """Replace the file's content atomically with mode ``0600``.

    A symlinked ``config.yaml`` keeps its link: the target is replaced.
    """
    path = config_path()
    _ensure_config_dir(path)
    target = Path(os.path.realpath(path))
    # mkstemp creates the file with mode 0600, so secrets are never readable
    # by others, not even before the replace.
    fd, tmp = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        target.chmod(0o600)
    except OSError:
        pass


def remove_config_file() -> None:
    """Delete the file if it exists (a symlink is removed, not its target)."""
    try:
        config_path().unlink()
    except FileNotFoundError:
        pass


UNSET: Any = object()
"""Default of ``expected_revision``: skip the revision check."""


def update_config_mapping(
    changes: Mapping[str, Any], *, expected_revision: Any = UNSET
) -> str | None:
    """Apply ``changes`` to the file under the lock; return the new revision.

    A ``None`` value removes the key, so its default applies. Keys not in
    ``changes`` keep their bytes' values, including unknown keys. With
    ``expected_revision`` (``None`` = "no file expected"), a file that changed
    since then raises :class:`RevisionMismatchError` and nothing is written.
    """
    with config_lock():
        raw = read_config_bytes()
        if expected_revision is not UNSET and revision_of(raw) != expected_revision:
            raise RevisionMismatchError(revision_of(raw))
        mapping = parse_config_mapping(raw)
        for key, value in changes.items():
            if value is None:
                mapping.pop(key, None)
            else:
                mapping[key] = value
        data = dump_config_mapping(mapping)
        write_config_bytes(data)
        return revision_of(data)
