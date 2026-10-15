"""The local secret: a value only this user's own programs can read.

The langgraph dev server keeps it in ``paths.SERVER_HOME / "local-secret"``:
on POSIX the file is ``0600`` in a ``0700`` folder; on Windows its access list
names this user, SYSTEM and Administrators, with nothing inherited (the
Administrators group can read it, as root can on POSIX). Clients on this
machine read the file and send the value in the ``X-EvoScientist-Local-Secret``
header; server routes can require it with :func:`require_local_secret`. It
shows that a caller runs as this user on this machine. It is not
authentication: anyone who can start a run can have the agent read the file.

The value is created once and reused by every server, so two servers running
at the same time accept the same clients. It is never printed. To get a new
value, delete the file while no EvoScientist server runs.
"""

from __future__ import annotations

import contextlib
import functools
import hmac
import logging
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from . import paths

logger = logging.getLogger(__name__)

HEADER = "X-EvoScientist-Local-Secret"
FILE_NAME = "local-secret"

# Windows: an antivirus scanner or the search indexer can hold a new file for a
# moment, so creating it is retried briefly.
_CREATE_ATTEMPTS = 5
_CREATE_RETRY_DELAY = 0.2

# The value this server process loaded at startup (``load_local_secret``).
_value: str | None = None


def load_local_secret() -> None:
    """Create or read the secret and keep it for :func:`require_local_secret`.

    Called once when the server starts. When the file cannot be used, the
    value stays unset and every route that needs it refuses.
    """
    global _value
    try:
        _value = ensure_local_secret()
    except OSError as exc:
        _value = None
        logger.error(
            "Local secret unavailable, so routes that need it refuse every request: %s",
            exc,
        )


def require_local_secret(request: Request) -> JSONResponse | None:
    """Return a ``403`` response unless *request* carries the local secret."""
    sent = request.headers.get(HEADER, "")
    if _value is not None and hmac.compare_digest(sent.encode(), _value.encode()):
        return None
    return JSONResponse({"error": "local_secret_required"}, status_code=403)


def ensure_local_secret() -> str:
    """Return the local secret, creating its file the first time.

    Raises ``OSError`` when the file cannot be used: it belongs to another
    user, it is empty, or it cannot be created or restricted to this user.
    """
    home = paths.SERVER_HOME
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        _keep_mode(home, 0o700)
    path = home / FILE_NAME
    if not path.exists():
        value = _create(path)
        if value is not None:
            return value
    _keep_private(path)
    raw = path.read_bytes().strip()
    if not raw or not raw.isascii():
        raise OSError(
            f"{path} does not hold a local secret; delete it while no "
            "EvoScientist server runs."
        )
    return raw.decode("ascii")


def _create(path: Path) -> str | None:
    """Create *path* with a new value; return None if it already exists.

    The value is written to a temporary file that is then hard-linked into
    place: the link fails if another server created the file first, and no
    reader ever sees a partly written file. Where the file system has no hard
    links, the file is created in place instead.
    """
    value = secrets.token_urlsafe(32)
    tmp = path.with_name(f".{FILE_NAME}.{os.getpid()}.{secrets.token_hex(4)}")
    fd = _open_private(tmp)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(value)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return None
        except OSError:
            return _create_in_place(path, value)
        return value
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            # Windows: a scanner may still hold it; the secret file is in place.
            logger.debug("Could not remove %s", tmp, exc_info=True)


def _create_in_place(path: Path, value: str) -> str | None:
    try:
        fd = _open_private(path)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(value)
    return value


def _open_private(path: Path) -> int:
    """Create *path*, which must not exist, readable by this user only.

    On Windows the access list is set while the file is still empty.
    """
    fd = _open_new(path)
    if sys.platform == "win32":
        try:
            _restrict_to_owner(path)
        except BaseException:
            os.close(fd)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            raise
    return fd


def _open_new(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    attempts = 1
    while True:
        try:
            return os.open(path, flags, 0o600)
        except PermissionError:
            if sys.platform != "win32" or attempts == _CREATE_ATTEMPTS:
                raise
            attempts += 1
            time.sleep(_CREATE_RETRY_DELAY)


def _keep_private(path: Path) -> None:
    """Make an existing secret file readable by this user only."""
    if sys.platform == "win32":
        _restrict_to_owner(path, existing=True)
    elif os.name == "posix":
        _keep_mode(path, 0o600)


def _keep_mode(path: Path, mode: int) -> None:
    """Refuse *path* if another user owns it; restrict it to *mode* if wider."""
    st = os.stat(path)
    if st.st_uid != os.getuid():
        raise PermissionError(f"{path} belongs to another user")
    if st.st_mode & 0o077:
        os.chmod(path, mode)
        if os.stat(path).st_mode & 0o077:
            raise PermissionError(
                f"{path} stays readable by other users: its file system ignores "
                "permissions. Put EVOSCIENTIST_DATA_DIR on one that keeps them."
            )
        logger.warning(
            "%s was readable by other users; it is now restricted to you. "
            "To get a new local secret, delete %s while no EvoScientist "
            "server runs.",
            path,
            paths.SERVER_HOME / FILE_NAME,
        )


def _restrict_to_owner(path: Path, *, existing: bool = False) -> None:
    """Windows: give *path* an access list for this user, SYSTEM and
    Administrators only, with nothing inherited. An OWNER RIGHTS entry can
    remain from the server folder, which Python creates with mode ``0o700`` and
    an access list of SYSTEM, Administrators and OWNER RIGHTS; it names the
    file's owner, so it lets no other account in.

    The grant names this user's SID rather than OWNER RIGHTS: a file created
    from an elevated prompt is owned by Administrators, and OWNER RIGHTS would
    then shut this user's own programs out. Applied to an *existing* file on
    every start, so one others could read is tightened: ``/reset`` first drops
    every explicit entry, since ``/grant:r`` only replaces those of the
    accounts it names. Between the two calls the file carries its folder's
    inherited entries, which only matters for a data folder outside the user
    profile. Fails, and the secret is refused, when this user may not change
    the file's access list (another account created it); the file then keeps
    the access list it had. The folder's own access list is not changed: in a
    data folder outside the profile, others who may write there could replace
    the file.
    """
    if existing:
        _run_system_tool("icacls", str(path), "/reset")
    _run_system_tool(
        "icacls",
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"*{_current_user_sid()}:F",
        "*S-1-5-18:F",  # SYSTEM
        "*S-1-5-32-544:F",  # Administrators
    )


@functools.cache
def _current_user_sid() -> str:
    # One CSV row: "DOMAIN\user","S-1-5-21-..."
    row = _run_system_tool("whoami", "/user", "/fo", "csv", "/nh")
    return row.strip().rsplit(",", 1)[-1].strip('"')


def _run_system_tool(name: str, *args: str) -> str:
    """Run a Windows system tool hidden; ``OSError`` when it fails.

    Found in System32 rather than on PATH, which the desktop app scrubs.
    """
    exe = Path(os.environ.get("SystemRoot", r"C:\Windows"), "System32", f"{name}.exe")
    try:
        result = subprocess.run(
            [str(exe), *args],
            check=True,
            capture_output=True,
            encoding="oem",  # console tools write the OEM code page
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.CalledProcessError as exc:
        raise OSError(f"{name} failed: {(exc.stderr or exc.stdout).strip()}") from exc
    return result.stdout
