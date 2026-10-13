"""``EvoSci update``: move an installation to another release together with
that release's ``constraints.txt``.

A uv tool install (the install scripts, or ``uv tool install`` by hand) is
reinstalled with ``uv tool install --force``, keeping what its receipt records:
the extras, the Python version and every ``--with`` requirement. The new
version's ``EvoSci setup`` runs afterwards. Every other kind of install is
left alone, and the command prints what to run instead: pip and conda
environments are often shared, an editable install updates with git, the
Docker image updates by pulling a new image, and a uv tool installed from a
directory, git or a URL is reinstalled from that source.

Every run reports through the setup event protocol (stage ``update``) and
ends with exactly one terminal event for that stage; with ``--json`` the
forwarded ``EvoSci setup`` lines keep their own stage ids.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from . import uv_tool
from .setup.download import _NETWORK_ERRORS
from .setup.protocol import Emitter, StageError, make_event

logger = logging.getLogger(__name__)

STAGE = "update"
# The last release that has no constraints.txt at its tag; it and every older
# release are refused, as in the install scripts.
LAST_UNSUPPORTED = "0.3.5"
RAW_BASE = "https://raw.githubusercontent.com/EvoScientist/EvoScientist"
CN_RAW_BASE = "https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist"
CN_PYTHON_MIRROR = "https://registry.npmmirror.com/-/binary/python-build-standalone/"
INSTALL_COMMAND = (
    "curl -fsSL https://raw.githubusercontent.com/EvoScientist/EvoScientist/"
    "main/scripts/install.sh | bash"
)
INSTALL_COMMAND_WINDOWS = (
    "irm https://raw.githubusercontent.com/EvoScientist/EvoScientist/"
    "main/scripts/install.ps1 | iex"
)
# The Docker image's /opt/venv: built by uv sync, without pip or a receipt.
DOCKER_MARKER = ".evoscientist-managed"
_LOOKUP_TIMEOUT = 30


class UpdateError(Exception):
    """Ends the update with an ``error`` event; *code* is one of
    ``ERROR_CODES``."""

    def __init__(
        self, code: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        StageError(code, message)  # rejects a code missing from ERROR_CODES
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


@dataclass
class UpdateOptions:
    version: str | None = None
    check: bool = False
    yes: bool = False
    json: bool = False
    skip: list[str] = field(default_factory=list)


# A question for the user (True = yes), or None where nothing may prompt.
Confirm = Callable[[str], bool] | None


def install_command() -> str:
    return INSTALL_COMMAND_WINDOWS if sys.platform == "win32" else INSTALL_COMMAND


def installed_version() -> str:
    from importlib.metadata import version

    return version("EvoScientist")


def _version(text: str):
    from packaging.version import InvalidVersion, Version

    try:
        return Version(text)
    except InvalidVersion as exc:
        raise UpdateError(
            "unknown_version", f"{text!r} is not a version number."
        ) from exc


# --------------------------------------------------------------------------- #
# Install kind
# --------------------------------------------------------------------------- #
def _manual(message: str) -> UpdateError:
    return UpdateError("manual_update", message)


def read_tool_receipt() -> uv_tool.Receipt | None:
    """The receipt of the uv tool environment EvoScientist runs from; None for
    any other install (pip, conda, the Docker image: their messages need the
    target version). Raises :class:`UpdateError` (``manual_update``) for an
    install this command must not reinstall and that needs no version: an
    unreadable receipt, an editable or source install."""
    from .ccproxy_manager import _is_editable_install

    env = uv_tool.tool_env()
    if env is not None:
        try:
            receipt = uv_tool.read_receipt(env)
        except uv_tool.ReceiptError as exc:
            raise _manual(
                f"{exc}. Run the install script again to reinstall: {install_command()}"
            ) from exc
        if "editable" in receipt.own:
            raise _manual(
                f"EvoScientist is installed in editable mode from "
                f"{receipt.source}: update that checkout with git (git pull)."
            )
        if receipt.source is not None:
            raise _manual(
                f"EvoScientist was installed from {receipt.source}. Reinstall "
                f"it from there, or run the install script to move to the "
                f"PyPI releases: {install_command()}"
            )
        return receipt
    if (Path(sys.prefix) / DOCKER_MARKER).is_file():
        return None  # the Docker image; its message needs the target version
    if _is_editable_install():
        raise _manual(
            "EvoScientist runs from a source checkout (an editable install): "
            "update it with git (git pull), then reinstall its dependencies "
            "(uv sync, or pip install -e .)."
        )
    return None


# --------------------------------------------------------------------------- #
# Version and constraints
# --------------------------------------------------------------------------- #
def lookup_versions(mirror: str) -> list[str]:
    from .update_check import fetch_versions, index_url

    index = index_url(mirror)
    try:
        return fetch_versions(index, _LOOKUP_TIMEOUT)
    except Exception as exc:
        raise UpdateError(
            "download_failed",
            f"Cannot read the EvoScientist versions from {index}: {exc}",
        ) from exc


def choose_target(requested: str | None, versions: list[str]) -> str:
    """The release to install: *requested*, which the index must list, else
    the newest final release. Refuses releases without ``constraints.txt``."""
    from .update_check import newest_release

    if requested:
        wanted = _version(requested.removeprefix("v"))
        listed = [v for v in versions if _safe_version(v) == wanted]
        if not listed:
            raise UpdateError(
                "unknown_version",
                f"The package index does not list EvoScientist {requested}.",
            )
        target = listed[0]
    else:
        target = newest_release(versions)
        if target is None:
            raise UpdateError(
                "unknown_version", "The package index lists no EvoScientist release."
            )
    if _version(target).release <= _version(LAST_UNSUPPORTED).release:
        raise UpdateError(
            "no_constraints",
            f"EvoScientist {target} cannot be installed by EvoSci update: "
            f"{LAST_UNSUPPORTED} is the last release without constraints.txt, "
            f"and every older release lacks it too.",
        )
    return target


def _safe_version(text: str):
    from packaging.version import InvalidVersion, Version

    try:
        return Version(text)
    except InvalidVersion:
        return None


def constraints_url(target: str, mirror: str) -> str:
    if mirror == "cn":
        return f"{CN_RAW_BASE}@v{target}/constraints.txt"
    return f"{RAW_BASE}/v{target}/constraints.txt"


def download_constraints(target: str, mirror: str, dest: Path) -> None:
    url = constraints_url(target, mirror)
    try:
        with urllib.request.urlopen(url, timeout=_LOOKUP_TIMEOUT) as resp:
            dest.write_bytes(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError(
                "no_constraints", f"EvoScientist {target} ships no constraints.txt."
            ) from exc
        raise UpdateError(
            "download_failed", f"Download of {url} failed: {exc}"
        ) from exc
    except _NETWORK_ERRORS as exc:
        raise UpdateError(
            "download_failed", f"Download of {url} failed: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Running server
# --------------------------------------------------------------------------- #
def check_server(opts: UpdateOptions, emit: Emitter, confirm: Confirm) -> None:
    """Stop the langgraph dev server this EvoSci started, after asking; a
    server it has no ownership record for is never touched. Raises
    ``server_running`` when the update must not go on."""
    from .cli.server_cmd import manual_stop_command
    from .config import get_effective_config
    from .deploy.busy import has_busy_thread, running_bg_process_names
    from .langgraph_dev.manager import (
        _DEFAULT_HOST,
        _DEFAULT_PORT,
        _base_url,
        _pid_serves_port,
        _read_workspace_sidecar,
        is_langgraph_dev_running,
        stop_recorded_server,
    )

    config = get_effective_config()
    port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    host = (
        str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or _DEFAULT_HOST)
    ).strip() or _DEFAULT_HOST
    # A stale record whose server does not answer is ignored.
    if not is_langgraph_dev_running(port=port, host=host):
        return
    sidecar = _read_workspace_sidecar()
    if sidecar is not None and _pid_serves_port(sidecar.get("pid"), port):
        url = _base_url(port, host)
        busy = []
        jobs = running_bg_process_names(url)
        if jobs:
            busy.append(f"background jobs still running: {', '.join(jobs)}")
        if has_busy_thread(url):
            busy.append("a task is running")
        question = (
            f"The EvoScientist server on port {port} runs the old version"
            + (f" ({'; '.join(busy)})" if busy else "")
            + ". Stop it and update?"
        )
        if not opts.yes:
            if confirm is None:
                raise UpdateError(
                    "server_running",
                    f"{question} Nothing was changed. Pass --yes to stop it, "
                    f"or run EvoSci server stop first.",
                )
            if not confirm(question):
                raise UpdateError(
                    "server_running", "Update cancelled; the server keeps running."
                )
        pid = stop_recorded_server()
        if pid is None:
            raise UpdateError(
                "server_running",
                f"Could not stop the EvoScientist server on port {port}; nothing "
                f"was changed. Stop it with EvoSci server stop, then run EvoSci "
                f"update again.",
            )
        emit(make_event(STAGE, "running", message=f"Stopped the server (pid {pid})"))
        return
    manual = manual_stop_command(port)
    message = (
        f"A langgraph dev server that EvoSci has no ownership record for answers "
        f"on port {port}, so it was not touched; it keeps running the old "
        f"version. If it is yours, stop it: {manual}"
    )
    if confirm is not None and not opts.yes:
        if confirm(f"{message}\nUpdate anyway?"):
            return
        raise UpdateError("server_running", "Update cancelled.")
    raise UpdateError("server_running", f"{message} Nothing was changed.")


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #
def backup_config(installed: str) -> Path | None:
    """Copy ``config.yaml`` before an older release reads it, which drops
    the keys and values it does not know on its next save."""
    from .config.settings import get_config_path

    config = get_config_path()
    if not config.is_file():
        return None
    backup = config.with_name(f"{config.name}.bak-{installed}")
    try:
        shutil.copy2(config, backup)
    except OSError as exc:
        raise UpdateError(
            "install_failed", f"Could not back up {config} to {backup}: {exc}"
        ) from exc
    return backup


# --------------------------------------------------------------------------- #
# uv
# --------------------------------------------------------------------------- #
def uv_environment(mirror: str) -> dict[str, str]:
    """The environment for uv: the system certificate store unless the user
    chose otherwise, and the mainland China mirrors with ``mirror: cn``."""
    from .update_check import CN_SIMPLE

    env = dict(os.environ)
    env.setdefault("UV_SYSTEM_CERTS", "1")
    if mirror == "cn":
        env["UV_DEFAULT_INDEX"] = CN_SIMPLE
        env["UV_PYTHON_INSTALL_MIRROR"] = CN_PYTHON_MIRROR
    return env


def uv_install_command(
    uv: str, receipt: uv_tool.Receipt, target: str, constraints: Path
) -> list[str]:
    """``uv tool install --force`` of *target*, keeping the receipt's extras,
    Python version and ``--with`` requirements, on uv's own Python.

    Run it from ``constraints.parent``: the file is passed by name, as uv
    splits a ``-c`` path at spaces (astral-sh/uv#12639), such as a temp dir
    under a Windows user name with a space.
    """
    extras = f"[{','.join(receipt.extras)}]" if receipt.extras else ""
    return [
        uv,
        "tool",
        "install",
        "--force",
        "--python",
        uv_tool.python_request(receipt),
        "--managed-python",
        "-c",
        constraints.name,
        f"EvoScientist{extras}=={target}",
        *receipt.with_args(),
    ]


def exclude_newer_hint() -> str:
    """Where an ``exclude-newer`` setting comes from, which keeps uv from
    seeing releases newer than its cutoff; empty when there is none."""
    sources = []
    if os.environ.get("UV_EXCLUDE_NEWER"):
        sources.append("UV_EXCLUDE_NEWER")
    if not os.environ.get("UV_NO_CONFIG"):
        candidates = [os.environ.get("UV_CONFIG_FILE")]
        if sys.platform == "win32":
            candidates += [
                os.path.join(os.environ.get("APPDATA", ""), "uv", "uv.toml"),
                os.path.join(os.environ.get("PROGRAMDATA", ""), "uv", "uv.toml"),
            ]
        else:
            xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
            candidates += [os.path.join(xdg, "uv", "uv.toml"), "/etc/uv/uv.toml"]
        for path in filter(None, candidates):
            try:
                text = Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
            if any(
                line.strip().startswith("exclude-newer") for line in text.splitlines()
            ):
                sources.append(path)
    if not sources:
        return ""
    return (
        f" {' and '.join(sources)} sets exclude-newer: uv ignores releases "
        f"uploaded after its cutoff."
    )


def tool_bin_dir(uv: str) -> Path:
    try:
        # uv writes UTF-8; the locale's code page (Windows) would garble a
        # path with non-ASCII characters.
        result = subprocess.run(
            [uv, "tool", "dir", "--bin"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpdateError("install_failed", f"uv tool dir --bin failed: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise UpdateError(
            "install_failed", f"uv tool dir --bin failed: {result.stderr.strip()}"
        )
    return Path(result.stdout.strip())


# --------------------------------------------------------------------------- #
# The new version's EvoSci setup
# --------------------------------------------------------------------------- #
def run_new_setup(
    evosci: Path, opts: UpdateOptions, emit_line: Callable[[str], None]
) -> None:
    """Run ``EvoSci setup`` of the new version. With ``--json`` its lines are
    forwarded unchanged; a stage that ended with ``error`` raises
    ``setup_failed`` naming the stage and its code."""
    cmd = [str(evosci), "setup"]
    for stage in opts.skip:
        cmd += ["--skip", stage]
    if not opts.json:
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise UpdateError(
                "setup_failed",
                f"EvoScientist is updated, but EvoSci setup failed (see above). "
                f"Run it again: {evosci} setup",
            )
        return
    cmd.append("--json")
    failed: dict[str, Any] | None = None
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            try:
                event = json.loads(line)
            except ValueError:
                # Not an event line: keep stdout for JSON only.
                print(line, file=sys.stderr)
                continue
            emit_line(line)
            if (
                isinstance(event, dict)
                and event.get("status") == "error"
                and failed is None
            ):
                failed = {"stage": event.get("stage"), "code": event.get("code")}
        rc = proc.wait()
    if failed is not None or rc != 0:
        stage = failed["stage"] if failed else "setup"
        raise UpdateError(
            "setup_failed",
            f"EvoScientist is updated, but the {stage} stage of EvoSci setup "
            f"failed. Run it again: {evosci} setup",
            detail=failed or {"stage": None, "code": None, "exit_code": rc},
        )


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def run_update(
    opts: UpdateOptions,
    emit: Emitter,
    *,
    confirm: Confirm,
    emit_line: Callable[[str], None] | None = None,
) -> int:
    """Run ``EvoSci update`` and return its exit code. Ends with exactly one
    terminal event for stage ``update``. *emit_line* writes a forwarded
    ``EvoSci setup --json`` line as it came."""

    def write_line(line: str) -> None:
        if emit_line is not None:
            emit_line(line)
        else:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    try:
        message, detail = _update(opts, emit, confirm, write_line)
    except HandedOver:
        return 0
    except click.exceptions.Abort:  # Ctrl+C at the terminal question
        emit(
            make_event(
                STAGE, "error", message="Update cancelled.", code="server_running"
            )
        )
        return 1
    except UpdateError as exc:
        emit(
            make_event(
                STAGE, "error", message=exc.message, detail=exc.detail, code=exc.code
            )
        )
        return 1
    except Exception as exc:  # every run ends with a terminal event
        logger.debug("EvoSci update failed", exc_info=True)
        emit(make_event(STAGE, "error", message=str(exc), code="install_failed"))
        return 1
    emit(make_event(STAGE, "done", message=message, detail=detail))
    return 0


def _update(
    opts: UpdateOptions,
    emit: Emitter,
    confirm: Confirm,
    write_line: Callable[[str], None],
) -> tuple[str, dict[str, Any]]:
    from .config import load_config

    installed = installed_version()
    mirror = load_config().mirror
    emit(make_event(STAGE, "running", progress=0.0, message="Checking for releases"))
    versions = lookup_versions(mirror)

    if opts.check:
        from .update_check import newest_release

        latest = newest_release(versions)
        return (
            f"Installed: {installed}; newest release: {latest or 'unknown'}",
            {"version": installed, "latest": latest},
        )

    receipt = read_tool_receipt()
    target = choose_target(opts.version, versions)
    if not opts.version and _version(target) <= _version(installed):
        return (
            f"EvoScientist {installed} is the newest release.",
            {"version": installed},
        )

    if receipt is None:
        url = constraints_url(target, mirror)
        if (Path(sys.prefix) / DOCKER_MARKER).is_file():
            raise _manual(
                f"This is the EvoScientist Docker image: pull the new image "
                f"(docker pull ghcr.io/evoscientist/evoscientist:{target}, or "
                f"docker compose pull) instead."
            )
        raise _manual(
            f"EvoScientist was installed with pip or conda; EvoSci update does "
            f"not change such an environment. Run: pip install --upgrade "
            f'"EvoScientist=={target}". In an environment used only by '
            f"EvoScientist, add the release's tested dependencies: pip install "
            f'--upgrade "EvoScientist=={target}" -c {url}'
        )

    uv = uv_tool.find_uv()
    if uv is None:
        raise _manual(
            "No uv was found (neither the one the install script recorded nor "
            "uv on PATH). Run the install script to update: "
            f"{install_command()}"
        )

    tmp = Path(tempfile.mkdtemp(prefix="evosci-update-"))
    handed_over = False
    try:
        constraints = tmp / "constraints.txt"
        emit(
            make_event(
                STAGE,
                "running",
                progress=0.1,
                message=f"Downloading the constraints of EvoScientist {target}",
            )
        )
        download_constraints(target, mirror, constraints)
        check_server(opts, emit, confirm)
        if _version(target) < _version(installed):
            backup = backup_config(installed)
            if backup is not None:
                emit(
                    make_event(
                        STAGE,
                        "running",
                        progress=0.2,
                        message=f"Copied the config to {backup} before the rollback",
                    )
                )
        cmd = uv_install_command(uv, receipt, target, constraints)
        evosci = tool_bin_dir(uv) / (
            "EvoSci.exe" if sys.platform == "win32" else "EvoSci"
        )
        if sys.platform == "win32":
            _hand_over(opts, emit, tmp, cmd, mirror, evosci, installed, target)
            handed_over = True
            raise HandedOver
        emit(
            make_event(
                STAGE,
                "running",
                progress=0.3,
                message=f"Installing EvoScientist {target}",
            )
        )
        # uv's own output goes to stderr under --json: stdout is for events.
        result = subprocess.run(
            cmd,
            cwd=tmp,
            env=uv_environment(mirror),
            stdout=sys.stderr if opts.json else None,
        )
        if result.returncode != 0:
            raise UpdateError(
                "install_failed",
                f"uv could not install EvoScientist {target} (see uv's output); "
                f"the installed version is unchanged.{exclude_newer_hint()}",
            )
    finally:
        if not handed_over:  # the helper deletes it when it ends
            shutil.rmtree(tmp, ignore_errors=True)

    emit(make_event(STAGE, "running", progress=0.8, message="Running EvoSci setup"))
    run_new_setup(evosci, opts, write_line)
    return f"Updated EvoScientist {installed} -> {target}", {"version": target}


class HandedOver(Exception):
    """The Windows helper took over and writes the terminal line itself."""


def _hand_over(
    opts: UpdateOptions,
    emit: Emitter,
    tmp: Path,
    cmd: list[str],
    mirror: str,
    evosci: Path,
    installed: str,
    target: str,
) -> None:
    from .update_helper import start_helper

    base = os.environ
    env = uv_environment(mirror)
    emit(
        make_event(
            STAGE,
            "running",
            progress=0.3,
            message=(
                "The install continues in a helper once EvoScientist has "
                "exited; wait for its last line"
                if opts.json
                else "The install continues in a new window once EvoScientist "
                "has exited"
            ),
        )
    )
    start_helper(
        tmp,
        {
            # Expanded, as the helper compares it with process paths.
            "env_dir": os.path.realpath(sys.prefix),
            "uv": cmd[0],
            "uv_args": cmd[1:],
            "uv_env": {k: v for k, v in env.items() if base.get(k) != v},
            "evosci": str(evosci),
            "skip": opts.skip,
            "installed": installed,
            "target": target,
            "install_command": install_command(),
        },
        json_mode=opts.json,
    )
