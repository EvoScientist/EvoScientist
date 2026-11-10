"""Tests for the Node setup stage. Downloads and ``node --version`` are mocked."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from EvoScientist.setup import _install, node
from EvoScientist.setup.protocol import StageError

V = node.NODE_VERSION
# The archive type the host would download, so extraction runs for real.
PLAT = "win-x64" if os.name == "nt" else "linux-x64"
FILENAME = node.archive_name(PLAT)
BASE = f"node-v{V}-{PLAT}"


def _archive_bytes(extra_member: str | None = None) -> bytes:
    buf = io.BytesIO()
    if os.name == "nt":
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"{BASE}/node.exe", b"fake")
            zf.writestr(f"{BASE}/npx.cmd", b"fake")
    else:
        with tarfile.open(fileobj=buf, mode="w:xz") as tf:
            for name in (
                f"{BASE}/bin/node",
                # The target of bin/npx, so shutil.which finds the link.
                f"{BASE}/lib/node_modules/npm/bin/npx-cli.js",
                extra_member,
            ):
                if name is None:
                    continue
                info = tarfile.TarInfo(name)
                info.size = 4
                info.mode = 0o755
                tf.addfile(info, io.BytesIO(b"fake"))
            link = tarfile.TarInfo(f"{BASE}/bin/npx")
            link.type = tarfile.SYMTYPE
            link.linkname = "../lib/node_modules/npm/bin/npx-cli.js"
            tf.addfile(link)
    return buf.getvalue()


class FakeNet:
    """Serves one archive; records requested URLs and the ``--cn`` hint flag."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []
        self.hints: list[bool] = []

    def download(self, url, dest: Path, progress=None, *, cn_hint=False) -> str:
        self.urls.append(url)
        self.hints.append(cn_hint)
        dest.write_bytes(self.data)
        if progress:
            progress(1.0)
        return hashlib.sha256(self.data).hexdigest()


# A probe result: a Windows policy refuses to start the program.
BLOCKED = object()


class ProbeTable(dict):
    """Fake ``node --version`` results keyed by executable path.

    Keys compare with ``os.path.normcase``: on Windows ``shutil.which`` returns
    the ``PATHEXT`` spelling (``node.EXE``) for a file created as ``node.exe``.
    """

    def __setitem__(self, exe, version) -> None:
        super().__setitem__(os.path.normcase(str(exe)), version)

    def get(self, exe, default=None):
        return super().get(os.path.normcase(str(exe)), default)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DATA_DIR with a space and non-ASCII characters, no node on PATH."""
    from EvoScientist import paths

    # Resolved like tools_dir() (on Windows this expands 8.3 short names), so
    # paths compare equal.
    data = tmp_path.resolve() / "Jan Kowalski ąę" / ".evoscientist"
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(node, "platform_id", lambda: PLAT)
    monkeypatch.setattr(node, "_failed_install", None)
    net = FakeNet(_archive_bytes())
    monkeypatch.setattr(node, "download", net.download)
    # Pin the fake archive, as NODE_SHA256 pins the real one.
    monkeypatch.setitem(node.NODE_SHA256, PLAT, hashlib.sha256(net.data).hexdigest())
    probes = ProbeTable()

    def probe(exe):
        # Anything under our tools dir probes as the pinned version by default.
        result = probes.get(exe, tuple(map(int, V.split("."))))
        if result is BLOCKED:
            raise node.PolicyBlocked(exe)
        return result

    monkeypatch.setattr(node, "_probe", probe)
    return {"data": data, "net": net, "probes": probes, "tmp": tmp_path}


def _record(data: Path) -> dict | None:
    path = data / "tools" / "node.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _system_node(
    env, monkeypatch, version: tuple[int, int, int], *, with_npx: bool = True
) -> Path:
    bin_dir = env["tmp"] / "system-bin"
    bin_dir.mkdir()
    names = ["node.exe", "npx.cmd"] if os.name == "nt" else ["node", "npx"]
    for name in names if with_npx else names[:1]:
        (bin_dir / name).write_text("")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    exe = bin_dir / names[0]
    env["probes"][str(exe)] = version
    return exe


def test_system_node_without_npx_triggers_download(env, monkeypatch):
    """Distros that package npm separately ship node without npx; every
    consumer needs npx, so that Node does not count."""
    _system_node(env, monkeypatch, (22, 11, 0), with_npx=False)
    assert node.ensure_node("default").source == "private"
    assert env["net"].urls


def test_stale_temp_dirs_are_removed_before_an_install(env):
    stale = env["data"] / "tools" / ".node-killed"
    stale.mkdir(parents=True)
    (stale / "partial.tar.xz").write_bytes(b"x")
    node.ensure_node("default")
    assert not stale.exists()


def test_log_progress_logs_every_step_and_thins_the_download(caplog):
    report = node.log_progress(node.logging.getLogger("test-node-progress"))
    with caplog.at_level("WARNING", logger="test-node-progress"):
        for i in range(101):
            report(0.05 + 0.8 * i / 100, "Downloading Node")
        report(0.86, "Verifying checksum")
        report(0.9, "Unpacking")
    lines = [r.getMessage() for r in caplog.records]
    assert 5 <= len(lines) <= 12
    assert lines[-2:] == [
        "Installing Node.js: Verifying checksum (86%)",
        "Installing Node.js: Unpacking (90%)",
    ]


def test_unwritable_tools_dir_raises_stage_error(env, monkeypatch):
    from EvoScientist import paths

    blocker = env["tmp"] / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(paths, "DATA_DIR", blocker / ".evoscientist")
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "install_failed"


def _deny_moving_node(monkeypatch, times: int) -> list[int]:
    """Make the move of the unpacked Node dir raise PermissionError ``times``
    times (what a Windows scanner holding node.exe causes), then succeed."""
    real_replace = os.replace
    denied = [0]

    def replace(src, dst):
        if Path(src).name == BASE and denied[0] < times:
            denied[0] += 1
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(_install, "_MOVE_RETRY_DELAYS", (0, 0, 0, 0, 0))
    monkeypatch.setattr(_install.os, "replace", replace)
    return denied


def test_brief_access_denied_on_move_is_retried(env, monkeypatch, caplog):
    denied = _deny_moving_node(monkeypatch, times=2)
    with caplog.at_level("WARNING", logger=_install.__name__):
        info = node.ensure_node("default")
    assert info.source == "private"
    assert denied[0] == 2
    assert _record(env["data"])["version"] == V
    # At least the faked denials: on Windows a real scanner lock can add more.
    match = re.search(r"denied (\d+) time\(s\)", caplog.text)
    assert match is not None
    assert int(match.group(1)) >= 2


def test_lasting_access_denied_on_move_is_install_failed(env, monkeypatch):
    _deny_moving_node(monkeypatch, times=100)
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "install_failed"
    assert _record(env["data"]) is None


def test_disk_error_while_unpacking_is_install_failed(env, monkeypatch):
    def disk_full(archive, dest, tick=None):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(node, "_extract", disk_full)
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "install_failed"
    assert _record(env["data"]) is None


def test_record_with_non_string_path_is_ignored(env):
    tools = env["data"] / "tools"
    tools.mkdir(parents=True)
    (tools / "node.json").write_text(json.dumps({"version": "24.21.0", "path": 1}))
    assert node.activate_runtime() is None
    assert node.ensure_node("default").source == "private"


def test_stage_reports_download_progress_lines(env, monkeypatch):
    from EvoScientist.setup import download as dl

    data = env["net"].data

    def urlopen(url, timeout):
        response = io.BytesIO(data)
        response.headers = {"Content-Length": str(len(data))}
        return response

    monkeypatch.setattr(dl, "_CHUNK", len(data) // 4)
    monkeypatch.setattr(dl, "open_url", urlopen)
    monkeypatch.setattr(node, "download", dl.download)
    events: list[dict] = []
    detail = node.run_stage(events.append, "default").detail
    assert {(e["stage"], e["status"]) for e in events} == {("node", "running")}
    downloading = [
        e["progress"] for e in events if e["message"].startswith("Downloading")
    ]
    assert len(downloading) >= 5
    assert downloading == sorted(downloading)
    assert detail["source"] == "private"


def test_system_node_20_9_or_newer_wins_without_download(env, monkeypatch):
    exe = _system_node(env, monkeypatch, (20, 9, 0))
    info = node.ensure_node("default")
    detail = info.detail()
    assert (detail["source"], detail["version"]) == ("system", "20.9.0")
    # shutil.which may return the PATHEXT spelling (node.EXE) on Windows.
    assert os.path.normcase(detail["path"]) == os.path.normcase(str(exe))
    assert env["net"].urls == []


def test_no_node_downloads_and_records(env):
    info = node.ensure_node("default")
    assert info.source == "private"
    # The pinned hash replaces SHASUMS256.txt: the archive is the only request.
    assert env["net"].urls == [f"https://nodejs.org/dist/v{V}/{FILENAME}"]
    record = _record(env["data"])
    assert record["version"] == V
    assert Path(record["path"]) == env["data"] / "tools" / f"node-v{V}"
    assert info.path.exists()
    # The temp dir is gone; only the install, the record and the lock remain.
    assert not [
        p for p in (env["data"] / "tools").iterdir() if p.name.startswith(".node-")
    ]


@pytest.mark.parametrize("version", [(19, 9, 0), (20, 8, 1)])
def test_old_system_node_triggers_download(env, monkeypatch, version):
    _system_node(env, monkeypatch, version)
    assert node.ensure_node("default").source == "private"
    assert env["net"].urls


def test_second_run_downloads_nothing(env):
    node.ensure_node("default")
    env["net"].urls.clear()
    assert node.ensure_node("default").source == "private"
    assert env["net"].urls == []


def test_cn_mirror_urls(env):
    node.ensure_node("cn")
    assert all(
        u.startswith("https://cdn.npmmirror.com/binaries/node/")
        for u in env["net"].urls
    )


@pytest.mark.parametrize(("mirror", "hint"), [("default", True), ("cn", False)])
def test_download_errors_name_cn_only_when_the_mirror_is_off(env, mirror, hint):
    node.ensure_node(mirror)
    assert env["net"].hints == [hint]


def test_pinned_hashes_are_sha256_for_the_supported_platforms():
    assert set(node.NODE_SHA256) == {
        "darwin-arm64",
        "linux-x64",
        "linux-arm64",
        "win-x64",
    }
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in node.NODE_SHA256.values())


def test_checksum_mismatch_keeps_current_node(env):
    node.ensure_node("default")
    before = _record(env["data"])
    # Break the current install's probe so a new download is attempted.
    env["probes"][str(node._node_exe(Path(before["path"])))] = None
    node.NODE_SHA256[PLAT] = "0" * 64
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "checksum_mismatch"
    assert _record(env["data"]) == before


def test_checksum_mismatch_leaves_no_partial_files(env):
    node.NODE_SHA256[PLAT] = "0" * 64
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "checksum_mismatch"
    assert "pinned" in ei.value.message
    assert _record(env["data"]) is None
    # filelock deletes its lock file on release on Windows only.
    assert {p.name for p in (env["data"] / "tools").iterdir()} <= {"node.lock"}


def test_probe_failure_records_nothing(env):
    final_exe = node._node_exe(env["data"] / "tools" / f"node-v{V}")
    env["probes"][str(final_exe)] = None
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "probe_failed"
    assert _record(env["data"]) is None
    assert not final_exe.exists()


def test_record_is_written_only_after_the_probe(env, monkeypatch):
    seen: list[dict | None] = []

    def probe(exe):
        seen.append(_record(env["data"]))
        return tuple(map(int, V.split(".")))

    monkeypatch.setattr(node, "_probe", probe)
    node.ensure_node("default")
    assert seen[-1] is None
    assert _record(env["data"])["version"] == V


def test_download_failure_keeps_current_node(env, monkeypatch):
    def offline(url, *_a, **_k):
        raise StageError("download_failed", "offline")

    monkeypatch.setattr(node, "download", offline)
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "download_failed"
    assert _record(env["data"]) is None


def test_failed_install_is_not_retried_in_the_same_process(env, monkeypatch):
    """Server spawn, MCP load and the WebUI check can each ask for Node in one
    process; after a failure they must not wait out the network again."""
    attempts: list[str] = []

    def offline(url, *_a, **_k):
        attempts.append(url)
        raise StageError("download_failed", "offline")

    monkeypatch.setattr(node, "download", offline)
    for _ in range(3):
        with pytest.raises(StageError) as ei:
            node.ensure_node("default")
        assert ei.value.code == "download_failed"
    assert len(attempts) == 1
    # A system Node that appears later is still used.
    _system_node(env, monkeypatch, (22, 0, 0))
    assert node.ensure_node("default").source == "system"


@pytest.mark.skipif(os.name == "nt", reason="tar.xz archives are not used on Windows")
def test_tar_member_escaping_dest_is_rejected(env):
    env["net"].data = _archive_bytes(extra_member="../../evil")
    node.NODE_SHA256[PLAT] = hashlib.sha256(env["net"].data).hexdigest()
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "download_failed"
    assert not (env["data"] / "evil").exists()


def test_zip_extracts_into_path_with_space_and_non_ascii(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("node-v1-win-x64/node.exe", b"x")
    archive = tmp_path / "a.zip"
    archive.write_bytes(buf.getvalue())
    dest = tmp_path / "Jan Kowalski ąę" / "unpack"
    node._extract(archive, dest)
    assert (dest / "node-v1-win-x64" / "node.exe").read_bytes() == b"x"


@pytest.fixture
def glibc(monkeypatch):
    """The glibc version platform_id() sees; (2, 28) unless a test sets it."""
    version = {"value": (2, 28)}
    monkeypatch.setattr(node, "_glibc_version", lambda: version["value"])
    return version


@pytest.mark.parametrize(
    ("sys_platform", "machine", "expected"),
    [
        ("darwin", "arm64", "darwin-arm64"),
        ("linux", "x86_64", "linux-x64"),
        ("linux", "aarch64", "linux-arm64"),
        ("win32", "AMD64", "win-x64"),
    ],
)
def test_platform_id(monkeypatch, glibc, sys_platform, machine, expected):
    monkeypatch.setattr(node.sys, "platform", sys_platform)
    monkeypatch.setattr(node.platform, "machine", lambda: machine)
    assert node.platform_id() == expected
    assert node.archive_name(expected).endswith(
        ".zip" if expected.startswith("win") else ".tar.xz"
    )


@pytest.mark.parametrize(
    ("sys_platform", "machine", "libc", "words"),
    [
        ("win32", "ARM64", None, "Windows on arm64"),
        # musl (Alpine): no glibc at all.
        ("linux", "x86_64", None, "apk add nodejs npm"),
        # CentOS 7 / RHEL 7.
        ("linux", "x86_64", (2, 17), "glibc 2.17"),
        ("linux", "aarch64", (2, 27), "glibc 2.27"),
        ("darwin", "x86_64", None, "No Node.js build"),
        ("linux", "riscv64", (2, 39), "No Node.js build"),
        ("freebsd14", "amd64", None, "No Node.js build"),
    ],
)
def test_platform_id_unsupported(
    monkeypatch, glibc, sys_platform, machine, libc, words
):
    monkeypatch.setattr(node.sys, "platform", sys_platform)
    monkeypatch.setattr(node.platform, "machine", lambda: machine)
    glibc["value"] = libc
    with pytest.raises(StageError) as ei:
        node.platform_id()
    assert ei.value.code == "unsupported_platform"
    assert words in ei.value.message
    assert "EvoSci setup" in ei.value.message


@pytest.mark.parametrize(
    ("confstr", "expected"),
    [
        (lambda _n: "glibc 2.28", (2, 28)),
        (lambda _n: "glibc 2.17", (2, 17)),
        (lambda _n: None, None),
        (lambda _n: "musl 1.2", None),
    ],
)
def test_glibc_version_reads_confstr(monkeypatch, confstr, expected):
    monkeypatch.setattr(node.os, "confstr", confstr, raising=False)
    assert node._glibc_version() == expected


def test_glibc_version_is_none_without_confstr_support(monkeypatch):
    def unsupported(_name):
        raise ValueError("unrecognized configuration name")

    monkeypatch.setattr(node.os, "confstr", unsupported, raising=False)
    assert node._glibc_version() is None


def _unsupported_here(monkeypatch):
    def refuse():
        raise StageError("unsupported_platform", "This Linux has no glibc.")

    monkeypatch.setattr(node, "platform_id", refuse)


def test_unsupported_platform_is_refused_before_any_download(env, monkeypatch):
    _unsupported_here(monkeypatch)
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "unsupported_platform"
    assert env["net"].urls == []


def test_unsupported_platform_still_uses_a_system_node(env, monkeypatch):
    _unsupported_here(monkeypatch)
    _system_node(env, monkeypatch, (22, 0, 0))
    assert node.ensure_node("default").source == "system"


def test_unsupported_platform_keeps_a_recorded_private_node(env, monkeypatch):
    node.ensure_node("default")
    _unsupported_here(monkeypatch)
    env["net"].urls.clear()
    assert node.ensure_node("default").source == "private"
    assert env["net"].urls == []


def test_activate_runtime_prepends_once(env):
    assert node.activate_runtime() is None
    node.ensure_node("default")
    bin_dir = node.activate_runtime()
    assert bin_dir == node._bin_dir(env["data"] / "tools" / f"node-v{V}")
    node.activate_runtime()
    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[0] == str(bin_dir)
    assert parts.count(str(bin_dir)) == 1


def test_system_node_drops_private_record(env, monkeypatch):
    node.ensure_node("default")
    assert _record(env["data"]) is not None
    _system_node(env, monkeypatch, (22, 0, 0))
    assert node.ensure_node("default").source == "system"
    assert _record(env["data"]) is None


def test_private_node_on_path_is_not_taken_for_system(env):
    node.ensure_node("default")
    node.activate_runtime()
    env["net"].urls.clear()
    # The private Node is first on PATH, yet it is reported as private.
    assert node.ensure_node("default").source == "private"


def test_node_in_the_current_directory_is_not_taken_for_system(env, monkeypatch):
    exe = _system_node(env, monkeypatch, (22, 0, 0))
    monkeypatch.chdir(exe.parent)
    # "." is how PATH reaches the current directory on POSIX; Windows
    # searches it without one.
    monkeypatch.setenv("PATH", os.pathsep.join([str(env["tmp"] / "empty-bin"), "."]))
    assert node.ensure_node("default").source == "private"


def test_npx_only_in_the_current_directory_does_not_count(env, monkeypatch):
    _system_node(env, monkeypatch, (22, 0, 0), with_npx=False)
    cwd = env["tmp"] / "project"
    cwd.mkdir()
    npx = cwd / ("npx.cmd" if os.name == "nt" else "npx")
    npx.write_text("")
    npx.chmod(0o755)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", os.pathsep.join([os.environ["PATH"], "."]))
    assert node.ensure_node("default").source == "private"


def test_npx_only_elsewhere_on_path_does_not_count(env, monkeypatch):
    _system_node(env, monkeypatch, (22, 0, 0), with_npx=False)
    other = env["tmp"] / "other-bin"
    other.mkdir()
    npx = other / ("npx.cmd" if os.name == "nt" else "npx")
    npx.write_text("")
    npx.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([os.environ["PATH"], str(other)]))
    assert node.ensure_node("default").source == "private"


def test_install_without_a_record_is_used_again_without_a_download(env, monkeypatch):
    node.ensure_node("default")
    _system_node(env, monkeypatch, (22, 0, 0))
    node.ensure_node("default")
    assert _record(env["data"]) is None
    monkeypatch.setenv("PATH", str(env["tmp"] / "empty-bin"))
    env["net"].urls.clear()
    assert node.ensure_node("default").source == "private"
    assert env["net"].urls == []
    assert _record(env["data"])["version"] == V


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "v20.19.2\n", (20, 19, 2)),
        (1, "v20.19.2\n", None),
        (0, "not a version\n", None),
    ],
)
def test_probe_reads_the_version_node_prints(monkeypatch, returncode, stdout, expected):
    def run(args, **_kw):
        return node.subprocess.CompletedProcess(args, returncode, stdout=stdout)

    monkeypatch.setattr(node.subprocess, "run", run)
    assert node._probe(Path("node")) == expected


def test_install_without_a_mirror_argument_uses_the_saved_mirror(env, monkeypatch):
    from EvoScientist.config import set_config_value

    set_config_value("mirror", "cn")
    node.ensure_node()
    assert env["net"].urls
    assert all(u.startswith(node.SOURCES["cn"]) for u in env["net"].urls)


def test_is_private_tells_the_private_npx_from_a_system_one(env):
    info = node.ensure_node("default")
    npx = info.path.parent / ("npx.cmd" if os.name == "nt" else "npx")
    assert node.is_private(npx)
    assert not node.is_private(env["tmp"] / "system-bin" / "npx")
    assert not node.is_private(None)


def test_node_child_env_keeps_only_the_npm_network_settings():
    kept = {
        "npm_config_registry": "r",
        "NPM_CONFIG_PROXY": "http://proxy:3128",
        "npm_config_https_proxy": "http://proxy:3128",
        "npm_config_noproxy": "corp.example",
        "npm_config_cafile": "/ca.pem",
        "npm_config_ca": "-----BEGIN",
        "npm_config_strict_ssl": "false",
    }
    env = {
        "PATH": "p",
        **kept,
        "NPM_CONFIG_PREFIX": "x",
        "npm_config_globalconfig": "/etc/npmrc",
        "NODE_OPTIONS": "--foo",
    }
    assert node.node_child_env(env, private=True) == {"PATH": "p", **kept}
    assert node.node_child_env(env, private=False) == env
    assert node.npm_network_env(env) == kept


HTTP = "http://proxy.example:3128"


@pytest.mark.parametrize(
    ("env", "flag"),
    [
        ({"HTTPS_PROXY": HTTP}, True),
        ({"https_proxy": "https://proxy.example:3128", "HTTP_PROXY": HTTP}, True),
        ({"HTTPS_PROXY": "HTTP://user:p%40ss@[::1]:3128/"}, True),
        ({"HTTPS_PROXY": HTTP, "https_proxy": ""}, True),
        ({"HTTPS_PROXY": f" {HTTP} "}, True),
        ({"HTTPS_PROXY": "http://user:p ss@proxy.example:3128"}, True),
        ({}, False),
        ({"HTTPS_PROXY": "socks4://127.0.0.1:1080"}, False),
        ({"HTTPS_PROXY": "socks5://127.0.0.1:1080"}, False),
        ({"HTTPS_PROXY": "ftp://proxy.example"}, False),
        ({"HTTPS_PROXY": "not a url"}, False),
        ({"HTTPS_PROXY": "http://"}, False),
        ({"HTTPS_PROXY": "http://127.0.0.1:bad"}, False),
        ({"HTTPS_PROXY": "http://127.0.0.1:99999"}, False),
        ({"HTTPS_PROXY": "http://proxy.example:80800"}, False),
        ({"HTTPS_PROXY": "http://proxy.example:3128/path"}, False),
        ({"HTTPS_PROXY": "http://us?er@proxy.example:3128"}, False),
        ({"HTTPS_PROXY": "http://us#er@proxy.example:3128"}, False),
        ({"HTTPS_PROXY": "http://user:50%off@proxy.example:3128"}, False),
        ({"HTTPS_PROXY": "http://user:caf%e9@proxy.example:3128"}, False),
        ({"HTTPS_PROXY": "http://proxy example:3128"}, False),
        ({"HTTPS_PROXY": "http://proxy.example: 3128"}, False),
        ({"HTTPS_PROXY": "http://pro\u00a0xy.example:3128"}, False),
        ({"HTTPS_PROXY": "http://DOMAIN\\user:pass@proxy.example:3128"}, False),
        ({"HTTPS_PROXY": "http://%00:3128"}, False),
        ({"HTTPS_PROXY": "http://256.256.256.256:3128"}, False),
        ({"HTTPS_PROXY": "http://proxy.123:3128"}, False),
        ({"HTTPS_PROXY": "http://[fe80::1%25en0]:3128"}, False),
        ({"HTTPS_PROXY": "http://[v1.fe]:3128"}, False),
        ({"HTTPS_PROXY": "http://proxy%E3%80%82123:3128"}, False),
        ({"HTTPS_PROXY": HTTP + "\r"}, False),
        ({"HTTPS_PROXY": HTTP, "https_proxy": " "}, False),
        # One refused value among good ones still leaves the flag off.
        ({"HTTPS_PROXY": HTTP, "http_proxy": "socks4://127.0.0.1:1080"}, False),
    ],
)
def test_node_network_env_sets_the_flag_only_for_http_proxies(env, flag):
    extra = node.node_network_env(env, private=True)
    assert (extra.get("NODE_USE_ENV_PROXY") == "1") is flag


def test_node_network_env_is_empty_for_a_system_node():
    assert node.node_network_env({"HTTPS_PROXY": HTTP}, private=False) == {}


def test_node_network_env_trusts_the_system_certificates_without_a_proxy():
    """A TLS-inspecting network re-signs sites with a company certificate
    that only the system store holds, with or without a proxy variable."""
    assert node.node_network_env({}, private=True) == {"NODE_USE_SYSTEM_CA": "1"}


@pytest.mark.parametrize("name", ["NODE_USE_SYSTEM_CA", "node_use_system_ca"])
def test_node_network_env_keeps_the_users_system_ca_choice(name):
    assert "NODE_USE_SYSTEM_CA" not in node.node_network_env({name: "0"}, private=True)


@pytest.mark.parametrize(
    ("name", "value", "loopback"),
    [
        ("NODE_USE_ENV_PROXY", "0", False),
        ("node_use_env_proxy", "0", False),
        ("NODE_USE_ENV_PROXY", "1", True),
    ],
)
def test_node_network_env_keeps_the_users_proxy_switch(name, value, loopback):
    extra = node.node_network_env({"HTTPS_PROXY": HTTP, name: value}, private=True)
    assert "NODE_USE_ENV_PROXY" not in extra
    assert ("NO_PROXY" in extra) is loopback


def test_node_network_env_adds_loopback_to_each_no_proxy_it_finds():
    extra = node.node_network_env(
        {"HTTPS_PROXY": HTTP, "no_proxy": "corp.example, localhost", "NO_PROXY": "a"},
        private=True,
    )
    assert extra["no_proxy"] == "corp.example,localhost,127.0.0.1,::1,[::1]"
    assert extra["NO_PROXY"] == "a,localhost,127.0.0.1,::1,[::1]"


def test_node_network_env_sets_only_no_proxy_when_none_is_set():
    """Two names that differ only in case would share one Windows env block."""
    extra = node.node_network_env({"HTTPS_PROXY": HTTP}, private=True)
    assert extra == {
        "NODE_USE_SYSTEM_CA": "1",
        "NODE_USE_ENV_PROXY": "1",
        "NO_PROXY": "localhost,127.0.0.1,::1,[::1]",
    }


@pytest.mark.parametrize(
    ("mirror", "configured", "expected"),
    [
        ("cn", None, "https://registry.npmmirror.com/"),
        ("cn", "https://registry.yarnpkg.com/", None),
        ("default", None, None),
    ],
)
def test_npm_registry_override(monkeypatch, mirror, configured, expected):
    monkeypatch.setattr(node, "configured_mirror", lambda: mirror)
    monkeypatch.setattr(node, "configured_npm_registry", lambda _env, _exe: configured)
    assert node.npm_registry_override({}, Path("node")) == expected


class FakeNpmConfig:
    """Fake ``npm config get registry``; records each call."""

    def __init__(self, stdout: str = "", returncode: int = 0, exc=None) -> None:
        self.stdout, self.returncode, self.exc = stdout, returncode, exc
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, "")


@pytest.fixture
def npm_env(env, monkeypatch):
    """A node + npm pair outside PATH and a clean per-process registry cache."""
    bin_dir = env["tmp"] / "node-bin"
    bin_dir.mkdir()
    names = ["node.exe", "npm.cmd"] if os.name == "nt" else ["node", "npm"]
    for name in names:
        (bin_dir / name).write_text("")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setattr(node, "_npm_config_registry", {})
    return bin_dir / names[0]


@pytest.mark.parametrize("name", ["npm_config_registry", "NPM_CONFIG_REGISTRY"])
def test_configured_registry_from_env_wins_without_asking_npm(
    npm_env, monkeypatch, name
):
    fake = FakeNpmConfig("https://npmrc.example/")
    monkeypatch.setattr(node.subprocess, "run", fake)
    env = {name: "https://env.example/"}
    assert node.configured_npm_registry(env, npm_env) == "https://env.example/"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("https://npmrc.example/\n", "https://npmrc.example/"),
        ("https://registry.npmjs.org/\n", None),
        ("https://registry.npmjs.org\n", None),
        ("undefined\n", None),
        ("", None),
    ],
)
def test_configured_registry_from_npm_config(npm_env, monkeypatch, stdout, expected):
    fake = FakeNpmConfig(stdout)
    monkeypatch.setattr(node.subprocess, "run", fake)
    assert node.configured_npm_registry({}, npm_env) == expected
    (call,) = fake.calls
    assert call["cmd"][1:] == ["config", "get", "registry"]
    assert Path(call["cmd"][0]).parent == npm_env.parent


def test_configured_registry_asks_npm_once_per_process(npm_env, monkeypatch):
    fake = FakeNpmConfig("https://npmrc.example/")
    monkeypatch.setattr(node.subprocess, "run", fake)
    for _ in range(3):
        assert node.configured_npm_registry({}, npm_env) == "https://npmrc.example/"
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "other",
    [
        {"npm_config_userconfig": "/srv/mcp/.npmrc"},
        {"HOME": "/srv/mcp"},
        {"NPM_CONFIG_GLOBALCONFIG": "/srv/npmrc"},
    ],
)
def test_configured_registry_asks_again_for_another_npm_config(
    npm_env, monkeypatch, other
):
    """A server whose env points npm at another ``.npmrc`` gets its own answer,
    not the parent's cached one."""
    answers = iter(["https://registry.npmjs.org/", "https://private.example/"])

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, next(answers), "")

    monkeypatch.setattr(node.subprocess, "run", run)
    assert node.configured_npm_registry({"HOME": "/home/me"}, npm_env) is None
    env = {"HOME": "/home/me", **other}
    assert node.configured_npm_registry(env, npm_env) == "https://private.example/"


@pytest.mark.parametrize(
    "fake",
    [
        FakeNpmConfig(exc=OSError("blocked")),
        FakeNpmConfig(exc=subprocess.TimeoutExpired("npm", 10)),
        FakeNpmConfig("https://npmrc.example/", returncode=1),
    ],
)
def test_configured_registry_is_none_when_npm_fails(npm_env, monkeypatch, fake):
    monkeypatch.setattr(node.subprocess, "run", fake)
    assert node.configured_npm_registry({}, npm_env) is None


def test_configured_registry_is_none_without_npm(env, monkeypatch):
    fake = FakeNpmConfig("https://npmrc.example/")
    monkeypatch.setattr(node.subprocess, "run", fake)
    monkeypatch.setattr(node, "_npm_config_registry", {})
    assert node.configured_npm_registry({}, env["tmp"] / "nowhere" / "node") is None
    assert fake.calls == []


def test_configured_registry_ignores_npm_in_the_current_directory(env, monkeypatch):
    """On Windows ``shutil.which`` would take an ``npm.cmd`` from there."""
    cwd = env["tmp"] / "project"
    cwd.mkdir()
    for name in ("npm", "npm.cmd"):
        (cwd / name).write_text("")
        (cwd / name).chmod(0o755)
    monkeypatch.chdir(cwd)
    fake = FakeNpmConfig("https://npmrc.example/")
    monkeypatch.setattr(node.subprocess, "run", fake)
    monkeypatch.setattr(node, "_npm_config_registry", {})
    assert node.configured_npm_registry({}, env["tmp"] / "nowhere" / "node") is None
    assert fake.calls == []


def test_configured_registry_runs_npm_outside_the_current_project(
    env, npm_env, monkeypatch
):
    """A project ``.npmrc`` in the cwd must not decide; NODE_OPTIONS is dropped;
    npm's launcher finds the Node next to it first on PATH."""
    (env["data"] / "tools").mkdir(parents=True)
    fake = FakeNpmConfig("https://npmrc.example/")
    monkeypatch.setattr(node.subprocess, "run", fake)
    node.configured_npm_registry(
        {"NODE_OPTIONS": "--x", "HOME": "h", "PATH": "elsewhere"}, npm_env
    )
    (call,) = fake.calls
    assert Path(call["cwd"]) == env["data"] / "tools"
    assert call["env"] == {
        "HOME": "h",
        "PATH": os.pathsep.join([str(npm_env.parent), "elsewhere"]),
    }


def test_run_stage_returns_done_with_the_node_detail(monkeypatch):
    info = node.NodeInfo("system", "22.11.0", Path("/usr/bin/node"))
    monkeypatch.setattr(node, "ensure_node", lambda mirror, report: info)
    result = node.run_stage(lambda event: None, "default")
    assert result.status == "done"
    assert result.detail == info.detail()


# --------------------------------------------------------------------------- #
# Heartbeats, lock waits, Windows policy
# --------------------------------------------------------------------------- #
def test_unpacking_repeats_its_line_between_members(env, monkeypatch):
    from EvoScientist.setup import protocol

    monkeypatch.setattr(protocol, "HEARTBEAT_INTERVAL", 0)
    lines: list[str] = []
    node.ensure_node("default", lambda _f, message: lines.append(message))
    assert lines.count("Unpacking") > 1


def test_second_install_waits_for_the_first_and_says_so(env, monkeypatch):
    import threading

    from filelock import FileLock

    from EvoScientist.setup import _install

    monkeypatch.setattr(_install, "HEARTBEAT_INTERVAL", 0.05)
    tools = env["data"] / "tools"
    tools.mkdir(parents=True)
    holder = FileLock(str(tools / "node.lock"))
    holder.acquire()
    lines: list[str] = []
    result = {}
    waiter = threading.Thread(
        target=lambda: result.update(
            info=node.ensure_node("default", lambda _f, m: lines.append(m))
        )
    )
    waiter.start()
    try:
        deadline = 50
        while not lines and deadline:
            waiter.join(0.05)
            deadline -= 1
        # Meanwhile the first install finishes.
        node._write_record(V, tools / f"node-v{V}")
    finally:
        holder.release()
    waiter.join(10)
    assert (
        lines[0]
        == "Waiting for another EvoScientist process to finish installing Node.js"
    )
    assert result["info"].source == "private"
    assert env["net"].urls == []


def test_policy_block_right_after_install_is_probe_failed_and_keeps_the_files(env):
    final = env["data"] / "tools" / f"node-v{V}"
    env["probes"][str(node._node_exe(final))] = BLOCKED
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "probe_failed"
    assert "Windows policy" in ei.value.message
    assert str(node._node_exe(final)) in ei.value.message
    assert _record(env["data"]) is None
    assert node._node_exe(final).exists()


def test_blocked_install_is_reported_again_without_a_new_download(env, monkeypatch):
    final = env["data"] / "tools" / f"node-v{V}"
    env["probes"][str(node._node_exe(final))] = BLOCKED
    with pytest.raises(StageError):
        node.ensure_node("default")
    monkeypatch.setattr(node, "_failed_install", None)  # a new process
    env["net"].urls.clear()
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "probe_failed"
    assert env["net"].urls == []


def test_blocked_recorded_node_is_probe_failed_without_a_download(env):
    info = node.ensure_node("default")
    env["probes"][str(info.path)] = BLOCKED
    env["net"].urls.clear()
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "probe_failed"
    assert env["net"].urls == []


def test_blocked_system_node_falls_back_to_the_private_node(env, monkeypatch):
    exe = _system_node(env, monkeypatch, (22, 0, 0))
    env["probes"][str(exe)] = BLOCKED
    assert node.ensure_node("default").source == "private"


def test_probe_turns_a_policy_error_into_policy_blocked(monkeypatch):
    def blocked(*_a, **_k):
        exc = OSError(13, "This program is blocked by group policy")
        exc.winerror = 1260
        raise exc

    monkeypatch.setattr(node.subprocess, "run", blocked)
    with pytest.raises(node.PolicyBlocked):
        node._probe(Path("node.exe"))


def test_probe_returns_none_for_other_start_errors(monkeypatch):
    def missing(*_a, **_k):
        raise FileNotFoundError(2, "not found")

    monkeypatch.setattr(node.subprocess, "run", missing)
    assert node._probe(Path("node.exe")) is None


def test_record_is_written_atomically(env, monkeypatch):
    written: list[Path] = []

    def atomic_write_text(path, text):
        written.append(path)
        _install.atomic_write_text(path, text)

    monkeypatch.setattr(node, "atomic_write_text", atomic_write_text)
    node.ensure_node("default")
    assert written == [env["data"] / "tools" / "node.json"]
