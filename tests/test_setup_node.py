"""Tests for the Node setup stage. Downloads and ``node --version`` are mocked."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from EvoScientist.setup import node
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
    """Serves one archive and its SHASUMS256.txt; records requested URLs."""

    def __init__(self, data: bytes, *, sums: str | None = None) -> None:
        self.data = data
        self.sums = sums or f"{hashlib.sha256(data).hexdigest()}  {FILENAME}\n"
        self.urls: list[str] = []

    def fetch_text(self, url: str) -> str:
        self.urls.append(url)
        return self.sums

    def download(self, url, dest: Path, progress=None) -> str:
        self.urls.append(url)
        dest.write_bytes(self.data)
        if progress:
            progress(1.0)
        return hashlib.sha256(self.data).hexdigest()


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

    data = tmp_path / "Jan Kowalski ąę" / ".evoscientist"
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(node, "platform_id", lambda: PLAT)
    net = FakeNet(_archive_bytes())
    monkeypatch.setattr(node, "fetch_text", net.fetch_text)
    monkeypatch.setattr(node, "download", net.download)
    probes = ProbeTable()
    # Anything under our tools dir probes as the pinned version by default.
    monkeypatch.setattr(
        node,
        "_probe",
        lambda exe: probes.get(exe, tuple(map(int, V.split(".")))),
    )
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

    monkeypatch.setattr(node, "_MOVE_RETRY_DELAYS", (0, 0, 0, 0, 0))
    monkeypatch.setattr(node.os, "replace", replace)
    return denied


def test_brief_access_denied_on_move_is_retried(env, monkeypatch, caplog):
    denied = _deny_moving_node(monkeypatch, times=2)
    with caplog.at_level("WARNING", logger=node.__name__):
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
    def disk_full(archive, dest):
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


@pytest.mark.parametrize(
    "exc",
    [http.client.IncompleteRead(b""), http.client.BadStatusLine("garbled")],
)
def test_http_client_errors_become_download_failed(tmp_path, monkeypatch, exc):
    from EvoScientist.setup import download as dl

    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(dl.urllib.request, "urlopen", boom)
    with pytest.raises(StageError) as ei:
        dl.download("https://example.invalid/x", tmp_path / "x")
    assert ei.value.code == "download_failed"
    with pytest.raises(StageError) as ei:
        dl.fetch_text("https://example.invalid/x")
    assert ei.value.code == "download_failed"


def test_download_that_ends_early_is_download_failed(tmp_path, monkeypatch):
    from EvoScientist.setup import download as dl

    def urlopen(url, timeout):
        response = io.BytesIO(b"half")
        response.headers = {"Content-Length": "8"}
        return response

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    with pytest.raises(StageError) as ei:
        dl.download("https://example.invalid/x", tmp_path / "x")
    assert ei.value.code == "download_failed"


def test_disk_error_while_downloading_is_install_failed(tmp_path, monkeypatch):
    from EvoScientist.setup import download as dl

    def urlopen(url, timeout):
        response = io.BytesIO(b"data")
        response.headers = {}
        return response

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    with pytest.raises(StageError) as ei:
        dl.download("https://example.invalid/x", tmp_path / "missing-dir" / "x")
    assert ei.value.code == "install_failed"


def test_stage_reports_download_progress_lines(env, monkeypatch):
    from EvoScientist.setup import download as dl

    data = env["net"].data

    def urlopen(url, timeout):
        response = io.BytesIO(data)
        response.headers = {"Content-Length": str(len(data))}
        return response

    monkeypatch.setattr(dl, "_CHUNK", len(data) // 4)
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(node, "download", dl.download)
    events: list[dict] = []
    _message, detail = node.run_stage(events.append, "default")
    assert {(e["stage"], e["status"]) for e in events} == {("node", "running")}
    downloading = [
        e["progress"] for e in events if e["message"].startswith("Downloading")
    ]
    assert len(downloading) >= 5
    assert downloading == sorted(downloading)
    assert detail["source"] == "private"


def test_system_node_20_or_newer_wins_without_download(env, monkeypatch):
    exe = _system_node(env, monkeypatch, (20, 0, 0))
    info = node.ensure_node("default")
    detail = info.detail()
    assert (detail["source"], detail["version"]) == ("system", "20.0.0")
    # shutil.which may return the PATHEXT spelling (node.EXE) on Windows.
    assert os.path.normcase(detail["path"]) == os.path.normcase(str(exe))
    assert env["net"].urls == []


def test_no_node_downloads_and_records(env):
    info = node.ensure_node("default")
    assert info.source == "private"
    assert env["net"].urls[0] == f"https://nodejs.org/dist/v{V}/SHASUMS256.txt"
    assert env["net"].urls[1] == f"https://nodejs.org/dist/v{V}/{FILENAME}"
    record = _record(env["data"])
    assert record["version"] == V
    assert Path(record["path"]) == env["data"] / "tools" / f"node-v{V}"
    assert info.path.exists()
    # The temp dir is gone; only the install, the record and the lock remain.
    assert not [
        p for p in (env["data"] / "tools").iterdir() if p.name.startswith(".node-")
    ]


def test_old_system_node_triggers_download(env, monkeypatch):
    _system_node(env, monkeypatch, (19, 9, 0))
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


def test_checksum_mismatch_keeps_current_node(env):
    node.ensure_node("default")
    before = _record(env["data"])
    # Break the current install's probe so a new download is attempted.
    env["probes"][str(node._node_exe(Path(before["path"])))] = None
    env["net"].sums = f"{'0' * 64}  {FILENAME}\n"
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "checksum_mismatch"
    assert _record(env["data"]) == before


def test_file_missing_from_sums_fails(env):
    env["net"].sums = f"{'0' * 64}  some-other-file.zip\n"
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "checksum_mismatch"
    assert _record(env["data"]) is None


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
    def offline(url):
        raise StageError("download_failed", "offline")

    monkeypatch.setattr(node, "fetch_text", offline)
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "download_failed"
    assert _record(env["data"]) is None


@pytest.mark.skipif(os.name == "nt", reason="tar.xz archives are not used on Windows")
def test_tar_member_escaping_dest_is_rejected(env):
    env["net"].data = _archive_bytes(extra_member="../../evil")
    env["net"].sums = f"{hashlib.sha256(env['net'].data).hexdigest()}  {FILENAME}\n"
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


@pytest.mark.parametrize(
    ("sys_platform", "machine", "expected"),
    [
        ("darwin", "arm64", "darwin-arm64"),
        ("darwin", "x86_64", "darwin-x64"),
        ("linux", "x86_64", "linux-x64"),
        ("linux", "aarch64", "linux-arm64"),
        ("win32", "AMD64", "win-x64"),
        ("win32", "ARM64", "win-arm64"),
    ],
)
def test_platform_id(monkeypatch, sys_platform, machine, expected):
    monkeypatch.setattr(node.sys, "platform", sys_platform)
    monkeypatch.setattr(node.platform, "machine", lambda: machine)
    monkeypatch.setattr(node, "_is_musl", lambda: False)
    assert node.platform_id() == expected
    assert node.archive_name(expected).endswith(
        ".zip" if expected.startswith("win") else ".tar.xz"
    )


def test_platform_id_musl_x64_uses_the_musl_build(monkeypatch):
    monkeypatch.setattr(node.sys, "platform", "linux")
    monkeypatch.setattr(node.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(node, "_is_musl", lambda: True)
    assert node.platform_id() == "linux-x64-musl"
    assert node.archive_name("linux-x64-musl") == f"node-v{V}-linux-x64-musl.tar.xz"


@pytest.mark.parametrize(
    ("sys_platform", "machine", "musl"),
    [
        ("linux", "riscv64", False),
        ("freebsd14", "amd64", False),
        # nodejs.org publishes no musl build for arm64.
        ("linux", "aarch64", True),
    ],
)
def test_platform_id_unsupported(monkeypatch, sys_platform, machine, musl):
    monkeypatch.setattr(node.sys, "platform", sys_platform)
    monkeypatch.setattr(node.platform, "machine", lambda: machine)
    monkeypatch.setattr(node, "_is_musl", lambda: musl)
    with pytest.raises(StageError) as ei:
        node.platform_id()
    assert ei.value.code == "unsupported_platform"


def test_is_musl_detects_the_musl_loader(monkeypatch, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    real_path = node.Path
    monkeypatch.setattr(node.sys, "platform", "linux")
    monkeypatch.setattr(node, "Path", lambda p: lib if p == "/lib" else real_path(p))
    # On musl, libc_ver() reports nothing (checked on Alpine 3.22).
    monkeypatch.setattr(node.platform, "libc_ver", lambda: ("", ""))
    assert node._is_musl() is False
    (lib / "ld-musl-x86_64.so.1").write_text("")
    assert node._is_musl() is True
    # Debian / Ubuntu with the `musl` package: the loader exists, but the
    # interpreter runs on glibc, so the glibc build is the right one.
    monkeypatch.setattr(node.platform, "libc_ver", lambda: ("glibc", "2.41"))
    assert node._is_musl() is False


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

    monkeypatch.setenv("XDG_CONFIG_HOME", str(env["tmp"] / "cfg"))
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


def test_node_child_env_strips_npm_config_and_node_options():
    env = {
        "PATH": "p",
        "npm_config_registry": "r",
        "NPM_CONFIG_PREFIX": "x",
        "NODE_OPTIONS": "--foo",
    }
    assert node.node_child_env(env, private=True) == {"PATH": "p"}
    assert node.node_child_env(env, private=False) == env
