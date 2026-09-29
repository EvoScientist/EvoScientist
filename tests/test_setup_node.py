"""Tests for the Node setup stage. Downloads and ``node --version`` are mocked."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
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
            for name in (f"{BASE}/bin/node", extra_member):
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
    probes: dict[str, tuple | None] = {}
    # Anything under our tools dir probes as the pinned version by default.
    monkeypatch.setattr(
        node,
        "_probe",
        lambda exe: probes.get(str(exe), tuple(map(int, V.split(".")))),
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


def test_unwritable_tools_dir_raises_stage_error(env, monkeypatch):
    from EvoScientist import paths

    blocker = env["tmp"] / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(paths, "DATA_DIR", blocker / ".evoscientist")
    with pytest.raises(StageError) as ei:
        node.ensure_node("default")
    assert ei.value.code == "download_failed"


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


def test_system_node_20_or_newer_wins_without_download(env, monkeypatch):
    exe = _system_node(env, monkeypatch, (22, 11, 0))
    info = node.ensure_node("default")
    assert info.detail() == {"source": "system", "version": "22.11.0", "path": str(exe)}
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
    _system_node(env, monkeypatch, (18, 20, 0))
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
    assert node.platform_id() == expected
    assert node.archive_name(expected).endswith(
        ".zip" if expected.startswith("win") else ".tar.xz"
    )


@pytest.mark.parametrize(
    ("sys_platform", "machine"), [("linux", "riscv64"), ("freebsd14", "amd64")]
)
def test_platform_id_unsupported(monkeypatch, sys_platform, machine):
    monkeypatch.setattr(node.sys, "platform", sys_platform)
    monkeypatch.setattr(node.platform, "machine", lambda: machine)
    with pytest.raises(StageError) as ei:
        node.platform_id()
    assert ei.value.code == "unsupported_platform"


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


def test_node_child_env_strips_npm_config_and_node_options():
    env = {
        "PATH": "p",
        "npm_config_registry": "r",
        "NPM_CONFIG_PREFIX": "x",
        "NODE_OPTIONS": "--foo",
    }
    assert node.node_child_env(env, private=True) == {"PATH": "p"}
    assert node.node_child_env(env, private=False) == env
