"""Tests for the WebUI setup stage (``EvoScientist/setup/webui.py``)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import sys
import tarfile
import textwrap
import urllib.error
from pathlib import Path

import pytest

from EvoScientist.setup import STAGES, webui
from EvoScientist.setup.protocol import ERROR_CODES, StageError

REG = "https://registry.npmjs.org"


def _tarball(version: str, *, name: str = webui.PACKAGE, extra=()) -> bytes:
    """An npm-style tarball: ``package/package.json`` + ``package/dist/server.js``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:

        def add(path: str, data: bytes) -> None:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        add(
            "package/package.json",
            json.dumps({"name": name, "version": version}).encode(),
        )
        add("package/dist/server.js", b"// server")
        add("package/dist/.next/chunk.js", b"// chunk")
        for member in extra:
            tf.addfile(member)
    return buf.getvalue()


def _sri(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


class FakeRegistry:
    """Serves metadata and tarballs; records requested URLs."""

    def __init__(self, versions: dict[str, bytes]) -> None:
        self.tarballs = {f"{REG}/webui-{v}.tgz": data for v, data in versions.items()}
        self.metadata = {
            "versions": {
                v: {
                    "dist": {"tarball": f"{REG}/webui-{v}.tgz", "integrity": _sri(data)}
                }
                for v, data in versions.items()
            }
        }
        self.urls: list[str] = []
        self.offline = False

    def fetch_metadata(self, registry: str) -> dict:
        self.urls.append(registry)
        if self.offline:
            raise StageError("download_failed", f"Could not read from {registry}")
        return self.metadata

    def download(self, url: str, dest: Path, progress=None) -> str:
        self.urls.append(url)
        data = self.tarballs[url]
        dest.write_bytes(data)
        if progress:
            progress(1.0)
        return hashlib.sha256(data).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DATA_DIR with a space and non-ASCII characters; fake registry
    and probe; no override of the range."""
    from EvoScientist import paths

    data = tmp_path.resolve() / "Jan Kowalski ąę" / ".evoscientist"
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.delenv(webui.COMPAT_ENV, raising=False)
    monkeypatch.setattr(webui, "_failed_install", None)
    reg = FakeRegistry({"0.3.0": _tarball("0.3.0"), "0.3.1": _tarball("0.3.1")})
    monkeypatch.setattr(webui, "fetch_metadata", reg.fetch_metadata)
    monkeypatch.setattr(webui, "download", reg.download)
    monkeypatch.setattr(webui, "registry_url", lambda mirror, node_exe: REG)
    probes: list[Path] = []
    monkeypatch.setattr(webui, "probe", lambda root, node_exe: probes.append(root))
    return {"data": data, "reg": reg, "probes": probes, "node": tmp_path / "node"}


def _record(data: Path) -> dict | None:
    path = data / "tools" / "webui.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #
# Range and version choice
# --------------------------------------------------------------------------- #
def test_newest_version_in_range_wins():
    versions = dict.fromkeys(["0.2.9", "0.3.0", "0.3.2", "0.3.10", "0.4.0"])
    assert webui.pick_version(versions, ">=0.3,<0.4") == "0.3.10"


def test_prereleases_and_unparsable_versions_are_skipped():
    versions = dict.fromkeys(
        ["0.3.0", "0.3.1rc1", "0.3.2-beta.1", "0.3.2-0", "0.3.2-r.1", "garbage"]
    )
    assert webui.pick_version(versions, ">=0.3,<0.4") == "0.3.0"


def test_no_version_in_range_is_no_compatible_version():
    with pytest.raises(StageError) as exc:
        webui.pick_version(dict.fromkeys(["0.2.9", "0.4.0"]), ">=0.3,<0.4")
    assert exc.value.code == "no_compatible_version"
    assert "no_compatible_version" in ERROR_CODES


def test_compat_range_default(monkeypatch):
    monkeypatch.delenv(webui.COMPAT_ENV, raising=False)
    assert webui.compat_range() == webui.WEBUI_COMPAT


def test_compat_range_override_warns(monkeypatch, caplog):
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.2,<0.4")
    monkeypatch.setattr(webui, "_warned_override", None)
    with caplog.at_level(logging.WARNING, logger=webui.__name__):
        assert webui.compat_range() == ">=0.2,<0.4"
    assert "For testing only" in caplog.text


def test_invalid_compat_override_is_ignored_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv(webui.COMPAT_ENV, "not a range")
    monkeypatch.setattr(webui, "_warned_override", None)
    with caplog.at_level(logging.WARNING, logger=webui.__name__):
        assert webui.compat_range() == webui.WEBUI_COMPAT
    assert "Ignoring" in caplog.text


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("user", "mirror", "expected"),
    [
        ("https://corp.example/npm/", "cn", "https://corp.example/npm"),
        ("https://corp.example/npm/", "default", "https://corp.example/npm"),
        (None, "cn", "https://registry.npmmirror.com"),
        (None, "default", REG),
    ],
)
def test_registry_priority(monkeypatch, tmp_path, user, mirror, expected):
    from EvoScientist.setup import node

    monkeypatch.setattr(node, "configured_npm_registry", lambda env, exe: user)
    assert webui.registry_url(mirror, tmp_path / "node") == expected


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_metadata_sends_the_abbreviated_accept_header(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["url"], seen["accept"] = request.full_url, request.get_header("Accept")
        return _Resp(json.dumps({"versions": {"0.3.0": {}}}).encode())

    monkeypatch.setattr(webui.urllib.request, "urlopen", urlopen)
    assert webui.fetch_metadata(REG)["versions"] == {"0.3.0": {}}
    assert seen["url"] == f"{REG}/@evoscientist%2Fwebui"
    assert seen["accept"].startswith("application/vnd.npm.install-v1+json")


@pytest.mark.parametrize(
    ("status", "words"),
    [
        (401, "needs a token"),
        (403, "needs a token"),
        (404, "does not host"),
        (500, "500"),
    ],
)
def test_fetch_metadata_http_errors_name_the_registry(monkeypatch, status, words):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "x", {}, None)

    monkeypatch.setattr(webui.urllib.request, "urlopen", urlopen)
    with pytest.raises(StageError) as exc:
        webui.fetch_metadata("https://corp.example/npm")
    assert exc.value.code == "download_failed"
    assert "https://corp.example/npm" in exc.value.message
    assert words in exc.value.message


@pytest.mark.parametrize("error", ["refused", "cut off"])
def test_fetch_metadata_network_error_is_download_failed(monkeypatch, error):
    import http.client

    def urlopen(request, timeout):
        if error == "cut off":
            raise http.client.IncompleteRead(b'{"versions": {"0.3', 4982)
        raise urllib.error.URLError(error)

    monkeypatch.setattr(webui.urllib.request, "urlopen", urlopen)
    with pytest.raises(StageError) as exc:
        webui.fetch_metadata(REG)
    assert exc.value.code == "download_failed"


# --------------------------------------------------------------------------- #
# Integrity and extraction
# --------------------------------------------------------------------------- #
def test_integrity_match_passes(tmp_path):
    archive = tmp_path / "a.tgz"
    archive.write_bytes(b"data")
    webui.verify_integrity(archive, {"integrity": _sri(b"data")}, "x")


@pytest.mark.parametrize(
    "dist",
    [
        {"integrity": _sri(b"other")},
        {"integrity": "sha1-abc", "shasum": hashlib.sha1(b"data").hexdigest()},
        {},
    ],
)
def test_integrity_mismatch_or_missing_sha512_fails_closed(tmp_path, dist):
    archive = tmp_path / "a.tgz"
    archive.write_bytes(b"data")
    with pytest.raises(StageError) as exc:
        webui.verify_integrity(archive, dist, "x")
    assert exc.value.code == "checksum_mismatch"


def test_extract_skips_links_and_files_outside_package(tmp_path):
    link = tarfile.TarInfo("package/dist/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    other = tarfile.TarInfo("other/file")
    archive = tmp_path / "a.tgz"
    archive.write_bytes(_tarball("0.3.0", extra=[link, other]))
    dest = tmp_path / "out"
    webui.extract_package(archive, dest)
    assert (dest / "dist" / "server.js").read_bytes() == b"// server"
    assert not (dest / "dist" / "link").exists()
    assert not (tmp_path / "other").exists()


def test_extract_rejects_path_traversal(tmp_path):
    evil = tarfile.TarInfo("package/../../evil.txt")
    archive = tmp_path / "a.tgz"
    archive.write_bytes(_tarball("0.3.0", extra=[evil]))
    with pytest.raises(StageError) as exc:
        webui.extract_package(archive, tmp_path / "out")
    assert exc.value.code == "download_failed"
    assert not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------------------- #
# ensure_webui
# --------------------------------------------------------------------------- #
def test_install_records_newest_after_the_probe(env):
    info = webui.ensure_webui(env["node"], "default")
    final = env["data"] / "tools" / "webui" / "0.3.1"
    assert (info.version, info.path) == ("0.3.1", final)
    assert _record(env["data"]) == {"version": "0.3.1", "path": str(final)}
    assert info.server_entry.is_file()
    # Probed in its temporary directory, before the move into place.
    (probed,) = env["probes"]
    assert probed != final
    # Only the version directory is left; no temporary directories.
    assert [p.name for p in (env["data"] / "tools" / "webui").iterdir()] == ["0.3.1"]


def test_probe_failure_records_nothing(env, monkeypatch):
    def failing_probe(root, node_exe):
        raise StageError("probe_failed", "no answer")

    monkeypatch.setattr(webui, "probe", failing_probe)
    with pytest.raises(StageError) as exc:
        webui.ensure_webui(env["node"], "default")
    assert exc.value.code == "probe_failed"
    assert _record(env["data"]) is None
    assert not list((env["data"] / "tools" / "webui").iterdir())


def test_integrity_mismatch_leaves_the_record_unchanged(env):
    webui.ensure_webui(env["node"], "default")
    before = _record(env["data"])
    env["reg"].metadata["versions"]["0.3.2"] = {
        "dist": {"tarball": f"{REG}/webui-0.3.2.tgz", "integrity": _sri(b"x")}
    }
    env["reg"].tarballs[f"{REG}/webui-0.3.2.tgz"] = _tarball("0.3.2")
    with pytest.raises(StageError) as exc:
        webui.ensure_webui(env["node"], "default", refresh=True)
    assert exc.value.code == "checksum_mismatch"
    assert _record(env["data"]) == before


def test_wrong_package_in_the_archive_is_download_failed(env):
    env["reg"].tarballs[f"{REG}/webui-0.3.1.tgz"] = data = _tarball("0.3.0")
    env["reg"].metadata["versions"]["0.3.1"]["dist"]["integrity"] = _sri(data)
    with pytest.raises(StageError) as exc:
        webui.ensure_webui(env["node"], "default")
    assert exc.value.code == "download_failed"
    assert "0.3.0" in exc.value.message
    assert _record(env["data"]) is None


def test_recorded_copy_is_used_without_the_network(env):
    webui.ensure_webui(env["node"], "default")
    env["reg"].urls.clear()
    env["reg"].offline = True
    assert webui.ensure_webui(env["node"], "default").version == "0.3.1"
    assert env["reg"].urls == []


@pytest.mark.parametrize("problem", ["offline", "nothing in range"])
def test_refresh_keeps_the_recorded_copy_and_says_why(env, caplog, problem):
    webui.ensure_webui(env["node"], "default")
    if problem == "offline":
        env["reg"].offline = True
    else:
        env["reg"].metadata["versions"] = {}
    with caplog.at_level(logging.WARNING, logger=webui.logger.name):
        version = webui.ensure_webui(env["node"], "default", refresh=True).version
    assert version == "0.3.1"
    assert "Keeping WebUI 0.3.1" in caplog.text


def test_refresh_with_the_newest_recorded_downloads_nothing(env):
    webui.ensure_webui(env["node"], "default")
    env["reg"].urls.clear()
    webui.ensure_webui(env["node"], "default", refresh=True)
    assert env["reg"].urls == [REG]  # metadata only


def test_offline_with_nothing_installed_is_download_failed(env):
    env["reg"].offline = True
    with pytest.raises(StageError) as exc:
        webui.ensure_webui(env["node"], "default")
    assert exc.value.code == "download_failed"


def test_a_failed_install_is_not_retried_in_the_same_process(env):
    env["reg"].offline = True
    with pytest.raises(StageError):
        webui.ensure_webui(env["node"], "default")
    env["reg"].offline = False
    env["reg"].urls.clear()
    with pytest.raises(StageError):
        webui.ensure_webui(env["node"], "default")
    assert env["reg"].urls == []


def test_recorded_copy_outside_the_range_is_replaced(env, monkeypatch):
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.3,<0.3.1")
    assert webui.ensure_webui(env["node"], "default").version == "0.3.0"
    # Still in the default range: kept as it is, without the network.
    monkeypatch.delenv(webui.COMPAT_ENV)
    assert webui.ensure_webui(env["node"], "default").version == "0.3.0"
    # The range moved above it (a core upgrade): replaced.
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.3.1,<0.4")
    assert webui.ensure_webui(env["node"], "default").version == "0.3.1"
    assert _record(env["data"])["version"] == "0.3.1"


def test_recorded_copy_without_its_server_is_reinstalled(env):
    info = webui.ensure_webui(env["node"], "default")
    info.server_entry.unlink()
    assert webui.recorded_webui() is None
    assert webui.ensure_webui(env["node"], "default").server_entry.is_file()


def test_unrecorded_leftover_copy_is_set_aside(env):
    leftover = env["data"] / "tools" / "webui" / "0.3.1"
    leftover.mkdir(parents=True)
    (leftover / "stale").write_text("x")
    info = webui.ensure_webui(env["node"], "default")
    assert not (info.path / "stale").exists()
    assert info.server_entry.is_file()


# --------------------------------------------------------------------------- #
# Probe (a Python stand-in for ``node dist/server.js``)
# --------------------------------------------------------------------------- #
_FAKE_SERVER = textwrap.dedent(
    """
    import http.server, os, sys
    LOGO = {logo}
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path == "/" or (self.path == "/evoscientist-logo.png" and LOGO)
            self.send_response(200 if ok else 404)
            self.end_headers()
        def log_message(self, *a):
            pass
    if {exit_at_once}:
        print("boom", flush=True)
        sys.exit(3)
    http.server.HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
    """
)


def _fake_copy(
    tmp_path: Path, *, logo: bool = True, exit_at_once: bool = False
) -> Path:
    root = tmp_path / "copy"
    (root / "dist").mkdir(parents=True)
    (root / "dist" / "server.js").write_text(
        _FAKE_SERVER.format(logo=logo, exit_at_once=exit_at_once)
    )
    return root


def test_probe_passes_when_root_and_logo_answer(tmp_path):
    webui.probe(_fake_copy(tmp_path), Path(sys.executable), timeout=30)


def test_probe_fails_when_the_logo_is_missing(tmp_path):
    with pytest.raises(StageError) as exc:
        webui.probe(_fake_copy(tmp_path, logo=False), Path(sys.executable), timeout=30)
    assert exc.value.code == "probe_failed"
    assert "evoscientist-logo.png" in exc.value.message


def test_probe_fails_with_the_output_when_the_server_exits(tmp_path):
    with pytest.raises(StageError) as exc:
        webui.probe(
            _fake_copy(tmp_path, exit_at_once=True), Path(sys.executable), timeout=30
        )
    assert exc.value.code == "probe_failed"
    assert "exited" in exc.value.message
    assert "boom" in exc.value.message


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def test_stage_runs_after_research_env():
    ids = [s.id for s in STAGES]
    assert ids[-1] == "webui"
    assert ids.index("research-env") < ids.index("webui")


def test_run_stage_reports_the_installed_copy(env, monkeypatch):
    from EvoScientist.setup import node

    monkeypatch.setattr(
        node,
        "ensure_node",
        lambda mirror, progress=None: node.NodeInfo("system", "22.0.0", env["node"]),
    )
    events: list[dict] = []
    result = webui.run_stage(events.append, "default")
    assert result.status == "done"
    assert result.detail["version"] == "0.3.1"
    assert {e["status"] for e in events} == {"running"}


def test_run_stage_reports_a_node_download_under_the_webui_stage(env, monkeypatch):
    """``--stage webui`` / ``--skip node`` on a machine without Node: the Node
    download is visible, not a silent wait."""
    from EvoScientist.setup import node

    def ensure_node(mirror, progress=None):
        progress(0.5, "Downloading Node 24.21.0")
        return node.NodeInfo("private", "24.21.0", env["node"])

    monkeypatch.setattr(node, "ensure_node", ensure_node)
    events: list[dict] = []
    webui.run_stage(events.append, "default")
    assert {
        "protocol": 1,
        "stage": "webui",
        "status": "running",
        "progress": 0.5,
        "message": "Downloading Node 24.21.0",
    } in events


# --------------------------------------------------------------------------- #
# Background update and promotion
# --------------------------------------------------------------------------- #
def _install_old(env, monkeypatch) -> webui.WebUIInfo:
    """Record 0.3.0 while 0.3.1 is the newest in range."""
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.3,<0.3.1")
    info = webui.ensure_webui(env["node"], "default")
    monkeypatch.delenv(webui.COMPAT_ENV)
    env["probes"].clear()
    env["reg"].urls.clear()
    return info


def test_background_update_stages_without_probing_or_switching(env, monkeypatch):
    old = _install_old(env, monkeypatch)
    assert webui.stage_update(env["node"], "default") == "0.3.1"
    record = _record(env["data"])
    staged = env["data"] / "tools" / "webui" / "0.3.1"
    # The running launch keeps its copy; the new one waits for the next launch.
    assert (record["version"], record["path"]) == ("0.3.0", str(old.path))
    assert record["staged"] == {"version": "0.3.1", "path": str(staged)}
    assert old.server_entry.is_file()
    assert (staged / "dist" / "server.js").is_file()
    assert env["probes"] == []


def test_background_update_runs_once_per_interval(env, monkeypatch):
    _install_old(env, monkeypatch)
    webui.stage_update(env["node"], "default")
    env["reg"].urls.clear()
    assert webui.stage_update(env["node"], "default") is None
    assert env["reg"].urls == []


def test_background_update_skips_when_the_lock_is_held(env, monkeypatch):
    """Another process installing holds the lock; the check does not wait."""
    import filelock

    _install_old(env, monkeypatch)

    def busy(self, *args, **kwargs):
        raise filelock.Timeout(self.lock_file)

    monkeypatch.setattr(filelock.FileLock, "acquire", busy)
    assert webui.stage_update(env["node"], "default") is None
    assert env["reg"].urls == []


def test_background_update_failures_stay_quiet(env, monkeypatch):
    _install_old(env, monkeypatch)
    env["reg"].offline = True
    assert webui.stage_update(env["node"], "default") is None
    assert "staged" not in _record(env["data"])
    assert webui._failed_install is None


def test_background_update_without_a_record_does_nothing(env):
    assert webui.stage_update(env["node"], "default") is None
    assert env["reg"].urls == []


def test_next_launch_probes_and_promotes_the_staged_copy(env, monkeypatch):
    _install_old(env, monkeypatch)
    webui.stage_update(env["node"], "default")
    env["reg"].offline = True  # promotion needs no network
    info = webui.ensure_webui(env["node"], "default")
    staged = env["data"] / "tools" / "webui" / "0.3.1"
    assert (info.version, info.path) == ("0.3.1", staged)
    assert env["probes"] == [staged]
    record = _record(env["data"])
    assert (record["version"], record["previous"]) == ("0.3.1", "0.3.0")
    assert "staged" not in record


def test_staged_copy_that_fails_its_probe_is_rejected(env, monkeypatch):
    _install_old(env, monkeypatch)
    webui.stage_update(env["node"], "default")

    def failing_probe(root, node_exe):
        raise StageError("probe_failed", "no answer")

    monkeypatch.setattr(webui, "probe", failing_probe)
    assert webui.ensure_webui(env["node"], "default").version == "0.3.0"
    record = _record(env["data"])
    assert record["version"] == "0.3.0"
    assert record["rejected"] == ["0.3.1"]
    assert "staged" not in record
    assert not (env["data"] / "tools" / "webui" / "0.3.1").exists()
    # The next day's check does not fetch it again.
    monkeypatch.setattr(webui, "_checked_recently", lambda now: False)
    env["reg"].urls.clear()
    assert webui.stage_update(env["node"], "default") is None
    assert all("webui-0.3.1" not in u for u in env["reg"].urls)


def test_staged_copy_outside_the_range_is_dropped_without_a_probe(env, monkeypatch):
    _install_old(env, monkeypatch)
    webui.stage_update(env["node"], "default")
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.3,<0.3.1")  # the range moved
    assert webui.ensure_webui(env["node"], "default").version == "0.3.0"
    assert env["probes"] == []
    assert "staged" not in _record(env["data"])


# --------------------------------------------------------------------------- #
# In-use markers and cleanup
# --------------------------------------------------------------------------- #
def _fake_version(env, version: str) -> Path:
    path = env["data"] / "tools" / "webui" / version
    (path / "dist").mkdir(parents=True)
    (path / "dist" / "server.js").write_text("")
    return path


def test_cleanup_keeps_current_previous_and_staged(env, monkeypatch):
    _install_old(env, monkeypatch)
    webui.stage_update(env["node"], "default")  # 0.3.0 current, 0.3.1 staged
    _fake_version(env, "0.2.9")
    assert webui.cleanup_old_versions() == ["0.2.9"]
    webui.ensure_webui(env["node"], "default")  # 0.3.1 current, 0.3.0 previous
    _fake_version(env, "0.2.8")
    assert webui.cleanup_old_versions() == ["0.2.8"]
    left = sorted(p.name for p in (env["data"] / "tools" / "webui").iterdir())
    assert left == ["0.3.0", "0.3.1"]


def test_cleanup_keeps_a_version_a_live_process_runs(env, monkeypatch):
    webui.ensure_webui(env["node"], "default")
    old = webui.WebUIInfo("0.2.9", _fake_version(env, "0.2.9"))
    marker = webui.mark_in_use(old)
    assert marker is not None
    assert marker.exists()
    assert webui.cleanup_old_versions() == []
    webui.release_in_use(marker)
    assert webui.cleanup_old_versions() == ["0.2.9"]


def test_cleanup_ignores_a_marker_whose_pid_was_reused(env, monkeypatch):
    webui.ensure_webui(env["node"], "default")
    path = _fake_version(env, "0.2.9")
    # This process's pid with another start time: a dead process's marker.
    start = webui._process_start_ms(os.getpid())
    (path / ".in-use").mkdir()
    (path / ".in-use" / f"{os.getpid()}-{start - 5000}").touch()
    (path / ".in-use" / "garbage").touch()
    assert webui.cleanup_old_versions() == ["0.2.9"]


def test_a_live_marker_survives_a_small_clock_step(env):
    """psutil derives the start time from the boot time, which moves when the
    clock is stepped; a live session's marker must still count."""
    webui.ensure_webui(env["node"], "default")
    path = _fake_version(env, "0.2.9")
    start = webui._process_start_ms(os.getpid())
    (path / ".in-use").mkdir()
    (path / ".in-use" / f"{os.getpid()}-{start - 1500}").touch()
    assert webui.cleanup_old_versions() == []
    assert (path / "dist" / "server.js").is_file()


def test_cleanup_keeps_a_directory_it_cannot_rename(env, monkeypatch):
    webui.ensure_webui(env["node"], "default")
    path = _fake_version(env, "0.2.9")

    def deny(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(webui.os, "replace", deny)
    assert webui.cleanup_old_versions() == []
    assert (path / "dist" / "server.js").is_file()


def test_cleanup_without_a_record_or_directory_does_nothing(env):
    assert webui.cleanup_old_versions() == []
    (env["data"] / "tools" / "webui").mkdir(parents=True)
    assert webui.cleanup_old_versions() == []


# --------------------------------------------------------------------------- #
# Review round 1: no downgrade, live leftovers, stamp, malformed metadata
# --------------------------------------------------------------------------- #
def test_a_lagging_registry_never_downgrades(env, monkeypatch):
    """A registry whose newest release is older than the copy we run (a
    mirror or company proxy that lags) leaves the copy alone."""
    webui.ensure_webui(env["node"], "default")  # 0.3.1
    del env["reg"].metadata["versions"]["0.3.1"]
    assert webui.stage_update(env["node"], "default") is None
    assert "staged" not in _record(env["data"])
    assert webui.ensure_webui(env["node"], "default", refresh=True).version == "0.3.1"
    assert _record(env["data"])["version"] == "0.3.1"


def test_background_update_does_not_replace_a_newer_staged_copy(env, monkeypatch):
    v032 = _tarball("0.3.2")
    env["reg"].tarballs[f"{REG}/webui-0.3.2.tgz"] = v032
    env["reg"].metadata["versions"]["0.3.2"] = {
        "dist": {"tarball": f"{REG}/webui-0.3.2.tgz", "integrity": _sri(v032)}
    }
    _install_old(env, monkeypatch)
    assert webui.stage_update(env["node"], "default") == "0.3.2"
    monkeypatch.setattr(webui, "_checked_recently", lambda now: False)
    del env["reg"].metadata["versions"]["0.3.2"]
    assert webui.stage_update(env["node"], "default") is None
    assert _record(env["data"])["staged"]["version"] == "0.3.2"


def test_the_next_daily_check_does_not_fetch_the_staged_copy_again(env, monkeypatch):
    _install_old(env, monkeypatch)
    assert webui.stage_update(env["node"], "default") == "0.3.1"
    monkeypatch.setattr(webui, "_checked_recently", lambda now: False)
    env["reg"].urls.clear()
    assert webui.stage_update(env["node"], "default") is None
    assert env["reg"].urls == [REG]


def test_a_leftover_copy_a_live_process_runs_is_not_replaced(env):
    leftover = webui.WebUIInfo("0.3.1", _fake_version(env, "0.3.1"))
    marker = webui.mark_in_use(leftover)
    with pytest.raises(StageError) as exc:
        webui.ensure_webui(env["node"], "default")
    assert exc.value.code == "install_failed"
    assert "in use" in exc.value.message
    assert marker.exists()
    assert leftover.server_entry.is_file()


def test_an_offline_check_does_not_stamp_the_day(env, monkeypatch):
    _install_old(env, monkeypatch)
    env["reg"].offline = True
    assert webui.stage_update(env["node"], "default") is None
    assert not (env["data"] / "tools" / "webui.check.json").exists()
    env["reg"].offline = False
    assert webui.stage_update(env["node"], "default") == "0.3.1"
    assert (env["data"] / "tools" / "webui.check.json").exists()


@pytest.mark.parametrize("entry", ["not a dict", {"dist": "not a dict"}, {}])
def test_malformed_version_entry_is_download_failed(env, entry):
    env["reg"].metadata["versions"]["0.3.1"] = entry
    with pytest.raises(StageError) as exc:
        webui.ensure_webui(env["node"], "default")
    assert exc.value.code == "download_failed"
