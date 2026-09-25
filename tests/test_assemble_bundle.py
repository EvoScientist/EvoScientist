"""Tests for the Windows bundle assembler's pure logic.

Network-touching steps (fetch_webui/fetch_node/fix_sharp) are exercised
manually against the live registry; here we pin the risky pure pieces: the
tar-subtree extractor's prefix stripping, symlink skipping and path-traversal
guard, plus the Node target->archive mapping.
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parents[1] / "packaging" / "windows" / "assemble_bundle.py"
)
_spec = importlib.util.spec_from_file_location("assemble_bundle", _MOD_PATH)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _make_tar(members: list[tuple]) -> bytes:
    """members: (name, bytes) for a file, or (name, ("sym", target)) for a link."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members:
            if isinstance(content, tuple) and content[0] == "sym":
                ti = tarfile.TarInfo(name)
                ti.type = tarfile.SYMTYPE
                ti.linkname = content[1]
                tar.addfile(ti)
            else:
                data = content or b""
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_extract_strips_prefix_and_ignores_outside(tmp_path):
    tgz = _make_tar(
        [
            ("package/dist/server.js", b"S"),
            ("package/dist/sub/a.txt", b"A"),
            ("package/other.txt", b"O"),  # outside the strip prefix
            ("package/dist/ln", ("sym", "server.js")),  # symlink skipped
        ]
    )
    files, skipped = ab._extract_tar_subtree(tgz, strip="package/dist/", dest=tmp_path)
    assert files == 2
    assert skipped == 1
    assert (tmp_path / "server.js").read_bytes() == b"S"
    assert (tmp_path / "sub" / "a.txt").read_bytes() == b"A"
    assert not (tmp_path / "other.txt").exists()


def test_extract_rejects_path_traversal(tmp_path):
    tgz = _make_tar([("package/dist/../../evil.txt", b"X")])
    with pytest.raises(RuntimeError, match="unsafe path"):
        ab._extract_tar_subtree(tgz, strip="package/dist/", dest=tmp_path)


def test_extract_rejects_sibling_prefix_path(tmp_path):
    # A same-parent sibling (``<dest>-evil``) shares ``dest``'s string prefix but
    # is outside it; the containment check must still reject it.
    dest = tmp_path / "dist"
    dest.mkdir()
    tgz = _make_tar([("package/dist/../dist-evil/x", b"X")])
    with pytest.raises(RuntimeError, match="unsafe path"):
        ab._extract_tar_subtree(tgz, strip="package/dist/", dest=dest)
    assert not (tmp_path / "dist-evil").exists()


def test_node_arch_mapping_known_and_unknown(tmp_path):
    assert "win32-x64" in ab._NODE_ARCH
    # Unknown target fails fast, before any network access.
    with pytest.raises(RuntimeError, match="no Node archive mapping"):
        ab.fetch_node("22.11.0", "linux-x64", tmp_path)


def test_python_arch_mapping_known_and_unknown(tmp_path):
    assert "win32-x64" in ab._PYTHON_ARCH
    # Unknown target fails fast, before any network access.
    with pytest.raises(RuntimeError, match="no Python archive mapping"):
        ab.fetch_python("3.12.7", "20241016", "linux-x64", tmp_path)


def test_verify_integrity_sha512_match_and_mismatch():
    import base64
    import hashlib

    data = b"webui-tarball-bytes"
    good = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    assert "sha512" in ab._verify_tarball_integrity(data, {"integrity": good})
    bad = "sha512-" + base64.b64encode(hashlib.sha512(b"other").digest()).decode()
    with pytest.raises(RuntimeError, match="integrity"):
        ab._verify_tarball_integrity(data, {"integrity": bad})


def test_verify_integrity_falls_back_to_shasum():
    import hashlib

    data = b"tarball"
    good = hashlib.sha1(data).hexdigest()
    assert "sha1" in ab._verify_tarball_integrity(data, {"shasum": good})
    with pytest.raises(RuntimeError, match="shasum"):
        ab._verify_tarball_integrity(data, {"shasum": "0" * 40})


def test_verify_integrity_no_published_checksum_does_not_raise():
    # Defensive: a missing integrity/shasum must warn, not break the build.
    assert "no npm-published checksum" in ab._verify_tarball_integrity(b"x", {})
