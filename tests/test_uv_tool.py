"""Tests for EvoScientist.uv_tool: receipt reading and the uv to run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from EvoScientist import paths, uv_tool

RECEIPT = """\
[tool]
requirements = [
    { name = "evoscientist", extras = ["oauth", "telegram"], specifier = "==0.3.6" },
    { name = "six", specifier = ">=1.10,<2" },
    { name = "idna", extras = ["all"], marker = "python_full_version >= '3.8'", specifier = ">=3" },
    { name = "pyjokes", git = "https://github.com/pyjokes/pyjokes?rev=v0.6.0#7f1a2b3c" },
    { name = "pkg-sub", git = "https://github.com/o/r?branch=dev&subdirectory=pkg" },
    { name = "pinned", git = "https://github.com/o/p#0123abcd" },
    { name = "wheel", url = "https://example.org/wheel-1.0-py3-none-any.whl" },
    { name = "local", directory = "@LOCAL@" },
    { name = "dev", editable = "/opt/src/dev" },
]
constraints = [
    { name = "six", specifier = "==1.17.0" },
    { name = "tomli", marker = "python_full_version <= '3.11'", specifier = "==2.5.0" },
]
python = "3.12"
entrypoints = [{ name = "EvoSci", install-path = "/x/bin/EvoSci", from = "evoscientist" }]

[tool.options]
exclude-newer = "2026-10-01T00:00:00Z"
"""
# An absolute path on every OS (on Windows a path needs a drive).
LOCAL = (Path(__file__).resolve().parent / "local-src").as_posix()
RECEIPT = RECEIPT.replace("@LOCAL@", LOCAL)


def _env(tmp_path: Path, text: str = RECEIPT) -> Path:
    env = tmp_path / "tools" / "evoscientist"
    env.mkdir(parents=True)
    (env / "uv-receipt.toml").write_text(text, encoding="utf-8")
    return env


def test_tool_env_from_sys_prefix(tmp_path, monkeypatch):
    env = _env(tmp_path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", str(env))
    assert uv_tool.tool_env() == env


def test_tool_env_none_without_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert uv_tool.tool_env() is None


def test_read_receipt_fields(tmp_path):
    receipt = uv_tool.read_receipt(_env(tmp_path))
    assert receipt.extras == ["oauth", "telegram"]
    assert receipt.source is None
    assert receipt.python == "3.12"
    assert receipt.constraint_lines() == [
        "six==1.17.0",
        "tomli==2.5.0 ; python_full_version <= '3.11'",
    ]


def test_with_args_keep_specifier_extras_marker_and_sources(tmp_path):
    receipt = uv_tool.read_receipt(_env(tmp_path))
    assert receipt.with_args() == [
        "--with",
        "six>=1.10,<2",
        "--with",
        "idna[all]>=3 ; python_full_version >= '3.8'",
        # A ref wins over the resolved commit in the fragment.
        "--with",
        "pyjokes @ git+https://github.com/pyjokes/pyjokes@v0.6.0",
        "--with",
        "pkg-sub @ git+https://github.com/o/r@dev#subdirectory=pkg",
        # Without a ref, the commit is the ref.
        "--with",
        "pinned @ git+https://github.com/o/p@0123abcd",
        "--with",
        "wheel @ https://example.org/wheel-1.0-py3-none-any.whl",
        "--with",
        f"local @ {Path(LOCAL).as_uri()}",
        "--with-editable",
        "/opt/src/dev",
    ]


def test_with_args_skip_matches_normalized_name(tmp_path):
    receipt = uv_tool.read_receipt(_env(tmp_path))
    args = receipt.with_args(skip="PyJokes")
    assert not any("pyjokes" in a for a in args)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("directory", "/opt/src/evo"),
        ("git", "https://github.com/EvoScientist/EvoScientist?branch=main"),
        ("url", "https://example.org/evoscientist-0.3.6-py3-none-any.whl"),
        ("editable", "/opt/src/evo"),
    ],
)
def test_source_of_own_requirement(tmp_path, key, value):
    text = f'[tool]\nrequirements = [{{ name = "evoscientist", {key} = "{value}" }}]\n'
    assert uv_tool.read_receipt(_env(tmp_path, text)).source == value


def test_receipt_without_python_key(tmp_path, monkeypatch):
    receipt = uv_tool.read_receipt(
        _env(tmp_path, '[tool]\nrequirements = [{ name = "evoscientist" }]\n')
    )
    assert receipt.python is None
    assert receipt.constraint_lines() == []
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert uv_tool.python_request(receipt) == py


@pytest.mark.parametrize(
    "text",
    [
        "not [ toml",
        "[other]\n",
        '[tool]\nrequirements = [{ name = "six" }]\n',
        '[tool]\nrequirements = "evoscientist"\n',
    ],
)
def test_unreadable_receipt_raises(tmp_path, text):
    with pytest.raises(uv_tool.ReceiptError):
        uv_tool.read_receipt(_env(tmp_path, text))


def test_missing_receipt_raises(tmp_path):
    with pytest.raises(uv_tool.ReceiptError):
        uv_tool.read_receipt(tmp_path)


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "v", "virtual": "/opt/v"},
        {"name": "rel", "directory": "relative/dir"},
        {"name": "g", "git": "https://github.com/o/r?path=sub&rev=x"},
        {"specifier": "==1"},
    ],
)
def test_requirement_args_refuse_unsupported(entry):
    with pytest.raises(uv_tool.ReceiptError):
        uv_tool.requirement_args(entry)


def test_find_uv_prefers_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    recorded = tmp_path / "tools" / "uv-0.12.23" / "uv"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("")
    (tmp_path / "tools" / "uv.json").write_text(
        json.dumps({"version": "0.12.23", "path": str(recorded)})
    )
    monkeypatch.setattr(uv_tool.shutil, "which", lambda name: "/usr/bin/uv")
    assert uv_tool.find_uv() == str(recorded)


@pytest.mark.parametrize("record", [None, "{broken", '{"path": "/no/such/uv"}'])
def test_find_uv_falls_back_to_path(tmp_path, monkeypatch, record):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    if record is not None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "uv.json").write_text(record)
    monkeypatch.setattr(uv_tool.shutil, "which", lambda name: "/usr/bin/uv")
    assert uv_tool.find_uv() == "/usr/bin/uv"


def test_runs_on_managed_python(tmp_path, monkeypatch):
    managed = tmp_path / "uv-python"
    base = managed / "cpython-3.12.15-linux-x86_64-gnu"
    base.mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))

    def fake_run(cmd, **kwargs):
        assert cmd == ["uv", "python", "dir"]
        assert kwargs["encoding"] == "utf-8"  # uv prints UTF-8 on Windows too
        return subprocess.CompletedProcess(cmd, 0, f"{managed}\n", "")

    monkeypatch.setattr(uv_tool.subprocess, "run", fake_run)
    assert uv_tool.runs_on_managed_python("uv") is True
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "usr"))
    assert uv_tool.runs_on_managed_python("uv") is False


def test_runs_on_managed_python_false_when_uv_fails(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("no uv")

    monkeypatch.setattr(uv_tool.subprocess, "run", fake_run)
    assert uv_tool.runs_on_managed_python("uv") is False
