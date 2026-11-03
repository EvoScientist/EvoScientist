"""Tests for EvoScientist.uv_tool: receipt reading and the uv to run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from EvoScientist import paths, uv_tool

# The real functions; conftest stubs them for every other test.
_FIND_UV = uv_tool.find_uv
_KNOWS_SYSTEM_CERTS = uv_tool._knows_system_certs


@pytest.fixture(autouse=True)
def _real_uv_lookups(monkeypatch):
    monkeypatch.setattr(uv_tool, "find_uv", _FIND_UV)
    monkeypatch.setattr(uv_tool, "_knows_system_certs", _KNOWS_SYSTEM_CERTS)
    _KNOWS_SYSTEM_CERTS.cache_clear()


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


def test_several_own_entries_join_their_extras(tmp_path):
    """A receipt written with ``--with 'evoscientist[x]'`` stays readable:
    the extras-only entries join the one with the version."""
    receipt = uv_tool.read_receipt(
        _env(
            tmp_path,
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist", extras = ["telegram"] },\n'
            '  { name = "EvoScientist", extras = ["oauth"], specifier = "==0.3.6" },\n'
            '  { name = "six" },\n'
            "]\n",
        )
    )
    assert receipt.own == {
        "name": "EvoScientist",
        "extras": ["telegram", "oauth"],
        "specifier": "==0.3.6",
    }
    assert receipt.with_args() == ["--with", "six"]


def test_two_versioned_own_entries_raise(tmp_path):
    text = (
        "[tool]\nrequirements = [\n"
        '  { name = "evoscientist", specifier = "==0.3.6" },\n'
        '  { name = "evoscientist", extras = ["x"], specifier = ">=0.3" },\n'
        "]\n"
    )
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


def test_runs_on_managed_python_unknown_when_uv_fails(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("no uv")

    monkeypatch.setattr(uv_tool.subprocess, "run", fake_run)
    assert uv_tool.runs_on_managed_python("uv") is None


def test_runs_on_managed_python_unknown_for_uv_python_elsewhere(tmp_path, monkeypatch):
    """uv run with other settings than at install time names another folder."""
    base = tmp_path / "old-uv-python" / "cpython-3.12.15-linux-x86_64-gnu"
    base.mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(
        uv_tool.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, f"{tmp_path / 'uv-python'}\n", ""
        ),
    )
    assert uv_tool.runs_on_managed_python("uv") is None


def _help(text: str):
    return lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, text, "")


@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        ("      --system-certs\n", True),
        ("      --native-tls\n", False),  # uv before UV_SYSTEM_CERTS
        ("", True),  # cannot tell: the current name
    ],
)
def test_knows_system_certs(monkeypatch, help_text, expected):
    monkeypatch.setattr(uv_tool.subprocess, "run", _help(help_text))
    assert uv_tool._knows_system_certs("uv") is expected


def test_uv_environment_old_uv_gets_native_tls(monkeypatch):
    for name in ("UV_SYSTEM_CERTS", "UV_NATIVE_TLS", "UV_HTTP_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(uv_tool.subprocess, "run", _help("      --native-tls\n"))
    env = uv_tool.uv_environment("old-uv", "default")
    assert env["UV_NATIVE_TLS"] == "1"
    assert "UV_SYSTEM_CERTS" not in env
    assert env["UV_HTTP_TIMEOUT"] == "300"


def test_uv_environment_cn_mirror(monkeypatch):
    for name in ("UV_SYSTEM_CERTS", "UV_NATIVE_TLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(uv_tool.subprocess, "run", _help("      --system-certs\n"))
    env = uv_tool.uv_environment("uv", "cn")
    assert env["UV_DEFAULT_INDEX"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert env["UV_PYTHON_INSTALL_MIRROR"] == uv_tool.CN_PYTHON_MIRROR
    assert env["UV_SYSTEM_CERTS"] == "1"
    assert "UV_NATIVE_TLS" not in env


def test_uv_environment_keeps_user_settings(monkeypatch):
    monkeypatch.setenv("UV_NATIVE_TLS", "0")
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "30")
    monkeypatch.delenv("UV_SYSTEM_CERTS", raising=False)
    monkeypatch.delenv("UV_DEFAULT_INDEX", raising=False)

    def no_probe(cmd, **kw):
        raise AssertionError("no probe when the user chose")

    monkeypatch.setattr(uv_tool.subprocess, "run", no_probe)
    env = uv_tool.uv_environment("uv", "default")
    assert env["UV_NATIVE_TLS"] == "0"
    assert "UV_SYSTEM_CERTS" not in env
    assert env["UV_HTTP_TIMEOUT"] == "30"
    assert "UV_DEFAULT_INDEX" not in env
