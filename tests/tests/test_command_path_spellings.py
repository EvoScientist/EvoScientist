"""Quoted and attached paths name the same workspace files as bare paths."""

import shlex
import sys

import pytest

from EvoScientist.backends import _convert_command_paths, validate_command


@pytest.mark.parametrize(
    "spelling",
    [
        "/tmp/out.txt",
        '"/tmp/out.txt"',
        "'/tmp/out.txt'",
        "--output=/tmp/out.txt",
        "-o/tmp/out.txt",
        "OUT=/tmp/out.txt",
        '--output="/tmp/out.txt"',
    ],
)
def test_tmp_path_spellings_are_mapped(tmp_path, spelling):
    command, uses_tmp = _convert_command_paths(
        "echo " + spelling,
        workspace_root=str(tmp_path),
        workspace_name=tmp_path.name,
    )
    assert uses_tmp
    assert "/tmp/out.txt" not in command.replace("./tmp/out.txt", "")
    assert validate_command(command, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize(
    "spelling",
    [
        "--output=/Users/person/out.txt",
        "-o/Users/person/out.txt",
        "OUT=/Users/person/out.txt",
    ],
)
def test_attached_unmapped_system_paths_are_refused(tmp_path, spelling):
    command, _ = _convert_command_paths(
        "echo " + spelling, workspace_root=str(tmp_path)
    )
    assert validate_command(command, workspace_root=str(tmp_path)) is not None


def test_quoted_tmp_after_cd_uses_workspace_root(tmp_path):
    command, uses_tmp = _convert_command_paths(
        'cd sub && cat "/tmp/out.txt"', workspace_root=str(tmp_path)
    )
    assert uses_tmp
    assert str(tmp_path / "tmp/out.txt") in shlex.split(command)


@pytest.mark.parametrize(
    "spelling", ['"/etc/result.txt"', "--output=/etc/result.txt", "-o/etc/result.txt"]
)
def test_existing_system_named_workspace_file_maps(tmp_path, spelling):
    folder = tmp_path / "etc"
    folder.mkdir()
    (folder / "result.txt").write_text("kept")
    command, _ = _convert_command_paths("cat " + spelling, workspace_root=str(tmp_path))
    assert "./etc/result.txt" in command
    assert validate_command(command, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize(
    "command",
    [
        "echo https://example.com/tmp/a",
        "export PATH=/usr/local/bin:$PATH",
        'python -c "print(6 / 3)"',
    ],
)
def test_urls_code_and_path_search_list_keep_their_spelling(tmp_path, command):
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == command
    assert validate_command(converted, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize(
    "spelling",
    ['"/tmp/out.txt"', "--output=/tmp/out.txt", "-o/tmp/out.txt", "OUT=/tmp/out.txt"],
)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell assignments")
def test_execute_creates_workspace_tmp_and_file(tmp_path, spelling):
    from EvoScientist.backends import CustomSandboxBackend

    (tmp_path / "writer.py").write_text(
        "import os, sys\n"
        'arg = sys.argv[1] if len(sys.argv) > 1 else os.environ["OUT"]\n'
        'path = arg.split("=", 1)[-1] if "=" in arg else arg[2:] if arg.startswith("-o/") or arg.startswith("-o.") else arg\n'
        'open(path, "w").write("row")\n'
    )
    command = "python writer.py " + spelling
    if spelling.startswith("OUT="):
        command = spelling + " python writer.py"
    backend = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = backend.execute(command)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tmp/out.txt").read_text() == "row"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell assignments")
def test_quoted_tmp_variables_are_not_frozen_after_cd(tmp_path):
    from EvoScientist.backends import CustomSandboxBackend

    (tmp_path / "sub").mkdir()
    backend = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = backend.execute('cd sub && FILE=out.txt; echo row > "/tmp/$FILE"')
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tmp/out.txt").read_text().strip() == "row"
    assert not (tmp_path / "sub/tmp").exists()
