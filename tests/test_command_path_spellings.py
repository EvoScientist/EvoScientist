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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell quoting")
def test_execute_creates_workspace_tmp_and_file(tmp_path):
    from EvoScientist.backends import CustomSandboxBackend

    (tmp_path / "writer.py").write_text(
        'import sys\nopen(sys.argv[1], "w").write("row")\n'
    )
    backend = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = backend.execute('python writer.py "/tmp/out.txt"')
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tmp/out.txt").read_text() == "row"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell assignments")
def test_quoted_tmp_variables_are_refused_instead_of_escaping_validation(tmp_path):
    from EvoScientist.backends import CustomSandboxBackend

    (tmp_path / "sub").mkdir()
    backend = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = backend.execute('cd sub && FILE=out.txt; echo row > "/tmp/$FILE"')
    assert result.exit_code != 0
    assert not (tmp_path / "tmp/out.txt").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host-path spelling")
@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_host_workspace_path_is_not_virtual_tmp(tmp_path, quote):
    path = str(tmp_path / "out.txt")
    converted, uses_tmp = _convert_command_paths(
        f"cat {quote}{path}{quote}", workspace_root=str(tmp_path)
    )
    assert not uses_tmp
    assert shlex.split(converted) == ["cat", "./out.txt"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX literal backslash semantics")
@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_path_retains_literal_backslash(tmp_path, quote):
    converted, _ = _convert_command_paths(
        f"cat {quote}/tmp/a\\b{quote}", workspace_root=str(tmp_path)
    )
    assert shlex.split(converted) == ["cat", "./tmp/a\\b"]


def test_quoted_path_search_list_is_untouched_and_valid(tmp_path):
    command = 'export PATH="/usr/bin:/bin:$PATH"'
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == command
    assert validate_command(converted, workspace_root=str(tmp_path)) is None


def test_wholly_quoted_option_path_is_mapped(tmp_path):
    converted, uses_tmp = _convert_command_paths(
        'echo "--output=/tmp/file"', workspace_root=str(tmp_path)
    )
    assert uses_tmp
    assert shlex.split(converted) == ["echo", "--output=./tmp/file"]


def test_quoted_null_device_is_not_shadowed(tmp_path):
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/null").touch()
    converted, _ = _convert_command_paths(
        'cat "/dev/null"', workspace_root=str(tmp_path)
    )
    assert shlex.split(converted) == ["cat", "/dev/null"]


def test_expanding_tmp_path_is_not_rewritten_past_validation(tmp_path):
    converted, _ = _convert_command_paths(
        'cat "/tmp/$ESCAPE/file"', workspace_root=str(tmp_path)
    )
    assert validate_command(converted, workspace_root=str(tmp_path)) is not None


@pytest.mark.parametrize(
    "command",
    [
        'PATH="$(cat /etc/passwd >&2)"',
        'PATH="`cat /etc/passwd >&2`"',
        'export PATH="/usr/bin:$(cat /etc/passwd >&2)"',
        'PATH="$(cat --input=/etc/passwd)"',
    ],
)
def test_path_assignment_substitutions_cannot_hide_host_paths(tmp_path, command):
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert validate_command(converted, workspace_root=str(tmp_path)) is not None


@pytest.mark.parametrize("prefix", ["--output=", "-o", "OUT="])
def test_attached_mixed_quoted_path_maps_its_unquoted_head(tmp_path, prefix):
    command, uses_tmp = _convert_command_paths(
        f'echo {prefix}/tmp/"out file"', workspace_root=str(tmp_path)
    )
    assert uses_tmp
    assert shlex.split(command) == ["echo", prefix + "./tmp/out file"]
    assert validate_command(command, workspace_root=str(tmp_path)) is None


def test_path_search_list_with_safe_substitution_remains_valid(tmp_path):
    command = 'export PATH="/usr/bin:/bin:$(printf local)"'
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == command
    assert validate_command(converted, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize(
    "command",
    [
        "export PATH='/usr/bin:$(cat /etc/passwd)'",
        'export PATH="/usr/bin:\\$(cat /etc/passwd)"',
    ],
)
def test_literal_path_search_text_is_not_a_substitution(tmp_path, command):
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == command
    assert validate_command(converted, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize("prefix", ["", "--output=", "-o", "OUT="])
@pytest.mark.parametrize("quote", ['"', "'"])
def test_mixed_quoted_traversal_cannot_bypass_confinement(tmp_path, prefix, quote):
    command = f"echo {prefix}/tmp/{quote}..{quote}/{quote}..{quote}/canary"
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert validate_command(converted, workspace_root=str(tmp_path)) is not None
    assert validate_command(command, workspace_root=str(tmp_path)) is not None


def test_mixed_quoted_assignment_cannot_read_outside_workspace(tmp_path):
    from EvoScientist.backends import prepare_sandbox_command

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _command, error = prepare_sandbox_command(
        'F=/tmp/".."/".."/canary; cat "$F"', str(workspace)
    )
    assert error is not None
    assert not (workspace / "tmp").exists()


@pytest.mark.parametrize("component", [r"\.\.", "'.'.", '."."', "..''"])
def test_shell_escaped_traversal_components_are_blocked(tmp_path, component):
    command = f"OUT=/tmp/{component}/{component}/canary"
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert validate_command(converted, workspace_root=str(tmp_path)) is not None


@pytest.mark.parametrize(
    "command",
    [
        "awk -F/ '{print $1}' f",
        "cut -d/ -f2 paths.txt",
        'IFS=/ read -r a b <<< "$p"',
        "SEP=/ python x.py",
        "sed -e/foo/d f",
        "uvicorn app:app --root-path=/api",
        "rsync -a --exclude=/build src/ dst/",
        'ENDPOINT=/api/v1; curl "localhost:8000$ENDPOINT"',
        'uvicorn app:app --root-path=/api/"v1"',
    ],
)
def test_attached_non_file_values_are_not_rewritten(tmp_path, command):
    converted, uses_tmp = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == command
    assert not uses_tmp
    assert validate_command(converted, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize(
    "name", ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CPATH", "PKG_CONFIG_PATH"]
)
@pytest.mark.parametrize("quote", ["", '"', "'"])
def test_search_list_assignments_remain_literal(tmp_path, name, quote):
    command = f"export {name}={quote}/tmp/a:/usr/lib/x{quote}"
    converted, uses_tmp = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == command
    assert not uses_tmp
    assert validate_command(converted, workspace_root=str(tmp_path)) is None


@pytest.mark.parametrize("name", ["PYTHONPATH", "LD_LIBRARY_PATH"])
@pytest.mark.parametrize("syntax", ['"$(cat /etc/passwd)"', '"`cat /etc/passwd`"'])
def test_search_list_substitutions_do_not_hide_paths(tmp_path, name, syntax):
    command = f"export {name}={syntax}"
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert validate_command(converted, workspace_root=str(tmp_path)) is not None


@pytest.mark.parametrize("prefix", ["--out=", "OUT=", "-o"])
def test_attached_existing_non_system_workspace_file_is_mapped(tmp_path, prefix):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/x.csv").touch()
    converted, _ = _convert_command_paths(
        f"echo {prefix}/results/x.csv", workspace_root=str(tmp_path)
    )
    assert shlex.split(converted) == ["echo", prefix + "./results/x.csv"]


@pytest.mark.parametrize(
    "command",
    [
        'MODEL_PATH="/home/alice/models/x.pt"; python eval.py',
        'export OUTPUT_PATH="/etc/cron.d/job"; echo x > "$OUTPUT_PATH"',
        'KEYPATH="/root/.ssh/id_rsa"; cat $KEYPATH',
        "SAVE_PATH=/home/alice/out; python x.py --save $SAVE_PATH",
        'CONFIG_PATH=/etc/passwd; cat "$CONFIG_PATH"',
    ],
)
def test_path_named_file_variable_is_validated(tmp_path, command):
    from EvoScientist.backends import prepare_sandbox_command

    assert prepare_sandbox_command(command, str(tmp_path))[1] is not None


@pytest.mark.parametrize("quote", ["", '"', "'"])
def test_path_named_file_variable_tmp_value_is_mapped(tmp_path, quote):
    from EvoScientist.backends import prepare_sandbox_command

    command, error = prepare_sandbox_command(
        f"OUTPUT_PATH={quote}/tmp/out.csv{quote} python train.py", str(tmp_path)
    )
    assert error is None
    assert shlex.split(command)[0] == "OUTPUT_PATH=./tmp/out.csv"


@pytest.mark.parametrize(
    "command",
    [
        "export CUDA_HOME=/usr/local/cuda",
        "CUDA_HOME=/usr/local/cuda pip install flash-attn",
        "export JAVA_HOME=/usr/lib/jvm/java-17",
        "make CC=/usr/bin/gcc",
        "env CC=/usr/bin/gcc make",
        "cmake -DCMAKE_C_COMPILER=/usr/bin/gcc .",
        "cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda .",
        'export CUDA_HOME="/usr/local/cuda"',
        "make CC=/opt/toolchain/gcc",
        "make CC=/bin/cc",
        "env TOOL=/sbin/tool build",
    ],
)
def test_literal_toolchain_assignments_preserve_compatibility(tmp_path, command):
    from EvoScientist.backends import prepare_sandbox_command

    converted, error = prepare_sandbox_command(command, str(tmp_path))
    assert error is None
    assert converted == command


@pytest.mark.parametrize(
    "command",
    [
        "X=/etc/passwd; cat $X",
        "OUT=/etc/x; echo hi > $OUT",
        "KEY_HOME=/root/.ssh; cat $KEY_HOME",
        "X=/usr/../etc/passwd; cat $X",
        'X="/usr/$(cat /etc/passwd)"; echo "$X"',
        "echo --output=/usr/local/x",
        "ls /usr/local/cuda",
    ],
)
def test_toolchain_assignment_rule_keeps_other_paths_refused(tmp_path, command):
    from EvoScientist.backends import prepare_sandbox_command

    assert prepare_sandbox_command(command, str(tmp_path))[1] is not None


@pytest.mark.parametrize(
    "name",
    [
        "LIBRARY_PATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "MANPATH",
        "CLASSPATH",
        "NODE_PATH",
    ],
)
def test_other_known_search_lists_are_not_file_variables(tmp_path, name):
    from EvoScientist.backends import prepare_sandbox_command

    original = f"{name}=/usr/a:/opt/b python x.py"
    converted, error = prepare_sandbox_command(original, str(tmp_path))
    assert error is None
    assert converted == original


@pytest.mark.parametrize(
    "command",
    [
        "OUT=/usr/local/x; echo hi > $OUT",
        "CUDA_HOME=/usr/local/cuda; ls $CUDA_HOME/bin",
    ],
)
def test_toolchain_compatibility_is_not_a_read_only_guarantee(tmp_path, command):
    from EvoScientist.backends import prepare_sandbox_command

    # Existing variable expansion is not tracked. This compatibility exception
    # therefore does not provide read-only isolation for toolchain locations.
    assert prepare_sandbox_command(command, str(tmp_path))[1] is None


@pytest.mark.parametrize(
    "raw", ["X=/usr_evil/file", "--output=/usr/x", "X=/optical/file"]
)
def test_toolchain_assignment_exemption_is_boundary_aware(raw):
    from EvoScientist.backends import _is_toolchain_path_assignment

    assert not _is_toolchain_path_assignment(raw)
