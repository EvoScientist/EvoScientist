"""Quoted and attached paths name the same workspace files as bare paths."""

import json
import shlex
import sys
from pathlib import Path

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
    tokens = shlex.split(command, posix=sys.platform != "win32")
    assert any(
        Path(token.strip('"')) == tmp_path / "tmp" / "out.txt" for token in tokens
    )


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


@pytest.mark.parametrize("prefix", ["--output=", "-o"])
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


@pytest.mark.parametrize("prefix", ["--out=", "-o"])
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


def _assignment_root(tmp_path):
    """How an assignment's value spells the workspace: its path, or ``.`` when
    the path needs quoting."""
    from EvoScientist.backends import _platform_quote

    root = _platform_quote(str(tmp_path))
    return "." if root[0] in "'\"" else root


@pytest.mark.parametrize("quote", ["", '"', "'"])
def test_path_named_file_variable_tmp_value_is_mapped(tmp_path, quote):
    from EvoScientist.backends import prepare_sandbox_command

    command, error = prepare_sandbox_command(
        f"OUTPUT_PATH={quote}/tmp/out.csv{quote} python train.py", str(tmp_path)
    )
    assert error is None
    root = _assignment_root(tmp_path)
    assert command == f"OUTPUT_PATH={root}/tmp/out.csv python train.py"


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


@pytest.mark.parametrize("prefix", ["--output=", "-o", "OUT="])
def test_attached_tmp_after_cd_uses_workspace_root(tmp_path, prefix):
    from EvoScientist.backends import _platform_quote

    command, uses_tmp = _convert_command_paths(
        f"cd sub && echo {prefix}/tmp/out.txt", workspace_root=str(tmp_path)
    )
    assert uses_tmp
    assert command == (
        f"cd sub && echo {prefix}{_platform_quote(str(tmp_path))}/tmp/out.txt"
    )


@pytest.mark.parametrize(
    "command",
    [
        "OUT=/tmp/x; cd sub && echo $OUT",
        'export OUT=/tmp/x && cd sub && echo "$OUT"',
        "OUT=/tmp/x python w.py",
        'OUT=/tmp/"x"; cd sub',
        "make -C src DESTDIR=/tmp/x install",
        "DESTDIR=/tmp/x make -C src install",
        "env DESTDIR=/tmp/x make -C src install",
        "OUT=/tmp/x sh -c 'cd sub && echo $OUT'",
    ],
)
def test_assigned_tmp_names_the_workspace_tmp(tmp_path, command):
    # A variable can be read after a cd or in another folder (make -C).
    from EvoScientist.backends import prepare_sandbox_command

    converted, uses_tmp = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert uses_tmp
    assert converted == command.replace("=/tmp/", f"={_assignment_root(tmp_path)}/tmp/")
    assert prepare_sandbox_command(command, str(tmp_path)) == (converted, None)


@pytest.mark.parametrize(
    "command", ['OUT="/tmp/x"', 'export "OUT=/tmp/x"', "export 'OUT=/tmp/x'"]
)
def test_quoted_assigned_tmp_names_the_workspace_tmp(tmp_path, command):
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    expected = command.split("OUT=")[0].rstrip("'\"")
    assert converted == f"{expected}OUT={_assignment_root(tmp_path)}/tmp/x"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('OUT=/tmp/"out file"; cd sub', 'OUT={root}/tmp/"out file"; cd sub'),
        ("OUT=/results/x.csv; cd sub", "OUT={root}/results/x.csv; cd sub"),
    ],
)
def test_assigned_value_spellings_name_the_workspace(tmp_path, command, expected):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/x.csv").touch()
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == expected.format(root=_assignment_root(tmp_path))


def test_assigned_tmp_stays_relative_when_the_root_needs_quoting(tmp_path):
    # An unquoted $OUT would split at the space in the root.
    root = tmp_path / "my ws"
    root.mkdir()
    converted, uses_tmp = _convert_command_paths(
        "OUT=/tmp/x; mkdir -p $OUT", workspace_root=str(root)
    )
    assert uses_tmp
    assert converted == "OUT=./tmp/x; mkdir -p $OUT"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell execution")
def test_execute_assigned_tmp_path_used_after_cd(tmp_path):
    from EvoScientist.backends import CustomSandboxBackend

    if _assignment_root(tmp_path) == ".":
        pytest.skip("a temp folder that needs quoting keeps ./")
    (tmp_path / "sub").mkdir()
    result = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True).execute(
        "OUT=/tmp/x; cd sub && echo payload > $OUT"
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tmp" / "x").read_text() == "payload\n"
    assert not (tmp_path / "sub" / "tmp").exists()


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "CUDA_HOME=/usr/local/cuda; python train.py > /var/tmp/log"',
        'bash -euo pipefail -c "CUDA_HOME=/usr/local/cuda; python train.py > /var/tmp/log"',
        'bash -o pipefail -c "PATH=/usr/bin:/bin; cat /etc/passwd"',
        'bash -O extglob -c "PATH=/usr/bin; cat /etc/passwd"',
        'eval "CUDA_HOME=/usr/local/cuda; cat /etc/passwd"',
        'su -c "PATH=/usr/bin; cat /etc/passwd"',
        'flock f -c "PATH=/usr/bin; cat /etc/passwd"',
        'watch "CUDA_HOME=/usr/local/cuda; cat /etc/passwd"',
        'ksh -c "PATH=/usr/bin; cat /etc/passwd"',
        'eval "CC=/usr/bin/gcc -m64; cat /etc/passwd"',
        'eval "CC=/usr/bin/gcc|cat /etc/passwd"',
        'eval "CC=/usr/bin/gcc&&cat /etc/passwd"',
        'bash -c "PATH=/usr/bin:/bin; cat /etc/passwd"',
        "sh -c 'X=/usr/x; cat /etc/passwd'",
        'bash -lc "PATH=/usr/bin:/bin; python train.py --config=/etc/x"',
        'bash -c "python x.py --config=/etc/x"',
        'bash -c "./configure --prefix=/opt/x && make"',
        "bash -c 'sh -c \"X=/usr/x; cat /etc/passwd\"'",
    ],
)
def test_shell_script_assignments_do_not_hide_file_operands(tmp_path, command):
    from EvoScientist.backends import prepare_sandbox_command

    assert prepare_sandbox_command(command, str(tmp_path))[1] is not None


@pytest.mark.parametrize(
    "command",
    [
        'make "CC=/usr/bin/clang -O3"',
        'make "CC=/usr/bin/gcc -m64 -std=c11"',
        'bash -euo pipefail -c "CUDA_HOME=/usr/local/cuda python setup.py build"',
        'CC="/usr/bin/clang -O3" make',
        'cmake -DCMAKE_PREFIX_PATH="/opt/a;/opt/b" .',
        'bash -c "CUDA_HOME=/usr/local/cuda python setup.py build"',
        'bash -c "CUDA_HOME=/usr/local/cuda; ls $CUDA_HOME/bin"',
        'bash -c "CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; python setup.py install"',
    ],
)
def test_shell_script_toolchain_settings_remain_supported(tmp_path, command):
    from EvoScientist.backends import prepare_sandbox_command

    converted, error = prepare_sandbox_command(command, str(tmp_path))
    assert error is None
    assert converted == command


@pytest.mark.parametrize("quote", ["", '"'])
def test_search_list_substitution_tmp_paths_are_mapped(tmp_path, quote):
    from EvoScientist.backends import prepare_sandbox_command

    command, error = prepare_sandbox_command(
        f"export PYTHONPATH={quote}$(cat /tmp/paths.txt){quote}", str(tmp_path)
    )
    assert error is None
    assert command == f"export PYTHONPATH={quote}$(cat ./tmp/paths.txt){quote}"
    assert (tmp_path / "tmp").is_dir()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('scp user@host:"/tmp/log.txt" .', 'scp user@host:"/tmp/log.txt" .'),
        ('kubectl cp pod:"/tmp/a.csv" a.csv', 'kubectl cp pod:"/tmp/a.csv" a.csv'),
        (
            'curl -F file=@"/tmp/plot.png" host',
            "curl -F file=@{root}/tmp/plot.png host",
        ),
    ],
)
def test_only_a_quoted_span_after_a_colon_is_kept(tmp_path, command, expected):
    converted, _ = _convert_command_paths(command, workspace_root=str(tmp_path))
    assert converted == expected.format(root=_assignment_root(tmp_path))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host-path spelling")
def test_quoted_host_path_into_workspace_skills_stays_protected(tmp_path):
    from EvoScientist.backends import prepare_sandbox_command

    (tmp_path / "skills/demo").mkdir(parents=True)
    _command, error = prepare_sandbox_command(
        f'rm -rf "{tmp_path}/skills/demo"', tmp_path, skills_dir=tmp_path / "skills"
    )
    assert error is not None


@pytest.mark.parametrize(
    "syntax", ['"$(cat /tmp/paths.txt)"', '"`cat /tmp/paths.txt`"']
)
def test_quoted_search_list_substitution_after_cd_uses_workspace_root(tmp_path, syntax):
    from EvoScientist.backends import _platform_quote, prepare_sandbox_command

    converted, error = prepare_sandbox_command(
        f"cd sub && export PYTHONPATH={syntax}", str(tmp_path)
    )
    assert error is None
    expected = syntax.replace(
        "/tmp/paths.txt", _platform_quote(str(tmp_path)) + "/tmp/paths.txt"
    )
    assert converted == f"cd sub && export PYTHONPATH={expected}"


def test_quoted_search_list_keeps_literal_entries_while_rewriting_substitution(
    tmp_path,
):
    from EvoScientist.backends import _platform_quote, prepare_sandbox_command

    converted, error = prepare_sandbox_command(
        'PATH="$(cd /tmp/x && pwd):$PATH" tool', str(tmp_path)
    )
    assert error is None
    assert (
        converted
        == f'PATH="$(cd {_platform_quote(str(tmp_path))}/tmp/x && pwd):$PATH" tool'
    )


@pytest.mark.parametrize(
    "options",
    [
        "-o pipefail",
        "-O extglob",
        "-euo pipefail",
        "+o pipefail",
        "-o pipefail -O extglob",
    ],
)
def test_named_shell_options_do_not_hide_script_paths(tmp_path, options):
    from EvoScientist.backends import prepare_sandbox_command

    command, error = prepare_sandbox_command(
        f'bash {options} -c "cat /tmp/a.txt"', str(tmp_path)
    )
    assert error is None
    assert command == f'bash {options} -c "cat ./tmp/a.txt"'
    assert (tmp_path / "tmp").is_dir()


@pytest.mark.parametrize(
    "operand",
    [
        "/tmp/x",
        '"/tmp/x"',
        "--output=/tmp/x",
        "-o/tmp/x",
        "OUT=/tmp/x",
        "/data.csv",
        "/skills/demo/SKILL.md",
    ],
)
def test_mapped_paths_after_cd_allow_a_spaced_workspace_root(tmp_path, operand):
    from EvoScientist.backends import prepare_sandbox_command

    root = tmp_path / "my project"
    root.mkdir()
    (root / "data.csv").write_text("row")
    converted, error = prepare_sandbox_command(f"cd sub && cat {operand}", str(root))
    assert error is None
    assert str(root).replace("\\", "/") in converted.replace('"', "").replace(
        "'", ""
    ).replace("\\", "/")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host-path quoting")
@pytest.mark.parametrize("quote", ["'", '"'])
def test_spaced_workspace_does_not_hide_later_system_paths(tmp_path, quote):
    from EvoScientist.backends import prepare_sandbox_command

    root = tmp_path / "my project"
    root.mkdir()
    command = f"eval {quote}{root}/ok; cat /etc/passwd{quote}"
    assert prepare_sandbox_command(command, str(root))[1] is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host-path quoting")
def test_spaced_workspace_keeps_substitution_operands_checked(tmp_path):
    from EvoScientist.backends import prepare_sandbox_command

    root = tmp_path / "my project"
    root.mkdir()
    assert (
        prepare_sandbox_command(f'cat "{root}/$(cat /etc/passwd)"', str(root))[1]
        is not None
    )


@pytest.mark.parametrize("operand", ["/tmp/../outside", '"/tmp/../outside"'])
def test_spaced_workspace_keeps_parent_components_refused(tmp_path, operand):
    from EvoScientist.backends import prepare_sandbox_command

    root = tmp_path / "my project"
    root.mkdir()
    assert prepare_sandbox_command(f"cd sub && cat {operand}", str(root))[1] is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize("root_name", ["ws", "my project", "Project (2)"])
@pytest.mark.parametrize(
    "command",
    [
        "cd sub && python x.py --files=/tmp/a.csv,/home/u/data.csv",
        "cd sub && python x.py --files=/tmp/a.csv,/tmp/b.csv",
        "cd sub && python x.py --files=/tmp/a.csv:/etc/passwd",
        "cd sub && python x.py --map=/tmp/a.csv=/etc/passwd",
        "python x.py --files=/memories/a:/etc/passwd",
        "cd sub && eval \"/tmp/tool '/etc/passwd'\"",
        "python x.py --files=/memories/a,/etc/passwd",
        "cd sub && python x.py --files=/memories/a,/etc/passwd",
    ],
)
def test_allowed_path_word_does_not_hide_a_later_system_path(
    tmp_path, monkeypatch, root_name, command
):
    from EvoScientist import paths
    from EvoScientist.backends import prepare_sandbox_command

    monkeypatch.setattr(paths, "MEMORIES_DIR", tmp_path / "memories")
    root = tmp_path / root_name
    root.mkdir()
    error = prepare_sandbox_command(command, str(root))[1]
    assert error is not None
    assert "absolute system path" in error


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize("root_name", ["ws", "my project", "Project (2)"])
@pytest.mark.parametrize(
    "command",
    [
        "cd sub && python x.py --files=/tmp/a.csv",
        "cd sub && python x.py --files=/data/a.csv,/data/b.csv",
        "cd sub && cat /tmp/{a,b}/x.csv",
        "python x.py --files=/memories/a,/memories/b",
        "cd sub && python x.py --files=/data/a.csv:/data/b.csv",
        "cd sub && cat /tmp/a.csv",
        "cd sub && echo {1..3} /tmp/{a,b}.csv",
    ],
)
def test_allowed_path_word_with_only_workspace_paths_still_runs(
    tmp_path, monkeypatch, root_name, command
):
    from EvoScientist import paths
    from EvoScientist.backends import prepare_sandbox_command

    monkeypatch.setattr(paths, "MEMORIES_DIR", tmp_path / "memories")
    root = tmp_path / root_name
    root.mkdir()
    assert prepare_sandbox_command(command, str(root))[1] is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize("root_name", ["c++", "data [v2]", "run=1", "a:b"])
def test_workspace_name_ending_in_a_symbol_keeps_its_own_paths(tmp_path, root_name):
    from EvoScientist.backends import prepare_sandbox_command

    root = tmp_path / root_name
    root.mkdir()
    assert prepare_sandbox_command("cd sub && cat /tmp/a.csv", str(root))[1] is None
    error = prepare_sandbox_command(
        "cd sub && python x.py --files=/tmp/a.csv,/etc/passwd", str(root)
    )[1]
    assert "/etc/passwd" in error


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize("folder", ["Project (2)", "c++", "data [v2]", "x)"])
def test_folder_name_ending_in_a_symbol_inside_workspace_runs(tmp_path, folder):
    from EvoScientist.backends import prepare_sandbox_command, validate_command

    root = tmp_path / "ws"
    root.mkdir()
    for rest in ("tmp/a.csv", "usr/lib.a", "etc/x"):
        command = f'cd sub && cat "/tmp/{folder}/{rest}"'
        assert prepare_sandbox_command(command, str(root))[1] is None
        command = f'cat "{root}/{folder}/{rest}"'
        assert validate_command(command, allow_prefixes=(str(root),)) is None
    for sep in (",", ":", "=", "@", "(", " '"):
        command = f'cd sub && python x.py --files="/tmp/{folder}/a.csv{sep}/etc/passwd"'
        assert "/etc/passwd" in prepare_sandbox_command(command, str(root))[1]


@pytest.mark.parametrize(
    "command",
    [
        "cat /tmp/{..,a}/{..,b}/.ssh/id_rsa",
        "cat ./{a,..}/x",
        "cat '{..,a}'/x",
        "ls {a,..,b}",
        # `..` built from a choice and the text next to it, or nested.
        "cat /tmp/.{.,a}/etc/passwd",
        "cat /tmp/{.,a}./etc/passwd",
        "cat /tmp/{a,{b,..}}/x",
        "cat /tmp/.{-../}/etc/passwd",
        # Glued to an option or assignment (checked per expanded word).
        "make CFLAGS=-I{..,x}/include",
        "env PYTHONPATH=a:{..,b}/lib python x.py",
        "gcc -I{a,..}/include x.c",
    ],
)
def test_brace_choice_of_parent_folder_is_traversal(command, tmp_path):
    from EvoScientist.backends import validate_command

    error = validate_command(command, allow_prefixes=(str(tmp_path),))
    assert error is not None
    assert "'..' path traversal" in error


_JSON_ROWS = json.dumps([{"id": i, "v": i / 2} for i in range(13)])
_JSON_POINTS = json.dumps([{"x": i, "y": i, "z": i} for i in range(8)])


@pytest.mark.parametrize(
    "command",
    [
        "echo {1..5}",
        "cat ./{a,b}..txt",
        "ls ./{a..c}/x",
        "cat a{..,b}",
        "for i in {1..100000}; do echo $i; done",
        "cat ./file{1,2}.txt",
        "cat ./{{a,b},c}/x",
        # JSON arguments: many `{...}` but no `.` next to a brace.
        f"curl -s -d '{_JSON_ROWS}' https://example.com/api",
        f"python plot.py --points '{_JSON_POINTS}'",
    ],
)
def test_brace_lists_without_a_parent_folder_are_no_traversal(command):
    from EvoScientist.backends import _has_traversal_component

    assert not _has_traversal_component(command)


@pytest.mark.parametrize(
    "word",
    [
        "{\x00..\U0010ffff}",  # a character range too big to expand
        "a{b,.c}" * 20,  # 2**20 words
        "{" * 50_000 + "}" * 50_000 + ".",  # too many braces to scan
        "x" * 70_000 + "{a,.b}",  # too long to expand
    ],
)
def test_brace_words_too_big_to_expand_are_refused(word):
    import time

    from EvoScientist.backends import _has_traversal_component

    started = time.monotonic()
    assert _has_traversal_component(f"cat {word}")
    assert time.monotonic() - started < 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize(
    ("files", "allowed"),
    [
        ("/tmp/my project/a.csv,/tmp/my project/b.csv", True),
        ("/tmp/my project/a.csv,/tmp/my project/b.csv,/etc/passwd", False),
        ("/tmp/my project/a.csv,/etc/passwd,/tmp/my project/b.csv", False),
        ("/tmp/my project/a.csv,/tmp/my projectX/b.csv", False),
        ("/tmp/my project/a.csv,/dev/null", True),
        ("/tmp/my project/a.csv,/dev/null,/etc/passwd", False),
        ('/tmp/my project/a.csv,/e""tc/passwd', False),
        ('/tmp/my project/a.csv,/h""ome/u/b.csv', False),
        ("/tmp/my project/a.csv:/etc/passwd", False),
        ("/tmp/my project/a.csv=/etc/passwd", False),
        ("/tmp/my project/a.csv:/tmp/my project/b.csv", True),
    ],
)
def test_allowed_path_word_may_repeat_an_allowed_spaced_folder(files, allowed):
    from EvoScientist.backends import validate_command

    error = validate_command(
        f'python x.py --files="{files}"', allow_prefixes=("/tmp/my project",)
    )
    assert (error is None) is allowed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize(
    "command",
    [
        'cmake -DCMAKE_PREFIX_PATH="/opt/ws/x;/usr/local" -S src',
        'make "LIBDIR=/opt/ws/lib;/usr/lib"',
    ],
)
def test_toolchain_setting_under_allowed_folder_keeps_later_entries(command):
    from EvoScientist.backends import validate_command

    assert validate_command(command, allow_prefixes=("/opt/ws",)) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX host paths")
@pytest.mark.parametrize("later", ['/e""tc/passwd', '/et""c/shadow', '/h""ome/u/b.csv'])
def test_quote_split_system_path_after_allowed_folder_is_refused(later):
    from EvoScientist.backends import validate_command

    command = f'python x.py --files="/tmp/work/a.csv,{later}"'
    assert validate_command(command, allow_prefixes=("/tmp/work",)) is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell execution")
def test_execute_mapped_path_after_cd_in_spaced_workspace(tmp_path):
    from EvoScientist.backends import CustomSandboxBackend

    root = tmp_path / "my project"
    (root / "sub").mkdir(parents=True)
    (root / "tmp").mkdir()
    (root / "tmp" / "x").write_text("expected payload")
    result = CustomSandboxBackend(root_dir=str(root), virtual_mode=True).execute(
        "cd sub && cat /tmp/x"
    )
    assert result.exit_code == 0, result.output
    assert "expected payload" in result.output


@pytest.mark.parametrize("uses_bash", [False, True])
@pytest.mark.parametrize(
    "operand", ["/tmp/x", '"/tmp/x"', "--output=/tmp/x", "-o/tmp/x", "OUT=/tmp/x"]
)
def test_windows_spaced_root_mapped_words_are_allowed(monkeypatch, uses_bash, operand):
    from EvoScientist import agent_shell, backends

    root = r"C:\Users\me\my project"
    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    monkeypatch.setattr(agent_shell, "uses_bash", lambda: uses_bash)
    converted, uses_tmp = backends._convert_command_paths(
        f"cd sub && cat {operand}", workspace_root=root
    )
    assert uses_tmp
    assert backends.validate_command(converted, (root,), workspace_root=root) is None
    assert "C:/Users/me/my project" in converted.replace("\\", "/")


@pytest.mark.parametrize("uses_bash", [False, True])
@pytest.mark.parametrize(
    "suffix", ["/../outside", "; cat /etc/passwd", "$(cat /etc/passwd)"]
)
def test_windows_spaced_root_does_not_hide_unsafe_words(monkeypatch, uses_bash, suffix):
    from EvoScientist import agent_shell, backends

    root = r"C:\Users\me\my project"
    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    monkeypatch.setattr(agent_shell, "uses_bash", lambda: uses_bash)
    command = f'cat "C:/Users/me/my project{suffix}"'
    assert backends.validate_command(command, (root,), workspace_root=root) is not None


@pytest.mark.parametrize(
    "root", [r"C:\Users\me\my project_evil", r"D:\Users\me\my project"]
)
def test_windows_workspace_exemption_keeps_drive_and_component_boundaries(
    monkeypatch, root
):
    from EvoScientist import backends

    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    allowed = r"C:\Users\me\my project"
    assert (
        backends.validate_command(
            f'cat "{root}"/tmp/x', (allowed,), workspace_root=allowed
        )
        is not None
    )


@pytest.mark.parametrize("uses_bash", [False, True])
@pytest.mark.parametrize("prefix", ["", "--output=", "-o", "OUT="])
def test_windows_spaced_root_normalizes_case_and_separators(
    monkeypatch, uses_bash, prefix
):
    from EvoScientist import agent_shell, backends

    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    monkeypatch.setattr(agent_shell, "uses_bash", lambda: uses_bash)
    root = r"C:\Users\me\my project"
    command = f'cat {prefix}"c:/users/ME/my project"/tmp/x'
    assert backends.validate_command(command, (root,), workspace_root=root) is None
