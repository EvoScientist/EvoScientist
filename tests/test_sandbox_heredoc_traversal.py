"""Traversal in code a shell reads from a here-document on stdin (issue #848).

A quoted or escaped ``..`` path in such a body ran even though the same line
in a direct command was refused. The body is code only when a shell reads it;
on ``cat`` the same body is data.
"""

import time

import pytest

from EvoScientist import backends, shell_syntax
from EvoScientist.backends import validate_command


class TestTraversalInShellCodeBodies:
    @pytest.mark.parametrize(
        "command",
        [
            "bash <<'EOF'\npython s.py --output='../out.txt'\nEOF",
            'bash <<EOF\ncd "../x"\nEOF',
            "bash -s <<EOF\nOUT='../x' python s.py\nEOF",
            "bash <<EOF\ncd ..\\/x\nEOF",
            "bash <<'EOF'\ncd ../x\nEOF",
            "bash <<'EOF'\npython s.py --output=../out.txt\nEOF",
        ],
    )
    def test_traversal_in_a_heredoc_the_shell_runs_is_blocked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF'\npython s.py --output='../out.txt'\nEOF",
            'cat <<EOF\ncd "../x"\nEOF',
            "cat <<'EOF'\nOUT='../x' python s.py\nEOF",
            "cat <<'EOF'\ncd ..\\/x\nEOF",
            "cat <<'EOF'\nRows: $(wc -l < /data/a.csv)\nEOF",
            "python3 - <<'EOF'\nprint(6 / 3)\nEOF",
        ],
    )
    def test_traversal_in_a_heredoc_without_a_shell_stays_data(self, command):
        assert validate_command(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            "env bash <<'EOF'\ncd '../x'\nEOF",
            "nohup bash <<EOF\ncd '../x'\nEOF",
            "setsid bash <<'EOF'\ncd '../x'\nEOF",
            "timeout 5 bash <<'EOF'\ncd '../x'\nEOF",
            "xargs bash -c <<'EOF'\ncd '../x'\nEOF",
        ],
    )
    def test_traversal_in_a_wrapped_shell_heredoc_is_blocked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    def test_traversal_in_a_nested_wrapped_shell_heredoc_is_blocked(self):
        command = "bash -c 'env bash <<EOF\ncd \"../x\"\nEOF'"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    def test_traversal_in_a_tab_stripped_heredoc_is_blocked(self):
        command = "bash <<-EOF\n\tcd '../x'\nEOF"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "sh <<'EOF'\ncd '../x'\nEOF",
            "sh -c \"bash <<'X'\ncd '../x'\nX\"",
        ],
    )
    def test_traversal_in_a_sh_heredoc_is_blocked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    def test_traversal_in_a_shell_heredoc_on_a_pipeline_is_blocked(self):
        command = "cat <<'A' | bash <<'B'\ncd '../a'\nA\ncd '../b'\nB"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    def test_traversal_in_multiple_heredocs_on_one_command_is_blocked(self):
        command = "bash <<EOF\ncd '../a'\nEOF\nbash <<EOF\ncd '../b'\nEOF"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF'\ncd '../x'\nEOF",
            "env cat <<'EOF'\ncd '../x'\nEOF",
        ],
    )
    def test_wrapped_non_shell_heredoc_stays_data(self, command):
        assert validate_command(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            "xargs bash '-c' <<'EOF'\ncd '../x'\nEOF",
            'xargs bash "-c" <<EOF\ncd "../x"\nEOF',
            'xargs bash \\-c <<EOF\ncd "../x"\nEOF',
            "xargs sh '-c' <<'EOF'\ncd '../x'\nEOF",
            "xargs bash -ec <<'EOF'\ncd '../x'\nEOF",
        ],
    )
    def test_quoted_or_clustered_c_option_still_runs_the_heredoc(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "bash -eu <<'EOF'\ncd '../x'\nEOF",
            'bash -euo pipefail <<EOF\ncd "../x"\nEOF',
            "bash -o pipefail <<'EOF'\ncd '../x'\nEOF",
        ],
    )
    def test_clustered_options_before_a_stdin_script_are_decoded(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "xargs bash -c <<'EOF'\n'sudo chmod 777 ./x'\nEOF",
            "xargs bash -c <<'EOF'\npkill -f myproc\nEOF",
        ],
    )
    def test_blocked_commands_in_an_xargs_script_are_found(self, command):
        assert validate_command(command) is not None

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF' | bash\ncd '../x'\nEOF",
            "cat <<'EOF' | bash -s\ncd '../x'\nEOF",
            "cat <<'EOF' | sh\ncd '../x'\nEOF",
            "cat <<'EOF' | zsh -s\ncd '../x'\nEOF",
            "{ cat <<'EOF'; } | bash\ncd '../x'\nEOF",
        ],
    )
    def test_heredoc_piped_into_a_stdin_reading_shell_is_blocked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "timeout 5 bash -c 'echo' <<'EOF'\ncd '../x'\nEOF",
            "bash -ec 'echo ok' <<'EOF'\ncd '../x'\nEOF",
            "bash -c 'echo \"$1\"' bash <<'EOF'\ncd '../x'\nEOF",
            "cat <<'EOF' | bash script.sh\ncd '../x'\nEOF",
            "cat <<'EOF' | grep -f -\ncd '../x'\nEOF",
            "cat <<'EOF' | grep foo\ncd '../x'\nEOF",
        ],
    )
    def test_heredoc_stays_data_when_no_shell_runs_it(self, command):
        assert validate_command(command) is None

    @pytest.mark.parametrize(
        "separator",
        [";", "&&", "|", "$(echo ../x)", "`echo ../x`"],
    )
    def test_operators_in_an_xargs_script_do_not_crash_the_check(self, separator):
        if "$(" in separator:
            body = f'cd "{separator}"'
        elif "`" in separator:
            body = f'cd "{separator}"'
        else:
            body = f"cd '../x' {separator} echo done"
        command = f"xargs bash '-c' <<'EOF'\n{body}\nEOF"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF' | xargs bash -c\ncd '../x'\nEOF",
            "cat <<'EOF' | xargs sh -c\ncd '../x'\nEOF",
        ],
    )
    def test_heredoc_piped_into_xargs_shell_is_blocked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    def test_multiline_braced_feeder_group_is_blocked(self):
        command = "{\n  cat <<'EOF'\ncd '../x'\nEOF\n  echo ok\n} | bash"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "( cat <<'EOF'\ncd '../x'\nEOF\n) | bash",
            "(\n  cat <<'EOF'\ncd '../x'\nEOF\n  echo ok\n) | bash",
        ],
    )
    def test_parenthesized_feeder_subshell_is_blocked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "options",
        ["-oe", "-O extglob", "-o pipefail", "-euo pipefail"],
    )
    def test_short_option_cluster_with_a_value_is_decoded(self, options):
        command = f"bash {options} <<'EOF'\ncd '../x'\nEOF"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    def test_c_cluster_takes_its_value_and_leaves_the_heredoc_as_data(self):
        # `-oc pipefail` consumes pipefail as the value of `-o`, so no `-c`
        # script is present and the here-document is not run by that shell.
        command = "bash -oc pipefail <<'EOF'\ncd '../x'\nEOF"
        assert validate_command(command) is None

    def test_xargs_supplied_script_with_quotes_is_checked(self):
        result = validate_command("xargs bash -c <<'EOF'\n'cd \"../x\"'\nEOF")
        assert result is not None
        assert "traversal" in result

    def test_xargs_input_is_parsed_without_shell_comment_rules(self):
        command = "xargs -n 1 bash -c <<'EOF'\n# 'cat \"../secret\"'\nEOF"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "xargs -I{} bash -c '{}' <<'EOF'\ncd '../x'\nEOF",
            "xargs -I % sh -c '%' <<'EOF'\ncd '../x'\nEOF",
            "cat <<'EOF' | xargs -I{} bash -c '{}'\ncd '../x'\nEOF",
            "xargs -i bash -c '{}' <<'EOF'\ncd '../x'\nEOF",
            "xargs --replace bash -c '{}' <<'EOF'\ncd '../x'\nEOF",
        ],
    )
    def test_xargs_replace_option_feeds_code_to_shell(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF' |& bash\ncat '../secret'\nEOF",
            "cat <<'EOF' |& xargs bash -c\ncd '../x'\nEOF",
        ],
    )
    def test_pipe_stderr_operator_feeds_shell_heredoc(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF' | cat | bash\ncat '../secret'\nEOF",
            "cat <<'EOF' | tee out.log | bash\ncd '../x'\nEOF",
            "cat <<'EOF' |& cat |& bash\ncd '../x'\nEOF",
        ],
    )
    def test_forwarding_pipeline_stages_are_followed(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'DATA' | bash <<'CODE'\ncd '../x'\nDATA\necho ok\nCODE",
            "cat <<'DATA' | bash < /dev/null\ncd '../x'\nDATA",
            "cat <<'A' | cat <<'B' | bash\ncd '../a'\nA\necho ok\nB",
            "cat <<'A'\ncd '../a'\nA\necho ok | bash",
        ],
    )
    def test_stdin_redirection_overrides_piped_heredoc(self, command):
        assert validate_command(command) is None

    def test_stderr_redirection_does_not_override_piped_heredoc(self):
        command = "cat <<'DATA' | bash 2>/dev/null\ncd '../x'\nDATA"
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "bash -c 'bash' <<'EOF'\ncd '../x'\nEOF",
            "bash -c 'sh -s' <<'EOF'\ncd '../x'\nEOF",
            "cat <<'EOF' | bash -c 'bash'\ncd '../x'\nEOF",
            "{ bash; } <<'EOF'\ncd '../x'\nEOF",
            "( bash ) <<'EOF'\ncd '../x'\nEOF",
            "cat <<'EOF' | { bash; }\ncd '../x'\nEOF",
            "cat <<'EOF' | ( bash )\ncd '../x'\nEOF",
        ],
    )
    def test_inherited_stdin_in_c_script_and_groups_is_checked(self, command):
        result = validate_command(command)
        assert result is not None
        assert "traversal" in result

    @pytest.mark.parametrize(
        "command",
        [
            "bash -c 'cat' <<'EOF'\ncd '../x'\nEOF",
            "{ cat; } <<'EOF'\ncd '../x'\nEOF",
            "( cat ) <<'EOF'\ncd '../x'\nEOF",
            "{ bash <<'CODE'; } <<'DATA'\necho ok\nCODE\ncd '../x'\nDATA",
            "{ echo ok | bash; } <<'DATA'\ncd '../x'\nDATA",
        ],
    )
    def test_group_or_c_script_heredoc_stays_data_when_not_run_as_code(self, command):
        assert validate_command(command) is None

    def test_nested_heredocs_scale_linearly_and_detect_deep_traversal(
        self, monkeypatch
    ):
        def build(inner: str, depth: int) -> str:
            text = inner
            for level in range(depth):
                tag = f"L{level}"
                text = f"bash <<'{tag}'\n{text}\n{tag}"
            return text

        calls = 0
        original = backends._has_traversal_component

        def counting(command: str, **kwargs):
            nonlocal calls
            calls += 1
            return original(command, **kwargs)

        monkeypatch.setattr(backends, "_has_traversal_component", counting)

        safe = build("echo ok", depth=20)
        started = time.monotonic()
        assert validate_command(safe) is None
        assert time.monotonic() - started < 2.0
        assert calls <= 25

        calls = 0
        unsafe = build("cd '../x'", depth=20)
        result = validate_command(unsafe)
        assert result is not None
        assert "traversal" in result
        assert calls <= 25

    @pytest.mark.parametrize(
        "command",
        [
            "zsh -O -c 'x'",
            "zsh -opipefail -c 'x'",
            "zsh -onomatch -c 'x'",
            "bash +c 'x'",
            "bash -euo pipefail -c 'x'",
        ],
    )
    def test_shell_code_handles_main_option_spellings(self, command):
        shell = shell_syntax.simple_commands(command)[0]
        script, _stdin = shell_syntax._shell_code(shell)
        assert script is not None
        assert shell.args[script] == "'x'"
