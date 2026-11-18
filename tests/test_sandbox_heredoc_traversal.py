"""Traversal in code a shell reads from a here-document on stdin (issue #848).

A quoted or escaped ``..`` path in such a body ran even though the same line
in a direct command was refused. The body is code only when a shell reads it;
on ``cat`` the same body is data.
"""

import pytest

from EvoScientist.backends import validate_command
from EvoScientist.shell_syntax import _SHELL_OPTION_VALUE_RE


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

    @pytest.mark.parametrize("cluster", ["-oc", "-oe", "-euo", "-o", "-O", "+o"])
    def test_value_taking_short_option_is_matched_anywhere_in_a_cluster(self, cluster):
        assert _SHELL_OPTION_VALUE_RE.fullmatch(cluster) is not None

    @pytest.mark.parametrize("cluster", ["-c", "-ec", "-e", "+n"])
    def test_value_taking_short_option_pattern_skips_plain_flags(self, cluster):
        assert _SHELL_OPTION_VALUE_RE.fullmatch(cluster) is None

    def test_c_cluster_takes_its_value_and_leaves_the_heredoc_as_data(self):
        # `-oc pipefail` consumes pipefail as the value of `-o`, so no `-c`
        # script is present and the here-document is not run by that shell.
        command = "bash -oc pipefail <<'EOF'\ncd '../x'\nEOF"
        assert validate_command(command) is None
