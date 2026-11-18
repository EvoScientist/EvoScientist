"""The shell syntax behind the sandbox's command checks and path rewriting."""

import time

import pytest

from EvoScientist import shell_syntax


def _values(command: str) -> list[str]:
    return [str(token["value"]) for token in shell_syntax.parse(command).tokens]


def _names(command: str) -> list[str]:
    words = shell_syntax.parse(command)
    return [
        str(token["value"])
        for token, is_name in zip(words.tokens, words.name_flags, strict=True)
        if is_name
    ]


class TestTokens:
    def test_quoted_parts_join_one_word(self):
        [word, other] = shell_syntax.tokenize("'a b'\"c\" x")
        assert (word["value"], word["raw"], word["quoted"]) == (
            "a bc",
            "'a b'\"c\"",
            True,
        )
        assert other["quoted"] is False

    def test_operators_and_redirections(self):
        assert _values("a && b | c >| f; d &> g") == [
            "a", "&&", "b", "|", "c", ">|", "f", ";", "d", "&>", "g",
        ]  # fmt: skip

    def test_a_backslash_in_double_quotes_escapes_only_what_bash_escapes(self):
        assert _values(r'cd "D:\data" "a\$b" "say \"hi\""') == [
            "cd",
            r"D:\data",
            "a$b",
            'say "hi"',
        ]

    def test_comments_and_line_continuations_are_not_words(self):
        assert _values("ls # it's a comment\necho \\\n  hi") == ["ls", "echo", "hi"]

    def test_a_hash_glued_to_a_substitution_is_a_word(self):
        assert "#" in _values("echo $(date)#; sudo whoami")


class TestHereDocuments:
    def test_bodies_are_not_tokenized_and_record_whether_they_expand(self):
        command = "cat <<EOF > a\n$(x)\nEOF\ncat <<'Q'\nb\nQ\nls"
        words = shell_syntax.parse(command)
        assert " ".join(_values(command)) == "cat << EOF > a cat << Q ls"
        assert [(command[a:b], expands) for a, b, expands in words.bodies] == [
            ("$(x)\nEOF\n", True),
            ("b\nQ\n", False),
        ]
        assert sorted(words.delimiters) == [command.index("EOF"), command.index("'Q'")]

    def test_a_shift_inside_arithmetic_is_no_here_document(self):
        words = shell_syntax.parse("echo $((1 << 2)); sudo x")
        inside = [
            str(t["value"]) for t in words.tokens if t["start"] in words.arithmetic
        ]
        assert not words.bodies
        assert inside == ["1", "<<", "2", ")", ")"]
        assert _names("echo $((1 << 2)); sudo x")[-1] == "sudo"


class TestCommandNames:
    def test_names_after_assignments_reserved_words_and_time(self):
        assert _names("FLAG=1 pkill a; if true; then sudo b; fi; time -p c x") == [
            "FLAG=1", "pkill", "if", "true", "then", "sudo", "fi", "time", "c",
        ]  # fmt: skip

    def test_names_inside_substitutions_and_expanding_here_documents(self):
        command = 'echo "$(sudo ls)"; cat <<EOF\n$(pkill x)\nEOF'
        assert shell_syntax.command_names(command) == ["echo", "cat", "sudo", "pkill"]

    def test_a_quoted_here_document_runs_nothing(self):
        assert shell_syntax.command_names("cat <<'EOF'\n$(pkill x)\nEOF") == ["cat"]


class TestWordsAndScripts:
    def test_a_substitution_inside_a_path_keeps_one_word(self):
        command = "/Users/$(whoami)/x"
        words = shell_syntax.glued_words(shell_syntax.parse(command).tokens)
        assert [command[a:b] for a, b, _quoted in words] == [command]

    def test_the_script_of_a_wrapped_shell_is_found(self):
        command = "timeout 5 bash -lc 'ls /x'"
        assert shell_syntax.shell_script_words(shell_syntax.parse(command).tokens) == {
            command.index("'")
        }

    @pytest.mark.parametrize(
        "options",
        [
            "-o pipefail",
            "-O extglob",
            "-euo pipefail",
            "+o pipefail",
            "-o pipefail -O extglob",
            "-o pipefail -lc",
        ],
    )
    def test_named_shell_options_are_skipped_before_the_script(self, options):
        option_c = "" if options.endswith("-lc") else " -c"
        command = f"timeout 5 bash {options}{option_c} 'cat /tmp/a'"
        assert shell_syntax.shell_script_words(shell_syntax.parse(command).tokens) == {
            command.index("'")
        }

    @pytest.mark.parametrize(
        "command",
        [
            "bash -o",
            "bash -O; echo x",
            "bash -o pipefail",
            "bash -c",
            "bash -o pipefail; echo -c 'not a script'",
        ],
    )
    def test_missing_named_option_or_script_does_not_consume_next_command(
        self, command
    ):
        assert not shell_syntax.shell_script_words(shell_syntax.parse(command).tokens)

    def test_substitutions_in_double_quotes_are_converted(self):
        assert shell_syntax.rewrite_substitutions(
            "\"$(ls /a) and `cat /b`\" '$(ls /c)'",
            lambda body: (body.upper(), body == "ls /a"),
        ) == ("\"$(LS /A) and `CAT /B`\" '$(ls /c)'", True)

    def test_substitutions_anywhere_in_an_expanding_body(self):
        assert shell_syntax.rewrite_substitutions(
            "x $(ls) 'q' `y`\n", lambda body: (body.upper(), False), heredoc=True
        ) == ("x $(LS) 'q' `Y`\n", False)

    def test_the_closing_paren_skips_quotes_and_comments(self):
        text = 'a (b) "c)" # x)\n) tail'
        assert shell_syntax.matching_paren(text, 0) == text.index("\n") + 1


class TestSimpleCommands:
    @staticmethod
    def _found(command):
        return [(c.name, c.via) for c in shell_syntax.simple_commands(command)]

    def test_wrapped_commands_and_their_wrappers(self):
        assert self._found("env 'RUN=a b' timeout -s KILL 5 pkill -f x") == [
            ("env", ()),
            ("timeout", ("env",)),
            ("pkill", ("env", "timeout")),
        ]

    def test_command_v_only_looks_the_name_up(self):
        assert self._found("command -v sudo >/dev/null") == [("command", ())]

    def test_scripts_of_shells_in_command_position_keep_their_wrappers(self):
        command = "pgrep -f t | xargs sh -c 'kill -9 \"$@\"' _"
        assert ("kill", ("xargs",)) in self._found(command)
        assert self._found('docker exec c bash -c "chmod +x a"') == [("docker", ())]

    def test_here_documents_a_shell_reads_as_code(self):
        assert ("pkill", ("timeout",)) in self._found(
            "timeout 30 sh <<'EOF'\npkill -f t\nEOF"
        )
        for data in (
            "bash -c 'cat' <<'EOF'\npkill\nEOF",
            "bash run.sh <<EOF\npkill\nEOF",
            "timeout 5 bash -n <<'EOF'\npkill\nEOF",
            "bash -n -c 'pkill x'",
        ):
            assert "pkill" not in [name for name, _ in self._found(data)]
        # After the script, -n is only $0; +n undoes an earlier -n.
        assert ("pkill", ()) in self._found("bash -c 'pkill x' -n")
        assert ("pkill", ()) in self._found("bash -n +n -c 'pkill x'")

    def test_nested_scripts_are_read_once(self):
        command = "echo hi"
        for _ in range(40):
            command = 'bash -c "$(' + command.replace('"', '\\"') + ')"'
        started = time.monotonic()
        assert len(shell_syntax.simple_commands(command)) < 400
        assert time.monotonic() - started < 2
        found = self._found("cat ids | xargs sh -c 'kill $0'; sh -c 'kill $0'")
        assert ("kill", ("xargs",)) in found
        assert ("kill", ()) in found

    def test_arguments_leave_out_redirection_targets(self):
        [rm, echo] = shell_syntax.simple_commands("rm -rf /a /b > /dev/null 2>&1; echo")
        assert rm.args == ("-rf", "/a", "/b")
        assert echo.args == ()

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("env -S 'pkill -f x'", ("pkill", ("-f", "x"))),
            ("env -S'pkill -f x'", ("pkill", ("-f", "x"))),
            ("env -iS 'pkill -f x'", ("pkill", ("-f", "x"))),
            ("env --split-string='pkill -f x'", ("pkill", ("-f", "x"))),
            ("env --split 'pkill -f x'", ("pkill", ("-f", "x"))),
            ("env -S 'kill -9' -1", ("kill", ("-9", "-1"))),
            ("env -S 'pkill\\_-f x'", ("pkill", ("-f", "x"))),
            ("env -S 'pkill\\c' -f worker", ("pkill", ("-f", "worker"))),
            ("env -S 'kill -9\\c ignored' -1", ("kill", ("-9", "-1"))),
            ("env -S 'sh -c \"echo safe\\npkill x\"'", ("pkill", ("x",))),
            ("env -S 'echo a\\nb\\tc d'", ("echo", ("a'\n'b'\t'c", "d"))),
            ("env -S \"echo 'a\\nb'\"", ("echo", ("'a\\nb'",))),
            ("env -S '-i -S \"pkill x\"'", ("pkill", ("x",))),
            (
                "env -S 'cat \"name with space.txt\"'",
                ("cat", ('"name with space.txt"',)),
            ),
            (
                "env -S \"cat 'name with space.txt'\"",
                ("cat", ("'name with space.txt'",)),
            ),
            (
                "env -S 'cat name\\ with\\ space.txt'",
                ("cat", ("name\\ with\\ space.txt",)),
            ),
            ("env -S \"cat 'a\\_b'\"", ("cat", ("'a\\_b'",))),
        ],
    )
    def test_env_runs_the_words_it_splits(self, command, expected):
        commands = shell_syntax.simple_commands(command)
        assert expected in [(found.name, found.args) for found in commands]
        assert all(found.via == ("env",) for found in commands[1:])

    def test_env_split_strings_are_also_read_as_code(self):
        assert ("pkill", ("env",)) in self._found("env -S 'a; pkill x'")
        assert ("pkill", ("env",)) in self._found("env -S 'sh -c \"pkill x\"'")
        # A bare `--` ends env's options; it is no `--split-string` prefix.
        assert self._found("env -- pkill x") == [("env", ()), ("pkill", ("env",))]
        assert self._found("env -u S python x.py") == [
            ("env", ()),
            ("python", ("env",)),
        ]

    @pytest.mark.parametrize(
        "command",
        [
            "env -iC build pkill x",
            "env --ch build pkill x",
            "env -iu A pkill x",
            "env -au pkill x",
            "env -P bin pkill x",
            "env --argv0 x pkill x",
        ],
    )
    def test_env_options_take_their_value_clustered_or_shortened(self, command):
        assert ("pkill", ("env",)) in self._found(command)

    def test_assignments_wherever_a_variable_is_set(self):
        assert shell_syntax.assignments(
            'A=1; B=$(ps aux | grep x) :; export C=`lsof -t`; local -r D="$(ps)" E'
        ) == [
            ("A", "1"),
            ("B", "$(ps aux | grep x)"),
            ("C", "`lsof -t`"),
            ("D", '"$(ps)"'),
        ]
        assert shell_syntax.assignments("arr[0]=$(ps) n+=1") == [
            ("arr", "$(ps)"),
            ("n", "1"),
        ]
        assert sorted(
            shell_syntax.assignments("bash -c 'P=$(pgrep x)'; echo $(Q=1)")
        ) == [("P", "$(pgrep x)"), ("Q", "1")]
        assert shell_syntax.assignments("echo A=1; FOO='x=1'") == [("FOO", "'x=1'")]
        # Quoted names; the value keeps the quotes opened before the `=`.
        assert shell_syntax.assignments(
            "export \"P=$(lsof -t)\" 'Q'=$(ps) 'R=$(ps)' -x=1 \"$X\"=1"
        ) == [("P", '"$(lsof -t)"'), ("Q", "''$(ps)"), ("R", "'$(ps)'")]

    @pytest.mark.parametrize(
        ("command", "folders"),
        [
            ("env -C build rm x", ("build",)),
            ("env -iC build rm x", ("build",)),
            ("env -iCbuild rm x", ("build",)),
            ("env --chdir=build rm x", ("build",)),
            ("env --ch build rm x", ("build",)),
            ("env -C 'my dir' rm x", ("my dir",)),
            ("env -C a env -C b rm x", ("a", "b")),
            # One env changes folder once, to its last -C.
            ("env -C a -C b rm x", ("b",)),
            ("env -C a --chdir=b env -C c -Cd rm x", ("b", "d")),
            ("nohup env -C a rm x", ("a",)),
            ("env -C a sh -c 'rm x'", ("a",)),
            ("env -C a -S \"sh -c 'rm x'\"", ("a",)),
            ("env -u C rm x", ()),
            ('env -C a echo "$(rm x)"', ()),  # expands before env runs
        ],
    )
    def test_commands_carry_the_env_folders_they_run_in(self, command, folders):
        removals = [
            each for each in shell_syntax.simple_commands(command) if each.name == "rm"
        ]
        assert removals
        assert all(each.folders == folders for each in removals)

    def test_a_function_body_is_read_as_commands(self):
        found = shell_syntax.simple_commands("function f { cd a; rm -rf x; }; f")
        assert {each.name for each in found} >= {"cd", "rm", "f", "function"}

    @pytest.mark.parametrize(
        ("command", "defines"),
        [
            ("f() { cd a; }; f", True),
            ("g () ( cd b )", True),
            ("bash -c 'f() { cd a; }'", True),
            ('python -c "main()"', False),
            ('git commit -m "rename function foo"', False),
        ],
    )
    def test_a_function_definition_is_a_command_named_function(self, command, defines):
        names = {each.name for each in shell_syntax.simple_commands(command)}
        assert ("function" in names) is defines

    @pytest.mark.parametrize(
        ("command", "name", "args"),
        [
            ("function w { cd $(mktemp -d) && touch x; }; w", "cd", ("$(mktemp -d)",)),
            ("function s { kill $(lsof -t -i:8000); }; s", "kill", ("$(lsof -t -i:8000)",)),
            ("function f { rm -rf x/$(date +%F); }; f", "rm", ("-rf", "x/$(date +%F)")),
        ],
    )  # fmt: skip
    def test_a_function_body_ending_in_a_substitution_is_read_whole(
        self, command, name, args
    ):
        found = shell_syntax.simple_commands(command)
        assert args in [each.args for each in found if each.name == name]

    def test_the_same_script_in_two_env_folders_is_read_in_both(self):
        found = shell_syntax.simple_commands("env -C a sh -c X; env -C b sh -c X")
        assert [each.folders for each in found if each.name == "X"] == [
            ("a",),
            ("b",),
        ]

    def test_assigned_values_as_the_shell_reads_them(self):
        assert shell_syntax.assigned_values(
            'T=..; A="$HOME"; export B=$A/x \'C\'=~; declare "D=a b"; E='
        ) == {
            "T": ("..",),
            "A": ("${HOME}",),
            "B": ("${HOME}/x",),
            "C": ("~",),
            "D": ("a b",),
            "E": ("",),
        }
        # In any order and in nested code; every value of a name.
        assert shell_syntax.assigned_values("cd \"$T\"; bash -c 'T=sub'; T=..") == {
            "T": ("..", "sub")
        }
        assert shell_syntax.assigned_values("X=1 Y=${X}2") == {
            "X": ("1",),
            "Y": ("12",),
        }

    def test_assigned_values_leave_out_what_only_the_run_knows(self):
        assert shell_syntax.assigned_values('A=$(pwd) B=`pwd` C="$(pwd)/x"') == {
            "A": (),
            "B": (),
            "C": (),
        }

    def test_a_name_that_refers_back_to_itself_is_not_followed(self):
        assert shell_syntax.assigned_values("A=$A/x") == {"A": None}
        assert shell_syntax.assigned_values("D=.evoscientist; D=$D/memories") == {
            "D": None
        }
        assert shell_syntax.assigned_values("B=skills; A=$B; B=$A") == {
            "A": None,
            "B": None,
        }
        # A chain without a cycle is followed.
        assert shell_syntax.assigned_values("A=..; B=$A/x") == {
            "A": ("..",),
            "B": ("../x",),
        }

    def test_variables_set_other_ways_are_not_followed(self):
        assert shell_syntax.assigned_values(
            "printf -v P x; read -r R; declare -a N=E; : ${S:=x}; "
            "for L in $X; do :; done; export -x E=a; builtin export B=b"
        ) == {
            "P": None,
            "R": None,
            "REPLY": None,  # read sets it when it names no variable
            "N": None,
            "S": None,
            "L": None,
            "E": ("a",),
            "B": ("b",),
        }
        assert shell_syntax.assigned_values("env 'Q=a b' sh -c :") == {"Q": ("a b",)}
        assert shell_syntax.assigned_values(
            "D=x; unset D; for B in {a,b}; do :; done; mapfile"
        ) == {"D": None, "B": None, "MAPFILE": None}
        # A nameref or eval may set any of them.
        assert shell_syntax.assigned_values("D=x; declare -n R=D") == {
            "D": None,
            "R": None,
        }
        assert shell_syntax.assigned_values("D=x; eval y") == {"D": None}
        assert shell_syntax.assigned_values('D=x; printf "-v" D y; read "E"') == {
            "D": None,
            "E": None,
            "REPLY": None,
        }

    def test_more_values_than_the_cap_are_reported(self):
        values = dict.fromkeys("VWXYZ", ("a", "b"))
        assert len(shell_syntax.expand_references("$V$W$X$Y", values)) == 16
        with pytest.raises(shell_syntax.UnknownValues):
            shell_syntax.expand_references("$V$W$X$Y$Z", values)
        assigned = shell_syntax.assigned_values(
            "; ".join(f"T=a{i}" for i in range(17)) + "; U=$T/x; V=1"
        )
        # Only the words that name them are refused.
        assert assigned == {"T": None, "U": None, "V": ("1",)}
        with pytest.raises(shell_syntax.UnknownValues) as raised:
            shell_syntax.word_values('"$U"', "$U", assigned)
        assert raised.value.word == '"$U"'
        assert shell_syntax.word_values("$V", "$V", assigned) == ["1"]

    def test_a_name_ends_where_it_is_written_not_where_quotes_end(self):
        assert shell_syntax.braced_value('"$D"ills') == "${D}ills"
        assert shell_syntax.braced_value("'$D'ills") == "$Dills"
        assert shell_syntax.braced_value("a b") is None
        assert shell_syntax.assigned_values('A=sk; D="$A"ills') == {
            "A": ("sk",),
            "D": ("skills",),
        }
        assigned = {"D": ("sk",)}
        assert shell_syntax.word_values('"$D"ills', "$Dills", assigned) == ["skills"]
        [env_rm] = [
            each
            for each in shell_syntax.simple_commands('env -C "$D"ills rm x')
            if each.name == "rm"
        ]
        assert env_rm.folders == ("${D}ills",)

    def test_a_variable_with_known_and_run_only_values_has_none_to_read(self):
        assert shell_syntax.assigned_values(
            "X=build; X=$(echo skills); Y=a; Y=(b c); Z=a; Z=$R; R=`pwd`; S=$(pwd)"
        ) == {"X": None, "Y": None, "Z": None, "R": (), "S": ()}

    def test_an_unquoted_value_splits_into_the_words_it_makes(self):
        assigned = {"D": ("x .evoscientist",)}
        assert shell_syntax.word_values("$D", "$D", assigned) == ["x", ".evoscientist"]
        assert shell_syntax.word_values('"$D"', "$D", assigned) == ["x .evoscientist"]
        with pytest.raises(shell_syntax.UnknownValues):
            shell_syntax.word_values("$D", "$D", {**assigned, "IFS": (":",)})
        assert shell_syntax.word_values('"$D"', "$D", {**assigned, "IFS": (":",)})

    @pytest.mark.parametrize(
        "command",
        [
            "D=x :",
            "D=x export E",
            "D=x declare E",
            "D=x cd .",
            "f() { :; }; D=x f",
            "D=x bash -c :",
            "D=x xargs sh -c :",
            "env D=x bash -c :",
            "D=x env -S 'ls'",
            "D=x time :",
        ],
    )
    def test_an_assignment_before_a_builtin_or_shell_code_counts(self, command):
        assert shell_syntax.assigned_values(command) == {"D": ("x",)}
        assert shell_syntax.assigned_values("D=x eval :") == {"D": None}

    @pytest.mark.parametrize(
        "command",
        ["IFS= read -r line", "D=x python y.py", "D=x A=y true", "env D=x python y.py"],
    )
    def test_an_assignment_before_any_other_command_is_that_commands_only(
        self, command
    ):
        assert "D" not in shell_syntax.assigned_values(command)
        assert "IFS" not in shell_syntax.assigned_values(command)

    def test_loop_words_and_env_assignments_are_values(self):
        assert shell_syntax.assigned_values(
            "for d in skills x; do :; done; env -i E='a b' bash x.sh; L=(a b)"
        ) == {"d": ("skills", "x"), "E": ("a b",), "L": None}
        with pytest.raises(shell_syntax.UnknownValues):
            shell_syntax.word_values('"${d%/}"', "${d%/}", {"d": ("x",)})
        assert shell_syntax.word_values('"${e%/}"', "${e%/}", {"d": ("x",)}) is None

    def test_a_variable_set_in_part_has_no_values_to_read(self):
        # Its value depends on the order of the assignments.
        assert shell_syntax.assigned_values(
            "D=sk; D+=ills; export E+=x; a[0]=x; F=$D/y; G=1"
        ) == {"D": None, "E": None, "a": None, "F": None, "G": ("1",)}

    def test_word_values_follow_unquoted_references_only(self):
        assigned = {"T": ("..", "sub"), "U": ()}
        assert shell_syntax.word_values('"$T/x"', "$T/x", assigned) == [
            "../x",
            "sub/x",
        ]
        assert shell_syntax.word_values("${T}", "${T}", assigned) == ["..", "sub"]
        assert shell_syntax.word_values("'$T'", "$T", assigned) is None
        assert shell_syntax.word_values("\\$T", "$T", assigned) is None
        assert shell_syntax.word_values("$OTHER", "$OTHER", assigned) is None
        assert shell_syntax.word_values("$U", "$U", assigned) == []

    def test_a_name_inside_a_substitution_is_not_the_words_own(self):
        assert shell_syntax.variable_references('"$(dirname "$f")"') == set()
        assert shell_syntax.variable_references("`echo $f`/$D") == {"D"}
        assert shell_syntax.variable_references('"$(echo "$(cat $x)")"$y') == {"y"}

    def test_a_substitution_is_one_argument(self):
        [kill, pgrep] = shell_syntax.simple_commands("kill `pgrep -f python`")
        assert kill.args == ("`pgrep -f python`",)
        assert pgrep.args == ("-f", "python")


def test_parse_is_shared_between_checks():
    assert shell_syntax.parse("ls -la") is shell_syntax.parse("ls -la")
