"""The shell syntax behind the sandbox's command checks and path rewriting."""

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


def test_parse_is_shared_between_checks():
    assert shell_syntax.parse("ls -la") is shell_syntax.parse("ls -la")
