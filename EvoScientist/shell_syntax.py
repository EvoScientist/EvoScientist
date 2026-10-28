"""Enough POSIX shell syntax for the sandbox's command checks and path rewriting.

Words, quotes, comments, operators, here-documents, ``$(...)`` / backtick
substitutions and command-name positions; not a full shell parser.
"""

import functools
import re
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

# Operators after which the next word is a command name.
COMMAND_START_OPS = frozenset({";", "&&", "||", "|", "&", "(", ")", "`"})
# Words after which the next word is still a command name.
COMMAND_PREFIX_WORDS = frozenset(
    {"do", "then", "else", "elif", "if", "while", "until", "!", "time", "{"}
)
# A newline that ends a command, not a backslash line continuation.
COMMAND_NEWLINE_RE = re.compile(r"(?<!\\)\n")
ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
HEREDOC_OP_RE = re.compile(r"\d*<<")
SHELL_NAMES = frozenset({"sh", "bash", "zsh", "dash"})
SHELL_C_OPTION_RE = re.compile(r"-[A-Za-z]*c[A-Za-z]*")
# Commands that run the command named by their first operand.
WRAPPER_NAMES = frozenset(
    {"env", "nohup", "timeout", "xargs", "exec", "command", "nice", "setsid"}
)
# Wrapper operands before the command: a duration or niceness (`5`, `1.5m`).
_WRAPPER_VALUE_RE = re.compile(r"-?\d+(\.\d+)?[smhd]?")
# Wrapper options whose value is the next word.
_WRAPPER_OPTIONS_WITH_VALUE = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "xargs": frozenset({"-n", "-I", "-P", "-L", "-s", "-d", "-E", "-a"}),
}
# Shell options whose value is the next word.
_SHELL_OPTIONS_WITH_VALUE = frozenset(
    {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}
)


def tokenize(command: str) -> list[dict[str, object]]:
    """Tokenize enough shell syntax for the command checks and path rewriting.

    This is intentionally small: it tracks words, quotes, comments and command
    separators, but does not try to be a full POSIX shell parser.
    """
    return list(iter_tokens(command))


def iter_tokens(command: str, start: int = 0) -> Iterator[dict[str, object]]:
    """The tokens of :func:`tokenize`, lazily and from *start*."""
    i = start
    n = len(command)

    def read_operator(index: int) -> str | None:
        if command.startswith(("&&", "||"), index):
            return command[index : index + 2]
        if command.startswith("&>", index):
            return "&>"
        ch = command[index]
        if ch in "`();|&":
            return ch
        if ch in "<>":
            # `>>`/`<<` and `>|` (force-clobber redirect) are single redirection
            # operators, NOT a pipe — the trailing `|` must not read as a boundary.
            if index + 1 < n and (
                command[index + 1] == ch or (ch == ">" and command[index + 1] == "|")
            ):
                return command[index : index + 2]
            return ch
        if ch.isdigit():
            j = index
            while j < n and command[j].isdigit():
                j += 1
            if j < n and command[j] in "<>":
                end = j + 1
                # `2>&1`, `2>>`, and `2>|` (fd force-clobber) are single
                # redirection operators — the trailing `|` is not a pipe.
                if end < n and command[end] in ("&", "|", command[j]):
                    end += 1
                return command[index:end]
        return None

    while i < n:
        if command[i].isspace():
            i += 1
            continue
        if command.startswith("\\\n", i):
            i += 2  # line continuation
            continue
        if command[i] == "#" and (i == 0 or command[i - 1] not in ")`"):
            # A comment: its quotes are text, so they open nothing. After a
            # ")" or backtick the "#" still belongs to the word.
            newline = command.find("\n", i)
            i = n if newline == -1 else newline
            continue
        operator = read_operator(i)
        if operator is not None:
            yield {
                "type": "op",
                "value": operator,
                "start": i,
                "end": i + len(operator),
            }
            i += len(operator)
            continue

        word_start = i
        value: list[str] = []
        quoted = False
        while i < n:
            ch = command[i]
            if ch.isspace():
                break
            if read_operator(i) is not None:
                break
            if ch in ("'", '"'):
                quoted = True
                quote = ch
                i += 1
                while i < n:
                    inner = command[i]
                    if inner == quote:
                        i += 1
                        break
                    if (
                        quote == '"'
                        and command.startswith("$(", i)
                        and not command.startswith("$((", i)
                    ):
                        # Quotes inside the substitution do not end the string.
                        close = matching_paren(command, i + 2)
                        if close is not None:
                            value.append(command[i : close + 1])
                            i = close + 1
                            continue
                    if inner == "\\" and quote == '"' and i + 1 < n:
                        value.append(command[i + 1])
                        i += 2
                    else:
                        value.append(inner)
                        i += 1
                continue
            if ch == "\\" and i + 1 < n:
                value.append(command[i + 1])
                i += 2
                continue
            value.append(ch)
            i += 1

        yield {
            "type": "word",
            "value": "".join(value),
            "raw": command[word_start:i],
            "start": word_start,
            "end": i,
            "quoted": quoted,
        }


def matching_paren(text: str, start: int) -> int | None:
    """Index of the ``)`` closing a ``$(`` whose body begins at *start*."""
    depth = 0
    quote: str | None = None
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and quote != "'":
            i += 2
            continue
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == start or text[i - 1] in " \t\n;|&("):
            newline = text.find("\n", i)
            if newline == -1:
                return None
            i = newline
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return i
            depth -= 1
        i += 1
    return None


def _heredoc_spans(
    command: str, start: int, delimiters: list[tuple[str, bool, bool]]
) -> list[tuple[int, int, bool]] | None:
    """``(start, end, expands)`` of the here-document bodies that begin at
    *start*, one per ``(delimiter, strip_tabs, expands)``; ``None`` when one
    is unterminated."""
    spans: list[tuple[int, int, bool]] = []
    pos = start
    for delimiter, strip_tabs, expands in delimiters:
        body_start = pos
        while True:
            if pos >= len(command):
                return None
            newline = command.find("\n", pos)
            line_end = len(command) if newline == -1 else newline
            line = command[pos:line_end]
            pos = line_end + 1
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
        spans.append((body_start, min(pos, len(command)), expands))
    return spans


class ShellWords(NamedTuple):
    tokens: tuple[dict[str, object], ...]
    bodies: tuple[tuple[int, int, bool], ...]  # start, end, expands $(...)
    delimiters: frozenset[int]  # starts of here-document delimiter words
    arithmetic: frozenset[int]  # starts of tokens inside $(( )) / (( ))
    name_flags: tuple[bool, ...]  # per token: a word in command-name position
    body_delimiters: tuple[int, ...]  # per body: the start of its delimiter word


@functools.lru_cache(maxsize=16)
def parse(command: str) -> ShellWords:
    """Tokens of *command* outside here-document bodies; a body is data and is
    never tokenized, so quotes in it cannot swallow the commands after it.

    Cached per command text, so the checks share one parse; callers must not
    modify the result."""
    tokens: list[dict[str, object]] = []
    bodies: list[tuple[int, int, bool]] = []
    delimiters: set[int] = set()
    in_arithmetic: set[int] = set()
    pending: list[tuple[str, bool, bool]] = []
    pending_starts: list[int] = []
    body_delimiters: list[int] = []
    # One flag per open "(", True inside arithmetic, where "<<" is a shift.
    parens: list[bool] = []
    awaiting: tuple[int, bool] | None = None  # heredoc operator end, strip tabs
    stream = iter_tokens(command)
    prev_end = 0
    previous: dict[str, object] | None = None
    while True:
        token = next(stream, None)
        token_start = len(command) if token is None else int(token["start"])
        newline = COMMAND_NEWLINE_RE.search(command, prev_end, token_start)
        if pending and newline is not None:
            spans = _heredoc_spans(command, newline.end(), pending)
            starts, pending, pending_starts = pending_starts, [], []
            if spans is not None:
                bodies.extend(spans)
                body_delimiters.extend(starts)
                stream = iter_tokens(command, spans[-1][1])
                continue
        if token is None:
            break
        tokens.append(token)
        if any(parens):
            in_arithmetic.add(token_start)
        value = str(token["value"])
        if token["type"] == "op":
            awaiting = None
            if value == "(":
                arithmetic = (
                    previous is not None
                    and previous["value"] == "("
                    and int(previous["end"]) == token_start
                )
                if arithmetic and parens:
                    parens[-1] = True
                parens.append(arithmetic)
            elif value == ")" and parens:
                parens.pop()
            elif HEREDOC_OP_RE.fullmatch(value) and not any(parens):
                awaiting = (int(token["end"]), False)
        elif awaiting is not None:
            op_end, strip_tabs = awaiting
            raw = str(token["raw"])
            if raw == "-" and token_start == op_end:
                awaiting = (int(token["end"]), True)
            else:
                if token_start == op_end and raw.startswith("-"):
                    strip_tabs, value = True, value[1:]
                if value:
                    # A quoted or escaped delimiter makes the body literal.
                    expands = not token["quoted"] and "\\" not in raw
                    pending.append((value, strip_tabs, expands))
                    pending_starts.append(token_start)
                    delimiters.add(token_start)
                awaiting = None
        prev_end = int(token["end"])
        previous = token
    return ShellWords(
        tuple(tokens),
        tuple(bodies),
        frozenset(delimiters),
        frozenset(in_arithmetic),
        tuple(command_name_flags(command, tokens)),
        tuple(body_delimiters),
    )


def command_name_flags(command: str, tokens: Sequence[dict[str, object]]) -> list[bool]:
    """For each token, whether it is a word in command-name position."""
    flags: list[bool] = []
    previous: dict[str, object] | None = None
    carried = timing = False
    for token in tokens:
        starts_command = (
            carried
            or previous is None
            or (previous["type"] == "op" and previous["value"] in COMMAND_START_OPS)
            or COMMAND_NEWLINE_RE.search(
                command, int(previous["end"]), int(token["start"])
            )
            is not None
        )
        value = str(token["value"])
        is_name = token["type"] == "word" and starts_command
        time_option = is_name and timing and value.startswith("-")
        flags.append(is_name and not time_option)
        timing = time_option or (is_name and value == "time")
        carried = timing or (
            is_name
            and (
                value in COMMAND_PREFIX_WORDS
                or bool(ASSIGNMENT_RE.match(str(token["raw"])))
            )
        )
        previous = token
    return flags


def glued_words(tokens: Sequence[dict[str, object]]) -> list[tuple[int, int, bool]]:
    """``(start, end, quoted)`` of each shell word: tokens with nothing between
    them, so ``/Users/$(whoami)/x`` is one word; an enclosing closer ends it."""
    words: list[tuple[int, int, bool]] = []
    in_backtick = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        previous = tokens[index - 1] if index else None
        if token["type"] == "op":
            if token["value"] == "`":
                in_backtick = not in_backtick
            index += 1
            continue
        continues_previous = (
            previous is not None
            and int(previous["end"]) == int(token["start"])
            and previous["type"] == "op"
            and (
                previous["value"] == ")"
                or (previous["value"] == "`" and not in_backtick)
            )
        )
        if continues_previous:
            index += 1
            continue
        start, end = int(token["start"]), int(token["end"])
        quoted = bool(token["quoted"])
        depth = 0
        own_backtick = False
        index += 1
        while index < len(tokens):
            following = tokens[index]
            if int(following["start"]) != end:
                break
            value = following["value"]
            if following["type"] == "word":
                quoted = quoted or bool(following["quoted"])
            elif value == "(":
                depth += 1
            elif value == ")":
                if depth == 0:
                    break
                depth -= 1
            elif value == "`":
                if not own_backtick and in_backtick:
                    break
                own_backtick = not own_backtick
            else:
                break
            end = int(following["end"])
            index += 1
        if own_backtick:
            in_backtick = not in_backtick
        words.append((start, end, quoted))
    return words


def shell_script_words(tokens: Sequence[dict[str, object]]) -> set[int]:
    """Starts of the script arguments of ``sh -c '...'`` / ``bash -lc "..."``."""
    starts: set[int] = set()
    for index, token in enumerate(tokens):
        name = re.split(r"[\\/]", str(token["value"]))[-1]
        if token["type"] != "word" or name not in SHELL_NAMES:
            continue
        for option, script in zip(
            tokens[index + 1 :], tokens[index + 2 :], strict=False
        ):
            if option["type"] != "word" or not str(option["raw"]).startswith("-"):
                break
            if SHELL_C_OPTION_RE.fullmatch(str(option["raw"])):
                if script["type"] == "word":
                    starts.add(int(script["start"]))
                break
    return starts


def rewrite_substitutions(
    raw: str, convert: Callable[[str], tuple[str, bool]], *, heredoc: bool = False
) -> tuple[str, bool]:
    """Convert the commands inside ``$(...)`` and backticks within double
    quotes (or anywhere in an expanding *heredoc* body); they run, while the
    text around them is data."""
    out: list[str] = []
    uses_tmp = False
    quote: str | None = '"' if heredoc else None
    i = 0
    while i < len(raw):
        ch = raw[i]
        body: tuple[int, int, str, str] | None = None
        if quote == "'":
            if ch == "'":
                quote = None
        elif ch == "\\":
            out.append(raw[i : i + 2])
            i += 2
            continue
        elif ch in "'\"" and not heredoc and (quote is None or quote == ch):
            quote = None if quote == ch else ch
        elif quote == '"' and raw.startswith("$(", i) and not raw.startswith("$((", i):
            close = matching_paren(raw, i + 2)
            if close is not None:
                body = (i + 2, close, "$(", ")")
        elif quote == '"' and ch == "`":
            close = i + 1
            while close < len(raw) and raw[close] != "`":
                close += 2 if raw[close] == "\\" else 1
            if close < len(raw):
                body = (i + 1, close, "`", "`")
        if body is None:
            out.append(ch)
            i += 1
            continue
        inner_start, close, opener, closer = body
        inner, tmp = convert(raw[inner_start:close])
        uses_tmp = uses_tmp or tmp
        out.append(opener + inner + closer)
        i = close + 1
    return "".join(out), uses_tmp


class SimpleCommand(NamedTuple):
    name: str  # basename of the command word
    args: tuple[str, ...]  # the words after it as written; redirection targets left out
    tokens: tuple[
        dict[str, object] | None, ...
    ]  # per arg: its token, or None if several
    via: tuple[str, ...]  # wrappers it runs under, outermost first


def _command_word_end(
    tokens: Sequence[dict[str, object]], index: int, closing_backtick: bool
) -> int:
    """Index after the last token of the shell word starting at *index*: glued
    tokens, a whole ``$(...)`` and a whole backtick substitution; with
    *closing_backtick* a backtick after the start ends the word."""
    first = tokens[index]
    end = int(first["end"])
    depth = 0
    in_backtick = first["type"] == "op" and first["value"] == "`"
    index += 1
    while index < len(tokens):
        token = tokens[index]
        if not depth and not in_backtick and int(token["start"]) != end:
            break
        if token["type"] == "op":
            value = token["value"]
            if value == "(":
                depth += 1
            elif value == ")":
                if not depth:
                    break
                depth -= 1
            elif value == "`":
                if closing_backtick and not in_backtick and not depth:
                    break
                in_backtick = not in_backtick
            elif not depth and not in_backtick:
                break
        end = int(token["end"])
        index += 1
    return index


def _command_arguments(
    command: str, words: ShellWords, index: int, in_backtick: bool
) -> tuple[list[tuple[str, dict[str, object] | None]], list[int]]:
    """The arguments of the command whose name is token *index*, and the
    here-document bodies it reads; inside a backtick substitution the
    closing backtick ends them."""
    tokens = words.tokens
    args: list[tuple[str, dict[str, object] | None]] = []
    heredocs: list[int] = []
    redirect_target = False
    previous_end = int(tokens[index]["end"])
    index += 1
    while index < len(tokens):
        token = tokens[index]
        if COMMAND_NEWLINE_RE.search(command, previous_end, int(token["start"])):
            break
        value = str(token["value"])
        if token["type"] == "op" and (value != "`" or in_backtick):
            if "<" not in value and ">" not in value:
                break
            redirect_target = True
            previous_end = int(token["end"])
            index += 1
            continue
        end = _command_word_end(tokens, index, in_backtick)
        start = int(token["start"])
        if start in words.delimiters:
            if start in words.body_delimiters:
                heredocs.append(words.body_delimiters.index(start))
        elif not redirect_target:
            single = token if end == index + 1 and token["type"] == "word" else None
            args.append((command[start : int(tokens[end - 1]["end"])], single))
        redirect_target = False
        previous_end = int(tokens[end - 1]["end"])
        index = end
    return args, heredocs


def _word_value(arg: str, token: dict[str, object] | None) -> str:
    """A word as the shell reads it: its decoded token, else as written."""
    return str(token["value"]) if token is not None else arg


def _unwrap(
    name: str, args: list[tuple[str, dict[str, object] | None]], via: tuple[str, ...]
) -> list[SimpleCommand]:
    """The command and, through wrappers such as ``env`` or ``xargs``, the
    commands it runs."""
    found = [
        SimpleCommand(name, tuple(a for a, _ in args), tuple(t for _, t in args), via)
    ]
    if name not in WRAPPER_NAMES:
        return found
    takes_value = _WRAPPER_OPTIONS_WITH_VALUE.get(name, frozenset())
    skip = False
    for position, (arg, token) in enumerate(args):
        value = _word_value(arg, token)
        if skip:
            skip = False
            continue
        if value in takes_value:
            skip = True
            continue
        if name == "command" and value in ("-v", "-V"):
            return found  # only looks the name up
        if (
            value.startswith("-")
            or ASSIGNMENT_RE.match(value)
            or _WRAPPER_VALUE_RE.fullmatch(value)
        ):
            continue
        inner = re.split(r"[\\/]", value)[-1]
        return found + _unwrap(inner, args[position + 1 :], (*via, name))
    return found


def _shell_code(shell: SimpleCommand) -> tuple[int | None, bool]:
    """The argument index of the script a shell runs from ``-c``, and whether
    it runs code from stdin (no ``-c`` and no script file, or ``-s``). Only
    the options before the first operand count; with ``-n`` the shell checks
    the syntax and runs neither."""
    letters = ""
    skip = False
    first: int | None = None
    for position, (arg, token) in enumerate(zip(shell.args, shell.tokens, strict=True)):
        value = _word_value(arg, token)
        if skip:
            skip = False
        elif value in _SHELL_OPTIONS_WITH_VALUE:
            skip = True
        elif value in ("-", "--"):
            first = position + 1 if position + 1 < len(shell.args) else None
            break
        elif value.startswith("--") or value.startswith("+"):
            continue
        elif value.startswith("-") and len(value) > 1:
            letters += value[1:]
        else:
            first = position
            break
    if "n" in letters:
        return None, False
    if "c" in letters:
        return first, False
    return None, first is None or "s" in letters


def simple_commands(command: str) -> list[SimpleCommand]:
    """Every command *command* runs, with its arguments: at each command-name
    position, through wrappers, and inside substitutions, the ``-c`` scripts of
    shells it runs and the here-documents those shells read as code."""
    return _simple_commands(command, set(), ())


def _simple_commands(
    command: str, seen: set[tuple[str, tuple[str, ...]]], outer: tuple[str, ...]
) -> list[SimpleCommand]:
    """:func:`simple_commands`; *seen* holds the nested texts already read, by
    the wrappers they run under (*outer* for this one)."""
    words = parse(command)
    tokens = words.tokens
    found: list[SimpleCommand] = []
    nested: list[tuple[str, tuple[str, ...]]] = []  # text, wrappers it runs under
    script_tokens: set[int] = set()  # read as scripts, not again as quoted words
    code_bodies: set[int] = set()  # read as code, not again for substitutions
    in_backtick = False
    for index, (token, is_name) in enumerate(
        zip(tokens, words.name_flags, strict=True)
    ):
        value = str(token["value"])
        if token["type"] == "op" and value == "`":
            in_backtick = not in_backtick
        if (
            not is_name
            or value in COMMAND_PREFIX_WORDS
            or ASSIGNMENT_RE.match(str(token["raw"]))
        ):
            continue
        args, heredocs = _command_arguments(command, words, index, in_backtick)
        for each in _unwrap(re.split(r"[\\/]", value)[-1], args, ()):
            found.append(each)
            if each.name not in SHELL_NAMES:
                continue
            script, reads_stdin = _shell_code(each)
            if script is not None:
                nested.append(
                    (_word_value(each.args[script], each.tokens[script]), each.via)
                )
                if each.tokens[script] is not None:
                    script_tokens.add(int(each.tokens[script]["start"]))
            if reads_stdin:
                code_bodies.update(heredocs)
                for body in heredocs:
                    start, end, _expands = words.bodies[body]
                    text = command[start:end].rstrip("\n")
                    # Without the end line.
                    nested.append((text[: text.rfind("\n") + 1], each.via))

    def collect(body: str) -> tuple[str, bool]:
        nested.append((body, ()))
        return body, False

    for token in tokens:
        if (
            token["type"] == "word"
            and token["quoted"]
            and int(token["start"]) not in script_tokens
        ):
            rewrite_substitutions(str(token["raw"]), collect)
    for body, (start, end, expands) in enumerate(words.bodies):
        if expands and body not in code_bodies:
            rewrite_substitutions(command[start:end], collect, heredoc=True)
    for text, via in nested:
        chain = (*outer, *via)
        if (text, chain) in seen:
            continue  # the same code reached through another nesting
        seen.add((text, chain))
        found.extend(
            inner._replace(via=(*via, *inner.via))
            for inner in _simple_commands(text, seen, chain)
        )
    return found


def command_names(command: str) -> list[str]:
    """The basenames of the commands *command* runs (see :func:`simple_commands`)."""
    return [found.name for found in simple_commands(command)]
