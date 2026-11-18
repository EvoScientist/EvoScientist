"""Enough POSIX shell syntax for the sandbox's command checks and path rewriting.

Words, quotes, comments, operators, here-documents, ``$(...)`` / backtick
substitutions and command-name positions; not a full shell parser.
"""

import functools
import itertools
import re
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

# Operators after which the next word is a command name.
COMMAND_START_OPS = frozenset({";", "&&", "||", "|", "&", "(", ")", "`"})
# Words after which the next word is still a command name.
COMMAND_PREFIX_WORDS = frozenset(
    {"do", "then", "else", "elif", "if", "while", "until", "!", "time", "{"}
)
_NEWLINE_RE = re.compile(r"(?<!\\)\n")


def command_newline(command: str, start: int, end: int) -> int | None:
    """The end of the first newline between two tokens (*start* to *end*)
    that ends a command, not a backslash line continuation. A backslash
    before *start* belongs to the word before it, as in ``x\\\\``, so it
    continues nothing; nor does one that ends a comment."""
    gap = command[start:end]
    comment = gap.find("#")  # before a newline, only a comment leaves a `#`
    match = _NEWLINE_RE.search(gap if comment == -1 else gap[:comment])
    if match is not None:
        return start + match.end()
    newline = -1 if comment == -1 else gap.find("\n", comment)
    return None if newline == -1 else start + newline + 1


ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
HEREDOC_OP_RE = re.compile(r"\d*<<")
SHELL_NAMES = frozenset({"sh", "bash", "zsh", "dash"})
SHELL_C_OPTION_RE = re.compile(r"-[A-Za-z]*c[A-Za-z]*")
# Commands that run the command named by their first operand.
WRAPPER_NAMES = frozenset(
    {"env", "nohup", "timeout", "xargs", "exec", "command", "builtin", "nice", "setsid"}
)
# Wrapper operands before the command: a duration or niceness (`5`, `1.5m`).
_WRAPPER_VALUE_RE = re.compile(r"-?\d+(\.\d+)?[smhd]?")
# Wrapper options whose value is the next word.
_WRAPPER_OPTIONS_WITH_VALUE = {
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
                    if (
                        inner == "\\"
                        and quote == '"'
                        and i + 1 < n
                        and command[i + 1] in '$`"\\\n'
                    ):
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
        newline = command_newline(command, prev_end, token_start)
        if pending and newline is not None:
            spans = _heredoc_spans(command, newline, pending)
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
            or command_newline(command, int(previous["end"]), int(token["start"]))
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
        options = iter(tokens[index + 1 :])
        for option in options:
            if option["type"] != "word":
                break
            raw = str(option["raw"])
            if not raw.startswith(("-", "+")):
                break
            if SHELL_C_OPTION_RE.fullmatch(raw):
                script = next(options, None)
                if script is not None and script["type"] == "word":
                    starts.add(int(script["start"]))
                break
            # -o/-O consume a separate option name, including in -euo.
            if re.fullmatch(r"[-+][A-Za-z]*[oO]", raw):
                name = next(options, None)
                if name is None or name["type"] != "word":
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
    # The folders `env -C` changes into before it runs, outermost first, as
    # the shell passes them to env.
    folders: tuple[str, ...] = ()
    # Inside a loop (its condition included) or a function body: it may run
    # again, or before the commands written ahead of it.
    repeats: bool = False


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
        if command_newline(command, previous_end, int(token["start"])) is not None:
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


# Short options of GNU and BSD env, other than `-S`, whose value is the rest
# of the word or the next word.
_ENV_VALUE_LETTERS = frozenset("aCPu")


def _env_split_string(
    value: str, args: list[tuple[str, dict[str, object] | None]], position: int
) -> tuple[str, int] | None:
    """The string an ``env`` option *value* splits into arguments (``-S``,
    ``-iS'...'``, ``--split-string[=]`` or an unambiguous prefix of it), and
    the position after it in *args*; ``None`` for any other word."""
    if value.startswith("--"):
        option, equals, attached = value.partition("=")
        if len(option) < 3 or not "--split-string".startswith(option):
            return None
        if equals:
            return attached, position
    elif value.startswith("-") and len(value) > 1:
        for index, letter in enumerate(value[1:], start=1):
            if letter in _ENV_VALUE_LETTERS:  # the rest of the word is its value
                return None
            if letter == "S":
                if value[index + 1 :]:
                    return value[index + 1 :], position
                break
        else:
            return None
    else:
        return None
    if position >= len(args):
        return None
    return _word_value(*args[position]), position + 1


def _env_option(value: str) -> tuple[bool, bool]:
    """For an ``env`` option word other than ``-S``: whether its value is the
    next word, and whether it changes the folder; short options also
    clustered (``-iC``), long ones also as a prefix (``--ch``)."""
    if value.startswith("--"):
        option, equals, _ = value.partition("=")
        for long in ("--argv0", "--chdir", "--env0-from", "--unset"):
            if len(option) > 2 and long.startswith(option):
                return not equals, long == "--chdir"
        return False, False
    for index, letter in enumerate(value[1:], start=1):
        if letter in _ENV_VALUE_LETTERS:
            return index == len(value) - 1, letter == "C"
    return False, False


# Escapes `env -S` turns into control characters, as GNU env documents.
_ENV_CONTROL_ESCAPES = {
    "\\n": "\n",
    "\\t": "\t",
    "\\r": "\r",
    "\\f": "\f",
    "\\v": "\v",
}


def _env_shell_text(text: str) -> str:
    """*text* as shell words that split like ``env -S`` splits it: env's
    ``\\_`` is a space (between words outside quotes, literal inside single
    quotes), ``\\n``, ``\\t``, ``\\r``, ``\\f`` and ``\\v`` are those characters
    inside the word, and an unquoted ``\\c`` ends the string."""
    shell_text: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote != "'" and char == "\\" and index + 1 < len(text):
            pair = text[index : index + 2]
            if pair == "\\c" and quote is None:
                break
            control = _ENV_CONTROL_ESCAPES.get(pair)
            if control is not None:
                # Single-quoted outside quotes, so it stays in the word.
                shell_text.append(control if quote else f"'{control}'")
            else:
                shell_text.append(" " if pair == "\\_" else pair)
            index += 2
            continue
        if char in "'\"" and quote in (None, char):
            quote = None if quote == char else char
        shell_text.append(char)
        index += 1
    return "".join(shell_text)


def _env_words(text: str) -> list[tuple[str, dict[str, object] | None]]:
    """The arguments ``env -S`` splits *text* into, with their tokens (see
    :func:`_env_shell_text`)."""
    joined = _env_shell_text(text)
    # Tokens with nothing between them are one argument; env does not read
    # shell operators, so `a;` is one too.
    groups: list[list[dict[str, object]]] = []
    for token in tokenize(joined):
        if groups and int(token["start"]) == int(groups[-1][-1]["end"]):
            groups[-1].append(token)
        else:
            groups.append([token])
    return [
        (
            joined[int(group[0]["start"]) : int(group[-1]["end"])],
            group[0] if len(group) == 1 and group[0]["type"] == "word" else None,
        )
        for group in groups
    ]


def _unwrap(
    name: str,
    args: list[tuple[str, dict[str, object] | None]],
    via: tuple[str, ...],
    scripts: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
    folders: tuple[str, ...] = (),
    assigned: list[tuple[str, str]] | None = None,
) -> list[SimpleCommand]:
    """The command and, through wrappers such as ``env`` or ``xargs``, the
    commands it runs, in the ``env -C`` *folders* they run in (the last
    ``-C`` of each ``env``, as GNU ``env`` changes folder once). Strings that
    ``env -S`` splits are read in place and also go to *scripts*, with the
    wrappers and folders they run under, to be read as shell code. The
    ``NAME=value`` words of ``env`` go to *assigned* (see
    :func:`assignments`)."""
    found = [
        SimpleCommand(
            name,
            tuple(a for a, _ in args),
            tuple(t for _, t in args),
            via,
            folders,
        )
    ]
    if name not in WRAPPER_NAMES:
        return found
    takes_value = _WRAPPER_OPTIONS_WITH_VALUE.get(name, frozenset())
    outer_folders = folders  # one env changes folder once, to its last -C
    rest = list(args)
    env_assigned: list[tuple[str, str]] = []
    scripts_before = len(scripts)
    position = 0
    while position < len(rest):
        value = _word_value(*rest[position])
        position += 1
        split = _env_split_string(value, rest, position) if name == "env" else None
        if split is not None:
            text, position = split
            scripts.append((_env_shell_text(text), (*via, name), folders))
            rest[position:position] = _env_words(text)
            continue
        if name == "env" and value.startswith("-"):
            takes_next, changes_folder = _env_option(value)
            if changes_folder:
                option = _braced_word_value(*rest[position - 1])
                folder = _env_folder(option, rest, position, takes_next)
                folders = (*outer_folders, folder)
            if takes_next:
                position += 1
            continue
        if value in takes_value:
            position += 1
            continue
        if name == "env" and assigned is not None:
            arg, token = rest[position - 1]
            declared = _declared_assignment(
                str(token["raw"]) if token is not None else arg
            )
            if declared is not None:
                env_assigned.append(declared)
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
        runs = _unwrap(inner, rest[position:], (*via, name), scripts, folders, assigned)
        if (
            assigned is not None
            and env_assigned
            and _keeps_prefix(runs, len(scripts) > scripts_before, "")
        ):
            assigned.extend(env_assigned)  # only the code it runs sees them
        return found + runs
    return found


def _braced_word_value(arg: str, token: dict[str, object] | None) -> str:
    """:func:`_word_value`, with each ``$NAME`` written ``${NAME}``."""
    braced = braced_value(str(token["raw"])) if token is not None else None
    return braced if braced is not None else _word_value(arg, token)


def _env_folder(
    value: str,
    args: list[tuple[str, dict[str, object] | None]],
    position: int,
    takes_next: bool,
) -> str:
    """The folder an ``env`` option *value* that changes it names: the next
    word in *args* (at *position*), or the rest of *value* (``-Cbuild``,
    ``-iCbuild``, ``--chdir=build``); a ``$NAME`` in it written ``${NAME}``
    (see :func:`braced_value`)."""
    if takes_next:
        return _braced_word_value(*args[position]) if position < len(args) else ""
    if value.startswith("--"):
        return value.partition("=")[2]
    return value[value.index("C") + 1 :]


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
        elif value.startswith("--"):
            continue
        elif value.startswith("+") and len(value) > 1:
            letters = "".join(c for c in letters if c not in value[1:])  # +n undoes -n
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


# Commands that change the folder the commands after them run in; `popd`
# only returns to a folder an earlier `pushd` left, as each command starts
# with an empty stack.
FOLDER_COMMANDS = frozenset({"cd", "pushd"})


# Builtins whose `NAME=value` arguments set variables.
DECLARATION_NAMES = frozenset({"export", "local", "declare", "typeset", "readonly"})


def assignments(command: str) -> list[tuple[str, str]]:
    """``(name, value as written)`` of each variable *command* sets with
    ``NAME=value``: alone, before a command, as an argument of ``export`` /
    ``local`` / ``declare`` or ``env``, and in the code it runs (see
    :func:`simple_commands`)."""
    return [(_assigned_name(head), value) for head, value in _variables(command)[0]]


# Declaration options that only export or mark read-only; any other (`-n`,
# `-a`, `-i`) changes what the value means.
_PLAIN_DECLARATION_OPTION_RE = re.compile(r"[-+][xrg]+")
# Builtins that set the variables named in their arguments from what the run
# reads.
_READ_SETTER_NAMES = frozenset({"read", "mapfile", "readarray"})
# Builtins that set them from their arguments in a way the check does not
# follow.
_OTHER_SETTER_NAMES = frozenset({"getopts", "let"})
# Builtins that run a string as code in this shell, where it may set any
# variable. A file run with `source` / `.` is code the check never sees in
# any case, so it is not one of them.
CODE_RUNNER_NAMES = frozenset({"eval"})
_NAME_RE = re.compile(r"[A-Za-z_]\w*")
# A printf `%n` conversion, with its flags, width, precision and length.
_PRINTF_N_RE = re.compile(r"%[-+ #0'*\d.]*(?:hh|h|ll|l|L|q|j|z|t)?n")


_SEQUENCE_RE = re.compile(
    r"(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?|([A-Za-z])\.\.([A-Za-z])"
)


def _brace_words(word: str) -> list[str] | None:
    """The words a brace expansion in *word*, as written, makes
    (``seed_{0..2}`` reads ``seed_0 seed_1 seed_2``, ``{a,b}x`` reads
    ``ax bx``); ``None`` for one it does not read: with quotes, an escape, an
    expansion or a brace that expands nothing, or with more words than
    :data:`_VALUES_CAP`."""
    if re.search(r"[\"'\\$`]", word):
        return None
    start = word.find("{")
    if start == -1:
        return None if "}" in word else [word]
    depth = 0
    commas: list[int] = []
    for end in range(start, len(word)):
        if word[end] == "{":
            depth += 1
        elif word[end] == "}":
            depth -= 1
            if not depth:
                break
        elif word[end] == "," and depth == 1:
            commas.append(end)
    else:
        return None
    body = word[start + 1 : end]
    if commas:
        bounds = [start, *commas, end]
        choices = [word[a + 1 : b] for a, b in itertools.pairwise(bounds)]
    else:
        sequence = _SEQUENCE_RE.fullmatch(body)
        if sequence is None:
            return None
        if sequence.group(4):
            first, last = ord(sequence.group(4)), ord(sequence.group(5))
            step = 1 if last >= first else -1
            choices = [chr(c) for c in range(first, last + step, step)]
        else:
            first, last = int(sequence.group(1)), int(sequence.group(2))
            step = abs(int(sequence.group(3) or 1)) or 1
            step = step if last >= first else -step
            if abs(last - first) // abs(step) >= _VALUES_CAP:
                return None
            width = max(len(sequence.group(1)), len(sequence.group(2)))
            padded = re.match(r"-?0\d", sequence.group(1)) or re.match(
                r"-?0\d", sequence.group(2)
            )
            choices = [
                str(n).zfill(width) if padded else str(n)
                for n in range(first, last + step, step)
            ]
    found: list[str] = []
    for choice in choices:
        expanded = _brace_words(word[:start] + choice + word[end + 1 :])
        if expanded is None or len(found) + len(expanded) > _VALUES_CAP:
            return None
        found.extend(expanded)
    return found


class _Variables(NamedTuple):
    """How *command* sets its variables (see :func:`_variables`)."""

    # Each `NAME=value` (see :func:`assignments`), with what is written
    # before its `=` (`NAME`, `NAME+` or `NAME[i]`), and each word of a
    # `for NAME in WORDS` loop whose words are written out.
    found: list[tuple[str, str]]
    # Set from what the run reads (`read`, `mapfile`, a loop over the words
    # of a substitution): values the check cannot know.
    run_only: set[str]
    # Set another way the check does not follow (`printf -v`, `declare -a`,
    # `${NAME:=value}`, `unset`, a loop over a variable's words), from values
    # it could read.
    in_part: set[str]
    # It may set any of them (a nameref, `eval`, a name only the run knows).
    opaque: bool


def _variables(command: str) -> _Variables:
    """How *command* sets its variables (see :class:`_Variables`). Options
    and names are read as the shell reads them (``declare "-n"``,
    ``read "D"``)."""
    found: list[tuple[str, str]] = []
    run_only: set[str] = set()
    in_part: set[str] = set()
    opaque = False
    for each in _simple_commands(command, set(), (), found):
        args = each.args
        words = [
            _word_value(arg, token)
            for arg, token in zip(args, each.tokens, strict=True)
        ]
        if each.name in CODE_RUNNER_NAMES:
            opaque = True
        elif each.name == "for" and args:
            loop_words = [word for word in args[2:] if word != "do"]
            if len(words) == 1:
                # `for NAME` alone: the arguments the shell was given, or a
                # loop whose `in WORDS` is on the next line.
                in_part.add(words[0])
            elif words[1:2] != ["in"]:
                run_only.add(words[0])  # the arguments the shell was given
            elif any(variable_references(word) for word in loop_words):
                in_part.add(words[0])  # a variable's words
            else:
                # Each word is a value, as of a `NAME=value`: one a
                # substitution prints only the run knows.
                values: list[str] | None = []
                for word in loop_words:
                    braced = _brace_words(word) if "{" in word else [word]
                    if braced is None or values is None:
                        values = None
                    else:
                        values.extend(braced)
                if values is None or len(values) > _VALUES_CAP:
                    in_part.add(words[0])  # a brace expansion not followed
                else:
                    found.extend((words[0], value) for value in values)
        elif each.name == "select" and args:
            # The word picked, or empty on a reply that picks none.
            in_part.add(words[0])
            run_only.add("REPLY")
        elif each.name == "unset":
            in_part.update(words)
        elif each.name in DECLARATION_NAMES:
            declared = [_declared_assignment(arg) for arg in args]
            plain = all(
                each_declared is not None
                or _NAME_RE.fullmatch(word)
                or _PLAIN_DECLARATION_OPTION_RE.fullmatch(word)
                for each_declared, word in zip(declared, words, strict=True)
            )
            if not plain and any(
                (word.startswith(("-", "+")) and "n" in word)
                or "$" in word
                or "`" in word
                for word in words
            ):
                opaque = True  # a nameref, or options only the run knows
            for each_declared, word in zip(declared, words, strict=True):
                if each_declared is not None:
                    found.append(each_declared)
                if not plain:
                    head = each_declared[0] if each_declared is not None else word
                    in_part.add(_assigned_name(head))
        elif each.name in _READ_SETTER_NAMES or each.name in _OTHER_SETTER_NAMES:
            names = {_assigned_name(word) for word in words}
            if each.name in _READ_SETTER_NAMES:
                run_only.update(names)
                # The variable set when no name is given.
                run_only.add("REPLY" if each.name == "read" else "MAPFILE")
            else:
                in_part.update(names)
                in_part.add("REPLY")
            if any("$" in word or "`" in word for word in words):
                opaque = True  # a name only the run knows
        elif each.name == "printf":
            # `%n` writes the count printed so far into a variable named by
            # an argument; which one is not followed.
            if any(_PRINTF_N_RE.search(word.replace("%%", "")) for word in words):
                opaque = True
            for at, word in enumerate(words):
                destination = None
                if word == "-v" and at + 1 < len(words):
                    destination = words[at + 1]
                elif word.startswith("-v") and len(word) > 2:
                    destination = word[2:]
                if destination is not None:
                    in_part.add(_assigned_name(destination))  # `D[0]` sets D
                    if "$" in destination or "`" in destination:
                        opaque = True  # a name only the run knows
    in_part.update(re.findall(r"\$\{([A-Za-z_]\w*)(?:\[[^\]]*\])?:?=", command))
    return _Variables(
        found,
        {name for name in run_only if _NAME_RE.fullmatch(name)},
        {name for name in in_part if _NAME_RE.fullmatch(name)},
        opaque,
    )


def _declared_assignment(arg: str) -> tuple[str, str] | None:
    """``(what is written before the =, value as written)`` of a
    ``NAME=value`` argument of a declaration builtin, quoted or not
    (``"PID=$(lsof -t)"``, ``'PID'=x``). The quotes opened before the ``=``
    stay on the value, so a substitution in it still runs exactly when the
    shell would run it."""
    equals = arg.find("=")
    if equals <= 0:
        return None
    head = arg[:equals]
    name = re.sub(r"[\"'\\]", "", head)
    if ASSIGNMENT_RE.fullmatch(f"{name}=") is None:
        return None
    return name, re.sub(r"[^\"']", "", head) + arg[equals + 1 :]


def _assigned_name(word: str) -> str:
    """The variable a ``NAME=value``, ``NAME+=value`` or ``NAME[i]=value`` sets."""
    return re.split(r"[\[+=]", word, maxsplit=1)[0]


_REFERENCE_RE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
# `${NAME` with an index or an operator: `${D[@]}`, `${D:-x}`, `${#D}`, `${!D}`.
_OPERATOR_REFERENCE_RE = re.compile(
    r"\$\{(?:[#!]([A-Za-z_]\w*)|([A-Za-z_]\w*)(?![\w}]))"
)
# `${NAME:?message}` / `${NAME?message}`: NAME's value, or an error.
_CHECKED_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_]\w*):?\?[^${}`]*\}")
# Most values one word is read with, so a long chain of assignments stays cheap.
_VALUES_CAP = 16


class UnknownValues(ValueError):
    """A word may read as values the check does not follow: more than
    :data:`_VALUES_CAP`, or of a variable set in a way it does not follow
    (see :func:`assigned_values`); *word* is it. *partial* when the check
    knows some of them but not all (``D=sk; D+=ills``), rather than none, as
    for a variable only the run sets (``read D``)."""

    def __init__(self, word: str, partial: bool = True) -> None:
        super().__init__(word)
        self.word = word
        self.partial = partial


class AssignedValues(dict[str, tuple[str, ...] | None]):
    """:func:`assigned_values`: each variable's values, ``None`` when they
    are not followed; *partial* holds those of them the check knows in part
    (see :class:`UnknownValues`)."""

    partial: frozenset[str] = frozenset()


def _references(word: str) -> list[tuple[re.Match[str], bool]]:
    """The ``$NAME`` / ``${NAME}`` of a word as written that the shell
    expands (outside single quotes and not escaped), each with whether it is
    inside double quotes. A name inside a ``$(...)`` or backtick
    substitution is the substitution's, whose output only the run knows, so
    it is not one of them."""
    found: list[tuple[re.Match[str], bool]] = []
    quote: str | None = None
    index = 0
    while index < len(word):
        char = word[index]
        if quote == "'":
            if char == "'":
                quote = None
        elif char == "\\":
            index += 1
        elif char == "'" and quote is None:
            quote = "'"
        elif char == '"':
            quote = None if quote == '"' else '"'
        elif char == "$" and word.startswith("$(", index):
            end = matching_paren(word, index + 2)
            if end is None:
                break
            index = end
        elif char == "`":
            end = word.find("`", index + 1)
            while end != -1 and word[end - 1] == "\\":
                end = word.find("`", end + 1)
            if end == -1:
                break
            index = end
        elif char == "$":
            match = _REFERENCE_RE.match(word, index)
            if match is not None:
                found.append((match, quote == '"'))
        index += 1
    return found


def variable_references(word: str) -> set[str]:
    """The variables a word as written expands with ``$NAME`` or ``${NAME}``:
    outside single quotes and not escaped."""
    return {match.group(1) or match.group(2) for match, _ in _references(word)}


def braced_value(word: str) -> str | None:
    """A word as written, as the shell reads it, with each ``$NAME`` it
    expands written ``${NAME}``, so quote removal keeps where the name ends
    (``"$D"ills`` reads ``${D}ills``); ``None`` when it is not one word."""
    parts: list[str] = []
    last = 0
    for match, _quoted in _references(word):
        name = match.group(1) or match.group(2)
        parts.append(word[last : match.start()] + "${" + name + "}")
        last = match.end()
    tokens = tokenize("".join(parts) + word[last:])
    if len(tokens) != 1 or tokens[0]["type"] != "word":
        return None
    return str(tokens[0]["value"])


def assigned_values(command: str, plain_only: bool = False) -> AssignedValues:
    """The values *command* sets each variable to with plain assignments
    (see :func:`assignments`), in any order, as the shell reads them: quotes
    removed, and a ``$NAME`` of another variable set in *command* replaced by
    its values. A value with a substitution is only known when the command
    runs, so it is left out; any other ``$NAME`` (``$HOME``) stays as
    written. ``None`` for a variable whose values the check does not follow:
    one set any other way (see :class:`_Variables`), every one when the
    command may set any variable (a nameref, ``eval``), one set in part
    (``NAME+=value``, ``NAME[i]=value``) or to an array, one whose value
    refers back to it (``D=$D/x``) or reads a variable with an operator
    (``${D%/}``), one with more than :data:`_VALUES_CAP` values, one with
    values known before the run and values only the run knows
    (``X=build; X=$(cmd)``), or one whose value names such a variable.
    Only one the run alone sets (``read D``, see :class:`_Variables`) is
    not in :attr:`AssignedValues.partial`. With *plain_only*, such a one is
    read from its plain assignments alone."""
    found, run_only, in_part, opaque = _variables(command)
    # Set in part, or to an array: the `=` values are only some of them.
    partial = {
        _assigned_name(head)
        for head, value in found
        if head != _assigned_name(head) or value.startswith("(")
    } | in_part
    if opaque:  # any of them may be set some other way
        partial.update(_assigned_name(head) for head, _ in found)
    # Read alone, the plain `=` values of one the run also sets are checked.
    other = set(partial) if plain_only else run_only | partial
    written: dict[str, list[str]] = {}
    for head, value in found:
        written.setdefault(_assigned_name(head), []).append(value)
    resolved: dict[str, tuple[str, ...] | None] = dict.fromkeys(other)
    pending: set[str] = set()

    def resolve(name: str) -> tuple[str, ...] | None:
        if name in resolved:
            return resolved[name]
        if name in pending:  # it refers back to itself
            raise UnknownValues(name)
        pending.add(name)
        try:
            values: tuple[str, ...] | None = resolve_values(name)
        except UnknownValues as exc:
            values = None
            if exc.partial:
                partial.add(name)
        pending.discard(name)
        resolved[name] = values
        return values

    def resolve_values(name: str) -> tuple[str, ...]:
        found: list[str] = []
        run_only = False  # a value only the run knows
        for value in written[name]:
            value = _CHECKED_REFERENCE_RE.sub(r"${\1}", value)
            # An operator form (`${X:-build}`) is read as written, as the
            # word it is in would be.
            if "$(" in value or "`" in value or _OPERATOR_REFERENCE_RE.search(value):
                run_only = True
                continue
            if not tokenize(value):
                found.append("")
                continue
            read = braced_value(value)
            if read is None:
                run_only = True
                continue
            names = variable_references(value) & (written.keys() | other)
            known = {each: resolve(each) for each in names}
            if any(known[each] is None for each in names):
                # Only the run knows it; just the run's value when the value
                # is that variable alone (`D=$R`), else known in part.
                raise UnknownValues(
                    name,
                    re.fullmatch(r"\$\{\w+\}", read) is None
                    or any(each in partial for each in names if known[each] is None),
                )
            expanded = expand_references(read, known)
            run_only = run_only or not expanded
            found.extend(expanded)
        values = tuple(dict.fromkeys(found))
        # Checking only the known values of a variable that may also take one
        # only the run knows would miss that one.
        if len(values) > _VALUES_CAP or (run_only and values):
            raise UnknownValues(name)
        return values

    for name in written:
        resolve(name)
    result = AssignedValues(resolved)
    result.partial = frozenset(partial)
    return result


def expand_references(
    text: str, values: dict[str, tuple[str, ...] | None]
) -> list[str]:
    """*text* with each ``$NAME`` / ``${NAME}`` of a variable in *values*
    replaced by each of its values; empty when one of them has none.
    :class:`UnknownValues` past :data:`_VALUES_CAP` combinations, or for a
    variable whose values are not followed (``None``)."""
    results = [""]
    last = 0
    for match in _REFERENCE_RE.finditer(text):
        name = match.group(1) or match.group(2)
        if name not in values:
            continue
        literal = text[last : match.start()]
        options = values[name]
        if options is None or len(results) * len(options) > _VALUES_CAP:
            raise UnknownValues(text)
        results = [result + literal + value for result in results for value in options]
        last = match.end()
    return [result + text[last:] for result in results]


def word_values(
    word: str, value: str, assigned: dict[str, tuple[str, ...] | None]
) -> list[str] | None:
    """The values a word reads as with the variables *assigned* in its command
    (see :func:`assigned_values`): *value* is the word as the shell reads it,
    *word* as written. ``None`` when it names none of them, and empty when
    one it names has no value known before the command runs; callers then
    read the word as written. A value of a name outside double quotes is
    split into the words it makes (``D="x skills"; rm -rf $D``).
    :class:`UnknownValues` (naming *word*) when it may read as more than
    :data:`_VALUES_CAP` values, names a variable whose values are not
    followed, reads one with an index or an operator (``${D[@]}``,
    ``${D%/}``; ``${D:?}`` reads as ``$D``), or splits one while the
    command sets ``IFS``."""
    word = _CHECKED_REFERENCE_RE.sub(r"${\1}", word)
    value = _CHECKED_REFERENCE_RE.sub(r"${\1}", value)
    # Known in part unless every unfollowed name it reads is one only the run
    # sets; a plain dict knows no such names.
    partial_names = getattr(assigned, "partial", None)

    def in_part(names: set[str]) -> bool:
        return partial_names is None or any(
            assigned[name] is not None or name in partial_names for name in names
        )

    operated = {
        match.group(1) or match.group(2)
        for match in _OPERATOR_REFERENCE_RE.finditer(word)
    } & assigned.keys()
    if operated:
        raise UnknownValues(word, in_part(operated))
    names = variable_references(word) & assigned.keys()
    if not names:
        return None
    value = braced_value(word) or value
    unquoted = any(
        not quoted
        for match, quoted in _references(word)
        if (match.group(1) or match.group(2)) in names
    )
    if unquoted and "IFS" in assigned:
        raise UnknownValues(word)
    unfollowed = {name for name in names if assigned[name] is None}
    if unfollowed:
        raise UnknownValues(word, in_part(unfollowed))
    try:
        values = expand_references(value, {name: assigned[name] for name in names})
    except UnknownValues:
        raise UnknownValues(word) from None
    if unquoted:
        values = [field for each in values for field in each.split() or [each]]
    return values


# Builtins after which a `NAME=value` written before them stays set (POSIX
# special builtins, and bash's own in POSIX mode).
SPECIAL_BUILTIN_NAMES = frozenset(
    {
        ":", ".", "break", "continue", "eval", "exec", "exit", "export",
        "readonly", "return", "set", "shift", "source", "times", "trap", "unset",
    }
)  # fmt: skip


def _keeps_prefix(
    unwrapped: list[SimpleCommand], adds_scripts: bool, command: str
) -> bool:
    """Whether a ``NAME=value`` written before a command, which sets the
    variable for that command only, still counts as one the shell sets:
    before a special builtin, a declaration, ``cd`` / ``pushd`` (which read
    ``HOME`` and ``CDPATH``) or a function, which keep or read it in this
    shell, or before a command that runs shell code the check reads, where
    the code sees it (``D=x bash -c 'rm -rf "$D"'``). Before any other
    command (``IFS= read``, ``OUT=x python``) the shell's own variable is
    unchanged."""
    name = unwrapped[0].name
    return (
        adds_scripts
        or name in SPECIAL_BUILTIN_NAMES
        or name in DECLARATION_NAMES
        or name in ("cd", "pushd")
        or re.search(
            rf"(?:^|[\s;&|(){{}}/]){re.escape(name)}\s*\(\s*\)|\bfunction\s+{re.escape(name)}\b",
            command,
        )
        is not None
        or any(
            each.name in SHELL_NAMES or each.name in SPECIAL_BUILTIN_NAMES
            for each in unwrapped
        )
    )


def _simple_commands(
    command: str,
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...], bool]],
    outer: tuple[str, ...],
    assigned: list[tuple[str, str]] | None = None,
    outer_folders: tuple[str, ...] = (),
    outer_repeats: bool = False,
    root: str | None = None,
) -> list[SimpleCommand]:
    """:func:`simple_commands`; *seen* holds the nested texts already read, by
    the wrappers and ``env -C`` folders they run under (*outer* and
    *outer_folders* for this one) and whether they repeat (*outer_repeats*,
    see :class:`SimpleCommand`); *root* is the outermost command, where a
    function a nested text calls may be defined. Assignments in command-name position go to
    *assigned* (see :func:`assignments`)."""
    root = command if root is None else root  # the whole command, for definitions
    words = parse(command)
    tokens = words.tokens
    found: list[SimpleCommand] = []
    # Text, and the wrappers and `env -C` folders it runs under.
    nested: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    # Tokens read as scripts, not again as quoted words; by identity, as an
    # `env -S` string's words carry offsets into that string, not *command*.
    script_tokens: set[int] = set()
    code_bodies: set[int] = set()  # read as code, not again for substitutions
    # Assignments written before a command name, by the index of its token.
    prefix: dict[int, list[tuple[str, str]]] = {}
    # The loops, function bodies and `{ }` groups open at the token being
    # read, and whether the next `{` opens a function body.
    blocks: list[str] = []
    function_next = False
    repeating: set[int] = set()  # indexes in *nested* of texts that repeat
    in_backtick = False
    for index, (token, is_name) in enumerate(
        zip(tokens, words.name_flags, strict=True)
    ):
        value = str(token["value"])
        if token["type"] == "op" and value == "`":
            in_backtick = not in_backtick
        if not is_name:
            continue
        # A keyword only as written: `"done"` or `\\done` runs a command.
        keyword = value if token.get("raw") == value else None
        after = tokens[index + 1] if index + 1 < len(tokens) else None
        if (
            blocks
            and blocks[-1] == "case"
            and after is not None
            and after["type"] == "op"
            and after["value"] in (")", "|")
        ):
            keyword = None  # a `case` pattern
        if keyword == "case":
            blocks.append("case")
        elif keyword == "esac":
            while blocks and blocks.pop() != "case":
                pass
        elif keyword in ("while", "until", "for", "select"):
            blocks.append("loop")
        elif keyword == "{":
            blocks.append("function" if function_next else "group")
            function_next = False
        elif keyword in ("done", "}") and blocks:
            opened = ("loop",) if keyword == "done" else ("function", "group")
            while blocks and blocks.pop() not in opened:
                pass
        repeats = outer_repeats or any(each in ("loop", "function") for each in blocks)
        if value in COMMAND_PREFIX_WORDS:
            if index in prefix:  # `D=x time cmd`: before the command after it
                prefix[index + 1] = prefix.pop(index)
            continue
        assignment = ASSIGNMENT_RE.match(str(token["raw"]))
        if assignment is not None:
            if assigned is not None:
                end = _command_word_end(tokens, index, in_backtick)
                text = command[int(token["start"]) : int(tokens[end - 1]["end"])]
                written = text[: assignment.end() - 1]
                chain = [*prefix.pop(index, []), (written, text[assignment.end() :])]
                if (
                    end < len(tokens)
                    and words.name_flags[end]
                    and tokens[end]["type"] == "word"
                    # A newline ends the command, as `;` does.
                    and command_newline(
                        command, int(tokens[end - 1]["end"]), int(tokens[end]["start"])
                    )
                    is None
                ):
                    prefix[end] = chain
                else:
                    assigned.extend(chain)  # alone, not before a command
            continue
        name = re.split(r"[\\/]", value)[-1]
        args, heredocs = _command_arguments(command, words, index, in_backtick)
        before = prefix.pop(index, [])
        # A function definition, `function NAME` or `NAME ( )`, is reported
        # as a command named `function`, as its body may run any number of
        # times, in any order with the commands around it.
        if name == "function" or [
            (str(token["type"]), str(token["value"]))
            for token in tokens[index + 1 : index + 3]
        ] == [("op", "("), ("op", ")")]:
            found.append(SimpleCommand("function", (), (), ()))
        if name != "function" and [
            (str(each["type"]), str(each["value"]))
            for each in tokens[index + 1 : index + 3]
        ] == [("op", "("), ("op", ")")]:
            following = tokens[index + 3] if index + 3 < len(tokens) else None
            if following is not None and following.get("raw") == "{":
                function_next = True  # its `{` is read as a command name next
            else:
                # `( BODY )`, `if`, `case`, ...: to the end of the text.
                blocks.append("function")
        if name == "function":
            # `function NAME [()] { BODY`: the body's first command is read
            # as these words.
            body = next(
                (at + 1 for at, (arg, _) in enumerate(args) if arg in ("{", "(")),
                len(args),
            )
            # Each argument is a slice of *command*, in order; a substitution
            # has no token of its own, so its place is found from its text.
            position = int(token["end"])
            body_start = None
            for at, (arg, _) in enumerate(args):
                position = command.index(arg, position)
                if at == body:
                    body_start = position
                position += len(arg)
            if body_start is not None:
                repeating.add(len(nested))
                nested.append((command[body_start:position], (), ()))
            # After `function NAME [( )]`, only a `{` waits for its `}`; any
            # other body repeats to the end of the text.
            at = index + 2
            if [
                (str(each["type"]), str(each["value"])) for each in tokens[at : at + 2]
            ] == [("op", "("), ("op", ")")]:
                at += 2
            following = tokens[at] if at < len(tokens) else None
            if (
                following is not None
                and following.get("raw") == "{"
                and words.name_flags[at]
            ):
                function_next = True  # the `{` is read as a command name next
            else:
                blocks.append("function")  # to its `}`, or to the end of the text
            continue
        scripts_before = len(nested)
        unwrapped = _unwrap(name, args, (), nested, (), assigned)
        if repeats:
            unwrapped = [each._replace(repeats=True) for each in unwrapped]
        if (
            assigned is not None
            and before
            and _keeps_prefix(unwrapped, len(nested) > scripts_before, root)
        ):
            assigned.extend(before)
        for each in unwrapped:
            found.append(each)
            if each.name not in SHELL_NAMES:
                continue
            script, reads_stdin = _shell_code(each)
            if script is not None:
                nested.append(
                    (
                        _word_value(each.args[script], each.tokens[script]),
                        each.via,
                        each.folders,
                    )
                )
                if each.tokens[script] is not None:
                    script_tokens.add(id(each.tokens[script]))
            if reads_stdin:
                code_bodies.update(heredocs)
                for body in heredocs:
                    start, end, _expands = words.bodies[body]
                    text = command[start:end].rstrip("\n")
                    # Without the end line.
                    nested.append(
                        (text[: text.rfind("\n") + 1], each.via, each.folders)
                    )
        if repeats:
            repeating.update(range(scripts_before, len(nested)))

    if assigned is not None:
        for chain in prefix.values():  # before no command name the parser marks
            assigned.extend(chain)

    def collect(body: str) -> tuple[str, bool]:
        nested.append((body, (), ()))  # runs before the command it is in
        return body, False

    for token in tokens:
        if (
            token["type"] == "word"
            and token["quoted"]
            and id(token) not in script_tokens
        ):
            rewrite_substitutions(str(token["raw"]), collect)
    for body, (start, end, expands) in enumerate(words.bodies):
        if expands and body not in code_bodies:
            rewrite_substitutions(command[start:end], collect, heredoc=True)
    for at, (text, via, folders) in enumerate(nested):
        chain = (*outer, *via)
        folder_chain = (*outer_folders, *folders)
        repeats = outer_repeats or at in repeating
        if (text, chain, folder_chain, repeats) in seen:
            continue  # the same code reached through another nesting
        seen.add((text, chain, folder_chain, repeats))
        found.extend(
            inner._replace(via=(*via, *inner.via), folders=(*folders, *inner.folders))
            for inner in _simple_commands(
                text, seen, chain, assigned, folder_chain, repeats, root
            )
        )
    return found


def command_names(command: str) -> list[str]:
    """The basenames of the commands *command* runs (see :func:`simple_commands`)."""
    return [found.name for found in simple_commands(command)]
