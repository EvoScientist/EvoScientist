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
    {"env", "nohup", "timeout", "xargs", "exec", "command", "nice", "setsid"}
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


def _command_end(command: str, words: ShellWords, index: int, in_backtick: bool) -> int:
    """Where the command whose name is token *index* ends, its redirections
    included (the words :func:`_command_arguments` reads)."""
    tokens = words.tokens
    end = int(tokens[index]["end"])
    index += 1
    while index < len(tokens):
        token = tokens[index]
        value = str(token["value"])
        if command_newline(command, end, int(token["start"])) is not None or (
            token["type"] == "op"
            and (value != "`" or in_backtick)
            and "<" not in value
            and ">" not in value
        ):
            break
        if token["type"] == "op" and (value != "`" or in_backtick):
            index += 1
        else:  # a word, or a backtick substitution that starts one
            index = _command_word_end(tokens, index, in_backtick)
        end = int(tokens[index - 1]["end"])
    return end


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
    starting_folder: bool = False,
    folders: tuple[str, ...] = (),
) -> list[SimpleCommand]:
    """The command and, through wrappers such as ``env`` or ``xargs``, the
    commands it runs, in the ``env -C`` *folders* they run in. Strings that
    ``env -S`` splits are read in place and also go to *scripts*, with the
    wrappers and folders they run under, to be read as shell code. With
    *starting_folder*, an ``env -C`` runs nothing here."""
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
    rest = list(args)
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
            if changes_folder and starting_folder:
                return found
            if changes_folder:
                folders = (*folders, _env_folder(value, rest, position, takes_next))
            if takes_next:
                position += 1
            continue
        if value in takes_value:
            position += 1
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
        return found + _unwrap(
            inner, rest[position:], (*via, name), scripts, starting_folder, folders
        )
    return found


def _env_folder(
    value: str,
    args: list[tuple[str, dict[str, object] | None]],
    position: int,
    takes_next: bool,
) -> str:
    """The folder an ``env`` option *value* that changes it names: the next
    word in *args* (at *position*), or the rest of *value* (``-Cbuild``,
    ``-iCbuild``, ``--chdir=build``)."""
    if takes_next:
        return _word_value(*args[position]) if position < len(args) else ""
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
# cannot before a `pushd`, as each command starts with an empty stack.
_FOLDER_COMMANDS = frozenset({"cd", "pushd"})


def starting_folder_commands(command: str) -> list[SimpleCommand]:
    """The commands of *command* that run in the folder it starts in (see
    :func:`simple_commands`): each text read ends at its first ``cd`` or
    ``pushd``, and the command an ``env -C`` runs is left out. A folder change
    inside a substitution or a subshell also ends its text, which misses a
    command rather than adding one."""
    return _simple_commands(command, set(), (), starting_folder=True)


# Builtins whose `NAME=value` arguments set variables.
DECLARATION_NAMES = frozenset({"export", "local", "declare", "typeset", "readonly"})


def assignments(command: str) -> list[tuple[str, str]]:
    """``(name, value as written)`` of each variable *command* sets with
    ``NAME=value``: alone, before a command, as an argument of ``export`` /
    ``local`` / ``declare``, and in the code it runs (see
    :func:`simple_commands`)."""
    found: list[tuple[str, str]] = []
    for each in _simple_commands(command, set(), (), found):
        if each.name in DECLARATION_NAMES:
            found.extend(
                declared
                for declared in map(_declared_assignment, each.args)
                if declared is not None
            )
    return found


def _declared_assignment(arg: str) -> tuple[str, str] | None:
    """``(name, value as written)`` of a ``NAME=value`` argument of a
    declaration builtin, quoted or not (``"PID=$(lsof -t)"``, ``'PID'=x``).
    The quotes opened before the ``=`` stay on the value, so a substitution
    in it still runs exactly when the shell would run it."""
    equals = arg.find("=")
    if equals <= 0:
        return None
    head = arg[:equals]
    name = re.sub(r"[\"'\\]", "", head)
    if ASSIGNMENT_RE.fullmatch(f"{name}=") is None:
        return None
    return _assigned_name(name), re.sub(r"[^\"']", "", head) + arg[equals + 1 :]


def _assigned_name(word: str) -> str:
    """The variable a ``NAME=value``, ``NAME+=value`` or ``NAME[i]=value`` sets."""
    return re.split(r"[\[+=]", word, maxsplit=1)[0]


_REFERENCE_RE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
# Most values one word is read with, so a long chain of assignments stays cheap.
_VALUES_CAP = 16


def variable_references(word: str) -> set[str]:
    """The variables a word as written expands with ``$NAME`` or ``${NAME}``:
    outside single quotes and not escaped."""
    names: set[str] = set()
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
        elif char == "$":
            match = _REFERENCE_RE.match(word, index)
            if match is not None:
                names.add(match.group(1) or match.group(2))
        index += 1
    return names


def assigned_values(command: str) -> dict[str, tuple[str, ...]]:
    """The values *command* sets each variable to (see :func:`assignments`),
    in any order, as the shell reads them: quotes removed, and a ``$NAME`` of
    another variable set in *command* replaced by its values. A value with a
    substitution is only known when the command runs, so it is left out; any
    other ``$NAME`` (``$HOME``) stays as written."""
    written: dict[str, list[str]] = {}
    for name, value in assignments(command):
        written.setdefault(name, []).append(value)
    resolved: dict[str, tuple[str, ...]] = {}
    pending: set[str] = set()  # a reference back to one stays as written

    def resolve(name: str) -> tuple[str, ...]:
        if name in resolved:
            return resolved[name]
        pending.add(name)
        found: list[str] = []
        for value in written[name]:
            if "$(" in value or "`" in value:
                continue
            tokens = tokenize(value)
            if not tokens:
                found.append("")
                continue
            if len(tokens) != 1 or tokens[0]["type"] != "word":
                continue
            others = (variable_references(value) & written.keys()) - pending
            found.extend(
                expand_references(
                    str(tokens[0]["value"]),
                    {other: resolve(other) for other in others},
                )
            )
        pending.discard(name)
        resolved[name] = tuple(dict.fromkeys(found))[:_VALUES_CAP]
        return resolved[name]

    return {name: resolve(name) for name in written}


def expand_references(text: str, values: dict[str, tuple[str, ...]]) -> list[str]:
    """*text* with each ``$NAME`` / ``${NAME}`` of a variable in *values*
    replaced by each of its values (at most a few combinations); empty when
    one of them has none."""
    results = [""]
    last = 0
    for match in _REFERENCE_RE.finditer(text):
        name = match.group(1) or match.group(2)
        if name not in values:
            continue
        literal = text[last : match.start()]
        results = [
            result + literal + value for result in results for value in values[name]
        ][:_VALUES_CAP]
        last = match.end()
    return [result + text[last:] for result in results]


def word_values(
    word: str, value: str, assigned: dict[str, tuple[str, ...]]
) -> list[str] | None:
    """The values a word reads as with the variables *assigned* in its command
    (see :func:`assigned_values`): *value* is the word as the shell reads it,
    *word* as written. ``None`` when it names none of them."""
    names = variable_references(word) & assigned.keys()
    if not names:
        return None
    return expand_references(value, {name: assigned[name] for name in names})


def _simple_commands(
    command: str,
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]],
    outer: tuple[str, ...],
    assigned: list[tuple[str, str]] | None = None,
    starting_folder: bool = False,
    outer_folders: tuple[str, ...] = (),
) -> list[SimpleCommand]:
    """:func:`simple_commands`; *seen* holds the nested texts already read, by
    the wrappers and ``env -C`` folders they run under (*outer* and
    *outer_folders* for this one). Assignments in command-name position go to
    *assigned* (see :func:`assignments`). With *starting_folder*, see
    :func:`starting_folder_commands`."""
    words = parse(command)
    tokens = words.tokens
    found: list[SimpleCommand] = []
    # Text, and the wrappers and `env -C` folders it runs under.
    nested: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    # Tokens read as scripts, not again as quoted words; by identity, as an
    # `env -S` string's words carry offsets into that string, not *command*.
    script_tokens: set[int] = set()
    code_bodies: set[int] = set()  # read as code, not again for substitutions
    in_backtick = False
    for index, (token, is_name) in enumerate(
        zip(tokens, words.name_flags, strict=True)
    ):
        value = str(token["value"])
        if token["type"] == "op" and value == "`":
            in_backtick = not in_backtick
        if not is_name or value in COMMAND_PREFIX_WORDS:
            continue
        assignment = ASSIGNMENT_RE.match(str(token["raw"]))
        if assignment is not None:
            if assigned is not None:
                end = _command_word_end(tokens, index, in_backtick)
                text = command[int(token["start"]) : int(tokens[end - 1]["end"])]
                assigned.append((_assigned_name(text), text[assignment.end() :]))
            continue
        name = re.split(r"[\\/]", value)[-1]
        if starting_folder and name in _FOLDER_COMMANDS:
            # Its own words expand here first, read as the words of a `:`
            # (dropped); the commands after it run elsewhere, so read none of
            # their substitutions either.
            elsewhere = _command_end(command, words, index, in_backtick)
            own = ": " + command[int(token["end"]) : elsewhere]
            found.extend(_simple_commands(own, seen, outer, None, True)[1:])
            script_tokens.update(id(t) for t in tokens[index:])
            code_bodies.update(
                body
                for body, delimiter in enumerate(words.body_delimiters)
                if delimiter >= elsewhere
            )
            break
        args, heredocs = _command_arguments(command, words, index, in_backtick)
        for each in _unwrap(name, args, (), nested, starting_folder):
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
                # Ending the script at its first `cd` must not drop what the
                # outer shell expands in it first (`bash -c "cd x; $(...)"`).
                if each.tokens[script] is not None and not starting_folder:
                    script_tokens.add(id(each.tokens[script]))
            if reads_stdin:
                if not starting_folder:  # likewise an unquoted here-document
                    code_bodies.update(heredocs)
                for body in heredocs:
                    start, end, _expands = words.bodies[body]
                    text = command[start:end].rstrip("\n")
                    # Without the end line.
                    nested.append(
                        (text[: text.rfind("\n") + 1], each.via, each.folders)
                    )

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
    for text, via, folders in nested:
        chain = (*outer, *via)
        folder_chain = (*outer_folders, *folders)
        if (text, chain, folder_chain) in seen:
            continue  # the same code reached through another nesting
        seen.add((text, chain, folder_chain))
        found.extend(
            inner._replace(via=(*via, *inner.via), folders=(*folders, *inner.folders))
            for inner in _simple_commands(
                text, seen, chain, assigned, starting_folder, folder_chain
            )
        )
    return found


def command_names(command: str) -> list[str]:
    """The basenames of the commands *command* runs (see :func:`simple_commands`)."""
    return [found.name for found in simple_commands(command)]
