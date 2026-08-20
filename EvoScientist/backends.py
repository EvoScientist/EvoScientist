"""Custom backends for EvoScientist agent."""

import os
import posixpath
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    WriteResult,
)

from . import paths
from .cancellation import current_cancel_event

if TYPE_CHECKING:
    from langgraph.types import Command

# Reproduced here to dodge a circular import from .EvoScientist (the canonical
# SKILLS_DIR constant).
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"

# System path prefixes that should never appear in virtual paths.
# If the agent hallucinates an absolute system path, we block it.
_SYSTEM_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/tmp/",
    "/var/",
    "/etc/",
    "/opt/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/dev/",
    "/proc/",
    "/sys/",
    "/root/",
)

# Path-confinement patterns: keep the agent inside the workspace. These are
# bypassed in dangerous mode (real-filesystem access).
_PATH_PATTERNS = [
    r"~/",  # home directory
    r"\bcd\s+/",  # cd to absolute path
]
# Destructive patterns: catastrophic regardless of mode — always enforced.
_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\s+/",  # rm -rf with absolute path
]

# Dangerous commands that should never be executed
BLOCKED_COMMANDS = [
    "sudo",
    "chmod",
    "chown",
    "mkfs",
    "dd",
    "shutdown",
    "reboot",
]


_active_shell_processes_lock = threading.RLock()
_active_shell_processes: dict[threading.Event, set[subprocess.Popen[str]]] = {}
_PROCESS_DRAIN_GRACE_SECONDS = 1.0


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Force-stop a shell and its descendants without waiting for reaping."""
    # A completed Popen has already reaped its PID, which the OS may reuse.
    # Inspect the recorded state rather than calling poll(): an exited but
    # unreaped shell can still have live descendants in its process group.
    if process.returncode is not None:
        return

    try:
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP alone does not make terminate() recursive.
            # taskkill is the native way to stop the complete descendant tree.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _stop_collecting_process_output(process: subprocess.Popen[str]) -> None:
    """Close inherited pipes and reap *process* without blocking the caller."""
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass

    if process.poll() is None:
        threading.Thread(target=process.wait, daemon=True).start()


def cancel_active_shell_processes(event: threading.Event) -> None:
    """Terminate every active shell command associated with *event*."""
    with _active_shell_processes_lock:
        processes = tuple(_active_shell_processes.get(event, ()))
    for process in processes:
        _terminate_process_tree(process)


def _register_shell_process(
    event: threading.Event | None,
    process: subprocess.Popen[str],
) -> None:
    if event is None:
        return
    with _active_shell_processes_lock:
        _active_shell_processes.setdefault(event, set()).add(process)
        cancel_now = event.is_set()
    if cancel_now:
        _terminate_process_tree(process)


def _unregister_shell_process(
    event: threading.Event | None,
    process: subprocess.Popen[str],
) -> None:
    if event is None:
        return
    with _active_shell_processes_lock:
        processes = _active_shell_processes.get(event)
        if processes is None:
            return
        processes.discard(process)
        if not processes:
            _active_shell_processes.pop(event, None)


def _shell_token_spans(command: str) -> list[dict[str, object]]:
    """Tokenize enough shell syntax to find quoted SSH remote commands.

    This is intentionally small: it tracks words, quotes, and command
    separators, but does not try to be a full POSIX shell parser.
    """
    tokens: list[dict[str, object]] = []
    i = 0
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
        operator = read_operator(i)
        if operator is not None:
            tokens.append(
                {"type": "op", "value": operator, "start": i, "end": i + len(operator)}
            )
            i += len(operator)
            continue

        start = i
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

        tokens.append(
            {
                "type": "word",
                "value": "".join(value),
                "raw": command[start:i],
                "start": start,
                "end": i,
                "quoted": quoted,
            }
        )
    return tokens


# Commands that are dangerous as the RIGHT-HAND SIDE of a pipe (they consume
# piped data as code or ship it off-box). Everything else piping is normal.
_PIPE_NETWORKING_RHS = frozenset(
    {
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "curl",
        "wget",
        "telnet",
        "socat",
        "scp",
        "sftp",
        "rsync",
        "ftp",
    }
)
_PIPE_INTERPRETER_RHS = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "ash",
        "ksh",
        "fish",
        "python",
        "python2",
        "python3",
        "node",
        "bun",
        "deno",
        "ruby",
        "perl",
        "php",
        "lua",
        "iex",
        "elixir",
    }
)
_PIPE_DANGEROUS_RHS = _PIPE_INTERPRETER_RHS | _PIPE_NETWORKING_RHS


def check_dangerous_command(command: str) -> str | None:
    """Return a reason if *command* pipes output into an interpreter or a
    network tool, else ``None``.

    Deliberately narrow: this guards indirect prompt injection (the agent
    ingests untrusted web content and could be induced to run
    ``curl … | bash``). Everyday research shell — pipes into ``grep``/``head``,
    redirects, ``python -c``, ``..``/``~`` paths — is NOT flagged here.
    Workspace confinement stays in :func:`validate_command`.

    Only the token immediately after the pipe is inspected, so wrapper
    commands like ``env bash``, ``xargs bash``, or ``timeout 5 bash`` are
    not detected — this is a known limitation, not a bug to fix here.
    """
    after_pipe = False
    for token in _shell_token_spans(command):
        if token.get("type") == "op":
            value = token.get("value")
            if value == "|":
                after_pipe = True
            elif value == "&" and after_pipe:
                # `|&` (pipe stdout+stderr) tokenizes as `|` then `&`;
                # keep the pipe context open across the `&`.
                pass
            else:
                after_pipe = False
            continue
        if after_pipe:
            base = str(token.get("value", "")).split("/")[-1]
            # strip trailing version digits: python3.11 -> python, lua5.4 -> lua
            normalized = re.sub(r"[0-9.]+$", "", base) or base
            if base in _PIPE_DANGEROUS_RHS or normalized in _PIPE_DANGEROUS_RHS:
                kind = (
                    "networking tool"
                    if base in _PIPE_NETWORKING_RHS
                    or normalized in _PIPE_NETWORKING_RHS
                    else "interpreter"
                )
                return f"pipes output into {kind} '{base}'"
            after_pipe = False
    return None


class ActionDecision(StrEnum):
    """Outcome of the shell-action policy."""

    APPROVE = "approve"
    REJECT = "reject"
    PROMPT = "prompt"


@dataclass(frozen=True)
class ActionVerdict:
    """A decision plus the reason to show the user or feed back to the agent."""

    decision: ActionDecision
    reason: str = ""


def resolve_action_decision(
    command: str,
    *,
    auto_approve: bool = False,
    dangerous_mode: bool = False,
    allow_list: list[str] | None = None,
) -> ActionVerdict:
    """Single source of truth for approve / reject / prompt.

    Precedence:
      1. ``dangerous_mode`` — the user asked for full power; run everything.
      2. dangerous detection — pipe into interpreter/network.
      3. ``auto_approve`` — opt-out means *never prompt*: approve, or reject
         a dangerous command with a reason the agent can act on.
      4. ``allow_list`` — case-sensitive match on a whole command or a
         command-plus-space prefix; blank entries are ignored. Every segment of
         a chain (``a; b``, ``a && b``, ``a | b``) must match, so an allow-listed
         prefix cannot carry a non-listed command in behind it.
    """
    if dangerous_mode:
        return ActionVerdict(ActionDecision.APPROVE)

    reason = check_dangerous_command(command)

    if auto_approve:
        if reason:
            return ActionVerdict(ActionDecision.REJECT, reason)
        return ActionVerdict(ActionDecision.APPROVE)

    if reason:
        return ActionVerdict(ActionDecision.PROMPT, reason)

    if allow_list:
        prefixes = [p.strip() for p in allow_list if p.strip()]
        # Match on a token boundary so allow-listing `ls` does not also approve
        # `lsof` (case-sensitive, like the shell). Require EVERY segment of a
        # chain to match, so `ls; rm -rf x` cannot ride in on an allow-listed
        # `ls`. ``None`` means an unparseable construct (substitution/newline) —
        # decline rather than risk approving a hidden command.
        segments = _split_command_segments(command)
        if segments is not None:
            segments = segments or [command.strip()]
            if prefixes and all(
                any(seg == p or seg.startswith(p + " ") for p in prefixes)
                for seg in segments
            ):
                return ActionVerdict(ActionDecision.APPROVE)

    return ActionVerdict(ActionDecision.PROMPT)


def build_hitl_resume(interrupt_id: str, decisions: list[dict]) -> "Command":
    """Build a HITL resume Command keyed by interrupt_id.

    Keying by id (not the flat ``{"decisions": …}``) is REQUIRED whenever the
    graph has more than one pending interrupt — parallel sub-agents that each
    call ``execute`` do exactly that, and a flat resume raises
    ``RuntimeError: When there are multiple pending interrupts …``. Resuming a
    single id resolves that interrupt and re-parks the rest (they re-emit on the
    next stream), so callers drain them one at a time. Safe for N=1 too.
    """
    from langgraph.types import Command

    return Command(resume={interrupt_id: {"decisions": decisions}})


_SSH_OPTIONS_WITH_VALUE = {
    "-B",
    "-b",
    "-c",
    "-D",
    "-E",
    "-e",
    "-F",
    "-I",
    "-i",
    "-J",
    "-L",
    "-l",
    "-m",
    "-O",
    "-o",
    "-p",
    "-Q",
    "-R",
    "-S",
    "-W",
    "-w",
}


def _ssh_option_consumes_next(token: str) -> bool:
    """Return whether an SSH option token consumes the following argument."""
    if token in _SSH_OPTIONS_WITH_VALUE:
        return True
    return False


def _is_ssh_executable(token: str) -> bool:
    return token == "ssh"


def _is_shell_assignment(token: dict[str, object]) -> bool:
    raw = str(token.get("raw", token.get("value", "")))
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", raw) is not None


def _ssh_executable_index(words: list[dict[str, object]]) -> int | None:
    idx = 0
    while idx < len(words) and _is_shell_assignment(words[idx]):
        idx += 1
    if idx < len(words) and _is_ssh_executable(str(words[idx].get("value", ""))):
        return idx
    return None


def _is_single_quoted_word(token: dict[str, object]) -> bool:
    raw = str(token.get("raw", ""))
    return raw.startswith("'") and raw.endswith("'")


def _ssh_host_index(words: list[dict[str, object]], ssh_idx: int) -> int:
    """Return the index of the host argument (first non-option after ssh)."""
    idx = ssh_idx + 1
    while idx < len(words):
        value = str(words[idx].get("value", ""))
        if value == "--":
            idx += 1
            break
        if value.startswith("-") and value != "-":
            idx += 2 if _ssh_option_consumes_next(value) else 1
            continue
        break
    return idx


def _ssh_invocations(
    command: str,
) -> list[tuple[list[dict[str, object]], int, int, int | None, int]]:
    """Return SSH invocations as ``(words, ssh_idx, host_idx, remote_idx, extra)``.

    Examples:
        >>> [(i, h, r, e) for _, i, h, r, e in _ssh_invocations("ssh host")]
        [(0, 1, None, 0)]
        >>> [(i, h, r, e) for _, i, h, r, e in _ssh_invocations("ssh host 'pwd'")]
        [(0, 1, 2, 0)]
        >>> [(i, h, r, e) for _, i, h, r, e in _ssh_invocations('ssh host "pwd"')]
        [(0, 1, 2, 0)]
        >>> [(i, h, r, e) for _, i, h, r, e in _ssh_invocations("ssh host 'pwd' extra")]
        [(0, 1, 2, 1)]
        >>> [(i, h, r, e) for _, i, h, r, e in _ssh_invocations("cat x && ssh -p 22 host 'pwd'")]
        [(0, 3, 4, 0)]
        >>> _ssh_invocations("/tmp/ssh host 'pwd'")
        []
    """
    tokens = _shell_token_spans(command)
    invocations: list[tuple[list[dict[str, object]], int, int, int | None, int]] = []
    segment: list[dict[str, object]] = []

    def flush_segment() -> None:
        if not segment:
            return
        words = [tok for tok in segment if tok.get("type") == "word"]
        ssh_idx = _ssh_executable_index(words)
        if ssh_idx is None:
            return

        host_idx = _ssh_host_index(words, ssh_idx)
        remote_idx = host_idx + 1 if host_idx + 1 < len(words) else None
        remote_extra_argv_count = (
            max(0, len(words) - remote_idx - 1) if remote_idx is not None else 0
        )
        invocations.append(
            (words, ssh_idx, host_idx, remote_idx, remote_extra_argv_count)
        )

    for token in tokens:
        if token.get("type") == "op":
            flush_segment()
            segment = []
        else:
            segment.append(token)
    flush_segment()
    return invocations


def _ssh_remote_command_spans(command: str) -> list[tuple[int, int]]:
    """Return remote-command argv spans in SSH invocations.

    Only the supported single-quoted token after the destination host is treated
    as remote argv. Plain ``ssh host`` has no remote argv and returns no spans.

    Examples:
        >>> _ssh_remote_command_spans("ssh host 'ls /home/u/project'")
        [(9, 29)]
        >>> _ssh_remote_command_spans('ssh host "ls /home/u/project"')
        []
        >>> _ssh_remote_command_spans("cat /tmp/x && ssh host 'pwd'")
        [(23, 28)]
    """
    spans: list[tuple[int, int]] = []
    for words, ssh_idx, host_idx, remote_idx, _ in _ssh_invocations(command):
        # Mask the SSH executable path itself (e.g., /usr/bin/ssh) so
        # virtual path conversion doesn't rewrite it.
        spans.append(
            (
                int(words[ssh_idx]["start"]),
                int(words[ssh_idx]["end"]),
            )
        )

        if (
            host_idx < len(words)
            and remote_idx is not None
            and _is_single_quoted_word(words[remote_idx])
        ):
            spans.append(
                (
                    int(words[remote_idx]["start"]),
                    int(words[remote_idx]["end"]),
                )
            )
    return spans


def _mask_spans(
    command: str, spans: list[tuple[int, int]]
) -> tuple[str, dict[str, str]]:
    """Replace spans with placeholders and return the restoration map."""
    if not spans:
        return command, {}
    pieces: list[str] = []
    replacements: dict[str, str] = {}
    cursor = 0
    nonce = uuid.uuid4().hex
    for index, (start, end) in enumerate(sorted(spans)):
        if start < cursor:
            continue
        placeholder = f"__EVOSCI_SSH_REMOTE_{nonce}_{index}__"
        pieces.append(command[cursor:start])
        pieces.append(placeholder)
        replacements[placeholder] = command[start:end]
        cursor = end
    pieces.append(command[cursor:])
    return "".join(pieces), replacements


def _restore_spans(command: str, replacements: dict[str, str]) -> str:
    for placeholder, original in replacements.items():
        command = command.replace(placeholder, original)
    return command


def _mask_ssh_remote_commands(command: str) -> tuple[str, dict[str, str]]:
    """Mask supported SSH remote argv so local path logic can skip it.

    Examples:
        >>> _restore_spans(*_mask_ssh_remote_commands("ssh host 'pwd'"))
        "ssh host 'pwd'"
    """
    return _mask_spans(command, _ssh_remote_command_spans(command))


def _validate_ssh_remote_command_format(command: str) -> str | None:
    """Require SSH remote commands to be one single-quoted token after the host."""

    def error() -> str:
        return (
            "SSH remote commands must be passed as a single quoted argument, "
            "for example: ssh host 'cd /home/user/project && python train.py'."
        )

    for words, _, _, remote_idx, remote_extra_argv_count in _ssh_invocations(command):
        if remote_idx is None:
            continue
        if not _is_single_quoted_word(words[remote_idx]):
            return error()
        if remote_extra_argv_count:
            return error()
    return None


def _split_shell_commands(command: str) -> list[str]:
    """Split a compound shell command into individual base commands.

    Handles command-boundary shell operators tracked by ``_shell_token_spans``.
    Redirection operators are not boundaries; their operands are filenames, not
    commands.
    """
    command_boundaries = {"&&", "||", ";", "|", "&", "(", ")", "`"}
    base_commands: list[str] = []
    segment: list[str] = []

    def flush_segment() -> None:
        words = [token for token in segment if token]
        if words:
            base_commands.append(words[0])

    for token in _shell_token_spans(command):
        if token.get("type") == "op" and token.get("value") in command_boundaries:
            flush_segment()
            segment = []
        else:
            segment.append(str(token.get("value", "")))
    flush_segment()
    return base_commands


def _split_command_segments(command: str) -> list[str] | None:
    """Split a compound command into raw segment strings on command boundaries.

    Quote-aware (via ``_shell_token_spans``). Boundaries are ``;`` ``&&`` ``||``
    ``|`` ``&`` and grouping; redirections are not boundaries. Lets the allow-list
    clear a chain only when *every* segment is allow-listed, not just the leading
    one (``ls; rm -rf x`` must not ride in on an allow-listed ``ls``).

    Returns ``None`` when the command contains a construct this small tokenizer
    cannot safely reason about — command substitution (``$(...)`` or backticks,
    which run a hidden command even inside double quotes) or a newline separator —
    so the caller declines to allow-list it rather than approve a hidden command.
    Deliberately a substring over-approximation: a literal/quoted ``$(``, backtick,
    or newline also declines (a safe extra prompt, never a bypass). Quote/escape
    awareness is intentionally not attempted — that fragility caused the original
    chaining gap.
    """
    if "$(" in command or "`" in command or "\n" in command or "\r" in command:
        return None
    boundaries = {"&&", "||", ";", "|", "&", "(", ")"}
    segments: list[str] = []
    seg_start = 0
    for token in _shell_token_spans(command):
        if token.get("type") == "op" and token.get("value") in boundaries:
            seg = command[seg_start : int(token["start"])].strip()
            if seg:
                segments.append(seg)
            seg_start = int(token["end"])
    tail = command[seg_start:].strip()
    if tail:
        segments.append(tail)
    return segments


def _has_traversal_component(command: str) -> bool:
    """Check if command contains '..' as a path component (not substring)."""
    from pathlib import PurePosixPath

    for token in command.split():
        if ".." in PurePosixPath(token).parts:
            return True
    return False


def _collect_executable_positions(command: str) -> set[int]:
    """Return the string offsets of executable tokens (first token per segment).

    These are command names/paths that appear in executable position (e.g.
    ``/usr/bin/python`` in ``/usr/bin/python script.py``) and should not be
    treated as dangerous operand paths.  Also covers the argument position
    right after ``pip install`` / ``pip3 install`` (package path).
    """
    offsets: set[int] = set()
    for segment in re.split(r"\s*(?:&&|\|\||;)\s*", command):
        for pipe_seg in segment.split("|"):
            pipe_seg_stripped = pipe_seg.strip()
            if not pipe_seg_stripped:
                continue
            # Offset of this pipe segment within *command*
            seg_start = command.find(pipe_seg_stripped)
            try:
                tokens = shlex.split(pipe_seg_stripped)
            except ValueError:
                tokens = pipe_seg_stripped.split()
            if not tokens:
                continue
            # First token is the executable itself — mark its offset
            offsets.add(seg_start)
            # pip install <path> — mark the install-target token
            if (
                len(tokens) >= 3
                and tokens[0] in ("pip", "pip3")
                and tokens[1] == "install"
            ):
                # Find position of the 3rd token (the package arg) onwards
                rest = pipe_seg_stripped
                for t in tokens[:2]:
                    idx = rest.find(t)
                    rest = rest[idx + len(t) :]
                pkg_offset = seg_start + (len(pipe_seg_stripped) - len(rest.lstrip()))
                offsets.add(pkg_offset)
    return offsets


def _is_under_allowed_prefix(path: str, allow_prefixes: tuple[str, ...]) -> bool:
    """True if *path* equals a prefix or is a strict descendant.

    Boundary-aware: ``str.startswith`` alone would let ``/A/skills_evil``
    match the prefix ``/A/skills`` — anchoring on ``/`` blocks neighbour
    directories that merely share a name prefix.
    """
    for prefix in allow_prefixes:
        normalized = prefix.rstrip("/")
        # Skip empty/root prefixes: they'd reduce the check to startswith("/")
        # and admit every absolute path, silently disabling the allowlist.
        if not normalized:
            continue
        if path == normalized or path.startswith(normalized + "/"):
            return True
    return False


def _extract_all_paths(
    command: str,
    allow_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Extract potential file paths from a command, including inside quoted strings.

    Scans both shell tokens and string literals (single/double quoted) to find
    paths that start with system prefixes like /Users/, /etc/, /tmp/, etc.
    Skips paths in executable position (command name) and pip install targets.

    Paths matched by ``allow_prefixes`` (via ``_is_under_allowed_prefix``)
    are dropped.
    """
    exe_offsets = _collect_executable_positions(command)
    paths: list[str] = []
    # Pattern: match absolute paths starting with / followed by word chars, dots,
    # dashes, slashes. Looks inside quotes and unquoted tokens alike.
    # Excludes URL-like patterns (preceded by ://)
    path_re = re.compile(
        r"(?<![:=/.\w])"  # not preceded by :, =, /, ., or word char (avoid URLs, env vars, ./paths)
        r"(/(?:Users|home|tmp|var|etc|opt|usr|bin|sbin|dev|proc|sys|root)"
        r'(?:/[^\s\'",;|&<>)}\]]*)?)'  # rest of the path
    )
    for m in path_re.finditer(command):
        # Skip paths that land at an executable-position offset
        if m.start(1) in exe_offsets:
            continue
        extracted = m.group(1)
        if _is_under_allowed_prefix(extracted, allow_prefixes):
            continue
        paths.append(extracted)
    return paths


def validate_command(
    command: str,
    allow_prefixes: tuple[str, ...] = (),
    *,
    dangerous: bool = False,
) -> str | None:
    """
    Validate a shell command for safety.

    Args:
        command: Shell command string.
        allow_prefixes: Absolute path prefixes exempt from the system-path
            block list (matching rules in ``_is_under_allowed_prefix``).
        dangerous: When True (real-filesystem mode), skip the path-confinement
            checks (``..`` traversal, ``~/``/``cd /`` patterns, absolute system
            paths). Privileged commands (:data:`BLOCKED_COMMANDS`) and
            catastrophic patterns (:data:`_DESTRUCTIVE_PATTERNS`) are still
            enforced.

    Returns:
        None if command is safe, error message string if blocked.
    """
    # Path-confinement checks — skipped in dangerous mode.
    if not dangerous:
        # Check for '..' path traversal as a path component
        if _has_traversal_component(command):
            return (
                "Command blocked: contains '..' path traversal. "
                "All commands must operate within the workspace directory. "
                "Use relative paths (e.g., './file.py') instead."
            )

        for pattern in _PATH_PATTERNS:
            if re.search(pattern, command):
                return (
                    f"Command blocked: contains forbidden pattern '{pattern}'. "
                    f"All commands must operate within the workspace directory. "
                    f"Use relative paths (e.g., './file.py') instead."
                )

    # Catastrophic patterns (e.g. `rm -rf /`) — always enforced.
    for pattern in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            return (
                f"Command blocked: contains forbidden pattern '{pattern}'. "
                f"All commands must operate within the workspace directory. "
                f"Use relative paths (e.g., './file.py') instead."
            )

    # Check for dangerous commands (pipeline-aware) — always enforced.
    for base_cmd in _split_shell_commands(command):
        if base_cmd in BLOCKED_COMMANDS:
            return (
                f"Command blocked: '{base_cmd}' is not allowed in sandbox mode. "
                f"Only standard development commands are permitted."
            )

    # Absolute-system-path check — skipped in dangerous mode.
    # Catches attacks like: python -c "os.remove('/Users/foo/file')"
    if not dangerous:
        escaped_paths = _extract_all_paths(command, allow_prefixes=allow_prefixes)
        if escaped_paths:
            path_sample = escaped_paths[0]
            return (
                f"Command blocked: contains absolute system path '{path_sample}'. "
                f"All file operations must use relative paths within the workspace. "
                f"Use relative paths (e.g., './file.py') instead."
            )

    return None


def _subpath_under_mount(token: str, mount: str) -> str | None:
    """Return the subpath of *token* under *mount*, or ``None`` if not under it.

    Bare ``mount`` and ``mount + "/"`` both return ``""`` so the caller can
    join uniformly (``Path(tier) / ""`` is the tier itself).
    """
    if token == mount or token == mount + "/":
        return ""
    prefix = mount + "/"
    if token.startswith(prefix):
        return token[len(prefix) :]
    return None


def _skills_tier_paths() -> tuple[Path, Path | None, Path]:
    """``(USER, GLOBAL or None, BUILTIN)`` — the tier priority chain that
    ``MergedSkillsBackend._backends()`` honors. Single source of truth so
    the resolver and the backend can't silently drift out of order.
    """
    return (paths.USER_SKILLS_DIR, paths.GLOBAL_SKILLS_DIR, _BUILTIN_SKILLS_DIR)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _cmd_quote(s: str) -> str:
    """Quote *s* for cmd.exe using double-quote wrapping.

    cmd.exe strips outer double quotes; content between them is taken
    literally. Backslashes are not escape chars inside double quotes, so
    Windows paths pass through unchanged. Embedded ``"`` is escaped as
    ``\"``; bare paths with no shell-special chars need no quoting at all.

    .. note::

       ``%VAR%`` expansion is **not** neutralised here.  Variable expansion
       happens before quote processing in cmd.exe, and ``%%`` collapsing
       only occurs inside ``.bat``/``.cmd`` files — not via ``cmd /c``.
       This is acceptable because virtual-mount paths (skills, memories)
       should never contain percent signs in practice.

    Mirrors the role of :func:`shlex.quote` for the Windows shell so the
    sandbox command can pass a single token through :func:`subprocess.run`
    with ``shell=True`` (which on Windows invokes cmd.exe, not /bin/sh).
    """
    if not s:
        return '""'
    if not any(c in s for c in ' \t\n"&|<>^()'):
        return s
    return '"' + s.replace('"', '\\"') + '"'


def _platform_quote(s: str) -> str:
    """Quote *s* for the host's default shell.

    On POSIX, delegates to :func:`shlex.quote` (single-quote wrapping).
    On Windows, uses double-quote wrapping compatible with cmd.exe —
    see :func:`_cmd_quote`. The platform check is read at call time, so
    tests can swap it via ``monkeypatch.setattr(backends, "_is_windows", ...)``
    without mutating :mod:`sys` module state.
    """
    if _is_windows():
        return _cmd_quote(s)
    return shlex.quote(s)


def _resolve_virtual_mount_path(token: str) -> str | None:
    """Resolve a virtual mount token to a shell-safe token, or ``None`` when
    *token* is not a registered virtual mount.

    For ``/skills/...``: walks ``_skills_tier_paths()`` priority (USER →
    GLOBAL → BUILTIN), returning :func:`_platform_quote` of the first tier
    where the path exists. On miss, returns a workspace-relative
    ``./skills/<rel>`` form — agent typed a virtual path, so the shell error
    should reference a location they recognise (`USER_SKILLS_DIR` defaults to
    ``WORKSPACE_ROOT / "skills"``, which is also where ``MergedSkillsBackend``
    would write a new skill).

    For ``/memories/...``: single tier (``paths.MEMORIES_DIR``), always
    absolute and :func:`_platform_quote`-wrapped. Memories live outside the
    workspace, so a relative form would point at an unrelated location.
    """
    rel = _subpath_under_mount(token, "/skills")
    if rel is not None:
        for tier in _skills_tier_paths():
            if tier is None:
                continue
            candidate = Path(tier) / rel
            if candidate.exists():
                return _platform_quote(str(candidate))
        return _platform_quote("./skills/" + rel if rel else "./skills")

    rel = _subpath_under_mount(token, "/memories")
    if rel is not None:
        return _platform_quote(str(Path(paths.MEMORIES_DIR) / rel))

    return None


def _guard_bare_absolute(result: str | None) -> str | None:
    """If *result* is a bare absolute path (no surrounding quotes),
    single-quote it so the post-process regex won't re-rewrite it."""
    if result and result.startswith("/") and result == result.strip("'\""):
        return "'" + result + "'"
    return result


def _rewrite_quoted_path(
    path: str,
    workspace_name: str | None,
) -> str | None:
    """Return the shell-quoted replacement for *path* (the decoded
    content of a quoted ``"..."`` or ``'...'`` argument),
    or ``None`` if no rewrite applies.
    """
    if not path or "://" in path[max(0, len(path) - 10) :]:
        return None
    if not path.startswith("/"):
        return None

    resolved = _resolve_virtual_mount_path(path)
    if resolved is not None:
        return _guard_bare_absolute(resolved)  # already shlex.quoted

    # Fix hallucinated system absolute paths that reference the workspace.
    if workspace_name:
        for prefix in _SYSTEM_PATH_PREFIXES:
            if path.startswith(prefix):
                marker = f"/{workspace_name}/"
                idx = path.rfind(marker)
                if idx != -1:
                    relative = path[idx + len(marker) :]
                    return _guard_bare_absolute(
                        shlex.quote("./" + relative if relative else ".")
                    )
                if path.endswith(f"/{workspace_name}"):
                    return _guard_bare_absolute(shlex.quote("."))
                break

    return None


def convert_virtual_paths_in_command(
    command: str,
    workspace_name: str | None = None,
) -> str:
    """Convert virtual paths (starting with ``/``) in commands to relative paths.

    Also auto-corrects hallucinated system absolute paths that reference the
    workspace directory (e.g. ``/Users/.../myproject/file.py`` → ``./file.py``).

    Pre-process: quoted arguments whose content resolves to a virtual
    mount (``/skills/...``, ``/memories/...``) or a workspace-prefixed
    system path are rewritten as a single shell token — this fixes #237
    where ``python "/skills/my skill/main.py"`` was truncated at the
    embedded space.  Bare quoted ``/...`` paths (e.g. ``echo "/hi"``)
    are left untouched since their semantics are ambiguous.
    After pre-processing, the original regex handles unquoted
    paths and workspace-name correction as before.
    """
    # Pre-process: rewrite quoted paths whose decoded content starts with /
    command = re.sub(
        r'(["\'])((?:\\.|(?!\1).)*?)\1',
        lambda m: (
            _rewrite_quoted_path(
                re.sub(r"\\(.)", r"\1", m.group(2)),
                workspace_name,
            )
            or m.group(0)
        ),
        command,
    )

    def replace_virtual_path(match: re.Match[str]) -> str:
        path = match.group(0)

        # Skip content that looks like a URL
        if "://" in command[max(0, match.start() - 10) : match.end() + 10]:
            return path

        resolved = _resolve_virtual_mount_path(path)
        if resolved is not None:
            return resolved

        # Fix hallucinated system absolute paths that reference the workspace.
        if workspace_name:
            for prefix in _SYSTEM_PATH_PREFIXES:
                if path.startswith(prefix):
                    marker = f"/{workspace_name}/"
                    idx = path.rfind(marker)
                    if idx != -1:
                        relative = path[idx + len(marker) :]
                        return "./" + relative if relative else "."
                    elif path.endswith(f"/{workspace_name}"):
                        return "."
                    break  # Matched system prefix but no workspace → fall through

        # Convert virtual path
        if path == "/":
            return "."
        return "." + path

    # Match pattern: paths starting with / (but not URLs)
    pattern = r'(?<=\s)/[^\s;|&<>\'"`]*|^/[^\s;|&<>\'"`]*'
    converted = re.sub(pattern, replace_virtual_path, command)

    return converted


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """
    Read-only filesystem backend.

    Allows read, ls, grep, glob operations but blocks write, edit, and upload.
    Used for skills directory — agent can read skill definitions but cannot
    modify them.
    """

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(
            error="This directory is read-only. Write operations are not permitted here."
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(
            error="This directory is read-only. Edit operations are not permitted here."
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(
                path=file_path,
                error="This directory is read-only. Upload operations are not permitted here.",
            )
            for file_path, _ in files
        ]

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(
            error="This directory is read-only. Delete operations are not permitted here."
        )


class MemoryFilesystemBackend(FilesystemBackend):
    """Filesystem backend for memory files with structured-write enforcement.

    Agents may read memory files and edit existing profile notes, but raw file
    creation is blocked so observations are recorded through memory tools.
    """

    _RAW_WRITE_ERROR = (
        "Raw writes to /memories are blocked. Edit existing "
        "/memories/profile/... files or use memory tools."
    )
    _RAW_EDIT_ERROR = (
        "Raw edits under /memories are limited to existing "
        "/memories/profile/... files. Use memory tools for observations."
    )
    _RAW_DELETE_ERROR = (
        "Deletes under /memories are blocked. Manage memory files through "
        "memory tools instead."
    )

    @staticmethod
    def _is_profile_path(file_path: str) -> bool:
        normalized = posixpath.normpath("/" + file_path.strip().lstrip("/"))
        return normalized == "/profile" or normalized.startswith("/profile/")

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=self._RAW_WRITE_ERROR)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if not self._is_profile_path(file_path):
            return EditResult(error=self._RAW_EDIT_ERROR)
        return super().edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=file_path, error=self._RAW_WRITE_ERROR)
            for file_path, _ in files
        ]

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=self._RAW_DELETE_ERROR)


def build_memory_agent_backend(
    *,
    workspace_dir: str | Path,
    memory_dir: str | Path,
):
    """Build the standard memory-agent backend with guarded `/memories/` routing."""
    from deepagents.backends import CompositeBackend

    return CompositeBackend(
        default=FilesystemBackend(root_dir=str(workspace_dir), virtual_mode=True),
        routes={
            "/memories/": MemoryFilesystemBackend(
                root_dir=str(memory_dir),
                virtual_mode=True,
            )
        },
    )


def build_memory_worker_backend(
    *,
    workspace_dir: str | Path,
    memory_dir: str | Path,
):
    """Build the memory-worker backend.

    Workers may update profile memory through /memories/profile/... and write
    observations through structured tools. The workspace itself is read-only.
    """
    from deepagents.backends import CompositeBackend

    return CompositeBackend(
        default=ReadOnlyFilesystemBackend(
            root_dir=str(workspace_dir),
            virtual_mode=True,
        ),
        routes={
            "/memories/": MemoryFilesystemBackend(
                root_dir=str(memory_dir),
                virtual_mode=True,
            )
        },
    )


def build_autoskill_agent_backend(
    *,
    memory_dir: str | Path,
    proposals_dir: str | Path,
    sandbox_timeout: int = 300,
):
    """Build the AutoSkills backend.

    AutoSkills has a different security model from ordinary memory
    maintenance: it can read memories and installed skills, write proposal
    folders, and run shell validation from the proposal root.
    """
    from deepagents.backends import CompositeBackend

    return CompositeBackend(
        default=AutoskillProposalSandboxBackend(
            root_dir=str(proposals_dir),
            timeout=sandbox_timeout,
        ),
        routes={
            "/memories/": ReadOnlyFilesystemBackend(
                root_dir=str(memory_dir),
                virtual_mode=True,
            ),
            "/skills/": MergedSkillsBackend(
                primary_dir=str(paths.USER_SKILLS_DIR),
                global_dir=str(paths.GLOBAL_SKILLS_DIR),
                secondary_dir=str(_BUILTIN_SKILLS_DIR),
                writable_primary=False,
            ),
            "/autoskill-proposals/": FilesystemBackend(
                root_dir=str(proposals_dir),
                virtual_mode=True,
            ),
        },
    )


class MergedSkillsBackend(BackendProtocol):
    """Skills backend that merges up to three skill directories.

    Priority (high → low):
    1. primary   — workspace/skills/  (project-local, writable)
    2. global    — ~/.evoscientist/skills/  (user global, read-only)
    3. secondary — EvoScientist/skills/  (built-in, PyPI, read-only)

    Higher-priority skills override lower-priority skills with the same name.
    All directories share the same virtual path namespace (/skills/).
    Only the workspace tier (primary) allows write and edit operations.
    """

    def __init__(
        self,
        primary_dir: str,
        secondary_dir: str,
        global_dir: str | None = None,
        writable_primary: bool = True,
    ):
        primary_backend = (
            FilesystemBackend if writable_primary else ReadOnlyFilesystemBackend
        )
        self._primary = primary_backend(root_dir=primary_dir, virtual_mode=True)
        self._global = (
            ReadOnlyFilesystemBackend(root_dir=global_dir, virtual_mode=True)
            if global_dir
            else None
        )
        self._secondary = ReadOnlyFilesystemBackend(
            root_dir=secondary_dir, virtual_mode=True
        )

    def _backends(self):
        """Yield backends in priority order: primary → global → secondary."""
        yield self._primary
        if self._global:
            yield self._global
        yield self._secondary

    # -- read: try each tier in priority order --

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        for backend in list(self._backends())[:-1]:
            try:
                result = backend.read(file_path, offset, limit)
                if hasattr(result, "error"):
                    if result.error is None:
                        return result
                elif not str(result).startswith("Error:"):
                    return result
            except (ValueError, FileNotFoundError, OSError):
                pass
        return self._secondary.read(file_path, offset, limit)

    # -- ls: merge all tiers, higher priority wins on name conflicts --

    def ls(self, path: str = "/") -> LsResult:
        merged: dict = {}
        for backend in reversed(list(self._backends())):
            result = backend.ls(path)
            for item in result.entries or []:
                merged[item["path"]] = item
        return LsResult(entries=sorted(merged.values(), key=lambda x: x["path"]))

    # -- grep: search all tiers --

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        matches = []
        for backend in self._backends():
            try:
                result = backend.grep(pattern, path, glob)
                matches.extend(result.matches or [])
            except Exception:
                pass
        return GrepResult(matches=matches)

    # -- glob: merge all tiers, higher priority wins on name conflicts --

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        merged: dict = {}
        for backend in reversed(list(self._backends())):
            try:
                result = backend.glob(pattern, path)
                for item in result.matches or []:
                    merged[item["path"]] = item
            except Exception:
                pass
        return GlobResult(matches=sorted(merged.values(), key=lambda x: x["path"]))

    # -- write / edit: only workspace/skills/ (primary) is writable --

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._primary.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._primary.edit(file_path, old_string, new_string, replace_all)

    # -- download / upload --

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, trying each tier in priority order."""
        backends = list(self._backends())
        responses: list[FileDownloadResponse] = []
        for path in paths:
            resp = backends[-1].download_files([path])[0]
            for backend in backends[:-1]:
                candidate = backend.download_files([path])[0]
                if candidate.error is None:
                    resp = candidate
                    break
            responses.append(resp)
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._primary.upload_files(files)


def prepare_sandbox_command(
    command: str,
    cwd: str | Path,
    *,
    virtual_mode: bool = True,
    dangerous: bool = False,
    guard_dangerous: bool = False,
) -> tuple[str, str | None]:
    """Normalize workspace paths in ``command`` and validate it for the sandbox.

    Shared by :meth:`CustomSandboxBackend.execute` and the background-process tools so
    both enforce *identical* workspace-path rewriting (so virtual ``/`` paths resolve to
    the workspace, not the host root) and the same command validation.

    Returns ``(prepared_command, error)``: ``error`` is a message string when the command
    is rejected (the caller must NOT run it), otherwise ``None``.

    ``guard_dangerous`` (see :func:`check_dangerous_command`) does not see inside an SSH
    remote payload: a dangerous pipe *inside* a quoted ``ssh host '...'`` argument is not
    detected, because the quoted payload is a single opaque token. Piping *into* ``ssh``
    itself (e.g. ``cat secret | ssh host x``) is detected — the check runs on the original,
    unmasked command so the SSH-masking done below (which also replaces the literal ``ssh``
    token) does not blind it.
    """
    original_command = command

    ssh_error = _validate_ssh_remote_command_format(command)
    if ssh_error:
        return command, ssh_error

    command, ssh_replacements = _mask_ssh_remote_commands(command)

    cwd_str = str(cwd).rstrip("/")
    # Replace literal workspace-root absolute paths with ./ after SSH masking so
    # remote paths that happen to contain the local cwd are preserved, and before
    # validation so local workspace paths are sanitized before the system-path
    # check fires. Skipped in dangerous mode: there is no virtual workspace, the
    # agent uses real absolute paths, and rewriting would corrupt any argument
    # (echo text, grep/git pattern) that merely contains the cwd string.
    if not dangerous:
        ws = cwd_str + "/"
        if ws in command:
            command = command.replace(ws, "./")
    if virtual_mode:
        command = convert_virtual_paths_in_command(
            command=command,
            workspace_name=Path(cwd_str).name,
        )
    # Skills/memory dirs must be allowlisted: the workspace-literal replace above runs
    # before the resolver, so any absolute path it later injects reaches validate unstripped.
    allow_prefixes = (
        str(paths.USER_SKILLS_DIR),
        str(paths.GLOBAL_SKILLS_DIR),
        str(paths.MEMORIES_DIR),
        str(_BUILTIN_SKILLS_DIR),
    )
    error = validate_command(
        command, allow_prefixes=allow_prefixes, dangerous=dangerous
    )
    if error:
        return command, error

    # No interactive approval is reachable here (unattended main agent, or an
    # async sub-agent on a remote thread), so refuse the narrow dangerous set
    # with a reason the agent can act on rather than running it blind.
    if guard_dangerous and not dangerous:
        dangerous_reason = check_dangerous_command(original_command)
        if dangerous_reason:
            return _restore_spans(command, ssh_replacements), (
                f"Command blocked: {dangerous_reason}. "
                f"Rewrite it to avoid that, or request approval from the user "
                f"(the orchestrator can re-issue it after approval)."
            )

    return _restore_spans(command, ssh_replacements), None


class CustomSandboxBackend(LocalShellBackend):
    """
    Custom sandbox backend - inherits LocalShellBackend with added safety.

    Features:
    - Inherits all file operations (ls, read, write, edit, grep, glob)
    - Inherits shell command execution with output truncation and timeout
    - Adds command validation to prevent directory traversal and dangerous operations
    - Adds path sanitization to auto-correct common LLM path mistakes
    - Compatible with LangGraph checkpointer (no thread locks)
    """

    def __init__(
        self,
        root_dir: str = ".",
        *,
        virtual_mode: bool = True,
        timeout: int = 300,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        dangerous: bool = False,
        guard_dangerous: bool = False,
        refuse_delete: bool = False,
    ):
        """
        Initialize custom sandbox backend.

        Args:
            root_dir: File system root directory
            virtual_mode: Whether to enable virtual path mode
            timeout: Command execution timeout in seconds
            max_output_bytes: Max output size before truncation (default 100KB)
            env: Extra environment variables for subprocess
            inherit_env: Whether to inherit parent process env (default True)
            dangerous: Real-filesystem mode — the agent operates on real absolute
                paths anywhere on disk (no workspace confinement). Forces
                ``virtual_mode=False`` and relaxes path validation while keeping
                the privileged-command blocklist. Defaults to False.
            guard_dangerous: Refuse the narrow dangerous-command set (see
                :func:`check_dangerous_command`) outright, for contexts where
                no interactive approval is reachable (unattended auto-approve
                runs, async sub-agents). Bypassed when ``dangerous=True``.
                Defaults to False.
            refuse_delete: Refuse the recursive ``delete`` FS tool outright,
                relaying an approval request to the orchestrator. Used for async
                research sub-agents (writing / data-analysis) that have no
                interactive approval path. Bypassed when ``dangerous=True``.
                Defaults to False.
        """
        self._dangerous = dangerous
        self._guard_dangerous = guard_dangerous
        self._refuse_delete = refuse_delete
        if dangerous:
            # Real paths require the legacy (non-virtual) resolution path so the
            # parent backend returns absolute paths as-is.
            virtual_mode = False
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=env,
            inherit_env=inherit_env,
        )
        # Override parent's "local-" prefix with our own
        self._sandbox_id = f"evosci-{uuid.uuid4().hex[:8]}"
        # Ensure working directory exists
        os.makedirs(str(self.cwd), exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve path with sanitization to prevent nested directories.

        Intercepts all file operations (read, write, edit, ls, grep, glob).
        Auto-corrects common LLM path mistakes instead of crashing:
          1. /Users/.../<cwd>/file.py      → /file.py (full cwd match — safest)
          2. /<ws_name>/file.py            → /file.py
          3. /Users/name/.../<ws_name>/f   → /f  (strip at LAST <ws_name>/)
          4. /Users/name/file.py           → /file.py (keep basename)

        In dangerous (real-filesystem) mode, skip all rewriting and let the
        parent resolve real absolute paths as-is.
        """
        if self._dangerous:
            return super()._resolve_path(key)

        cwd_str = str(self.cwd).rstrip("/")
        ws_name = Path(cwd_str).name  # e.g. "workspace", "my-project"

        # Prefer the full cwd match so a parent path that happens to contain
        # "/<ws_name>/" (e.g. cwd = /Users/u/workspace/.../workspace) doesn't
        # confuse the basename-based fallback below.
        if key == cwd_str:
            return super()._resolve_path("/")
        if key.startswith(cwd_str + "/"):
            return super()._resolve_path("/" + key[len(cwd_str) + 1 :])

        # Auto-strip /<ws_name>/ prefix to prevent nesting
        ws_prefix = f"/{ws_name}/"
        if key.startswith(ws_prefix):
            key = key[len(ws_prefix) - 1 :]  # "/<ws>/main.py" → "/main.py"
        elif key == f"/{ws_name}":
            key = "/"

        # Auto-correct system absolute paths
        for prefix in _SYSTEM_PATH_PREFIXES:
            if key.startswith(prefix):
                # rfind, not find: the cwd's parent path may itself contain
                # "/<ws_name>/" as a substring, and we want the boundary
                # nearest the file — the workspace mount.
                idx = key.rfind(ws_prefix)
                if idx != -1:
                    key = "/" + key[idx + len(ws_prefix) :]
                elif key.endswith(f"/{ws_name}"):
                    key = "/"
                else:
                    # Fall back to basename
                    key = "/" + Path(key).name
                break

        return super()._resolve_path(key)

    _DELETE_APPROVAL_ERROR = (
        "Delete blocked: needs approval. Report it to the orchestrator, which "
        "can re-issue it after approval."
    )

    def delete(self, file_path: str) -> DeleteResult:
        """Refuse ``delete`` for guarded async sub-agents (no approval path).

        No ``adelete`` override is needed: the inherited ``BackendProtocol.adelete``
        runs ``asyncio.to_thread(self.delete, ...)``, so async sub-agents reach
        this refusal too.
        """
        if self._refuse_delete and not self._dangerous:
            return DeleteResult(error=self._DELETE_APPROVAL_ERROR)
        return super().delete(file_path)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """
        Execute shell command in sandbox environment.

        Commands are validated before execution to prevent:
        - Directory traversal (../)
        - Access to paths outside workspace
        - Dangerous system commands

        The validated command is handed to the owned process runner so
        cancelling an agent turn can terminate the complete process tree.
        """
        # Preserve LocalShellBackend's public validation contract.  This
        # override cannot delegate execution to the base implementation because
        # it must retain the Popen handle for cancellation, so validate before
        # command preparation and process launch instead.
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        command, error = prepare_sandbox_command(
            command,
            self.cwd,
            virtual_mode=self.virtual_mode,
            dangerous=self._dangerous,
            guard_dangerous=self._guard_dangerous,
        )
        if error:
            return ExecuteResponse(output=error, exit_code=1, truncated=False)

        return self._execute_prepared_command(command, timeout=timeout)

    def _execute_prepared_command(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute an already validated command in an owned process group."""

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        cancel_event = current_cancel_event()
        if cancel_event is not None and cancel_event.is_set():
            return ExecuteResponse(
                output="Command cancelled before execution.",
                exit_code=130,
                truncated=False,
            )

        process: subprocess.Popen[str] | None = None
        termination_reason: str | None = None
        output_abandoned = False
        try:
            process_options: dict[str, object] = {}
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True

            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                env=self._env,
                cwd=str(self.cwd),
                **process_options,
            )
            _register_shell_process(cancel_event, process)
            deadline = time.monotonic() + effective_timeout
            drain_deadline: float | None = None

            while True:
                now = time.monotonic()
                if (
                    termination_reason is None
                    and cancel_event is not None
                    and cancel_event.is_set()
                ):
                    termination_reason = "cancelled"
                    _terminate_process_tree(process)
                    drain_deadline = now + _PROCESS_DRAIN_GRACE_SECONDS
                elif termination_reason is None and now >= deadline:
                    termination_reason = "timed_out"
                    _terminate_process_tree(process)
                    drain_deadline = now + _PROCESS_DRAIN_GRACE_SECONDS

                if drain_deadline is not None and now >= drain_deadline:
                    _stop_collecting_process_output(process)
                    stdout = stderr = ""
                    output_abandoned = True
                    break

                communicate_deadline = (
                    drain_deadline if drain_deadline is not None else deadline
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=max(
                            0.01,
                            min(0.1, communicate_deadline - time.monotonic()),
                        )
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue

            if termination_reason == "timed_out":
                if timeout is not None:
                    timeout_output = (
                        "Error: Command timed out after "
                        f"{effective_timeout} seconds (custom timeout). The command "
                        "may be stuck or require more time."
                    )
                else:
                    timeout_output = (
                        f"Error: Command timed out after {effective_timeout} seconds. "
                        "For long-running commands, re-run using the timeout parameter."
                    )
                response = ExecuteResponse(
                    output=timeout_output,
                    exit_code=124,
                    truncated=output_abandoned,
                )
            elif termination_reason == "cancelled" or (
                cancel_event is not None and cancel_event.is_set()
            ):
                response = ExecuteResponse(
                    output="Command cancelled.",
                    exit_code=130,
                    truncated=output_abandoned,
                )
            else:
                output_parts = []
                if stdout:
                    output_parts.append(stdout)
                if stderr:
                    stderr_lines = stderr.strip().split("\n")
                    output_parts.extend(f"[stderr] {line}" for line in stderr_lines)
                output = "\n".join(output_parts) if output_parts else "<no output>"

                truncated = False
                if len(output) > self._max_output_bytes:
                    output = output[: self._max_output_bytes]
                    output += (
                        f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                    )
                    truncated = True
                if process.returncode != 0:
                    output = f"{output.rstrip()}\n\nExit code: {process.returncode}"
                response = ExecuteResponse(
                    output=output,
                    exit_code=process.returncode,
                    truncated=truncated,
                )
        except Exception as exc:
            if process is not None:
                _terminate_process_tree(process)
            response = ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )
        finally:
            if process is not None:
                _unregister_shell_process(cancel_event, process)

        # Enhance timeout errors with actionable recovery guidance
        if response.exit_code == 124:
            cmd_words = command.split()
            grep_hint = cmd_words[0] if cmd_words else "process"
            # In dangerous mode `/` is the host root; use a workspace-relative
            # log path so the suggested command doesn't fail or write to `/`.
            output_log = "./output.log" if self._dangerous else "/output.log"
            bg_cmd = f'{command} > {output_log} 2>&1 & echo "PID: $!"'
            response = ExecuteResponse(
                output=(
                    f"{response.output}\n\n"
                    f"Recovery — pick one:\n"
                    f"  1. Needs more time? Re-run with a larger timeout (up to 3600s): "
                    f"execute(command=..., timeout=600)\n"
                    f"  2. Runs indefinitely? Run it in the background and keep the PID:\n"
                    f"       {bg_cmd}\n"
                    f"     Check: ps -p <PID>  (or: ps aux | grep {grep_hint})  ·  "
                    f"Read: cat {output_log}  ·  Stop: kill <PID>"
                ),
                exit_code=response.exit_code,
                truncated=response.truncated,
            )

        return response


class AutoskillProposalSandboxBackend(CustomSandboxBackend):
    """Shell backend rooted at the autoskill proposal directory.

    File-tool writes through this backend are blocked; proposal writes go
    through the `/autoskill-proposals/` route. Shell commands run with cwd set
    to the proposal root so validation commands can inspect generated skill
    folders without executing in the user's project workspace.
    """

    _RAW_WRITE_ERROR = (
        "Raw workspace writes are blocked for AutoSkills. Write proposal files "
        "under /autoskill-proposals/<skill-name>/."
    )

    @staticmethod
    def _rewrite_autoskill_mount(command: str) -> str:
        return re.sub(
            r"(^|[\s'\"(=<>])/autoskill-proposals(?=/|$|[\s'\";|&)])",
            r"\1.",
            command,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=self._RAW_WRITE_ERROR)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(error=self._RAW_WRITE_ERROR)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=file_path, error=self._RAW_WRITE_ERROR)
            for file_path, _ in files
        ]

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(
            error="Deletes are blocked for AutoSkills. Manage proposal files "
            "under /autoskill-proposals/ instead."
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return super().execute(
            self._rewrite_autoskill_mount(command),
            timeout=timeout,
        )
