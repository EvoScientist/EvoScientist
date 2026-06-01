"""Custom backends for EvoScientist agent."""

import os
import re
import shlex
import uuid
from pathlib import Path

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import (
    BackendProtocol,
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

# Dangerous patterns that could escape the workspace
BLOCKED_PATTERNS = [
    r"~/",  # home directory
    r"\bcd\s+/",  # cd to absolute path
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
            if index + 1 < n and command[index + 1] == ch:
                return command[index : index + 2]
            return ch
        if ch.isdigit():
            j = index
            while j < n and command[j].isdigit():
                j += 1
            if j < n and command[j] in "<>":
                end = j + 1
                if end < n and command[end] in ("&", command[j]):
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
) -> str | None:
    """
    Validate a shell command for safety.

    Args:
        command: Shell command string.
        allow_prefixes: Absolute path prefixes exempt from the system-path
            block list (matching rules in ``_is_under_allowed_prefix``).

    Returns:
        None if command is safe, error message string if blocked.
    """
    # Check for '..' path traversal as a path component
    if _has_traversal_component(command):
        return (
            "Command blocked: contains '..' path traversal. "
            "All commands must operate within the workspace directory. "
            "Use relative paths (e.g., './file.py') instead."
        )

    # Check for dangerous patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return (
                f"Command blocked: contains forbidden pattern '{pattern}'. "
                f"All commands must operate within the workspace directory. "
                f"Use relative paths (e.g., './file.py') instead."
            )

    # Check for dangerous commands (pipeline-aware)
    for base_cmd in _split_shell_commands(command):
        if base_cmd in BLOCKED_COMMANDS:
            return (
                f"Command blocked: '{base_cmd}' is not allowed in sandbox mode. "
                f"Only standard development commands are permitted."
            )

    # Check for absolute system paths (including inside quoted strings).
    # This catches attacks like: python -c "os.remove('/Users/foo/file')"
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


def _resolve_virtual_mount_path(token: str) -> str | None:
    """Resolve a virtual mount token to a shell-safe token, or ``None`` when
    *token* is not a registered virtual mount.

    For ``/skills/...``: walks ``_skills_tier_paths()`` priority (USER →
    GLOBAL → BUILTIN), returning ``shlex.quote`` of the first tier where the
    path exists. On miss, returns a workspace-relative ``./skills/<rel>``
    form — agent typed a virtual path, so the shell error should reference a
    location they recognise (`USER_SKILLS_DIR` defaults to
    ``WORKSPACE_ROOT / "skills"``, which is also where ``MergedSkillsBackend``
    would write a new skill).

    For ``/memories/...``: single tier (``paths.MEMORIES_DIR``), always
    absolute and ``shlex.quote``-wrapped. Memories live outside the
    workspace, so a relative form would point at an unrelated location.
    """
    rel = _subpath_under_mount(token, "/skills")
    if rel is not None:
        for tier in _skills_tier_paths():
            if tier is None:
                continue
            candidate = Path(tier) / rel
            if candidate.exists():
                return shlex.quote(str(candidate))
        return shlex.quote("./skills/" + rel if rel else "./skills")

    rel = _subpath_under_mount(token, "/memories")
    if rel is not None:
        return shlex.quote(str(Path(paths.MEMORIES_DIR) / rel))

    return None


def convert_virtual_paths_in_command(
    command: str,
    workspace_name: str | None = None,
) -> str:
    """
    Convert virtual paths (starting with /) in commands to relative paths.

    Also auto-corrects hallucinated system absolute paths that reference the
    workspace directory (e.g. ``/Users/.../myproject/file.py`` → ``./file.py``).

    Tier-aware mounts (``/skills/...``, ``/memories/...``) are expanded to
    absolute paths via ``_resolve_virtual_mount_path``. Callers that pass
    the result through ``validate_command`` MUST whitelist the tier roots
    via ``allow_prefixes`` to avoid false-positive system-path blocks.

    Args:
        command: Original command.
        workspace_name: Basename of the workspace directory (e.g. ``"workspace"``,
            ``"my-project"``).  When provided, system paths containing
            ``/<workspace_name>/`` are auto-corrected.

    Examples:
        >>> convert_virtual_paths_in_command("python /main.py")
        'python ./main.py'
        >>> convert_virtual_paths_in_command("ls /")
        'ls .'
        >>> convert_virtual_paths_in_command(
        ...     "mkdir -p /Users/u/proj/dir", workspace_name="proj")
        'mkdir -p ./dir'
    """

    def replace_virtual_path(match: re.Match[str]) -> str:
        path = match.group(0)

        # Skip content that looks like a URL
        if "://" in command[max(0, match.start() - 10) : match.end() + 10]:
            return path

        resolved = _resolve_virtual_mount_path(path)
        if resolved is not None:
            return resolved

        # Fix hallucinated system absolute paths that reference the workspace.
        # E.g. /Users/user/.../myproject/file.py → ./file.py
        # This mirrors _resolve_path() logic but for shell command strings.
        if workspace_name:
            for prefix in _SYSTEM_PATH_PREFIXES:
                if path.startswith(prefix):
                    marker = f"/{workspace_name}/"
                    # rfind, not find: the workspace's parent path may itself
                    # contain "/<workspace_name>/" (e.g. dev tree under
                    # ~/workspace/.../workspace). Last occurrence is the
                    # boundary closest to the file.
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
        else:
            return "." + path

    # Match pattern: paths starting with / (but not URLs)
    pattern = r'(?<=\s)/[^\s;|&<>\'"`]*|^/[^\s;|&<>\'"`]*'
    converted = re.sub(pattern, replace_virtual_path, command)

    return converted


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """
    Read-only filesystem backend.

    Allows read, ls, grep, glob operations but blocks write and edit.
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
    ):
        self._primary = FilesystemBackend(root_dir=primary_dir, virtual_mode=True)
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
    command: str, cwd: str | Path, *, virtual_mode: bool = True
) -> tuple[str, str | None]:
    """Normalize workspace paths in ``command`` and validate it for the sandbox.

    Shared by :meth:`CustomSandboxBackend.execute` and the background-process tools so
    both enforce *identical* workspace-path rewriting (so virtual ``/`` paths resolve to
    the workspace, not the host root) and the same command validation.

    Returns ``(prepared_command, error)``: ``error`` is a message string when the command
    is rejected (the caller must NOT run it), otherwise ``None``.
    """
    ssh_error = _validate_ssh_remote_command_format(command)
    if ssh_error:
        return command, ssh_error

    command, ssh_replacements = _mask_ssh_remote_commands(command)

    cwd_str = str(cwd).rstrip("/")
    # Replace literal workspace-root absolute paths with ./ after SSH masking so
    # remote paths that happen to contain the local cwd are preserved, and before
    # validation so local workspace paths are sanitized before the system-path
    # check fires.
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
    error = validate_command(command, allow_prefixes=allow_prefixes)
    if error:
        return command, error
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
        """
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
        """
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

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """
        Execute shell command in sandbox environment.

        Commands are validated before execution to prevent:
        - Directory traversal (../)
        - Access to paths outside workspace
        - Dangerous system commands

        Then delegates to LocalShellBackend.execute() for actual execution.
        """
        command, error = prepare_sandbox_command(
            command, self.cwd, virtual_mode=self.virtual_mode
        )
        if error:
            return ExecuteResponse(output=error, exit_code=1, truncated=False)

        # Delegate to parent for subprocess execution
        response = super().execute(command, timeout=timeout)

        # Enhance timeout errors with actionable recovery guidance
        if response.exit_code == 124:
            cmd_words = command.split()
            grep_hint = cmd_words[0] if cmd_words else "process"
            bg_cmd = f'{command} > /output.log 2>&1 & echo "PID: $!"'
            response = ExecuteResponse(
                output=(
                    f"{response.output}\n\n"
                    f"Recovery — pick one:\n"
                    f"  1. Needs more time? Re-run with a larger timeout (up to 3600s): "
                    f"execute(command=..., timeout=600)\n"
                    f"  2. Runs indefinitely? Run it in the background and keep the PID:\n"
                    f"       {bg_cmd}\n"
                    f"     Check: ps -p <PID>  (or: ps aux | grep {grep_hint})  ·  "
                    f"Read: cat /output.log  ·  Stop: kill <PID>"
                ),
                exit_code=response.exit_code,
                truncated=response.truncated,
            )

        return response
