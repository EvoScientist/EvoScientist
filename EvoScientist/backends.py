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
    WriteResult,
)

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


def _split_shell_commands(command: str) -> list[str]:
    """Split a compound shell command into individual base commands.

    Handles &&, ||, ;, and | operators. Returns base command names.
    """
    base_commands: list[str] = []
    # Split by sequential operators first
    for segment in re.split(r"\s*(?:&&|\|\||;)\s*", command):
        # Then split by pipe
        for pipe_seg in segment.split("|"):
            pipe_seg = pipe_seg.strip()
            if not pipe_seg:
                continue
            try:
                tokens = shlex.split(pipe_seg)
            except ValueError:
                tokens = pipe_seg.split()
            if tokens:
                base_commands.append(tokens[0])
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


def _extract_all_paths(command: str) -> list[str]:
    """Extract potential file paths from a command, including inside quoted strings.

    Scans both shell tokens and string literals (single/double quoted) to find
    paths that start with system prefixes like /Users/, /etc/, /tmp/, etc.
    Skips paths in executable position (command name) and pip install targets.
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
        paths.append(m.group(1))
    return paths


def validate_command(command: str) -> str | None:
    """
    Validate a shell command for safety.

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
    escaped_paths = _extract_all_paths(command)
    if escaped_paths:
        path_sample = escaped_paths[0]
        return (
            f"Command blocked: contains absolute system path '{path_sample}'. "
            f"All file operations must use relative paths within the workspace. "
            f"Use relative paths (e.g., './file.py') instead."
        )

    return None


def convert_virtual_paths_in_command(
    command: str,
    workspace_name: str | None = None,
) -> str:
    """
    Convert virtual paths (starting with /) in commands to relative paths.

    Also auto-corrects hallucinated system absolute paths that reference the
    workspace directory (e.g. ``/Users/.../myproject/file.py`` → ``./file.py``).

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

        # Fix hallucinated system absolute paths that reference the workspace.
        # E.g. /Users/user/.../myproject/file.py → ./file.py
        # This mirrors _resolve_path() logic but for shell command strings.
        if workspace_name:
            for prefix in _SYSTEM_PATH_PREFIXES:
                if path.startswith(prefix):
                    marker = f"/{workspace_name}/"
                    idx = path.find(marker)
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


class MergedReadOnlyBackend(BackendProtocol):
    """Read-only backend that merges two directories.

    Reads from *primary* first (user skills in workspace/skills/),
    falls back to *secondary* (system skills in ./skills/).
    User skills override system skills with the same name.

    Both directories share the same virtual path namespace — the agent
    sees all skills under /skills/ regardless of which backend serves them.
    """

    def __init__(self, primary_dir: str, secondary_dir: str):
        self._primary = ReadOnlyFilesystemBackend(
            root_dir=primary_dir, virtual_mode=True
        )
        self._secondary = ReadOnlyFilesystemBackend(
            root_dir=secondary_dir, virtual_mode=True
        )

    # -- read: try primary first, fall back to secondary --

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        try:
            result = self._primary.read(file_path, offset, limit)
            if not result.startswith("Error:"):
                return result
        except (ValueError, FileNotFoundError, OSError):
            pass
        return self._secondary.read(file_path, offset, limit)

    # -- ls_info: merge both, primary wins on name conflicts --

    def ls_info(self, path: str = "/") -> list:
        secondary_items = {item["path"]: item for item in self._secondary.ls_info(path)}
        primary_items = {item["path"]: item for item in self._primary.ls_info(path)}
        secondary_items.update(primary_items)  # primary overrides
        return sorted(secondary_items.values(), key=lambda x: x["path"])

    # -- grep_raw: search both, deduplicate --

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list:
        results = self._secondary.grep_raw(pattern, path, glob)
        try:
            results += self._primary.grep_raw(pattern, path, glob)
        except Exception:
            pass
        return results

    # -- glob_info: merge both --

    def glob_info(self, pattern: str, path: str = "/") -> list:
        secondary = {
            item["path"]: item for item in self._secondary.glob_info(pattern, path)
        }
        try:
            primary = {
                item["path"]: item for item in self._primary.glob_info(pattern, path)
            }
            secondary.update(primary)
        except Exception:
            pass
        return sorted(secondary.values(), key=lambda x: x["path"])

    # -- write / edit: blocked --

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

    # -- download / upload --

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, trying primary then secondary."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            resp = self._primary.download_files([path])[0]
            if resp.error is not None:
                resp = self._secondary.download_files([path])[0]
            responses.append(resp)
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=path, error="permission_denied")
            for path, _ in files
        ]


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
          1. /<ws_name>/file.py            → /file.py
          2. /Users/name/.../<ws_name>/f   → /f  (strip up to ws dir)
          3. /Users/name/file.py           → /file.py (keep basename)
        """
        ws_name = Path(str(self.cwd)).name  # e.g. "workspace", "my-project"

        # Auto-strip /<ws_name>/ prefix to prevent nesting
        ws_prefix = f"/{ws_name}/"
        if key.startswith(ws_prefix):
            key = key[len(ws_prefix) - 1 :]  # "/<ws>/main.py" → "/main.py"
        elif key == f"/{ws_name}":
            key = "/"

        # Auto-correct system absolute paths
        for prefix in _SYSTEM_PATH_PREFIXES:
            if key.startswith(prefix):
                # Try to extract path after "<ws_name>/"
                idx = key.find(ws_prefix)
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
        # Replace literal workspace-root absolute paths with ./
        # Must happen BEFORE validation so workspace paths (e.g. /tmp/...)
        # are sanitized before the system-path check fires.
        ws = str(self.cwd).rstrip("/") + "/"
        if ws in command:
            command = command.replace(ws, "./")

        # Convert virtual paths to relative paths
        if self.virtual_mode:
            command = convert_virtual_paths_in_command(
                command=command,
                workspace_name=Path(str(self.cwd)).name,
            )

        # Validate command safety (after path sanitization)
        error = validate_command(command)
        if error:
            return ExecuteResponse(
                output=error,
                exit_code=1,
                truncated=False,
            )

        # Delegate to parent for subprocess execution
        response = super().execute(command, timeout=timeout)

        # Enhance timeout errors with actionable recovery guidance
        if response.exit_code == 124:
            cmd_words = command.split()
            grep_hint = cmd_words[0] if cmd_words else "process"
            bg_cmd = f"{command} > /output.log 2>&1 &"
            response = ExecuteResponse(
                output=(
                    f"{response.output}\n\n"
                    f"Recovery: re-run in background to avoid the sandbox timeout:\n"
                    f"  {bg_cmd}\n"
                    f"Then check progress: ps aux | grep {grep_hint}\n"
                    f"Read results: cat /output.log"
                ),
                exit_code=response.exit_code,
                truncated=response.truncated,
            )

        return response
