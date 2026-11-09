"""Windows host paths must survive the file tools' virtual-path validation."""

import os
from dataclasses import asdict
from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest
from deepagents import FilesystemMiddleware
from deepagents.backends.protocol import GlobResult, GrepResult, LsResult
from deepagents.backends.utils import file_data_to_string, validate_path

from EvoScientist import agent_shell, backends, paths
from EvoScientist.channels.base import Channel, RawIncoming
from EvoScientist.cli.file_mentions import resolve_file_mentions
from EvoScientist.stream.events import build_agent_stream_input


@pytest.mark.parametrize(
    "path", ["C:/project/media/paper.pdf", r"C:\project\media\paper.pdf"]
)
async def test_attachment_reference_passes_file_tool_validation(path):
    result = await build_agent_stream_input("Read the paper", media=[path])
    reference = result["messages"][0]["content"][-1]["text"]
    file_path = reference.split("path: ", 1)[1].split(";", 1)[0]

    assert validate_path(file_path) == "/C:/project/media/paper.pdf"
    assert "shell path: C:/project/media/paper.pdf" in reference


async def test_posix_attachment_reference_unchanged():
    result = await build_agent_stream_input(
        "Read the paper", media=["/project/media/paper.pdf"]
    )
    reference = result["messages"][0]["content"][-1]["text"]
    assert "path: /project/media/paper.pdf" in reference
    assert "file tools path:" not in reference
    assert "shell path:" not in reference


@pytest.mark.parametrize(
    "key", ["/C:/project/media/paper.pdf", "/c:/project/media/paper.pdf"]
)
def test_windows_file_tool_alias_reaches_media_folder(monkeypatch, key):
    # Exercise Windows path semantics without replacing the process-wide os.name.
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "Path", PureWindowsPath)
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._media_dir = PureWindowsPath("C:/project/media")
    backend._media_real = backend._media_dir

    assert backend._media_relative(validate_path(key)) == PureWindowsPath("paper.pdf")
    assert backend._media_relative("/C:/project/media-other/paper.pdf") is None


@pytest.mark.parametrize("dangerous", [False, True])
@pytest.mark.parametrize("drive", ["C", "c"])
def test_windows_file_tool_alias_resolves_working_directory(
    monkeypatch, dangerous, drive
):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "Path", PureWindowsPath)
    monkeypatch.setattr(
        backends.LocalShellBackend, "_resolve_path", lambda _self, key: key
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = dangerous
    backend._media_dir = None
    backend.cwd = PureWindowsPath("C:/project")

    expected = f"{drive}:/project/paper.pdf" if dangerous else "/paper.pdf"
    assert (
        backend._resolve_path(validate_path(f"/{drive}:/project/paper.pdf")) == expected
    )


def _windows_backend(monkeypatch, *, dangerous=False):
    """A sandbox at ``C:/project`` with Windows path semantics; the parent
    resolver returns the key it is handed."""
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "Path", PureWindowsPath)
    monkeypatch.setattr(
        backends.LocalShellBackend, "_resolve_path", lambda _self, key: key
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = dangerous
    backend._media_dir = None
    backend.cwd = PureWindowsPath("C:/project")
    return backend


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("/c/project/artifacts/a.md", "/artifacts/a.md"),
        ("/C/Project/artifacts/a.md", "/artifacts/a.md"),
        ("/c/project", "/"),
        # Outside the working folder: a workspace folder named with one letter.
        ("/c/data.csv", "/c/data.csv"),
        ("/c/Users/me/notes.md", "/c/Users/me/notes.md"),
        ("/x/data.csv", "/x/data.csv"),
    ],
)
def test_git_bash_path_resolves_into_working_folder(monkeypatch, key, expected):
    backend = _windows_backend(monkeypatch)
    assert backend._resolve_path(validate_path(key)) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("/c/Users/me/a.md", "c:/Users/me/a.md"),
        ("/d/data", "d:/data"),
        ("/c", "c:/"),
        ("/tmp/a.md", "/tmp/a.md"),
    ],
)
def test_dangerous_git_bash_path_resolves_to_drive(monkeypatch, key, expected):
    backend = _windows_backend(monkeypatch, dangerous=True)
    assert backend._resolve_path(validate_path(key)) == expected


@pytest.mark.parametrize("key", ["/C:/other/a.md", "/d:/data/a.md", "/C:/"])
def test_drive_path_outside_working_folder_is_refused(monkeypatch, key):
    backend = _windows_backend(monkeypatch)
    with pytest.raises(PermissionError, match="outside the workspace"):
        backend._resolve_path(validate_path(key))


def test_git_bash_path_not_read_off_windows(monkeypatch):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(backends, "Path", PurePosixPath)
    monkeypatch.setattr(
        backends.LocalShellBackend, "_resolve_path", lambda _self, key: key
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = False
    backend._media_dir = None
    backend.cwd = PurePosixPath("/home/u/project")

    assert paths.git_bash_drive_path("/c/home/u/project/a.md") is None
    key = "/c/home/u/project/a.md"
    assert backend._resolve_path(key) == key


_TOOL_ARGS = {
    "ls": {"path": "/C:/other"},
    "read_file": {"file_path": "/C:/other/a.md"},
    "write_file": {"file_path": "/C:/other/a.md", "content": "x"},
    "edit_file": {"file_path": "/C:/other/a.md", "old_string": "x", "new_string": "y"},
    "delete": {"file_path": "/C:/other/a.md"},
    "glob": {"pattern": "*.md", "path": "/C:/other"},
    "grep": {"pattern": "x", "path": "/C:/other"},
}


@pytest.mark.parametrize("tool", sorted(_TOOL_ARGS))
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_file_tools_report_drive_path_outside_workspace(
    tmp_path, monkeypatch, tool, asynchronous
):
    backend = backends.CustomSandboxBackend(str(tmp_path))
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "Path", PureWindowsPath)
    backend.cwd = PureWindowsPath("C:/project")
    tools = {t.name: t for t in FilesystemMiddleware(backend=backend).tools}
    runtime = SimpleNamespace(tool_call_id="call-1", state={})

    if asynchronous:
        message = await tools[tool].coroutine(runtime=runtime, **_TOOL_ARGS[tool])
    else:
        message = tools[tool].func(runtime=runtime, **_TOOL_ARGS[tool])

    assert message.status == "error"
    assert "outside the workspace" in message.content


@pytest.mark.skipif(os.name != "nt", reason="real Windows drive paths")
def test_windows_file_tools_reach_workspace_by_git_bash_path(tmp_path):
    workspace = tmp_path / "EvoScientist"
    (workspace / "artifacts").mkdir(parents=True)
    backend = backends.CustomSandboxBackend(str(workspace))
    tools = {t.name: t for t in FilesystemMiddleware(backend=backend).tools}
    runtime = SimpleNamespace(tool_call_id="call-1", state={})
    root = backend.cwd.as_posix()
    git_bash = f"/{root[0].lower()}{root[2:]}"
    outside = tmp_path / "outside" / "a.md"

    inside = tools["write_file"].func(
        runtime=runtime, file_path=f"{git_bash}/artifacts/a.md", content="hello"
    )
    refused = tools["write_file"].func(
        runtime=runtime, file_path=paths.file_tool_path(outside), content="hello"
    )

    assert inside.status == "success"
    assert (workspace / "artifacts" / "a.md").read_text(encoding="utf-8") == "hello"
    assert not (workspace / root[0].lower()).exists()
    assert refused.status == "error"
    assert "outside the workspace" in refused.content
    assert not outside.exists()


@pytest.mark.skipif(os.name != "nt", reason="real Windows drive paths")
def test_windows_git_bash_path_reaches_media_folder(tmp_path):
    media = tmp_path / "workspace" / "media"
    media.mkdir(parents=True)
    (media / "paper.txt").write_text("hello", encoding="utf-8")
    run = tmp_path / "workspace" / "runs" / "r1"
    run.mkdir(parents=True)
    backend = backends.CustomSandboxBackend(str(run), media_dir=media)
    real = media.resolve().as_posix()

    read = backend.read(f"/{real[0].lower()}{real[2:]}/paper.txt")

    assert read.error is None
    assert file_data_to_string(read.file_data) == "hello"


@pytest.mark.parametrize("key", ["/D:/", "/d:/"])
def test_windows_drive_root_alias_is_decoded(monkeypatch, key):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    assert paths.host_file_tool_path(validate_path(key)) == f"{key[1]}:/"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "python parse.py /C:/proj/media/paper.pdf",
            "python parse.py C:/proj/media/paper.pdf",
        ),
        ("cp '/C:/proj/media/my paper.pdf' .", "cp 'C:/proj/media/my paper.pdf' ."),
        ("python parse.py /data.csv", "python parse.py ./data.csv"),
    ],
)
def test_windows_file_tool_alias_in_commands(monkeypatch, command, expected):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: object())
    assert backends.convert_virtual_paths_in_command(command, "r1") == expected


def test_windows_alias_conversion_does_not_change_posix_paths():
    assert paths.file_tool_path("/project/paper.pdf") == "/project/paper.pdf"
    assert paths.file_tool_path("C:relative.pdf") == "C:relative.pdf"
    assert paths.file_tool_path("/C:/project/paper.pdf") == "/C:/project/paper.pdf"


def test_drive_alias_not_decoded_on_posix(monkeypatch):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="posix"))
    assert paths.host_file_tool_path("/C:/project/paper.pdf") == "/C:/project/paper.pdf"


def _query_result(tool, items):
    if tool == "ls":
        return LsResult(entries=items, error="partial listing")
    if tool == "glob":
        return GlobResult(
            matches=items,
            error="partial search",
            truncated=True,
            truncation_reason="budget",
        )
    return GrepResult(matches=items, error="partial search", truncated=True)


async def _query(backend, tool, asynchronous, path):
    args = (path,) if tool == "ls" else ("*.txt" if tool == "glob" else "needle", path)
    result = getattr(backend, "a" + tool if asynchronous else tool)(*args)
    return await result if asynchronous else result


@pytest.mark.parametrize("tool", ["ls", "glob", "grep"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "native_path",
    [r"C:\project\paper.txt", "C:/project/paper.txt", "c:/project/paper.txt"],
)
async def test_dangerous_windows_query_paths_round_trip(
    monkeypatch, tool, asynchronous, native_path
):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "Path", PureWindowsPath)
    monkeypatch.setattr(
        backends.LocalShellBackend,
        "_resolve_path",
        lambda _self, key: PureWindowsPath(key),
    )
    item = {"path": native_path}
    item.update(
        {"line_number": 2, "line": "needle"}
        if tool == "grep"
        else {"size": 17, "is_dir": False}
    )
    items = [item]
    parent_result = _query_result(tool, items)
    expected = asdict(parent_result)
    field = "entries" if tool == "ls" else "matches"
    expected[field][0]["path"] = "/" + native_path.replace("\\", "/")
    monkeypatch.setattr(
        backends.LocalShellBackend, tool, lambda _self, *args, **kwargs: parent_result
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = True

    result = await _query(backend, tool, asynchronous, "/C:/project")

    assert result is parent_result
    assert asdict(result) == expected
    alias = getattr(result, field)[0]["path"]
    assert backend._resolve_path(validate_path(alias)) == PureWindowsPath(native_path)
    assert items[0]["path"] == native_path


@pytest.mark.parametrize("tool", ["ls", "glob", "grep"])
@pytest.mark.parametrize("items", [None, []])
async def test_dangerous_query_preserves_empty_results(monkeypatch, tool, items):
    result = _query_result(tool, items)
    expected = asdict(result)
    monkeypatch.setattr(
        backends.LocalShellBackend, tool, lambda _self, *args, **kwargs: result
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = True

    assert await _query(backend, tool, False, "/C:/project") is result
    assert asdict(result) == expected


@pytest.mark.parametrize("tool", ["ls", "glob", "grep"])
@pytest.mark.parametrize("dangerous", [False, True])
@pytest.mark.parametrize("path", ["/project/paper.txt", "/C:/project/paper.txt"])
async def test_query_preserves_posix_and_virtual_paths(
    monkeypatch, tool, dangerous, path
):
    result = _query_result(tool, [{"path": path}])
    expected = asdict(result)
    monkeypatch.setattr(
        backends.LocalShellBackend, tool, lambda _self, *args, **kwargs: result
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = dangerous
    backend._media_dir = None

    assert await _query(backend, tool, False, "/project") is result
    assert asdict(result) == expected


@pytest.mark.parametrize("tool", ["ls", "glob", "grep"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dangerous", [False, True])
async def test_query_results_can_be_read(tmp_path, tool, asynchronous, dangerous):
    content = "before\nneedle\nafter\n"
    (tmp_path / "paper.txt").write_text(content, encoding="utf-8")
    backend = backends.CustomSandboxBackend(str(tmp_path), dangerous=dangerous)
    root = paths.file_tool_path(tmp_path) if dangerous else "/"

    result = await _query(backend, tool, asynchronous, root)

    assert result.error is None
    items = result.entries if tool == "ls" else result.matches
    assert len(items) == 1
    file_path = validate_path(items[0]["path"])
    read = await backend.aread(file_path) if asynchronous else backend.read(file_path)
    assert read.error is None
    assert file_data_to_string(read.file_data) == content


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("notes.txt", b"before\nneedle\nafter\n"),
        ("big.txt", b"x" * (300 * 1024)),
        ("fig.png", b"\x89PNG\r\n\x1a\n\x00\x00"),
    ],
    ids=["inline", "too-large", "binary"],
)
@pytest.mark.parametrize("dangerous", [False, True])
def test_file_mention_path_can_be_read(tmp_path, name, data, dangerous):
    (tmp_path / name).write_bytes(data)
    backend = backends.CustomSandboxBackend(str(tmp_path), dangerous=dangerous)

    _, prompt, _ = resolve_file_mentions(f"read @{name}", str(tmp_path))
    file_path = validate_path(prompt.split("Path: `", 1)[1].split("`", 1)[0])
    read = backend.read(file_path)

    assert read.error is None
    if name == "notes.txt":
        assert file_data_to_string(read.file_data) == data.decode()


def test_channel_annotation_names_the_file_tool_path():
    local = r"C:\project\media\paper.pdf"
    raw = RawIncoming(
        sender_id="u",
        chat_id="c",
        text="Read the paper",
        media_files=[local],
        content_annotations=[f"[document: {local}]"],
    )
    message = Channel._raw_to_inbound(SimpleNamespace(name="test"), raw)

    assert message.content == "Read the paper\n[document: /C:/project/media/paper.pdf]"
    assert message.media == [local]
