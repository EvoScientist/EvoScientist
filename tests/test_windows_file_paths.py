"""Windows host paths must survive the file tools' virtual-path validation."""

import asyncio
import os
import subprocess
from dataclasses import asdict
from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from deepagents import FilesystemMiddleware
from deepagents.backends.protocol import GlobResult, GrepResult, LsResult
from deepagents.backends.utils import file_data_to_string, validate_path
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt.tool_node import ToolCallRequest

from EvoScientist import agent_shell, backends, paths
from EvoScientist.channels.base import Channel, RawIncoming
from EvoScientist.cli.file_mentions import resolve_file_mentions
from EvoScientist.middleware import windows_paths
from EvoScientist.middleware.windows_paths import WindowsFilePathMiddleware
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
    "key",
    [
        "/C:/project/media/paper.pdf",
        "/c:/project/media/paper.pdf",
        "/c/project/media/paper.pdf",
        "/C/Project/media/paper.pdf",
    ],
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


class _WinPath(PureWindowsPath):
    """Windows path semantics with a fake ``resolve()``: ``Z:`` is a ``subst``
    drive for ``C:/project``, ``Y:`` an unreachable drive."""

    def resolve(self, strict=False):
        drive = self.drive.upper()
        if drive == "Z:":
            return _WinPath("C:/project", *self.parts[1:])
        if drive == "Y:":
            raise OSError("drive not ready")
        return self


def _windows_backend(monkeypatch, *, dangerous=False):
    """A sandbox at ``C:/project`` with Windows path semantics; the parent
    resolver returns the key it is handed."""
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    monkeypatch.setattr(backends, "Path", _WinPath)
    monkeypatch.setattr(
        backends.LocalShellBackend, "_resolve_path", lambda _self, key: key
    )
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = dangerous
    backend._media_dir = None
    backend.cwd = _WinPath("C:/project")
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


@pytest.mark.parametrize(
    "key", ["/C:/other/a.md", "/d:/data/a.md", "/C:/", "/Y:/project/a.md"]
)
def test_drive_path_outside_working_folder_is_refused(monkeypatch, key):
    backend = _windows_backend(monkeypatch)
    with pytest.raises(PermissionError, match="outside the workspace"):
        backend._resolve_path(validate_path(key))


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("/Z:/artifacts/a.md", "/artifacts/a.md"),
        ("/z:/", "/"),
        ("/z/artifacts/a.md", "/artifacts/a.md"),
    ],
)
def test_other_name_of_working_folder_resolves_into_it(monkeypatch, key, expected):
    backend = _windows_backend(monkeypatch)
    assert backend._resolve_path(validate_path(key)) == expected


@pytest.mark.skipif(os.name != "nt", reason="real Windows junction")
def test_windows_junction_to_workspace_reaches_it(tmp_path):
    workspace = tmp_path / "EvoScientist"
    (workspace / "artifacts").mkdir(parents=True)
    junction = tmp_path / "link"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(workspace)],
        check=True,
        capture_output=True,
    )
    backend = backends.CustomSandboxBackend(str(workspace))
    link = junction.as_posix()
    git_bash = f"/{link[0].lower()}{link[2:]}"

    first = backend.write(paths.file_tool_path(junction / "artifacts" / "a.md"), "a")
    second = backend.write(f"{git_bash}/artifacts/b.md", "b")

    assert first.error is None
    assert second.error is None
    assert (workspace / "artifacts" / "a.md").read_text(encoding="utf-8") == "a"
    assert (workspace / "artifacts" / "b.md").read_text(encoding="utf-8") == "b"
    assert not (workspace / link[0].lower()).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX path semantics")
def test_drive_shaped_name_unchanged_off_windows(tmp_path):
    backend = backends.CustomSandboxBackend(str(tmp_path))
    assert backend._resolve_path("C:/notes.md") == backend.cwd / "C:" / "notes.md"


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
    monkeypatch.setattr(backends, "_is_windows", lambda: True)
    monkeypatch.setattr(backends, "Path", _WinPath)
    backend.cwd = _WinPath("C:/project")
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


@pytest.mark.parametrize(
    "key", ["/C:/project/media/paper.pdf", "/c/project/media/paper.pdf"]
)
def test_media_folder_stays_read_only_by_every_spelling(monkeypatch, key):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends, "Path", PureWindowsPath)
    backend = object.__new__(backends.CustomSandboxBackend)
    backend._dangerous = False
    backend._refuse_delete = False
    backend._media_dir = PureWindowsPath("C:/project/media")
    backend._media_real = backend._media_dir

    error = backends.CustomSandboxBackend._MEDIA_READ_ONLY_ERROR
    assert backend.write(key, "x").error == error
    assert backend.edit(key, "a", "b").error == error
    assert backend.delete(key).error == error


@pytest.mark.skipif(os.name != "nt", reason="real Windows drive paths")
def test_windows_git_bash_path_cannot_change_media_folder(tmp_path):
    media = tmp_path / "workspace" / "media"
    media.mkdir(parents=True)
    (media / "paper.txt").write_text("hello", encoding="utf-8")
    run = tmp_path / "workspace" / "runs" / "r1"
    run.mkdir(parents=True)
    backend = backends.CustomSandboxBackend(str(run), media_dir=media)
    real = media.resolve().as_posix()
    folder = f"/{real[0].lower()}{real[2:]}"

    assert backend.write(f"{folder}/paper.txt", "x").error
    assert backend.edit(f"{folder}/paper.txt", "hello", "x").error
    assert backend.delete(folder).error
    assert [e["path"] for e in backend.ls(folder).entries] == [
        paths.file_tool_path(media / "paper.txt")
    ]
    assert (media / "paper.txt").read_text(encoding="utf-8") == "hello"
    assert not (run / real[0].lower()).exists()


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


def _tool_request(name, args):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,
    )


def _rewritten(request, *, asynchronous=False):
    seen = []

    def handler(req):
        seen.append(req)
        return "done"

    async def ahandler(req):
        seen.append(req)
        return "done"

    mw = WindowsFilePathMiddleware()
    if asynchronous:
        assert asyncio.run(mw.awrap_tool_call(request, ahandler)) == "done"
    else:
        assert mw.wrap_tool_call(request, handler) == "done"
    return seen[0]


_PATH_ARG = windows_paths._PATH_ARGS


def test_middleware_covers_every_file_tool_path_argument(tmp_path):
    backend = backends.CustomSandboxBackend(str(tmp_path))
    tools = {t.name: t for t in FilesystemMiddleware(backend=backend).tools}

    assert set(_PATH_ARG) == set(tools) - {"execute"}
    for name, arg in _PATH_ARG.items():
        assert arg in tools[name].args, name


@pytest.mark.parametrize("tool", sorted(_PATH_ARG))
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "drive_path",
    [r"C:\Users\me\ws\a.md", "C:/Users/me/ws/a.md", r"c:\Users/me/ws/a.md"],
)
def test_middleware_rewrites_the_path_argument(
    monkeypatch, tool, asynchronous, drive_path
):
    monkeypatch.setattr(windows_paths, "os", SimpleNamespace(name="nt"))
    arg = _PATH_ARG[tool]
    request = _tool_request(tool, {arg: drive_path, "content": r"C:\keep"})

    seen = _rewritten(request, asynchronous=asynchronous)

    expected = "/" + drive_path.replace("\\", "/")
    assert seen.tool_call["args"] == {arg: expected, "content": r"C:\keep"}
    assert seen.tool_call["id"] == "call-1"
    assert request.tool_call["args"][arg] == drive_path


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("write_file", {"file_path": "/artifacts/a.md"}),
        ("write_file", {"file_path": "/C:/Users/me/ws/a.md"}),
        ("write_file", {"file_path": "/c/Users/me/ws/a.md"}),
        ("grep", {"pattern": "x", "path": None}),
        ("read_file", {"path": r"C:\Users\me\ws\a.md"}),
        ("execute", {"command": r"type C:\Users\me\ws\a.md"}),
    ],
)
def test_middleware_leaves_other_calls_alone(monkeypatch, name, args):
    monkeypatch.setattr(windows_paths, "os", SimpleNamespace(name="nt"))
    request = _tool_request(name, args)
    assert _rewritten(request) is request


def test_middleware_does_nothing_off_windows(monkeypatch):
    monkeypatch.setattr(windows_paths, "os", SimpleNamespace(name="posix"))
    request = _tool_request("write_file", {"file_path": r"C:\Users\me\ws\a.md"})
    assert _rewritten(request) is request


class _ScriptedToolModel(BaseChatModel):
    replies: list

    @property
    def _llm_type(self) -> str:
        return "scripted-tools"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def test_agent_file_tool_takes_a_drive_path(tmp_path):
    """Through a real agent: on Windows the file lands in the workspace; elsewhere
    a drive path stays refused and nothing is written."""
    workspace = tmp_path / "EvoScientist"
    workspace.mkdir()
    backend = backends.CustomSandboxBackend(str(workspace))
    target = (
        str(backend.cwd / "artifacts" / "a.md")
        if os.name == "nt"
        else (r"C:\Users\me\EvoScientist\artifacts\a.md")
    )
    model = _ScriptedToolModel(
        replies=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": target, "content": "hello"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage("done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            FilesystemMiddleware(backend=backend, tools=["read_file", "write_file"]),
            WindowsFilePathMiddleware(),
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("write it")]})

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    if os.name == "nt":
        assert tool_message.status == "success"
        assert (workspace / "artifacts" / "a.md").read_text(encoding="utf-8") == "hello"
    else:
        assert tool_message.status == "error"
        assert "Windows absolute paths are not supported" in tool_message.content
        assert [p.name for p in workspace.iterdir()] == []


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
@pytest.mark.parametrize("for_async_subagent", [False, True])
def test_main_agent_stack_mounts_the_middleware(
    mock_config, mock_model, mock_ts, for_async_subagent, workspace
):
    from EvoScientist.EvoScientist import _get_default_middleware

    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    mw = _get_default_middleware(
        workspace=workspace, for_async_subagent=for_async_subagent
    )
    assert sum(isinstance(m, WindowsFilePathMiddleware) for m in mw) == 1


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_subagent_stack_mounts_the_middleware(mock_model, workspace):
    from EvoScientist.EvoScientist import _inject_subagent_middleware

    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    subs = [{"name": "test-agent"}]

    _inject_subagent_middleware(subs, workspace=workspace)

    assert (
        sum(isinstance(m, WindowsFilePathMiddleware) for m in subs[0]["middleware"])
        == 1
    )


def test_grader_mounts_the_middleware(tmp_path):
    from deepagents.backends import FilesystemBackend

    from EvoScientist.middleware.rubric import create_rubric_middleware
    from tests.fakes.rubric import ScriptedGrader

    rubric = create_rubric_middleware(
        model=ScriptedGrader(verdicts=[]),
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=3,
    )
    assert any(
        isinstance(m, WindowsFilePathMiddleware) for m in rubric._grader_middleware
    )


def test_goal_drafter_mounts_the_middleware(monkeypatch):
    import langchain.agents

    from EvoScientist.middleware.goal import GoalDrafter

    captured = {}
    monkeypatch.setattr(
        langchain.agents,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    GoalDrafter(model=MagicMock(), backend=SimpleNamespace())._ensure_agent()

    assert any(isinstance(m, WindowsFilePathMiddleware) for m in captured["middleware"])
