"""Windows host paths must survive the file tools' virtual-path validation."""

from pathlib import PureWindowsPath
from types import SimpleNamespace

import pytest
from deepagents.backends.utils import validate_path

from EvoScientist import backends, paths
from EvoScientist.stream.events import build_agent_stream_input


@pytest.mark.parametrize(
    "path", ["C:/project/media/paper.pdf", r"C:\project\media\paper.pdf"]
)
async def test_attachment_reference_passes_file_tool_validation(path):
    result = await build_agent_stream_input("Read the paper", media=[path])
    reference = result["messages"][0]["content"][-1]["text"]
    file_path = reference.split("path: ", 1)[1].split(";", 1)[0]

    assert validate_path(file_path) == "/C:/project/media/paper.pdf"
    assert reference == (
        "[attached file: paper.pdf] file tools path: /C:/project/media/paper.pdf; "
        "shell path: C:/project/media/paper.pdf"
    )


async def test_posix_attachment_reference_unchanged():
    result = await build_agent_stream_input(
        "Read the paper", media=["/project/media/paper.pdf"]
    )
    assert result["messages"][0]["content"][-1]["text"] == (
        "[attached file: paper.pdf] path: /project/media/paper.pdf"
    )


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


def test_windows_alias_conversion_does_not_change_posix_paths():
    assert paths.file_tool_path("/project/paper.pdf") == "/project/paper.pdf"
    assert paths.file_tool_path("C:relative.pdf") == "C:relative.pdf"
    assert paths.file_tool_path("/C:/project/paper.pdf") == "/C:/project/paper.pdf"


def test_drive_alias_not_decoded_on_posix(monkeypatch):
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="posix"))
    assert paths.host_file_tool_path("/C:/project/paper.pdf") == "/C:/project/paper.pdf"
