"""Media sent to a chat is the file as it was when the tool result arrived (#779)."""

from __future__ import annotations

from pathlib import Path

from EvoScientist.stream import display


def _result_event(state: display.StreamState, path: str) -> dict:
    state.handle_event(
        {
            "type": "tool_call",
            "id": "tc1",
            "name": "write_file",
            "args": {"file_path": path},
        }
    )
    result = {
        "type": "tool_result",
        "id": "tc1",
        "name": "write_file",
        "content": "ok",
        "success": True,
    }
    state.handle_event(result)
    return result


def _cleanup(path: str) -> None:
    cleanup = getattr(path, "cleanup", None)
    cleanup()


def test_snapshot_is_sent_and_cleanup_removes_it(tmp_path):
    original = tmp_path / "fig.png"
    original.write_bytes(b"original")
    state = display.StreamState()
    result = _result_event(state, str(original))
    sent: list[str] = []

    display.forward_media_result(state, result, set(), sent.append, tmp_path)
    assert len(sent) == 1
    snapshot = Path(sent[0])
    assert snapshot.read_bytes() == b"original"

    # A later write or delete in the same turn cannot change what is delivered.
    original.write_bytes(b"changed")
    assert snapshot.read_bytes() == b"original"
    original.unlink()
    assert snapshot.read_bytes() == b"original"

    _cleanup(sent[0])
    assert not snapshot.exists()


def test_snapshot_is_not_queued_when_the_file_is_already_gone(tmp_path):
    missing = tmp_path / "gone.png"
    state = display.StreamState()
    result = _result_event(state, str(missing))
    sent: list[str] = []

    display.forward_media_result(state, result, set(), sent.append, tmp_path)
    assert sent == []


def test_a_snapshot_keeps_the_file_suffix(tmp_path):
    original = tmp_path / "report.pdf"
    original.write_bytes(b"pdf")
    state = display.StreamState()
    result = _result_event(state, str(original))
    sent: list[str] = []

    display.forward_media_result(state, result, set(), sent.append, tmp_path)
    assert Path(sent[0]).suffix == ".pdf"
    _cleanup(sent[0])
