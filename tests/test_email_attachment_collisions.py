"""Email MIME parts must retain separate bytes and safe local paths."""

from email.message import EmailMessage
from pathlib import Path
from unittest.mock import Mock

import pytest

from EvoScientist.channels.email.channel import EmailChannel, EmailConfig


def make_message(message_id, filenames, disposition):
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    if message_id:
        msg["Message-ID"] = message_id
    msg.set_content("body")
    for index, filename in enumerate(filenames):
        msg.add_attachment(
            f"payload-{index}".encode(),
            maintype="image",
            subtype="png",
            filename=filename,
            disposition=disposition,
        )
    return msg.as_bytes()


def make_channel(tmp_path, monkeypatch):
    channel = EmailChannel(EmailConfig())
    channel.set_media_dir(tmp_path)
    monkeypatch.setattr(channel, "_reconnect_imap", lambda: None)
    channel._imap = Mock()
    channel._imap.search.return_value = ("OK", [b"1"])
    channel._imap.store.return_value = ("OK", [])
    return channel


def fetch(channel, message_id, filenames, disposition="attachment"):
    channel._imap.fetch.return_value = (
        "OK",
        [(b"1 (RFC822)", make_message(message_id, filenames, disposition))],
    )
    results = channel._fetch_unseen()
    assert len(results) == 1
    paths = [Path(item["path"]) for item in results[0]["attachments"]]
    assert all(
        str(path) in item["annotation"]
        for path, item in zip(paths, results[0]["attachments"], strict=False)
    )
    return paths


@pytest.mark.parametrize("disposition", ["attachment", "inline"])
def test_email_same_named_parts_keep_their_bytes(tmp_path, monkeypatch, disposition):
    channel = make_channel(tmp_path, monkeypatch)
    paths = fetch(
        channel, "<same@example.com>", ["image.png", "image.png"], disposition
    )
    assert len(set(paths)) == 2
    assert [path.read_bytes() for path in paths] == [b"payload-0", b"payload-1"]


@pytest.mark.parametrize(
    "message_ids", [("<first@example.com>", "<second@example.com>"), ("", "")]
)
def test_email_reused_sequence_number_does_not_overwrite(
    tmp_path, monkeypatch, message_ids
):
    channel = make_channel(tmp_path, monkeypatch)
    first = fetch(channel, message_ids[0], ["image.png"])[0]
    first.write_bytes(b"earlier-email")
    second = fetch(channel, message_ids[1], ["image.png"])[0]
    assert first != second
    assert first.read_bytes() == b"earlier-email"
    assert second.read_bytes() == b"payload-0"


@pytest.mark.parametrize(
    "filename", ["Invoice 01/2026.png", r"..\image.png", "图" * 100 + ".png"]
)
def test_email_filename_is_safe_and_preserves_suffix(tmp_path, monkeypatch, filename):
    channel = make_channel(tmp_path, monkeypatch)
    path = fetch(channel, "<safe@example.com>", [filename])[0]
    assert path.parent == tmp_path
    assert "/" not in path.name
    assert "\\" not in path.name
    assert path.suffix == ".png"
    assert len(path.name.encode()) <= 255
    assert path.read_bytes() == b"payload-0"
