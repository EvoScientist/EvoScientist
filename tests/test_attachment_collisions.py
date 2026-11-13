"""Inbound attachments must retain their own bytes, not just their names."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


async def receive_signal(
    channel, message_index, attachment_indexes, sources, named=True
):
    await channel._handle_rpc(
        {
            "method": "receive",
            "params": {
                "envelope": {
                    "source": "sender",
                    "dataMessage": {
                        "timestamp": 1000 + message_index,
                        "attachments": [
                            {
                                "file": str(sources[i]),
                                **({"filename": "image.jpeg"} if named else {}),
                            }
                            for i in attachment_indexes
                        ],
                    },
                }
            },
        }
    )


async def receive_imessage(channel, message_index, attachment_indexes, sources):
    await channel._handle_message(
        {
            "message": {
                "id": message_index + 1,
                "sender": "sender",
                "chat_id": 1,
                "attachments": [str(sources[i]) for i in attachment_indexes],
            }
        }
    )


async def receive_qq(channel, message_index, attachment_indexes, sources):
    await channel._on_msg(
        SimpleNamespace(
            id=str(message_index),
            content="",
            author=SimpleNamespace(user_openid="sender"),
            attachments=[
                SimpleNamespace(filename="image.jpeg", url=f"https://example.com/{i}")
                for i in attachment_indexes
            ],
        ),
        "c2c",
    )


async def receive_dingtalk(
    channel, message_index, attachment_indexes, sources, content_only=False
):
    keys = ["content"] if content_only else ["imageContent", "fileContent"]
    payload = {"senderStaffId": "sender", "msgtype": "image"}
    for key, i in zip(keys, attachment_indexes, strict=False):
        payload[key] = {
            "fileName": "image.jpeg",
            "downloadUrl": f"https://example.com/{i}",
        }
    await channel._on_ws_message(
        {
            "type": "CALLBACK",
            "headers": {"messageId": str(message_index)},
            "data": payload,
        }
    )


async def receive_signal_unnamed(channel, message_index, attachment_indexes, sources):
    await receive_signal(
        channel, message_index, attachment_indexes, sources, named=False
    )


async def receive_dingtalk_content(channel, message_index, attachment_indexes, sources):
    await receive_dingtalk(
        channel, message_index, attachment_indexes, sources, content_only=True
    )


@pytest.mark.parametrize(
    ("channel_name", "receiver", "same_message"),
    [
        pytest.param(
            channel_name,
            receiver,
            same_message,
            id=f"{channel_name}-{'within' if same_message else 'across'}",
        )
        for channel_name, receiver in (
            ("signal", receive_signal),
            ("signal_unnamed", receive_signal_unnamed),
            ("imessage", receive_imessage),
            ("qq", receive_qq),
            ("dingtalk", receive_dingtalk),
            ("dingtalk_content", receive_dingtalk_content),
        )
        for same_message in (False, True)
        if not (channel_name == "dingtalk_content" and same_message)
    ],
)
async def test_same_named_attachments_keep_their_bytes(
    tmp_path, monkeypatch, channel_name, receiver, same_message
):
    from EvoScientist.channels.dingtalk.channel import DingTalkChannel, DingTalkConfig
    from EvoScientist.channels.imessage.channel_rpc import (
        IMessageChannelRpc,
        IMessageConfig,
    )
    from EvoScientist.channels.qq.channel import QQChannel, QQConfig
    from EvoScientist.channels.signal.channel import SignalChannel, SignalConfig

    channels = {
        "signal": lambda: SignalChannel(SignalConfig()),
        "signal_unnamed": lambda: SignalChannel(SignalConfig()),
        "imessage": lambda: IMessageChannelRpc(
            IMessageConfig(include_attachments=True)
        ),
        "qq": lambda: QQChannel(QQConfig()),
        "dingtalk": lambda: DingTalkChannel(DingTalkConfig()),
        "dingtalk_content": lambda: DingTalkChannel(DingTalkConfig()),
    }
    channel = channels[channel_name]()
    media_dir = tmp_path / "media"
    channel.set_media_dir(media_dir)
    channel._enqueue_raw = AsyncMock()
    channel._ws_send_json = AsyncMock()
    channel._ensure_token = AsyncMock(return_value="test-token")
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=request.url.path.encode())
    )
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    sources = []
    for index in range(2):
        source = tmp_path / str(index) / "image.jpeg"
        source.parent.mkdir()
        source.write_bytes(f"/{index}".encode())
        sources.append(source)

    if same_message:
        await receiver(channel, 0, [0, 1], sources)
    else:
        await receiver(channel, 0, [0], sources)
        await receiver(channel, 1, [1], sources)
    paths = [
        Path(p)
        for call in channel._enqueue_raw.await_args_list
        for p in call.args[0].media_files
    ]
    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert [p.read_bytes() for p in paths] == [b"/0", b"/1"]
    prefix = {
        "signal": "signal_",
        "signal_unnamed": "signal_",
        "imessage": "imsg_",
        "qq": "qq_",
    }.get(channel_name, "dingtalk_")
    assert all(
        p.name.startswith(prefix) and not p.name.startswith(prefix * 2) for p in paths
    )
    assert all(p.parent == media_dir for p in paths)
    if channel_name != "signal_unnamed":
        assert all(p.suffix == ".jpeg" for p in paths)


@pytest.mark.parametrize("filename", ["image.jpeg", r"..\image.jpeg", "../image.jpeg"])
def test_attachment_filename_is_safe_and_keeps_suffix(filename):
    from EvoScientist.channels.base import attachment_filename

    first = attachment_filename(filename, "sender/a", 0)
    assert first != attachment_filename(filename, "sender_a", 0)
    assert first != attachment_filename(filename, "sender/a", 1)
    assert "/" not in first
    assert "\\" not in first
    assert Path(first).suffix == ".jpeg"


def test_attachment_filename_without_message_id_is_unique():
    from EvoScientist.channels.base import attachment_filename

    assert attachment_filename("image.jpeg", "", 0) != attachment_filename(
        "image.jpeg", "", 0
    )


@pytest.mark.parametrize("message_id", ["sender/123", ""])
@pytest.mark.parametrize("filename", ["x" * 250 + ".jpeg", "图" * 80 + ".jpeg"])
def test_long_attachment_filename_fits_filesystem(tmp_path, message_id, filename):
    from EvoScientist.channels.base import attachment_filename

    name = attachment_filename(filename, message_id, 0)
    for prefix in (
        "signal_",
        "imsg_",
        "qq_",
        "dingtalk_",
        "future_channel_with_long_prefix_",
    ):
        path = tmp_path / (prefix + name)
        assert len(path.name.encode()) <= 255
        assert path.suffix == ".jpeg"
        path.write_bytes(b"attachment")
        assert path.read_bytes() == b"attachment"
