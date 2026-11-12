"""Inbound attachments must retain their own bytes, not just their names."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.mark.parametrize(
    "channel_name",
    ["signal", "signal_unnamed", "imessage", "qq", "dingtalk", "dingtalk_content"],
)
@pytest.mark.parametrize("same_message", [False, True])
async def test_same_named_attachments_keep_their_bytes(
    tmp_path, monkeypatch, channel_name, same_message
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

    async def receive(message_index, attachment_indexes):
        if channel_name.startswith("signal"):
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
                                        **(
                                            {"filename": "image.jpeg"}
                                            if channel_name == "signal"
                                            else {}
                                        ),
                                    }
                                    for i in attachment_indexes
                                ],
                            },
                        }
                    },
                }
            )
        elif channel_name == "imessage":
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
        elif channel_name == "qq":
            await channel._on_msg(
                SimpleNamespace(
                    id=str(message_index),
                    content="",
                    author=SimpleNamespace(user_openid="sender"),
                    attachments=[
                        SimpleNamespace(
                            filename="image.jpeg", url=f"https://example.com/{i}"
                        )
                        for i in attachment_indexes
                    ],
                ),
                "c2c",
            )
        else:
            keys = (
                ["content"]
                if channel_name == "dingtalk_content"
                else ["imageContent", "fileContent"]
            )
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

    if same_message and channel_name != "dingtalk_content":
        await receive(0, [0, 1])
    else:
        await receive(0, [0])
        await receive(1, [1])
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
