from ..channel_manager import register_channel
from .channel import WebUIChannel, WebUIConfig

__all__ = ["WebUIChannel", "WebUIConfig"]


def create_from_config(config) -> WebUIChannel:
    return WebUIChannel(
        WebUIConfig(
            webhook_port=getattr(config, "webui_port", 8010),
            api_key=getattr(config, "webui_api_key", ""),
            base_path=getattr(config, "webui_base_path", "/webui") or "/webui",
        )
    )


register_channel("webui", create_from_config)
