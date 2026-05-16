from ..channel_manager import register_channel
from .channel import WebUIChannel, WebUIConfig

__all__ = ["WebUIChannel", "WebUIConfig"]


def create_from_config(config) -> WebUIChannel:
    base_path = getattr(config, "webui_base_path", "/webui") or "/webui"
    if not str(base_path).startswith("/"):
        base_path = f"/{base_path}"
    return WebUIChannel(
        WebUIConfig(
            bind_host=getattr(config, "webui_bind_host", "127.0.0.1") or "127.0.0.1",
            webhook_port=int(getattr(config, "webui_port", 8010) or 8010),
            api_key=getattr(config, "webui_api_key", "") or "",
            base_path=str(base_path).rstrip("/") or "/webui",
            workspace_mode=getattr(config, "default_mode", "daemon") or "daemon",
            workspace_root=getattr(config, "default_workdir", "") or "",
        )
    )


register_channel("webui", create_from_config)
