"""Native, structured UI helpers for WebUI and future interactive channels."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _normalize_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serializable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    return value


_MAX_FILE_PREVIEW_BYTES = 256 * 1024
_BINARY_PROBE_BYTES = 8192
_LANGUAGE_BY_SUFFIX = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".htm": "html",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".md": "markdown",
    ".mjs": "javascript",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "bash",
    ".sql": "sql",
    ".swift": "swift",
    ".toml": "ini",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".txt": "plaintext",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(" .-_")
    return token or "workspace"


def _normalize_relative_path(path: str) -> str:
    normalized = (path or "").strip().replace("\\", "/")
    normalized = normalized.lstrip("/")
    if not normalized or normalized == ".":
        return ""

    parts = [part for part in normalized.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError("path traversal is not allowed")
    return "/".join(parts)


async def _resolve_thread_workspace(thread_id: str) -> Path:
    from ..sessions import get_thread_metadata

    metadata = await get_thread_metadata(thread_id)
    workspace_dir = (metadata or {}).get("workspace_dir")
    if not isinstance(workspace_dir, str) or not workspace_dir.strip():
        raise LookupError(f"workspace not found for thread {thread_id}")

    workspace_root = Path(workspace_dir).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise FileNotFoundError(f"workspace does not exist: {workspace_root}")
    return workspace_root


def _resolve_workspace_path(
    workspace_root: Path, relative_path: str
) -> tuple[Path, str]:
    normalized_rel = _normalize_relative_path(relative_path)
    base = workspace_root.resolve()
    candidate = base if not normalized_rel else (base / normalized_rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("path is outside workspace") from exc
    return candidate, normalized_rel


def _looks_binary(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:_BINARY_PROBE_BYTES]
    if b"\x00" in sample:
        return True
    non_printable = sum(
        1 for byte in sample if byte < 32 and byte not in (9, 10, 12, 13)
    )
    return (non_printable / len(sample)) > 0.3


def _language_for_name(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    return _LANGUAGE_BY_SUFFIX.get(suffix)


def _serialize_dir_entry(entry: Path, workspace_root: Path) -> dict[str, Any] | None:
    if entry.is_symlink():
        return None

    try:
        relative = entry.relative_to(workspace_root).as_posix()
    except ValueError:
        return None

    is_directory = entry.is_dir()
    stats = entry.stat()
    return {
        "name": entry.name,
        "relativePath": relative,
        "kind": "directory" if is_directory else "file",
        "size": None if is_directory else int(stats.st_size),
        "modifiedAt": datetime.fromtimestamp(stats.st_mtime, UTC).isoformat(),
    }


def _write_workspace_zip(workspace_root: Path, archive_path: Path) -> int:
    count = 0
    root = workspace_root.resolve()
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for current_root, dirs, files in os.walk(root, topdown=True, followlinks=False):
            current = Path(current_root)
            try:
                current.relative_to(root)
            except ValueError:
                dirs[:] = []
                continue

            dirs[:] = sorted(dirs, key=str.lower)
            for filename in sorted(files, key=str.lower):
                candidate = current / filename
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                try:
                    relative = candidate.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                zf.write(candidate, arcname=relative)
                count += 1
    return count


def cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        return


async def get_skills_overview(
    *,
    tag: str = "",
    search: str = "",
) -> dict[str, Any]:
    from ..tools.skills_manager import fetch_remote_skill_index, list_skills

    normalized_tag = tag.strip().lower()
    normalized_search = search.strip().lower()

    installed = list_skills(include_system=True)
    installed_names = {skill.name for skill in installed}
    installed_payload = [
        {
            "name": skill.name,
            "description": skill.description,
            "source": skill.source,
            "tags": list(skill.tags),
            "path": str(skill.path),
        }
        for skill in installed
    ]

    marketplace_error: str | None = None
    marketplace_raw: list[dict[str, Any]] = []
    try:
        marketplace_raw = await asyncio.to_thread(fetch_remote_skill_index)
    except Exception as exc:
        marketplace_error = str(exc)

    marketplace_payload: list[dict[str, Any]] = []
    for item in marketplace_raw:
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        description = str(item.get("description", "") or "")
        tags = [str(tag) for tag in item.get("tags", []) if str(tag).strip()]
        haystack = " ".join([name, description, " ".join(tags)]).lower()

        if normalized_tag and normalized_tag not in [tag.lower() for tag in tags]:
            continue
        if normalized_search and normalized_search not in haystack:
            continue

        marketplace_payload.append(
            {
                "name": name,
                "description": description,
                "tags": tags,
                "installSource": str(item.get("install_source", "") or ""),
                "installed": name in installed_names,
            }
        )

    marketplace_payload.sort(
        key=lambda item: (
            item.get("installed", False),
            str(item.get("name", "")).lower(),
        )
    )
    installed_payload.sort(key=lambda item: str(item.get("name", "")).lower())

    return {
        "installed": installed_payload,
        "marketplace": marketplace_payload,
        "marketplaceError": marketplace_error,
    }


async def install_skills(
    *,
    sources: list[str],
    local: bool = False,
) -> dict[str, Any]:
    from ..tools.skills_manager import install_skill

    normalized_sources = _dedupe_preserve_order(
        [source.strip() for source in sources if source.strip()]
    )
    if not normalized_sources:
        return {
            "ok": False,
            "error": "sources are required",
            "installed": [],
            "failed": [],
        }

    installed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for source in normalized_sources:
        result = await asyncio.to_thread(
            install_skill,
            source,
            None,
            not local,
        )
        if result.get("batch"):
            for item in result.get("installed", []):
                installed.append(
                    {
                        "source": source,
                        "name": str(item.get("name", "") or ""),
                        "description": str(item.get("description", "") or ""),
                    }
                )
            for item in result.get("failed", []):
                failed.append(
                    {
                        "source": source,
                        "name": str(item.get("name", "") or ""),
                        "error": str(item.get("error", "failed") or "failed"),
                    }
                )
            continue

        if result.get("success"):
            installed.append(
                {
                    "source": source,
                    "name": str(result.get("name", "") or ""),
                    "description": str(result.get("description", "") or ""),
                }
            )
            continue

        failed.append(
            {
                "source": source,
                "name": source,
                "error": str(result.get("error", "failed") or "failed"),
            }
        )

    return {
        "ok": bool(installed),
        "installed": installed,
        "failed": failed,
        "reloadRecommended": bool(installed),
    }


async def uninstall_skill(name: str) -> dict[str, Any]:
    from ..tools.skills_manager import uninstall_skill as _uninstall_skill

    normalized_name = name.strip()
    if not normalized_name:
        return {"ok": False, "error": "name is required"}

    result = await asyncio.to_thread(_uninstall_skill, normalized_name)
    return {
        "ok": bool(result.get("success")),
        "name": normalized_name,
        "error": str(result.get("error", "") or ""),
    }


async def get_mcp_overview(
    *,
    tag: str = "",
    search: str = "",
) -> dict[str, Any]:
    from ..mcp import load_mcp_config
    from ..mcp.registry import fetch_marketplace_index, get_installed_names

    normalized_tag = tag.strip().lower()
    normalized_search = search.strip().lower()

    config = load_mcp_config()
    configured_payload = []
    for name, entry in sorted(config.items(), key=lambda item: item[0].lower()):
        configured_payload.append(
            {
                "name": name,
                "transport": str(entry.get("transport", "") or ""),
                "command": entry.get("command"),
                "args": entry.get("args", []),
                "url": entry.get("url"),
                "tools": entry.get("tools"),
                "exposeTo": entry.get("expose_to"),
            }
        )

    installed_names = set(get_installed_names())

    marketplace_error: str | None = None
    marketplace_entries: list[Any] = []
    try:
        marketplace_entries = await asyncio.to_thread(fetch_marketplace_index)
    except Exception as exc:
        marketplace_error = str(exc)

    marketplace_payload: list[dict[str, Any]] = []
    for entry in marketplace_entries:
        name = str(getattr(entry, "name", "") or "").strip()
        if not name:
            continue
        description = str(getattr(entry, "description", "") or "")
        tags = [str(tag) for tag in getattr(entry, "tags", []) if str(tag).strip()]
        haystack = " ".join([name, description, " ".join(tags)]).lower()

        if normalized_tag and normalized_tag not in [tag.lower() for tag in tags]:
            continue
        if normalized_search and normalized_search not in haystack:
            continue

        marketplace_payload.append(
            {
                "name": name,
                "label": str(getattr(entry, "label", "") or name),
                "description": description,
                "tags": tags,
                "transport": str(getattr(entry, "transport", "") or "stdio"),
                "installed": name in installed_names,
                "configured": name in config,
                "envKey": str(getattr(entry, "env_key", "") or ""),
                "envHint": str(getattr(entry, "env_hint", "") or ""),
                "envOptional": bool(getattr(entry, "env_optional", False)),
            }
        )

    marketplace_payload.sort(
        key=lambda item: (
            item.get("configured", False),
            item.get("installed", False),
            str(item.get("name", "")).lower(),
        )
    )

    return {
        "configured": configured_payload,
        "marketplace": marketplace_payload,
        "marketplaceError": marketplace_error,
    }


async def install_mcp_servers(names: list[str]) -> dict[str, Any]:
    from ..mcp.registry import (
        fetch_marketplace_index,
        find_server_by_name,
        install_mcp_server,
    )

    normalized_names = _dedupe_preserve_order(
        [name.strip() for name in names if name.strip()]
    )
    if not normalized_names:
        return {
            "ok": False,
            "error": "names are required",
            "installed": [],
            "failed": [],
            "logs": [],
        }

    marketplace = await asyncio.to_thread(fetch_marketplace_index)

    logs: list[dict[str, str]] = []
    installed: list[str] = []
    failed: list[dict[str, str]] = []

    for name in normalized_names:
        entry = find_server_by_name(name, marketplace)
        if entry is None:
            failed.append({"name": name, "error": "server not found"})
            continue

        def _log(message: str, style: str = "dim") -> None:
            logs.append({"text": str(message), "style": str(style)})

        success = await asyncio.to_thread(
            install_mcp_server,
            entry,
            print_fn=_log,
        )
        if success:
            installed.append(str(getattr(entry, "name", name)))
        else:
            failed.append(
                {
                    "name": str(getattr(entry, "name", name)),
                    "error": "failed to configure server",
                }
            )

    return {
        "ok": bool(installed),
        "installed": installed,
        "failed": failed,
        "logs": logs,
        "reloadRecommended": bool(installed),
    }


async def remove_mcp(name: str) -> dict[str, Any]:
    from ..mcp import remove_mcp_server

    normalized_name = name.strip()
    if not normalized_name:
        return {"ok": False, "error": "name is required"}
    removed = await asyncio.to_thread(remove_mcp_server, normalized_name)
    return {
        "ok": bool(removed),
        "name": normalized_name,
        "error": "" if removed else "server not found",
    }


def get_channels_overview() -> dict[str, Any]:
    from ..channels.channel_manager import available_channels
    from ..cli import channel as cli_channel
    from ..config import load_config

    config = load_config()
    running = cli_channel._channels_running_list()
    details = {}
    if running and getattr(cli_channel, "_manager", None):
        details = cli_channel._manager.get_detailed_status()

    return {
        "available": sorted(available_channels()),
        "configured": _normalize_csv(str(config.channel_enabled or "")),
        "running": running,
        "isRunning": bool(cli_channel._channels_is_running()),
        "details": _serializable(details),
        "hasAgent": getattr(cli_channel, "_cli_agent", None) is not None,
        "threadId": getattr(cli_channel, "_cli_thread_id", None),
    }


def start_channels(
    channel_types: list[str],
    *,
    fallback_thread_id: str = "",
    persist: bool = False,
) -> dict[str, Any]:
    from ..channels.channel_manager import available_channels
    from ..cli import channel as cli_channel
    from ..config import load_config, save_config

    requested = _dedupe_preserve_order(
        [channel_type.strip() for channel_type in channel_types if channel_type.strip()]
    )
    if not requested:
        return {"ok": False, "error": "channelTypes are required"}

    known = set(available_channels())
    unknown = [channel for channel in requested if channel not in known]
    if unknown:
        return {
            "ok": False,
            "error": f"unknown channel type(s): {', '.join(unknown)}",
        }

    config = load_config()
    outcome: dict[str, str] = {}

    if cli_channel._channels_is_running():
        running = set(cli_channel._channels_running_list())
        for channel in requested:
            if channel in running:
                outcome[channel] = "already_running"
                continue
            try:
                cli_channel._add_channel_to_running_bus(
                    channel,
                    config,
                    send_thinking=bool(config.channel_send_thinking),
                )
                outcome[channel] = "started"
            except Exception as exc:
                outcome[channel] = f"error: {exc}"
    else:
        agent = getattr(cli_channel, "_cli_agent", None)
        if agent is None:
            return {
                "ok": False,
                "error": "no active EvoScientist agent in this process",
            }

        thread_id = (
            str(getattr(cli_channel, "_cli_thread_id", "") or "").strip()
            or fallback_thread_id.strip()
            or "webui"
        )
        original_enabled = config.channel_enabled
        try:
            config.channel_enabled = ",".join(requested)
            cli_channel._start_channels_bus_mode(
                config,
                agent,
                thread_id,
                send_thinking=bool(config.channel_send_thinking),
            )
            for channel in requested:
                outcome[channel] = "started"
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            config.channel_enabled = original_enabled

    if persist:
        current = _normalize_csv(str(config.channel_enabled or ""))
        config.channel_enabled = ",".join(_dedupe_preserve_order(current + requested))
        save_config(config)

    return {
        "ok": True,
        "status": outcome,
        "channels": get_channels_overview(),
    }


def stop_channels(
    *,
    channel_type: str = "",
    persist: bool = False,
) -> dict[str, Any]:
    from ..cli import channel as cli_channel
    from ..config import load_config, save_config

    target = channel_type.strip()
    try:
        if target:
            cli_channel._channels_stop(target)
        else:
            cli_channel._channels_stop()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if persist:
        config = load_config()
        if target:
            current = _normalize_csv(str(config.channel_enabled or ""))
            config.channel_enabled = ",".join(
                [channel for channel in current if channel != target]
            )
        else:
            config.channel_enabled = ""
        save_config(config)

    return {
        "ok": True,
        "channels": get_channels_overview(),
    }


async def get_workspace_tree(
    *,
    thread_id: str,
    relative_path: str = "",
) -> dict[str, Any]:
    workspace_root = await _resolve_thread_workspace(thread_id)
    target_dir, normalized_rel = _resolve_workspace_path(workspace_root, relative_path)

    if not target_dir.exists():
        raise FileNotFoundError(f"path does not exist: {normalized_rel or '.'}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"path is not a directory: {normalized_rel or '.'}")

    entries: list[dict[str, Any]] = []
    for entry in target_dir.iterdir():
        serialized = _serialize_dir_entry(entry, workspace_root)
        if serialized is None:
            continue
        entries.append(serialized)

    entries.sort(key=lambda item: (item["kind"] != "directory", item["name"].lower()))

    return {
        "ok": True,
        "threadId": thread_id,
        "workspaceDir": str(workspace_root),
        "path": normalized_rel,
        "entries": entries,
    }


async def read_workspace_file_preview(
    *,
    thread_id: str,
    relative_path: str,
    max_bytes: int = _MAX_FILE_PREVIEW_BYTES,
) -> dict[str, Any]:
    workspace_root = await _resolve_thread_workspace(thread_id)
    target_file, normalized_rel = _resolve_workspace_path(workspace_root, relative_path)

    if not normalized_rel:
        raise ValueError("path is required")
    if not target_file.exists():
        raise FileNotFoundError(f"path does not exist: {normalized_rel}")
    if not target_file.is_file():
        raise ValueError(f"path is not a file: {normalized_rel}")
    if target_file.is_symlink():
        raise ValueError("symlinks are not supported")

    stats = target_file.stat()
    with target_file.open("rb") as handle:
        raw = handle.read(max_bytes + 1)

    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]

    is_text = not _looks_binary(raw)
    content = raw.decode("utf-8", errors="replace") if is_text else ""
    message = None if is_text else "Binary or unsupported file type."

    return {
        "ok": True,
        "threadId": thread_id,
        "workspaceDir": str(workspace_root),
        "path": normalized_rel,
        "name": target_file.name,
        "extension": target_file.suffix.lower(),
        "language": _language_for_name(target_file.name),
        "isText": is_text,
        "content": content,
        "truncated": truncated,
        "size": int(stats.st_size),
        "message": message,
    }


async def create_workspace_archive(*, thread_id: str) -> dict[str, Any]:
    workspace_root = await _resolve_thread_workspace(thread_id)
    fd, tmp_name = tempfile.mkstemp(prefix="evosci-workspace-", suffix=".zip")
    os.close(fd)
    archive_path = Path(tmp_name)

    try:
        file_count = await asyncio.to_thread(
            _write_workspace_zip,
            workspace_root,
            archive_path,
        )
    except Exception:
        cleanup_temp_file(str(archive_path))
        raise

    workspace_token = _safe_filename_token(workspace_root.name)
    thread_token = _safe_filename_token(thread_id)
    download_name = f"{workspace_token}-{thread_token}.zip"
    return {
        "ok": True,
        "threadId": thread_id,
        "workspaceDir": str(workspace_root),
        "archivePath": str(archive_path),
        "downloadName": download_name,
        "fileCount": file_count,
    }


def schedule_shutdown(delay_seconds: float = 0.2) -> None:
    """Gracefully stop the current process shortly after responding."""
    loop = asyncio.get_running_loop()
    loop.call_later(delay_seconds, signal.raise_signal, signal.SIGINT)
