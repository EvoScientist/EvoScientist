"""A serve run with its dependencies faked, and a fake server client for
the proactive tests."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

from langgraph_sdk.errors import NotFoundError

from EvoScientist.cli import commands
from EvoScientist.config import MemoryObservationWriter
from EvoScientist.runtime import AsyncRuntime


def serve_config(
    *,
    default_workdir: str = "",
    channel_send_thinking: bool = True,
    log_level: str = "warning",
    channel_debug_tracing: bool = False,
    auto_approve: bool = False,
    auto_mode: bool = False,
    enable_ask_user: bool = True,
    dangerous_mode: bool = False,
):
    return SimpleNamespace(
        channel_enabled="telegram",
        default_workdir=default_workdir,
        channel_send_thinking=channel_send_thinking,
        log_level=log_level,
        channel_debug_tracing=channel_debug_tracing,
        auto_approve=auto_approve,
        auto_mode=auto_mode,
        enable_ask_user=enable_ask_user,
        dangerous_mode=dangerous_mode,
        enable_async_subagents=False,
        enable_scheduler=False,
        memory_profile_enabled=True,
        memory_observations_enabled=True,
        memory_observation_writer=MemoryObservationWriter.ALL,
        memory_workers_enabled=False,
        memory_skill_synthesis_enabled=False,
        model="test-model",
        provider="anthropic",
        anthropic_auth_mode="api_key",
        openai_auth_mode="api_key",
    )


def run_serve_once(
    monkeypatch,
    config,
    *,
    workdir: str | None = None,
    host: str | None = None,
    no_thinking: bool = False,
    debug: bool = False,
    cwd: str | None = None,
    auto_approve: bool = False,
    auto_mode: bool = False,
    ask_user: bool = False,
    dangerous: bool = False,
    message_queue=None,
    process_message=None,
):
    import EvoScientist.config as config_mod

    order: list[tuple[str, str | None]] = []
    captured: dict[str, object] = {}

    def _fake_ensure_dirs():
        order.append(("ensure_dirs", None))

    def _fake_load_agent(
        work_dir=None,
        checkpointer=None,
        config=None,
        *,
        workspace,
        runtime=None,
    ):
        captured["work_dir"] = work_dir
        captured["workspace"] = workspace
        captured["async_runtime"] = runtime
        return object()

    def _fake_start_channels_bus_mode(
        cfg, agent, thread_id, *, media_dir, send_thinking=None
    ):
        captured["started"] = True
        captured["media_dir"] = media_dir
        captured["send_thinking"] = send_thinking
        captured["thread_id"] = thread_id

    def _fake_channels_stop(channel_type=None, *, runtime=None):
        captured["stopped"] = True

    class _InterruptQueue:
        """A fake queue whose get() immediately raises KeyboardInterrupt."""

        def get(self, timeout=None):
            raise KeyboardInterrupt()

    def _fake_ensure_async_server(cfg, *, dirs, backend=None):
        captured["dirs"] = dirs
        captured["ensure_config"] = cfg
        captured["ensure_backend"] = backend

    monkeypatch.setattr(commands, "ensure_dirs", _fake_ensure_dirs)
    monkeypatch.setattr(
        commands, "_ensure_async_subagent_server", _fake_ensure_async_server
    )
    monkeypatch.setattr(commands, "_load_agent", _fake_load_agent)
    monkeypatch.setattr(
        commands, "_start_channels_bus_mode", _fake_start_channels_bus_mode
    )
    monkeypatch.setattr(commands, "_channels_stop", _fake_channels_stop)
    monkeypatch.setattr(
        commands,
        "_message_queue",
        message_queue if message_queue is not None else _InterruptQueue(),
    )
    if process_message is not None:
        monkeypatch.setattr(commands, "_serve_process_message", process_message)

    def _fake_get_effective_config(cli_overrides=None):
        captured["cli_overrides"] = dict(cli_overrides or {})
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)

    if cwd is not None:
        monkeypatch.setattr(commands.os, "getcwd", lambda: cwd)

    with AsyncRuntime(thread_name="test-serve-runtime") as runtime:
        monkeypatch.setattr(commands, "_get_cli_async_runtime", lambda _ctx: runtime)
        commands.serve(
            object(),
            no_thinking=no_thinking,
            workdir=workdir,
            host=host,
            debug=debug,
            auto_approve=auto_approve,
            auto_mode=auto_mode,
            ask_user=ask_user,
            dangerous=dangerous,
        )
    return order, captured


MAIN_GRAPH = {
    "nodes": [],
    "edges": [
        {"source": "__start__", "target": "model"},
        {"source": "model", "target": "HumanInTheLoopMiddleware.after_model"},
        {
            "source": "HumanInTheLoopMiddleware.after_model",
            "target": "Todo.after_model",
        },
        {"source": "Todo.after_model", "target": "tools", "conditional": True},
        {
            "source": "Todo.after_model",
            "target": "memory.after_agent",
            "conditional": True,
        },
        {"source": "tools", "target": "model"},
        {"source": "memory.after_agent", "target": "Interp.after_agent"},
        {"source": "Interp.after_agent", "target": "__end__"},
    ],
}


class FakeCrons:
    def __init__(self):
        self.rows = []
        self.created = []
        self.updated = []
        self._ids = itertools.count(1)

    def search(self, *, metadata, limit, offset):
        rows = [
            r
            for r in self.rows
            if all(r["metadata"].get(k) == v for k, v in metadata.items())
        ]
        return rows[offset : offset + limit]

    def delete(self, cron_id):
        self.rows = [r for r in self.rows if r["cron_id"] != cron_id]

    def create_for_thread(self, thread_id, assistant_id, **kwargs):
        row = {
            "cron_id": f"c{next(self._ids)}",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            **kwargs,
        }
        self.rows.append(row)
        self.created.append(row)
        return row

    def update(self, cron_id, **kwargs):
        if not any(r["cron_id"] == cron_id for r in self.rows):
            raise NotFoundError("no cron", response=MagicMock(), body=None)
        self.updated.append((cron_id, kwargs))


class FakeThreads:
    def __init__(self):
        self.threads = {}
        self.states = {}
        self.created = []
        self.writes = []
        self.deleted = []

    def get(self, thread_id):
        if thread_id not in self.threads:
            raise NotFoundError("gone", response=MagicMock(), body=None)
        return self.threads[thread_id]

    def get_state(self, thread_id):
        return self.states.get(thread_id, {"values": {}})

    def search(self, *, metadata, limit, sort_by=None, sort_order=None):
        return [
            {"thread_id": tid}
            for tid, t in self.threads.items()
            if all((t.get("metadata") or {}).get(k) == v for k, v in metadata.items())
        ][:limit]

    def delete(self, thread_id):
        self.threads.pop(thread_id, None)
        self.deleted.append(thread_id)

    def create(self, *, metadata):
        tid = f"cron-thread-{len(self.created) + 1}"
        self.threads[tid] = {"thread_id": tid, "metadata": metadata, "status": "idle"}
        self.created.append(tid)
        return {"thread_id": tid}

    def update(self, thread_id, *, metadata):
        self.threads[thread_id].setdefault("metadata", {}).update(metadata)

    def update_state(self, thread_id, values, *, as_node):
        self.writes.append((thread_id, values, as_node))
        return {"checkpoint": {"checkpoint_id": f"after-{len(self.writes)}"}}


class FakeRuns:
    def __init__(self):
        self.running = {}
        self.cancelled = []

    def list(self, thread_id, *, status):
        return self.running.get(thread_id, []) if status == "running" else []

    def cancel(self, thread_id, run_id, *, action):
        self.cancelled.append((thread_id, run_id, action))


class FakeAssistants:
    def __init__(self, graph):
        self.graph = graph

    def get_graph(self, assistant_id):
        assert assistant_id == "EvoScientist"
        return self.graph


class FakeProactiveClient:
    def __init__(self, graph=MAIN_GRAPH):
        self.crons = FakeCrons()
        self.threads = FakeThreads()
        self.runs = FakeRuns()
        self.assistants = FakeAssistants(graph)


def seed_source_thread(client, tid="t1", *, head="h1", status="idle", interrupts=()):
    client.threads.threads[tid] = {"thread_id": tid, "status": status, "metadata": {}}
    client.threads.states[tid] = {
        "checkpoint": {"checkpoint_id": head},
        "next": [],
        "interrupts": list(interrupts),
    }
