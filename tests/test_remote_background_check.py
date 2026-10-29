"""The local check_process tool can observe processes owned by the dev server."""

import signal
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from starlette.testclient import TestClient

from EvoScientist import background, local_secret, paths
from EvoScientist.langgraph_dev.http import app
from EvoScientist.langgraph_dev.sdk import langgraph_dev_headers
from EvoScientist.middleware.background import check_process

RUNTIME = SimpleNamespace(
    tool_call_id="check-remote", config={"configurable": {"thread_id": "thread-1"}}
)
RECORD = {
    "process_id": "server-process",
    "name": "training",
    "status": "success",
    "returncode": 0,
    "origin_thread_id": "thread-1",
}


@pytest.fixture
def remote(monkeypatch):
    local_secret.ensure_local_secret()
    monkeypatch.setattr(background, "state_record", lambda _: None)
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_config", lambda: SimpleNamespace()
    )
    needs = Mock(return_value=True)
    monkeypatch.setattr("EvoScientist.langgraph_dev.manager.needs_langgraph_dev", needs)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.sdk.configured_langgraph_dev_url",
        lambda: "http://dev-server:6174",
    )
    response = httpx.Response(
        200,
        json={
            "text": "Process server-process EXITED code 0\nfinished training",
            "record": RECORD,
        },
        request=httpx.Request("GET", "http://dev-server:6174/api/bg_process_check"),
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(httpx, "get", request)
    return needs, request


def test_check_process_mirrors_remote_terminal_record(remote, monkeypatch):
    needs, request = remote
    config = SimpleNamespace()
    monkeypatch.setattr("EvoScientist.EvoScientist._ensure_config", lambda: config)
    result = check_process.func(process_id="server-process", runtime=RUNTIME)

    assert result.update["bg_processes"] == {"server-process": RECORD}
    assert "finished training" in result.update["messages"][0].content
    assert result.update["messages"][0].tool_call_id == "check-remote"
    needs.assert_called_once_with(config, backend="local")
    assert request.call_args.args == ("http://dev-server:6174/api/bg_process_check",)
    assert request.call_args.kwargs["params"] == {
        "process_id": "server-process",
        "thread_id": "thread-1",
    }
    assert request.call_args.kwargs["timeout"] == 10.0
    assert request.call_args.kwargs["trust_env"] is False
    assert request.call_args.kwargs["headers"] == {
        **langgraph_dev_headers(),
        local_secret.HEADER: local_secret.ensure_local_secret(),
    }


def test_known_local_process_does_not_query_server(remote, monkeypatch):
    needs, request = remote
    monkeypatch.setattr(background, "state_record", lambda _: RECORD)
    monkeypatch.setattr(background, "status", lambda *args, **kwargs: "local output")

    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert result.update["messages"][0].content == "local output"
    needs.assert_not_called()
    request.assert_not_called()


def test_disabled_server_keeps_unknown_process_response(remote):
    needs, request = remote
    needs.return_value = False
    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert "No such background process" in result
    request.assert_not_called()


def test_dev_server_does_not_query_itself(remote, monkeypatch):
    needs, request = remote
    monkeypatch.setenv("EVOSCIENTIST_SERVER_PROCESS", "1")
    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert "No such background process" in result
    needs.assert_not_called()
    request.assert_not_called()


def test_offline_server_keeps_unknown_process_response(remote):
    _, request = remote
    request.side_effect = httpx.ConnectError("offline")
    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert "No such background process" in result


def test_remote_timeout_does_not_mark_process_observed(remote):
    _, request = remote
    request.side_effect = httpx.ReadTimeout("timed out")
    with pytest.raises(httpx.ReadTimeout):
        check_process.func(process_id="server-process", runtime=RUNTIME)


def test_remote_check_does_not_request_without_a_readable_secret(remote, monkeypatch):
    """An unusable secret must not cause an unauthenticated process-output request."""
    _, request = remote
    monkeypatch.setattr(local_secret, "read_local_secret", Mock(return_value=None))
    assert "No such background process" in check_process.func(
        process_id="server-process", runtime=RUNTIME
    )
    request.assert_not_called()


def test_remote_check_never_creates_the_server_secret(remote, monkeypatch, tmp_path):
    _, request = remote
    server_home = tmp_path / "server-never-started"
    monkeypatch.setattr(paths, "SERVER_HOME", server_home)
    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert "No such background process" in result
    assert not server_home.exists()
    request.assert_not_called()


@pytest.mark.parametrize("status_code", [403, 404])
def test_remote_version_or_secret_mismatch_keeps_local_response(remote, status_code):
    _, request = remote
    request.return_value = httpx.Response(
        status_code,
        request=httpx.Request("GET", "http://dev-server:6174/api/bg_process_check"),
    )
    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert "No such background process" in result


def test_remote_server_error_still_propagates(remote):
    _, request = remote
    request.return_value = httpx.Response(
        503, request=httpx.Request("GET", "http://dev-server:6174/api/bg_process_check")
    )
    with pytest.raises(httpx.HTTPStatusError):
        check_process.func(process_id="server-process", runtime=RUNTIME)


def test_remote_running_process_explains_where_to_stop_it(remote):
    _, request = remote
    record = {**RECORD, "status": "running", "returncode": None}
    request.return_value = httpx.Response(
        200,
        json={"text": "Process server-process RUNNING", "record": record},
        request=httpx.Request("GET", "http://dev-server:6174/api/bg_process_check"),
    )
    result = check_process.func(process_id="server-process", runtime=RUNTIME)
    assert result.update["bg_processes"] == {"server-process": record}
    assert "owned by the dev server" in result.update["messages"][0].content


def test_remote_unknown_process_does_not_write_state(remote):
    _, request = remote
    request.return_value = httpx.Response(
        200,
        json={"text": "No such background process", "record": None},
        request=httpx.Request("GET", "http://dev-server:6174/api/bg_process_check"),
    )
    assert (
        check_process.func(process_id="server-process", runtime=RUNTIME)
        == "No such background process"
    )


def test_check_route_returns_status_tail_and_state_record(monkeypatch):
    status = Mock(
        return_value="Process server-process EXITED code 0\nfinished training"
    )
    monkeypatch.setattr(background, "status", status)
    monkeypatch.setattr(background, "state_record", lambda _: RECORD)
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/bg_process_check",
            params={"process_id": "server-process", "thread_id": "thread-1"},
            headers={local_secret.HEADER: local_secret.ensure_local_secret()},
        )
    assert response.status_code == 200
    assert response.json() == {"text": status.return_value, "record": RECORD}
    status.assert_called_with("server-process", thread_id="thread-1")


def test_check_route_returns_null_record_for_missing_process():
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/bg_process_check",
            params={"process_id": "missing-process"},
            headers={local_secret.HEADER: local_secret.ensure_local_secret()},
        )
    assert response.status_code == 200
    assert response.json()["record"] is None
    assert "No such background process" in response.json()["text"]


@pytest.fixture
def held_process(tmp_path, monkeypatch):
    """Control the poll boundary without a watcher, child interpreter or sleep."""
    log_path = tmp_path / "process.log"

    class HeldPopen:
        returncode = None

        def poll(self):
            return self.returncode

        def exit(self, returncode):
            self.returncode = returncode
            log_path.write_text("process-finished\n")

    process = background.BgProcess(
        process_id="controlled-process",
        name="controlled",
        command="controlled process",
        popen=HeldPopen(),
        pid=12345,
        log_path=log_path,
        started_at="2026-10-25T00:00:00Z",
        started_ts=time.time(),
        origin_thread_id="thread-1",
    )
    monkeypatch.setattr(background, "_PROCESSES", {process.process_id: process})
    return process


@pytest.mark.parametrize(
    ("returncode", "expected_status"),
    [
        (0, "success"),
        (7, "error"),
        (-signal.SIGTERM, "interrupted"),
    ],
)
def test_check_route_reports_exit_during_log_read(
    held_process, monkeypatch, returncode, expected_status
):
    """A terminal mirror must report the exit and mark it observed by this thread."""
    process = held_process
    read_tail = background._read_tail

    def exit_during_read(log_path, tail_bytes):
        if process.popen.poll() is None:
            process.popen.exit(returncode)
        return read_tail(log_path, tail_bytes)

    monkeypatch.setattr(background, "_read_tail", exit_during_read)
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/bg_process_check",
            params={"process_id": process.process_id, "thread_id": "thread-1"},
            headers={local_secret.HEADER: local_secret.ensure_local_secret()},
        )
    assert response.status_code == 200
    result = response.json()
    assert result["record"]["status"] == expected_status
    assert result["record"]["returncode"] == returncode
    assert f"EXITED code {returncode}" in result["text"]
    assert "RUNNING" not in result["text"]
    if returncode >= 0:
        assert "process-finished" in result["text"]
    assert background.was_observed_done(process.process_id, "thread-1")
    assert not background.was_observed_done(process.process_id, "another-thread")


@pytest.mark.parametrize("returncode", [0, 7, -signal.SIGTERM])
def test_local_check_reports_exit_during_log_read(
    held_process, monkeypatch, returncode
):
    process = held_process
    read_tail = background._read_tail

    def exit_during_read(log_path, tail_bytes):
        process.popen.exit(returncode)
        return read_tail(log_path, tail_bytes)

    monkeypatch.setattr(background, "_read_tail", exit_during_read)
    result = check_process.func(process_id=process.process_id, runtime=RUNTIME)
    record = result.update["bg_processes"][process.process_id]
    assert record["returncode"] == returncode
    assert record["status"] != "running"
    assert f"EXITED code {returncode}" in result.update["messages"][0].content
    assert background.was_observed_done(process.process_id, "thread-1")


def test_local_check_keeps_a_later_exit_notifiable(held_process):
    process = held_process
    result = check_process.func(process_id=process.process_id, runtime=RUNTIME)
    assert result.update["bg_processes"][process.process_id]["status"] == "running"
    process.popen.exit(0)
    assert background.poll_status(process.process_id) == "success"
    assert not background.was_observed_done(process.process_id, "thread-1")


def test_check_route_keeps_a_later_exit_notifiable(held_process):
    """Returning a running snapshot must not acknowledge an exit that happens later."""
    process = held_process
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/bg_process_check",
            params={"process_id": process.process_id, "thread_id": "thread-1"},
            headers={local_secret.HEADER: local_secret.ensure_local_secret()},
        )
    assert response.status_code == 200
    result = response.json()
    assert result["record"]["status"] == "running"
    assert result["record"]["returncode"] is None
    assert "RUNNING" in result["text"]
    process.popen.exit(0)
    assert background.poll_status(process.process_id) == "success"
    assert not background.was_observed_done(process.process_id, "thread-1")


@pytest.mark.parametrize(
    "headers", [{}, {local_secret.HEADER: ""}, {local_secret.HEADER: "wrong"}]
)
def test_check_route_refuses_before_reading_process_output(monkeypatch, headers):
    """Neither process output nor observation state is accessed without the secret."""
    status = Mock(return_value="private process output")
    state_record = Mock(return_value=RECORD)
    monkeypatch.setattr(background, "status", status)
    monkeypatch.setattr(background, "state_record", state_record)
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/bg_process_check",
            params={"process_id": "server-process", "thread_id": "thread-1"},
            headers=headers,
        )
    assert response.status_code == 403
    assert response.json() == {"error": "local_secret_required"}
    status.assert_not_called()
    state_record.assert_not_called()


def test_check_route_refuses_when_server_secret_is_unavailable(monkeypatch):
    """A server that cannot load its secret fails closed before accessing a process."""
    status = Mock(return_value="private process output")
    state_record = Mock(return_value=RECORD)
    monkeypatch.setattr(background, "status", status)
    monkeypatch.setattr(background, "state_record", state_record)
    monkeypatch.setattr(
        local_secret, "ensure_local_secret", Mock(side_effect=OSError("unreadable"))
    )
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/bg_process_check",
            params={"process_id": "server-process"},
            headers={local_secret.HEADER: "anything"},
        )
    assert response.status_code == 403
    assert response.json() == {"error": "local_secret_required"}
    status.assert_not_called()
    state_record.assert_not_called()


def test_remote_check_mirrors_a_real_completed_process(tmp_path, monkeypatch):
    from EvoScientist.middleware import background as middleware

    monkeypatch.setattr(background, "_PROCESSES", {})
    process_id = background.launch(
        "echo remote-process-finished", str(tmp_path), origin_thread_id="thread-1"
    )
    process = background._PROCESSES[process_id]
    try:
        process.popen.wait(timeout=5)
        # The CLI has its own empty registry; the route keeps the real server registry.
        monkeypatch.setattr(
            middleware,
            "background",
            SimpleNamespace(
                check=lambda *args, **kwargs: ("No such background process", None),
            ),
        )
        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_config", lambda: SimpleNamespace()
        )
        monkeypatch.setattr(
            "EvoScientist.langgraph_dev.manager.needs_langgraph_dev",
            lambda *args, **kwargs: True,
        )
        monkeypatch.setattr(
            "EvoScientist.langgraph_dev.sdk.configured_langgraph_dev_url",
            lambda: "http://localhost",
        )
        with TestClient(app, base_url="http://localhost") as client:

            def request(url, *, timeout, trust_env, **kwargs):
                assert timeout == 10.0
                assert trust_env is False
                return client.get(url, **kwargs)

            monkeypatch.setattr(httpx, "get", request)
            result = check_process.func(process_id=process_id, runtime=RUNTIME)

        record = result.update["bg_processes"][process_id]
        assert record == background.state_record(process_id)
        assert record["status"] == "success"
        assert record["returncode"] == 0
        assert record["origin_thread_id"] == "thread-1"
        assert "remote-process-finished" in result.update["messages"][0].content
        assert "thread-1" in process.last_checked_by_thread
    finally:
        if process.popen.poll() is None:
            background._kill_process_tree(process.popen, forceful=True)
        process.popen.wait(timeout=5)
