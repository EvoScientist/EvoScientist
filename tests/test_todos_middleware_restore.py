"""deepagents 0.7.0 made TodoListMiddleware opt-in; EvoScientist opts back in everywhere."""


def _names(middleware_list):
    return {m.name for m in middleware_list}


def test_default_middleware_includes_todos():
    from EvoScientist.EvoScientist import _get_default_middleware

    assert "TodoListMiddleware" in _names(_get_default_middleware())


def test_async_subagent_middleware_includes_todos():
    from EvoScientist.EvoScientist import _get_default_middleware

    assert "TodoListMiddleware" in _names(
        _get_default_middleware(for_async_subagent=True)
    )


def test_injected_subagent_middleware_includes_todos(tmp_path):
    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "research-agent"}]
    _inject_subagent_middleware(subs, workspace_dir=str(tmp_path))
    assert "TodoListMiddleware" in _names(subs[0]["middleware"])
