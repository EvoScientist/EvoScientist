"""Config keys, the per-run key and where the retry middleware is mounted."""

from __future__ import annotations

import logging

import pytest

from EvoScientist.backends import MODEL_RETRY_RUN_KEY, model_retry_for_run
from EvoScientist.config import EvoScientistConfig
from EvoScientist.middleware.model_retry import (
    ModelCallBoundaryMiddleware,
    ModelRetryMiddleware,
)


def test_config_defaults_and_env_names():
    from EvoScientist.config.settings import _ENV_MAPPINGS

    cfg = EvoScientistConfig()
    assert cfg.model_retry_wait_seconds == 60
    assert cfg.auto_mode_retry_wait_seconds == 1800
    assert (
        _ENV_MAPPINGS["model_retry_wait_seconds"]
        == "EVOSCIENTIST_MODEL_RETRY_WAIT_SECONDS"
    )
    assert (
        _ENV_MAPPINGS["auto_mode_retry_wait_seconds"]
        == "EVOSCIENTIST_AUTO_MODE_RETRY_WAIT_SECONDS"
    )


def test_model_retry_for_run_follows_auto_mode():
    cfg = EvoScientistConfig(
        model_retry_wait_seconds=12, auto_mode_retry_wait_seconds=345
    )
    assert model_retry_for_run(cfg) == {"unattended": False, "wait_budget_seconds": 12}
    cfg.auto_mode = True
    assert model_retry_for_run(cfg) == {"unattended": True, "wait_budget_seconds": 345}
    cfg.auto_mode_retry_wait_seconds = -5
    assert model_retry_for_run(cfg)["wait_budget_seconds"] == 0


@pytest.mark.parametrize("bad", ["60s", "30m", "90", 1.5, float("inf"), True, [60]])
def test_a_malformed_budget_goes_back_to_its_default_with_a_warning(bad, caplog):
    """A value that is not a whole number of seconds must not switch the
    retry off without a word; only 0 does that."""
    with caplog.at_level(logging.WARNING, logger="EvoScientist.config.settings"):
        cfg = EvoScientistConfig(
            model_retry_wait_seconds=bad, auto_mode_retry_wait_seconds=bad
        )
    assert (cfg.model_retry_wait_seconds, cfg.auto_mode_retry_wait_seconds) == (
        60,
        1800,
    )
    assert "Invalid model_retry_wait_seconds" in caplog.text
    assert "Invalid auto_mode_retry_wait_seconds" in caplog.text


def test_zero_and_whole_seconds_are_kept_as_written(caplog):
    with caplog.at_level(logging.WARNING, logger="EvoScientist.config.settings"):
        cfg = EvoScientistConfig(
            model_retry_wait_seconds=0, auto_mode_retry_wait_seconds=45
        )
    assert (cfg.model_retry_wait_seconds, cfg.auto_mode_retry_wait_seconds) == (0, 45)
    assert "Invalid" not in caplog.text


def test_a_negative_budget_is_kept_and_switches_the_retry_off(caplog):
    from EvoScientist.EvoScientist import _model_retry_middleware

    with caplog.at_level(logging.WARNING, logger="EvoScientist.config.settings"):
        cfg = EvoScientistConfig(model_retry_wait_seconds=-5)
    assert cfg.model_retry_wait_seconds == -5
    assert "Invalid" not in caplog.text
    assert _model_retry_middleware(cfg)._tier.wait_budget_seconds == 0.0
    assert model_retry_for_run(cfg)["wait_budget_seconds"] == 0


def test_a_malformed_budget_in_the_config_file_keeps_the_default(tmp_path, monkeypatch):
    """From the file, through the loader, to the retry layer and the run's key."""
    from EvoScientist import paths
    from EvoScientist.config.settings import load_config
    from EvoScientist.EvoScientist import _model_retry_middleware

    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "evoscientist")
    (tmp_path / "evoscientist").mkdir()
    (tmp_path / "evoscientist" / "config.yaml").write_text(
        "model_retry_wait_seconds: 60s\nauto_mode_retry_wait_seconds: .inf\n",
        encoding="utf-8",
    )
    cfg = load_config()
    assert (cfg.model_retry_wait_seconds, cfg.auto_mode_retry_wait_seconds) == (
        60,
        1800,
    )
    assert _model_retry_middleware(cfg)._tier.wait_budget_seconds == 60.0
    assert model_retry_for_run(cfg)["wait_budget_seconds"] == 60


@pytest.mark.parametrize("value", ["60s", None, float("inf")])
@pytest.mark.parametrize("auto_mode", [False, True])
def test_the_readers_survive_a_value_the_config_did_not_normalise(value, auto_mode):
    """A config object built some other way: the agent still starts and the
    run's key agrees with the mounted layer, both without retries."""
    from types import SimpleNamespace

    from EvoScientist.EvoScientist import _model_retry_middleware

    cfg = SimpleNamespace(
        auto_mode=auto_mode,
        model_retry_wait_seconds=value,
        auto_mode_retry_wait_seconds=value,
    )
    retry = _model_retry_middleware(cfg)
    assert (retry._tier.wait_budget_seconds, retry._tier.unattended) == (0.0, auto_mode)
    assert model_retry_for_run(cfg)["wait_budget_seconds"] == 0


def test_resolve_per_run_config_writes_the_key_after_extras():
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config(
        "t1",
        {MODEL_RETRY_RUN_KEY: {"unattended": False, "wait_budget_seconds": 1}},
        model_retry={"unattended": True, "wait_budget_seconds": 1800},
    )
    assert run_config["configurable"][MODEL_RETRY_RUN_KEY] == {
        "unattended": True,
        "wait_budget_seconds": 1800,
    }
    assert MODEL_RETRY_RUN_KEY not in resolve_per_run_config("t1", None)["configurable"]


def _mw_names(middleware):
    return [getattr(m, "name", type(m).__name__) for m in middleware]


@pytest.fixture(autouse=True)
def no_configured_model(monkeypatch):
    """The lists are built with the user's configured model, which needs its key."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_chat_model",
        lambda: FakeMessagesListChatModel(responses=[]),
    )


def _retry_of(middleware) -> ModelRetryMiddleware:
    return next(m for m in middleware if isinstance(m, ModelRetryMiddleware))


def test_default_middleware_mounts_retry_first_and_always(workspace):
    from EvoScientist.EvoScientist import _get_default_middleware

    cfg = EvoScientistConfig(model_retry_wait_seconds=0, auto_mode_retry_wait_seconds=0)
    mw = _get_default_middleware(workspace=workspace, cfg=cfg)
    names = _mw_names(mw)
    assert names.index("model_retry") < names.index("error_normalization")
    retry = _retry_of(mw)
    assert retry._tier.wait_budget_seconds == 0
    assert retry._honor_run_key is True


def test_default_middleware_tier_follows_auto_mode_and_the_unattended_flag(workspace):
    from EvoScientist.EvoScientist import _get_default_middleware

    cfg = EvoScientistConfig(
        model_retry_wait_seconds=60, auto_mode_retry_wait_seconds=1800
    )
    interactive = _retry_of(_get_default_middleware(workspace=workspace, cfg=cfg))
    assert (interactive._tier.wait_budget_seconds, interactive._tier.unattended) == (
        60.0,
        False,
    )
    cfg.auto_mode = True
    auto = _retry_of(_get_default_middleware(workspace=workspace, cfg=cfg))
    assert (auto._tier.wait_budget_seconds, auto._tier.unattended) == (1800.0, True)
    cfg.auto_mode = False
    scheduler = _retry_of(
        _get_default_middleware(
            workspace=workspace,
            cfg=cfg,
            unattended=True,
            model_retry_honors_run_key=False,
        )
    )
    assert scheduler._tier.unattended is True
    assert scheduler._honor_run_key is False


def test_subagent_lists_get_the_retry_first(workspace):
    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "research", "middleware": []}]
    _inject_subagent_middleware(
        subs, workspace=workspace, cfg=EvoScientistConfig(), unattended=True
    )
    names = _mw_names(subs[0]["middleware"])
    assert names[0] == "model_retry"
    assert names[1] == "error_normalization"
    assert _retry_of(subs[0]["middleware"])._tier.unattended is True


def test_grader_list_contains_the_retry_when_given(tmp_path):
    from deepagents.backends import FilesystemBackend

    from EvoScientist.middleware.rubric import create_rubric_middleware
    from tests.fakes.rubric import ScriptedGrader

    retry = ModelRetryMiddleware(wait_budget_seconds=60, unattended=False)
    rubric = create_rubric_middleware(
        model=ScriptedGrader(verdicts=[]),
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=3,
        model_retry=retry,
    )
    assert rubric._grader_middleware[0] is retry


def test_grader_boundary_comes_with_its_retry_layer(tmp_path):
    """Without the retry layer nothing rejects an error-finished grader response, so no marks either."""
    from deepagents.backends import FilesystemBackend

    from EvoScientist.middleware.rubric import create_rubric_middleware
    from tests.fakes.rubric import ScriptedGrader

    def _build(**kwargs):
        return create_rubric_middleware(
            model=ScriptedGrader(verdicts=[]),
            backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            max_iterations=3,
            **kwargs,
        )._grader_middleware

    retry = ModelRetryMiddleware(wait_budget_seconds=60, unattended=False)
    with_retry = _build(model_retry=retry)
    assert with_retry[0] is retry
    assert isinstance(with_retry[-1], ModelCallBoundaryMiddleware)
    assert not any(isinstance(m, ModelCallBoundaryMiddleware) for m in _build())


def _index_of(middleware, name: str) -> int:
    return _mw_names(middleware).index(name)


def test_default_middleware_mounts_the_boundary_inside_the_fallback_chain(workspace):
    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware(workspace=workspace, cfg=EvoScientistConfig())
    names = _mw_names(mw)
    assert names.count("model_call_boundary") == 1
    boundary = _index_of(mw, "model_call_boundary")
    assert boundary > _index_of(mw, "model_fallback")
    assert boundary > _index_of(mw, "error_normalization")
    assert boundary > _index_of(mw, "context_overflow_mapper")
    assert boundary > _index_of(mw, "conditional_tool_selector")


def test_subagent_lists_get_the_boundary_inside_the_normalization_layer(workspace):
    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "research", "middleware": []}]
    _inject_subagent_middleware(
        subs,
        workspace=workspace,
        cfg=EvoScientistConfig(),
        model_retry_honors_run_key=False,
    )
    mw = subs[0]["middleware"]
    assert _mw_names(mw).count("model_call_boundary") == 1
    assert _index_of(mw, "model_call_boundary") > _index_of(
        mw, "context_overflow_mapper"
    )
    assert _retry_of(mw)._honor_run_key is False


@pytest.mark.parametrize("auto_mode", [False, True])
def test_memory_agents_mount_retry_then_normalization(tmp_path, monkeypatch, auto_mode):
    """Workers and the linker keep the short budget even under auto_mode, so a
    provider outage cannot park them on the server's run slots; AutoSkills
    runs alone on a schedule and waits as long as the scheduler."""
    from EvoScientist.config import MemoryObservationWriter
    from EvoScientist.memory.agents import autoskills, memory_worker, observation_linker
    from EvoScientist.memory.types import MemorySourceType
    from EvoScientist.paths import Workspace

    monkeypatch.setattr(
        "EvoScientist.EvoScientist._config",
        EvoScientistConfig(
            auto_mode=auto_mode,
            model_retry_wait_seconds=60,
            auto_mode_retry_wait_seconds=1800,
        ),
    )
    stacks = {
        source.value: memory_worker._memory_worker_middleware(
            memory_dir=tmp_path / "memories",
            workspace_dir=tmp_path / "workspace",
            source_type=source,
            observation_writer=MemoryObservationWriter.ALL,
        )
        for source in MemorySourceType
    }
    for module, build, key in (
        (observation_linker, "build_observation_linker_graph", "linker"),
        (autoskills, "build_autoskills_graph", "autoskills"),
    ):

        def capture(*, key=key, **kwargs):
            stacks[key] = kwargs["middleware"]

        monkeypatch.setattr(module, "build_memory_agent_graph", capture)
        getattr(module, build)(
            workspace=Workspace(tmp_path / "workspace"),
            memory_dir=tmp_path / "memories",
        )

    tiers = {}
    for key, middleware in stacks.items():
        assert isinstance(middleware[0], ModelRetryMiddleware), key
        assert type(middleware[1]).__name__ == "ErrorNormalizationMiddleware", key
        tier = middleware[0]._tier
        tiers[key] = (tier.wait_budget_seconds, tier.unattended)
    assert tiers == {
        "turn": (60.0, False),
        "subagent": (60.0, False),
        "linker": (60.0, False),
        "autoskills": (1800.0, True),
    }
