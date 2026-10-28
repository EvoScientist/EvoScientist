"""MiniMax selection and real SDK serialization, without network inference."""

import asyncio
import json

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from EvoScientist.llm.context_window import get_context_window
from EvoScientist.llm.models import get_chat_model
from EvoScientist.llm.registry import (
    _MODEL_ENTRIES,
    MODELS,
    get_models_for_provider,
    list_model_picker_entries,
    list_models,
    list_models_by_provider,
)
from tests.fakes import anthropic_httpx


def test_current_choices_preserve_legacy_resolution():
    current = [
        ("minimax-m3.1-flash", "MiniMax-M3.1-Flash"),
        ("minimax-m3", "MiniMax-M3"),
    ]
    assert get_models_for_provider("minimax") == current
    assert [
        (name, model_id)
        for name, model_id in get_models_for_provider("openrouter")
        if name.startswith("minimax-")
    ] == [
        ("minimax-m3.1-flash", "minimax/minimax-m3.1-flash:free"),
        ("minimax-m3", "minimax/minimax-m3"),
    ]
    picker = asyncio.run(list_model_picker_entries(None, include_custom_ollama=False))
    direct_picker = [
        (name, model_id)
        for name, model_id, provider in picker
        if provider == "minimax"
    ]
    assert direct_picker == current
    assert not any(name.startswith("minimax-m2") for name, _ in direct_picker)
    legacy_names = {
        "minimax-m2.5",
        "minimax-m2.5-highspeed",
        "minimax-m2.7",
        "minimax-m2.7-highspeed",
    }
    assert legacy_names <= MODELS.keys()
    assert legacy_names <= {
        name
        for name, _, provider in _MODEL_ENTRIES
        if provider == "minimax"
    }
    assert ("minimax-m2.5", "minimaxai/minimax-m2.5", "nvidia") in picker
    assert ("minimax-m2.5", "Pro/MiniMaxAI/MiniMax-M2.5", "siliconflow") in picker
    assert {"minimax-m2.1", "minimax-m2.5"} <= set(list_models())
    assert picker == list_models_by_provider()


@pytest.mark.parametrize("model_id", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
@pytest.mark.parametrize(
    "base_url",
    ["https://api.minimaxi.com/anthropic", "https://api.minimax.io/anthropic"],
)
def test_current_direct_request_and_history(monkeypatch, model_id, base_url):
    sdk_httpx = anthropic_httpx()
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", base_url)
    model = get_chat_model(model_id.lower(), provider="minimax")
    assert get_context_window(model) == 1_000_000
    captured = []

    def handler(request):
        captured.append((request.url, json.loads(request.content)))
        return sdk_httpx.Response(
            200,
            json={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    history = [
        {"type": "thinking", "thinking": "consider", "signature": "fixture-signature"},
        {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"q": "x"}},
    ]
    with sdk_httpx.Client(transport=sdk_httpx.MockTransport(handler)) as client:
        model._client = anthropic.Anthropic(
            api_key="fixture-key", base_url=base_url, http_client=client, max_retries=0
        )
        model.invoke(
            [
                HumanMessage("hello"),
                AIMessage(content=history),
                ToolMessage(content="found", tool_call_id="tool_1"),
                HumanMessage("continue"),
            ]
        )
    url, payload = captured[0]
    assert str(url) == base_url + "/v1/messages"
    assert payload["model"] == model_id
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["max_tokens"] == 16000
    assert "service_tier" not in payload
    assert "reasoning_effort" not in payload
    assert payload["messages"][1]["content"] == history
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"


@pytest.mark.parametrize(
    ("short_name", "provider", "model_id"),
    [
        ("minimax-m2.7", "minimax", "MiniMax-M2.7"),
        ("minimax-m2.5-highspeed", "minimax", "MiniMax-M2.5-highspeed"),
        ("minimax-m2.5", "nvidia", "minimaxai/minimax-m2.5"),
        ("minimax-m2.5", "siliconflow", "Pro/MiniMaxAI/MiniMax-M2.5"),
    ],
)
def test_explicit_legacy_routes_unchanged(monkeypatch, short_name, provider, model_id):
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fixture-key")
    if provider == "nvidia":
        pytest.importorskip("langchain_nvidia_ai_endpoints")
    model = get_chat_model(short_name, provider=provider)
    assert (getattr(model, "model_name", None) or model.model) == model_id
    if provider == "minimax":
        assert model.thinking == {"type": "enabled", "budget_tokens": 10000}


def test_explicit_thinking_and_custom_route_preserved(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-key")
    model = get_chat_model(
        "minimax-m3", provider="minimax", thinking={"type": "disabled"}
    )
    assert model.thinking == {"type": "disabled"}
    monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "https://custom.example/anthropic")
    monkeypatch.setenv("CUSTOM_ANTHROPIC_API_KEY", "fixture-key")
    custom = get_chat_model("custom-model", provider="custom-anthropic")
    assert custom.model == "custom-model"
    assert custom.thinking is None
    assert custom.anthropic_api_url == "https://custom.example/anthropic"


def test_openrouter_flash_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    model = get_chat_model("minimax-m3.1-flash", provider="openrouter")
    assert get_context_window(model) == 1_000_000
    captured = []

    def handler(request):
        captured.append((request.url, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "chat_fixture",
                "object": "chat.completion",
                "created": 1,
                "system_fingerprint": None,
                "model": "minimax/minimax-m3.1-flash:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    import openrouter

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model.client = openrouter.OpenRouter(
            api_key="fixture-key",
            server_url="https://openrouter.ai/api/v1",
            client=client,
        )
        model.invoke("hello")
    url, payload = captured[0]
    assert str(url) == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["model"] == "minimax/minimax-m3.1-flash:free"
    assert payload["reasoning"]["effort"] == "high"
    assert "thinking" not in payload
