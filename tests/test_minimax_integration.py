"""Integration tests for MiniMax direct provider.

These tests validate that the MiniMax provider can connect to the real
MiniMax API (api.minimaxi.com/anthropic by default) and produce chat completions.

Requires MINIMAX_API_KEY environment variable to be set.
Optionally set MINIMAX_BASE_URL to https://api.minimax.io/anthropic for Global keys.
"""

import os

import pytest

from EvoScientist.llm import get_chat_model

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


class TestMiniMaxIntegration:
    def test_minimax_m25_chat_completion(self):
        """Test that MiniMax M2.5 can produce a chat completion."""
        model = get_chat_model("minimax-m2.5", provider="minimax", temperature=0)
        response = model.invoke("Reply with exactly: hello")
        assert response.content
        assert len(response.content) > 0

    def test_minimax_m25_highspeed_chat_completion(self):
        """Test that MiniMax M2.5-highspeed can produce a chat completion."""
        model = get_chat_model(
            "minimax-m2.5-highspeed", provider="minimax", temperature=0
        )
        response = model.invoke("Reply with exactly: world")
        assert response.content
        assert len(response.content) > 0

    def test_minimax_with_full_model_id(self):
        """Test using the full model ID directly."""
        model = get_chat_model("MiniMax-M2.5", provider="minimax", temperature=0)
        response = model.invoke("What is 2+2? Answer with just the number.")
        assert response.content
        assert "4" in response.content

    def test_minimax_m27_chat_completion(self):
        """Test that MiniMax M2.7 can produce a chat completion."""
        model = get_chat_model("minimax-m2.7", provider="minimax", temperature=0)
        response = model.invoke("Reply with exactly: test")
        assert response.content
        assert len(response.content) > 0
