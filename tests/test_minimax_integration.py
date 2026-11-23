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
    def test_minimax_m31_flash_chat_completion(self):
        """Test that MiniMax M3.1 Flash can produce a chat completion."""
        model = get_chat_model("minimax-m3.1-flash", provider="minimax", temperature=0)
        response = model.invoke("Reply with exactly: hello")
        assert response.text

    def test_minimax_with_full_model_id(self):
        """Test using the full model ID directly."""
        model = get_chat_model("MiniMax-M3", provider="minimax", temperature=0)
        response = model.invoke("What is 2+2? Answer with just the number.")
        assert "4" in response.text

    def test_minimax_m3_chat_completion(self):
        """Test that MiniMax M3 can produce a chat completion."""
        model = get_chat_model("minimax-m3", provider="minimax", temperature=0)
        response = model.invoke("Reply with exactly: test")
        assert response.text
