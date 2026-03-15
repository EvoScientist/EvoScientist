"""Tests for EvoScientist memory middleware fallback extraction."""

from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.middleware.memory import EvoMemoryMiddleware


class _StructuredModelFailure:
    def invoke(self, prompt):
        raise ValueError("Structured Output response does not have a parsed field")


class _FallbackModel:
    def with_structured_output(self, schema, **kwargs):
        return _StructuredModelFailure()

    def invoke(self, prompt):
        return AIMessage(
            content=[
                {
                    "type": "text",
                    "text": '{"user_profile":{"language":"Chinese"},"learned_preferences":["prefers codex proxy"]}',
                }
            ]
        )


def test_extract_falls_back_to_raw_json_response():
    """Fallback should parse raw JSON text from Responses API style content."""
    middleware = EvoMemoryMiddleware(backend=None)
    model = _FallbackModel()

    result = middleware._extract(model, "", [HumanMessage(content="Please remember I prefer Chinese.")])

    assert result["user_profile"]["language"] == "Chinese"
    assert result["learned_preferences"] == ["prefers codex proxy"]
