"""Tests for provider-agnostic context-window resolution."""

from types import SimpleNamespace

from EvoScientist.llm.context_window import (
    DEFAULT_CONTEXT_WINDOW_FALLBACK,
    apply_known_context_window,
    get_context_window,
    resolve_context_window,
)


def test_prefers_direct_context_window_over_profile():
    model = SimpleNamespace(
        context_window=512_000,
        profile={"max_input_tokens": 200_000},
    )

    assert get_context_window(model) == 512_000


def test_uses_direct_context_length_attribute():
    model = SimpleNamespace(context_length=1_000_000)

    assert get_context_window(model) == 1_000_000


def test_uses_ollama_num_ctx():
    model = SimpleNamespace(model="llama3.2", num_ctx=65_536, profile=None)

    assert get_context_window(model) == 65_536


def test_uses_profile_context_length_before_max_input_tokens():
    model = SimpleNamespace(
        profile={
            "context_length": 1_000_000,
            "max_input_tokens": 128_000,
        }
    )

    assert get_context_window(model) == 1_000_000


def test_uses_profile_max_input_tokens_when_needed():
    model = SimpleNamespace(profile={"max_input_tokens": 200_000})

    assert get_context_window(model) == 200_000


def test_accepts_numeric_string_values():
    model = SimpleNamespace(profile={"context_length": "1_048_576"})

    assert get_context_window(model) == 1_048_576


def test_resolve_context_window_falls_back_when_missing():
    model = SimpleNamespace(profile=None)

    assert get_context_window(model) is None
    assert resolve_context_window(model) == DEFAULT_CONTEXT_WINDOW_FALLBACK
    assert resolve_context_window(model, fallback=42_000) == 42_000


def test_patch_table_resolves_known_model_name():
    model = SimpleNamespace(model_name="claude-opus-4.7", profile=None)

    assert get_context_window(model) == 1_000_000


def test_patch_table_strips_provider_prefix():
    full = SimpleNamespace(model_name="openai/gpt-5.5", profile=None)
    short = SimpleNamespace(model_name="gpt-5.5", profile=None)

    assert get_context_window(full) == 1_050_000
    assert get_context_window(short) == 1_050_000


def test_real_attribute_beats_patch_table():
    model = SimpleNamespace(
        model_name="claude-opus-4.7",
        context_window=500_000,
        profile=None,
    )

    assert get_context_window(model) == 500_000


def test_unknown_model_falls_through_to_fallback():
    model = SimpleNamespace(model_name="some-unreleased-model", profile=None)

    assert get_context_window(model) is None
    assert resolve_context_window(model) == DEFAULT_CONTEXT_WINDOW_FALLBACK


def test_claude_family_pattern_covers_all_variants():
    # Native Anthropic (dash form)
    native = SimpleNamespace(model_name="claude-opus-4-7", profile=None)
    # OpenRouter (dot form, with vendor prefix)
    openrouter = SimpleNamespace(model_name="anthropic/claude-sonnet-4.6", profile=None)
    sonnet = SimpleNamespace(model_name="claude-sonnet-4-5", profile=None)
    # Haiku 4.5 is a dict-level exception (200K, not 1M).
    haiku = SimpleNamespace(model_name="claude-haiku-4-5", profile=None)

    assert get_context_window(native) == 1_000_000
    assert get_context_window(openrouter) == 1_000_000
    assert get_context_window(sonnet) == 1_000_000
    assert get_context_window(haiku) == 200_000


def test_dict_entry_overrides_family_pattern():
    # Qwen 3.6 closed-source default is 1M (family), but open-source -27b
    # variant is 262K (dict override). Dict wins because it's checked first.
    closed = SimpleNamespace(model_name="qwen/qwen3.6-flash", profile=None)
    open_27b = SimpleNamespace(model_name="qwen/qwen3.6-27b", profile=None)
    open_plus = SimpleNamespace(model_name="qwen3.6-plus", profile=None)

    assert get_context_window(closed) == 1_000_000
    assert get_context_window(open_27b) == 262_000
    assert get_context_window(open_plus) == 1_000_000


def test_claude_3x_matches_family_but_upstream_profile_wins_in_practice():
    # The broad ``claude-`` family pattern intentionally also matches older
    # Claude 3.x. This is safe because langchain-anthropic ships upstream
    # profiles for those (200K), and profile lookup runs before the patch
    # table — so the family value never actually reaches downstream callers
    # for normal Anthropic usage.
    old_sonnet_no_profile = SimpleNamespace(
        model_name="claude-3-5-sonnet-20241022", profile=None
    )
    old_sonnet_with_profile = SimpleNamespace(
        model_name="claude-3-5-sonnet-20241022",
        profile={"max_input_tokens": 200_000},
    )

    # No profile (rare custom-anthropic edge case): family fires.
    assert get_context_window(old_sonnet_no_profile) == 1_000_000
    # Normal langchain-anthropic path: upstream profile wins.
    assert get_context_window(old_sonnet_with_profile) == 200_000


def test_apply_injects_into_empty_profile():
    model = SimpleNamespace(model_name="claude-opus-4-7", profile=None)

    apply_known_context_window(model)

    assert model.profile == {"max_input_tokens": 1_000_000}


def test_apply_preserves_existing_profile_keys():
    model = SimpleNamespace(
        model_name="qwen3.6-flash",
        profile={"name": "Qwen 3.6 Flash"},
    )

    apply_known_context_window(model)

    assert model.profile == {
        "name": "Qwen 3.6 Flash",
        "max_input_tokens": 1_000_000,
    }


def test_apply_does_not_overwrite_existing_max_input_tokens():
    model = SimpleNamespace(
        model_name="claude-opus-4-7",
        profile={"max_input_tokens": 500_000},
    )

    apply_known_context_window(model)

    assert model.profile == {"max_input_tokens": 500_000}


def test_apply_skips_unknown_models():
    model = SimpleNamespace(model_name="some-unknown-model", profile=None)

    apply_known_context_window(model)

    assert model.profile is None


def test_apply_handles_non_dict_profile_safely():
    # Non-dict profile (e.g. namespace from a future langchain version) must
    # not raise — fall back to a fresh dict so deepagents can still read it.
    model = SimpleNamespace(
        model_name="claude-opus-4-7",
        profile=SimpleNamespace(name="opaque object"),
    )

    apply_known_context_window(model)

    assert model.profile == {"max_input_tokens": 1_000_000}


def test_lookup_is_case_insensitive():
    # SiliconFlow uses capitalized IDs like "Pro/moonshotai/Kimi-K2.5"
    siliconflow = SimpleNamespace(model_name="Pro/moonshotai/Kimi-K2.5", profile=None)
    assert get_context_window(siliconflow) == 262_000

    # GLM-5 capitalized variant
    glm = SimpleNamespace(model_name="Pro/zai-org/GLM-5", profile=None)
    assert get_context_window(glm) == 203_000
