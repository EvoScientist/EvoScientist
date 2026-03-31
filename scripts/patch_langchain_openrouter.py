#!/usr/bin/env python3
"""Patch langchain-openrouter v0.2.1 reasoning_details multi-turn bug.

Bug: reasoning_details entries have type-specific schemas:
  - "thinking"           → requires "content" field
  - "reasoning.summary"  → requires "summary" field
  - "reasoning.encrypted"→ requires "data" field

But streaming chunks store everything as "content" in additional_kwargs.
When sent back to OpenRouter API in conversation history, Pydantic
validation fails on non-"thinking" types.

Additionally, LangChain's AIMessageChunk.__add__ list-concatenates
reasoning_details, fragmenting them into multiple entries.

Fix: Drop reasoning_details entirely from _convert_message_to_dict().
The "reasoning" field (plain text) already carries reasoning content —
reasoning_details is not needed for history replay.

This patch is lost on reinstall — reapply after `pip install langchain-openrouter`.

Usage:
    python scripts/patch_langchain_openrouter.py
"""

from __future__ import annotations

import importlib
import pathlib
import sys

PATCH_MARKER = "# --- PATCH: drop reasoning_details from conversation history ---"

# The buggy code in _convert_message_to_dict (v0.2.1, lines ~1182-1185)
OLD_CODE = """\
        if "reasoning_details" in message.additional_kwargs:
            message_dict["reasoning_details"] = message.additional_kwargs[
                "reasoning_details"
            ]"""

# The patched replacement: drop reasoning_details entirely
NEW_CODE = """\
        {marker}
        # langchain-openrouter v0.2.1: reasoning_details entries have
        # type-specific schemas (thinking->content, reasoning.summary->summary,
        # reasoning.encrypted->data) but streaming chunks store everything as
        # "content", causing Pydantic validation errors on multi-turn.
        # The "reasoning" field (plain text) already carries the reasoning
        # content -- reasoning_details is not needed for history replay.
        # Intentionally NOT sending reasoning_details back to the API.""".format(  # noqa: UP032
    marker=PATCH_MARKER,
)


def main() -> None:
    try:
        mod = importlib.import_module("langchain_openrouter.chat_models")
    except ImportError:
        print("ERROR: langchain-openrouter is not installed.")
        sys.exit(1)

    src = pathlib.Path(mod.__file__)  # type: ignore[arg-type]
    text = src.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print(f"Already patched: {src}")
        return

    if OLD_CODE not in text:
        print(f"WARNING: Could not find expected code block in {src}")
        print("The package version may have changed. Manual patching required.")
        sys.exit(1)

    patched = text.replace(OLD_CODE, NEW_CODE, 1)
    src.write_text(patched, encoding="utf-8")

    # Clear .pyc cache
    cache = src.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob("chat_models*.pyc"):
            pyc.unlink()
            print(f"Removed cache: {pyc}")

    print(f"Patched: {src}")
    print("langchain-openrouter reasoning_details bug fixed.")


if __name__ == "__main__":
    main()
