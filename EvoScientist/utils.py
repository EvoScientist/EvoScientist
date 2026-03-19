"""Utility functions for EvoScientist.

This module primarily contains helpers for displaying messages and prompts in
notebooks, and lightweight configuration loaders used by the agent runtime.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


def format_message_content(message):
    """Convert message content to displayable string.

    Args:
        message: A LangChain message object with content attribute.

    Returns:
        Formatted string representation of the message content.
    """
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "tool_use":
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting.

    Args:
        messages: List of LangChain message objects to display.
    """
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")
        content = format_message_content(m)

        if msg_type == "Human":
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == "Ai":
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == "Tool":
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # Highlight headers
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )


def load_subagents(
    config_path: Path,
    *,
    tool_registry: dict[str, Any],
    prompt_refs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load subagent definitions from YAML and wire up tools.

    NOTE: This is a custom utility. deepagents does not natively load subagents
    from files - they're normally defined inline in the create_deep_agent() call.
    We externalize to YAML here to keep configuration separate from code.

    Supported YAML schemas:

    1) Mapping style (recommended):
       planner-agent:
         description: "..."
         tools: [think_tool]
         system_prompt: |
           ...
       research-agent:
         description: "..."
         tools: [tavily_search, think_tool]
         system_prompt_ref: RESEARCHER_INSTRUCTIONS

    2) List style (legacy):
       subagents:
         - name: planner-agent
           description: "..."
           tools: [think_tool]
           system_prompt: |
             ...
    """
    prompt_refs = prompt_refs or {}

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict) or not config:
        raise ValueError("subagent.yaml must be a mapping or contain 'subagents:'")

    subagents: list[dict[str, Any]] = []

    def _build_one(name: str, spec: dict[str, Any]) -> dict[str, Any]:
        subagent: dict[str, Any] = {
            "name": name,
            "description": spec.get("description", ""),
        }

        if "system_prompt_ref" in spec:
            ref = spec["system_prompt_ref"]
            if ref not in prompt_refs:
                raise ValueError(
                    f"Unknown system_prompt_ref '{ref}' for subagent '{name}'"
                )
            subagent["system_prompt"] = prompt_refs[ref]
        else:
            subagent["system_prompt"] = spec.get("system_prompt", "")

        if "model" in spec:
            subagent["model"] = spec["model"]

        if "skills" in spec:
            subagent["skills"] = spec["skills"]

        if "tools" in spec:
            resolved = []
            for t in spec["tools"]:
                if t in tool_registry:
                    resolved.append(tool_registry[t])
                else:
                    logger.warning(
                        "Subagent %r: tool %r not in registry, skipping", name, t
                    )
            subagent["tools"] = resolved

        return subagent

    # Legacy list style
    if "subagents" in config:
        items = config.get("subagents")
        if not isinstance(items, list) or not items:
            raise ValueError("subagent.yaml must contain a non-empty 'subagents:' list")
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                raise ValueError("Each subagent entry must have a 'name'")
            subagents.append(_build_one(name, item))
        return subagents

    # Mapping style: {<name>: <spec>}
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        subagents.append(_build_one(name, spec))

    return subagents


def load_subagent(
    config_path: Path,
    name: str,
    *,
    tool_registry: dict[str, Any],
    prompt_refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load a single sub-agent by name from YAML."""
    for agent in load_subagents(
        config_path,
        tool_registry=tool_registry,
        prompt_refs=prompt_refs,
    ):
        if agent.get("name") == name:
            return agent
    raise KeyError(f"Sub-agent not found: {name}")
