"""
ToolResultFormatter - content-aware tool result formatting with Rich.

Detects content type (success/error/json/markdown/text) and formats accordingly.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .utils import FAILURE_PREFIX, SUCCESS_PREFIX, truncate
from .utils import is_success as _is_success


class ContentType(Enum):
    """Content type categories."""

    SUCCESS = "success"
    ERROR = "error"
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass
class FormattedResult:
    """Formatted result container."""

    content_type: ContentType
    elements: list[Any]  # Rich renderable elements
    success: bool = True


class ToolResultFormatter:
    """Tool result formatter with content type detection.

    Usage:
        formatter = ToolResultFormatter()
        result = formatter.format("execute", output, max_length=800)
        for elem in result.elements:
            console.print(elem)
    """

    def detect_type(self, content: str) -> ContentType:
        """Detect content type."""
        content = content.strip()

        if content.startswith(SUCCESS_PREFIX):
            body = self._extract_body(content)
            if self._is_json(body):
                return ContentType.JSON
            return ContentType.SUCCESS

        if content.startswith(FAILURE_PREFIX):
            return ContentType.ERROR

        if self._is_json(content):
            return ContentType.JSON

        if self._is_error(content):
            return ContentType.ERROR

        if self._is_markdown(content):
            return ContentType.MARKDOWN

        return ContentType.TEXT

    def format(self, name: str, content: str, max_length: int = 800) -> FormattedResult:
        """Format tool result based on detected content type."""
        content_type = self.detect_type(content)
        success = _is_success(content)

        formatter_map = {
            ContentType.SUCCESS: self._format_success,
            ContentType.ERROR: self._format_error,
            ContentType.JSON: self._format_json,
            ContentType.MARKDOWN: self._format_markdown,
            ContentType.TEXT: self._format_text,
        }

        formatter = formatter_map.get(content_type, self._format_text)
        elements = formatter(name, content, max_length)

        return FormattedResult(
            content_type=content_type, elements=elements, success=success
        )

    def _extract_body(self, content: str) -> str:
        """Extract body after status prefix."""
        lines = content.split("\n", 2)
        return lines[2].strip() if len(lines) > 2 else ""

    def _is_json(self, content: str) -> bool:
        content = content.strip()
        if not content:
            return False
        if (content.startswith("{") and content.endswith("}")) or (
            content.startswith("[") and content.endswith("]")
        ):
            try:
                json.loads(content)
                return True
            except (json.JSONDecodeError, ValueError):
                pass
        return False

    def _is_error(self, content: str) -> bool:
        head = "\n".join(content.splitlines()[:3])
        error_patterns = [
            "Traceback (most recent call last)",
            "Exception:",
            "Error:",
            "Error invoking tool",
            "Failed ",
        ]
        return any(pattern in head for pattern in error_patterns)

    def _is_markdown(self, content: str) -> bool:
        md_patterns = ["```", "**", "##", "- **"]
        return content.startswith("#") or any(p in content for p in md_patterns)

    def _format_success(self, name: str, content: str, max_length: int) -> list[Any]:
        display = truncate(content, max_length)
        return [
            Panel(
                Text(display, style="green"),
                title=f"{escape(name)} OK",
                border_style="green",
            )
        ]

    def _format_error(self, name: str, content: str, max_length: int) -> list[Any]:
        display = truncate(content, max_length)
        return [
            Panel(
                Text(display, style="red"),
                title=f"{escape(name)} FAILED",
                border_style="red",
            )
        ]

    def _format_json(self, name: str, content: str, max_length: int) -> list[Any]:
        json_content = content
        if content.startswith(SUCCESS_PREFIX):
            json_content = self._extract_body(content)

        try:
            data = json.loads(json_content)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            formatted = truncate(formatted, max_length)
            return [
                Text(f"{name} OK", style="cyan bold"),
                Syntax(formatted, "json", theme="monokai", line_numbers=False),
            ]
        except (json.JSONDecodeError, ValueError):
            return self._format_text(name, content, max_length)

    def _format_markdown(self, name: str, content: str, max_length: int) -> list[Any]:
        display = truncate(content, max_length)
        return [
            Panel(
                Markdown(display),
                title=escape(name),
                border_style="cyan dim",
            )
        ]

    def _format_text(self, name: str, content: str, max_length: int) -> list[Any]:
        display = truncate(content, max_length)
        return [
            Text(f"{name}:", style="cyan bold"),
            Text(f"   {display}", style="dim"),
        ]
