"""Clipboard utilities for EvoScientist TUI.

Provides copy-on-select and paste for the Textual TUI with fallback methods:

Copy (3 methods):
  1. pyperclip  — preferred on local machines (uses pbcopy on macOS)
  2. Textual    — built-in app.copy_to_clipboard()
  3. OSC 52     — escape sequence for SSH / tmux remote sessions

Paste (3 methods):
  1. pyperclip  — preferred on local machines
  2. Platform-native — pbpaste (macOS), xclip/xsel (Linux), PowerShell (Windows)
  3. Textual    — built-in app.paste() on supported terminals
"""

from __future__ import annotations

import base64
import logging
import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)

_PREVIEW_MAX = 40


# ── Platform-native clipboard read ────────────────────────────────


def _paste_native() -> str | None:
    """Read clipboard using platform-native commands.

    Returns:
        Clipboard text, or None if unavailable.
    """
    if sys.platform == "darwin":
        # macOS: pbpaste
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    elif sys.platform == "win32":
        # Windows: PowerShell Get-Clipboard
        try:
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                # Try UTF-8 first, fall back to system encoding
                for encoding in ("utf-8", "gbk", "cp936"):
                    try:
                        return result.stdout.decode(encoding).rstrip("\r\n")
                    except UnicodeDecodeError:
                        continue
                # Last resort: decode with errors ignored
                return result.stdout.decode("utf-8", errors="ignore").rstrip("\r\n")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    else:
        # Linux: try xclip, then xsel
        for cmd in (
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
        ):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    return result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
    return None


# ── OSC 52 (remote / SSH / tmux) ──────────────────────────────────


def _copy_osc52(text: str) -> None:
    """Copy text using OSC 52 escape sequence (works over SSH/tmux)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    seq = f"\033]52;c;{encoded}\a"
    if os.environ.get("TMUX"):
        seq = f"\033Ptmux;\033{seq}\033\\"
    with pathlib.Path("/dev/tty").open("w", encoding="utf-8") as tty:
        tty.write(seq)
        tty.flush()


# ── Preview helper ─────────────────────────────────────────────────


def _shorten(texts: list[str]) -> str:
    """Return a short preview string for the notification toast."""
    dense = "⏎".join(texts).replace("\n", "⏎")
    if len(dense) > _PREVIEW_MAX:
        return dense[: _PREVIEW_MAX - 1] + "…"
    return dense


# ── Public API ─────────────────────────────────────────────────────


def copy_selection_to_clipboard(app: App) -> None:
    """Copy mouse-selected text from any widget to the system clipboard.

    Called from ``on_mouse_up`` in the Textual app so that selecting
    text with the mouse automatically copies it.
    """
    selected_texts: list[str] = []

    for widget in app.query("*"):
        if not hasattr(widget, "text_selection") or not widget.text_selection:
            continue
        selection = widget.text_selection
        if selection.end is None:
            continue
        try:
            result = widget.get_selection(selection)
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            logger.debug(
                "Failed to get selection from %s: %s",
                type(widget).__name__,
                exc,
            )
            continue
        if not result:
            continue
        text, _ = result
        if text.strip():
            selected_texts.append(text)

    if not selected_texts:
        return

    combined = "\n".join(selected_texts)

    # Try methods in priority order
    copy_methods = [app.copy_to_clipboard]

    try:
        import pyperclip

        copy_methods.insert(0, pyperclip.copy)
    except ImportError:
        pass

    copy_methods.append(_copy_osc52)

    for fn in copy_methods:
        try:
            fn(combined)
            app.notify(
                f'"{_shorten(selected_texts)}" copied',
                severity="information",
                timeout=2,
                markup=False,
            )
        except (OSError, RuntimeError, TypeError) as exc:
            logger.debug(
                "Clipboard method %s failed: %s", getattr(fn, "__name__", repr(fn)), exc
            )
            continue
        else:
            return

    app.notify(
        "Failed to copy — no clipboard method available",
        severity="warning",
        timeout=3,
    )


def get_clipboard_text() -> str | None:
    """Read text from the system clipboard.

    Tries multiple methods in priority order:
      1. pyperclip (if installed)
      2. Platform-native commands (pbpaste, xclip, PowerShell)

    Returns:
        Clipboard text, or None if unavailable or empty.
    """
    # 1. Try pyperclip first
    try:
        import pyperclip

        text = pyperclip.paste()
        if text:
            return text
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pyperclip.paste() failed: %s", exc)

    # 2. Try platform-native commands
    text = _paste_native()
    if text:
        return text

    return None
