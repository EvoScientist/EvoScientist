"""Slash command for managing the model fallback chain.

Provides ``/model-fallback`` (alias ``/fallback``) with subcommands to
add, remove, list, clear, save, and display help for fallback models.
"""

from __future__ import annotations

from typing import ClassVar

from ..base import Argument, Command, CommandContext
from ..manager import manager


class ModelFallbackCommand(Command):
    """Manage the model fallback chain.

    Subcommands:
        add     -- Append a model (interactive picker when no args).
        remove  -- Remove by position (interactive picker in TUI).
        list    -- Display the current chain.
        clear   -- Remove all entries.
        save    -- Persist the chain to the config file.
        help    -- Show subcommand reference.
    """

    name = "/model-fallback"
    alias: ClassVar[list[str]] = ["/fallback"]
    description = "Manage fallback models (add/remove/list/clear)"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="action",
            type=str,
            description="add|remove|list|clear|save|help",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...llm.models import MODELS
        from ...middleware.model_fallback import (
            add_fallback,
            clear_fallbacks,
            get_fallback_chain,
            remove_fallback,
            remove_fallback_at,
            serialize_fallback_chain,
        )

        save = "--save" in args
        args = [a for a in args if a != "--save"]

        if not args:
            await self._show_list(ctx, get_fallback_chain())
            return

        action = args[0].lower()

        if action == "list":
            await self._show_list(ctx, get_fallback_chain())

        elif action == "add":
            if len(args) >= 2:
                model_name = args[1]
                provider = args[2] if len(args) > 2 else None

                if provider is None:
                    if model_name in MODELS:
                        _, provider = MODELS[model_name]
                    else:
                        ctx.ui.append_system(
                            f"Unknown model '{model_name}'. Specify provider explicitly: "
                            f"/model-fallback add {model_name} <provider>",
                            style="red",
                        )
                        return
            else:
                picked = await self._pick_model(ctx)
                if picked is None:
                    return
                model_name, provider = picked

            if add_fallback(model_name, provider):
                ctx.ui.append_system(
                    f"Added {model_name} ({provider}) to fallback chain", style="green"
                )
            else:
                ctx.ui.append_system(
                    f"{model_name} ({provider}) is already in the fallback chain",
                    style="yellow",
                )
                return

            if save:
                self._save_to_config(serialize_fallback_chain())

        elif action == "remove":
            chain = get_fallback_chain()
            if not chain:
                ctx.ui.append_system("Fallback chain is empty", style="yellow")
                return

            if len(args) >= 2:
                arg = args[1]
                try:
                    idx = int(arg) - 1
                except ValueError:
                    ctx.ui.append_system(
                        f"Expected a position number (1-{len(chain)}), got '{arg}'. "
                        "Use /model-fallback list to see positions.",
                        style="red",
                    )
                    return
                removed = remove_fallback_at(idx)
                if removed is None:
                    ctx.ui.append_system(
                        f"Invalid position {arg}. "
                        f"Use a number between 1 and {len(chain)}.",
                        style="red",
                    )
                    return
                model_name, provider = removed
            else:
                picked = await self._pick_fallback_to_remove(ctx, chain)
                if picked is None:
                    return
                model_name, provider = picked
                remove_fallback(model_name)

            ctx.ui.append_system(
                f"Removed {model_name} ({provider}) from fallback chain",
                style="green",
            )

            if save:
                self._save_to_config(serialize_fallback_chain())

        elif action == "clear":
            clear_fallbacks()
            ctx.ui.append_system("Cleared all fallback models", style="green")

            if save:
                self._save_to_config("")

        elif action == "save":
            self._save_to_config(serialize_fallback_chain())
            ctx.ui.append_system("Fallback chain saved to config", style="green")

        elif action == "help":
            self._show_help(ctx)

        else:
            self._show_help(ctx)

    async def _pick_model(self, ctx: CommandContext) -> tuple[str, str] | None:
        """Open the interactive model picker to select a fallback model.

        Falls back to a usage hint when the UI does not support interactive
        widgets (CLI mode without a model argument).

        Args:
            ctx: Current command context.

        Returns:
            ``(model_name, provider)`` tuple, or ``None`` if cancelled.
        """
        if not ctx.ui.supports_interactive:
            ctx.ui.append_system(
                "Usage: /model-fallback add <model> [provider]", style="yellow"
            )
            return None

        from ...EvoScientist import _ensure_config
        from ...llm.models import list_models_by_provider

        cfg = _ensure_config()
        entries = list_models_by_provider()

        ollama_base_url = getattr(cfg, "ollama_base_url", None)
        if ollama_base_url:
            from ...llm.ollama_discovery import discover_ollama_models

            detected = await discover_ollama_models(ollama_base_url, timeout=1.5)
            for detected_name in detected:
                entries.append((detected_name, detected_name, "ollama"))
            entries.append(("Custom Ollama model...", "__custom_ollama__", "ollama"))

        result = await ctx.ui.wait_for_model_pick(
            entries,
            current_model=cfg.model,
            current_provider=cfg.provider,
        )
        if result is None:
            return None

        name, provider = result
        if provider == "ollama" and name in (
            "Custom Ollama model...",
            "__custom_ollama__",
        ):
            return None
        return name, provider

    async def _pick_fallback_to_remove(
        self, ctx: CommandContext, chain: list[tuple[str, str]]
    ) -> tuple[str, str] | None:
        """Open the model picker populated with the current fallback chain.

        Falls back to a usage hint in CLI mode.

        Args:
            ctx: Current command context.
            chain: The current fallback chain to choose from.

        Returns:
            ``(model_name, provider)`` tuple, or ``None`` if cancelled.
        """
        if not ctx.ui.supports_interactive:
            ctx.ui.append_system(
                "Usage: /model-fallback remove <position>  "
                "(use /model-fallback list to see positions)",
                style="yellow",
            )
            return None

        entries = [(m, m, p) for m, p in chain]
        result = await ctx.ui.wait_for_model_pick(
            entries, current_model=None, current_provider=None
        )
        if result is None:
            return None
        return result

    def _show_help(self, ctx: CommandContext) -> None:
        """Render the subcommand reference table.

        Args:
            ctx: Current command context.
        """
        from rich.text import Text

        text = Text("/model-fallback subcommands:\n", style="bold")
        for cmd, desc in (
            (
                "add [model] [provider]",
                "Add a fallback model (opens picker if omitted)",
            ),
            (
                "remove [position]",
                "Remove a fallback by position (opens picker in TUI)",
            ),
            ("list", "Show the current fallback chain"),
            ("clear", "Remove all fallback models"),
            ("save", "Save current fallback chain to config file"),
            ("help", "Show this help message"),
        ):
            text.append(f"  {cmd:<26}", style="cyan")
            text.append(f"{desc}\n", style="dim")
        text.append(
            "\nAdd --save to add/remove/clear to persist the change immediately.\n",
            style="dim",
        )
        ctx.ui.mount_renderable(text)

    async def _show_list(
        self, ctx: CommandContext, chain: list[tuple[str, str]]
    ) -> None:
        """Display the current fallback chain as a numbered list.

        Args:
            ctx: Current command context.
            chain: The fallback chain to display.
        """
        if not chain:
            ctx.ui.append_system("No fallback models configured", style="dim")
            ctx.ui.append_system(
                "Use /model-fallback add <model> [provider] to add one",
                style="dim",
            )
            return

        from rich.text import Text

        text = Text("Fallback chain:\n", style="bold")
        for idx, (model, provider) in enumerate(chain, 1):
            text.append(f"  {idx}. ", style="dim")
            text.append(model, style="cyan")
            text.append(f" ({provider})\n", style="dim")
        ctx.ui.mount_renderable(text)

    def _save_to_config(self, value: str) -> None:
        """Persist the fallback chain string to the config file.

        Args:
            value: Serialized chain (``"model:provider,..."``).
        """
        from ...config.settings import set_config_value

        set_config_value("model_fallbacks", value)


manager.register(ModelFallbackCommand())
