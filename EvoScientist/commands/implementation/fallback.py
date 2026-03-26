from __future__ import annotations

import asyncio
from typing import ClassVar

from ...config import get_effective_config, load_config
from ...config.onboard import _step_model, _step_provider
from ...config.settings import save_config
from ..base import Argument, Command, CommandContext
from ..manager import manager


def _run_prompts() -> str | None:
    config = get_effective_config()
    try:
        provider = _step_provider(config)
        model = _step_model(config, provider)
        return f"{provider}:{model}"
    except KeyboardInterrupt:
        return None


def _update_running_configs(ctx: CommandContext) -> None:
    """Force an active configuration reload so new agents see changes."""
    try:
        import sys

        if "EvoScientist.EvoScientist" in sys.modules:
            sys.modules["EvoScientist.EvoScientist"]._needs_update = True
    except Exception:
        ctx.ui.append_system(
            "Failed to schedule config update, reload the app.", style="red"
        )


async def pick_model() -> str | None:
    # Run the onboarding prompts to select a provider and model
    try:
        from textual._context import active_app

        app = active_app.get()
        # Textual TUI: suspend the app to use questionary
        with app.suspend():
            return await asyncio.to_thread(_run_prompts)
    except Exception:
        return None


class FallbackCommand(Command):
    """Manage fallback models (append, insert, remove, list)."""

    name = "/fallback"
    description = "Manage fallback models"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="action",
            type=str,
            description="Action to perform: append, insert, remove, list",
            required=True,
        ),
        Argument(
            name="idx",
            type=int,
            description="Index for insert or remove actions",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        if not args:
            ctx.ui.append_system(
                "Usage: /fallback <append|insert|remove|list> [idx]", style="red"
            )
            return

        action = args[0].lower()

        if action == "list":
            config = load_config()
            models = list(getattr(config, "fallback_models", []) or [])
            if not models:
                ctx.ui.append_system("No fallback models configured.", style="yellow")
            else:
                ctx.ui.append_system("Fallback models:", style="green")
                for i, m in enumerate(models):
                    ctx.ui.append_system(f"  [{i}] {m}")
            return

        if action == "append":
            model = await pick_model()
            if not model:
                ctx.ui.append_system("Cancelled.", style="dim")
                return

            config = load_config()
            models = list(getattr(config, "fallback_models", []) or [])
            models.append(model)
            config.fallback_models = models
            save_config(config)
            _update_running_configs(ctx)

            ctx.ui.append_system(f"Appended fallback model: {model}", style="green")
            ctx.ui.append_system("Reload with /new to apply.", style="green")
            return

        if action == "insert":
            if len(args) < 2:
                ctx.ui.append_system("Usage: /fallback insert <idx>", style="red")
                return
            try:
                idx = int(args[1])
            except ValueError:
                ctx.ui.append_system("Invalid index. Must be an integer.", style="red")
                return

            model = await pick_model()
            if not model:
                ctx.ui.append_system("Cancelled.", style="dim")
                return

            config = load_config()
            models = list(getattr(config, "fallback_models", []) or [])
            models.insert(idx, model)
            config.fallback_models = models
            save_config(config)
            _update_running_configs(ctx)

            ctx.ui.append_system(
                f"Inserted fallback model {model} at index {idx}", style="green"
            )
            ctx.ui.append_system("Reload with /new to apply.", style="green")
            return

        if action == "remove":
            if len(args) < 2:
                ctx.ui.append_system("Usage: /fallback remove <idx>", style="red")
                return
            try:
                idx = int(args[1])
            except ValueError:
                ctx.ui.append_system("Invalid index. Must be an integer.", style="red")
                return

            config = load_config()
            models = list(getattr(config, "fallback_models", []) or [])
            if not models:
                ctx.ui.append_system("No fallback models configured.", style="yellow")
                return

            try:
                removed = models.pop(idx)
                config.fallback_models = models
                save_config(config)
                _update_running_configs(ctx)
                ctx.ui.append_system(
                    f"Removed fallback model: {removed}", style="green"
                )
                ctx.ui.append_system("Reload with /new to apply.", style="green")
            except IndexError:
                ctx.ui.append_system(f"Index {idx} out of range.", style="red")
            return

        ctx.ui.append_system(
            f"Unknown action: {action}. Use append, insert, remove, or list.",
            style="red",
        )


manager.register(FallbackCommand())
