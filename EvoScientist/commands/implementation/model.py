from __future__ import annotations

from typing import ClassVar

from ..base import Argument, Command, CommandContext
from ..manager import manager


def extract_model_and_provider(
    args: list[str], fallback_provider: str
) -> tuple[str, str]:
    """Parse model name and provider from argument list.

    Args:
        args: Non-empty argument list (model_name [provider]).
        fallback_provider: Provider to use when not specified and model
            is not in the registry.

    Returns:
        ``(model_name, provider)`` tuple.

    Raises:
        ValueError: If the model is unknown and no provider was given.
    """
    from ...llm.models import MODELS

    model_name = args[0]
    provider_override = args[1] if len(args) > 1 else None

    if model_name not in MODELS and provider_override is None:
        raise ValueError(f"Unknown model '{model_name}'")

    if provider_override:
        provider = provider_override
    elif model_name in MODELS:
        _, provider = MODELS[model_name]
    else:
        provider = fallback_provider

    return model_name, provider


class ModelCommand(Command):
    """Switch the LLM model for the current session."""

    name = "/model"
    description = "Switch model (--save to persist)"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="model_name",
            type=str,
            description="Model short name (e.g. claude-sonnet-4-6). Opens picker if omitted.",
            required=False,
        ),
        Argument(
            name="--save",
            type=str,
            description="Save the choice to config file",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...EvoScientist import _ensure_config
        from ...llm.models import list_models_by_provider

        cfg = _ensure_config()
        current_model = cfg.model
        current_provider = cfg.provider

        # Parse --save flag
        save = "--save" in args
        args = [a for a in args if a != "--save"]

        if args:
            try:
                model_name, provider = extract_model_and_provider(
                    args, fallback_provider=current_provider
                )
            except ValueError:
                ctx.ui.append_system(
                    f"Unknown model '{args[0]}'. Use /model to browse available models.",
                    style="red",
                )
                return

            await self._apply_model(ctx, model_name, provider, save=save)
            return

        # Interactive picker
        entries = list_models_by_provider()
        result = await ctx.ui.wait_for_model_pick(
            entries,
            current_model=current_model,
            current_provider=current_provider,
        )
        if result is None:
            return

        name, provider = result
        await self._apply_model(ctx, name, provider, save=save)

    async def _apply_model(
        self,
        ctx: CommandContext,
        model_name: str,
        provider: str,
        *,
        save: bool = False,
    ) -> None:
        from ...EvoScientist import _ensure_config, set_chat_model

        try:
            set_chat_model(model_name, provider=provider)
        except Exception as e:
            ctx.ui.append_system(f"Failed to switch model: {e}", style="red")
            return

        # Update cached config so status bar and new agents pick up the change
        cfg = _ensure_config()
        cfg.model = model_name
        cfg.provider = provider

        # Persist to config file if --save was given
        if save:
            from ...config.settings import set_config_value

            set_config_value("model", model_name)
            set_config_value("provider", provider)

        # Rebuild the agent so subsequent turns use the new model
        from ...cli.agent import _load_agent

        new_agent = _load_agent(
            workspace_dir=ctx.workspace_dir,
            checkpointer=ctx.checkpointer,
            config=cfg,
        )
        ctx.agent = new_agent

        # Propagate to channel module if channels are running
        try:
            import EvoScientist.cli.channel as _ch_mod

            if getattr(_ch_mod, "_cli_agent", None) is not None:
                _ch_mod._cli_agent = new_agent
        except Exception:
            pass

        # Update status bar if available
        update_model_fn = getattr(ctx.ui, "update_status_after_model_change", None)
        if callable(update_model_fn):
            update_model_fn(model_name)

        saved_note = " (saved to config)" if save else ""
        ctx.ui.append_system(
            f"Switched to {model_name} ({provider}){saved_note}", style="green"
        )


manager.register(ModelCommand())
