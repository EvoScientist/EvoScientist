from __future__ import annotations

from typing import ClassVar

from ..base import Argument, Command, CommandContext
from ..manager import manager


def extract_model_and_provider(args: list[str]) -> tuple[str, str]:
    """Parse model name and provider from argument list.

    Args:
        args: Non-empty argument list (model_name [provider]).

    Returns:
        ``(model_name, provider)`` tuple.

    Raises:
        ValueError: If the model is not in the registry. Skipped when
            ``provider_override == "ollama"``, since Ollama models are
            locally-installed and never appear in ``MODELS``.
    """
    from ...llm.models import MODELS

    model_name = args[0]
    provider_override = args[1] if len(args) > 1 else None

    # Ollama models are locally-installed — not in the registry. Pass the name
    # through verbatim; get_chat_model's "Assume full model ID" fallback
    # (models.py) accepts them.
    if provider_override == "ollama":
        return model_name, "ollama"

    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'")

    if provider_override:
        provider = provider_override
    else:
        _, provider = MODELS[model_name]

    return model_name, provider


class ModelCommand(Command):
    """Switch the LLM model for the current session."""

    name = "/model"
    description = "Switch model (--save to persist)"
    # ``--save`` is parsed manually in ``execute`` via ``"--save" in args``;
    # ``type=bool`` below is declarative metadata, not enforced by the manager.
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="model_name",
            type=str,
            description="Model short name (e.g. claude-sonnet-4-6). Opens picker if omitted.",
            required=False,
        ),
        Argument(
            name="--save",
            type=bool,
            description="Save the choice to config file",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...EvoScientist import _ensure_config
        from ...llm.models import list_model_picker_entries

        cfg = _ensure_config()
        current_model = cfg.model
        current_provider = cfg.provider

        # Parse --save flag
        save = "--save" in args
        args = [a for a in args if a != "--save"]

        if args:
            try:
                model_name, provider = extract_model_and_provider(args)
            except ValueError:
                ctx.ui.append_system(
                    f"Unknown model '{args[0]}'. Use /model to browse available models.",
                    style="red",
                )
                return

            await self._apply_model(ctx, model_name, provider, save=save)
            return

        # Interactive picker
        if not ctx.ui.supports_interactive:
            ctx.ui.append_system(
                "Usage: /model <name> [provider] [--save]",
                style="yellow",
            )
            return

        entries = await list_model_picker_entries(
            getattr(cfg, "ollama_base_url", None),
            include_custom_ollama=True,
        )

        result = await ctx.ui.wait_for_model_pick(
            entries,
            current_model=current_model,
            current_provider=current_provider,
        )
        if result is None:
            return

        name, provider = result
        # Defense-in-depth: the widget should have replaced the sentinel with
        # the user-typed name. If it didn't, treat as cancel rather than try
        # to switch to a literal "__custom_ollama__" model.
        if provider == "ollama" and name in (
            "Custom Ollama model...",
            "__custom_ollama__",
        ):
            return
        await self._apply_model(ctx, name, provider, save=save)

    async def _apply_model(
        self,
        ctx: CommandContext,
        model_name: str,
        provider: str,
        *,
        save: bool = False,
    ) -> None:
        import copy

        from ...cli.agent import _load_agent
        from ...EvoScientist import (
            _build_chat_model,
            _ensure_config,
            set_active_config,
            set_chat_model_instance,
        )

        cfg = _ensure_config()

        # Build a temporary config + its chat model and verify the agent can be
        # built before committing anything. ``create_cli_agent(config=...,
        # chat_model=...)`` is pure (issue #183) — it writes none of the cached
        # config/model module globals — so a failure below leaves the session
        # on the original model with no snapshot/restore needed.
        temp_cfg = copy.copy(cfg)
        temp_cfg.model = model_name
        temp_cfg.provider = provider

        try:
            new_chat_model = _build_chat_model(temp_cfg)
            new_agent = _load_agent(
                workspace_dir=ctx.workspace_dir,
                checkpointer=ctx.checkpointer,
                config=temp_cfg,
                chat_model=new_chat_model,
            )
        except Exception as e:
            ctx.ui.append_system(f"Failed to switch model: {e}", style="red")
            return

        # Agent built with no global mutation — commit the switch atomically.
        # These are pure assignments and cannot fail, so the session can never
        # be left half-switched. Apply the switch to the LIVE ``cfg`` in place
        # (the active config object) instead of rebinding ``_config`` to the
        # fresh ``temp_cfg`` — callers that hold the active config by reference
        # (e.g. serve's ``agent_holder["config"]`` and its workspace-changing
        # ``/resume`` reload) must observe the new model/provider. The verify
        # build above used the ``temp_cfg`` copy, so a failed build never reaches
        # here and the live ``cfg`` stays untouched (failure still no-ops).
        cfg.model = model_name
        cfg.provider = provider
        set_active_config(cfg)
        set_chat_model_instance(new_chat_model, (model_name, provider))
        ctx.agent = new_agent

        # Persist to config file if --save was given
        if save:
            from ...config.settings import set_config_value

            set_config_value("model", model_name)
            set_config_value("provider", provider)

        # Propagate to the channel runtime if channels are running so the
        # bus picks up the new agent on the next inbound message.
        if ctx.channel_runtime is not None and ctx.channel_runtime.agent is not None:
            ctx.channel_runtime.agent = new_agent

        # Update status bar if available
        update_model_fn = getattr(ctx.ui, "update_status_after_model_change", None)
        if callable(update_model_fn):
            update_model_fn(model_name, provider)

        saved_note = " (saved to config)" if save else ""
        ctx.ui.append_system(
            f"Switched to {model_name} ({provider}){saved_note}", style="green"
        )


manager.register(ModelCommand())
