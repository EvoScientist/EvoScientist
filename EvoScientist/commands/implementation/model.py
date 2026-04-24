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
        ValueError: If the model is not in the registry.
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
        from ...llm.models import list_models_by_provider

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

        entries = list_models_by_provider()

        # Ollama models are locally-installed — probe the daemon for the list
        # the user has actually pulled. Gated on ollama_base_url being set
        # (issue non-goal forbids implicit localhost detection).
        ollama_base_url = getattr(cfg, "ollama_base_url", None)
        if ollama_base_url:
            from ...llm.ollama_discovery import discover_ollama_models

            detected = await discover_ollama_models(ollama_base_url, timeout=1.5)
            for detected_name in detected:
                entries.append((detected_name, detected_name, "ollama"))
            # Always append the sentinel so users can type a name even when
            # the daemon is down or no models have been pulled yet. The widget
            # swaps the sentinel name for the typed value before posting Picked.
            entries.append(("Custom Ollama model...", "__custom_ollama__", "ollama"))

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

        from ... import EvoScientist as _mod
        from ...cli.agent import _load_agent
        from ...EvoScientist import _ensure_config, set_chat_model

        cfg = _ensure_config()

        # Build a temporary config to verify the agent can be created
        # before mutating any global state.
        temp_cfg = copy.copy(cfg)
        temp_cfg.model = model_name
        temp_cfg.provider = provider

        # create_cli_agent(config=temp_cfg) calls _ensure_config and
        # _ensure_chat_model before finishing, so a failure further
        # down (middleware build, MCP reconnect, deepagents wiring)
        # would leave the session pointing at the new model. Snapshot
        # the four globals those helpers write so we can restore on
        # error. Best-effort: references already captured by concurrent
        # readers (e.g. a channel thread mid-turn) are not retroactively
        # patched, but /model is user-initiated from an idle prompt in
        # practice.
        snap = (
            _mod._config,
            _mod._chat_model,
            _mod._chat_model_key,
            _mod._EvoScientist_agent,
        )

        try:
            new_agent = _load_agent(
                workspace_dir=ctx.workspace_dir,
                checkpointer=ctx.checkpointer,
                config=temp_cfg,
            )
        except Exception as e:
            (
                _mod._config,
                _mod._chat_model,
                _mod._chat_model_key,
                _mod._EvoScientist_agent,
            ) = snap
            ctx.ui.append_system(f"Failed to switch model: {e}", style="red")
            return

        # Agent built successfully — now commit the change globally.
        try:
            set_chat_model(model_name, provider=provider)
        except Exception as e:
            ctx.ui.append_system(f"Failed to switch model: {e}", style="red")
            return

        cfg.model = model_name
        cfg.provider = provider
        ctx.agent = new_agent

        # Persist to config file if --save was given
        if save:
            from ...config.settings import set_config_value

            set_config_value("model", model_name)
            set_config_value("provider", provider)

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
            update_model_fn(model_name, provider)

        saved_note = " (saved to config)" if save else ""
        ctx.ui.append_system(
            f"Switched to {model_name} ({provider}){saved_note}", style="green"
        )


manager.register(ModelCommand())
