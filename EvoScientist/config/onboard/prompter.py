"""UI-agnostic prompt abstraction for the onboarding wizard.

Each step function takes an optional ``prompter: Prompter | None``. When None,
it falls back to ``QuestionaryPrompter()``, which preserves the current
interactive UX. The non-interactive code path (``NonInteractivePrompter``)
returns pre-filled answers keyed by ``prompt_id``, so ``EvoSci onboard
--provider anthropic --workspace-mode daemon`` can skip those prompts.

The split mirrors openclaw's ``WizardPrompter`` interface in
``meals/openclaw/src/wizard/prompts.ts``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import questionary
from prompt_toolkit.styles import Style

from .style import QMARK, WIZARD_STYLE


# Used by the provider/auth sub-loop in wizard.py. Mirrors openclaw's
# ``BACK_VALUE = "__back"`` (see
# meals/openclaw/src/commands/auth-choice-prompt.ts).
class GoBack(Exception):
    """Raised inside the provider sub-loop to rewind to provider selection."""


# Sentinel value the back-keybinding writes into the prompt result, and the
# value the trailing ``← Back`` menu item carries. Same string so the two
# code paths converge to a single ``GoBack`` raise.
BACK_SENTINEL = "__back__"


def install_navigation_keys(
    question,
    *,
    with_back: bool = False,
    sentinel: str = BACK_SENTINEL,
) -> None:
    """Add keyboard shortcuts on a questionary select ``Question``.

    Bindings (merged in front of questionary's defaults — Ctrl+C/Ctrl+D still
    cancel the wizard):

    - ``→`` — accept the option under the cursor and advance (mirrors Enter).
    - ``Esc`` / ``←`` (only when ``with_back=True``) — exit with ``sentinel``
      so the wizard can rewind. Used in the provider sub-loop's auth_mode
      prompts.
    """
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

    kb = KeyBindings()

    @kb.add("right", eager=True)
    def _confirm(event):
        # Locate questionary's InquirerControl in the layout and exit with
        # the value the cursor currently points at.
        try:
            from questionary.prompts.common import InquirerControl
        except ImportError:  # pragma: no cover — defensive
            return
        for window in event.app.layout.find_all_windows():
            ctrl = getattr(window, "content", None)
            if isinstance(ctrl, InquirerControl):
                pointed = ctrl.get_pointed_at()
                ctrl.is_answered = True
                event.app.exit(result=pointed.value)
                return

    if with_back:

        @kb.add("escape", eager=True)
        @kb.add("left", eager=True)
        def _back(event):
            event.app.exit(result=sentinel)

    question.application.key_bindings = merge_key_bindings(
        [kb, question.application.key_bindings]
    )


from contextlib import contextmanager  # noqa: E402 — colocated with usage below


@contextmanager
def select_navigation_active():
    """Wrap every ``questionary.select`` call inside the block.

    Returned ``Question`` objects get ``→ = confirm current selection``
    bound automatically. Used in ``run_onboard`` so all wizard select
    prompts share the same arrow-key UX without touching 20 call sites.

    Restores the original ``questionary.select`` on exit. Safe under tests
    that patch the local ``questionary`` name in each onboard submodule —
    those patches replace the module reference and bypass our wrapper.
    """
    import questionary

    original = questionary.select

    def _wrapped(*args, **kwargs):
        q = original(*args, **kwargs)
        try:
            install_navigation_keys(q, with_back=False)
        except Exception:
            # Don't let a stray keybinding error block the wizard.
            pass
        return q

    questionary.select = _wrapped
    try:
        yield
    finally:
        questionary.select = original


@runtime_checkable
class Prompter(Protocol):
    """Methods the wizard's step functions invoke.

    ``prompt_id`` is a stable key (``"provider"``, ``"model"``, …) used by
    ``NonInteractivePrompter`` to look up the pre-filled answer.
    """

    def select(
        self,
        prompt_id: str,
        message: str,
        choices: list,
        *,
        default: Any | None = None,
    ) -> Any: ...

    def text(
        self,
        prompt_id: str,
        message: str,
        *,
        default: str = "",
        validate=None,
        placeholder: Any | None = None,
    ) -> str: ...

    def password(
        self,
        prompt_id: str,
        message: str,
        *,
        placeholder: Any | None = None,
    ) -> str: ...

    def confirm(
        self,
        prompt_id: str,
        message: str,
        *,
        default: bool = True,
        style: Style | None = None,
    ) -> bool: ...

    def multiselect(
        self,
        prompt_id: str,
        message: str,
        choices: list,
    ) -> list: ...


class QuestionaryPrompter:
    """Default implementation — delegates to ``questionary`` with EvoSci styling."""

    def select(self, prompt_id, message, choices, *, default=None):
        result = questionary.select(
            message,
            choices=choices,
            default=default,
            style=WIZARD_STYLE,
            qmark=QMARK,
            use_indicator=True,
        ).ask()
        if result is None:
            raise KeyboardInterrupt()
        return result

    def text(self, prompt_id, message, *, default="", validate=None, placeholder=None):
        kwargs: dict = {"style": WIZARD_STYLE, "qmark": QMARK}
        if default:
            kwargs["default"] = default
        if validate is not None:
            kwargs["validate"] = validate
        if placeholder is not None:
            kwargs["placeholder"] = placeholder
        result = questionary.text(message, **kwargs).ask()
        if result is None:
            raise KeyboardInterrupt()
        return result

    def password(self, prompt_id, message, *, placeholder=None):
        kwargs: dict = {"style": WIZARD_STYLE, "qmark": QMARK}
        if placeholder is not None:
            kwargs["placeholder"] = placeholder
        result = questionary.password(message, **kwargs).ask()
        if result is None:
            raise KeyboardInterrupt()
        return result

    def confirm(self, prompt_id, message, *, default=True, style=None):
        result = questionary.confirm(
            message,
            default=default,
            style=style or WIZARD_STYLE,
            qmark=QMARK,
        ).ask()
        if result is None:
            raise KeyboardInterrupt()
        return result

    def multiselect(self, prompt_id, message, choices):
        # Reuse the existing ✓-rendering helper for visual consistency with
        # already-installed items.
        from .style import _checkbox_ask

        result = _checkbox_ask(choices, message)
        if result is None:
            raise KeyboardInterrupt()
        return result


class NonInteractivePrompter:
    """Pre-filled answers keyed by ``prompt_id``.

    Used by ``EvoSci onboard --provider X --model Y``. If the wizard asks for
    a prompt whose id is not in ``answers``, it raises — callers should preset
    every required prompt before invoking.

    ``skip_set`` marks sections that should be SKIPPED entirely. Step functions
    check this before any prompt: if their section is in ``skip_set`` they
    short-circuit without calling the prompter.
    """

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        skip_set: set[str] | None = None,
        strict: bool = False,
    ):
        """
        Args:
            answers: pre-filled prompt answers keyed by ``prompt_id``.
            skip_set: section ids to skip entirely.
            strict: when True, the wizard treats this as "must run without any
                interactive prompt" — optional sections (skills/mcp/latex/
                channels) and provider sub-prompts (base_url / auth_mode)
                that have no preset are auto-skipped instead of falling back
                to interactive. Set this through the constructor; do not
                monkey-set the attribute on an existing instance.
        """
        self.answers: dict[str, Any] = dict(answers or {})
        self.skip_set: set[str] = set(skip_set or ())
        self.strict: bool = bool(strict)

    def has(self, prompt_id: str) -> bool:
        return prompt_id in self.answers

    def should_skip(self, section: str) -> bool:
        return section in self.skip_set

    def _get(self, prompt_id: str, message: str):
        if prompt_id not in self.answers:
            flag = "--" + prompt_id.replace("_", "-")
            raise RuntimeError(
                f"Non-interactive mode: missing answer for {prompt_id!r} "
                f"(would have asked: {message!r}). Pass {flag} on the command "
                "line, or drop --non-interactive."
            )
        return self.answers[prompt_id]

    def select(self, prompt_id, message, choices, *, default=None):
        return self._get(prompt_id, message)

    def text(self, prompt_id, message, *, default="", validate=None, placeholder=None):
        value = self._get(prompt_id, message)
        resolved = value if value != "" else default
        # Validate even when ``resolved`` is empty — a strict validator
        # (e.g. ``ChoiceValidator(allow_empty=False)``) needs the chance to
        # reject blank input, matching what the interactive prompt would do.
        # Validators that tolerate empty (``IntegerValidator``,
        # ``ChoiceValidator`` with the default ``allow_empty=True``) are
        # unaffected.
        if validate is not None:
            # ``validate`` follows one of two prompt_toolkit-compatible forms:
            # (a) a ``Validator`` subclass instance — has ``.validate(document)``
            #     and raises ``ValidationError`` on failure;
            # (b) a plain callable taking the string and returning ``True`` /
            #     ``False`` / a string error message.
            # ``QuestionaryPrompter`` passes either form straight to
            # questionary, which handles both; we must do the same here so the
            # non-interactive path doesn't silently accept invalid presets.
            from prompt_toolkit.document import Document
            from prompt_toolkit.validation import ValidationError, Validator

            def _reject(msg: str) -> None:
                raise RuntimeError(
                    f"Non-interactive mode: preset value for {prompt_id!r} "
                    f"({resolved!r}) rejected by validator: {msg}"
                )

            if isinstance(validate, Validator):
                try:
                    validate.validate(Document(text=str(resolved)))
                except ValidationError as exc:
                    _reject(exc.message or "invalid value")
            elif callable(validate):
                try:
                    result = validate(resolved)
                except ValidationError as exc:
                    _reject(exc.message or "invalid value")
                else:
                    if result is False or isinstance(result, str):
                        _reject(result if isinstance(result, str) else "invalid value")
            # Any other type (None ruled out above) — silently ignore: the
            # interactive path would also have nothing reasonable to do with it.
        return resolved

    def password(self, prompt_id, message, *, placeholder=None):
        return self._get(prompt_id, message)

    def confirm(self, prompt_id, message, *, default=True, style=None):
        return self._get(prompt_id, message)

    def multiselect(self, prompt_id, message, choices):
        value = self._get(prompt_id, message)
        return list(value) if value is not None else []


__all__ = [
    "NonInteractivePrompter",
    "Prompter",
    "QuestionaryPrompter",
]
