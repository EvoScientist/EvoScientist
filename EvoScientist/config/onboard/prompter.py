"""Non-interactive preset container + questionary navigation helpers."""

from __future__ import annotations

from typing import Any


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
        try:
            from questionary.prompts.common import InquirerControl
        except ImportError:  # pragma: no cover
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


from contextlib import contextmanager  # noqa: E402


@contextmanager
def select_navigation_active():
    """Bind ``→`` to confirm on every ``questionary.select`` in the block.

    ``checkbox`` is NOT wrapped: the shared ``→`` handler returns a single
    value, but checkbox must return a list of checked items.
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


class NonInteractivePrompter:
    """Container for CLI-supplied wizard answers (``--provider``, ``--model``…).

    ``strict=True`` makes missing presets fatal instead of falling back to
    interactive. Wizard reads ``answers`` / ``skip_set`` / ``strict`` directly.
    """

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        skip_set: set[str] | None = None,
        strict: bool = False,
    ):
        self.answers: dict[str, Any] = dict(answers or {})
        self.skip_set: set[str] = set(skip_set or ())
        self.strict: bool = bool(strict)

    def has(self, prompt_id: str) -> bool:
        return prompt_id in self.answers


__all__ = [
    "BACK_SENTINEL",
    "GoBack",
    "NonInteractivePrompter",
    "install_navigation_keys",
    "select_navigation_active",
]
