"""Shared UI helpers for skill-management commands (picker used by /evoskills)."""

from ..stream.console import console


def _pick_skills_interactive(
    index: list[dict],
    installed_names: set[str],
    pre_filter_tag: str,
) -> list[str] | None:
    """Interactive questionary picker for EvoSkills browse.

    Two-phase picker:
    1. tag filter — ``questionary.select`` (skipped if ``pre_filter_tag``)
    2. multi-select — ``questionary.checkbox`` with installed items disabled

    Returns:
        list of ``install_source`` strings selected by the user,
        ``None`` if the user cancelled at either phase, or
        ``[]`` if nothing was selectable / all-installed in the filter.
    """
    from collections import Counter

    import questionary
    from questionary import Choice

    from .widgets.thread_selector import PICKER_STYLE

    # Installed-item indicator style for disabled checkbox choices.
    _INSTALLED_INDICATOR = ("fg:#4caf50", "✓ ")

    def _checkbox_ask(choices, message: str, **kwargs):
        """questionary.checkbox that renders disabled items with checkmark."""
        from questionary.prompts.common import InquirerControl

        original = InquirerControl._get_choice_tokens

        def _patched(self):
            tokens = original(self)
            return [
                _INSTALLED_INDICATOR
                if cls == "class:disabled" and text == "- "
                else (cls, text)
                for cls, text in tokens
            ]

        InquirerControl._get_choice_tokens = _patched
        try:
            return questionary.checkbox(
                message,
                choices=choices,
                style=PICKER_STYLE,
                qmark="❯",
                **kwargs,
            ).ask()
        finally:
            InquirerControl._get_choice_tokens = original

    pre_filter_tag = (pre_filter_tag or "").strip().lower()

    # Phase 1: tag filter (skip if pre-filtered via args)
    if pre_filter_tag:
        filtered = [
            s for s in index if pre_filter_tag in [t.lower() for t in s.get("tags", [])]
        ]
        if not filtered:
            console.print(
                f"[yellow]No skills found with tag: {pre_filter_tag}[/yellow]"
            )
            tag_counter: Counter[str] = Counter()
            for s in index:
                for t in s.get("tags", []):
                    tag_counter[t.lower()] += 1
            if tag_counter:
                sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
                tags_str = ", ".join(f"{tag} ({count})" for tag, count in sorted_tags)
                console.print(f"[dim]Available tags: {tags_str}[/dim]")
            return []
    else:
        tag_counter = Counter()
        for s in index:
            for t in s.get("tags", []):
                tag_counter[t.lower()] += 1

        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        tag_choices = [Choice(title=f"All skills ({len(index)})", value="__all__")]
        for tag, count in sorted_tags:
            tag_choices.append(Choice(title=f"{tag} ({count})", value=tag))

        selected_tag = questionary.select(
            "Filter by tag:",
            choices=tag_choices,
            style=PICKER_STYLE,
            qmark="❯",
        ).ask()

        if selected_tag is None:
            return None

        if selected_tag == "__all__":
            filtered = index
        else:
            filtered = [
                s
                for s in index
                if selected_tag in [t.lower() for t in s.get("tags", [])]
            ]

    # Phase 2: skill selection checkbox
    if all(s["name"] in installed_names for s in filtered):
        console.print(
            "[green]All skills in this category are already installed.[/green]"
        )
        return []

    choices = []
    for s in filtered:
        if s["name"] in installed_names:
            choices.append(
                Choice(
                    title=[
                        ("", f"{s['name']} — {s['description'][:80]}"),
                        ("class:instruction", "  (installed)"),
                    ],
                    value=s["install_source"],
                    disabled=True,
                )
            )
        else:
            choices.append(
                Choice(
                    title=f"{s['name']} — {s['description'][:80]}",
                    value=s["install_source"],
                )
            )

    selected = _checkbox_ask(choices, "Select skills to install:")

    if selected is None:
        return None
    return list(selected)
