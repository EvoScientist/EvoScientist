"""Channel selection + per-channel configuration.

`_step_channels` is the big one — over 700 lines that walk the user through
selecting which messaging channels to enable and collecting credentials for
each.
"""

from __future__ import annotations

import questionary
from questionary import Choice

from ..settings import EvoScientistConfig
from .helpers import (
    _setup_imessage,
)
from .style import (
    QMARK,
    WIZARD_STYLE,
    console,
)


def _step_channels(config: EvoScientistConfig) -> dict[str, object]:
    """Step: Select channels to enable on startup.

    Presents a multi-select list of supported channels.
    For each selected channel, prompts for required credentials
    and validates them via the channel's probe function.

    Args:
        config: Current configuration.

    Returns:
        Dict mapping config field names to their new values.
        Empty dict when the user skips or selects nothing.
    """
    # Currently enabled channels
    _currently_enabled = {
        t.strip()
        for t in (getattr(config, "channel_enabled", "") or "").split(",")
        if t.strip()
    }
    # Legacy iMessage compat
    if (
        getattr(config, "imessage_enabled", False)
        and "imessage" not in _currently_enabled
    ):
        _currently_enabled.add("imessage")

    # Direct pip packages for each channel extra.  Used to install the
    # exact dependency without requiring the evoscientist package itself
    # to be resolvable on PyPI (e.g. editable / dev installs).
    _CHANNEL_PIP_DEPS: dict[str, list[str]] = {
        "telegram": ["python-telegram-bot>=21.0"],
        "discord": ["discord.py>=2.3"],
        "slack": ["slack-sdk>=3.27", "aiohttp>=3.9"],
        "feishu": ["aiohttp>=3.9", "qrcode>=7.4"],
        "dingtalk": ["aiohttp>=3.9"],
        "wechat": [
            "pycryptodome>=3.20",
            "aiohttp>=3.9",
            "qrcode>=7.4",
            "certifi>=2024.0",
        ],
        "qq": ["qq-botpy>=1.0", "cryptography>=41.0", "qrcode>=7.4"],
    }

    # Channel definitions:
    #   (value, display_name, required_fields, import_check, pip_extra)
    # required_fields entries are (field_name, prompt_label, is_secret).
    # ``is_secret=True`` triggers a password prompt (no echo, no default echo)
    # so bot tokens / OAuth secrets / IMAP+SMTP passwords don't leak into
    # terminal scrollback, screen recordings, or support sessions.
    _CHANNELS = [
        (
            "telegram",
            "Telegram",
            [("telegram_bot_token", "Bot token (from @BotFather)", True)],
            "telegram",
            "telegram",
        ),
        (
            "discord",
            "Discord",
            [("discord_bot_token", "Bot token", True)],
            "discord",
            "discord",
        ),
        (
            "slack",
            "Slack",
            [
                ("slack_bot_token", "Bot token (xoxb-...)", True),
                ("slack_app_token", "App token for Socket Mode (xapp-...)", True),
            ],
            "slack_sdk",
            "slack",
        ),
        (
            "feishu",
            "Feishu",
            [
                ("feishu_app_id", "App ID", False),
                ("feishu_app_secret", "App Secret", True),
            ],
            "aiohttp",
            "feishu",
        ),
        (
            "dingtalk",
            "DingTalk",
            [
                ("dingtalk_client_id", "Client ID (AppKey)", False),
                ("dingtalk_client_secret", "Client Secret (AppSecret)", True),
            ],
            "aiohttp",
            "dingtalk",
        ),
        (
            "wechat",
            "WeChat",
            [],  # backend-specific fields prompted in the wechat branch below
            ("aiohttp", "qrcode", "Crypto", "certifi"),
            "wechat",
        ),
        (
            "email",
            "Email",
            [
                ("email_imap_host", "IMAP host", False),
                ("email_imap_username", "IMAP username", False),
                ("email_imap_password", "IMAP password", True),
                ("email_smtp_host", "SMTP host", False),
                ("email_smtp_username", "SMTP username", False),
                ("email_smtp_password", "SMTP password", True),
                ("email_from_address", "From address", False),
            ],
            None,
            None,
        ),
        (
            "qq",
            "QQ",
            [
                ("qq_app_id", "App ID", False),
                ("qq_app_secret", "App Secret", True),
            ],
            "botpy",
            "qq",
        ),
        (
            "signal",
            "Signal",
            [("signal_phone_number", "Phone number (E.164)", False)],
            None,
            None,
        ),
        ("imessage", "iMessage", [], None, None),  # handled via _setup_imessage()
    ]

    choices = [
        Choice(
            title=display,
            value=value,
            checked=value in _currently_enabled,
        )
        for value, display, *_ in _CHANNELS
    ]

    selected = questionary.checkbox(
        "Select channels to enable (Space to toggle, Enter to confirm):",
        choices=choices,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if selected is None:
        raise KeyboardInterrupt()

    updates: dict[str, object] = {}

    if not selected:
        updates["channel_enabled"] = ""
        updates["imessage_enabled"] = False
        return updates

    from ...mcp.registry import install_library, pip_install_hint

    # Build a lookup for channel definitions
    _ch_lookup = {
        v: (v, d, fields, imp, extra) for v, d, fields, imp, extra in _CHANNELS
    }

    enabled_channels: list[str] = []

    for ch_name in selected:
        _, display, required_fields, import_check, pip_extra = _ch_lookup[ch_name]
        console.print(f"\n  [bold cyan]── {display} ──[/bold cyan]")

        # Check pip dependency before proceeding
        if import_check:
            _required_imports: tuple[str, ...] = (
                (import_check,)
                if isinstance(import_check, str)
                else tuple(import_check)
            )
            _pkg_ready = False
            try:
                for _module_name in _required_imports:
                    __import__(_module_name)
                _pkg_ready = True
            except ImportError:
                console.print("  [yellow]✗ Required package not installed.[/yellow]")
                # Determine packages to install
                _pip_pkgs = _CHANNEL_PIP_DEPS.get(pip_extra, []) if pip_extra else []
                _pkg_display = (
                    " ".join(f'"{p}"' for p in _pip_pkgs)
                    if _pip_pkgs
                    else f'"evoscientist[{pip_extra}]"'
                )
                install_now = questionary.confirm(
                    f"Install {_pkg_display} now?",
                    default=True,
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                ).ask()
                if install_now is None:
                    raise KeyboardInterrupt() from None
                if install_now:
                    console.print(f"  [dim]Installing {_pkg_display}...[/dim]")
                    if _pip_pkgs:
                        _ok = all(install_library(p) for p in _pip_pkgs)
                    else:
                        _ok = install_library(f"evoscientist[{pip_extra}]")
                    if _ok:
                        # Verify the imports actually work now
                        try:
                            for _module_name in _required_imports:
                                __import__(_module_name)
                            console.print("  [green]✓ Installed successfully.[/green]")
                            _pkg_ready = True
                        except ImportError:
                            console.print(
                                "  [red]✗ Package installed but import failed.[/red]"
                            )
                            console.print(
                                "  [dim]Try restarting and running:[/dim] evosci channel setup"
                            )
                    else:
                        console.print("  [red]✗ Installation failed.[/red]")
                        console.print(
                            f"  [dim]Run manually:[/dim] {pip_install_hint()} {_pkg_display}"
                        )
            if not _pkg_ready:
                # Previously-enabled channels are silently dropped from
                # ``channel_enabled`` if we just ``continue`` — warn.
                if ch_name in _currently_enabled:
                    console.print(
                        f"  [bold yellow]⚠ {display} will be DISABLED[/bold yellow]"
                        " [dim](dependency missing — re-run after install)[/dim]"
                    )
                else:
                    console.print(
                        f"  [dim]Skipping {display} — dependency not installed.[/dim]"
                    )
                continue

        # Special handling for iMessage
        if ch_name == "imessage":
            ready = _setup_imessage()
            if not ready:
                console.print()
                enable_anyway = questionary.confirm(
                    "Enable iMessage anyway? (will try to connect on startup)",
                    default=False,
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                ).ask()
                if enable_anyway is None:
                    raise KeyboardInterrupt()
                if not enable_anyway:
                    continue
            # Allowed senders
            senders = questionary.text(
                "Allowed senders (comma-separated, empty = all):",
                default=getattr(config, "imessage_allowed_senders", ""),
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
            ).ask()
            if senders is None:
                raise KeyboardInterrupt()
            updates["imessage_enabled"] = True
            updates["imessage_allowed_senders"] = senders.strip()
            enabled_channels.append("imessage")
            continue

        # QQ: offer scan-to-configure before falling back to manual entry.
        # The bot must already exist at q.qq.com — scanning binds the
        # developer's QQ account to it and returns app_id + client_secret.
        _qq_scanned = False
        _feishu_scanned = False
        if ch_name == "qq":
            scan_choices = [
                Choice(
                    title="Scan QR code  (recommended — auto-fill App ID & Secret)",
                    value="scan",
                ),
                Choice(title="Enter App ID and Secret manually", value="manual"),
            ]
            scan_choice = questionary.select(
                "Configure QQ Bot:",
                choices=scan_choices,
                default="scan",
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
                use_indicator=True,
            ).ask()
            if scan_choice is None:
                raise KeyboardInterrupt()

            if scan_choice == "scan":
                # Preflight: AES-GCM decryption needs `cryptography`.
                # `qrcode` is a soft dep — onboard.py degrades to URL-only display.
                try:
                    import cryptography  # noqa: F401
                except ImportError:
                    console.print(
                        '  [yellow]✗ QR scan requires "cryptography".[/yellow]'
                    )
                    install_now = questionary.confirm(
                        'Install "cryptography" now?',
                        default=True,
                        style=WIZARD_STYLE,
                        qmark=f"  {QMARK}",
                    ).ask()
                    if install_now is None:
                        raise KeyboardInterrupt() from None
                    if install_now and install_library("cryptography>=41.0"):
                        console.print("  [green]✓ Installed cryptography.[/green]")
                    else:
                        console.print(
                            "  [yellow]⚠ Falling back to manual entry.[/yellow]"
                        )
                        scan_choice = "manual"

            if scan_choice == "scan":
                from ...channels.qq.onboard import qr_register

                console.print(
                    "  [dim]Make sure the bot is registered at"
                    " https://q.qq.com first — scanning binds an"
                    " existing app, it does not create one.[/dim]"
                )
                try:
                    creds = qr_register()
                except Exception as exc:
                    console.print(f"  [red]✗ Scan failed: {exc}[/red]")
                    creds = None

                if creds:
                    updates["qq_app_id"] = creds["app_id"]
                    updates["qq_app_secret"] = creds["client_secret"]
                    console.print(
                        f"  [green]✓ Bound QQ Bot (App ID: {creds['app_id']})[/green]"
                    )
                    _qq_scanned = True
                else:
                    console.print(
                        "  [yellow]⚠ Scan did not complete — falling"
                        " back to manual entry.[/yellow]"
                    )

        # Feishu: offer scan-to-create before falling back to manual entry.
        # Unlike QQ, this provisions a brand-new PersonalAgent app with the
        # required IM permissions attached, then returns app_id + app_secret.
        if ch_name == "feishu":
            scan_choices = [
                Choice(
                    title="Scan QR code  (recommended — auto-create app, fill App ID & Secret)",
                    value="scan",
                ),
                Choice(title="Enter App ID and Secret manually", value="manual"),
            ]
            scan_choice = questionary.select(
                "Configure Feishu / Lark:",
                choices=scan_choices,
                default="scan",
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
                use_indicator=True,
            ).ask()
            if scan_choice is None:
                raise KeyboardInterrupt()

            if scan_choice == "scan":
                # `qrcode` is the only soft dep needed — onboard prints the URL
                # if it's missing, but the UX is much worse, so offer to install.
                try:
                    import qrcode  # noqa: F401
                except ImportError:
                    console.print(
                        '  [yellow]✗ QR scan looks best with "qrcode".[/yellow]'
                    )
                    install_now = questionary.confirm(
                        'Install "qrcode" now?',
                        default=True,
                        style=WIZARD_STYLE,
                        qmark=f"  {QMARK}",
                    ).ask()
                    if install_now is None:
                        raise KeyboardInterrupt() from None
                    if install_now and install_library("qrcode>=7.4"):
                        console.print("  [green]✓ Installed qrcode.[/green]")
                    else:
                        console.print(
                            "  [yellow]⚠ Falling back to manual entry.[/yellow]"
                        )
                        scan_choice = "manual"

            if scan_choice == "scan":
                # Region selection — accounts.feishu.cn vs accounts.larksuite.com.
                # The poll endpoint auto-switches if the scanning user is on the
                # other tenant, so this is just a starting hint.
                region_choices = [
                    Choice(title="Feishu (飞书, mainland China)", value="feishu"),
                    Choice(title="Lark (overseas)", value="lark"),
                ]
                region = questionary.select(
                    "Region:",
                    choices=region_choices,
                    default="feishu",
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                    use_indicator=True,
                ).ask()
                if region is None:
                    raise KeyboardInterrupt()

            if scan_choice == "scan":
                from ...channels.feishu.onboard import qr_register

                console.print(
                    "  [dim]A QR code will be printed below — open Feishu or"
                    " Lark on your phone and scan it. The platform will"
                    " auto-create a bot app with IM permissions and return"
                    " the credentials here.[/dim]"
                )
                try:
                    creds = qr_register(initial_domain=region)
                except Exception as exc:
                    console.print(f"  [red]✗ Scan failed: {exc}[/red]")
                    creds = None

                if creds:
                    updates["feishu_app_id"] = creds["app_id"]
                    updates["feishu_app_secret"] = creds["app_secret"]
                    # Sync open-platform domain to the resolved region
                    updates["feishu_domain"] = (
                        "https://open.larksuite.com"
                        if creds.get("domain") == "lark"
                        else "https://open.feishu.cn"
                    )
                    bot_name = creds.get("bot_name")
                    if bot_name:
                        console.print(
                            f'  [green]✓ Bound Feishu bot "{bot_name}"'
                            f" (App ID: {creds['app_id']})[/green]"
                        )
                    else:
                        console.print(
                            f"  [green]✓ Bound Feishu app"
                            f" (App ID: {creds['app_id']})[/green]"
                        )
                    _feishu_scanned = True
                else:
                    console.print(
                        "  [yellow]⚠ Scan did not complete — falling"
                        " back to manual entry.[/yellow]"
                    )

        # WeChat: pick backend (wecom / wechatmp / personal), then prompt
        # backend-specific fields. Personal-WeChat has no static credentials —
        # we offer an interactive QR-scan that obtains and persists them.
        if ch_name == "wechat":
            backend_choices = [
                Choice(
                    title="WeCom (企业微信应用) — most stable, official API",
                    value="wecom",
                ),
                Choice(
                    title="Official Account (微信公众号) — public-facing bots",
                    value="wechatmp",
                ),
                Choice(
                    title="Personal WeChat (个人微信, iLink) — QR-code scan login",
                    value="personal",
                ),
            ]
            wechat_backend = questionary.select(
                "WeChat backend:",
                choices=backend_choices,
                default=getattr(config, "wechat_backend", "") or "wecom",
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
                use_indicator=True,
            ).ask()
            if wechat_backend is None:
                raise KeyboardInterrupt()
            updates["wechat_backend"] = wechat_backend

            # Both WeCom and WeChat MP need the same non-empty-required
            # treatment as the generic required_fields loop below — newly
            # enabling either with blank credentials would leave the channel
            # half-configured and only fail at first message.
            wechat_newly_enabled = "wechat" not in _currently_enabled
            wechat_fields_for_backend: list[tuple[str, str, bool]] = []
            if wechat_backend == "wecom":
                wechat_fields_for_backend = [
                    ("wechat_wecom_corp_id", "WeCom Corp ID", False),
                    ("wechat_wecom_agent_id", "WeCom Agent ID", False),
                    ("wechat_wecom_secret", "WeCom Secret", True),
                ]
            elif wechat_backend == "wechatmp":
                wechat_fields_for_backend = [
                    ("wechat_mp_app_id", "Official Account App ID", False),
                    ("wechat_mp_app_secret", "Official Account App Secret", True),
                ]

            if wechat_backend in ("wecom", "wechatmp"):
                for field_name, prompt_label, is_secret in wechat_fields_for_backend:
                    current = getattr(config, field_name, "")
                    while True:
                        if is_secret:
                            masked_hint = (
                                f" (current: ***{current[-4:]})" if current else ""
                            )
                            value = questionary.password(
                                f"{prompt_label}{masked_hint}:",
                                style=WIZARD_STYLE,
                                qmark=f"  {QMARK}",
                            ).ask()
                        else:
                            value = questionary.text(
                                f"{prompt_label}:",
                                default=current,
                                style=WIZARD_STYLE,
                                qmark=f"  {QMARK}",
                            ).ask()
                        if value is None:
                            raise KeyboardInterrupt()
                        value = value.strip()
                        if not value and current:
                            break  # keep existing
                        if not value and wechat_newly_enabled:
                            console.print(
                                f"  [yellow]{prompt_label} is required to "
                                "enable WeChat. Press Ctrl+C to cancel.[/yellow]"
                            )
                            continue
                        updates[field_name] = value
                        break
            elif wechat_backend == "personal":
                personal_choices = [
                    Choice(
                        title="Scan QR code now (recommended — login to a personal WeChat account)",
                        value="scan",
                    ),
                    Choice(
                        title="I already have an account_id — enter it manually",
                        value="manual",
                    ),
                ]
                personal_choice = questionary.select(
                    "Personal WeChat login:",
                    choices=personal_choices,
                    default="scan",
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                    use_indicator=True,
                ).ask()
                if personal_choice is None:
                    raise KeyboardInterrupt()

                if personal_choice == "scan":
                    from ...channels.wechat.personal import _account_dir as _wp_dir

                    _accounts_path = _wp_dir()
                    console.print(
                        "  [dim]A QR code will be printed below — open WeChat on"
                        " your phone and scan it. The session token is saved"
                        f" to {_accounts_path}.[/dim]"
                    )
                    try:
                        import asyncio

                        from ...channels.wechat.personal import qr_login

                        creds = asyncio.run(qr_login())
                    except Exception as exc:
                        console.print(f"  [red]✗ Scan failed: {exc}[/red]")
                        creds = None

                    if creds:
                        updates["wechat_personal_account_id"] = creds["account_id"]
                        # Token is persisted on disk by qr_login(); the channel
                        # reads it from the per-account store at runtime, so we
                        # intentionally do NOT copy it into the main config here
                        # (avoids stale duplicates and broader secret exposure).
                        console.print(
                            f"  [green]✓ Logged in (account_id: "
                            f"{creds['account_id'][:12]}…)[/green]"
                        )
                    else:
                        console.print(
                            "  [yellow]⚠ QR login did not complete — falling"
                            " back to manual entry.[/yellow]"
                        )
                        personal_choice = "manual"

                if personal_choice == "manual":
                    current_id = getattr(config, "wechat_personal_account_id", "")
                    wechat_newly_enabled = "wechat" not in _currently_enabled
                    while True:
                        account_id = questionary.text(
                            "iLink account_id (from a previous --qr-login run):",
                            default=current_id,
                            style=WIZARD_STYLE,
                            qmark=f"  {QMARK}",
                        ).ask()
                        if account_id is None:
                            raise KeyboardInterrupt()
                        account_id = account_id.strip()
                        if not account_id and current_id:
                            break  # keep existing
                        if not account_id and wechat_newly_enabled:
                            console.print(
                                "  [yellow]account_id is required to enable "
                                "WeChat Personal. Press Ctrl+C to cancel.[/yellow]"
                            )
                            continue
                        updates["wechat_personal_account_id"] = account_id
                        break

        # Prompt for required fields. Secret fields use ``questionary.password``
        # so the entered value (and the existing one shown as a hint) are
        # never echoed to the terminal — see _CHANNELS docstring above.
        if not _qq_scanned and not _feishu_scanned:
            # "Newly enabled" = this channel wasn't in the user's prior
            # ``channel_enabled`` list. Required fields with no existing
            # value must be non-empty for newly enabled channels — saving
            # blanks leaves the channel half-configured and only surfaces
            # the problem on the first message.
            newly_enabled = ch_name not in _currently_enabled
            for field_name, prompt_label, is_secret in required_fields:
                current = getattr(config, field_name, "")
                while True:
                    if is_secret:
                        masked_hint = (
                            f" (current: ***{current[-4:]})" if current else ""
                        )
                        value = questionary.password(
                            f"{prompt_label}{masked_hint}:",
                            style=WIZARD_STYLE,
                            qmark=f"  {QMARK}",
                        ).ask()
                    else:
                        value = questionary.text(
                            f"{prompt_label}:",
                            default=current,
                            style=WIZARD_STYLE,
                            qmark=f"  {QMARK}",
                        ).ask()
                    if value is None:
                        raise KeyboardInterrupt()
                    value = value.strip()

                    # Empty input + existing value → keep existing (this is
                    # the "re-run wizard, no change to this field" path).
                    if not value and current:
                        break
                    # Empty input + newly enabling channel → not OK; the
                    # channel would be enabled with broken creds. Re-prompt.
                    if not value and newly_enabled:
                        console.print(
                            f"  [yellow]{prompt_label} is required to enable "
                            f"{display}. Press Ctrl+C to cancel instead.[/yellow]"
                        )
                        continue
                    # Empty input + previously enabled but never set
                    # (unlikely, but tolerate) → still allow blank-through
                    # so the user isn't blocked re-running configure later.
                    updates[field_name] = value
                    break

        # Feishu: subscription mode + optional fields
        if ch_name == "feishu":
            mode_choices = [
                Choice(
                    title="Webhook (requires public IP / port forwarding)",
                    value="webhook",
                ),
                Choice(
                    title="WebSocket long connection (no public IP needed)",
                    value="websocket",
                ),
            ]
            sub_mode = questionary.select(
                "Subscription mode:",
                choices=mode_choices,
                default="webhook",
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
                use_indicator=True,
            ).ask()
            if sub_mode is None:
                raise KeyboardInterrupt()
            updates["feishu_subscription_mode"] = sub_mode

            if sub_mode == "websocket":
                # WebSocket mode needs lark-oapi SDK
                try:
                    __import__("lark_oapi")
                except ImportError:
                    console.print(
                        '  [yellow]✗ WebSocket mode requires "lark-oapi".[/yellow]'
                    )
                    install_sdk = questionary.confirm(
                        'Install "lark-oapi>=1.4.0" now?',
                        default=True,
                        style=WIZARD_STYLE,
                        qmark=f"  {QMARK}",
                    ).ask()
                    if install_sdk is None:
                        raise KeyboardInterrupt() from None
                    if install_sdk:
                        console.print('  [dim]Installing "lark-oapi"...[/dim]')
                        if install_library("lark-oapi>=1.4.0"):
                            console.print("  [green]✓ Installed successfully.[/green]")
                        else:
                            console.print("  [red]✗ Installation failed.[/red]")
                            console.print(
                                f"  [dim]Run manually:[/dim] {pip_install_hint()} "
                                '"lark-oapi>=1.4.0"'
                            )
            else:
                # Webhook mode: prompt optional verification/encryption fields.
                # Both are credentials — use password() so they don't echo.
                console.print(
                    "  [dim]The following fields are optional"
                    " (press Enter to skip):[/dim]"
                )
                for field_name, prompt_label in [
                    ("feishu_verification_token", "Verification Token (optional)"),
                    ("feishu_encrypt_key", "Encrypt Key (optional)"),
                ]:
                    current = getattr(config, field_name, "")
                    masked_hint = f" (current: ***{current[-4:]})" if current else ""
                    value = questionary.password(
                        f"{prompt_label}{masked_hint}:",
                        style=WIZARD_STYLE,
                        qmark=f"  {QMARK}",
                    ).ask()
                    if value is None:
                        raise KeyboardInterrupt()
                    value = value.strip()
                    if not value and current:
                        # Keep existing value when user just presses Enter.
                        continue
                    updates[field_name] = value

        # Allowed senders (common for all channels)
        senders_field = f"{ch_name}_allowed_senders"
        if hasattr(config, senders_field):
            senders = questionary.text(
                "Allowed senders (comma-separated, empty = all):",
                default=getattr(config, senders_field, ""),
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
            ).ask()
            if senders is None:
                raise KeyboardInterrupt()
            updates[senders_field] = senders.strip()

        # Probe validation
        _probe_channel(ch_name, config, updates)

        enabled_channels.append(ch_name)

    updates["channel_enabled"] = ",".join(enabled_channels)
    # Keep legacy field in sync
    updates["imessage_enabled"] = "imessage" in enabled_channels

    # --- Common prompt: send thinking (shown when any channel is enabled) ---
    if enabled_channels:
        console.print("\n  [bold cyan]── Channel Settings ──[/bold cyan]")
        thinking_choices = [
            Choice(title="On (forward model reasoning)", value=True),
            Choice(title="Off (only send final responses)", value=False),
        ]

        send_thinking = questionary.select(
            "Send thinking panel in channel?",
            choices=thinking_choices,
            default=config.channel_send_thinking,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
            use_indicator=True,
        ).ask()

        if send_thinking is None:
            raise KeyboardInterrupt()

        updates["channel_send_thinking"] = send_thinking

    return updates


def _probe_channel(
    ch_name: str,
    config: EvoScientistConfig,
    updates: dict[str, object],
) -> None:
    """Run the probe for a channel type and print the result.

    Non-fatal: prints a warning on failure but does not prevent enabling.
    """
    import asyncio

    def _val(key: str, fallback: str = "") -> str:
        """Get a value from updates first, then config, then fallback."""
        if key in updates:
            return str(updates[key])
        return str(getattr(config, key, fallback))

    console.print("  [dim]Validating credentials...[/dim]")

    async def _run() -> tuple[bool, str]:
        if ch_name == "telegram":
            from ...channels.telegram.probe import validate_telegram_token

            return await validate_telegram_token(
                _val("telegram_bot_token"),
                _val("telegram_proxy") or None,
            )
        elif ch_name == "discord":
            from ...channels.discord.probe import validate_discord_token

            return await validate_discord_token(
                _val("discord_bot_token"),
                _val("discord_proxy") or None,
            )
        elif ch_name == "slack":
            from ...channels.slack.probe import validate_slack_tokens

            return await validate_slack_tokens(
                _val("slack_bot_token"),
                _val("slack_app_token") or None,
                _val("slack_proxy") or None,
            )
        elif ch_name == "wechat":
            backend = _val("wechat_backend", "wecom")
            if backend == "wechatmp":
                from ...channels.wechat.probe import validate_wechat_mp

                return await validate_wechat_mp(
                    _val("wechat_mp_app_id"),
                    _val("wechat_mp_app_secret"),
                    _val("wechat_proxy") or None,
                )
            elif backend == "personal":
                from ...channels.wechat.probe import validate_wechat_personal

                return await validate_wechat_personal(
                    _val("wechat_personal_account_id"),
                    _val("wechat_personal_token"),
                )
            else:
                from ...channels.wechat.probe import validate_wecom

                return await validate_wecom(
                    _val("wechat_wecom_corp_id"),
                    _val("wechat_wecom_secret"),
                    _val("wechat_proxy") or None,
                )
        elif ch_name == "feishu":
            from ...channels.feishu.probe import validate_feishu_credentials

            return await validate_feishu_credentials(
                _val("feishu_app_id"),
                _val("feishu_app_secret"),
                _val("feishu_domain", "https://open.feishu.cn"),
            )
        elif ch_name == "dingtalk":
            from ...channels.dingtalk.probe import validate_dingtalk

            return await validate_dingtalk(
                _val("dingtalk_client_id"),
                _val("dingtalk_client_secret"),
                _val("dingtalk_proxy") or None,
            )
        elif ch_name == "email":
            from ...channels.email.probe import validate_email_imap

            return await validate_email_imap(
                _val("email_imap_host"),
                int(_val("email_imap_port", "993")),
                _val("email_imap_username"),
                _val("email_imap_password"),
                _val("email_imap_use_ssl", "True").lower() not in ("false", "0", "no"),
            )
        elif ch_name == "qq":
            from ...channels.qq.probe import validate_qq

            return await validate_qq(
                _val("qq_app_id"),
                _val("qq_app_secret"),
            )
        elif ch_name == "signal":
            from ...channels.signal.probe import validate_signal

            return await validate_signal(
                _val("signal_phone_number"),
                _val("signal_cli_path", "signal-cli"),
                int(_val("signal_rpc_port", "7583")),
            )
        else:
            return True, "No probe available"

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio  # type: ignore[import-untyped]

                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        ok, detail = loop.run_until_complete(_run())
        if ok:
            console.print(f"  [green]✓ {detail}[/green]")
        else:
            console.print(f"  [yellow]⚠ {detail}[/yellow]")
            console.print(
                "  [dim]Channel will still be enabled — check credentials later.[/dim]"
            )
    except Exception as e:
        console.print(f"  [yellow]⚠ Could not validate: {e}[/yellow]")
        console.print(
            "  [dim]Channel will still be enabled — check credentials later.[/dim]"
        )


# =============================================================================
# Progress Rendering (for tests and potential future use)
# =============================================================================
