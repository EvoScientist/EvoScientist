"""Cross-step helpers: API key prompt loop, ccproxy login, npx/node bootstrapping,
LaTeX detection/install, iMessage setup.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import questionary

from ..settings import EvoScientistConfig
from .style import QMARK, WIZARD_STYLE, console
from .validators import (
    validate_anthropic_key,
    validate_dashscope_code_key,
    validate_dashscope_key,
    validate_deepseek_key,
    validate_google_key,
    validate_kimi_key,
    validate_minimax_key,
    validate_moonshot_key,
    validate_nvidia_key,
    validate_openai_key,
    validate_openrouter_key,
    validate_siliconflow_key,
    validate_volcengine_key,
    validate_zhipu_key,
)


def _provider_key_info(config: EvoScientistConfig, provider: str):
    """Return (display_name, current_value, validate_fn) for a provider."""
    mapping = {
        "anthropic": (
            "Anthropic",
            config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            validate_anthropic_key,
        ),
        "minimax": (
            "MiniMax",
            config.minimax_api_key or os.environ.get("MINIMAX_API_KEY", ""),
            lambda key: validate_minimax_key(
                key,
                base_url=config.minimax_base_url
                or os.environ.get(
                    "MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"
                ),
            ),
        ),
        "nvidia": (
            "NVIDIA",
            config.nvidia_api_key or os.environ.get("NVIDIA_API_KEY", ""),
            validate_nvidia_key,
        ),
        "google-genai": (
            "Google",
            config.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
            validate_google_key,
        ),
        "siliconflow": (
            "SiliconFlow",
            config.siliconflow_api_key or os.environ.get("SILICONFLOW_API_KEY", ""),
            validate_siliconflow_key,
        ),
        "openrouter": (
            "OpenRouter",
            config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            validate_openrouter_key,
        ),
        "deepseek": (
            "DeepSeek",
            config.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            validate_deepseek_key,
        ),
        "zhipu": (
            "ZhipuAI",
            config.zhipu_api_key or os.environ.get("ZHIPU_API_KEY", ""),
            validate_zhipu_key,
        ),
        "zhipu-code": (
            "ZhipuAI CodePlan",
            config.zhipu_api_key or os.environ.get("ZHIPU_API_KEY", ""),
            validate_zhipu_key,
        ),
        "volcengine": (
            "Volcengine",
            config.volcengine_api_key or os.environ.get("VOLCENGINE_API_KEY", ""),
            validate_volcengine_key,
        ),
        "dashscope": (
            "DashScope",
            config.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            validate_dashscope_key,
        ),
        "dashscope-code": (
            "DashScope Coding Plan",
            config.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            validate_dashscope_code_key,
        ),
        "moonshot": (
            "Moonshot",
            config.moonshot_api_key or os.environ.get("MOONSHOT_API_KEY", ""),
            validate_moonshot_key,
        ),
        "kimi-coding": (
            "Kimi Coding Plan",
            config.kimi_api_key or os.environ.get("KIMI_API_KEY", ""),
            validate_kimi_key,
        ),
        "custom-openai": (
            "OpenAI-compatible",
            config.custom_openai_api_key or os.environ.get("CUSTOM_OPENAI_API_KEY", ""),
            None,
        ),
        "custom-anthropic": (
            "Custom Anthropic",
            config.custom_anthropic_api_key
            or os.environ.get("CUSTOM_ANTHROPIC_API_KEY", ""),
            None,
        ),
        "ollama": ("Ollama", "__no_key__", None),
    }
    return mapping.get(
        provider,
        (
            "OpenAI",
            config.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            validate_openai_key,
        ),
    )


def _prompt_and_validate_api_key(
    prompt_text: str,
    current: str,
    validate_fn,
    skip_validation: bool = False,
    placeholder=None,
) -> str | None:
    """Prompt user for an API key, validate, offer save-anyway on failure.

    Args:
        prompt_text: The question shown to the user.
        current: Currently stored key value (may be empty).
        validate_fn: Callable(key) -> (bool, str).
        skip_validation: If True, skip the validation step entirely.
        placeholder: Optional placeholder for the password input.

    Returns:
        New key string if the user entered one, or None to keep existing.
    """
    kwargs: dict = {"style": WIZARD_STYLE, "qmark": QMARK}
    if placeholder is not None:
        kwargs["placeholder"] = placeholder

    new_key = questionary.password(prompt_text, **kwargs).ask()
    if new_key is None:
        raise KeyboardInterrupt()

    new_key = new_key.strip()

    # Determine which key to validate: new input or existing
    key_to_validate = new_key or current

    if not key_to_validate:
        return None

    if not skip_validation and validate_fn is not None:
        console.print("  [dim]Validating...[/dim]", end="")
        valid, msg = validate_fn(key_to_validate)
        if valid:
            console.print(f"\r  [green]\u2713 {msg}[/green]      ")
            return new_key or None
        else:
            console.print(f"\r  [red]\u2717 {msg}[/red]      ")
            if not new_key:
                # Existing key is invalid — warn but keep (user didn't change it)
                return None
            save_anyway = questionary.confirm(
                "Save anyway?",
                default=False,
                style=WIZARD_STYLE,
                qmark=QMARK,
            ).ask()
            if save_anyway is None:
                raise KeyboardInterrupt()
            return new_key if save_anyway else None

    return new_key or None


def _prompt_ccproxy_port(config: EvoScientistConfig) -> None:
    """Prompt the user for a ccproxy port and save it to config."""

    def valid_port(value: str) -> bool:
        if not value:  # empty = keep default
            return True
        try:
            return 0 < int(value) < 2**16
        except (ValueError, TypeError):
            return False

    current_port = getattr(config, "ccproxy_port", 8000)
    try:
        raw = questionary.text(
            f"Enter port number for ccproxy to run on (Current: {current_port}, Enter to keep):",
            validate=valid_port,
            style=WIZARD_STYLE,
            qmark=QMARK,
        ).ask()
        if raw is None:
            raise KeyboardInterrupt()
        raw = raw.strip()
        ccproxy_port = int(raw) if raw else current_port
    except (ValueError, TypeError):
        ccproxy_port = current_port
        console.print(f"  [dim]Using default port: {ccproxy_port}[/dim]")

    config.ccproxy_port = ccproxy_port
    console.print(
        f"  [green]✓ ccproxy will run on http://127.0.0.1:{ccproxy_port}[/green]"
    )


def _run_ccproxy_login(provider: str, label: str) -> None:
    """Run ccproxy auth login for the given provider and show status."""
    from ...ccproxy_manager import _ccproxy_exe, check_ccproxy_auth

    console.print("  [dim]Opening browser for authentication...[/dim]")
    try:
        proc = subprocess.run(
            [_ccproxy_exe() or "ccproxy", "auth", "login", provider],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in proc.stdout.splitlines():
            if line.strip().startswith("https://"):
                console.print(f"  [dim]Visit: {line.strip()}[/dim]")
                break
        authed, msg = check_ccproxy_auth(provider)
        if authed:
            console.print(f"  [green]✓ {label}: {msg}[/green]")
        else:
            console.print(f"  [red]Authentication failed: {msg}[/red]")
    except subprocess.TimeoutExpired:
        console.print("  [red]Login timed out.[/red]")
    except Exception as exc:
        console.print(f"  [red]Login error: {exc}[/red]")


def _check_npx() -> bool:
    """Check if npx is available on the system.

    Uses shutil.which() to resolve the executable path, which correctly
    finds .cmd/.bat wrappers on Windows (e.g., npx.cmd).

    Returns:
        True if npx is found and working.
    """
    npx = shutil.which("npx")
    if not npx:
        return False
    try:
        result = subprocess.run(
            [npx, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_node_install_method() -> tuple[str, str]:
    """Detect the best way to install Node.js for this environment.

    Returns:
        Tuple of (method_name, install_command).
    """
    # Conda environment (any platform)
    if os.environ.get("CONDA_PREFIX"):
        return "conda", "conda install -y nodejs"

    # macOS with Homebrew
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["brew", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "brew", "brew install node"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Windows: winget (built-in on Win 10+) or chocolatey
    if sys.platform == "win32":
        if shutil.which("winget"):
            return "winget", "winget install OpenJS.NodeJS.LTS"
        if shutil.which("choco"):
            return "choco", "choco install nodejs-lts -y"

    return "manual", "https://nodejs.org"


def _install_node(method: str, command: str) -> bool:
    """Install Node.js using the detected method.

    Returns:
        True if installation succeeded.
    """
    if method == "manual":
        return False

    parts = command.split()
    exe = shutil.which(parts[0]) or parts[0]
    try:
        proc = subprocess.run(
            [exe, *parts[1:]],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        console.print(f"  [red]✗ {method} not found[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("  [red]✗ Installation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]✗ Installation failed: {e}[/red]")
        return False


def _ensure_npx(reason: str) -> bool:
    """Check for npx and offer to install Node.js if missing.

    Args:
        reason: Why npx is needed (shown in the warning message).

    Returns:
        True if npx is available (was already present or just installed).
    """
    if _check_npx():
        return True

    console.print(f"  [yellow]✗ npx not found — {reason}[/yellow]")
    method, command = _detect_node_install_method()

    if method != "manual":
        install_node = questionary.confirm(
            f"Install Node.js via {method}? ({command})",
            default=True,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
        ).ask()
        if install_node is None:
            raise KeyboardInterrupt()
        if install_node:
            console.print("  [dim]Installing Node.js...[/dim]")
            if _install_node(method, command):
                if _check_npx():
                    console.print("  [green]✓ npx now available[/green]")
                    return True
                else:
                    console.print(
                        "  [yellow]✗ npx still not found after install[/yellow]"
                    )
            else:
                console.print("  [red]✗ Installation failed[/red]")
    else:
        console.print(f"  [dim]Install Node.js: {command}[/dim]")

    return False


# =============================================================================
# TinyTeX (LaTeX) helpers
# =============================================================================


def _check_latex_components() -> dict[str, bool]:
    """Check which LaTeX components are available.

    Returns:
        Dict mapping component name to availability:
        ``{"pdflatex": bool, "latexmk": bool, "tlmgr": bool}``.
    """
    result: dict[str, bool] = {}
    for cmd in ("pdflatex", "latexmk", "tlmgr"):
        exe = shutil.which(cmd)
        if not exe:
            result[cmd] = False
            continue
        try:
            proc = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            result[cmd] = proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result[cmd] = False
    return result


def _check_tinytex() -> bool:
    """Check if a usable LaTeX distribution is available.

    Returns:
        True if pdflatex is found and working.
    """
    return _check_latex_components().get("pdflatex", False)


def _detect_tinytex_install_method() -> tuple[str, str]:
    """Detect the best way to install TinyTeX for this platform.

    Returns:
        Tuple of (method_name, install_command_or_url).
    """
    if sys.platform == "win32":
        if shutil.which("choco"):
            return "choco", "choco install tinytex -y"
        if shutil.which("scoop"):
            return "scoop", "scoop install tinytex"
        return "manual", "https://yihui.org/tinytex/"

    # macOS and Linux: use the official install script
    if shutil.which("curl"):
        return (
            "curl",
            'curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh',
        )
    if shutil.which("wget"):
        return (
            "wget",
            'wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh',
        )

    return "manual", "https://yihui.org/tinytex/"


def _install_tinytex(method: str, command: str) -> bool:
    """Install TinyTeX using the detected method.

    Returns:
        True if installation succeeded.
    """
    if method == "manual":
        return False

    if method in ("curl", "wget"):
        # Pipe-to-shell commands must run through the shell
        try:
            proc = subprocess.run(
                command,
                shell=True,  # user confirmed install in wizard
                capture_output=True,
                text=True,
                timeout=300,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            console.print("  [red]✗ Installation timed out[/red]")
            return False
        except Exception as e:
            console.print(f"  [red]✗ Installation failed: {e}[/red]")
            return False

    # choco / scoop
    parts = command.split()
    exe = shutil.which(parts[0]) or parts[0]
    try:
        proc = subprocess.run(
            [exe, *parts[1:]],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        console.print(f"  [red]✗ {method} not found[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("  [red]✗ Installation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]✗ Installation failed: {e}[/red]")
        return False


def _print_latex_status(components: dict[str, bool]) -> None:
    """Print a single-line status showing all LaTeX components."""
    parts: list[str] = []
    for cmd, _role in (
        ("pdflatex", "compiler"),
        ("latexmk", "build tool"),
        ("tlmgr", "package manager"),
    ):
        if components.get(cmd, False):
            parts.append(f"[green]✓ {cmd}[/green]")
        else:
            parts.append(f"[yellow]✗ {cmd}[/yellow]")
    console.print("  " + "  ".join(parts))


def _auto_install_latexmk() -> None:
    """Auto-install latexmk via tlmgr when it is missing."""
    console.print("  [dim]Installing latexmk via tlmgr...[/dim]")
    tlmgr = shutil.which("tlmgr")
    if not tlmgr:
        return
    try:
        proc = subprocess.run(
            [tlmgr, "install", "latexmk"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and shutil.which("latexmk"):
            console.print("  [green]✓ latexmk installed[/green]")
        else:
            console.print(
                "  [yellow]⚠ Failed to install latexmk"
                " (run: tlmgr install latexmk)[/yellow]"
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print(
            "  [yellow]⚠ Failed to install latexmk"
            " (run: tlmgr install latexmk)[/yellow]"
        )


def validate_imessage() -> tuple[bool, str]:
    """Validate iMessage environment by checking for the imsg CLI.

    Returns:
        Tuple of (is_valid, message).
    """
    # macOS only
    if sys.platform != "darwin":
        return False, "iMessage requires macOS"

    from ...channels.imessage.probe import find_cli

    cli_path = find_cli()
    if not cli_path:
        return False, "not_installed"

    # Check version
    try:
        result = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        version = None

    # Check RPC support
    try:
        result = subprocess.run(
            [cli_path, "rpc", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rpc_ok = result.returncode == 0
    except Exception:
        rpc_ok = False

    if not rpc_ok:
        return (
            False,
            f"imsg found at {cli_path} but RPC not supported (update with: brew upgrade imsg)",
        )

    version_str = f" ({version})" if version else ""
    return True, f"imsg{version_str} at {cli_path}"


def _install_ccproxy() -> bool:
    """Run pip install for ccproxy (evoscientist[oauth]).

    Uses uv pip install when available (uv-managed envs don't ship pip).

    Returns:
        True if installation succeeded and ccproxy is available.
    """
    from ...ccproxy_manager import is_ccproxy_available
    from ...mcp.registry import install_library

    ok = install_library("evoscientist[oauth]")
    if not ok:
        console.print("  [red]✗ Installation failed.[/red]")
        return False
    return is_ccproxy_available()


def _install_imsg() -> bool:
    """Run brew install for imsg CLI.

    Returns:
        True if installation succeeded.
    """
    try:
        proc = subprocess.run(
            ["brew", "install", "steipete/tap/imsg"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        console.print("  [red]✗ Homebrew not found[/red]")
        console.print("  [dim]Install Homebrew first: https://brew.sh[/dim]")
        return False
    except subprocess.TimeoutExpired:
        console.print("  [red]✗ Installation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]✗ Installation failed: {e}[/red]")
        return False


def _setup_imessage() -> bool:
    """Guide the user through iMessage setup: install, validate, test.

    Returns:
        True if iMessage is ready to use.
    """
    # Step 1: Validate
    console.print("  [dim]Checking iMessage environment...[/dim]")
    valid, msg = validate_imessage()

    if valid:
        console.print(f"  [green]✓ {msg}[/green]")
        return True

    if msg == "iMessage requires macOS":
        console.print(f"  [red]✗ {msg}[/red]")
        return False

    if msg == "not_installed":
        console.print("  [yellow]✗ imsg CLI not installed[/yellow]")
        console.print()

        # Step 2: Offer to install
        install = questionary.confirm(
            "Install imsg via Homebrew? (brew install steipete/tap/imsg)",
            default=True,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
        ).ask()

        if install is None:
            raise KeyboardInterrupt()

        if install:
            console.print()
            if _install_imsg():
                console.print()
                # Re-validate after install
                valid, msg = validate_imessage()
                if valid:
                    console.print(f"  [green]✓ {msg}[/green]")
                    return True
                else:
                    console.print(f"  [red]✗ {msg}[/red]")
                    return False
            else:
                return False
        else:
            console.print(
                "  [dim]Skipped. Install manually: brew install steipete/tap/imsg[/dim]"
            )
            return False
    else:
        # RPC not supported or other issue
        console.print(f"  [red]✗ {msg}[/red]")
        return False
