"""First-run setup for the Windows desktop shell.

Shown in the desktop window *before* the backend boots when the active provider
has no usable API key, so a fresh user can pick a provider/model, enter their
key and choose a workspace without hand-editing ``config.yaml``.

GUI-free: the pure pieces here (trigger, provider→field mapping, config write,
validation, form HTML) are unit-tested; the pywebview glue that shows the form
and receives the answer lives in :mod:`EvoScientist.desktop.shell`. The
provider→key-field mapping and the "is this provider configured" check are
reused from the CLI onboarding wizard (via its public accessors) so the GUI and
the terminal wizard never diverge.
"""

from __future__ import annotations

import html
from typing import Any

# Curated common providers for the dropdown (value = config ``provider``). The
# key field for each is resolved through the wizard's public accessor, so any
# provider it knows works even though the dropdown lists the mainstream ones.
PROVIDERS: list[tuple[str, str]] = [
    ("anthropic", "Anthropic (Claude)"),
    ("openai", "OpenAI (GPT)"),
    ("google-genai", "Google (Gemini)"),
    ("openrouter", "OpenRouter"),
    ("deepseek", "DeepSeek"),
    ("nvidia", "NVIDIA"),
]

_KNOWN_PROVIDERS = frozenset(value for value, _ in PROVIDERS)

_STYLE = """
  :root { color-scheme: light dark; }
  body { margin: 0; font: 15px -apple-system, Segoe UI, system-ui, sans-serif;
         display: flex; min-height: 100vh; align-items: center;
         justify-content: center; background: #f6f7f9; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #16181c; color: #e6e6e6; }
    .card { background: #202329; }
    input, select { background: #16181c; color: #e6e6e6; border-color: #3a3f47; }
  }
  .card { width: 460px; max-width: 90vw; padding: 28px 32px; border-radius: 12px;
          background: #fff; box-shadow: 0 2px 24px rgba(0,0,0,.08); }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { margin: 0 0 20px; opacity: .7; font-size: 13px; }
  label { display: block; margin: 14px 0 4px; font-size: 13px; font-weight: 600; }
  input, select { width: 100%; box-sizing: border-box; padding: 8px 10px;
          border: 1px solid #cfd4da; border-radius: 7px; font-size: 14px; }
  .hint { margin: 4px 0 0; font-size: 12px; opacity: .6; }
  button { margin-top: 22px; width: 100%; padding: 10px; border: 0;
           border-radius: 7px; background: #2563eb; color: #fff; font-size: 15px;
           font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .6; cursor: default; }
  .error { margin-top: 14px; padding: 8px 10px; border-radius: 7px;
           background: #fdecec; color: #b42318; font-size: 13px; }
  @media (prefers-color-scheme: dark) { .error { background: #3a1d1d; color: #f7b4b4; } }
"""


def setup_needed(config: Any) -> bool:
    """Whether the first-run setup screen should be shown.

    True when the active provider has no usable key. Reuses the wizard's
    ``is_provider_configured``; since ``get_effective_config`` folds
    ``<PROVIDER>_API_KEY`` env vars into the config fields, a key set in either
    the config file or the environment counts as configured (setup skipped).
    """
    from ..config.onboard.wizard import is_provider_configured

    return not is_provider_configured(config)


def validate_setup(provider: str, model: str, api_key: str) -> str | None:
    """Return an actionable error message if the answers are unusable, else None."""
    if provider not in _KNOWN_PROVIDERS:
        return "Choose a model provider."
    if not model.strip():
        return "Enter a model name."
    if not api_key.strip():
        return "Enter an API key."
    return None


def apply_setup(
    provider: str, model: str, api_key: str, workspace: str, *, config: Any
) -> None:
    """Write the first-run answers onto *config* and persist them.

    Mutates the dataclass (provider, model, the provider's key field, and the
    default workspace) then saves via the shared ``save_config`` (which creates
    the config dir and writes the file 0600). The provider→field mapping comes
    from the wizard so it matches the CLI onboarding exactly.
    """
    from ..config import save_config
    from ..config.onboard.wizard import provider_key_attr

    config.provider = provider
    config.model = model.strip()
    setattr(config, provider_key_attr(provider), api_key.strip())
    config.default_workdir = workspace.strip()
    save_config(config)


def render_setup_html(
    *,
    provider: str = "",
    model: str = "",
    workspace: str = "",
    error: str | None = None,
) -> str:
    """Render the first-run setup form.

    Field values are HTML-escaped; the API-key field is never prefilled. The
    submit button calls ``window.pywebview.api.submit(...)`` and shows any
    validation error inline without a full reload.
    """
    options = []
    for value, text in PROVIDERS:
        selected = " selected" if value == provider else ""
        options.append(
            f'<option value="{html.escape(value)}"{selected}>'
            f"{html.escape(text)}</option>"
        )
    error_block = f'<div class="error">{html.escape(error)}</div>' if error else ""
    return (
        f"<!doctype html><meta charset=utf-8><style>{_STYLE}</style>"
        f"<div class=card>"
        f"<h1>Welcome to EvoScientist</h1>"
        f"<p class=sub>Set up your model provider to get started.</p>"
        f"<label for=provider>Model provider</label>"
        f"<select id=provider>{''.join(options)}</select>"
        f"<label for=model>Model</label>"
        f'<input id=model type=text value="{html.escape(model)}" '
        f"autocomplete=off spellcheck=false>"
        f"<label for=apikey>API key</label>"
        f"<input id=apikey type=password autocomplete=off spellcheck=false>"
        f"<p class=hint>Stored locally in your config file.</p>"
        f"<label for=workspace>Workspace folder</label>"
        f'<input id=workspace type=text value="{html.escape(workspace)}" '
        f'autocomplete=off spellcheck=false placeholder="(current folder)">'
        f"<button id=save onclick=submitSetup()>Save and start</button>"
        f"{error_block}"
        f"</div>"
        "<script>"
        "async function submitSetup(){"
        "var b=document.getElementById('save');b.disabled=true;b.textContent='Starting…';"
        "var r=await window.pywebview.api.submit({"
        "provider:document.getElementById('provider').value,"
        "model:document.getElementById('model').value,"
        "api_key:document.getElementById('apikey').value,"
        "workspace:document.getElementById('workspace').value});"
        "if(r&&!r.ok){b.disabled=false;b.textContent='Save and start';"
        "var e=document.querySelector('.error');"
        "if(!e){e=document.createElement('div');e.className='error';"
        "document.querySelector('.card').appendChild(e);}"
        "e.textContent=r.error||'Please check your entries.';}}"
        "</script>"
    )
