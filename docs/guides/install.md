# One-Line Installation Guide

The one-line installer sets up EvoScientist on a completely blank machine—no `uv`, Python, Node.js, or Git required.

**Supported platforms:** macOS (Apple silicon), Linux (`x86_64` / `arm64`, glibc ≥ 2.28), Windows 10/11 (`x64`). *(Not supported: Intel Macs, Windows ARM64, musl Linux such as Alpine.)*

## Install Commands

**macOS, Linux, WSL2:**
```bash
curl -fsSL https://raw.githubusercontent.com/EvoScientist/EvoScientist/main/scripts/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/EvoScientist/EvoScientist/main/scripts/install.ps1 | iex
```

**Mainland China mirrors (`--cn` / `-Cn`):**
```bash
curl -fsSL https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@main/scripts/install.sh | bash -s -- --cn
```
```powershell
& ([scriptblock]::Create((irm https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@main/scripts/install.ps1))) -Cn
```

When it finishes, **open a new terminal** and run:

```bash
EvoSci onboard
EvoSci            # TUI (or `EvoSci --ui webui` for the browser UI)
```

## Installer Options

*(In PowerShell, `irm ... | iex` cannot accept flags—use the `[scriptblock]::Create(...)` form above.)*

| bash | PowerShell | What it does |
|---|---|---|
| `--cn` | `-Cn` | Uses mainland China mirrors (Tsinghua PyPI, npmmirror for Python/Node/WebUI/npx, jsDelivr for `constraints.txt`) and saves `mirror: cn` for future updates. Override `npx` registry anytime via `npm_config_registry` or `~/.npmrc`. |
| `--version <ver>` | `-Version <ver>` | Install a specific release (requires ≥ `0.3.6`, when `constraints.txt` was introduced). |
| `--no-modify-path` | `-NoModifyPath` | Skip adding `uv`'s tool bin folder to `PATH`. |
| `--no-setup` | `-NoSetup` | Install the CLI package only; skip `EvoSci setup`. |

## What the Installer Actually Does

Data lives in `~/.evoscientist` (or `$EVOSCIENTIST_DATA_DIR` if set). Your existing system Python, Node.js, and Git are never modified:

1. Downloads a SHA-256-verified `uv` (0.12.23) into `~/.evoscientist/tools/uv-0.12.23/` (recorded in `uv.json`).
2. Fetches the release's `constraints.txt` and installs EvoScientist as a `uv` tool on `uv`'s own Python 3.12 with exact tested dependency versions.
3. Adds `uv`'s tool bin directory (`~/.local/bin`) to your shell profile or Windows user `PATH`.
4. Runs `EvoSci setup`, which installs only what's missing: Node.js (if none or `< 20.9`), Git for Windows (Windows only), a Python environment for the agent's shell (if no usable `python` exists), and the local WebUI.

*(The scripts set `UV_SYSTEM_CERTS=1` and `UV_HTTP_TIMEOUT=300` automatically unless you already set them. If you later install Node.js ≥ 20.9 yourself via `nvm`/Homebrew, EvoScientist automatically prefers yours from terminals where it's on `PATH`.)*

## Updating & Repairing

Because the installer pins exact versions, `uv tool upgrade EvoScientist` will report *"Nothing to upgrade"*. Instead, update with:

```bash
EvoSci update
```

`EvoSci update` stages the new release into a temporary folder first (so a failed download leaves your install untouched), prompts before stopping the background server, waits for open sessions/MCP servers to close, and restarts the server when done.

You can also **re-run the install script** to repair or switch versions:
- Stop EvoScientist and `EvoSci server stop` first (on Windows, running processes lock files and the script will list them).
- Extras recorded in `uv`'s tool receipt are preserved; packages added outside the receipt (e.g., manually on Windows or older releases) will need to be re-added.
- Interrupted downloads of Node/Git/WebUI retry up to 3 times with resume; if one still fails, re-run `EvoSci setup`.

## Uninstalling

First stop the background server (`EvoSci server stop`, or `EvoSci service stop` on v0.4.0–0.4.2).

<details>
<summary><strong>macOS / Linux uninstall steps</strong></summary>

```bash
# 1. Uninstall EvoScientist via the installer's uv (do this before step 2 deletes uv.json)
uv=$(sed -E 's/.*"path": *"([^"]*)".*/\1/' ~/.evoscientist/tools/uv.json)
"$uv" tool uninstall EvoScientist
# Optional (only if no other uv tool uses them):
# "$uv" python uninstall 3.12 && "$uv" cache clean

# 2. Remove downloaded tools and agent Python env
rm -rf ~/.evoscientist/tools ~/.evoscientist/envs

# 3. Optional: delete settings, memories, skills, and sessions too
# rm -rf ~/.config/evoscientist ~/.evoscientist
```

Remove the `# uv` / `export PATH=...` lines from your shell profile if nothing else uses `~/.local/bin`.

</details>

<details>
<summary><strong>Windows (PowerShell) uninstall steps</strong></summary>

```powershell
# 1. Uninstall EvoScientist
$uv = (Get-Content -Raw -Encoding UTF8 -LiteralPath "$env:USERPROFILE\.evoscientist\tools\uv.json" | ConvertFrom-Json).path
& $uv tool uninstall EvoScientist
# Optional: & $uv python uninstall 3.12; & $uv cache clean

# 2. Remove downloaded tools and agent Python env
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath "$env:USERPROFILE\.evoscientist\tools", "$env:USERPROFILE\.evoscientist\envs"

# 3. Optional: remove $env:USERPROFILE\.config\evoscientist and $env:USERPROFILE\.evoscientist
```

Remove `%USERPROFILE%\.local\bin` from your user `PATH` if no other tools live there.

</details>

## Proxies & Corporate TLS Inspection

### Proxies

- **macOS / Linux:** `curl` and `uv` ignore GUI "system proxy" toggles (Clash, V2Ray, etc.). Either turn on TUN mode or export proxy vars first:
  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
  ```
- **Windows:** PowerShell and `uv` respect manual system proxies, but `uv` ignores PAC/WPAD scripts and per-protocol lists—export `$env:HTTPS_PROXY = "http://proxy.example.com:8080"` in those cases.

### Networks That Inspect TLS

Point these environment variables at your company's root CA PEM file if HTTPS connections fail with certificate errors:

| Variable | Used by |
|---|---|
| `SSL_CERT_FILE` | Python & `EvoSci setup` downloads on macOS/Linux |
| `UV_SYSTEM_CERTS=1` | `uv` calls made later by EvoScientist (e.g., MCP installer) |
| `PIP_CERT` | `pip` |
| `NODE_EXTRA_CA_CERTS` | Node.js (only needed for your own Node install, or certs not in OS store—`EvoSci`'s bundled Node sets `NODE_USE_SYSTEM_CA=1`) |
| `npm_config_cafile` | `npm` and `npx` (forwarded to `npx` MCP servers along with `npm_config_ca`, `npm_config_strict_ssl`, `npm_config_registry`, `npm_config_proxy`, `npm_config_https_proxy`, `npm_config_noproxy`) |
