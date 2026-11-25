# One-Line Installation Guide

The one-line installer sets up EvoScientist on a completely blank machine—no `uv`, Python, Node.js, or Git required.

**Supported platforms:** macOS (Apple silicon), Linux (`x86_64` / `arm64`, glibc ≥ 2.28), Windows 10/11 (`x64`). *(Not supported: Intel Macs, Windows ARM64, musl Linux such as Alpine.)*

## Install Commands

**macOS, Linux, WSL2:**
```bash
curl -fsSL https://evoscientist.ai/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://evoscientist.ai/install.ps1 | iex
```

*(These URLs serve [`scripts/install.sh`](../../scripts/install.sh) / [`install.ps1`](../../scripts/install.ps1) from `main`; the `raw.githubusercontent.com` paths to those files work too.)*

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

1. Downloads a SHA-256-verified `uv` into `~/.evoscientist/tools/` (recorded in `uv.json`).
2. Fetches the release's `constraints.txt` and installs EvoScientist as a `uv` tool on `uv`'s own Python 3.12 with exact tested dependency versions.
3. Adds `uv`'s tool bin directory (`~/.local/bin`) to `PATH` via `uv tool update-shell`: on macOS/Linux it appends to your shell profile (`~/.bashrc` and `~/.bash_profile` or `~/.profile` for bash, `~/.zshenv` for zsh, the fish config); on Windows it updates your user `PATH`.
4. Runs `EvoSci setup`, which installs only what's missing: Node.js (if none or `< 20.9`), Git for Windows (Windows only), a Python environment for the agent's shell (if no usable `python` exists), and the local WebUI.

*(The scripts set `UV_SYSTEM_CERTS=1` and `UV_HTTP_TIMEOUT=300` automatically unless you already set them. If you later install Node.js ≥ 20.9 yourself via `nvm`/Homebrew, EvoScientist automatically prefers yours from terminals where it's on `PATH`.)*

## Updating & Repairing

Because the installer pins exact versions, `uv tool upgrade EvoScientist` will report *"Nothing to upgrade"*. Instead, update with:

```bash
EvoSci update
```

`EvoSci update` stages the new release into a temporary folder first (so a failed download leaves your install untouched), prompts before stopping the background server, waits for open sessions/MCP servers to close, and restarts the server when done.

*(Very old releases have no `EvoSci update` and weren't pinned: if `EvoSci update` doesn't exist, run `uv tool upgrade EvoScientist` once with EvoScientist closed. Docker images update by pulling a new image.)*

You can also **re-run the install script** to repair or switch versions:
- Stop EvoScientist and `EvoSci server stop` first (on Windows, running processes lock files and the script will list them).
- Extras recorded in `uv`'s tool receipt are preserved. On Windows (and with older releases), packages added when you installed a channel or MCP server aren't in the receipt and are removed—add them again afterwards.
- Interrupted downloads of Node/Git/WebUI retry up to 3 times with resume; if one still fails, re-run `EvoSci setup`.

## Uninstalling

First stop the background server (`EvoSci server stop`).

<details>
<summary><strong>macOS / Linux uninstall steps</strong></summary>

```bash
# 1. Uninstall EvoScientist via the installer's uv (do this before step 2 deletes uv.json)
uv=$(sed -E 's/.*"path": *"([^"]*)".*/\1/' ~/.evoscientist/tools/uv.json)
"$uv" tool uninstall EvoScientist
# Optional (only if no other uv on the machine uses them):
# "$uv" python uninstall 3.12 && "$uv" cache clean

# 2. Remove downloaded tools and agent Python env
rm -rf ~/.evoscientist/tools ~/.evoscientist/envs

# 3. Optional: delete settings, memories, skills, and sessions too
# rm -rf ~/.config/evoscientist ~/.evoscientist
```

Remove the `# uv` / `export PATH=...` lines from the shell profile the installer edited (see [step 3 above](#what-the-installer-actually-does)) if nothing else uses `~/.local/bin`.

</details>

<details>
<summary><strong>Windows (PowerShell) uninstall steps</strong></summary>

```powershell
# 1. Uninstall EvoScientist
$uv = (Get-Content -Raw -Encoding UTF8 -LiteralPath "$env:USERPROFILE\.evoscientist\tools\uv.json" | ConvertFrom-Json).path
& $uv tool uninstall EvoScientist
# Optional (only if no other uv on the machine uses them): & $uv python uninstall 3.12; & $uv cache clean

# 2. Remove downloaded tools and agent Python env
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath "$env:USERPROFILE\.evoscientist\tools", "$env:USERPROFILE\.evoscientist\envs"

# 3. Optional: remove $env:USERPROFILE\.config\evoscientist and $env:USERPROFILE\.evoscientist
```

Remove `%USERPROFILE%\.local\bin` from your user `PATH` if no other tools live there.

</details>

## Docker

A pre-built image on [GitHub Container Registry](https://github.com/EvoScientist/EvoScientist/pkgs/container/evoscientist) bundles EvoScientist with the cross-platform channel extras (`all-channels`), `uv`, and Node.js/`npx` for MCP servers. File edits and shell commands stay confined to the volumes you mount. *(iMessage needs macOS's Messages.app and doesn't work in the container.)*

```bash
mkdir -p workspace   # create first on Linux so Docker doesn't make it root-owned
docker run --rm -v evosci-data:/home/evosci/.evoscientist ghcr.io/evoscientist/evoscientist:latest setup   # once
docker run -it --rm \
  --env-file .env \
  -v "$(pwd)/workspace:/workspace" \
  -v evosci-data:/home/evosci/.evoscientist \
  ghcr.io/evoscientist/evoscientist:latest
```

| Mount | Purpose |
|---|---|
| `--env-file .env` | API keys (`ANTHROPIC_API_KEY=...`, one `NAME=value` per line), set as the container's environment |
| `./workspace:/workspace` | The agent's working directory |
| `evosci-data:/home/evosci/.evoscientist` | Sessions, skills, memories, `config.yaml`/`mcp.yaml`, and the research Python that `setup` builds |

The image's entrypoint is `evosci`, so `setup`, `config set ...`, or `serve` after the image name run those subcommands. Update by pulling a new image.

**Compose:** the repo's [`docker-compose.yml`](../../docker-compose.yml) reads a `.env` file next to it, which you create. Run `docker compose run --rm evoscientist setup` once, then `docker compose run --rm evoscientist`.

### Linux file permissions

The image runs as a non-root user (`evosci`, UID `1000`). Docker Desktop (macOS, Windows) handles bind-mount permissions itself; on Linux:

- **Your UID is 1000:** nothing to do beyond `mkdir -p workspace`.
- **You work as root (UID 0):** run `chown -R 1000:1000 ./workspace` once. (A local build with `UID=0` fails, since that user already exists in the image.)
- **Any other UID:** build the image as yourself from a clone of this repo and use `evoscientist:local` everywhere, including `setup` (with compose, set `image: evoscientist:local`):
  ```bash
  docker build --build-arg UID="$(id -u)" -t evoscientist:local .
  ```
  If you already used the published image, hand the data volume to your UID once:
  ```bash
  docker run --rm --user 0 --entrypoint chown -v evosci-data:/home/evosci/.evoscientist evoscientist:local -R "$(id -u):$(id -g)" /home/evosci/.evoscientist
  # compose names the volume <project>_evosci-data, so run it through compose instead:
  docker compose run --rm --user 0 --entrypoint chown evoscientist -R "$(id -u):$(id -g)" /home/evosci/.evoscientist
  ```
  Alternatively, `sudo chown -R 1000:1000 ./workspace` once, at the cost of needing `sudo` to edit those files from the host.

Apart from those one-off `chown` commands, don't pass `--user` to `docker run` or set `user:` in compose: the data folder belongs to the build-time UID, so any other UID fails with `Permission denied`.

### Adding extras or LaTeX

The `stt` (speech-to-text) and `oauth` (Claude Pro/Max sign-in via `ccproxy`) extras and TinyTeX aren't in the image. Derive your own:

```dockerfile
FROM ghcr.io/evoscientist/evoscientist:latest

USER root
RUN uv pip install --python /opt/venv/bin/python "EvoScientist[stt,oauth]"
USER evosci

# TinyTeX (official install pipes a remote script into sh; for a pinned setup,
# fetch a release tarball from https://github.com/rstudio/tinytex-releases instead)
RUN curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh \
 && /home/evosci/.TinyTeX/bin/*/tlmgr install latexmk
```

## Proxies & Corporate TLS Inspection

### Proxies

- **macOS / Linux:** `curl` and `uv` ignore GUI "system proxy" toggles (Clash, V2Ray, etc.). Either turn on TUN mode or export proxy vars first:
  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
  ```
- **Windows:** PowerShell and `uv` respect manual system proxies, but `uv` ignores PAC/WPAD scripts and per-protocol lists—export `$env:HTTPS_PROXY = "http://proxy.example.com:8080"` in those cases.

### Networks That Inspect TLS

If HTTPS connections fail with certificate errors, point the file variables below at a PEM file containing your company's root CA (`UV_SYSTEM_CERTS=1` is a flag that makes `uv` use the OS certificate store):

| Variable | Used by |
|---|---|
| `SSL_CERT_FILE` | Python & `EvoSci setup` downloads on macOS/Linux |
| `UV_SYSTEM_CERTS=1` | `uv` calls made later by EvoScientist (e.g., MCP installer) |
| `PIP_CERT` | `pip` |
| `NODE_EXTRA_CA_CERTS` | Node.js (only needed for your own Node install, or certs not in OS store—`EvoSci`'s bundled Node sets `NODE_USE_SYSTEM_CA=1`) |
| `npm_config_cafile` | `npm` and `npx` (forwarded to `npx` MCP servers along with `npm_config_ca`, `npm_config_strict_ssl`, `npm_config_registry`, `npm_config_proxy`, `npm_config_https_proxy`, `npm_config_noproxy`) |
