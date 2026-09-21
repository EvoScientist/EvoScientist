# Windows desktop bundle

Build tooling for the terminal-free Windows installer (issue #484, phase 1). The
installed app is a thin pywebview shell (`EvoScientist/desktop/`) that hosts the
existing WebUI through the shell-agnostic launcher (`EvoScientist/deploy/launcher.py`).

## Quick build (on Windows)

`build.ps1` runs all four steps below with consistent paths and a sanity gate
that aborts if the merged app tree is missing any half (the manual xcopy merge
could silently drop `runtime\python`):

```
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
# -> packaging\windows\dist\EvoScientist-Setup.exe
```

Useful switches: `-SkipAssemble` (reuse the fetched bundle when only the Python
side changed), `-SkipInstaller` (stop after the merged tree), `-Iscc <path>`
(non-default ISCC.exe), `-AppVersion <v>`. It kills a running
`EvoScientist.exe`/`langgraph.exe` first so PyInstaller can overwrite `_internal`.
The sections below document each step for when you need to run them by hand.

## Bundle layout

The installed app directory (what `EvoScientist/desktop/app_paths.py` resolves):

```
<app_root>/                 (PyInstaller onedir)
  EvoScientist.exe          windowed desktop shell                                (evoscientist.spec)
  langgraph.exe             bundled langgraph CLI (found by manager._langgraph_exe) (evoscientist.spec)
  _internal/...             bundled Python + all deps                             (evoscientist.spec)
  runtime/node/node.exe     pinned Node runtime                                   (assemble_bundle.py)
  runtime/python/python.exe standalone CPython for the agent's code execution     (assemble_bundle.py)
  webui/dist/server.js      pinned prebuilt @evoscientist/webui standalone        (assemble_bundle.py)
  webui/dist/.next, node_modules, public
  manifest.json             pinned versions + sha256 of the fetched inputs
```

`runtime/python/` is a bare python-build-standalone CPython (with pip, no
third-party packages) that the agent's `execute` shell runs code with — so it
never depends on whatever python is on the end-user's PATH. On-demand
`pip install`s are routed to a writable per-user dir (`~/.evoscientist/pypackages`)
via `PYTHONUSERBASE`, since the install dir is read-only for a non-admin user.

## assemble_bundle.py

Produces the `webui/`, `runtime/node/` and `runtime/python/` halves. Runs on any
host OS (Linux CI included) — everything is fetched for the *target* platform,
not the build host.

```
uv run python packaging/windows/assemble_bundle.py --out build/bundle
# pin explicitly:
uv run python packaging/windows/assemble_bundle.py --out build/bundle \
    --webui-version 0.2.7 --node-version 22.11.0 \
    --python-version 3.12.7 --python-tag 20241016 --target win32-x64
```

The Python half is a python-build-standalone `install_only` CPython (pin via
`--python-version` + `--python-tag`; if the default 404s, pick a release from
github.com/astral-sh/python-build-standalone/releases). It ships `python3.exe`
alongside `python.exe` so both names resolve on Windows.

Why it is not just "extract the tarball": the published `@evoscientist/webui`
tarball bundles the sharp native binary of whatever machine published it
(`@img/sharp-darwin-arm64`). Next.js's runtime image optimizer requires sharp,
so a Windows bundle carrying only the macOS binary fails at runtime. The script
removes non-target sharp natives and installs `@img/sharp-<target>` at the same
sharp version (win32 statically bundles libvips; no separate package needed).

## evoscientist.spec (PyInstaller)

Freezes the Python side into the onedir above: `EvoScientist.exe` (windowed
desktop shell) and `langgraph.exe` (the backend CLI), sharing one `_internal`.

```
# on Windows:
uv run --extra winbuild pyinstaller packaging/windows/evoscientist.spec --noconfirm
# -> dist/EvoScientist/  (EvoScientist.exe, langgraph.exe, _internal/, ...)
```

Two entry scripts feed it: `desktop_entry.py` and `langgraph_entry.py`. The spec
authors the tricky bits (dynamic graph imports from `langgraph.json`, the
langgraph/langchain/deepagents collect list, the pywebview WinForms backend). It
is authored on Linux and must be shaken out on a real Windows build — extend
`COLLECT_PACKAGES` / `HIDDEN` in the spec as the first build surfaces missing
modules.

`build.ps1` (or the manual step below) copies `assemble_bundle.py`'s `webui/` +
`runtime/` (both `node/` and `python/`) into `dist/EvoScientist/` next to the exes
before the installer packages it.

## evoscientist.iss (Inno Setup installer)

Wraps the merged app directory into `EvoScientist-Setup.exe`. It ensures the
Edge WebView2 Evergreen runtime is present (registry detect; download +
silent-install the Microsoft bootstrapper only if missing), lays the app tree
down under `Program Files`, and creates a Start-menu shortcut (desktop shortcut
optional). Needs Inno Setup 6.1+ (for `DownloadTemporaryFile`).

Its input is a single directory holding BOTH halves of the bundle — the
PyInstaller onedir with the `assemble_bundle.py` output copied in next to the
exes. Full build (on Windows):

```
# 1. runtime half (webui/ + runtime/node/)
uv run python packaging\windows\assemble_bundle.py --out build\bundle
# 2. Python half (EvoScientist.exe, langgraph.exe, _internal\)
uv run --extra winbuild pyinstaller packaging\windows\evoscientist.spec --noconfirm
# 3. merge the runtime half into the onedir
xcopy /E /I build\bundle\webui   dist\EvoScientist\webui
xcopy /E /I build\bundle\runtime dist\EvoScientist\runtime
copy       build\bundle\manifest.json dist\EvoScientist\
# 4. compile the installer -> packaging\windows\dist\EvoScientist-Setup.exe
iscc packaging\windows\evoscientist.iss
```

The installer lands in `packaging\windows\dist\` (`OutputDir=dist` is relative to
the `.iss`), not the repo-root `dist\`.

`iscc` defines override the pins: `iscc /DAppVersion=0.3.0 /DSourceDir=..\..\dist\EvoScientist packaging\windows\evoscientist.iss`.

Shortcut `WorkingDir` is `{userdocs}`, not `{app}`: Program Files is read-only,
and the launcher defaults its workspace (`runs/`, `skills/`, `media/`) to the
working directory (see `build_launcher_config`).

Authorable on Linux; compile/verify only on Windows.
