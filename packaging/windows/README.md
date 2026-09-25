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
(non-default ISCC.exe), `-AppVersion <v>`, `-WebuiVersion <v>` (WebUI npm
version; defaults to the `latest` dist-tag, pass an exact version to pin). When
rebuilding an older EvoScientist tag, pass the WebUI version that was current
for that release: `latest` would pull a newer WebUI than the one it shipped
with. It kills a running
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
  webui/dist/server.js      prebuilt @evoscientist/webui standalone               (assemble_bundle.py)
  webui/dist/.next, node_modules, public
  manifest.json             pinned versions + sha256 of the fetched inputs
```

`runtime/python/` is a bare python-build-standalone CPython (with pip, no
third-party packages) that the agent's `execute` shell runs code with — so it
never depends on whatever python is on the end-user's PATH. On-demand
`pip install`s are routed to a per-user dir (`~/.evoscientist/pypackages`) via
`PYTHONUSERBASE` plus a `runtime/python/pip.ini` that defaults the bundled
interpreter's installs to `--user`, keeping them out of the pinned bundle tree
(replaced wholesale on upgrade) rather than writing into `runtime/python`. A
venv the agent creates has its own prefix, so it ignores that `pip.ini` and
installs into itself.

## assemble_bundle.py

Produces the `webui/`, `runtime/node/` and `runtime/python/` halves. Runs on any
host OS (Linux CI included) — everything is fetched for the *target* platform,
not the build host.

```
# --webui-version defaults to the npm `latest` dist-tag (resolved version is
# recorded in manifest.json either way):
uv run python packaging/windows/assemble_bundle.py --out build/bundle
# pin explicitly for a reproducible build:
uv run python packaging/windows/assemble_bundle.py --out build/bundle \
    --webui-version 0.3.0 --node-version 22.11.0 \
    --python-version 3.12.7 --python-tag 20241016 --target win32-x64
```

Every download is checked against its publisher's checksum and a mismatch
fails the build: the WebUI and sharp tarballs against npm's `integrity`, Node
against `SHASUMS256.txt`, and Python against the release's `SHA256SUMS`.

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
uv run --extra winbuild --extra desktop pyinstaller packaging/windows/evoscientist.spec --noconfirm
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
down in a per-user location (`{localappdata}\Programs`, no elevation), and
creates a Start-menu shortcut (desktop shortcut optional). Needs Inno Setup 6.3+
(`x64compatible` needs 6.3; `DownloadTemporaryFile` 6.1).

Its input is a single directory holding BOTH halves of the bundle — the
PyInstaller onedir with the `assemble_bundle.py` output copied in next to the
exes. Full build (on Windows):

```
# 1. runtime half (webui/ + runtime/node/ + runtime/python/)
uv run python packaging\windows\assemble_bundle.py --out build\bundle
# 2. Python half (EvoScientist.exe, langgraph.exe, _internal\)
uv run --extra winbuild --extra desktop pyinstaller packaging\windows\evoscientist.spec --noconfirm
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

Shortcut `WorkingDir` is `{userdocs}`, not `{app}`: `{app}` holds the app (and is
wiped on uninstall), so user data does not belong there. The desktop defaults its
workspace (`runs/`, `skills/`, `media/`) to a `Documents\EvoScientist` subfolder
(`app_paths.default_workspace()`), never the whole working directory.

Authorable on Linux; compile/verify only on Windows.
