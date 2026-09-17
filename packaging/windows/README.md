# Windows desktop bundle

Build tooling for the terminal-free Windows installer (issue #484, phase 1). The
installed app is a thin pywebview shell (`EvoScientist/desktop/`) that hosts the
existing WebUI through the shell-agnostic launcher (`EvoScientist/deploy/launcher.py`).

## Bundle layout

The installed app directory (what `EvoScientist/desktop/app_paths.py` resolves):

```
<app_root>/                 (PyInstaller onedir)
  EvoScientist.exe          windowed desktop shell                                (evoscientist.spec)
  langgraph.exe             bundled langgraph CLI (found by manager._langgraph_exe) (evoscientist.spec)
  _internal/...             bundled Python + all deps                             (evoscientist.spec)
  runtime/node/node.exe     pinned Node runtime                                   (assemble_bundle.py)
  webui/dist/server.js      pinned prebuilt @evoscientist/webui standalone        (assemble_bundle.py)
  webui/dist/.next, node_modules, public
  manifest.json             pinned versions + sha256 of the fetched inputs
```

## assemble_bundle.py

Produces the `webui/` and `runtime/node/` halves. Runs on any host OS (Linux CI
included) — everything is fetched for the *target* platform, not the build host.

```
uv run python packaging/windows/assemble_bundle.py --out dist/win-bundle
# pin explicitly:
uv run python packaging/windows/assemble_bundle.py --out dist/win-bundle \
    --webui-version 0.2.7 --node-version 22.11.0 --target win32-x64
```

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

The installer (Task 6) copies `assemble_bundle.py`'s `webui/` + `runtime/node/`
into `dist/EvoScientist/` next to the exes.

## Remaining phase-1 steps (not yet built)

- WebView2 Evergreen runtime detect/install.
- Inno Setup (or NSIS) installer wrapping the app_root + shortcut. Windows-only to build/verify.
