# Windows desktop bundle

Build tooling for the terminal-free Windows installer (issue #484, phase 1). The
installed app is a thin pywebview shell (`EvoScientist/desktop/`) that hosts the
existing WebUI through the shell-agnostic launcher (`EvoScientist/deploy/launcher.py`).

## Bundle layout

The installed app directory (what `EvoScientist/desktop/app_paths.py` resolves):

```
<app_root>/
  EvoScientist.exe          windowed launcher -> python -m EvoScientist.desktop   (PyInstaller step)
  runtime/python/...        bundled Python + EvoScientist package                 (PyInstaller step)
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

## Remaining phase-1 steps (not yet built)

- PyInstaller onedir for `runtime/python/` + `EvoScientist.exe`.
- WebView2 Evergreen runtime detect/install.
- Inno Setup (or NSIS) installer wrapping the app_root + shortcut. Windows-only to build/verify.
