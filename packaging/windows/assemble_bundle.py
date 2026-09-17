#!/usr/bin/env python
"""Assemble the runtime half of the Windows desktop bundle.

Produces the ``webui/`` and ``runtime/node/`` parts of the app directory that
``EvoScientist/desktop/app_paths.py`` expects::

    <out>/
      webui/dist/server.js        pinned @evoscientist/webui standalone build
      webui/dist/.next/ ...        (+ node_modules, public)
      runtime/node/node.exe        pinned Node runtime for the target platform
      manifest.json                pinned versions + sha256 of the inputs

This runs on any host OS (Linux CI included): everything is downloaded for the
*target* platform, not the build host. The Python + EvoScientist half of the
bundle (``runtime/python/`` + ``EvoScientist.exe``) is produced separately by
the PyInstaller step; this script only handles the WebUI + Node.

Cross-platform gotcha it handles: the published ``@evoscientist/webui`` tarball
bundles the sharp native binary of *whatever machine published it*
(``@img/sharp-darwin-arm64`` at time of writing). Next.js's runtime image
optimizer requires sharp, so a Windows bundle carrying only the macOS binary
would fail at runtime. This script removes the non-target sharp binaries and
installs the target one (``@img/sharp-<target>`` at the same sharp version).

Usage:
    uv run python packaging/windows/assemble_bundle.py --out dist/win-bundle
    uv run python packaging/windows/assemble_bundle.py --out X \\
        --webui-version 0.2.7 --node-version 22.11.0 --target win32-x64
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REGISTRY = "https://registry.npmjs.org"
NODE_DIST = "https://nodejs.org/dist"

# npm target triple -> node.js dist archive infix. Extend as targets are added.
_NODE_ARCH = {
    "win32-x64": "win-x64",
    "win32-arm64": "win-arm64",
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_tar_subtree(tgz: bytes, *, strip: str, dest: Path) -> tuple[int, int]:
    """Extract members of ``tgz`` whose name starts with ``strip`` into
    ``dest``, dropping the ``strip`` prefix. Returns (files, skipped_symlinks).

    Regular files and directories only; symlinks are skipped (the standalone
    Node runtime resolves modules by path, not via ``node_modules/.bin`` links)
    and counted so the caller can report them. Every write is confined to
    ``dest`` (path-traversal guard)."""
    files = 0
    skipped = 0
    dest = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.name.startswith(strip):
                continue
            rel = m.name[len(strip) :].lstrip("/")
            if not rel:
                continue
            target = (dest / rel).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
            if m.issym() or m.islnk():
                skipped += 1
                continue
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                skipped += 1
                continue
            target.write_bytes(src.read())
            files += 1
    return files, skipped


def _npm_tarball_url(pkg: str, version: str) -> str:
    """Resolve the exact tarball URL from the version packument (authoritative,
    rather than guessing the ``-/`` path)."""
    quoted = pkg.replace("/", "%2F")
    meta = _get_json(f"{REGISTRY}/{quoted}/{version}")
    return meta["dist"]["tarball"]


def fetch_webui(version: str, out: Path) -> dict:
    """Download @evoscientist/webui@<version> and extract package/dist ->
    <out>/webui/dist. Returns manifest fields."""
    webui_dir = out / "webui"
    if webui_dir.exists():
        shutil.rmtree(webui_dir)
    url = _npm_tarball_url("@evoscientist/webui", version)
    print(f"[webui] {url}")
    tgz = _get(url)
    files, skipped = _extract_tar_subtree(
        tgz, strip="package/dist/", dest=webui_dir / "dist"
    )
    server = webui_dir / "dist" / "server.js"
    if not server.exists():
        raise RuntimeError(
            f"{server} missing after extraction — tarball layout changed?"
        )
    print(f"[webui] extracted {files} files ({skipped} symlinks skipped)")
    return {"webui_version": version, "webui_sha256": _sha256(tgz)}


def _read_sharp_version(webui_dir: Path) -> str:
    pkg = webui_dir / "dist" / "node_modules" / "sharp" / "package.json"
    return json.loads(pkg.read_text())["version"]


def fix_sharp(out: Path, target: str) -> dict:
    """Swap the bundled (build-host) sharp native binary for the target's.

    Removes every ``@img/sharp-<plat>`` / ``@img/sharp-libvips-<plat>`` that is
    not ``target``, then installs ``@img/sharp-<target>`` at the bundled sharp
    version (plus its libvips subpackage if that package declares one — win32
    statically bundles libvips and declares none)."""
    img = out / "webui" / "dist" / "node_modules" / "@img"
    if not img.exists():
        print("[sharp] no @img dir — skipping (WebUI may not use sharp)")
        return {}

    sharp_ver = _read_sharp_version(out / "webui")
    keep_prefixes = ("colour",)  # pure-JS @img packages to preserve
    removed = []
    for child in sorted(img.iterdir()):
        name = child.name
        if name.startswith(keep_prefixes):
            continue
        if name.startswith("sharp-") and target not in name:
            shutil.rmtree(child)
            removed.append(name)
    if removed:
        print(f"[sharp] removed non-target natives: {', '.join(removed)}")

    installed = []
    wanted = [f"@img/sharp-{target}"]
    quoted = wanted[0].replace("/", "%2F")
    meta = _get_json(f"{REGISTRY}/{quoted}/{sharp_ver}")
    for dep in meta.get("dependencies") or {}:
        if dep.startswith("@img/sharp-libvips-"):
            wanted.append(dep)  # platforms that ship libvips separately
    for pkg in wanted:
        # libvips dep version may differ from sharp's; resolve via its own dep
        # spec when present, else match sharp's version.
        ver = sharp_ver
        if pkg != wanted[0]:
            ver = (meta.get("dependencies") or {})[pkg].lstrip("^~>=")
        url = _npm_tarball_url(pkg, ver)
        print(f"[sharp] + {pkg}@{ver}")
        tgz = _get(url)
        _extract_tar_subtree(tgz, strip="package/", dest=img / pkg.split("/")[-1])
        installed.append(f"{pkg}@{ver}")
    return {"sharp_version": sharp_ver, "sharp_natives": installed}


def fetch_node(version: str, target: str, out: Path) -> dict:
    """Download the target-platform Node archive and extract node.exe ->
    <out>/runtime/node/node.exe."""
    arch = _NODE_ARCH.get(target)
    if arch is None:
        raise RuntimeError(
            f"no Node archive mapping for target {target!r} "
            f"(known: {', '.join(_NODE_ARCH)})"
        )
    node_dir = out / "runtime" / "node"
    if node_dir.exists():
        shutil.rmtree(node_dir)
    node_dir.mkdir(parents=True)
    base = f"node-v{version}-{arch}"
    url = f"{NODE_DIST}/v{version}/{base}.zip"
    print(f"[node] {url}")
    data = _get(url)
    member = f"{base}/node.exe"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        try:
            exe = zf.read(member)
        except KeyError as exc:
            raise RuntimeError(f"{member} not found in Node archive") from exc
    (node_dir / "node.exe").write_bytes(exe)
    print(f"[node] node.exe {len(exe)} bytes")
    return {"node_version": version, "node_sha256": _sha256(data)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, type=Path, help="output bundle dir")
    p.add_argument("--webui-version", default="0.2.7")
    p.add_argument("--node-version", default="22.11.0")
    p.add_argument("--target", default="win32-x64", choices=list(_NODE_ARCH))
    p.add_argument("--skip-node", action="store_true")
    p.add_argument(
        "--no-sharp-fix",
        action="store_true",
        help="leave the tarball's bundled sharp native as-is (debug only)",
    )
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "target": args.target,
        "assembled_at": datetime.now(UTC).isoformat(),
    }
    manifest.update(fetch_webui(args.webui_version, out))
    if not args.no_sharp_fix:
        manifest.update(fix_sharp(out, args.target))
    if not args.skip_node:
        manifest.update(fetch_node(args.node_version, args.target, out))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Final sanity gate.
    server = out / "webui" / "dist" / "server.js"
    node = out / "runtime" / "node" / "node.exe"
    ok = server.exists() and (args.skip_node or node.exists())
    print(f"\n{'✓' if ok else '✗'} bundle at {out}")
    print(f"  webui/dist/server.js : {'present' if server.exists() else 'MISSING'}")
    print(
        f"  runtime/node/node.exe: "
        f"{'skipped' if args.skip_node else ('present' if node.exists() else 'MISSING')}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
