#!/usr/bin/env python
"""Assemble the runtime half of the Windows desktop bundle.

Produces the ``webui/``, ``runtime/node/`` and ``runtime/python/`` parts of the
app directory that ``EvoScientist/desktop/app_paths.py`` expects::

    <out>/
      webui/dist/server.js        pinned @evoscientist/webui standalone build
      webui/dist/.next/ ...        (+ node_modules, public)
      runtime/node/node.exe        pinned Node runtime for the target platform
      runtime/python/python.exe    standalone CPython for agent code execution
      manifest.json                pinned versions + sha256 of the inputs

This runs on any host OS (Linux CI included): everything is downloaded for the
*target* platform, not the build host. The frozen EvoScientist.exe / langgraph.exe
(``_internal``) are produced separately by the PyInstaller step; this script
handles the WebUI, Node, and the agent's standalone Python runtime.

Cross-platform gotcha it handles: the published ``@evoscientist/webui`` tarball
bundles the sharp native binary of *whatever machine published it*
(``@img/sharp-darwin-arm64`` at time of writing). Next.js's runtime image
optimizer requires sharp, so a Windows bundle carrying only the macOS binary
would fail at runtime. This script removes the non-target sharp binaries and
installs the target one (``@img/sharp-<target>`` at the same sharp version).

Usage:
    # --webui-version defaults to the npm ``latest`` dist-tag; pin it for a
    # reproducible build. Either way the resolved version lands in manifest.json.
    uv run python packaging/windows/assemble_bundle.py --out dist/win-bundle
    uv run python packaging/windows/assemble_bundle.py --out X \\
        --webui-version 0.3.0 --node-version 22.11.0 \\
        --python-version 3.12.7 --python-tag 20241016 --target win32-x64
"""

from __future__ import annotations

import argparse
import base64
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
# python-build-standalone release assets (Astral). Relocatable, self-contained
# CPython with pip — the agent's code-execution interpreter, so its shell never
# depends on whatever python happens to be on the end-user's PATH.
PBS_RELEASES = "https://github.com/astral-sh/python-build-standalone/releases/download"

# npm target triple -> node.js dist archive infix. Extend as targets are added.
_NODE_ARCH = {
    "win32-x64": "win-x64",
    "win32-arm64": "win-arm64",
}

# npm target triple -> python-build-standalone platform triple.
_PYTHON_ARCH = {
    "win32-x64": "x86_64-pc-windows-msvc",
    "win32-arm64": "aarch64-pc-windows-msvc",
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
            # Path-containment check on the resolved path, not a string prefix:
            # ``startswith(str(dest))`` would accept a same-parent sibling like
            # ``<dest>-evil`` as if it were inside ``dest``.
            if dest != target and dest not in target.parents:
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


def _npm_dist(pkg: str, version: str) -> tuple[str, str, dict]:
    """Resolve ``(tarball URL, concrete version, dist)`` from the packument.

    ``version`` may be a dist-tag (e.g. ``latest``); the registry resolves it
    and the returned version is the concrete one, so callers record what was
    actually fetched rather than the tag. Authoritative — reads ``dist.tarball``
    rather than guessing the ``-/`` path. ``dist`` also carries the published
    ``integrity`` / ``shasum`` (see :func:`_verify_tarball_integrity`).
    """
    quoted = pkg.replace("/", "%2F")
    meta = _get_json(f"{REGISTRY}/{quoted}/{version}")
    dist = meta["dist"]
    return dist["tarball"], meta["version"], dist


def _verify_tarball_integrity(tgz: bytes, dist: dict) -> str:
    """Verify downloaded bytes against the checksum npm published for this
    version, and return a short description of what was checked.

    Closes the gap where the WebUI tarball's hash was recorded but never
    verified — a corrupted or tampered download now fails the build — without
    pinning a version (the check is against the registry's own published value
    for whatever ``latest`` resolved to). Prefers the SRI ``integrity`` field
    (sha512/384/256), falling back to the legacy ``shasum`` (sha1). Defensive:
    if the registry published neither (unexpected for npm), it warns rather than
    failing, so a metadata quirk never blocks a build.
    """
    integrity = dist.get("integrity")
    if isinstance(integrity, str) and "-" in integrity:
        algo, _, rest = integrity.partition("-")
        algo = algo.strip().lower()
        expected = rest.split()[0] if rest.split() else ""  # SRI may list several
        if algo in ("sha512", "sha384", "sha256") and expected:
            actual = base64.b64encode(hashlib.new(algo, tgz).digest()).decode()
            if actual != expected:
                raise RuntimeError(
                    f"WebUI tarball {algo} does not match npm's published "
                    f"integrity — corrupted or tampered download."
                )
            return f"{algo} matches npm integrity"
    shasum = dist.get("shasum")
    if isinstance(shasum, str) and shasum:
        actual = hashlib.sha1(tgz).hexdigest()
        if actual != shasum:
            raise RuntimeError(
                f"WebUI tarball sha1 {actual} does not match npm's published "
                f"shasum {shasum} — corrupted or tampered download."
            )
        return "sha1 matches npm shasum"
    return "no npm-published checksum to verify against"


def fetch_webui(version: str, out: Path) -> dict:
    """Download @evoscientist/webui@<version> and extract package/dist ->
    <out>/webui/dist. ``version`` may be a dist-tag (e.g. ``latest``); the
    resolved concrete version is recorded in the manifest. Returns manifest
    fields."""
    webui_dir = out / "webui"
    if webui_dir.exists():
        shutil.rmtree(webui_dir)
    url, resolved, dist = _npm_dist("@evoscientist/webui", version)
    print(f"[webui] {version} -> {resolved}: {url}")
    tgz = _get(url)
    print(f"[webui] integrity: {_verify_tarball_integrity(tgz, dist)}")
    files, skipped = _extract_tar_subtree(
        tgz, strip="package/dist/", dest=webui_dir / "dist"
    )
    server = webui_dir / "dist" / "server.js"
    if not server.exists():
        raise RuntimeError(
            f"{server} missing after extraction — tarball layout changed?"
        )
    print(f"[webui] extracted {files} files ({skipped} symlinks skipped)")
    return {"webui_version": resolved, "webui_sha256": _sha256(tgz)}


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
        url, _, _ = _npm_dist(pkg, ver)
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


def _write_pip_config(py_dir: Path) -> None:
    """Make the bundled interpreter's ``pip install`` default to ``--user``.

    pip reads ``<sys.prefix>/pip.ini`` as its site config, so this applies to
    the bundled interpreter only: with the agent shell's ``PYTHONUSERBASE``,
    its installs land in a per-user dir instead of the bundle tree (replaced
    wholesale on upgrade). A venv created from it has its own prefix, so it
    does not read this file and installs into itself as usual (a ``PIP_USER``
    env var would leak into the venv and make pip refuse there).
    """
    (py_dir / "pip.ini").write_text("[install]\nuser = true\n")


def fetch_python(version: str, tag: str, target: str, out: Path) -> dict:
    """Download a python-build-standalone ``install_only`` CPython for the
    target and extract it to <out>/runtime/python/ (python.exe at its root).

    This is the interpreter the packaged agent's shell runs code with, so it
    never depends on the end-user's PATH python. It ships with pip and no
    third-party packages — the agent installs what it needs on demand.
    """
    arch = _PYTHON_ARCH.get(target)
    if arch is None:
        raise RuntimeError(
            f"no Python archive mapping for target {target!r} "
            f"(known: {', '.join(_PYTHON_ARCH)})"
        )
    py_dir = out / "runtime" / "python"
    if py_dir.exists():
        shutil.rmtree(py_dir)
    py_dir.mkdir(parents=True)
    asset = f"cpython-{version}+{tag}-{arch}-install_only.tar.gz"
    url = f"{PBS_RELEASES}/{tag}/{asset}"
    print(f"[python] {url}")
    data = _get(url)
    # install_only archives contain a single top-level ``python/`` directory.
    files, skipped = _extract_tar_subtree(data, strip="python/", dest=py_dir)
    exe = py_dir / "python.exe"
    if not exe.exists():
        raise RuntimeError(f"{exe} missing after extraction — archive layout changed?")
    # The agent invokes ``python3`` (POSIX habit); Windows builds ship only
    # python.exe, so provide a python3.exe alias so that name resolves too.
    exe3 = py_dir / "python3.exe"
    if not exe3.exists():
        shutil.copyfile(exe, exe3)
    _write_pip_config(py_dir)
    print(f"[python] extracted {files} files ({skipped} symlinks skipped)")
    return {
        "python_version": version,
        "python_tag": tag,
        "python_sha256": _sha256(data),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, type=Path, help="output bundle dir")
    # Default to the npm ``latest`` dist-tag so a release build tracks the
    # current WebUI; the resolved concrete version is recorded in the manifest.
    # Pin an exact version for a reproducible build.
    p.add_argument("--webui-version", default="latest")
    p.add_argument("--node-version", default="22.11.0")
    # python-build-standalone: version is the CPython version, tag is the PBS
    # release date. Both pin one release asset; override from the releases page
    # if the default 404s. 3.12.x matches the repo's .python-version.
    p.add_argument("--python-version", default="3.12.7")
    p.add_argument("--python-tag", default="20241016")
    p.add_argument("--target", default="win32-x64", choices=list(_NODE_ARCH))
    p.add_argument("--skip-node", action="store_true")
    p.add_argument("--skip-python", action="store_true")
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
    if not args.skip_python:
        manifest.update(
            fetch_python(args.python_version, args.python_tag, args.target, out)
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Final sanity gate.
    server = out / "webui" / "dist" / "server.js"
    node = out / "runtime" / "node" / "node.exe"
    python = out / "runtime" / "python" / "python.exe"
    ok = (
        server.exists()
        and (args.skip_node or node.exists())
        and (args.skip_python or python.exists())
    )
    print(f"\n{'✓' if ok else '✗'} bundle at {out}")
    print(f"  webui/dist/server.js : {'present' if server.exists() else 'MISSING'}")
    print(
        f"  runtime/node/node.exe: "
        f"{'skipped' if args.skip_node else ('present' if node.exists() else 'MISSING')}"
    )
    print(
        f"  runtime/python/python.exe: "
        f"{'skipped' if args.skip_python else ('present' if python.exists() else 'MISSING')}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
