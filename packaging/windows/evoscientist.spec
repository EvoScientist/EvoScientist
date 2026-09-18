# PyInstaller spec — Windows desktop bundle (issue #484, phase 1, Task 3).
#
# Builds a single onedir containing TWO executables that share one _internal:
#   EvoScientist.exe  windowed desktop shell  (packaging/windows/desktop_entry.py)
#   langgraph.exe     console langgraph CLI    (packaging/windows/langgraph_entry.py)
#
# Why two exes: the desktop shell launches the backend with
# subprocess.Popen(["langgraph", "dev", ...]) via manager._langgraph_exe, which
# resolves a langgraph[.exe] sitting next to sys.executable. Shipping langgraph.exe
# in the same onedir makes that work with no PATH and no separate install.
#
# The webui/ and runtime/node/ trees come from assemble_bundle.py and are placed
# next to these exes by the installer (Task 6) — they are NOT built here.
#
# Build (on Windows):
#   uv run --extra winbuild pyinstaller packaging/windows/evoscientist.spec --noconfirm
# Output: dist/EvoScientist/  (EvoScientist.exe, langgraph.exe, _internal/, ...)
#
# This spec is authored on Linux and MUST be shaken out on a real Windows build:
# the langgraph/langchain/deepagents stack loads a lot dynamically, so expect to
# extend COLLECT_PACKAGES / HIDDEN below as the first build surfaces missing
# modules. Keep additions here, not scattered across --hidden-import flags.

import os

from PyInstaller.utils.hooks import collect_all

# The entry scripts live in packaging/windows/, but the EvoScientist package
# lives at the repo root (and is installed editable, which PyInstaller's module
# graph does not resolve on its own). Put the repo root on pathex so the
# analyzer can *find and analyze* EvoScientist's code and follow its imports —
# without this, EvoScientist modules are copied as data but their third-party
# deps (typer, rich, httpx, langchain, …) are never collected. SPECPATH is the
# directory of this spec, injected by PyInstaller.
_REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

# Packages with dynamic imports and/or data files PyInstaller cannot infer from
# the entry scripts alone. collect_all pulls submodules + data + binaries.
COLLECT_PACKAGES = [
    "EvoScientist",
    # langgraph runtime + dev server (the langgraph.exe path)
    "langgraph",
    "langgraph_api",
    "langgraph_runtime_inmem",
    "langgraph_cli",
    "langgraph_sdk",
    "langgraph.checkpoint",
    # langchain stack
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "langchain_openai",
    "langchain_deepseek",
    "langchain_google_genai",
    "langchain_ollama",
    "langchain_openrouter",
    "langchain_nvidia_ai_endpoints",
    "langchain_mcp_adapters",
    # agent construction
    "deepagents",
    "langchain_quickjs",
    # server + misc runtime deps that hook poorly
    "starlette",
    "uvicorn",
    "sse_starlette",
    "pydantic",
    # desktop window
    "webview",
]

# Graphs/checkpointer/http referenced by string in langgraph_dev/langgraph.json —
# imported at runtime by langgraph_api, so invisible to static analysis.
HIDDEN = [
    "EvoScientist.langgraph_dev.main_graph",
    "EvoScientist.langgraph_dev.graphs",
    "EvoScientist.langgraph_dev.http",
    "EvoScientist.sessions",
    # pywebview's Windows backend (WinForms + WebView2 via pythonnet)
    "webview.platforms.winforms",
    "clr",
]

datas, binaries, hiddenimports = [], [], list(HIDDEN)
for pkg in COLLECT_PACKAGES:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# langgraph_api reads these data files via ``Path(__file__).parent.parent`` — the
# directory ABOVE the package (site-packages), so they ship at the site-packages
# root, not inside langgraph_api/, and collect_all misses them. Ship them at the
# bundle root so the frozen ``__file__.parent.parent`` (_internal) resolves them.
# (validation.py -> openapi.json; queue_entrypoint.py -> logging.json)
import importlib.util

_lg_spec = importlib.util.find_spec("langgraph_api")
if _lg_spec and _lg_spec.origin:
    _site = os.path.dirname(os.path.dirname(_lg_spec.origin))
    for _fname in ("openapi.json", "logging.json"):
        _fp = os.path.join(_site, _fname)
        if os.path.exists(_fp):
            datas.append((_fp, "."))

# wasmtime (via quickjs_rs → langchain_quickjs, the code-interpreter middleware)
# loads its native lib through ctypes at runtime from wasmtime/<platform>/, which
# PyInstaller's static analysis and collect_all both miss (non-standard name in a
# platform subdir). Collect whatever native lib the installed package ships and
# place it at the same package-relative path the frozen wasmtime/_ffi.py expects.
_wt_spec = importlib.util.find_spec("wasmtime")
if _wt_spec and _wt_spec.origin:
    _wt_dir = os.path.dirname(_wt_spec.origin)
    for _plat in os.listdir(_wt_dir):
        _pdir = os.path.join(_wt_dir, _plat)
        if os.path.isdir(_pdir):
            for _f in os.listdir(_pdir):
                if _f.endswith((".dll", ".so", ".dylib")):
                    datas.append((os.path.join(_pdir, _f), f"wasmtime/{_plat}"))

# quickjs_rs ships the code-interpreter guest + transform modules as .wasm data
# files (quickjs_rs/_guest.wasm, quickjs_rs/_transform.wasm) and reads them via
# ``importlib.resources.files("quickjs_rs") / "<name>.wasm"`` at runtime. The
# package is only reached through langchain_quickjs's imports, so its code lands
# in the PYZ but these data files are never collected — the guest run then fails
# with "guest wasm not found". Ship every .wasm at the package-relative path the
# resource lookup expects (quickjs_rs/).
_qjs_spec = importlib.util.find_spec("quickjs_rs")
if _qjs_spec and _qjs_spec.origin:
    _qjs_dir = os.path.dirname(_qjs_spec.origin)
    for _f in os.listdir(_qjs_dir):
        if _f.endswith(".wasm"):
            datas.append((os.path.join(_qjs_dir, _f), "quickjs_rs"))

a_desktop = Analysis(
    ["desktop_entry.py"],
    pathex=[_REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
a_langgraph = Analysis(
    ["langgraph_entry.py"],
    pathex=[_REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Share collected libs across both exes in one _internal (avoids double-shipping
# the whole langgraph/langchain stack).
MERGE(
    (a_desktop, "desktop_entry", "EvoScientist"),
    (a_langgraph, "langgraph_entry", "langgraph"),
)

pyz_desktop = PYZ(a_desktop.pure)
pyz_langgraph = PYZ(a_langgraph.pure)

exe_desktop = EXE(
    pyz_desktop,
    a_desktop.scripts,
    [],
    exclude_binaries=True,
    name="EvoScientist",
    console=False,  # windowed: no console flashes behind the WebUI window
    disable_windowed_traceback=False,
    icon=None,
)
exe_langgraph = EXE(
    pyz_langgraph,
    a_langgraph.scripts,
    [],
    exclude_binaries=True,
    name="langgraph",
    console=True,  # backend subprocess; its stdout/stderr go to the dev log
    icon=None,
)

coll = COLLECT(
    exe_desktop,
    a_desktop.binaries,
    a_desktop.datas,
    exe_langgraph,
    a_langgraph.binaries,
    a_langgraph.datas,
    name="EvoScientist",
)
