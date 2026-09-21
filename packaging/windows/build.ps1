<#
.SYNOPSIS
  One-command Windows installer build for EvoScientist (issue #484, phase 1).

.DESCRIPTION
  Runs the four build steps with consistent paths so the manual xcopy merge can
  no longer silently drop a subtree (e.g. runtime\python) from the packaged app:

    1. assemble_bundle.py  -> <BundleDir>\  (webui\, runtime\node\, runtime\python\)
    2. pyinstaller spec    -> <DistDir>\    (EvoScientist.exe, langgraph.exe, _internal\)
    3. merge the bundle halves into <DistDir>\, then a sanity gate that ABORTS if
       any required file is missing from the merged tree.
    4. iscc                -> packaging\windows\dist\EvoScientist-Setup.exe

  Running EvoScientist.exe / langgraph.exe are killed first: PyInstaller's clean
  step file-locks on _internal\*.pyd otherwise (WinError 5).

.EXAMPLE
  # from anywhere, on Windows:
  powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1

.EXAMPLE
  # skip the (slow) bundle fetch when only the Python side changed:
  packaging\windows\build.ps1 -SkipAssemble
#>
[CmdletBinding()]
param(
    [string]$AppVersion = "0.3.0",
    # Relative paths are resolved against the repo root, not the current dir.
    [string]$BundleDir = "build\bundle",
    [string]$DistDir = "dist\EvoScientist",
    # ISCC.exe: the test machine installs it here; override for another location.
    [string]$Iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    [string[]]$AssembleArgs = @(),   # e.g. -AssembleArgs '--python-version','3.12.8'
    [switch]$SkipAssemble,
    [switch]$SkipPyInstaller,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PkgDir = $PSScriptRoot

function Resolve-RepoPath([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $RepoRoot $p)
}

# PowerShell does not stop on a native exe's nonzero exit even under
# ErrorActionPreference=Stop; check $LASTEXITCODE explicitly after each.
function Invoke-Checked([string]$Label, [scriptblock]$Block) {
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit code $LASTEXITCODE)."
    }
}

$BundleFull = Resolve-RepoPath $BundleDir
$DistFull = Resolve-RepoPath $DistDir

Write-Host "Repo root : $RepoRoot"
Write-Host "Bundle    : $BundleFull"
Write-Host "Dist      : $DistFull"

# --- kill any running instances so PyInstaller can overwrite _internal ---
# Nothing to kill (or no taskkill) must not abort the build, so swallow errors.
foreach ($proc in "EvoScientist", "langgraph") {
    try { taskkill /F /IM "$proc.exe" 2>$null | Out-Null } catch {}
}

Push-Location $RepoRoot
try {
    # --- 1. runtime half (webui + node + python) ---
    if (-not $SkipAssemble) {
        Invoke-Checked "assemble_bundle.py -> $BundleDir" {
            uv run python (Join-Path $PkgDir "assemble_bundle.py") --out $BundleFull @AssembleArgs
        }
    }
    else {
        Write-Host "==> skipping assemble_bundle (using existing $BundleDir)" -ForegroundColor Yellow
    }

    # --- 2. Python half (frozen exes + _internal) ---
    if (-not $SkipPyInstaller) {
        Invoke-Checked "pyinstaller -> $DistDir" {
            uv run --extra winbuild pyinstaller (Join-Path $PkgDir "evoscientist.spec") --noconfirm
        }
    }
    else {
        Write-Host "==> skipping pyinstaller (using existing $DistDir)" -ForegroundColor Yellow
    }

    # --- 3. merge the runtime half into the onedir ---
    Write-Host "==> merging bundle into $DistDir" -ForegroundColor Cyan
    foreach ($sub in "webui", "runtime") {
        $src = Join-Path $BundleFull $sub
        $dst = Join-Path $DistFull $sub
        if (-not (Test-Path $src)) { throw "bundle is missing '$sub' at $src (run without -SkipAssemble)" }
        if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
        Copy-Item $src $DistFull -Recurse -Force
    }
    Copy-Item (Join-Path $BundleFull "manifest.json") $DistFull -Force

    # --- sanity gate: every half must be present in the MERGED tree ---
    $required = @(
        "EvoScientist.exe",
        "langgraph.exe",
        "runtime\python\python.exe",
        "runtime\node\node.exe",
        "webui\dist\server.js"
    )
    $missing = $required | Where-Object { -not (Test-Path (Join-Path $DistFull $_)) }
    if ($missing) {
        throw "merged app tree is incomplete - missing:`n  " + ($missing -join "`n  ")
    }
    Write-Host "    all required files present in $DistDir" -ForegroundColor Green

    # --- 4. compile the installer ---
    if (-not $SkipInstaller) {
        if (-not (Test-Path $Iscc)) {
            throw "ISCC.exe not found at '$Iscc' - install Inno Setup 6.1+ or pass -Iscc <path>."
        }
        Invoke-Checked "iscc -> EvoScientist-Setup.exe" {
            & $Iscc "/DAppVersion=$AppVersion" "/DSourceDir=$DistFull" (Join-Path $PkgDir "evoscientist.iss")
        }
        $setup = Join-Path $PkgDir "dist\EvoScientist-Setup.exe"
        Write-Host "`nInstaller: $setup" -ForegroundColor Green
    }
    else {
        Write-Host "==> skipping installer; merged app tree ready at $DistFull" -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
}
