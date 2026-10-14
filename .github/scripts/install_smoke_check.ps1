# Checks for the installer smoke workflow (install-smoke.yml), run after
# scripts/install.ps1 on a clean Windows runner. Same checks as
# install_smoke_check.sh:
#
#   online    the version, where the package was imported from, the installed
#             versions against constraints.txt, a second setup that downloads
#             nothing, and the WebUI launch with the oldest and the newest
#             WebUI release in the core's range
#   block     adds outbound firewall rules for the installed Python and Node
#             executables, then checks that they cannot reach https://pypi.org
#   offline   EvoSci setup and the WebUI launch; run after block
#
# EXPECTED_VERSION (else the newest release on INDEX_URL) and
# EVOSCI_INSTALL_CONSTRAINTS (else constraints.txt at the release tag) are
# optional.
param([Parameter(Mandatory = $true)][ValidateSet('online', 'block', 'offline')][string]$Mode)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# EvoSci's output goes to pipes and files here; with the ANSI code page Python
# fails on its progress symbols, as in install.ps1.
$env:PYTHONIOENCODING = 'utf-8'

$IndexUrl = if ($env:INDEX_URL) { $env:INDEX_URL } else { 'https://pypi.org/simple' }
$DataDir = if ($env:EVOSCIENTIST_DATA_DIR) { $env:EVOSCIENTIST_DATA_DIR } else { Join-Path $HOME '.evoscientist' }
$BackendPort = if ($env:EVOSCIENTIST_LANGGRAPH_DEV_PORT) { $env:EVOSCIENTIST_LANGGRAPH_DEV_PORT } else { '6174' }
$WebuiPort = if ($env:EVOSCIENTIST_WEBUI_PORT) { $env:EVOSCIENTIST_WEBUI_PORT } else { '4716' }
$BackendUrl = "http://127.0.0.1:$BackendPort/ok"
$WebuiUrl = "http://127.0.0.1:$WebuiPort/"

function Fail([string]$Text) {
    Write-Host "::error::$Text"
    exit 1
}

# Runs a native command and returns its stdout as text (stderr goes to the
# log); native stderr must not become a terminating error.
function Invoke-Native([string]$Exe, [string[]]$Arguments, [switch]$WithStderr) {
    $ErrorActionPreference = 'Continue'
    if ($WithStderr) {
        $out = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
    } else {
        $out = & $Exe @Arguments | ForEach-Object { "$_" }
    }
    $script:NativeExitCode = $LASTEXITCODE
    return ($out -join "`n")
}

$Uv = (Get-Content -Raw (Join-Path $DataDir 'tools\uv.json') | ConvertFrom-Json).path
if (-not (Test-Path $Uv)) { Fail "No uv recorded in $DataDir\tools\uv.json" }
$BinDir = (Invoke-Native $Uv @('tool', 'dir', '--bin')).Trim()
$ToolDir = (Invoke-Native $Uv @('tool', 'dir')).Trim()
$EvoSci = Join-Path $BinDir 'EvoSci.exe'
$Python = Join-Path $ToolDir 'evoscientist\Scripts\python.exe'

function Test-Answers([string]$Url) {
    try {
        $request = [Net.WebRequest]::Create($Url)
        $request.Proxy = $null
        $request.Timeout = 5000
        $response = $request.GetResponse()
        $code = [int]$response.StatusCode
        $response.Close()
    } catch [Net.WebException] {
        if (-not $_.Exception.Response) { return $false }
        $code = [int]$_.Exception.Response.StatusCode
    }
    return ($code -ge 200 -and $code -lt 500)
}

function Show-Logs([string]$Log) {
    Write-Host '--- EvoSci output'
    Get-Content -ErrorAction SilentlyContinue $Log, "$Log.err" | Write-Host
    foreach ($file in @("$HOME\.config\evoscientist\langgraph_dev.log", "$HOME\.config\evoscientist\webui.log")) {
        Write-Host "--- $file"
        Get-Content -Tail 80 -ErrorAction SilentlyContinue $file | Write-Host
    }
}

# Starts the WebUI, waits until the backend's /ok and the WebUI answer, then
# stops the whole process tree and checks that both ports close.
function Test-Launch {
    $work = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work | Out-Null
    $log = "$work.log"
    $proc = Start-Process -FilePath $EvoSci -ArgumentList @('--ui', 'webui', '--workdir', "`"$work`"") -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden
    $up = $false
    for ($i = 0; $i -lt 240; $i++) {
        if ((Test-Answers $BackendUrl) -and (Test-Answers $WebuiUrl)) { $up = $true; break }
        if ($proc.HasExited) { break }
        Start-Sleep -Seconds 1
    }
    # Windows has no signal that runs EvoSci's own cleanup: stop the tree,
    # then the recorded backend.
    & taskkill.exe /PID $proc.Id /T /F | Out-Null
    Invoke-Native $EvoSci @('server', 'stop') | Out-Null
    if (-not $up) {
        Show-Logs $log
        Fail "The backend ($BackendUrl) and the WebUI ($WebuiUrl) did not both answer within $i s"
    }
    Start-Sleep -Seconds 3
    if ((Test-Answers $BackendUrl) -or (Test-Answers $WebuiUrl)) {
        Show-Logs $log
        Fail 'The backend or the WebUI still answers after EvoSci was stopped'
    }
    Write-Host 'WebUI launch ok'
}

function Get-ExpectedVersion {
    if ($env:EXPECTED_VERSION) { return $env:EXPECTED_VERSION }
    $listing = Invoke-RestMethod -UseBasicParsing -Uri "$IndexUrl/evoscientist/" -Headers @{ Accept = 'application/vnd.pypi.simple.v1+json' }
    return @($listing.versions | Where-Object { $_ -match '^\d+(\.\d+){1,3}$' } | Sort-Object { [version]$_ }) | Select-Object -Last 1
}

function Test-Online {
    $expected = Get-ExpectedVersion
    $actual = (Invoke-Native $EvoSci @('--version', '--json') | ConvertFrom-Json).version
    if ($actual -ne $expected) { Fail "EvoSci --version --json reports $actual, expected $expected" }
    Write-Host "version ok: $actual"

    $file = Invoke-Native $Python @('-c', 'import EvoScientist; print(EvoScientist.__file__)')
    if (-not $file.StartsWith($ToolDir + '\', [StringComparison]::OrdinalIgnoreCase)) {
        Fail "EvoScientist is imported from $file, outside $ToolDir"
    }
    Write-Host "imported from the tool environment: $file"

    $constraints = $env:EVOSCI_INSTALL_CONSTRAINTS
    if (-not $constraints) {
        $constraints = Join-Path ([IO.Path]::GetTempPath()) 'evosci-constraints.txt'
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/EvoScientist/EvoScientist/v$actual/constraints.txt" -OutFile $constraints
    }
    # Re-resolving the installed set under the constraints changes nothing only
    # when every installed version is the one constraints.txt pins.
    $frozen = Join-Path ([IO.Path]::GetTempPath()) 'evosci-frozen.txt'
    [IO.File]::WriteAllText($frozen, (Invoke-Native $Uv @('pip', 'freeze', '--python', $Python)))
    # uv reports "Would make no changes" on stderr.
    $out = Invoke-Native $Uv @('pip', 'install', '--python', $Python, '--dry-run', '-r', $frozen, '-c', $constraints) -WithStderr
    if ($NativeExitCode -ne 0 -or $out -notmatch 'Would make no changes') {
        Fail "The installed versions differ from constraints.txt: $out"
    }
    Write-Host 'installed versions match constraints.txt'

    $out = Invoke-Native $EvoSci @('setup', '--json')
    if ($NativeExitCode -ne 0) { Fail "The second EvoSci setup failed: $out" }
    if ($out -match '"(Downloading |Installing |Creating the virtual)') { Fail "The second EvoSci setup downloaded or installed something: $out" }
    Write-Host 'second setup downloaded nothing'

    # The oldest and the newest WebUI release in the core's range, through the
    # test-only override of the range; the newest stays installed.
    $code = @'
from packaging.version import Version
from EvoScientist.config import load_config
from EvoScientist.setup.node import DEFAULT_NPM_REGISTRY, NPM_REGISTRIES
from EvoScientist.setup.webui import WEBUI_COMPAT, fetch_metadata, in_range
registry = NPM_REGISTRIES.get(load_config().mirror, DEFAULT_NPM_REGISTRY).rstrip("/")
found = sorted((v for v in fetch_metadata(registry)["versions"] if in_range(v, WEBUI_COMPAT)), key=Version)
print(found[0], found[-1])
'@
    $script = Join-Path ([IO.Path]::GetTempPath()) 'evosci-webui-range.py'
    [IO.File]::WriteAllText($script, $code)
    $oldest, $newest = (Invoke-Native $Python @($script)).Trim().Split(' ')
    foreach ($version in @($oldest, $newest) | Select-Object -Unique) {
        Write-Host "WebUI $version"
        $env:EVOSCIENTIST_WEBUI_COMPAT = "==$version"
        try {
            $out = Invoke-Native $EvoSci @('setup', '--stage', 'webui', '--json')
            if ($NativeExitCode -ne 0) { Fail "EvoSci setup --stage webui failed for WebUI ${version}: $out" }
            if ($out -notmatch [regex]::Escape("`"Using WebUI $version ")) { Fail "WebUI $version was not installed: $out" }
            Test-Launch
        } finally {
            Remove-Item Env:\EVOSCIENTIST_WEBUI_COMPAT
        }
    }
}

# Blocks outbound traffic of every Python and Node executable that the
# install uses; loopback traffic is not filtered by Windows Firewall.
function Block-Network {
    $basePython = Invoke-Native $Python @('-c', 'import sys; print(sys._base_executable)')
    $programs = @($basePython.Trim(), $Python)
    $programs += @(Get-ChildItem (Join-Path $ToolDir 'evoscientist\Scripts') -Filter *.exe | ForEach-Object { $_.FullName })
    $programs += @(Get-ChildItem $BinDir -Filter *.exe | ForEach-Object { $_.FullName })
    $node = Get-Content -Raw -ErrorAction SilentlyContinue (Join-Path $DataDir 'tools\node.json') | ConvertFrom-Json
    if ($node) { $programs += Join-Path $node.path 'node.exe' }
    $envPython = Join-Path $DataDir 'envs\default\Scripts\python.exe'
    if (Test-Path $envPython) { $programs += $envPython }
    $i = 0
    foreach ($program in $programs | Select-Object -Unique) {
        $i++
        New-NetFirewallRule -DisplayName "evosci-smoke-block-$i" -Direction Outbound -Program $program -Action Block | Out-Null
        Write-Host "blocked: $program"
    }
    $probe = 'import urllib.request; urllib.request.urlopen("https://pypi.org/simple/", timeout=15)'
    foreach ($exe in @($basePython.Trim(), $Python)) {
        Invoke-Native $exe @('-c', $probe) | Out-Null
        if ($NativeExitCode -eq 0) { Fail "$exe can still reach https://pypi.org: the network is not blocked" }
    }
    if ($node) {
        Invoke-Native (Join-Path $node.path 'node.exe') @('-e', 'fetch("https://registry.npmjs.org/").then(() => process.exit(0), () => process.exit(1))') | Out-Null
        if ($NativeExitCode -eq 0) { Fail 'node can still reach https://registry.npmjs.org: the network is not blocked' }
    }
    Write-Host 'network blocked'
}

function Test-Offline {
    $out = Invoke-Native $EvoSci @('setup', '--json')
    if ($NativeExitCode -ne 0) { Fail "EvoSci setup failed offline: $out" }
    if ($out -notmatch '"stage": *"webui", *"status": *"done"') { Fail "The webui stage did not end with done offline: $out" }
    Write-Host 'offline setup ok'
    Test-Launch
}

switch ($Mode) {
    'online' { Test-Online }
    'block' { Block-Network }
    'offline' { Test-Offline }
}
