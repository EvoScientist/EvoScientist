"""The Windows half of ``EvoSci update``: a PowerShell helper that installs
once EvoScientist has exited.

On Windows a process running from the uv tool environment keeps its files
open, and a ``uv tool install --force`` that fails on them leaves the
environment broken. ``EvoSci`` itself runs from that environment, so it
cannot stay alive while uv replaces it: after the version lookup, the
constraints download and the server check, it writes this helper, starts it
and exits. The helper (Windows PowerShell, not the environment's Python)
waits with no time limit until no process runs from the environment, runs
uv (with one retry after another wait), runs the new ``EvoSci setup`` and
writes the update's terminal line.

While it waits it prints ``running`` lines ("Update pending until you
close: ...", ``detail.waiting_for``), a new one whenever that set changes.
Ctrl+C in a terminal, or the end of its stdin under ``--json`` (the caller
closed it, or it was already closed), cancels the wait: nothing is changed
and the update ends with ``environment_busy``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HELPER_NAME = "evosci-update-helper.ps1"

HELPER = r"""
param([Parameter(Mandatory = $true)][string]$ConfigPath)

$ErrorActionPreference = 'Stop'
$cfg = Get-Content -Raw -Encoding UTF8 -LiteralPath $ConfigPath | ConvertFrom-Json
$Json = [bool]$cfg.json
$finished = $false
$blockers = @()

function Emit([string]$Status, [string]$Message, $Detail, [string]$Code, $Progress) {
    if ($Json) {
        $e = [ordered]@{ protocol = 1; stage = 'update'; status = $Status }
        if ($null -ne $Progress) { $e.progress = $Progress }
        if ($Message) { $e.message = $Message }
        if ($null -ne $Detail) { $e.detail = $Detail }
        if ($Code) { $e.code = $Code }
        $line = $e | ConvertTo-Json -Compress -Depth 6
        # ASCII only, like EvoSci's own lines.
        $line = [regex]::Replace($line, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
        [Console]::Out.WriteLine($line)
    } elseif ($Status -eq 'error') {
        [Console]::Out.WriteLine("  x update: $Message ($Code)")
    } elseif ($Status -eq 'done') {
        [Console]::Out.WriteLine("  ok update: $Message")
    } else {
        [Console]::Out.WriteLine("  update: $Message")
    }
    [Console]::Out.Flush()
}

# Get-Process and %TEMP% can return 8.3 short names (C:\Users\ABCDEF~1\...):
# both sides of a comparison are expanded.
if (-not ('EvoSciUpdate.Win32' -as [type])) {
    Add-Type -Namespace EvoSciUpdate -Name Win32 -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern uint GetLongPathName(string shortPath, System.Text.StringBuilder longPath, uint bufferSize);
'@
}
function Get-LongPath([string]$Path) {
    try { $full = [IO.Path]::GetFullPath($Path) } catch { return $Path }
    $sb = New-Object System.Text.StringBuilder 32768
    try {
        if ([EvoSciUpdate.Win32]::GetLongPathName($full, $sb, [uint32]$sb.Capacity) -gt 0) { $full = $sb.ToString() }
    } catch { }
    return $full
}
$sep = [IO.Path]::DirectorySeparatorChar
$Root = (Get-LongPath $cfg.env_dir).TrimEnd($sep) + $sep

function Get-Blockers {
    # The leading comma keeps a one-element result an array (a JSON list).
    return , @(Get-Process | ForEach-Object {
            $p = $null
            try { $p = $_.Path } catch { }
            if ($p -and $_.Id -ne $PID) {
                $long = Get-LongPath $p
                if ($long.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) {
                    [ordered]@{ pid = $_.Id; name = $_.ProcessName; path = $long }
                }
            }
        })
}

$stdin = $null
$pending = $null
if ($Json) {
    $stdin = [Console]::OpenStandardInput()
    $buf = New-Object byte[] 1
    $pending = $stdin.BeginRead($buf, 0, 1, $null, $null)
}

# Waits until nothing runs from the environment. False when the caller
# cancelled: under --json its stdin ended (also one that had already ended).
function Wait-Free([double]$Progress) {
    $last = $null
    while ($true) {
        $script:blockers = Get-Blockers
        if ($script:blockers.Count -eq 0) { return $true }
        $key = ($script:blockers | ForEach-Object { $_.pid }) -join ','
        if ($key -ne $last) {
            $names = ($script:blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', '
            Emit 'running' "Update pending until you close: $names" @{ waiting_for = $script:blockers } $null $Progress
            $last = $key
        }
        if ($Json -and $script:pending.IsCompleted) {
            if ($script:stdin.EndRead($script:pending) -eq 0) { return $false }
            $script:pending = $script:stdin.BeginRead($buf, 0, 1, $null, $null)
        }
        Start-Sleep -Milliseconds 500
    }
}

function Invoke-Uv {
    $ErrorActionPreference = 'Continue'
    # stdout is for event lines: uv's output goes to stderr.
    & $cfg.uv @($cfg.uv_args) 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
    return $LASTEXITCODE
}

function Invoke-OnConsole([string]$Exe, [string[]]$Arguments) {
    $quoted = foreach ($arg in $Arguments) {
        if ($arg -and $arg -notmatch '[\s"]') { $arg } else {
            $escaped = [regex]::Replace($arg, '(\\*)"', '$1$1\"')
            '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
        }
    }
    $info = New-Object Diagnostics.ProcessStartInfo $Exe
    $info.Arguments = $quoted -join ' '
    $info.UseShellExecute = $false
    $process = [Diagnostics.Process]::Start($info)
    $process.WaitForExit()
    return $process.ExitCode
}

try {
    foreach ($p in $cfg.uv_env.PSObject.Properties) { Set-Item -Path "Env:\$($p.Name)" -Value $p.Value }

    if (-not (Wait-Free 0.35)) { return }
    Emit 'running' "Installing EvoScientist $($cfg.target)" $null $null 0.4
    $rc = Invoke-Uv
    if ($rc -ne 0) {
        Emit 'running' "uv failed (exit $rc); trying once more when nothing runs from the environment" $null $null 0.5
        if (-not (Wait-Free 0.5)) { return }
        $rc = Invoke-Uv
    }
    if ($rc -ne 0) {
        Emit 'error' "uv could not install EvoScientist $($cfg.target) (exit $rc, see its output). If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
        $finished = $true
        return
    }

    Emit 'running' 'Running EvoSci setup' $null $null 0.8
    $setupArgs = @('setup')
    foreach ($s in @($cfg.skip)) { if ($s) { $setupArgs += @('--skip', $s) } }
    $failed = $null
    if ($Json) {
        $ErrorActionPreference = 'Continue'
        & $cfg.evosci @($setupArgs + '--json') | ForEach-Object {
            $line = "$_"
            [Console]::Out.WriteLine($line)
            [Console]::Out.Flush()
            try { $ev = $line | ConvertFrom-Json } catch { $ev = $null }
            if ($ev -and $ev.status -eq 'error' -and $null -eq $failed) {
                $failed = [ordered]@{ stage = $ev.stage; code = $ev.code }
            }
        }
        $setupRc = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
    } else {
        $setupRc = Invoke-OnConsole $cfg.evosci $setupArgs
    }
    if ($null -ne $failed -or $setupRc -ne 0) {
        $stage = if ($failed) { $failed.stage } else { 'setup' }
        Emit 'error' "EvoScientist is updated, but the $stage stage of EvoSci setup failed. Run it again: $($cfg.evosci) setup" $failed 'setup_failed'
        $finished = $true
        return
    }
    Emit 'done' "Updated EvoScientist $($cfg.installed) -> $($cfg.target)" @{ version = $cfg.target }
    $finished = $true
} finally {
    if (-not $finished) {
        # The wait was cancelled: Ctrl+C, or the end of stdin under --json.
        $names = ($blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', '
        Emit 'error' "Update cancelled while these still ran: $names. Nothing was changed." @{ waiting_for = $blockers } 'environment_busy'
    }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath $cfg.temp_dir
}
"""


def powershell() -> str:
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")


def start_helper(temp_dir: Path, config: dict[str, Any], *, json_mode: bool) -> None:
    """Write the helper and its config into *temp_dir* (the helper deletes it
    when it ends) and start it. Under ``--json`` it gets this process's stdin
    and stdout in a new process group without a console window, so a caller
    reading the pipe gets its lines after EvoSci exits; in a terminal it
    shares the console, so Ctrl+C reaches it."""
    script = temp_dir / HELPER_NAME
    script.write_text(HELPER, encoding="utf-8-sig")
    config_path = temp_dir / "config.json"
    config_path.write_text(
        json.dumps({**config, "temp_dir": str(temp_dir), "json": json_mode}),
        encoding="utf-8",
    )
    cmd = [
        powershell(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-ConfigPath",
        str(config_path),
    ]
    if json_mode:
        # Not DETACHED_PROCESS: a helper started that way never ran.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        subprocess.Popen(cmd, creationflags=flags, stdin=sys.stdin, stdout=sys.stdout)
    else:
        subprocess.Popen(cmd)
