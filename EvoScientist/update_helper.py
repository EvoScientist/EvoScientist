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
Cancelling the wait changes nothing and ends the update with
``environment_busy``:

- In a terminal the helper runs in a window of its own: Ctrl+C there cancels.
  The window the update was started from has its prompt back by then, and a
  prompt (PSReadLine) reads Ctrl+C as a key, so no Ctrl+C event would reach
  a helper sharing it.
- Under ``--json`` the waiting lines carry ``detail.cancel_file``: the caller
  creates that file to cancel. Closing stdin cannot be the signal: Node
  destroys a child's stdin when the child (``EvoSci``) exits, which is right
  after the handover. A stdin that is not a pipe (``NUL``, or a console:
  ``--json`` is for programs) marks an unattended run, which fails at once
  instead of waiting.
- A wait cancelled after a failed uv run (before the retry) ends with
  ``install_failed``, not ``environment_busy``: that uv run may have left the
  environment broken.
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
# uv and EvoSci write UTF-8: decode their output, and write this window's
# lines, as UTF-8 rather than the console's code page (paths with non-ASCII
# characters). Fails harmlessly without a console (--json).
try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { }
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
[DllImport("kernel32.dll")]
public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll")]
public static extern uint GetFileType(IntPtr hFile);
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

$CancelFile = Join-Path $cfg.temp_dir 'cancel'
# Under --json a stdin that is not a pipe (NUL: nobody is attached; or a
# console) marks an unattended run, which fails at once when something blocks.
$Unattended = $false
if ($Json) {
    try {
        # STD_INPUT_HANDLE; FILE_TYPE_PIPE = 3
        $Unattended = [EvoSciUpdate.Win32]::GetFileType([EvoSciUpdate.Win32]::GetStdHandle(-10)) -ne 3
    } catch { }
}

# Waits until nothing runs from the environment. False when the wait was
# cancelled: the cancel file exists, or the run is unattended.
function Wait-Free([double]$Progress) {
    $last = $null
    while ($true) {
        $script:blockers = Get-Blockers
        if ($script:blockers.Count -eq 0) { return $true }
        $key = ($script:blockers | ForEach-Object { $_.pid }) -join ','
        if ($key -ne $last) {
            $names = ($script:blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', '
            if ($Json) {
                Emit 'running' "Update pending until you close: $names" @{ waiting_for = $script:blockers; cancel_file = $CancelFile } $null $Progress
            } else {
                Emit 'running' "Update pending until you close: $names (Ctrl+C here cancels)" $null $null $Progress
            }
            $last = $key
        }
        if ($Unattended -or (Test-Path -LiteralPath $CancelFile)) { return $false }
        Start-Sleep -Milliseconds 500
    }
}

function Invoke-Uv {
    $ErrorActionPreference = 'Continue'
    # stdout is for event lines: uv's output goes to stderr.
    # From the temp dir: the constraints file is passed by name (uv splits a
    # -c path at spaces).
    Push-Location -LiteralPath $cfg.temp_dir
    try {
        & $cfg.uv @($cfg.uv_args) 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
        return $LASTEXITCODE
    } finally {
        Pop-Location
    }
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

# What the helper is doing, for the terminal line of an interruption.
$phase = 'waiting'
try {
    foreach ($p in $cfg.uv_env.PSObject.Properties) { Set-Item -Path "Env:\$($p.Name)" -Value $p.Value }

    if (-not (Wait-Free 0.35)) { return }
    $phase = 'installing'
    Emit 'running' "Installing EvoScientist $($cfg.target)" $null $null 0.4
    $rc = Invoke-Uv
    if ($rc -ne 0) {
        Emit 'running' "uv failed (exit $rc); trying once more when nothing runs from the environment" $null $null 0.5
        # Not 'waiting': the failed run may have changed the environment.
        $phase = 'retry-waiting'
        if (-not (Wait-Free 0.5)) { return }
        $phase = 'installing'
        $rc = Invoke-Uv
    }
    if ($rc -ne 0) {
        Emit 'error' "uv could not install EvoScientist $($cfg.target) (exit $rc, see its output). If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
        $finished = $true
        return
    }

    $phase = 'setup'
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
        $what = if ($failed) { "the $($failed.stage) stage of EvoSci setup" } else { 'EvoSci setup' }
        Emit 'error' "EvoScientist is updated, but $what failed (see above). Run it again: $($cfg.evosci) setup" $failed 'setup_failed'
        $finished = $true
        return
    }
    Emit 'done' "Updated EvoScientist $($cfg.installed) -> $($cfg.target)" @{ version = $cfg.target }
    $finished = $true
} catch {
    $reason = $_.Exception.Message
    if ($phase -eq 'setup') {
        Emit 'error' "EvoScientist $($cfg.target) is installed, but EvoSci setup could not run ($reason). Run it again: $($cfg.evosci) setup" @{ stage = $null; code = $null } 'setup_failed'
    } elseif ($phase -eq 'installing' -or $phase -eq 'retry-waiting') {
        Emit 'error' "The install failed ($reason). If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
    } else {
        Emit 'error' "The update helper failed before installing ($reason). Nothing was changed." $null 'install_failed'
    }
    $finished = $true
} finally {
    if (-not $finished) {
        if ($phase -eq 'waiting') {
            # The wait was cancelled: Ctrl+C, the cancel file, or nobody attached.
            $names = ($blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', '
            Emit 'error' "Update cancelled while these still ran: $names. Nothing was changed." @{ waiting_for = $blockers } 'environment_busy'
        } elseif ($phase -eq 'retry-waiting') {
            # Cancelled before the retry: the failed uv run may have left the
            # environment broken, so this is not "nothing was changed".
            $names = ($blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', '
            Emit 'error' "Update cancelled after a failed install, while these still ran: $names. If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" @{ waiting_for = $blockers } 'install_failed'
        } elseif ($phase -eq 'setup') {
            # Ctrl+C while EvoSci setup ran.
            Emit 'error' "EvoScientist $($cfg.target) is installed, but EvoSci setup was interrupted. Run it again: $($cfg.evosci) setup" @{ stage = $null; code = $null } 'setup_failed'
        } else {
            # Ctrl+C while uv ran.
            Emit 'error' "The install was interrupted. If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
        }
    }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath $cfg.temp_dir
    # The helper's own window closes when it exits. Without a visible
    # desktop (an SSH session, a scheduled task) nobody could answer.
    if (-not $Json -and [Environment]::UserInteractive) {
        try { Read-Host 'Press Enter to close this window' | Out-Null } catch { }
    }
}
"""


def powershell() -> str:
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")


def start_helper(temp_dir: Path, config: dict[str, Any], *, json_mode: bool) -> None:
    """Write the helper and its config into *temp_dir* (the helper deletes it
    when it ends) and start it. Under ``--json`` it gets this process's stdin
    and stdout in a new process group without a console window, so a caller
    reading the pipe gets its lines after EvoSci exits; in a terminal it opens
    a console window of its own, where Ctrl+C reaches it."""
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
        subprocess.Popen(
            cmd, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        )
