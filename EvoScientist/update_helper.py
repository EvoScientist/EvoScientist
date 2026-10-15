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
writes the update's terminal line. Whatever fails, from reading its config
on, ends with exactly one terminal line.

One helper runs at a time: a second update while one waits or installs ends
with ``environment_busy`` (an empty ``detail.waiting_for``) and changes
nothing.

While it waits it prints ``running`` lines ("Update pending until you
close: ...", ``detail.waiting_for``), a new one whenever that set changes.
Cancelling the wait changes nothing and ends the update with
``environment_busy``:

- In a terminal the helper runs in a window of its own: Ctrl+C there cancels.
  The window the update was started from has its prompt back by then, and a
  prompt (PSReadLine) reads Ctrl+C as a key, so no Ctrl+C event would reach
  a helper sharing it. Without a visible desktop (an SSH session) nobody
  would see that window: the helper then writes its lines to the caller's
  console and runs unattended.
- Under ``--json`` the waiting lines carry ``detail.cancel_file``: the caller
  creates that file to cancel. Closing stdin cannot be the signal: Node
  destroys a child's stdin when the child (``EvoSci``) exits, which is right
  after the handover. A stdin that is not a pipe (``NUL``, or a console:
  ``--json`` is for programs) marks an unattended run, which fails at once
  instead of waiting. The first wait also ends once the program reading the
  helper's stdout has gone (a zero-byte write to the pipe fails), so a
  crashed caller does not leave an update to run hours later unwatched.
- A wait cancelled after a failed uv run (before the retry) ends with
  ``install_failed``, not ``environment_busy``: that uv run may have left the
  environment broken. That wait goes on when the reader has gone.

When ``Add-Type`` fails (a TEMP its compiler cannot write to: a character
outside the ANSI code page), the helper cannot tell an unattended run or a
gone reader, and waits for the cancel file as an attended run.
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
param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$TempDir,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
# uv and EvoSci write UTF-8: decode their output, and write this window's
# lines, as UTF-8 rather than the console's code page (paths with non-ASCII
# characters). Fails harmlessly without a console (--json).
try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { }
$cfg = $null
$finished = $false
$blockers = @()
$lock = $null
$Native = $false
$Unattended = $false
$WatchReader = $false
$ReaderGone = $false
$CancelFile = Join-Path $TempDir 'cancel'

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

# Expands 8.3 short names (C:\Users\JOHNSM~1\...), which Get-Process and
# %TEMP% can return, so both sides of a comparison match. A folder search
# matches short names too and returns the long one; no Add-Type, whose
# compiler fails when %TEMP% has a character outside the ANSI code page.
function Get-LongPath([string]$Path) {
    try { $full = [IO.Path]::GetFullPath($Path) } catch { return $Path }
    # Only an 8.3 name has a ~ (it runs for every process on every poll).
    if ($full.IndexOf('~') -lt 0) { return $full }
    $long = [IO.Path]::GetPathRoot($full)
    foreach ($part in $full.Substring($long.Length).Split([char[]]'\/', [StringSplitOptions]::RemoveEmptyEntries)) {
        $found = @()
        try { if ([IO.Directory]::Exists($long)) { $found = @([IO.Directory]::GetFileSystemEntries($long, $part)) } } catch { }
        $long = if ($found.Count -eq 1) { $found[0] } else { [IO.Path]::Combine($long, $part) }
    }
    return $long
}

function Get-Blockers {
    # The leading comma keeps a one-element result an array (a JSON list).
    return , @(Get-Process | ForEach-Object {
            $p = $null
            try { $p = $_.Path } catch { }
            if ($p -and $_.Id -ne $PID) {
                $long = Get-LongPath $p
                if ($long.StartsWith($script:Root, [StringComparison]::OrdinalIgnoreCase)) {
                    [ordered]@{ pid = $_.Id; name = $_.ProcessName; path = $long }
                }
            }
        })
}

# True once the program reading this helper's stdout pipe has exited: a
# zero-byte write then fails (ERROR_NO_DATA). [Console]::Out cannot tell, as
# Windows PowerShell 5.1 turns a write to a broken pipe into success.
function Test-ReaderGone {
    if (-not $script:WatchReader) { return $false }
    [uint32]$written = 0
    return -not [EvoSciUpdate.Win32]::WriteFile([EvoSciUpdate.Win32]::GetStdHandle(-11), (New-Object byte[] 1), 0, [ref]$written, [IntPtr]::Zero)
}

# Waits until nothing runs from the environment. False when the wait was
# cancelled: the cancel file exists, the run is unattended, or (-Reader) the
# program reading stdout has gone.
function Wait-Free([double]$Progress, [switch]$Reader) {
    $last = $null
    while ($true) {
        $script:blockers = Get-Blockers
        if ($script:blockers.Count -eq 0) { return $true }
        $key = ($script:blockers | ForEach-Object { $_.pid }) -join ','
        if ($key -ne $last) {
            $names = ($script:blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', '
            if ($Json) {
                Emit 'running' "Update pending until you close: $names" @{ waiting_for = $script:blockers; cancel_file = $CancelFile } $null $Progress
            } elseif ($Unattended) {
                Emit 'running' "Update pending until you close: $names" $null $null $Progress
            } else {
                Emit 'running' "Update pending until you close: $names (Ctrl+C here cancels)" $null $null $Progress
            }
            $last = $key
        }
        if ($Unattended -or (Test-Path -LiteralPath $CancelFile)) { return $false }
        if ($Reader -and (Test-ReaderGone)) {
            $script:ReaderGone = $true
            return $false
        }
        Start-Sleep -Milliseconds 500
    }
}

function Invoke-Uv {
    $ErrorActionPreference = 'Continue'
    # stdout is for event lines: uv's output goes to stderr.
    # From the temp dir: the constraints file is passed by name (uv splits a
    # -c path at spaces).
    Push-Location -LiteralPath $TempDir
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

function Get-Names { return ($blockers | ForEach-Object { "$($_.name).exe ($($_.pid))" }) -join ', ' }

# What the helper is doing, for the terminal line of an interruption.
$phase = 'starting'
try {
    $cfg = Get-Content -Raw -Encoding UTF8 -LiteralPath $ConfigPath | ConvertFrom-Json
    $sep = [IO.Path]::DirectorySeparatorChar
    $Root = (Get-LongPath $cfg.env_dir).TrimEnd($sep) + $sep

    try {
        if (-not ('EvoSciUpdate.Win32' -as [type])) {
            Add-Type -Namespace EvoSciUpdate -Name Win32 -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll")]
public static extern uint GetFileType(IntPtr hFile);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool WriteFile(IntPtr hFile, byte[] buffer, uint count, out uint written, IntPtr overlapped);
'@
        }
        $Native = $true
    } catch { }
    $Unattended = [bool]$cfg.unattended
    if ($Json -and $Native) {
        try {
            # FILE_TYPE_PIPE = 3. A stdin that is not a pipe (NUL: nobody is
            # attached; or a console) marks an unattended run, which fails at
            # once when something blocks.
            $Unattended = [EvoSciUpdate.Win32]::GetFileType([EvoSciUpdate.Win32]::GetStdHandle(-10)) -ne 3
            $WatchReader = [EvoSciUpdate.Win32]::GetFileType([EvoSciUpdate.Win32]::GetStdHandle(-11)) -eq 3
        } catch { }
    }

    # One helper at a time, whichever window started it: two would race uv
    # on the same environment.
    try {
        $lock = [IO.File]::Open($cfg.lock, 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.DirectoryNotFoundException] {
        throw
    } catch [IO.IOException] {
        Emit 'error' 'Another EvoSci update is already waiting or installing. Nothing was changed.' @{ waiting_for = @() } 'environment_busy'
        $finished = $true
        return
    }

    $phase = 'waiting'
    foreach ($p in $cfg.uv_env.PSObject.Properties) { Set-Item -Path "Env:\$($p.Name)" -Value $p.Value }

    # EvoSci, which started this helper, is still exiting: wait for it (at
    # most 30 s) before looking for processes that block the environment, or
    # an unattended run would end at once on its own caller.
    $deadline = (Get-Date).AddSeconds(30)
    foreach ($exiting in @($cfg.wait_pids)) {
        $left = [int]($deadline - (Get-Date)).TotalMilliseconds
        if ($left -le 0) { break }
        try { [void](Get-Process -Id $exiting -ErrorAction Stop).WaitForExit($left) } catch { }
    }

    if (-not (Wait-Free 0.35 -Reader)) { return }
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
        Emit 'error' "uv could not install EvoScientist $($cfg.target) (exit $rc, see its output).$($cfg.install_hint) If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
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
            try { $ev = $line | ConvertFrom-Json } catch { $ev = $null }
            if ($ev -and $null -ne $ev.protocol) {
                [Console]::Out.WriteLine($line)
                [Console]::Out.Flush()
            } else {
                # Not an event line (a stray print): stdout is for JSON only.
                [Console]::Error.WriteLine($line)
                $ev = $null
            }
            if ($ev -and $ev.status -eq 'error' -and $null -eq $failed) {
                $failed = [ordered]@{ stage = $ev.stage; code = $ev.code }
            }
        }
        $setupRc = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
    } elseif ($Unattended) {
        & $cfg.evosci @setupArgs
        $setupRc = $LASTEXITCODE
    } else {
        $setupRc = Invoke-OnConsole $cfg.evosci $setupArgs
    }
    if ($null -ne $failed -or $setupRc -ne 0) {
        $what = if ($failed) { "the $($failed.stage) stage of EvoSci setup" } else { 'EvoSci setup' }
        Emit 'error' "EvoScientist is updated, but $what failed (see above). Run it again: $($cfg.setup_command)" $failed 'setup_failed'
        $finished = $true
        return
    }
    Emit 'done' "Updated EvoScientist $($cfg.installed) -> $($cfg.target)" @{ version = $cfg.target }
    $finished = $true
} catch {
    $reason = $_.Exception.Message
    if ($phase -eq 'setup') {
        Emit 'error' "EvoScientist $($cfg.target) is installed, but EvoSci setup could not run ($reason). Run it again: $($cfg.setup_command)" @{ stage = $null; code = $null } 'setup_failed'
    } elseif ($phase -eq 'installing' -or $phase -eq 'retry-waiting') {
        Emit 'error' "The install failed ($reason). If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
    } else {
        Emit 'error' "The update helper failed before installing ($reason). Nothing was changed." $null 'install_failed'
    }
    $finished = $true
} finally {
    if (-not $finished) {
        if ($phase -eq 'waiting') {
            # The wait was cancelled: Ctrl+C, the cancel file, nobody
            # attached, or the program reading stdout has gone.
            $why = if ($ReaderGone) { 'the program that started it has exited' } else { 'cancelled' }
            if ($blockers.Count -eq 0) {
                # Before the first look (while the caller was still exiting).
                Emit 'error' "Update $why. Nothing was changed." @{ waiting_for = @() } 'environment_busy'
            } else {
                Emit 'error' "Update $why while these still ran: $(Get-Names). Nothing was changed." @{ waiting_for = $blockers } 'environment_busy'
            }
        } elseif ($phase -eq 'retry-waiting') {
            # Cancelled before the retry: the failed uv run may have left the
            # environment broken, so this is not "nothing was changed".
            Emit 'error' "Update cancelled after a failed install, while these still ran: $(Get-Names). If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" @{ waiting_for = $blockers } 'install_failed'
        } elseif ($phase -eq 'setup') {
            # Ctrl+C while EvoSci setup ran.
            Emit 'error' "EvoScientist $($cfg.target) is installed, but EvoSci setup was interrupted. Run it again: $($cfg.setup_command)" @{ stage = $null; code = $null } 'setup_failed'
        } elseif ($phase -eq 'starting') {
            Emit 'error' 'The update helper stopped before installing. Nothing was changed.' $null 'install_failed'
        } else {
            # Ctrl+C while uv ran.
            Emit 'error' "The install was interrupted. If EvoSci no longer starts, run the install script again once nothing runs from it: $($cfg.install_command)" $null 'install_failed'
        }
    }
    if ($lock) { $lock.Dispose() }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath $TempDir
    # The helper's own window closes when it exits. Without a visible
    # desktop (an SSH session, a scheduled task) nobody could answer.
    if (-not $Json -and -not $Unattended -and [Environment]::UserInteractive) {
        try { Read-Host 'Press Enter to close this window' | Out-Null } catch { }
    }
}
"""

# Runs the helper as a script block, not with -File: an execution policy set
# by Group Policy (AllSigned) overrides -ExecutionPolicy Bypass and refuses an
# unsigned script file, but does not apply to a script block. Dot-sourced, so
# its $script: variables are the session's, as they would be in a script
# file. The paths come through the environment, so nothing needs quoting.
_START = (
    ". ([scriptblock]::Create([IO.File]::ReadAllText($env:EVOSCI_UPDATE_HELPER))) "
    "-ConfigPath $env:EVOSCI_UPDATE_CONFIG -TempDir $env:EVOSCI_UPDATE_TEMP"
)


def powershell() -> str:
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")


def has_visible_desktop() -> bool:
    """Whether this process's window station is visible, as
    ``[Environment]::UserInteractive`` tells: not in an SSH session or a
    service. True when it cannot tell."""
    try:
        import ctypes
        from ctypes import wintypes

        class _Flags(ctypes.Structure):
            _fields_ = [
                ("fInherit", wintypes.BOOL),
                ("fReserved", wintypes.BOOL),
                ("dwFlags", wintypes.DWORD),
            ]

        user32 = ctypes.WinDLL("user32")
        user32.GetProcessWindowStation.restype = wintypes.HANDLE
        station = user32.GetProcessWindowStation()
        flags = _Flags()
        needed = wintypes.DWORD()
        uoi_flags, wsf_visible = 1, 1
        if not station or not user32.GetUserObjectInformationW(
            station,
            uoi_flags,
            ctypes.byref(flags),
            ctypes.sizeof(flags),
            ctypes.byref(needed),
        ):
            return True
        return bool(flags.dwFlags & wsf_visible)
    except (AttributeError, OSError):
        return True


def start_helper(
    temp_dir: Path, config: dict[str, Any], *, json_mode: bool, own_window: bool
) -> None:
    """Write the helper and its config into *temp_dir* (the helper deletes it
    when it ends) and start it. Under ``--json``, or without *own_window*
    (no visible desktop: ``config["unattended"]``), it gets this process's
    stdin and stdout in a new process group without a console window, so the
    caller gets its lines after EvoSci exits; otherwise it opens a console
    window of its own, where Ctrl+C reaches it."""
    script = temp_dir / HELPER_NAME
    script.write_text(HELPER, encoding="utf-8-sig")
    config_path = temp_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cmd = [
        powershell(),
        "-NoProfile",
        "-Command",
        _START + (" -Json" if json_mode else ""),
    ]
    env = {
        **os.environ,
        "EVOSCI_UPDATE_HELPER": str(script),
        "EVOSCI_UPDATE_CONFIG": str(config_path),
        "EVOSCI_UPDATE_TEMP": str(temp_dir),
    }
    if json_mode or not own_window:
        # Not DETACHED_PROCESS: a helper started that way never ran.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        subprocess.Popen(
            cmd, creationflags=flags, stdin=sys.stdin, stdout=sys.stdout, env=env
        )
    else:
        subprocess.Popen(
            cmd, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0), env=env
        )
