# Install EvoScientist on Windows with nothing else installed.
#
#   irm https://raw.githubusercontent.com/EvoScientist/EvoScientist/main/scripts/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@main/scripts/install.ps1))) -Cn
#
# Options (irm | iex cannot pass any; use the scriptblock form):
#   -Cn             use mainland China mirrors (uv, Python, PyPI, Node, WebUI)
#   -Version <ver>  install this release instead of the newest one
#   -NoModifyPath   do not put uv's tool bin directory on the user PATH
#   -NoSetup        stop after installing the package, before `EvoSci setup`
#
# Steps: download a pinned uv and check its SHA-256, keep it under
# <data dir>\tools, install EvoScientist as a uv tool with the release's
# constraints.txt on uv's own Python (3.12, or on a re-run the one the existing
# install asked for), put the tool bin directory on the user PATH, then run `EvoSci setup`. The data dir is $env:EVOSCIENTIST_DATA_DIR
# when set and not empty, else ~\.evoscientist. A uv installed by the user is
# never touched.
#
# Test-only variables (the installer smoke workflow): EVOSCI_INSTALL_SOURCE
# installs this wheel instead of the release from the index, and
# EVOSCI_INSTALL_CONSTRAINTS uses this local constraints file instead of the
# one at the release tag.
#
# Under irm | iex this code runs in the caller's session: everything lives in
# one function, environment, TLS and proxy settings are restored at the end, so
# nothing stays behind; errors are thrown, never `exit`, which would close the
# window.

# Refuses an unknown or misspelled option instead of ignoring it.
[CmdletBinding()]
param(
    [switch]$Cn,
    [string]$Version = '',
    [switch]$NoModifyPath,
    [switch]$NoSetup
)

function Install-EvoScientist {
    param([bool]$Cn, [string]$Version, [bool]$NoModifyPath, [bool]$NoSetup)

    $ErrorActionPreference = 'Stop'
    # Invoke-WebRequest's progress bar slows downloads in Windows PowerShell 5.1.
    $ProgressPreference = 'SilentlyContinue'

    $UvVersion = '0.12.23'
    $UvSha256 = '75d05de6762778c31ee183398de7dd15093fad0ed90b1f236d8205ea5ec00c90'
    $UvWheelPath = 'packages/af/fd/816f4e8a76197312ec8037369e5b5c8a5ebde7aa118b6250259e031d3444/uv-0.12.23-py3-none-win_amd64.whl'
    $UvWheelSha256 = 'fb8a4117a5224d73abe2204a1e744ae34b544f1b9a0b761576851deaa7215c12'
    # The last release that has no constraints.txt at its tag; it and every
    # older release are refused.
    $LastUnsupported = '0.3.5'
    $PythonVersion = '3.12'
    $PypiSimple = 'https://pypi.org/simple'
    $CnSimple = 'https://pypi.tuna.tsinghua.edu.cn/simple'
    $CnPythonMirror = 'https://registry.npmmirror.com/-/binary/python-build-standalone/'
    $RawBase = 'https://raw.githubusercontent.com/EvoScientist/EvoScientist'
    $CnRawBase = 'https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist'
    $EntryPoints = @('EvoSci.exe', 'EvoScientist.exe', 'evosci.exe', 'evoscientist.exe')

    # Reads an existing uv-receipt.toml (argv[1]) with the old tool
    # environment's own Python and prints what a reinstall needs to keep, one
    # tab-separated line each: "python<TAB><request>" for the receipt's Python
    # (else this interpreter's X.Y), "extras<TAB>a,b" for EvoScientist's
    # extras, "with<TAB><requirement>" and "with-editable<TAB><path>" for the
    # other requirements (added with --with, e.g. by the MCP installer),
    # "warn<TAB><text>" for anything it cannot keep. Same code as install.sh;
    # tests/test_install_scripts.py runs both on the same receipts.
    $ReceiptReader = @'
import re, sys, tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def git_spec(url):
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    if "path" in query:
        raise ValueError("git path source")
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    ref = next((query[k][0] for k in ("branch", "tag", "rev") if k in query), parts.fragment)
    spec = "git+" + base + ("@" + ref if ref else "")
    if "subdirectory" in query:
        spec += "#subdirectory=" + query["subdirectory"][0]
    return spec

def requirement(entry):
    name = entry["name"]
    extras = entry.get("extras") or []
    head = name + ("[" + ",".join(extras) + "]" if extras else "")
    marker = entry.get("marker")
    tail = " ; " + marker if marker else ""
    if "editable" in entry:
        return "with-editable", entry["editable"]
    if "git" in entry:
        return "with", head + " @ " + git_spec(entry["git"]) + tail
    if "url" in entry:
        url = entry["url"]
        if entry.get("subdirectory"):
            url += "#subdirectory=" + entry["subdirectory"]
        return "with", head + " @ " + url + tail
    for key in ("path", "directory"):
        if key in entry:
            return "with", head + " @ " + Path(entry[key]).as_uri() + tail
    if "virtual" in entry:
        raise ValueError("virtual source")
    return "with", head + entry.get("specifier", "") + tail

try:
    tool = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["tool"]
    entries = tool["requirements"]
except Exception as exc:
    print("warn\tcannot read " + sys.argv[1] + ": " + str(exc))
    sys.exit(0)
python = tool.get("python")
print("python\t" + (python if isinstance(python, str) else "%d.%d" % sys.version_info[:2]))
own_extras = []
for entry in entries:
    try:
        if norm(entry["name"]) == "evoscientist":
            # Several entries when an extra was added with --with: all extras.
            own_extras += [x for x in entry.get("extras") or [] if x not in own_extras]
            continue
        kind, value = requirement(entry)
    except Exception as exc:
        print("warn\tcannot keep " + repr(entry.get("name", entry)) + " (" + str(exc) + ")")
        continue
    print(kind + "\t" + value)
if own_extras:
    print("extras\t" + ",".join(own_extras))
'@

    function Write-Info([string]$Text) { Write-Host $Text }
    function Write-Warn([string]$Text) { Write-Host "Warning: $Text" -ForegroundColor Yellow }

    # The dotted release numbers of a version, as [version]; a pre-release such
    # as 0.3.5rc1 counts as 0.3.5.
    function Get-ReleaseVersion([string]$Text) {
        $release = ([regex]::Match($Text, '^\d+(\.\d+)*')).Value
        $parts = @($release.Split('.') | Select-Object -First 4)
        while ($parts.Count -lt 2) { $parts += '0' }
        return [version]($parts -join '.')
    }

    # Expands 8.3 short names (C:\Users\JOHNSM~1\...), which Get-Process and
    # %TEMP% can return, so paths compare equal. A folder search matches short
    # names too and returns the long one. Not GetLongPathName through Add-Type:
    # its compiler fails when %TEMP% has a character outside the ANSI code page.
    function Get-LongPath([string]$Path) {
        try { $full = [IO.Path]::GetFullPath($Path) } catch { return $Path.TrimEnd('\') }
        # Only an 8.3 name has a ~.
        if ($full.IndexOf('~') -lt 0) { return $full.TrimEnd('\') }
        $long = [IO.Path]::GetPathRoot($full)
        foreach ($part in $full.Substring($long.Length).Split([char[]]'\', [StringSplitOptions]::RemoveEmptyEntries)) {
            $found = @()
            try { if ([IO.Directory]::Exists($long)) { $found = @([IO.Directory]::GetFileSystemEntries($long, $part)) } } catch { }
            $long = if ($found.Count -eq 1) { $found[0] } else { [IO.Path]::Combine($long, $part) }
        }
        return $long.TrimEnd('\')
    }

    function Test-SamePath([string]$A, [string]$B) {
        return [string]::Equals((Get-LongPath $A), (Get-LongPath $B), [StringComparison]::OrdinalIgnoreCase)
    }

    # Runs a native command and returns its output lines (stdout, and stderr
    # with -WithStderr) and exit code. Windows PowerShell decodes captured
    # native output with the console's code page, while uv and Python with
    # -X utf8 write UTF-8: a path with non-ASCII characters would come back
    # garbled, so the console encoding is UTF-8 during the call.
    function Invoke-Captured([string]$Exe, [string[]]$Arguments, [switch]$WithStderr) {
        $ErrorActionPreference = 'Continue'
        $saved = [Console]::OutputEncoding
        try {
            try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { }
            if ($WithStderr) {
                $lines = @(& $Exe @Arguments 2>&1 | ForEach-Object { "$_" })
            } else {
                $lines = @(& $Exe @Arguments | ForEach-Object { "$_" })
            }
            return @{ Lines = $lines; ExitCode = $LASTEXITCODE }
        } finally {
            try { [Console]::OutputEncoding = $saved } catch { }
        }
    }

    # Runs a native command on this console (its output is not read by
    # PowerShell, so EvoSci's progress output keeps the console's encoding)
    # and returns its exit code.
    function Invoke-OnConsole([string]$Exe, [string[]]$Arguments, [string]$WorkingDirectory) {
        $quoted = foreach ($arg in $Arguments) {
            if ($arg -and $arg -notmatch '[\s"]') {
                $arg
            } else {
                # The quoting rules of CommandLineToArgvW.
                $escaped = [regex]::Replace($arg, '(\\*)"', '$1$1\"')
                '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
            }
        }
        $info = New-Object Diagnostics.ProcessStartInfo $Exe
        $info.Arguments = $quoted -join ' '
        $info.UseShellExecute = $false
        if ($WorkingDirectory) { $info.WorkingDirectory = $WorkingDirectory }
        $process = [Diagnostics.Process]::Start($info)
        $process.WaitForExit()
        return $process.ExitCode
    }

    # Not -OutFile, which Windows PowerShell 5.1 resolves as a wildcard path:
    # a profile folder with [ or ] in its name would fail.
    function Get-Download([string]$Url, [string]$OutFile) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url
            [IO.File]::WriteAllBytes($OutFile, $response.RawContentStream.ToArray())
        } catch {
            throw "Download failed: $Url ($($_.Exception.Message))"
        }
    }

    function Assert-Platform {
        if (-not [Environment]::Is64BitOperatingSystem) {
            throw 'This is 32-bit Windows. EvoScientist supports 64-bit Windows on x64.'
        }
        # The machine's architecture, not this process's: an x64 PowerShell runs
        # under emulation on Windows on Arm and reports x64 for itself.
        $arch = $null
        try {
            $assembly = [System.Reflection.Assembly]::LoadWithPartialName('System.Runtime.InteropServices.RuntimeInformation')
            $type = $assembly.GetType('System.Runtime.InteropServices.RuntimeInformation')
            $arch = $type.GetProperty('OSArchitecture').GetValue($null).ToString()
        } catch { }
        $native = $null
        try {
            $native = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -Name PROCESSOR_ARCHITECTURE).PROCESSOR_ARCHITECTURE
        } catch { }
        if ($arch -eq 'Arm64' -or $native -eq 'ARM64') {
            throw 'Windows on arm64 is not supported. EvoScientist supports Windows on x64.'
        }
    }

    # TLS 1.2 for Windows PowerShell 5.1 on older .NET; leaves .NET's
    # SystemDefault alone, which already negotiates it.
    function Enable-Tls12 {
        $protocol = [Net.ServicePointManager]::SecurityProtocol
        $names = [Enum]::GetNames([Net.SecurityProtocolType])
        if (-not ($names -contains 'SystemDefault' -and $protocol -eq [Net.SecurityProtocolType]::SystemDefault)) {
            [Net.ServicePointManager]::SecurityProtocol = $protocol -bor [Net.SecurityProtocolType]::Tls12
        }
        # A system proxy that asks for the Windows login.
        try {
            [Net.WebRequest]::DefaultWebProxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials
        } catch { }
    }

    # The data dir rule of EvoScientist's paths module: EVOSCIENTIST_DATA_DIR
    # when set and not empty (with ~ expanded), else ~\.evoscientist; made
    # absolute. The home folder as Python finds it: USERPROFILE first, while
    # Windows PowerShell 5.1 sets $HOME from HOMEDRIVE + HOMEPATH, which point
    # at a network home folder on some domain accounts.
    function Get-DataDir {
        $userHome = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
        $dir = $env:EVOSCIENTIST_DATA_DIR
        if (-not $dir) { $dir = Join-Path $userHome '.evoscientist' }
        if ($dir -eq '~') {
            $dir = $userHome
        } elseif ($dir.StartsWith('~\') -or $dir.StartsWith('~/')) {
            $dir = Join-Path $userHome $dir.Substring(2)
        }
        if (-not [IO.Path]::IsPathRooted($dir)) {
            $dir = Join-Path (Get-Location).ProviderPath $dir
        }
        New-Item -ItemType Directory -Force -Path (Join-Path $dir 'tools') | Out-Null
        return Get-LongPath $dir
    }

    function Get-InstallVersion {
        if ($Version) {
            $v = $Version.TrimStart('v')
            if ($v -notmatch '^\d+(\.\d+)*((a|b|rc)\d+)?$') {
                throw "-Version expects a release number such as 0.3.6, got: $Version"
            }
            return $v
        }
        if ($env:EVOSCI_INSTALL_SOURCE) {
            $name = Split-Path -Leaf $env:EVOSCI_INSTALL_SOURCE
            if ($name -match '^evoscientist-([^-]+)-.*\.whl$') { return $Matches[1] }
            throw "Cannot read the version from EVOSCI_INSTALL_SOURCE: $env:EVOSCI_INSTALL_SOURCE"
        }
        $index = if ($Cn) { $CnSimple } else { $PypiSimple }
        try {
            $listing = Invoke-RestMethod -UseBasicParsing -Uri "$index/evoscientist/" -Headers @{ Accept = 'application/vnd.pypi.simple.v1+json' }
        } catch {
            throw "Cannot read the EvoScientist versions from $index/evoscientist/ ($($_.Exception.Message))"
        }
        # The mirror sorts the list as strings, so filter to final releases and
        # sort numerically; never take the last entry as it comes.
        $latest = @($listing.versions | Where-Object { $_ -match '^\d+(\.\d+){1,3}$' } |
                Sort-Object { [version]$_ }) | Select-Object -Last 1
        if (-not $latest) { throw "No EvoScientist release found on $index" }
        return $latest
    }

    # Downloads the pinned uv, checks its hash and keeps it in
    # <data dir>\tools\uv-<version>\; records it in tools\uv.json.
    function Install-Uv([string]$DataDir, [string]$TempDir) {
        $tools = Join-Path $DataDir 'tools'
        $final = Join-Path $tools "uv-$UvVersion"
        $staged = Join-Path $TempDir 'uv-bin'
        if ($Cn) {
            # Expand-Archive accepts only a .zip name; a wheel is a zip.
            $archive = Join-Path $TempDir 'uv.zip'
            $expected = $UvWheelSha256
            Write-Info "Downloading uv $UvVersion from the Tsinghua mirror..."
            Get-Download "https://pypi.tuna.tsinghua.edu.cn/$UvWheelPath" $archive
        } else {
            $archive = Join-Path $TempDir 'uv.zip'
            $expected = $UvSha256
            Write-Info "Downloading uv $UvVersion..."
            Get-Download "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $archive
        }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLower()
        if ($actual -ne $expected) {
            throw "The uv download does not match its pinned SHA-256 (expected $expected, got $actual). Nothing was installed."
        }
        $unpacked = Join-Path $TempDir 'uv-unpacked'
        Expand-Archive -LiteralPath $archive -DestinationPath $unpacked -Force
        $exe = if ($Cn) { Join-Path $unpacked "uv-$UvVersion.data\scripts\uv.exe" } else { Join-Path $unpacked 'uv.exe' }
        New-Item -ItemType Directory -Force -Path $staged | Out-Null
        Move-Item -LiteralPath $exe -Destination (Join-Path $staged 'uv.exe')
        if (Test-Path -LiteralPath $final) { Remove-Item -LiteralPath $final -Recurse -Force }
        Move-Item -LiteralPath $staged -Destination $final
        $uv = Join-Path $final 'uv.exe'
        # UTF-8 without a BOM: EvoScientist reads it with json.loads.
        $json = [ordered]@{ version = $UvVersion; path = $uv } | ConvertTo-Json -Compress
        $record = Join-Path $tools 'uv.json'
        [IO.File]::WriteAllText("$record.tmp", $json, (New-Object Text.UTF8Encoding $false))
        Move-Item -Force -LiteralPath "$record.tmp" -Destination $record
        return $uv
    }

    function Get-Constraints([string]$InstallVersion, [string]$TempDir) {
        $file = Join-Path $TempDir 'constraints.txt'
        if ($env:EVOSCI_INSTALL_CONSTRAINTS) {
            Copy-Item -LiteralPath $env:EVOSCI_INSTALL_CONSTRAINTS -Destination $file
            return $file
        }
        $base = if ($Cn) { "$CnRawBase@v$InstallVersion" } else { "$RawBase/v$InstallVersion" }
        Write-Info "Downloading the constraints of EvoScientist $InstallVersion..."
        try {
            Get-Download "$base/constraints.txt" $file
        } catch {
            throw "Cannot download $base/constraints.txt, so EvoScientist $InstallVersion cannot be installed with its tested dependencies."
        }
        return $file
    }

    # Processes started from the tool environment (EvoSci.exe, the server, MCP
    # servers, the env's python.exe) or from its entry points in the bin dir.
    # uv cannot replace an environment that a running process uses, and a
    # failed replace leaves it broken.
    function Get-BlockingProcesses([string]$EnvDir, [string]$BinDir) {
        $root = (Get-LongPath $EnvDir) + '\'
        $bin = Get-LongPath $BinDir
        return @(Get-Process | ForEach-Object {
                $path = $null
                try { $path = $_.Path } catch { }
                if ($path) {
                    $long = Get-LongPath $path
                    $inEnv = $long.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)
                    $isEntry = (Test-SamePath (Split-Path -Parent $long) $bin) -and ($EntryPoints -contains (Split-Path -Leaf $long))
                    if ($inEnv -or $isEntry) { "$($_.ProcessName) (PID $($_.Id)): $long" }
                }
            })
    }

    # Names the uv config that sets exclude-newer, which blocks releases newer
    # than its cutoff. uv prints its own hint; this says where it comes from.
    function Write-ExcludeNewerHint {
        if ($env:UV_EXCLUDE_NEWER) {
            Write-Warn "UV_EXCLUDE_NEWER is set ($env:UV_EXCLUDE_NEWER): uv ignores releases uploaded after that date."
        }
        if ($env:UV_NO_CONFIG) { return }
        $files = @($env:UV_CONFIG_FILE)
        if ($env:APPDATA) { $files += Join-Path $env:APPDATA 'uv\uv.toml' }
        if ($env:ProgramData) { $files += Join-Path $env:ProgramData 'uv\uv.toml' }
        foreach ($file in $files) {
            if ($file -and (Test-Path -LiteralPath $file) -and (Select-String -LiteralPath $file -Pattern '^\s*exclude-newer\s*=' -Quiet)) {
                Write-Warn "$file sets exclude-newer: uv ignores releases uploaded after its cutoff, so a newer EvoScientist cannot install. Remove that line, or wait until the release is older than the cutoff."
            }
        }
    }

    Assert-Platform

    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ("evosci-install-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    # Environment, TLS and proxy changes are undone at the end, so the caller's
    # session keeps its own values under irm | iex.
    $envNames = @('UV_SYSTEM_CERTS', 'UV_HTTP_TIMEOUT', 'UV_DEFAULT_INDEX', 'UV_PYTHON_INSTALL_MIRROR', 'PYTHONIOENCODING')
    $savedEnv = @{}
    foreach ($name in $envNames) { $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    $savedProtocol = [Net.ServicePointManager]::SecurityProtocol
    $savedCredentials = $null
    try { $savedCredentials = [Net.WebRequest]::DefaultWebProxy.Credentials } catch { }
    try {
        Enable-Tls12
        $installVersion = Get-InstallVersion
        if ((Get-ReleaseVersion $installVersion) -le (Get-ReleaseVersion $LastUnsupported)) {
            throw "EvoScientist $installVersion cannot be installed with this script: $LastUnsupported is the last release without constraints.txt, and every older release lacks it too. Choose a newer version."
        }

        $dataDir = Get-DataDir
        # uv trusts the Windows certificate store, as Invoke-WebRequest did, so
        # a network that inspects TLS works; longer timeout for slow mirrors.
        if (-not $env:UV_SYSTEM_CERTS) { $env:UV_SYSTEM_CERTS = '1' }
        if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = '300' }
        # EvoSci setup writes its progress with symbols such as a check mark: to a pipe (CI,
        # the desktop app) Python would encode them in the ANSI code page and
        # fail. Its console output is unaffected.
        if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = 'utf-8' }
        if ($Cn) {
            $env:UV_DEFAULT_INDEX = $CnSimple
            $env:UV_PYTHON_INSTALL_MIRROR = $CnPythonMirror
        }

        $uv = Install-Uv $dataDir $tempDir
        $constraints = Get-Constraints $installVersion $tempDir
        $toolDir = Invoke-Captured $uv @('tool', 'dir')
        $binDir = Invoke-Captured $uv @('tool', 'dir', '--bin')
        if ($toolDir.ExitCode -ne 0 -or $binDir.ExitCode -ne 0) {
            throw "uv tool dir failed (exit codes $($toolDir.ExitCode), $($binDir.ExitCode)); see uv's output above."
        }
        $toolDir = ($toolDir.Lines -join '').Trim()
        $binDir = ($binDir.Lines -join '').Trim()
        $envDir = Join-Path $toolDir 'evoscientist'

        $extras = ''
        $withArgs = @()
        $force = Test-Path -LiteralPath $envDir
        if (-not $force) {
            # pipx and pip --user use the same bin dir; uv would refuse to
            # overwrite their command.
            foreach ($name in $EntryPoints) {
                $existing = Join-Path $binDir $name
                if (Test-Path -LiteralPath $existing) {
                    throw "$existing exists but was not installed by uv; pipx or pip --user may have put it there. Remove that install (for example pipx uninstall EvoScientist or pip uninstall EvoScientist), then run this script again."
                }
            }
        }
        if ($force) {
            $blocking = Get-BlockingProcesses $envDir $binDir
            if ($blocking.Count -gt 0) {
                throw ("EvoScientist is running, and the install cannot replace it while it runs:`n  " + ($blocking -join "`n  ") +
                    "`nClose these windows (a background server: EvoSci server stop), then run this script again. Nothing was changed.")
            }
            $receipt = Join-Path $envDir 'uv-receipt.toml'
            $python = Join-Path $envDir 'Scripts\python.exe'
            $lost = 'Extras and packages added with --with (such as MCP servers) may be dropped by the reinstall.'
            if ((Test-Path -LiteralPath $receipt) -and (Test-Path -LiteralPath $python)) {
                $reader = Join-Path $tempDir 'read_receipt.py'
                [IO.File]::WriteAllText($reader, $ReceiptReader, (New-Object Text.UTF8Encoding $false))
                $read = Invoke-Captured $python @('-X', 'utf8', '-I', $reader, $receipt) -WithStderr
                if ($read.ExitCode -ne 0) {
                    Write-Warn "Cannot read $receipt with $python. $lost"
                } else {
                    foreach ($line in $read.Lines) {
                        $kind, $value = "$line".Split("`t", 2)
                        switch ($kind) {
                            'python' { $PythonVersion = $value }
                            'extras' { $extras = "[$value]" }
                            'with' { $withArgs += @('--with', $value) }
                            'with-editable' { $withArgs += @('--with-editable', $value) }
                            'warn' { Write-Warn "From ${receipt}: $value; it is not kept by the reinstall." }
                        }
                    }
                }
            } else {
                Write-Warn "Cannot read the settings of the existing install in $envDir. $lost"
            }
        }

        $spec = "EvoScientist$extras==$installVersion"
        if ($env:EVOSCI_INSTALL_SOURCE) {
            $wheel = (Resolve-Path -LiteralPath $env:EVOSCI_INSTALL_SOURCE).ProviderPath
            $spec = "EvoScientist$extras @ " + ([Uri]$wheel).AbsoluteUri
        }
        # uv splits a -c path at spaces (astral-sh/uv#12639), such as a %TEMP%
        # under a user name with a space: uv runs from the file's folder and
        # gets its bare name.
        $uvArgs = @('tool', 'install', '--python', $PythonVersion, '--managed-python', '-c', (Split-Path -Leaf $constraints))
        if ($force) { $uvArgs += '--force' }
        $uvArgs += $withArgs
        $uvArgs += $spec
        Write-Info "Installing EvoScientist $installVersion..."
        if ((Invoke-OnConsole $uv $uvArgs (Split-Path -Parent $constraints)) -ne 0) {
            Write-ExcludeNewerHint
            throw "uv could not install EvoScientist $installVersion (see uv's output above). If EvoScientist was started meanwhile, close it and run this script again."
        }

        $pathChanged = $false
        $onPath = @($env:Path.Split(';') | Where-Object { $_ -and (Test-SamePath $_ $binDir) }).Count -gt 0
        if (-not $onPath) {
            if ($NoModifyPath) {
                Write-Info "Not changing PATH (-NoModifyPath). Add $binDir to PATH to run EvoSci."
            } else {
                # On Windows uv exits 0 also when the user PATH already has the
                # directory ("already in PATH"); any other exit is an error.
                $shell = Invoke-Captured $uv @('tool', 'update-shell') -WithStderr
                $shell.Lines | Out-Host
                if ($shell.ExitCode -eq 0) {
                    $pathChanged = $true
                } else {
                    Write-Warn "uv tool update-shell failed; add $binDir to PATH yourself."
                }
            }
        }

        if ($NoSetup) {
            Write-Info 'Skipping EvoSci setup (-NoSetup).'
        } else {
            $setupArgs = @('setup')
            if ($Cn) { $setupArgs += '--cn' }
            # By absolute path: this window does not see the PATH change yet.
            $evosci = Join-Path $binDir 'EvoSci.exe'
            if ((Invoke-OnConsole $evosci $setupArgs) -ne 0) {
                throw "EvoScientist $installVersion is installed, but EvoSci setup failed (see above). Run it again with: & '$evosci' setup"
            }
        }

        # Another EvoSci earlier on PATH (pip, conda) would run instead. Compare
        # the directories that hold the commands.
        $found = Get-Command EvoSci -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found -and -not (Test-SamePath (Split-Path -Parent $found.Source) $binDir)) {
            Write-Warn "Another EvoSci comes first on PATH: $($found.Source). It will run instead of this install; remove it (for example pip uninstall EvoScientist) or put $binDir earlier on PATH."
        }

        Write-Info ''
        Write-Info "EvoScientist $installVersion is installed."
        if ($pathChanged) {
            Write-Info "Your user PATH now includes $binDir (uv tool update-shell); open a new terminal first."
        }
        Write-Info 'Next steps:'
        Write-Info '  EvoSci onboard          set up your model provider and API keys'
        Write-Info '  EvoSci                  start the terminal UI'
        Write-Info '  EvoSci --ui webui       start the WebUI'
    } finally {
        foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process') }
        [Net.ServicePointManager]::SecurityProtocol = $savedProtocol
        try { [Net.WebRequest]::DefaultWebProxy.Credentials = $savedCredentials } catch { }
        Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

try {
    Install-EvoScientist -Cn $Cn.IsPresent -Version $Version -NoModifyPath $NoModifyPath.IsPresent -NoSetup $NoSetup.IsPresent
    # The outcome for a caller of irm | iex (CI, the desktop app), which gets
    # no exit code; the last native command may have set another value.
    $global:LASTEXITCODE = 0
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    $global:LASTEXITCODE = 1
    # Only a script run from a file may exit: under irm | iex or the
    # scriptblock form, exit would close the caller's window.
    if ($MyInvocation.MyCommand.Path) { exit 1 }
} finally {
    Remove-Item -Path Function:\Install-EvoScientist -ErrorAction SilentlyContinue
}
