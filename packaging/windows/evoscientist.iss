; Inno Setup script — Windows desktop installer (issue #484, phase 1, Task 6).
;
; Wraps the assembled app directory into EvoScientist-Setup.exe. It:
;   (a) ensures the Microsoft Edge WebView2 Evergreen runtime is present
;       (detect via registry; download + silent-install the bootstrapper if not),
;   (b) lays down the merged app tree (PyInstaller onedir + assemble_bundle.py
;       output) into one install dir under Program Files,
;   (c) creates a Start-menu shortcut (and an optional desktop shortcut).
;
; Input: a single directory that already contains BOTH halves of the bundle —
; the PyInstaller onedir (EvoScientist.exe, langgraph.exe, _internal\) with the
; assemble_bundle.py output (webui\, runtime\node\) copied in next to the exes:
;
;   <SourceDir>\
;     EvoScientist.exe
;     langgraph.exe
;     _internal\...
;     runtime\node\node.exe
;     webui\dist\server.js  (+ .next, node_modules, public)
;     manifest.json
;
; Build (on Windows, with Inno Setup 6.1+ installed for DownloadTemporaryFile):
;   1. uv run python packaging\windows\assemble_bundle.py --out build\bundle
;   2. uv run --extra winbuild pyinstaller packaging\windows\evoscientist.spec --noconfirm
;   3. xcopy /E /I build\bundle\webui   dist\EvoScientist\webui
;      xcopy /E /I build\bundle\runtime dist\EvoScientist\runtime
;      copy    build\bundle\manifest.json dist\EvoScientist\
;   4. iscc packaging\windows\evoscientist.iss
;      (override defaults: iscc /DAppVersion=0.3.0 /DSourceDir=..\..\dist\EvoScientist ...)
;   -> dist\EvoScientist-Setup.exe
;
; This script is authored on Linux and can only be compiled/verified on Windows.

#define MyAppName "EvoScientist"
#define MyAppPublisher "EvoScientist"
#define MyAppExeName "EvoScientist.exe"

; Overridable at compile time: iscc /DAppVersion=... /DSourceDir=...
#ifndef AppVersion
  #define AppVersion "0.3.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\EvoScientist"
#endif

[Setup]
; NOTE: AppId is the stable identity used for upgrades/uninstall — never change it.
AppId={{7F3A1B62-4C8E-4E2A-9D1F-6B0A2C5E9A84}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=dist
OutputBaseFilename=EvoScientist-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The bundle is 64-bit (Node/Python/native libs); refuse to install on 32-bit.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-machine install into Program Files; the WebView2 bootstrapper also
; installs per-machine, so admin is required either way.
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole assembled app tree. recursesubdirs+createallsubdirs pull in
; _internal\, webui\, runtime\ verbatim.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; WorkingDir is a writable per-user location, not {app} (Program Files is
; read-only): the launcher defaults its workspace to the working directory,
; and runs/skills/media are written under it. See build_launcher_config.
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{userdocs}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{userdocs}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; WorkingDir: "{userdocs}"; Flags: nowait postinstall skipifsilent

[Code]
const
  { WebView2 Runtime application GUID (constant across versions). }
  WV2_CLIENT_KEY = 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  { Evergreen Bootstrapper — stable Microsoft fwlink. Downloads a tiny
    (~2 MB) stub that fetches and installs the current runtime. }
  WV2_BOOTSTRAPPER_URL = 'https://go.microsoft.com/fwlink/p/?LinkId=2124703';

function ReadPv(RootKey: Integer): String;
var
  Value: String;
begin
  Result := '';
  if RegQueryStringValue(RootKey, WV2_CLIENT_KEY, 'pv', Value) then
    Result := Value;
end;

{ True when the Evergreen runtime is registered. The x64 runtime registers its
  client key under the 32-bit (WOW6432Node) HKLM view; a per-user install lands
  in HKCU. Check the native and 32-bit HKLM views plus HKCU, and treat an empty
  or 0.0.0.0 version as "not installed" (Microsoft's own detection rule). }
function WebView2Installed(): Boolean;
var
  Pv: String;
begin
  Pv := ReadPv(HKLM32);
  if Pv = '' then Pv := ReadPv(HKLM64);
  if Pv = '' then Pv := ReadPv(HKCU);
  Result := (Pv <> '') and (Pv <> '0.0.0.0');
end;

{ Runs after the wizard, before files are copied. A non-empty return aborts the
  install with that message. Only touches the network when WebView2 is actually
  missing (rare on current Windows, which ships it Evergreen). }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Installer: String;
  ResultCode: Integer;
begin
  Result := '';
  if WebView2Installed() then
    exit;

  WizardForm.StatusLabel.Caption := 'Downloading the Microsoft Edge WebView2 runtime...';
  try
    DownloadTemporaryFile(WV2_BOOTSTRAPPER_URL, 'MicrosoftEdgeWebview2Setup.exe', '', nil);
  except
    Result :=
      'EvoScientist needs the Microsoft Edge WebView2 runtime, which could not be downloaded.' + #13#10 +
      'Connect to the internet and re-run Setup, or install WebView2 manually from' + #13#10 +
      'https://developer.microsoft.com/microsoft-edge/webview2/ and try again.' + #13#10#13#10 +
      '(' + GetExceptionMessage + ')';
    exit;
  end;

  Installer := ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe');
  WizardForm.StatusLabel.Caption := 'Installing the Microsoft Edge WebView2 runtime...';
  if not Exec(Installer, '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    Result := 'Could not start the WebView2 installer: ' + SysErrorMessage(ResultCode);
    exit;
  end;
  if ResultCode <> 0 then
    Result := Format('The WebView2 runtime installer failed (exit code %d).', [ResultCode]);
end;
