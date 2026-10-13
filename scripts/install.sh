#!/usr/bin/env bash
# Install EvoScientist on macOS or Linux with nothing else installed.
#
#   curl -fsSL https://raw.githubusercontent.com/EvoScientist/EvoScientist/main/scripts/install.sh | bash
#   curl -fsSL https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@main/scripts/install.sh | bash -s -- --cn
#
# Options:
#   --cn               use mainland China mirrors (uv, Python, PyPI, Node, WebUI)
#   --version <ver>    install this release instead of the newest one
#   --no-modify-path   do not put uv's tool bin directory on PATH
#   --no-setup         stop after installing the package, before `EvoSci setup`
#
# Steps: download a pinned uv and check its SHA-256, keep it under
# <data dir>/tools, install EvoScientist as a uv tool with the release's
# constraints.txt on uv's own Python 3.12, put the tool bin directory on PATH,
# then run `EvoSci setup`. The data dir is $EVOSCIENTIST_DATA_DIR when set and
# not empty, else ~/.evoscientist. A uv installed by the user is never touched.
#
# Test-only variables (the installer smoke workflow): EVOSCI_INSTALL_SOURCE
# installs this wheel instead of the release from the index, and
# EVOSCI_INSTALL_CONSTRAINTS uses this local constraints file instead of the
# one at the release tag.
#
# Everything runs inside main, called on the last line, so a download cut off
# halfway never runs half a script.

set -euo pipefail

UV_VERSION="0.12.23"
# The last release that has no constraints.txt at its tag; it and every older
# release are refused.
LAST_UNSUPPORTED="0.3.5"
PYTHON_VERSION="3.12"
PYPI_SIMPLE="https://pypi.org/simple"
CN_SIMPLE="https://pypi.tuna.tsinghua.edu.cn/simple"
CN_PYTHON_MIRROR="https://registry.npmmirror.com/-/binary/python-build-standalone/"
RAW_BASE="https://raw.githubusercontent.com/EvoScientist/EvoScientist"
CN_RAW_BASE="https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist"

# Reads an existing uv-receipt.toml (argv[1]) with the old tool environment's
# own Python and prints what a reinstall needs to keep, one tab-separated line
# each: "extras<TAB>a,b" for EvoScientist's extras, "with<TAB><requirement>"
# and "with-editable<TAB><path>" for the other requirements (added with --with,
# e.g. by the MCP installer), "warn<TAB><text>" for anything it cannot keep.
RECEIPT_READER='
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
    entries = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["tool"]["requirements"]
except Exception as exc:
    print("warn\tcannot read " + sys.argv[1] + ": " + str(exc))
    sys.exit(0)
for entry in entries:
    try:
        if norm(entry["name"]) == "evoscientist":
            if entry.get("extras"):
                print("extras\t" + ",".join(entry["extras"]))
            continue
        kind, value = requirement(entry)
    except Exception as exc:
        print("warn\tcannot keep " + repr(entry.get("name", entry)) + " (" + str(exc) + ")")
        continue
    print(kind + "\t" + value)
'

info() { printf '%s\n' "$*"; }
warn() { printf 'Warning: %s\n' "$*" >&2; }
die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

TMP_DIR=""
cleanup() {
  if [ -n "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}

usage() {
  cat <<'EOF'
Install EvoScientist on macOS or Linux.

Options:
  --cn               use mainland China mirrors (uv, Python, PyPI, Node, WebUI)
  --version <ver>    install this release instead of the newest one
  --no-modify-path   do not put uv's tool bin directory on PATH
  --no-setup         stop after installing the package, before `EvoSci setup`
EOF
}

# True when version $1 <= $2. Compares the dotted release numbers only, so a
# pre-release such as 0.3.5rc1 counts as 0.3.5.
version_le() {
  local a b i x y
  IFS=. read -r -a a <<<"$(release_part "$1")"
  IFS=. read -r -a b <<<"$(release_part "$2")"
  for ((i = 0; i < ${#a[@]} || i < ${#b[@]}; i++)); do
    x=${a[i]:-0}
    y=${b[i]:-0}
    if ((10#$x < 10#$y)); then return 0; fi
    if ((10#$x > 10#$y)); then return 1; fi
  done
  return 0
}

release_part() {
  printf '%s' "$1" | sed -E 's/^([0-9]+(\.[0-9]+)*).*/\1/'
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    die "Neither sha256sum nor shasum is installed, so the uv download cannot be checked. Install one of them and run this script again."
  fi
}

download() {
  curl -fsSL --retry 3 -o "$2" "$1" || die "Download failed: $1"
}

# Sets OS_NAME, UV_TARGET, UV_SHA256, UV_WHEEL_PATH and UV_WHEEL_SHA256, or
# stops on a platform that EvoScientist does not support.
detect_platform() {
  local os arch glibc
  os=$(uname -s)
  arch=$(uname -m)
  case "$os" in
    Darwin)
      OS_NAME=macos
      # A shell running under Rosetta reports x86_64 on an Apple Silicon Mac.
      if [ "$arch" = x86_64 ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
        arch=arm64
      fi
      ;;
    Linux)
      OS_NAME=linux
      if ! glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null); then
        die "This Linux has no glibc (musl, as on Alpine). EvoScientist supports Linux with glibc 2.28 or newer."
      fi
      glibc=${glibc##* }
      if version_le "$glibc" 2.27; then
        die "This Linux has glibc $glibc. EvoScientist supports Linux with glibc 2.28 or newer."
      fi
      ;;
    *)
      die "Unsupported operating system: $os. Use install.ps1 on Windows."
      ;;
  esac
  case "$OS_NAME/$arch" in
    macos/arm64)
      UV_TARGET=aarch64-apple-darwin
      UV_SHA256=50487ae565ccd96e499056b4674d438f4c53170202617b4c759defe0c6a1b544
      UV_WHEEL_PATH=packages/23/e9/b75db33954e985a763bd3277a477cabd551aaba7d08872b22400838dace2/uv-0.12.23-py3-none-macosx_11_0_arm64.whl
      UV_WHEEL_SHA256=99197c022afbd1edd0a295e9fdac5078c61eddf2e77657a77294cfe8ea0d94b9
      ;;
    macos/x86_64)
      UV_TARGET=x86_64-apple-darwin
      UV_SHA256=960da44cb4b73685206ddd250b19e0a117fa41095710c1038f081f5cb613efb4
      UV_WHEEL_PATH=packages/01/15/a2514d061b92d4b7c0ebefc3da51f3ff5a528e9f4c067ea2cf48c5573e5a/uv-0.12.23-py3-none-macosx_10_12_x86_64.whl
      UV_WHEEL_SHA256=4951286e68f116fa41897ecf8e248cc2836cb013be6d4f37afae4049f9ef4df7
      ;;
    linux/x86_64 | linux/amd64)
      UV_TARGET=x86_64-unknown-linux-gnu
      UV_SHA256=9167d72b3319674b6303c4cbe071854bba13ebdf3d76b1a7cbdc175471fb66d6
      UV_WHEEL_PATH=packages/e5/83/85a6c63c24905af4924fddb11a499b934913f59a134248367a1ef1a4716f/uv-0.12.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      UV_WHEEL_SHA256=565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff
      ;;
    linux/aarch64 | linux/arm64)
      UV_TARGET=aarch64-unknown-linux-gnu
      UV_SHA256=6524bd338177ed50d035d39354e12545e993bbeba2ecbddf0480c5b3a81d313f
      UV_WHEEL_PATH=packages/df/50/84d0424b63718ac6f277c0b77206751e078a0adea16813320508a033e718/uv-0.12.23-py3-none-manylinux_2_28_aarch64.whl
      UV_WHEEL_SHA256=9346c741175c7fc0d380f30fcfcb8d5e13f5f900b30536230c8d34208524e47a
      ;;
    *)
      die "Unsupported processor: $arch on $os."
      ;;
  esac
  if [ "$UV_TARGET" = x86_64-apple-darwin ]; then
    info "Note: Intel Macs are supported on a best-effort basis; the stt extra cannot be installed there."
  fi
}

# The data dir rule of EvoScientist's paths module: $EVOSCIENTIST_DATA_DIR when
# set and not empty (with ~ expanded), else ~/.evoscientist; made absolute.
resolve_data_dir() {
  local dir="${EVOSCIENTIST_DATA_DIR:-}"
  if [ -z "$dir" ]; then
    dir="$HOME/.evoscientist"
  fi
  case "$dir" in
    \~) dir="$HOME" ;;
    \~/*) dir="$HOME/${dir#\~/}" ;;
  esac
  mkdir -p "$dir/tools"
  DATA_DIR=$(cd "$dir" && pwd -P)
}

# Sets VERSION: --version as given, or the version of the test wheel, or the
# newest final release that the index lists.
choose_version() {
  if [ -n "$VERSION" ]; then
    VERSION=${VERSION#v}
    if ! printf '%s' "$VERSION" | grep -Eq '^[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?$'; then
      die "--version expects a release number such as 0.3.6, got: $VERSION"
    fi
    return
  fi
  if [ -n "${EVOSCI_INSTALL_SOURCE:-}" ]; then
    VERSION=$(basename "$EVOSCI_INSTALL_SOURCE" | sed -nE 's/^[Ee]vo[Ss]cientist-([^-]+)-.*\.whl$/\1/p')
    [ -n "$VERSION" ] || die "Cannot read the version from EVOSCI_INSTALL_SOURCE: $EVOSCI_INSTALL_SOURCE"
    return
  fi
  local index="$PYPI_SIMPLE"
  if [ "$CN" = 1 ]; then
    index="$CN_SIMPLE"
  fi
  local listing
  listing=$(curl -fsSL --retry 3 -H 'Accept: application/vnd.pypi.simple.v1+json' "$index/evoscientist/") ||
    die "Cannot read the EvoScientist versions from $index/evoscientist/"
  # The mirror sorts the list as strings, so filter to final releases and sort
  # numerically; never take the last entry as it comes.
  VERSION=$(printf '%s' "$listing" | tr -d '\n' |
    sed -nE 's/.*"versions"[[:space:]]*:[[:space:]]*\[([^]]*)\].*/\1/p' |
    tr ',' '\n' | tr -d ' "' |
    { grep -E '^[0-9]+(\.[0-9]+)*$' || true; } |
    sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | tail -n 1)
  [ -n "$VERSION" ] || die "No EvoScientist release found on $index"
}

check_version_floor() {
  if version_le "$VERSION" "$LAST_UNSUPPORTED"; then
    die "EvoScientist $VERSION cannot be installed with this script: $LAST_UNSUPPORTED is the last release without constraints.txt, and every older release lacks it too. Choose a newer version."
  fi
}

# Downloads the pinned uv, checks its hash and keeps it in
# <data dir>/tools/uv-<version>/; sets UV and records it in tools/uv.json.
install_uv() {
  local tools="$DATA_DIR/tools" archive actual staged
  local final="$tools/uv-$UV_VERSION"
  staged="$TMP_DIR/uv-bin"
  mkdir -p "$staged"
  if [ "$CN" = 1 ]; then
    command -v unzip >/dev/null 2>&1 || die "--cn needs unzip to unpack uv; install unzip and run this script again."
    archive="$TMP_DIR/uv.whl"
    info "Downloading uv $UV_VERSION from the Tsinghua mirror..."
    download "https://pypi.tuna.tsinghua.edu.cn/$UV_WHEEL_PATH" "$archive"
    actual=$(sha256_of "$archive")
    [ "$actual" = "$UV_WHEEL_SHA256" ] || die "The uv download does not match its pinned SHA-256 (expected $UV_WHEEL_SHA256, got $actual). Nothing was installed."
    unzip -p "$archive" "uv-$UV_VERSION.data/scripts/uv" >"$staged/uv" || die "Cannot unpack uv from $archive"
  else
    archive="$TMP_DIR/uv.tar.gz"
    info "Downloading uv $UV_VERSION..."
    download "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TARGET.tar.gz" "$archive"
    actual=$(sha256_of "$archive")
    [ "$actual" = "$UV_SHA256" ] || die "The uv download does not match its pinned SHA-256 (expected $UV_SHA256, got $actual). Nothing was installed."
    tar -xzf "$archive" -C "$TMP_DIR" || die "Cannot unpack $archive"
    mv "$TMP_DIR/uv-$UV_TARGET/uv" "$staged/uv"
  fi
  chmod +x "$staged/uv"
  rm -rf "$final"
  mv "$staged" "$final"
  UV="$final/uv"
  local escaped=${UV//\\/\\\\}
  escaped=${escaped//\"/\\\"}
  printf '{"version": "%s", "path": "%s"}' "$UV_VERSION" "$escaped" >"$tools/uv.json.tmp"
  mv "$tools/uv.json.tmp" "$tools/uv.json"
}

fetch_constraints() {
  CONSTRAINTS="$TMP_DIR/constraints.txt"
  if [ -n "${EVOSCI_INSTALL_CONSTRAINTS:-}" ]; then
    cp "$EVOSCI_INSTALL_CONSTRAINTS" "$CONSTRAINTS"
    return
  fi
  local base="$RAW_BASE/v$VERSION"
  if [ "$CN" = 1 ]; then
    base="$CN_RAW_BASE@v$VERSION"
  fi
  info "Downloading the constraints of EvoScientist $VERSION..."
  curl -fsSL --retry 3 -o "$CONSTRAINTS" "$base/constraints.txt" ||
    die "Cannot download $base/constraints.txt, so EvoScientist $VERSION cannot be installed with its tested dependencies."
}

# When EvoScientist is already a uv tool, fills WITH_ARGS and EXTRAS from its
# receipt so that the reinstall keeps them, and sets FORCE. Stops when the bin
# dir already holds an EvoScientist command that is not a uv tool's (pipx and
# pip --user use ~/.local/bin too): uv would refuse to overwrite it.
read_existing_install() {
  local tool_dir env receipt python kind value name
  tool_dir=$("$UV" tool dir </dev/null)
  env="$tool_dir/evoscientist"
  receipt="$env/uv-receipt.toml"
  python="$env/bin/python"
  if [ ! -d "$env" ]; then
    for name in EvoSci EvoScientist evosci evoscientist; do
      if [ -e "$BIN_DIR/$name" ]; then
        die "$BIN_DIR/$name exists but was not installed by uv; pipx or pip --user may have put it there. Remove that install (for example pipx uninstall EvoScientist or pip uninstall EvoScientist), then run this script again."
      fi
    done
    return 0
  fi
  FORCE=1
  local lost="Extras and packages added with --with (such as MCP servers) may be dropped by the reinstall."
  if [ ! -f "$receipt" ] || [ ! -x "$python" ]; then
    warn "Cannot read the settings of the existing install in $env. $lost"
    return 0
  fi
  local out
  if ! out=$("$python" -I -c "$RECEIPT_READER" "$receipt" </dev/null 2>&1); then
    warn "Cannot read $receipt with $python. $lost"
    return 0
  fi
  while IFS=$'\t' read -r kind value; do
    case "$kind" in
      extras) EXTRAS="[$value]" ;;
      with) WITH_ARGS+=(--with "$value") ;;
      with-editable) WITH_ARGS+=(--with-editable "$value") ;;
      warn) warn "From $receipt: $value; it is not kept by the reinstall." ;;
    esac
  done <<<"$out"
}

# Names the uv config that sets exclude-newer, which blocks releases newer than
# its cutoff. uv prints its own hint; this says where the setting comes from.
exclude_newer_hint() {
  if [ -n "${UV_EXCLUDE_NEWER:-}" ]; then
    warn "UV_EXCLUDE_NEWER is set ($UV_EXCLUDE_NEWER): uv ignores releases uploaded after that date."
  fi
  if [ -n "${UV_NO_CONFIG:-}" ]; then
    return
  fi
  local file
  for file in "${UV_CONFIG_FILE:-}" "${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml" /etc/uv/uv.toml; do
    if [ -n "$file" ] && [ -f "$file" ] && grep -Eq '^[[:space:]]*exclude-newer[[:space:]]*=' "$file"; then
      warn "$file sets exclude-newer: uv ignores releases uploaded after its cutoff, so a newer EvoScientist cannot install. Remove that line, or wait until the release is older than the cutoff."
    fi
  done
}

install_package() {
  local spec="EvoScientist$EXTRAS==$VERSION"
  if [ -n "${EVOSCI_INSTALL_SOURCE:-}" ]; then
    local wheel
    wheel=$(cd "$(dirname "$EVOSCI_INSTALL_SOURCE")" && pwd -P)/$(basename "$EVOSCI_INSTALL_SOURCE")
    spec="EvoScientist$EXTRAS @ file://$wheel"
  fi
  local args=(tool install --python "$PYTHON_VERSION" --managed-python -c "$CONSTRAINTS")
  if [ "$FORCE" = 1 ]; then
    args+=(--force)
  fi
  info "Installing EvoScientist $VERSION..."
  if ! "$UV" "${args[@]}" ${WITH_ARGS[@]+"${WITH_ARGS[@]}"} "$spec" </dev/null; then
    exclude_newer_hint
    die "uv could not install EvoScientist $VERSION (see uv's output above)."
  fi
}

# Puts uv's tool bin directory on PATH through the shell profile, unless it is
# already there or --no-modify-path was given.
update_path() {
  case ":$PATH:" in
    *":$BIN_DIR:"*) return 0 ;;
  esac
  if [ "$NO_MODIFY_PATH" = 1 ]; then
    info "Not changing PATH (--no-modify-path). Add $BIN_DIR to PATH to run EvoSci."
    return 0
  fi
  local rc=0 out
  out=$("$UV" tool update-shell </dev/null 2>&1) || rc=$?
  case "$rc" in
    0)
      info "$out"
      PATH_CHANGED=1
      ;;
    # Exit 2 is uv's code for any error; only this message means the profiles
    # already add the directory and this shell has not read them yet, e.g. on
    # a second run of this script.
    2)
      case "$out" in
        *"already up-to-date"*) PATH_CHANGED=1 ;;
        *)
          printf '%s\n' "$out" >&2
          warn "uv tool update-shell failed; add $BIN_DIR to PATH yourself."
          ;;
      esac
      ;;
    *)
      printf '%s\n' "$out" >&2
      warn "uv tool update-shell failed; add $BIN_DIR to PATH yourself."
      ;;
  esac
}

# Warns when another EvoSci comes first on PATH (pip, conda). Compares the
# directories that hold the commands, resolved as pwd -P does, not the entry
# point's link target, which lies inside the tool environment.
check_shadowing() {
  local found found_dir bin_dir
  found=$(command -v EvoSci 2>/dev/null || true)
  [ -n "$found" ] || return 0
  found_dir=$(cd "$(dirname "$found")" 2>/dev/null && pwd -P) || return 0
  bin_dir=$(cd "$BIN_DIR" 2>/dev/null && pwd -P) || return 0
  if [ "$found_dir" != "$bin_dir" ]; then
    warn "Another EvoSci comes first on PATH: $found. It will run instead of this install; remove it (for example pip uninstall EvoScientist) or put $BIN_DIR earlier on PATH."
  fi
}

main() {
  CN=0
  VERSION=""
  NO_MODIFY_PATH=0
  NO_SETUP=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --cn) CN=1 ;;
      --version)
        [ $# -ge 2 ] || die "--version needs a value"
        VERSION=$2
        shift
        ;;
      --version=*) VERSION=${1#--version=} ;;
      --no-modify-path) NO_MODIFY_PATH=1 ;;
      --no-setup) NO_SETUP=1 ;;
      -h | --help)
        usage
        return 0
        ;;
      *) die "Unknown option: $1 (options: --cn, --version <ver>, --no-modify-path, --no-setup)" ;;
    esac
    shift
  done

  command -v curl >/dev/null 2>&1 || die "This script needs curl."
  detect_platform
  choose_version
  check_version_floor

  TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/evosci-install.XXXXXX")
  trap cleanup EXIT
  resolve_data_dir

  # uv trusts the system certificate store, as curl did above, so a network
  # that inspects TLS works; longer timeout for slow mirrors.
  export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-1}"
  export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
  if [ "$CN" = 1 ]; then
    export UV_DEFAULT_INDEX="$CN_SIMPLE"
    export UV_PYTHON_INSTALL_MIRROR="$CN_PYTHON_MIRROR"
  fi

  install_uv
  fetch_constraints
  FORCE=0
  EXTRAS=""
  WITH_ARGS=()
  BIN_DIR=$("$UV" tool dir --bin </dev/null)
  read_existing_install
  install_package

  PATH_CHANGED=0
  update_path

  if [ "$NO_SETUP" = 1 ]; then
    info "Skipping EvoSci setup (--no-setup)."
  else
    local setup_args=(setup)
    if [ "$CN" = 1 ]; then
      setup_args+=(--cn)
    fi
    # By absolute path: this shell does not see the PATH change yet.
    "$BIN_DIR/EvoSci" "${setup_args[@]}" </dev/null ||
      die "EvoScientist $VERSION is installed, but EvoSci setup failed (see above). Run it again with: $BIN_DIR/EvoSci setup"
  fi

  check_shadowing

  info ""
  info "EvoScientist $VERSION is installed."
  if [ "$PATH_CHANGED" = 1 ]; then
    info "Your shell profile puts $BIN_DIR on PATH (uv tool update-shell); open a new terminal first."
  fi
  info "Next steps:"
  info "  EvoSci onboard          set up your model provider and API keys"
  info "  EvoSci                  start the terminal UI"
  info "  EvoSci --ui webui       start the WebUI"
}

main "$@"
