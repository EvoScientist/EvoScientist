#!/usr/bin/env bash
# Checks for the installer smoke workflow (install-smoke.yml), run after
# scripts/install.sh on a clean macOS or Linux runner.
#
#   online    the version, where the package was imported from, the installed
#             versions against constraints.txt, a second setup that downloads
#             nothing, and the WebUI launch with the oldest and the newest
#             WebUI release in the core's range
#   offline   EvoSci setup and the WebUI launch; run with the network blocked
#   blocked   succeeds only when https://pypi.org cannot be reached
#   update    EvoSci update to a release (install_smoke_update.py)
#
# EXPECTED_VERSION (else the newest release on INDEX_URL) and
# EVOSCI_INSTALL_CONSTRAINTS (else constraints.txt at the release tag) are
# optional.
set -euo pipefail

INDEX_URL="${INDEX_URL:-https://pypi.org/simple}"
DATA_DIR="${EVOSCIENTIST_DATA_DIR:-$HOME/.evoscientist}"
BACKEND_URL="http://127.0.0.1:${EVOSCIENTIST_LANGGRAPH_DEV_PORT:-6174}/ok"
WEBUI_URL="http://127.0.0.1:${EVOSCIENTIST_WEBUI_PORT:-4716}/"

fail() {
  echo "::error::$*"
  exit 1
}

UV=$(sed -nE 's/.*"path": *"([^"]*)".*/\1/p' "$DATA_DIR/tools/uv.json")
[ -x "$UV" ] || fail "No uv recorded in $DATA_DIR/tools/uv.json"
BIN_DIR=$("$UV" tool dir --bin)
TOOL_DIR=$("$UV" tool dir)
EVOSCI="$BIN_DIR/EvoSci"
PYTHON="$TOOL_DIR/evoscientist/bin/python"

# Any answer below $2 (default 500); the backend's /ok must be a success (300).
answers() {
  local code
  code=$(curl -s --noproxy '*' --max-time 5 -o /dev/null -w '%{http_code}' "$1" || true)
  [ "$code" -ge 200 ] 2>/dev/null && [ "$code" -lt "${2:-500}" ]
}

show_logs() {
  echo "--- EvoSci output"
  cat "$1" || true
  for f in "$HOME/.config/evoscientist/langgraph_dev.log" "$HOME/.config/evoscientist/webui.log"; do
    echo "--- $f"
    tail -n 80 "$f" || true
  done
}

# Starts the WebUI in the background, waits until the backend's /ok and the
# WebUI answer, then stops it with SIGTERM and checks that both ports close.
launch_check() {
  local work log pid i up=0
  work=$(mktemp -d)
  log="$work.log"
  "$EVOSCI" --ui webui --workdir "$work" </dev/null >"$log" 2>&1 &
  pid=$!
  for i in $(seq 1 240); do
    if answers "$BACKEND_URL" 300 && answers "$WEBUI_URL"; then
      up=1
      break
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$up" != 1 ]; then
    kill -TERM "$pid" 2>/dev/null || true
    show_logs "$log"
    fail "The backend ($BACKEND_URL) and the WebUI ($WEBUI_URL) did not both answer within $i s"
  fi
  # The signal handlers that stop the children are installed right after the
  # ready panel, a moment after both URLs answer.
  sleep 3
  if ! kill -TERM "$pid" 2>/dev/null; then
    show_logs "$log"
    fail "EvoSci exited by itself after the WebUI came up"
  fi
  for i in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" || true
    show_logs "$log"
    fail "EvoSci did not stop within 30 s of SIGTERM"
  fi
  sleep 2
  if answers "$BACKEND_URL" || answers "$WEBUI_URL"; then
    show_logs "$log"
    fail "The backend or the WebUI still answers after EvoSci stopped"
  fi
  echo "WebUI launch ok"
}

expected_version() {
  if [ -n "${EXPECTED_VERSION:-}" ]; then
    echo "$EXPECTED_VERSION"
    return
  fi
  "$PYTHON" - "$INDEX_URL" <<'EOF'
import json, re, sys, urllib.request
from packaging.version import Version
req = urllib.request.Request(sys.argv[1] + "/evoscientist/", headers={"Accept": "application/vnd.pypi.simple.v1+json"})
with urllib.request.urlopen(req, timeout=60) as resp:
    versions = json.load(resp)["versions"]
print(max((v for v in versions if re.fullmatch(r"\d+(\.\d+)*", v)), key=Version))
EOF
}

online() {
  local expected actual constraints frozen out
  expected=$(expected_version)
  actual=$("$EVOSCI" --version --json | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["version"])')
  [ "$actual" = "$expected" ] || fail "EvoSci --version --json reports $actual, expected $expected"
  echo "version ok: $actual"

  local file
  file=$("$PYTHON" -c 'import EvoScientist; print(EvoScientist.__file__)')
  case "$file" in
    "$TOOL_DIR"/*) echo "imported from the tool environment: $file" ;;
    *) fail "EvoScientist is imported from $file, outside $TOOL_DIR" ;;
  esac

  constraints="${EVOSCI_INSTALL_CONSTRAINTS:-}"
  if [ -z "$constraints" ]; then
    constraints=$(mktemp)
    curl -fsSL -o "$constraints" "https://raw.githubusercontent.com/EvoScientist/EvoScientist/v$actual/constraints.txt"
  fi
  # Re-resolving the installed set under the constraints changes nothing only
  # when every installed version is the one constraints.txt pins.
  frozen=$(mktemp)
  "$UV" pip freeze --python "$PYTHON" >"$frozen"
  # By name from its folder: uv splits a -c path at spaces (astral-sh/uv#12639).
  out=$(cd "$(dirname "$constraints")" && "$UV" pip install --python "$PYTHON" --dry-run -r "$frozen" -c "$(basename "$constraints")" 2>&1) ||
    fail "The installed versions do not satisfy constraints.txt: $out"
  echo "$out" | grep -q 'Would make no changes' || fail "The installed versions differ from constraints.txt: $out"
  echo "installed versions match constraints.txt"

  out=$("$EVOSCI" setup --json </dev/null) || fail "The second EvoSci setup failed: $out"
  if echo "$out" | grep -Eq '"(Downloading |Installing |Creating the virtual)'; then
    fail "The second EvoSci setup downloaded or installed something: $out"
  fi
  echo "second setup downloaded nothing"

  # The oldest and the newest WebUI release in the core's range, through the
  # test-only override of the range; the newest stays installed.
  local versions oldest newest version
  versions=$("$PYTHON" - <<'EOF'
from packaging.version import Version
from EvoScientist.config import load_config
from EvoScientist.setup.node import DEFAULT_NPM_REGISTRY, NPM_REGISTRIES
from EvoScientist.setup.webui import WEBUI_COMPAT, fetch_metadata, in_range
registry = NPM_REGISTRIES.get(load_config().mirror, DEFAULT_NPM_REGISTRY).rstrip("/")
found = sorted((v for v in fetch_metadata(registry)["versions"] if in_range(v, WEBUI_COMPAT)), key=Version)
print(found[0], found[-1])
EOF
  )
  read -r oldest newest <<<"$versions"
  for version in $(printf '%s\n' "$oldest" "$newest" | uniq); do
    echo "WebUI $version"
    out=$(EVOSCIENTIST_WEBUI_COMPAT="==$version" "$EVOSCI" setup --stage webui --json </dev/null) ||
      fail "EvoSci setup --stage webui failed for WebUI $version: $out"
    echo "$out" | grep -q "\"Using WebUI $version " || fail "WebUI $version was not installed: $out"
    EVOSCIENTIST_WEBUI_COMPAT="==$version" launch_check
  done
}

offline() {
  local out
  out=$("$EVOSCI" setup --json </dev/null) || fail "EvoSci setup failed offline: $out"
  echo "$out" | grep -Eq '"stage": *"webui", *"status": *"done"' ||
    fail "The webui stage did not end with done offline: $out"
  echo "offline setup ok"
  launch_check
}

blocked() {
  if curl -fsS --max-time 15 -o /dev/null https://pypi.org/simple/; then
    fail "https://pypi.org is reachable: the network is not blocked"
  fi
  echo "network blocked"
}

# EvoSci update to a release through this code's updater. With
# EVOSCI_INSTALL_SOURCE (a pull request) the install is the newest release
# from the scripts, whose receipt the update reads, and this wheel goes over
# it with uv pip, which leaves the receipt alone; the update then moves it to
# the newest release. Without it, the installed release is reinstalled, once
# a release has EvoSci update.
update() {
  local key=version check target base
  if [ -n "${EVOSCI_INSTALL_SOURCE:-}" ]; then
    # uv splits a -c path at spaces (astral-sh/uv#12639): by name, from its folder.
    (cd "$(dirname "$EVOSCI_INSTALL_CONSTRAINTS")" &&
      "$UV" pip install -q --python "$PYTHON" --reinstall-package evoscientist \
        -c "$(basename "$EVOSCI_INSTALL_CONSTRAINTS")" "$EVOSCI_INSTALL_SOURCE")
    key=latest
  elif ! "$EVOSCI" update --help >/dev/null 2>&1; then
    echo "This release has no EvoSci update yet; nothing to check."
    return 0
  fi
  check=$("$EVOSCI" update --check --json </dev/null | tail -n 1)
  target=$("$PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1])["detail"][sys.argv[2]])' "$check" "$key")
  # Not the environment's Python: it would run from the install being replaced.
  base=$("$PYTHON" -c 'import sys; print(sys._base_executable)')
  "$base" "$(dirname "$0")/install_smoke_update.py" "$EVOSCI" "$target"
}

case "${1:-}" in
  online) online ;;
  offline) offline ;;
  blocked) blocked ;;
  update) update ;;
  *) fail "usage: $0 online|offline|blocked|update" ;;
esac
