#!/usr/bin/env bash
# Release checks shared by the workflows. Run from the repository root.
#
#   release  (publish.yml, docker.yml) The ref is a version tag, the
#            pyproject.toml version matches it, uv.lock is up to date and
#            constraints.txt matches its export. A release that fails any of
#            these is published neither to PyPI nor as versioned images.
#   pr       (release-check.yml) The uv.lock and constraints.txt checks, only
#            when the pull request changes the pyproject.toml version, so a
#            release PR is fixed before its tag exists. BASE_SHA is the pull
#            request's base commit; the checkout has only the merge commit, so
#            it is fetched here.
set -euo pipefail

EXPORT_ARGS=(--frozen --no-hashes --no-dev --no-emit-project --all-extras --no-header --no-annotate)
EXPORT_CMD="uv export ${EXPORT_ARGS[*]} -o constraints.txt"

# Prints the version of the pyproject.toml on stdin, or nothing when it has no
# version line.
pyproject_version() {
  { grep -m1 '^version = ' || true; } | sed -E 's/^version = "(.*)"/\1/'
}

require_version() {
  if [ -z "$1" ]; then
    echo "::error::No version line (version = \"...\") in $2."
    exit 1
  fi
}

check_lock_and_constraints() {
  # Printed so a mismatch caused by a change in uv's export format can be
  # reproduced with the same uv: uvx uv@<version> export ...
  uv --version
  # Fails both when uv.lock is out of date and when pyproject.toml cannot be
  # resolved at all; uv's own output above says which.
  if ! uv lock --check; then
    echo "::error::uv lock --check failed (see uv's output above). Run: uv lock, then: ${EXPORT_CMD}"
    exit 1
  fi
  local expected
  expected=$(mktemp)
  uv export "${EXPORT_ARGS[@]}" --quiet -o "$expected"
  if [ ! -f constraints.txt ] || ! diff -u --label constraints.txt --label "export of uv.lock" constraints.txt "$expected"; then
    echo "::error::constraints.txt does not match uv.lock. Run: ${EXPORT_CMD}"
    exit 1
  fi
  echo "constraints.txt matches uv.lock ($(wc -l < constraints.txt) lines)"
}

case "${1:-}" in
  release)
    if [ "${GITHUB_REF_TYPE:-}" != "tag" ]; then
      echo "::error::Releases must run from a version tag (got ${GITHUB_REF_TYPE:-no ref type} '${GITHUB_REF_NAME:-}'); dispatch the workflow from the release tag."
      exit 1
    fi
    VERSION=$(pyproject_version < pyproject.toml)
    require_version "$VERSION" pyproject.toml
    TAG="${GITHUB_REF_NAME#v}"
    echo "pyproject version: $VERSION | release tag: $TAG"
    if [ "$VERSION" != "$TAG" ]; then
      echo "::error::pyproject version ($VERSION) does not match release tag ($TAG); refusing to publish."
      exit 1
    fi
    check_lock_and_constraints
    ;;
  pr)
    VERSION=$(pyproject_version < pyproject.toml)
    require_version "$VERSION" pyproject.toml
    git fetch --quiet --depth=1 origin "${BASE_SHA:?BASE_SHA must be the pull request base commit}"
    BASE_VERSION=$(git show "$BASE_SHA:pyproject.toml" | pyproject_version)
    require_version "$BASE_VERSION" "the base commit's pyproject.toml"
    if [ "$VERSION" = "$BASE_VERSION" ]; then
      echo "pyproject version unchanged ($VERSION): not a release PR, nothing to check."
      exit 0
    fi
    echo "Release PR: pyproject version $BASE_VERSION -> $VERSION"
    check_lock_and_constraints
    ;;
  *)
    echo "usage: $0 release|pr" >&2
    exit 2
    ;;
esac
